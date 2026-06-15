"""``deep-dream explore`` — multi-strategy semantic exploration.

Combines five retrieval strategies into a single unified result set:

1. **Document file search** — grep-like line matches across indexed files.
2. **Semantic concept search** — embedding similarity against the concept graph.
3. **Graph neighbour expansion** — walk outward from seed concepts.
4. **Relation evidence collection** — find co-mention evidence for concept pairs.
5. **Evidence card assembly** — merge raw file hits and graph source text into
   a unified card format.

Human output uses Rich panels and tables.  ``--json`` returns the same
structure as the legacy argparse command.
"""
from __future__ import annotations

import json as _json
from typing import Any, Dict, List

import click

from ._ctx import CliContext
from ._exit_codes import ERROR, OK
from ._output import OutputManager


def _to_dict(item: Any) -> dict:
    """Normalize an item to a plain dict.

    ``agent_semantic_search`` may return Entity DTO objects instead of
    dicts.  This helper ensures safe ``.get()`` access in all cases.
    """
    if isinstance(item, dict):
        return item
    if hasattr(item, "__dict__"):
        return vars(item)
    return {}


def _like_pattern(term: str) -> str:
    """Escape a term for use in a SQL LIKE pattern wrapped in wildcards."""
    escaped = term.replace("!", "!!").replace("%", "!%").replace("_", "!_")
    return f"%{escaped}%"


def _semantic_search_for_role(
    storage: Any,
    term: str,
    *,
    role: str | None,
    top_k: int,
    threshold: float,
) -> dict:
    """Run a semantic search for the given role.

    The storage ``agent_semantic_search`` only supports ``entity``/``relation``
    (and a LIKE fallback for ``entity``).  ``document`` and ``episode`` have no
    embedding branch there, so we run a name/content LIKE lookup for those
    roles here so every role returns *something*.

    Returns ``{"results": [...], "fallback": bool}`` where ``fallback`` is
    ``True`` whenever the result is a name/content match rather than a real
    cosine similarity (i.e. embeddings were unavailable).
    """
    if role in (None, "entity", "relation"):
        result = storage.agent_semantic_search(
            term, role=role, top_k=top_k, threshold=threshold,
        )
        results = result.get("results", [])
        # Detect the LIKE-fallback signature: storage assigns every LIKE row
        # a uniform pseudo-score of ``threshold * 0.95``.  When the embedding
        # client is unavailable, the similarity search returns nothing and the
        # fallback fills results, so flag it honestly.
        fallback = _is_fallback(storage, results, threshold)
        return {"results": results, "fallback": fallback}

    # Container roles: no embedding path exists; do a content/name LIKE.
    results = _like_search_container(storage, term, role=role, limit=top_k)
    return {"results": results, "fallback": True}


def _is_fallback(storage: Any, results: list, threshold: float) -> bool:
    """Return True if *results* came from the LIKE name-match fallback.

    The fallback fires when embeddings are unavailable (the CLI storage never
    wires an embedding client), so check that first.  As a secondary signal we
    also recognise the uniform ``threshold * 0.95`` pseudo-score that storage
    stamps onto every LIKE row.
    """
    emb_client = getattr(storage, "embedding_client", None)
    if emb_client is None or not getattr(emb_client, "is_available", lambda: False)():
        return True
    if not results:
        return False
    pseudo = threshold * 0.95
    return all(
        abs(float(_to_dict(r).get("_score") or 0.0) - pseudo) < 1e-9
        for r in results
    )


def _like_search_container(
    storage: Any, term: str, *, role: str, limit: int,
) -> list[dict]:
    """LIKE-based name/content search for the document/episode container roles.

    These roles have no embedding branch in storage, so we match on title /
    heading / content so every role returns something useful instead of 0.
    """
    like = _like_pattern(term)
    conn = storage._conn()
    rows: list[dict] = []
    if role == "document":
        try:
            cur = conn.execute(
                "SELECT document_version_id, document_family_id, title, "
                "source_mode, read_path "
                "FROM v_document_files "
                "WHERE title LIKE ? ESCAPE '!' "
                "ORDER BY processed_time DESC LIMIT ?",
                (like, limit),
            ).fetchall()
        except Exception:
            cur = []
        for r in cur:
            rows.append({
                "family_id": r[1] or r[0],
                "name": r[2] or "",
                "content": "",
                "role": "document",
                "document_version_id": r[0],
                "read_path": r[4] or "",
            })
        return rows

    if role == "episode":
        try:
            cur = conn.execute(
                "SELECT version_id, family_id, heading_path, "
                "memory_content, source_text "
                "FROM v_episodes "
                "WHERE (heading_path LIKE ? ESCAPE '!' "
                "       OR COALESCE(memory_content, '') LIKE ? ESCAPE '!' "
                "       OR COALESCE(source_text, '') LIKE ? ESCAPE '!') "
                "ORDER BY start_offset LIMIT ?",
                (like, like, like, limit),
            ).fetchall()
        except Exception:
            cur = []
        for r in cur:
            heading = r[2] or ""
            content = (r[3] or r[4] or "")
            # Fall back to a short content snippet when there's no heading.
            name = heading or (content[:40] + ("..." if len(content) > 40 else ""))
            rows.append({
                "family_id": r[1] or r[0],
                "name": name,
                "content": content,
                "role": "episode",
                "episode_version_id": r[0],
            })
        return rows

    return rows


# ------------------------------------------------------------------
# Click command
# ------------------------------------------------------------------

@click.command()
@click.argument("question")
@click.option(
    "--role",
    type=click.Choice(["document", "episode", "entity", "relation"]),
    default=None,
    help="Semantic search role filter.",
)
@click.option(
    "--limit",
    type=int,
    default=20,
    show_default=True,
    help="Maximum total semantic results.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.2,
    show_default=True,
    help="Minimum similarity threshold for semantic search.",
)
@click.option(
    "--file-limit",
    type=int,
    default=20,
    show_default=True,
    help="Maximum total document file hits.",
)
@click.option(
    "--per-term-file-limit",
    type=int,
    default=5,
    show_default=True,
    help="Maximum file hits per individual query term.",
)
@click.option(
    "--expand-query/--no-expand-query",
    default=True,
    show_default=True,
    help="Use all query terms (question + explicit --terms) for semantic "
         "search. --no-expand-query narrows semantic fan-out to the question "
         "term only; explicit --terms are still used for document search.",
)
@click.option(
    "--terms",
    default=None,
    help="Comma-separated query expansion terms generated by the caller/agent.",
)
@click.option(
    "--semantic-queries",
    type=int,
    default=5,
    show_default=True,
    help="Maximum number of query terms used for semantic search.",
)
@click.option(
    "--min-semantic-score",
    type=float,
    default=0.0,
    show_default=True,
    help="Discard semantic results below this score.",
)
@click.option(
    "--evidence-limit",
    type=int,
    default=12,
    show_default=True,
    help="Maximum number of assembled evidence cards.",
)
@click.option(
    "--neighbor-seeds",
    type=int,
    default=3,
    show_default=True,
    help="Number of top semantic concepts used as seeds for neighbour expansion.",
)
@click.option(
    "--neighbor-limit",
    type=int,
    default=50,
    show_default=True,
    help="Maximum number of neighbour results (applied AFTER fetch, so it "
         "does not starve deeper traversals).",
)
@click.option(
    "--depth",
    type=int,
    default=1,
    show_default=True,
    help="Graph traversal depth for neighbour expansion. Higher depth fetches "
         "more candidates internally (scaled by depth) and then trims to "
         "--neighbor-limit, so results from deeper hops are not saturated out.",
)
@click.option(
    "--relation-seed-count",
    type=int,
    default=5,
    show_default=True,
    help="Number of top concepts used to build relation pairs.",
)
@click.option(
    "--relation-pair-limit",
    type=int,
    default=8,
    show_default=True,
    help="Maximum number of concept pairs checked for relations.",
)
@click.option(
    "--relation-evidence-limit",
    type=int,
    default=10,
    show_default=True,
    help="Maximum total relation evidence rows.",
)
@click.pass_context
def explore(
    ctx: click.Context,
    question: str,
    role: str | None,
    limit: int,
    threshold: float,
    file_limit: int,
    per_term_file_limit: int,
    expand_query: bool,
    terms: str | None,
    semantic_queries: int,
    min_semantic_score: float,
    evidence_limit: int,
    neighbor_seeds: int,
    neighbor_limit: int,
    depth: int,
    relation_seed_count: int,
    relation_pair_limit: int,
    relation_evidence_limit: int,
) -> None:
    """Multi-strategy semantic exploration of the concept graph.

    QUESTION is the natural-language query to explore.  The command searches
    raw documents, runs semantic concept lookups, expands into graph
    neighbours, and collects relation evidence -- returning everything in
    a single unified result set.
    """
    from ._helpers import (
        concept_source_evidence,
        evidence_cards,
        expand_query_terms,
        relation_evidence,
        search_document_terms,
    )

    out = OutputManager(ctx)
    cli_ctx: CliContext = ctx.obj

    graph_id = cli_ctx.get_active_graph()

    with cli_ctx.get_storage(graph_id) as storage:
        # ------------------------------------------------------------------
        # 1. Query expansion
        # ------------------------------------------------------------------
        # ``expand_query_terms`` never auto-generates terms — every non-original
        # term is an explicit ``--terms`` entry the caller supplied.  The old
        # ``query_terms[:1]`` chop under ``--no-expand-query`` silently dropped
        # those explicit terms.  We now keep ALL terms for document search;
        # ``--no-expand-query`` only narrows the *semantic* fan-out below.
        query_terms = expand_query_terms(question, terms)

        # ------------------------------------------------------------------
        # 2. Document file search
        # ------------------------------------------------------------------
        with out.spinner("Searching document files..."):
            file_hits = search_document_terms(
                storage,
                query_terms,
                per_term_limit=per_term_file_limit,
                total_limit=file_limit,
            )

        # ------------------------------------------------------------------
        # 3. Semantic concept search
        # ------------------------------------------------------------------
        semantic_results: list[dict] = []
        semantic_seen: set[str] = set()
        # When embeddings are unavailable, storage falls back to a LIKE name
        # lookup that assigns every row a uniform pseudo-score. Track that so
        # we can present results honestly instead of as fake cosine scores.
        semantic_fallback_used = False
        # Fan out across the surviving query terms (capped by --semantic-queries).
        # Under ``--no-expand-query`` only the original question term is used for
        # semantic lookups, but explicit ``--terms`` are never dropped.
        semantic_query_terms = query_terms if expand_query else query_terms[:1]

        with out.spinner("Running semantic search..."):
            for term_info in semantic_query_terms[:semantic_queries]:
                semantic = _semantic_search_for_role(
                    storage,
                    term_info["term"],
                    role=role,
                    top_k=limit,
                    threshold=threshold,
                )
                results = semantic.get("results", [])
                if semantic.get("fallback"):
                    semantic_fallback_used = True
                for raw_item in results:
                    item = _to_dict(raw_item)
                    score = item.get("score")
                    if score is None:
                        score = item.get("_score")
                    if (
                        score is not None
                        and not semantic.get("fallback")
                        and float(score or 0.0) < min_semantic_score
                    ):
                        continue
                    fid = item.get("family_id", "")
                    if not fid or fid in semantic_seen:
                        continue
                    semantic_seen.add(fid)
                    item["matched_query"] = term_info["term"]
                    item["query_source"] = term_info.get("source", "expanded")
                    item["match_mode"] = (
                        "name-like fallback"
                        if semantic.get("fallback")
                        else "embedding"
                    )
                    semantic_results.append(item)
                    if len(semantic_results) >= limit:
                        break
                if len(semantic_results) >= limit:
                    break

        semantic_results.sort(
            key=lambda x: float(x.get("score") or x.get("_score") or 0.0),
            reverse=True,
        )

        concept_ids = [
            r.get("family_id", "")
            for r in semantic_results
            if r.get("family_id")
        ]

        # ------------------------------------------------------------------
        # 4. Source evidence
        # ------------------------------------------------------------------
        with out.spinner("Collecting source evidence..."):
            source_evidence = concept_source_evidence(
                storage, concept_ids, limit=limit
            )

        # ------------------------------------------------------------------
        # 5. Graph neighbour expansion
        # ------------------------------------------------------------------
        # Fetch more internally than the user-facing neighbour limit so that
        # deeper traversals (``--depth`` > 1) are not saturated out by the
        # first seed's depth-1 fan-out. The list is trimmed to the requested
        # limit below.
        neighbors: list[dict] = []
        per_seed_fetch = max(neighbor_limit, neighbor_limit * max(1, depth))
        with out.spinner("Expanding graph neighbours..."):
            for fid in concept_ids[:neighbor_seeds]:
                try:
                    for nb in storage.get_concept_neighbors(
                        fid, max_depth=depth, max_results=per_seed_fetch
                    ):
                        neighbors.append(_to_dict(nb))
                except Exception:
                    continue

        # ------------------------------------------------------------------
        # 6. Relation evidence
        # ------------------------------------------------------------------
        relation_samples: list[dict] = []
        relation_pairs: list[tuple[str, str]] = []
        with out.spinner("Collecting relation evidence..."):
            for i, left in enumerate(concept_ids[:relation_seed_count]):
                for right in concept_ids[i + 1 : relation_seed_count]:
                    if left != right:
                        relation_pairs.append((left, right))
            for left, right in relation_pairs[:relation_pair_limit]:
                evidence = relation_evidence(
                    storage, left, right, limit=relation_evidence_limit
                )
                for raw_item in evidence:
                    item = _to_dict(raw_item)
                    item["query_pair"] = [left, right]
                    relation_samples.append(item)
                if len(relation_samples) >= relation_evidence_limit:
                    relation_samples = relation_samples[:relation_evidence_limit]
                    break

        # ------------------------------------------------------------------
        # 7. Evidence cards
        # ------------------------------------------------------------------
        cards = evidence_cards(
            file_hits, source_evidence, query_terms, limit=evidence_limit
        )

        # Trim neighbors to the requested limit
        trimmed_neighbors = neighbors[:neighbor_limit]

        # ------------------------------------------------------------------
        # Build result payload
        # ------------------------------------------------------------------
        data = {
            "question": question,
            "query_terms": query_terms,
            "file_hits": file_hits,
            "semantic_hits": semantic_results,
            "semantic_total": len(semantic_results),
            "semantic_fallback": semantic_fallback_used,
            "source_evidence": source_evidence,
            "evidence_cards": cards,
            "neighbors": trimmed_neighbors,
            "relation_evidence": relation_samples,
            "depth": depth,
            "coverage": {
                "file_hits": len(file_hits),
                "semantic_hits": len(semantic_results),
                "source_evidence": len(source_evidence),
                "evidence_cards": len(cards),
                "neighbors": len(trimmed_neighbors),
                "relation_evidence": len(relation_samples),
                "relation_pairs_checked": min(
                    len(relation_pairs), relation_pair_limit
                ),
                "semantic_mode": (
                    "name-like fallback" if semantic_fallback_used else "embedding"
                ),
            },
        }

        meta = {
            "graph_id": graph_id,
            "used": {
                "raw_files": True,
                "sqlite": True,
                "semantic": True,
                "graph_traversal": True,
                "api": False,
            },
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if out.is_json:
        from ._output import json_result

        payload = json_result("explore", data, meta=meta)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    _render_human(out, data, meta)


# ------------------------------------------------------------------
# Rich human-readable rendering
# ------------------------------------------------------------------

def _render_human(
    out: OutputManager,
    data: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    """Render the explore results as Rich panels and tables."""
    from rich.panel import Panel
    from rich.markup import escape as _esc
    from rich.table import Table

    coverage = data["coverage"]
    fallback = bool(data.get("semantic_fallback"))

    # -- Quiet mode: collapse to a single one-liner of coverage -----------
    if out.is_quiet:
        out.console.print(
            f"{_esc(data['question'])} | "
            f"files={coverage['file_hits']} "
            f"semantic={coverage['semantic_hits']} "
            f"evidence={coverage['evidence_cards']} "
            f"neighbors={coverage['neighbors']} "
            f"relations={coverage['relation_evidence']}"
            + (" (semantic: name-match fallback)" if fallback else "")
        )
        return

    # -- Header panel ---------------------------------------------------
    header_lines = [
        f"[bold]Question:[/bold] {_esc(data['question'])}",
        f"[bold]Graph:[/bold]   {_esc(meta.get('graph_id', '?'))}",
    ]
    terms_display = ", ".join(
        _esc(t["term"]) for t in data.get("query_terms", [])
    )
    if terms_display:
        header_lines.append(f"[bold]Terms:[/bold]    {terms_display}")
    out.console.print(Panel(
        "\n".join(header_lines),
        title="Explore",
        border_style="cyan",
    ))

    # -- Honest note when semantic search fell back to name matches ------
    if fallback:
        out.console.print(
            "[yellow]semantic embeddings unavailable — showing name "
            "matches (no real similarity scores).[/yellow]"
        )

    # -- Coverage summary table -----------------------------------------
    cov_table = Table(
        title="Coverage Summary",
        show_header=True,
        header_style="bold",
        border_style="dim",
    )
    cov_table.add_column("Strategy", style="cyan", min_width=22)
    cov_table.add_column("Hits", justify="right", min_width=6)
    cov_table.add_column("Detail", min_width=20)
    cov_table.add_row(
        "Document file search",
        str(coverage["file_hits"]),
        f"per-term limit applied",
    )
    cov_table.add_row(
        "Semantic concept search",
        str(coverage["semantic_hits"]),
        "name-match fallback" if fallback else "threshold filtered",
    )
    cov_table.add_row(
        "Source evidence",
        str(coverage["source_evidence"]),
        "",
    )
    cov_table.add_row(
        "Evidence cards",
        str(coverage["evidence_cards"]),
        "merged file + graph",
    )
    cov_table.add_row(
        "Graph neighbours",
        str(coverage["neighbors"]),
        f"depth={data.get('depth', '?')}",
    )
    cov_table.add_row(
        "Relation evidence",
        str(coverage["relation_evidence"]),
        f"{coverage['relation_pairs_checked']} pairs checked",
    )
    out.console.print(cov_table)

    # -- File hits ------------------------------------------------------
    file_hits = data.get("file_hits", [])
    if file_hits:
        _render_file_hits(out, file_hits)
    else:
        out.console.print("[dim]No document file hits.[/dim]")

    # -- Semantic hits --------------------------------------------------
    semantic_hits = data.get("semantic_hits", [])
    if semantic_hits:
        _render_semantic_hits(out, semantic_hits)
    else:
        out.console.print("[dim]No semantic concept hits.[/dim]")

    # -- Neighbours -----------------------------------------------------
    neighbors = data.get("neighbors", [])
    if neighbors:
        _render_neighbors(out, neighbors)
    else:
        out.console.print("[dim]No graph neighbours found.[/dim]")

    # -- Relation evidence ----------------------------------------------
    rel_evidence = data.get("relation_evidence", [])
    if rel_evidence:
        _render_relation_evidence(out, rel_evidence)
    else:
        out.console.print("[dim]No relation evidence found.[/dim]")

    # -- Evidence cards -------------------------------------------------
    cards = data.get("evidence_cards", [])
    if cards:
        _render_evidence_cards(out, cards)
    else:
        out.console.print("[dim]No evidence cards assembled.[/dim]")

    out.console.print()


def _render_file_hits(out: OutputManager, hits: List[dict]) -> None:
    """Render document file hits as a Rich table."""
    from rich.markup import escape as _esc
    from rich.table import Table

    table = Table(
        title="Document File Hits",
        show_header=True,
        header_style="bold",
        border_style="green",
    )
    table.add_column("#", justify="right", style="dim", min_width=3)
    table.add_column("Term", style="cyan", min_width=10)
    table.add_column("Title / Path", min_width=30)
    table.add_column("Line", justify="right", min_width=5)
    table.add_column("Excerpt", min_width=40, max_width=80)

    for idx, hit in enumerate(hits, 1):
        doc = hit.get("document") or {}
        title = doc.get("title") or doc.get("read_path") or "?"
        line = str(doc.get("line_start", ""))
        term = hit.get("matched_term", "")
        excerpt = hit.get("text", "")
        if len(excerpt) > 80:
            excerpt = excerpt[:77] + "..."
        table.add_row(
            str(idx), _esc(term), _esc(title), _esc(line), _esc(excerpt),
        )
    out.console.print(table)


def _render_semantic_hits(out: OutputManager, hits: List[dict]) -> None:
    """Render semantic concept hits as a Rich table."""
    from rich.markup import escape as _esc
    from rich.table import Table

    table = Table(
        title="Semantic Concept Hits",
        show_header=True,
        header_style="bold",
        border_style="magenta",
    )
    table.add_column("#", justify="right", style="dim", min_width=3)
    table.add_column("Score", justify="right", min_width=6)
    table.add_column("Family ID", style="cyan", min_width=12)
    table.add_column("Name", min_width=20)
    table.add_column("Query", style="dim", min_width=10)

    for idx, item in enumerate(hits, 1):
        # Honest score rendering: LIKE-fallback matches have no real cosine
        # similarity, so show "—" instead of a fabricated number.
        mode = item.get("match_mode")
        if mode == "name-like fallback":
            score = "[dim]name-match[/dim]"
        else:
            raw = item.get("score")
            if raw is None:
                raw = item.get("_score")
            score = f"{float(raw or 0.0):.3f}" if raw is not None else "[dim]—[/dim]"
        fid = item.get("family_id", "")
        name = item.get("name") or item.get("target_name") or ""
        query = item.get("matched_query", "")
        table.add_row(str(idx), score, _esc(fid), _esc(name), _esc(query))
    out.console.print(table)


def _render_neighbors(out: OutputManager, neighbors: List[dict]) -> None:
    """Render graph neighbour expansion results as a Rich table."""
    from rich.markup import escape as _esc
    from rich.table import Table

    table = Table(
        title="Graph Neighbours",
        show_header=True,
        header_style="bold",
        border_style="yellow",
    )
    table.add_column("#", justify="right", style="dim", min_width=3)
    table.add_column("Family ID", style="cyan", min_width=12)
    table.add_column("Name", min_width=20)
    table.add_column("Depth", justify="right", min_width=5)

    for idx, nb in enumerate(neighbors, 1):
        fid = nb.get("family_id", "")
        name = nb.get("name") or ""
        d = str(nb.get("depth", ""))
        table.add_row(str(idx), _esc(fid), _esc(name), _esc(d))
    out.console.print(table)


def _render_relation_evidence(out: OutputManager, evidence: List[dict]) -> None:
    """Render relation evidence rows as a Rich table."""
    from rich.markup import escape as _esc
    from rich.table import Table

    from ._helpers import compact_text

    table = Table(
        title="Relation Evidence",
        show_header=True,
        header_style="bold",
        border_style="blue",
    )
    table.add_column("#", justify="right", style="dim", min_width=3)
    table.add_column("Relation", style="cyan", min_width=14)
    table.add_column("Entities", min_width=20)
    table.add_column("Source", min_width=20)
    table.add_column("Excerpt", min_width=40, max_width=80)

    for idx, ev in enumerate(evidence, 1):
        rel_name = ev.get("relation_name") or ev.get("relation_family_id", "")
        e1 = ev.get("entity1_name", "")
        e2 = ev.get("entity2_name", "")
        entities = f"{e1} -- {e2}" if e1 and e2 else e1 or e2 or "?"
        source = ev.get("title") or ev.get("read_path") or ""
        excerpt = compact_text(ev.get("source_text", ""))
        table.add_row(
            str(idx),
            _esc(rel_name),
            _esc(entities),
            _esc(source),
            _esc(excerpt),
        )
    out.console.print(table)


def _render_evidence_cards(out: OutputManager, cards: List[dict]) -> None:
    """Render assembled evidence cards as Rich panels."""
    from rich.markup import escape as _esc
    from rich.panel import Panel

    from ._helpers import compact_text

    for idx, card in enumerate(cards, 1):
        claim = _esc(card.get("claim_hint", ""))
        doc = card.get("document") or {}
        title = _esc(doc.get("title") or doc.get("read_path") or "(no title)")
        verification = _esc(card.get("verification", ""))
        matched = [_esc(m) for m in card.get("matched_terms", [])]
        excerpt = _esc(card.get("source_excerpt", ""))

        body_parts: list[str] = []
        if claim:
            body_parts.append(f"[bold]Claim:[/bold] {claim}")
        body_parts.append(f"[bold]Document:[/bold] {title}")
        if verification:
            body_parts.append(f"[bold]Verification:[/bold] {verification}")
        if matched:
            body_parts.append(f"[bold]Matched terms:[/bold] {', '.join(matched)}")
        if excerpt:
            body_parts.append(f"[dim]{excerpt}[/dim]")

        concepts = card.get("concepts") or []
        if concepts:
            cpts = ", ".join(
                _esc(c.get("name", c.get("family_id", "?"))) for c in concepts
            )
            body_parts.append(f"[bold]Concepts:[/bold] {cpts}")

        episode = card.get("episode")
        if episode and episode.get("heading_path"):
            body_parts.append(
                f"[bold]Heading:[/bold] {_esc(episode['heading_path'])}"
            )

        out.console.print(Panel(
            "\n".join(body_parts),
            title=f"Evidence Card #{idx}",
            border_style="green",
        ))
