"""``concept`` command group — search, inspect, and manage concepts.

Subcommands
-----------
search      Search concepts by BM25, semantic, or hybrid mode.
get         Display full concept details in a Rich panel.
trace       Trace concept provenance back to source episodes.
neighbors   Expand concept graph neighbors.
versions    List all versions of a concept family.
mentions    Get episodes mentioning a concept.
update      Manually update a concept's name, content, or confidence.
suggest     Suggest concept names by prefix.
duplicates  Detect potential duplicate entities.
merge       Merge two entity families into one.

All commands respect the global ``--json`` / ``--quiet`` / ``--no-color``
flags handled by :class:`OutputManager`.
"""
from __future__ import annotations

import json as _json
from typing import Any, Dict, List, Optional

import click

from rich.markup import escape as _rich_escape
from rich.panel import Panel as _Panel

from ._ctx import CliContext
from ._exit_codes import ERROR, NOT_FOUND, OK
from ._helpers import resolve_concept_id
from ._output import OutputManager, format_confidence, format_timestamp


def _escape_like(value: str) -> str:
    """Escape LIKE wildcard characters (%_) so they match literally.

    Uses '!' as the ESCAPE character to avoid backslash quoting issues
    in Python triple-quoted SQL strings.
    """
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def _format_line_range(line_start: Any, line_end: Any) -> str:
    """Format a line range, showing '-' when both are 0 or missing."""
    try:
        s = int(line_start) if line_start is not None else 0
        e = int(line_end) if line_end is not None else 0
    except (ValueError, TypeError):
        return "-"
    if s == 0 and e == 0:
        return "-"
    if s == e:
        return str(s)
    return f"{s}-{e}"


# ------------------------------------------------------------------
# Click group
# ------------------------------------------------------------------

@click.group()
def concept() -> None:
    """Search, inspect, and manage concepts."""
    pass


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _resolve_config_path(ctx: click.Context) -> str:
    """Extract the ``--config`` path from the Click root context."""
    cur = ctx
    while cur.parent is not None:
        cur = cur.parent
    params = getattr(ctx.obj, "_click_params", None) or {}
    return params.get("config", "service_config.json")


def _get_graph_id(ctx: click.Context, explicit: Optional[str] = None) -> str:
    """Return the active graph ID (always LIBRARY_ID in single-library mode)."""
    obj: CliContext = ctx.obj
    return obj.get_active_graph(explicit)


# ------------------------------------------------------------------
# concept search
# ------------------------------------------------------------------

@concept.command()
@click.argument("query")
@click.option(
    "--mode",
    type=click.Choice(["name", "bm25", "semantic", "hybrid"], case_sensitive=False),
    default="name",
    show_default=True,
    help="Search mode: name (entity name LIKE), bm25 (episode FTS), semantic (embedding), or hybrid.",
)
@click.option(
    "--role",
    type=click.Choice(["document", "episode", "entity", "relation"]),
    default=None,
    help="Filter by concept role.",
)
@click.option(
    "--limit",
    type=int,
    default=20,
    show_default=True,
    help="Maximum number of results.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.3,
    show_default=True,
    help="Minimum similarity threshold (semantic/hybrid mode).",
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    mode: str,
    role: Optional[str],
    limit: int,
    threshold: float,
) -> None:
    """Search concepts by query string."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)
    graph_id = _get_graph_id(ctx)

    with obj.get_storage(graph_id) as storage:
        if mode == "name":
            # Direct SQL LIKE search on entity canonical_name — works without embeddings or FTS.
            conn = storage._conn()
            like_pattern = f"%{_escape_like(query)}%"
            rows = conn.execute(
                """
                SELECT ef.entity_family_id AS family_id,
                       ef.canonical_name AS name,
                       'entity' AS role,
                       (SELECT count(*) FROM entity_observations eo
                        WHERE eo.entity_family_id = ef.entity_family_id
                          AND eo.status = 'active') AS observation_count
                FROM entity_families ef
                WHERE ef.canonical_name LIKE ? ESCAPE '!'
                ORDER BY observation_count DESC
                LIMIT ?
                """,
                (like_pattern, limit),
            ).fetchall()
            # Use explicit column names since row_factory may vary
            concepts = [
                {"family_id": r[0], "name": r[1], "role": r[2], "observation_count": r[3]}
                for r in rows
            ]
        elif mode == "semantic":
            result = storage.agent_semantic_search(
                query, role=role, top_k=limit, threshold=threshold,
            )
            concepts = result.get("results", [])
        elif mode == "hybrid":
            if hasattr(storage, "search_entities_by_bm25"):
                raw_entities = storage.search_entities_by_bm25(query, limit=limit)
                bm25_results = []
                seen_fids: set[str] = set()
                for ent in raw_entities:
                    if isinstance(ent, dict):
                        fid = ent.get("entity_family_id") or ent.get("family_id", "")
                        name = ent.get("canonical_name") or ent.get("name", "")
                        score = ent.get("_score", "")
                    else:
                        fid = getattr(ent, "family_id", "")
                        name = getattr(ent, "name", "")
                        score = getattr(ent, "_score", "")
                    if fid and fid not in seen_fids:
                        seen_fids.add(fid)
                        bm25_results.append({
                            "family_id": fid,
                            "name": name,
                            "role": "entity",
                            "observation_count": "",
                            "score": score,
                        })
            else:
                bm25_results = storage.search_concepts_by_bm25(
                    query, role=role, limit=limit,
                )
            sem_result = storage.agent_semantic_search(
                query, role=role, top_k=limit, threshold=threshold,
            )
            semantic_results = sem_result.get("results", [])
            # De-duplicate: BM25 first, then append unique semantic hits.
            seen_ids = {c.get("family_id") for c in bm25_results}
            for c in semantic_results:
                fid = c.get("family_id")
                if fid and fid not in seen_ids:
                    bm25_results.append(c)
                    seen_ids.add(fid)
            concepts = bm25_results
        else:  # bm25
            # search_concepts_by_bm25 returns raw episode-level hits
            # without family_id/role.  Use search_entities_by_bm25 instead
            # to get actual entity results with proper family resolution.
            if hasattr(storage, "search_entities_by_bm25"):
                raw_entities = storage.search_entities_by_bm25(query, limit=limit)
                concepts = []
                seen_fids: set[str] = set()
                for ent in raw_entities:
                    if isinstance(ent, dict):
                        fid = ent.get("entity_family_id") or ent.get("family_id", "")
                        name = ent.get("canonical_name") or ent.get("name", "")
                        score = ent.get("_score", "")
                    else:
                        fid = getattr(ent, "family_id", "")
                        name = getattr(ent, "name", "")
                        score = getattr(ent, "_score", "")
                    if fid and fid not in seen_fids:
                        seen_fids.add(fid)
                        concepts.append({
                            "family_id": fid,
                            "name": name,
                            "role": "entity",
                            "observation_count": "",
                            "score": score,
                        })
            else:
                concepts = storage.search_concepts_by_bm25(
                    query, role=role, limit=limit,
                )

    data = {
        "concepts": concepts,
        "total": len(concepts),
        "query": query,
        "mode": mode,
    }
    meta = {
        "used": {
            "raw_files": False,
            "sqlite": True,
            "semantic": mode in ("semantic", "hybrid"),
            "graph_traversal": False,
            "api": False,
        },
    }

    if out.is_json:
        from ._output import json_result
        payload = json_result("concept search", data, meta=meta)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    if not concepts:
        out.console.print("[dim]No concepts found.[/dim]")
        return

    # Build table columns based on available data
    has_obs_count = any(c.get("observation_count") for c in concepts)
    has_score = any(c.get("score") for c in concepts)
    if has_obs_count:
        columns = ("Family ID", "Name", "Role", "Observations")
        rows = []
        for c in concepts:
            rows.append([
                c.get("family_id", ""),
                c.get("name", ""),
                c.get("role", ""),
                str(c.get("observation_count", "")),
            ])
    elif has_score:
        columns = ("Family ID", "Name", "Role", "Score")
        rows = []
        for c in concepts:
            sc = c.get("score")
            try:
                sc_str = f"{float(sc):.3f}" if sc != "" else ""
            except (ValueError, TypeError):
                sc_str = str(sc) if sc else ""
            rows.append([
                c.get("family_id", ""),
                c.get("name", ""),
                c.get("role", ""),
                sc_str,
            ])
    else:
        columns = ("Family ID", "Name", "Role", "Confidence")
        rows = []
        for c in concepts:
            rows.append([
                c.get("family_id", ""),
                c.get("name", ""),
                c.get("role", ""),
                format_confidence(c.get("confidence")),
            ])
    out.table(f"Concept Search ({mode})", columns, rows)


# ------------------------------------------------------------------
# concept get
# ------------------------------------------------------------------

@concept.command()
@click.argument("family_id")
@click.pass_context
def get(ctx: click.Context, family_id: str) -> None:
    """Display full details for a concept."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)
    graph_id = _get_graph_id(ctx)

    with obj.get_storage(graph_id) as storage:
        concept_data = storage.get_concept_by_family_id(family_id)
        if concept_data is None:
            out.error(
                f"Concept not found: {family_id}",
                hint="Use 'deep-dream concept search <query>' to find concepts.",
                code=NOT_FOUND,
            )
            return  # unreachable

        # Gather relations for this concept.
        relations: List[Dict[str, Any]] = []
        try:
            neighbors = storage.get_concept_neighbors(
                concept_data["family_id"], max_depth=1, max_results=20,
            )
            relations = neighbors
            # Resolve neighbor names while storage is still open.
            for r in relations:
                target_fid = r.get("family_id", "")
                if target_fid:
                    try:
                        _nbr = storage.get_concept_by_family_id(target_fid)
                        if _nbr:
                            r["name"] = _nbr.get("name", target_fid)
                    except Exception:
                        pass
        except Exception:
            pass

        # Gather source mentions.
        mentions: List[Dict[str, Any]] = []
        try:
            from ._helpers import concept_source_evidence
            mentions = concept_source_evidence(
                storage, [concept_data["family_id"]], limit=5,
            )
        except Exception:
            pass

    data = {
        "concept": concept_data,
        "relations": relations,
        "mentions": mentions,
    }

    if out.is_json:
        from ._output import json_result
        payload = json_result("concept get", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    # Build Rich panel content directly (bypass out.panel which escapes markup).
    name_val = concept_data.get("name", "")
    panel_lines: list[str] = []
    panel_lines.append(f"[bold]Family ID:[/bold] {_rich_escape(concept_data.get('family_id', ''))}")
    panel_lines.append(f"[bold]Name:[/bold]      {_rich_escape(name_val)}")
    panel_lines.append(f"[bold]Role:[/bold]      {_rich_escape(concept_data.get('role', ''))}")
    panel_lines.append(f"[bold]Confidence:[/bold] {format_confidence(concept_data.get('confidence'))}")

    content = concept_data.get("content", "")
    if content:
        panel_lines.append("")
        panel_lines.append("[bold]Content:[/bold]")
        panel_lines.append(_rich_escape(content))

    created = concept_data.get("created_at") or concept_data.get("first_seen")
    if created:
        panel_lines.append(f"[dim]First seen: {format_timestamp(created)}[/dim]")

    out.console.print(_Panel(
        "\n".join(panel_lines),
        title=f"Concept: {_rich_escape(name_val or family_id)}",
    ))

    # Show relations table — names were resolved in the storage block above.
    if relations:
        out.console.print()
        columns = ("Relation", "Target", "Depth")
        rows = []
        for r in relations:
            target_name = r.get("name") or r.get("target_name") or r.get("family_id", "")
            rows.append([
                r.get("relation_name", r.get("edge_type", "")),
                target_name,
                str(r.get("depth", "")),
            ])
        out.table("Relations", columns, rows)

    # Show source mentions.
    if mentions:
        out.console.print()
        columns = ("Source", "Title", "Lines")
        rows = []
        for m in mentions[:5]:
            rows.append([
                m.get("source_mode", ""),
                m.get("title", ""),
                _format_line_range(m.get("line_start"), m.get("line_end")),
            ])
        out.table("Source Mentions (top 5)", columns, rows)


# ------------------------------------------------------------------
# concept trace
# ------------------------------------------------------------------

@concept.command()
@click.argument("family_id")
@click.option(
    "--time-point",
    default=None,
    help="ISO timestamp to trace at a specific point in time.",
)
@click.pass_context
def trace(ctx: click.Context, family_id: str, time_point: Optional[str]) -> None:
    """Trace concept provenance back to source episodes."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)
    graph_id = _get_graph_id(ctx)

    with obj.get_storage(graph_id) as storage:
        # Validate concept exists
        concept_data = storage.get_concept_by_family_id(family_id)
        if concept_data is None:
            out.error(
                f"Concept not found: {family_id}",
                hint="Use 'deep-dream concept search <query>' to find concepts.",
                code=NOT_FOUND,
            )
            return  # unreachable
        provenance = storage.get_concept_provenance(
            family_id, time_point=time_point,
        )

    data = {
        "family_id": family_id,
        "provenance": provenance,
    }

    if out.is_json:
        from ._output import json_result
        payload = json_result("concept trace", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    if not provenance:
        out.console.print(f"[dim]No provenance found for {family_id}.[/dim]")
        return

    # Show provenance details.
    if isinstance(provenance, dict):
        lines: list[str] = []
        for key, value in provenance.items():
            if isinstance(value, list):
                lines.append(f"[bold]{_rich_escape(str(key))}:[/bold] {len(value)} items")
            else:
                lines.append(f"[bold]{_rich_escape(str(key))}:[/bold] {_rich_escape(str(value))}")
        out.console.print(_Panel(
            "\n".join(lines),
            title=f"Provenance: {_rich_escape(family_id)}",
        ))

        # If provenance has episodes, show as table.
        episodes = provenance.get("episodes") or provenance.get("sources") or []
        if episodes and isinstance(episodes, list):
            out.console.print()
            columns = ("Episode ID", "Source", "Heading")
            rows = []
            for ep in episodes:
                rows.append([
                    ep.get("version_id", ep.get("episode_id", "")),
                    ep.get("source_text", "")[:60],
                    ep.get("heading_path", ""),
                ])
            out.table("Source Episodes", columns, rows)

    elif isinstance(provenance, list):
        # Provenance is a list of mention/evidence dicts.
        columns = ("Edge Type", "Episode ID", "Surface Text", "Offset")
        rows: list[list[str]] = []
        for item in provenance:
            evidence = item.get("evidence", {})
            if isinstance(evidence, dict):
                surface = evidence.get("surface_text", "")
                offset = f"{evidence.get('start_offset', '?')}-{evidence.get('end_offset', '?')}"
            else:
                surface = ""
                offset = ""
            rows.append([
                item.get("edge_type", ""),
                item.get("episode_id", ""),
                _rich_escape(str(surface)),
                offset,
            ])
        out.table(
            f"Provenance: {_rich_escape(family_id)} ({len(provenance)} mention(s))",
            columns,
            rows,
        )
    else:
        out.console.print(str(provenance))


# ------------------------------------------------------------------
# concept neighbors
# ------------------------------------------------------------------

@concept.command()
@click.argument("family_id")
@click.option(
    "--depth",
    type=int,
    default=1,
    show_default=True,
    help="Maximum traversal depth.",
)
@click.option(
    "--limit",
    type=int,
    default=50,
    show_default=True,
    help="Maximum number of neighbor results.",
)
@click.pass_context
def neighbors(
    ctx: click.Context,
    family_id: str,
    depth: int,
    limit: int,
) -> None:
    """Expand concept graph neighbors."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)
    graph_id = _get_graph_id(ctx)

    with obj.get_storage(graph_id) as storage:
        fid = resolve_concept_id(storage, family_id)
        if fid is None:
            out.error(
                f"Concept not found: {family_id}",
                hint="Use 'deep-dream concept search <query>' to find concepts.",
                code=NOT_FOUND,
            )
            return  # unreachable
        results = storage.get_concept_neighbors(
            fid, max_depth=depth, max_results=limit,
        )
        # Resolve neighbor names while storage is still open.
        for n in results:
            target_fid = n.get("family_id", "")
            if target_fid and not n.get("name"):
                try:
                    _nbr = storage.get_concept_by_family_id(target_fid)
                    if _nbr:
                        n["name"] = _nbr.get("name", target_fid)
                except Exception:
                    pass

    data = {
        "family_id": fid,
        "neighbors": results,
        "total": len(results),
    }
    meta = {
        "used": {
            "raw_files": False,
            "sqlite": True,
            "semantic": False,
            "graph_traversal": True,
            "api": False,
        },
    }

    if out.is_json:
        from ._output import json_result
        payload = json_result("concept neighbors", data, meta=meta)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    if not results:
        out.console.print(f"[dim]No neighbors found for {fid}.[/dim]")
        return

    columns = ("Family ID", "Name", "Relation", "Depth")
    rows = []
    for n in results:
        rows.append([
            n.get("family_id", ""),
            n.get("name", ""),
            n.get("relation_name", n.get("edge_type", "")),
            str(n.get("depth", "")),
        ])
    out.table(f"Neighbors of {fid} (depth={depth})", columns, rows)


# ------------------------------------------------------------------
# concept versions
# ------------------------------------------------------------------

@concept.command()
@click.argument("family_id")
@click.pass_context
def versions(ctx: click.Context, family_id: str) -> None:
    """List all versions of a concept family."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)
    graph_id = _get_graph_id(ctx)

    with obj.get_storage(graph_id) as storage:
        version_list: List[Dict[str, Any]] = []
        # Validate concept exists
        concept_data = storage.get_concept_by_family_id(family_id)
        if concept_data is None:
            # Try resolve before giving up
            fid = resolve_concept_id(storage, family_id)
            if fid is None:
                out.error(
                    f"Concept not found: {family_id}",
                    hint="Use 'deep-dream concept search <query>' to find concepts.",
                    code=NOT_FOUND,
                )
                return  # unreachable
            family_id = fid
        if hasattr(storage, "get_concept_versions"):
            version_list = storage.get_concept_versions(family_id)
        if not version_list:
            # Fallback: resolve and try again with family_id.
            fid = resolve_concept_id(storage, family_id)
            if fid and fid != family_id and hasattr(storage, "get_concept_versions"):
                version_list = storage.get_concept_versions(fid)

    data = {
        "family_id": family_id,
        "versions": version_list,
        "total": len(version_list),
    }

    if out.is_json:
        from ._output import json_result
        payload = json_result("concept versions", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    if not version_list:
        out.console.print(f"[dim]No versions found for {family_id}.[/dim]")
        return

    columns = ("Version ID", "Name", "Episode", "Created")
    rows = []
    for v in version_list:
        rows.append([
            v.get("absolute_id", v.get("version_id", "")),
            v.get("name", ""),
            v.get("episode_id", ""),
            format_timestamp(v.get("processed_time")),
        ])
    out.table(f"Version History: {family_id}", columns, rows)


# ------------------------------------------------------------------
# concept mentions
# ------------------------------------------------------------------

@concept.command()
@click.argument("family_id")
@click.option(
    "--limit",
    type=int,
    default=20,
    show_default=True,
    help="Maximum number of episodes to return.",
)
@click.pass_context
def mentions(
    ctx: click.Context,
    family_id: str,
    limit: int,
) -> None:
    """Get episodes mentioning a concept."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)
    graph_id = _get_graph_id(ctx)

    with obj.get_storage(graph_id) as storage:
        mention_list: List[Dict[str, Any]] = []
        # Validate concept exists
        concept_data = storage.get_concept_by_family_id(family_id)
        if concept_data is None:
            out.error(
                f"Concept not found: {family_id}",
                hint="Use 'deep-dream concept search <query>' to find concepts.",
                code=NOT_FOUND,
            )
            return  # unreachable
        if hasattr(storage, "get_concept_mentions"):
            mention_list = storage.get_concept_mentions(family_id)
        if not mention_list:
            # Fallback: use SQL helper.
            fid = resolve_concept_id(storage, family_id)
            if fid:
                from ._helpers import concept_source_evidence
                mention_list = concept_source_evidence(
                    storage, [fid], limit=limit,
                )

    data = {
        "family_id": family_id,
        "mentions": mention_list,
        "total": len(mention_list),
    }

    if out.is_json:
        from ._output import json_result
        payload = json_result("concept mentions", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    if not mention_list:
        out.console.print(f"[dim]No mentions found for {family_id}.[/dim]")
        return

    columns = ("Episode ID", "Document", "Heading", "Lines", "Excerpt")
    rows = []
    for m in mention_list:
        excerpt = (m.get("source_text") or m.get("surface_text") or "")[:60]
        rows.append([
            m.get("episode_id", m.get("episode_version_id", m.get("version_id", ""))),
            m.get("title", ""),
            m.get("heading_path", ""),
            _format_line_range(m.get("line_start"), m.get("line_end")),
            excerpt,
        ])
    out.table(f"Mentions of {family_id}", columns, rows)


# ------------------------------------------------------------------
# concept update
# ------------------------------------------------------------------

@concept.command()
@click.argument("family_id")
@click.option(
    "--name",
    "new_name",
    default=None,
    help="New name for the concept.",
)
@click.option(
    "--content",
    "new_content",
    default=None,
    help="New content for the concept.",
)
@click.option(
    "--confidence",
    "new_confidence",
    type=float,
    default=None,
    help="New confidence value (0.0 - 1.0).",
)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
@click.pass_context
def update(
    ctx: click.Context,
    family_id: str,
    new_name: Optional[str],
    new_content: Optional[str],
    new_confidence: Optional[float],
    yes: bool,
) -> None:
    """Manually update a concept's name, content, or confidence."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)
    graph_id = _get_graph_id(ctx)

    # Validate that at least one field is specified.
    if new_name is None and new_content is None and new_confidence is None:
        out.error(
            "At least one of --name, --content, or --confidence is required.",
            hint="Specify the field(s) to update.",
            code=ERROR,
        )
        return  # unreachable

    # Validate confidence range.
    if new_confidence is not None and not (0.0 <= new_confidence <= 1.0):
        out.error(
            f"Confidence must be between 0.0 and 1.0, got {new_confidence}",
            code=ERROR,
        )
        return  # unreachable

    with obj.get_storage(graph_id) as storage:
        concept_data = storage.get_concept_by_family_id(family_id)
        if concept_data is None:
            out.error(
                f"Concept not found: {family_id}",
                hint="Use 'deep-dream concept search <query>' to find concepts.",
                code=NOT_FOUND,
            )
            return  # unreachable

        # Confirmation prompt.
        if not yes:
            if out.is_json:
                # In JSON mode, require --yes.
                out.error(
                    "Confirmation required. Use --yes to confirm.",
                    code=ERROR,
                )
                return  # unreachable
            out.console.print(
                f"  Updating concept [bold]{family_id}[/bold]:"
            )
            if new_name is not None:
                out.console.print(
                    f"    name: {concept_data.get('name', '')!r} -> {new_name!r}"
                )
            if new_content is not None:
                old_preview = (concept_data.get("content") or "")[:40]
                new_preview = new_content[:40]
                out.console.print(
                    f"    content: {old_preview!r}... -> {new_preview!r}..."
                )
            if new_confidence is not None:
                out.console.print(
                    f"    confidence: {concept_data.get('confidence', '')} -> {new_confidence}"
                )
            if not click.confirm("Apply this update?", default=False):
                out.console.print("[dim]Cancelled.[/dim]")
                raise SystemExit(0)

        # Perform the update.
        result: Dict[str, Any] = {"updated": False}
        if hasattr(storage, "update_concept_manual"):
            updates: Dict[str, Any] = {}
            if new_name is not None:
                updates["name"] = new_name
            if new_content is not None:
                updates["content"] = new_content
            if new_confidence is not None:
                updates["confidence"] = new_confidence
            result = storage.update_concept_manual(family_id, updates)
        else:
            result = {
                "updated": True,
                "family_id": family_id,
                "warning": "storage.update_concept_manual() not available",
            }

    data = {
        "family_id": family_id,
        "update": result,
        "fields": {
            "name": new_name,
            "content": new_content,
            "confidence": new_confidence,
        },
    }

    if out.is_json:
        from ._output import json_result
        payload = json_result("concept update", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    out.success(f"Updated concept {family_id}")


# ------------------------------------------------------------------
# concept suggest
# ------------------------------------------------------------------

@concept.command()
@click.argument("prefix")
@click.pass_context
def suggest(ctx: click.Context, prefix: str) -> None:
    """Suggest concept names by prefix."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)
    graph_id = _get_graph_id(ctx)

    with obj.get_storage(graph_id) as storage:
        suggestions: List[Dict[str, Any]] = []
        if hasattr(storage, "suggest_concepts"):
            suggestions = storage.suggest_concepts(prefix)

    data = {
        "prefix": prefix,
        "suggestions": suggestions,
        "total": len(suggestions),
    }

    if out.is_json:
        from ._output import json_result
        payload = json_result("concept suggest", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    if not suggestions:
        out.console.print(f"[dim]No suggestions for prefix '{_rich_escape(prefix)}'.[/dim]")
        return

    columns = ("Family ID", "Name", "Role")
    rows = []
    for s in suggestions:
        rows.append([
            s.get("family_id", ""),
            s.get("name", ""),
            s.get("role", ""),
        ])
    out.table(f"Suggestions for '{prefix}'", columns, rows)


# ------------------------------------------------------------------
# concept duplicates
# ------------------------------------------------------------------

@concept.command()
@click.pass_context
def duplicates(ctx: click.Context) -> None:
    """Detect potential duplicate entities."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)
    graph_id = _get_graph_id(ctx)

    with obj.get_storage(graph_id) as storage:
        dupes: List[Dict[str, Any]] = []
        if hasattr(storage, "find_duplicate_entities_fast"):
            dupes = storage.find_duplicate_entities_fast()

    data = {
        "duplicates": dupes,
        "total": len(dupes),
    }

    if out.is_json:
        from ._output import json_result
        payload = json_result("concept duplicates", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    if not dupes:
        out.console.print("[dim]No duplicate entities detected.[/dim]")
        return

    columns = ("Entity A", "Entity B", "Similarity", "Family A", "Family B")
    rows = []
    for d in dupes:
        rows.append([
            d.get("entity_a_name", d.get("name_a", "")),
            d.get("entity_b_name", d.get("name_b", "")),
            f"{d.get('similarity', d.get('score', 0)):.3f}",
            d.get("family_a_id", d.get("family_id_a", "")),
            d.get("family_b_id", d.get("family_id_b", "")),
        ])
    out.table("Duplicate Entities", columns, rows)


# ------------------------------------------------------------------
# concept merge
# ------------------------------------------------------------------

@concept.command()
@click.argument("source")
@click.argument("target")
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Confirm the merge operation.",
)
@click.pass_context
def merge(
    ctx: click.Context,
    source: str,
    target: str,
    yes: bool,
) -> None:
    """Merge two entity families (source into target).

    The SOURCE entity family is merged into the TARGET entity family.
    All relations and mentions of SOURCE are redirected to TARGET.
    This operation cannot be easily undone.
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)
    graph_id = _get_graph_id(ctx)

    if not yes:
        out.error(
            "Merge requires explicit confirmation. Use --yes to confirm.",
            hint=(
                "This operation merges SOURCE into TARGET and cannot be "
                "easily undone. Verify the family IDs before proceeding."
            ),
            code=ERROR,
        )
        return  # unreachable

    with obj.get_storage(graph_id) as storage:
        # Verify both entities exist.
        source_concept = storage.get_concept_by_family_id(source)
        target_concept = storage.get_concept_by_family_id(target)

        if source_concept is None:
            out.error(
                f"Source concept not found: {source}",
                code=NOT_FOUND,
            )
            return  # unreachable

        if target_concept is None:
            out.error(
                f"Target concept not found: {target}",
                code=NOT_FOUND,
            )
            return  # unreachable

        # Perform the merge.
        merge_result: Dict[str, Any] = {"merged": False}
        if hasattr(storage, "merge_entity_families"):
            merge_result = storage.merge_entity_families(target, [source])

        # Redirect relations.
        redirect_result: Dict[str, Any] = {"redirected": False}
        if hasattr(storage, "redirect_entity_relations"):
            redirect_result = storage.redirect_entity_relations(source, target)

    data = {
        "source": {
            "family_id": source,
            "name": source_concept.get("name", ""),
        },
        "target": {
            "family_id": target,
            "name": target_concept.get("name", ""),
        },
        "merge": merge_result,
        "redirect": redirect_result,
    }

    if out.is_json:
        from ._output import json_result
        payload = json_result("concept merge", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    out.success(
        f"Merged {source_concept.get('name', source)} -> "
        f"{target_concept.get('name', target)}"
    )
    if merge_result:
        out.console.print(
            f"  [dim]Merge: {_json.dumps(merge_result, ensure_ascii=False)}[/dim]"
        )
    if redirect_result:
        out.console.print(
            f"  [dim]Redirect: {_json.dumps(redirect_result, ensure_ascii=False)}[/dim]"
        )
