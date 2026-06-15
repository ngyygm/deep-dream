"""``deep-dream find`` -- quick concept search.

Default path searches entity families by name (most useful for users). With
``--role`` the search stays coherent: each role filters a base of its own kind
rather than collapsing every role onto episode-content BM25 expansion.

For semantic search, use ``deep-dream concept search --semantic`` instead.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import click

from ._exit_codes import ARGS, OK


def _escape_like(value: str) -> str:
    """Escape LIKE wildcard characters (%_) so they match literally.

    Uses '!' as the ESCAPE character to avoid backslash quoting issues
    in Python triple-quoted SQL strings.
    """
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


# Characters that FTS5 treats as query syntax / operators. A raw query
# containing these (e.g. ``[水滴]``, ``a AND b``, ``foo*``) raises
# ``fts5: syntax error`` from SQLite. We strip the operators down to a
# safe bareword so MATCH never leaks a syntax-error traceback to the user.
_FTS5_SPECIAL = re.compile(r'[\[\]()":*^+\-{}\s]+')


def _sanitize_fts_query(query: str) -> str:
    """Reduce a user query to an FTS5-safe bareword list.

    FTS5 MATCH chokes on bracketed terms (``[水滴]``) and on operator
    characters (``*``, ``"``, ``-``...). Rather than risk a syntax error,
    we split on those special chars and emit each non-empty token quoted
    with double quotes (FTS5 phrase form), joined by implicit AND. A query
    that reduces to nothing (e.g. only punctuation) returns "" so callers
    can fall back to a LIKE search or report 'no concepts found'.
    """
    tokens = [t for t in _FTS5_SPECIAL.split(query) if t]
    if not tokens:
        return ""
    # Quote each token as a phrase; FTS5 treats adjacent phrases as AND.
    # A literal double-quote inside a token is impossible here because
    # double-quote is in the split class.
    return " ".join(f'"{t}"' for t in tokens)


# ------------------------------------------------------------------
# Row formatting
# ------------------------------------------------------------------


def _concept_row(c: Dict[str, Any]) -> List[str]:
    """Format a single concept dict into a table row."""
    fid = c.get("family_id") or ""
    name = c.get("name") or c.get("canonical_name") or fid or ""
    role = c.get("role") or ""
    confidence = c.get("confidence") or c.get("_score")
    conf_str = ""
    if isinstance(confidence, (int, float)):
        conf_str = f"{confidence:.2f}"
    summary = (c.get("content") or c.get("canonical_content") or name)[:50]
    return [
        fid,
        name[:30],
        role,
        conf_str,
        summary,
    ]


# ------------------------------------------------------------------
# Per-role search helpers
# ------------------------------------------------------------------


def _entity_search(storage, conn, query: str, limit: int,
                   time_point: Optional[str]) -> List[Dict[str, Any]]:
    """Entity role: name LIKE first (consistent with default), BM25 fallback.

    The default ``find X`` path searches entity_families.canonical_name.
    ``--role entity`` must agree with that, so we run the same name search
    first; only if it finds nothing do we expand entities from
    episode-content BM25 (honoring --time-point).
    """
    concepts = _entity_name_search(conn, query, limit)
    if concepts:
        return concepts

    sanitized = _sanitize_fts_query(query)
    if not sanitized:
        return []
    try:
        bm25_results = storage.search_concepts_by_bm25(
            sanitized, role="entity", limit=limit, time_point=time_point,
        )
    except Exception:
        return []

    seen: set = set()
    out: List[Dict[str, Any]] = []
    for ep in bm25_results:
        ep_id = ep.get("episode_id")
        if not ep_id:
            # search_concepts_by_bm25 already lifts matched episodes to
            # entity dicts when role="entity".
            fid = ep.get("family_id")
            if fid and fid not in seen:
                seen.add(fid)
                out.append({
                    "family_id": fid,
                    "name": ep.get("name", ""),
                    "role": "entity",
                    "content": ep.get("content", ""),
                    "_score": ep.get("_score"),
                })
            if len(out) >= limit:
                break
            continue
        ent_rows = conn.execute(
            """
            SELECT DISTINCT ef.entity_family_id, ef.canonical_name, ef.canonical_content
            FROM entity_mentions em
            JOIN entity_families ef ON ef.entity_family_id = em.entity_family_id
            WHERE em.episode_id = ?
            LIMIT 5
            """,
            (ep_id,),
        ).fetchall()
        for er in ent_rows:
            fid = er[0]
            if fid not in seen:
                seen.add(fid)
                out.append({
                    "family_id": fid,
                    "name": er[1],
                    "role": "entity",
                    "content": er[2],
                })
        if len(out) >= limit:
            break
    return out


def _entity_name_search(conn, query: str, limit: int) -> List[Dict[str, Any]]:
    """Search entity_families.canonical_name by LIKE (the default base)."""
    like_pattern = f"%{_escape_like(query)}%"
    rows = conn.execute(
        """
        SELECT ef.entity_family_id AS family_id,
               ef.canonical_name AS name,
               'entity' AS role,
               ef.canonical_content AS content,
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
    return [
        {"family_id": r[0], "name": r[1], "role": r[2],
         "content": r[3], "observation_count": r[4]}
        for r in rows
    ]


def _relation_search(conn, query: str, limit: int) -> List[Dict[str, Any]]:
    """Relation role: rank relations whose content mentions the query.

    Previously ``--role relation`` collapsed onto episode-content BM25 and
    returned one arbitrary ``fetchone`` relation per matched episode. That
    gave incoherent, near-random results. Here we instead score relations
    directly by their ``canonical_content`` (LIKE), resolve endpoint names
    so the row shows what each relation connects, and rank by mention
    frequency as a tie-breaker.
    """
    like_pattern = f"%{_escape_like(query)}%"
    rows = conn.execute(
        """
        SELECT rf.relation_family_id AS family_id,
               rf.subject_entity_family_id,
               rf.object_entity_family_id,
               rf.canonical_content AS content,
               (SELECT count(*) FROM relation_assertions ra
                WHERE ra.relation_family_id = rf.relation_family_id
                  AND ra.status = 'active') AS assertion_count
        FROM relation_families rf
        WHERE rf.canonical_content LIKE ? ESCAPE '!'
           OR EXISTS (
               SELECT 1 FROM entity_families ef
               WHERE ef.entity_family_id IN (
                   rf.subject_entity_family_id, rf.object_entity_family_id)
                 AND ef.canonical_name LIKE ? ESCAPE '!'
           )
        ORDER BY assertion_count DESC
        LIMIT ?
        """,
        (like_pattern, like_pattern, limit),
    ).fetchall()
    if not rows:
        return []

    # Resolve endpoint entity names in one pass.
    fids: List[str] = []
    seen_fid: set = set()
    for r in rows:
        for fid in (r[1], r[2]):
            if fid and fid not in seen_fid:
                seen_fid.add(fid)
                fids.append(fid)
    name_map: Dict[str, str] = {}
    if fids:
        placeholders = ",".join("?" for _ in fids)
        for fid, nm in conn.execute(
            f"SELECT entity_family_id, canonical_name "
            f"FROM entity_families "
            f"WHERE entity_family_id IN ({placeholders})",
            fids,
        ).fetchall():
            name_map[fid] = nm or ""

    out: List[Dict[str, Any]] = []
    for r in rows:
        fid, subj, obj, content = r[0], r[1], r[2], r[3]
        e1 = name_map.get(subj, "")
        e2 = name_map.get(obj, "")
        if content and len(content) <= 80:
            label = f"{e1} {content} {e2}".strip()
        else:
            label = f"{e1} → {e2}".strip(" →") if (e1 or e2) else (content or "")
        out.append({
            "family_id": fid,
            "id": fid,
            "name": label,
            "content": content,
            "role": "relation",
            "entity1_name": e1,
            "entity2_name": e2,
        })
    return out


def _document_search(conn, query: str, limit: int) -> List[Dict[str, Any]]:
    """Document role: JOIN documents to populate name=title + a snippet.

    Previously ``--role document`` showed the raw ``doc_`` id as the name
    with no content. Here we JOIN to documents (title/relative_path) so the
    name is human-readable, and attach a short episode snippet where the
    query actually matched.
    """
    like_pattern = f"%{_escape_like(query)}%"
    rows = conn.execute(
        """
        SELECT d.document_id AS family_id,
               COALESCE(NULLIF(d.title, ''), d.relative_path, d.absolute_path,
                       d.document_id) AS name,
               d.title,
               d.relative_path,
               (SELECT count(*) FROM episodes ep
                WHERE ep.document_id = d.document_id AND ep.status = 'active')
                   AS episode_count
        FROM documents d
        WHERE d.status = 'active'
          AND (
              d.title LIKE ? ESCAPE '!'
              OR d.relative_path LIKE ? ESCAPE '!'
              OR d.absolute_path LIKE ? ESCAPE '!'
          )
        ORDER BY episode_count DESC
        LIMIT ?
        """,
        (like_pattern, like_pattern, like_pattern, limit),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        did, name, title, rel_path, ep_count = (
            r[0], r[1], r[2], r[3], r[4]
        )
        snippet = title or rel_path or ""
        out.append({
            "family_id": did,
            "id": did,
            "name": name,
            "content": snippet,
            "role": "document",
            "episode_count": ep_count,
        })
    return out


def _episode_search(storage, query: str, limit: int,
                    time_point: Optional[str]) -> List[Dict[str, Any]]:
    """Episode role: episode-content BM25 expansion (episodes ARE content)."""
    sanitized = _sanitize_fts_query(query)
    if not sanitized:
        return []
    try:
        raw = storage.search_concepts_by_bm25(
            sanitized, role="episode", limit=limit, time_point=time_point,
        )
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for r in raw:
        fid = r.get("family_id") or r.get("episode_family_id") or ""
        out.append({
            "family_id": fid,
            "id": r.get("episode_id", fid),
            "name": r.get("name", "") or r.get("heading_path", "") or "",
            "content": r.get("source_text", "") or r.get("content", "") or "",
            "role": "episode",
            "_score": r.get("_score"),
        })
    return out


# ------------------------------------------------------------------
# Click command
# ------------------------------------------------------------------

@click.command()
@click.argument("query")
@click.option(
    "--role",
    type=click.Choice(["document", "episode", "entity", "relation"]),
    default=None,
    help=(
        "Filter by concept role. 'entity' searches entity names; "
        "'relation' searches relation content; 'document' searches document "
        "titles/paths; 'episode' searches episode content via BM25."
    ),
)
@click.option(
    "--limit",
    default=20,
    type=click.IntRange(min=1),
    show_default=True,
    help="Maximum number of results (>= 1).",
)
@click.option(
    "--time-point",
    default=None,
    help=(
        "Temporal snapshot (ISO timestamp). Applies to content/episode "
        "search only (entity name and document title searches are static)."
    ),
)
@click.pass_context
def find(
    ctx: click.Context,
    query: str,
    role: Optional[str],
    limit: int,
    time_point: Optional[str],
) -> None:
    """Quick concept search.

    By default searches entity names. Use --role to search relations,
    documents, or episodes instead. For semantic (embedding) search, use
    'deep-dream concept search --semantic'.

    \b
    Examples:
      deep-dream find "machine learning"
      deep-dream find "transformer" --role entity --limit 10
      deep-dream find "causes" --role relation
    """
    from ._output import OutputManager

    out = OutputManager(ctx)
    cli_ctx = ctx.obj

    # Reject empty / whitespace-only queries up front (previously returned
    # 20 arbitrary entity families because '%' matches everything).
    if not query or not query.strip():
        out.error(
            "Query must not be empty.",
            hint="Provide a search term, e.g. 'deep-dream find \"三体\"'.",
            code=ARGS,
        )

    graph_id = cli_ctx.get_active_graph()

    concepts: List[Dict[str, Any]] = []
    with cli_ctx.get_storage(graph_id) as storage:
        try:
            conn = storage._conn()
            if role == "entity":
                concepts = _entity_search(
                    storage, conn, query, limit=limit, time_point=time_point,
                )
            elif role == "relation":
                concepts = _relation_search(conn, query, limit=limit)
            elif role == "document":
                concepts = _document_search(conn, query, limit=limit)
            elif role == "episode":
                concepts = _episode_search(
                    storage, query, limit=limit, time_point=time_point,
                )
            else:
                # Default: search entity_families by name (most useful for
                # users). Falls back to episode-content BM25 expansion if
                # the name search finds nothing.
                concepts = _entity_name_search(conn, query, limit)
                if not concepts:
                    concepts = _entity_search(
                        storage, conn, query, limit=limit,
                        time_point=time_point,
                    )

        except (ZeroDivisionError, ValueError) as exc:
            out.error(
                f"Search failed: {exc}",
                hint="Try a different query or use 'deep-dream concept search --semantic'.",
                code=1,
            )

    data = {"concepts": concepts, "total": len(concepts)}

    if out.is_json:
        out.result(data, meta={"graph_id": graph_id})
        return

    # ---- Rich human-readable output ----
    if not concepts:
        click.echo("No concepts found.", err=True)
        return

    columns = ["Family ID", "Concept", "Role", "Conf.", "Summary"]
    rows = [_concept_row(c) for c in concepts]

    out.table(
        f'Results for "{query}" ({len(concepts)} found)',
        columns,
        rows,
    )
    # Role-condition the footer hint: documents/episodes have their own
    # get commands, not 'concept get'.
    if role == "document":
        hint = '\n  Use "deep-dream docs <document_id>" for details.'
    elif role == "episode":
        hint = '\n  Use "deep-dream episode get <episode_id>" for details.'
    elif role == "relation":
        hint = '\n  Use "deep-dream relation get <family_id>" for details.'
    else:
        hint = '\n  Use "deep-dream concept get <family_id>" for details.'
    click.echo(hint, err=True)
