"""``deep-dream find`` -- quick concept search using BM25 full-text search.

For semantic search, use ``deep-dream concept search --semantic`` instead.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import click

from ._exit_codes import OK


def _escape_like(value: str) -> str:
    """Escape LIKE wildcard characters (%_) so they match literally.

    Uses '!' as the ESCAPE character to avoid backslash quoting issues
    in Python triple-quoted SQL strings.
    """
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


# ------------------------------------------------------------------
# Row formatting
# ------------------------------------------------------------------

def _concept_row(c: Dict[str, Any]) -> List[str]:
    """Format a single concept dict into a table row."""
    fid = c.get("family_id") or ""
    name = c.get("name") or c.get("canonical_name") or fid or ""
    role = c.get("role") or ""
    confidence = c.get("confidence")
    conf_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else ""
    summary = (c.get("content") or c.get("canonical_content") or name)[:50]
    return [
        fid,
        name[:30],
        role,
        conf_str,
        summary,
    ]


# ------------------------------------------------------------------
# Click command
# ------------------------------------------------------------------

@click.command()
@click.argument("query")
@click.option(
    "--graph",
    default=None,
    help="Graph ID [default: library]",
)
@click.option(
    "--role",
    type=click.Choice(["document", "episode", "entity", "relation"]),
    default=None,
    help="Filter by concept role.",
)
@click.option(
    "--limit",
    default=20,
    type=int,
    show_default=True,
    help="Maximum number of results.",
)
@click.option(
    "--time-point",
    default=None,
    help="Temporal snapshot (ISO timestamp).",
)
@click.pass_context
def find(
    ctx: click.Context,
    query: str,
    graph: Optional[str],
    role: Optional[str],
    limit: int,
    time_point: Optional[str],
) -> None:
    """Quick concept search using BM25 full-text search.

    For semantic search, use 'deep-dream concept search --semantic'.

    \b
    Examples:
      deep-dream find "machine learning"
      deep-dream find "transformer" --role entity --limit 10
    """
    from ._output import OutputManager, format_confidence

    out = OutputManager(ctx)
    cli_ctx = ctx.obj

    graph_id = cli_ctx.get_active_graph(graph)

    concepts: List[Dict[str, Any]] = []
    with cli_ctx.get_storage(graph_id) as storage:
        try:
            # Primary: search entity_families by name (most useful for users).
            conn = storage._conn()
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
            concepts = [
                {"family_id": r[0], "name": r[1], "role": r[2],
                 "content": r[3], "observation_count": r[4]}
                for r in rows
            ]

            # Fallback: if entity name search found nothing, try BM25 on episodes
            # and resolve to entities from there.
            if not concepts:
                bm25_results = storage.search_concepts_by_bm25(
                    query, role=role, limit=limit, time_point=time_point,
                )
                seen = set()
                for ep in bm25_results:
                    ep_id = ep.get("episode_id")
                    if not ep_id:
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
                            concepts.append({
                                "family_id": fid,
                                "name": er[1],
                                "role": "entity",
                                "content": er[2],
                            })
                    if len(concepts) >= limit:
                        break

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
    click.echo(
        f'\n  Use "deep-dream concept get <family_id>" for details.',
        err=True,
    )
