"""``episode`` command group — episode mapping and inspection.

Subcommands
-----------
from-file   Map a file path/line to episodes.
concepts    List concepts mentioned by an episode.
get         Get episode details.
content     Read episode source content.

All commands respect the global ``--json`` / ``--quiet`` / ``--no-color``
flags handled by :class:`OutputManager`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from rich.markup import escape as _rich_escape
from rich.panel import Panel as _Panel

from ._ctx import CliContext
from ._exit_codes import NOT_FOUND
from ._helpers import (
    document_file_payload,
    map_path_to_documents,
    read_sql,
)
from ._output import OutputManager, format_timestamp


# ------------------------------------------------------------------
# Shared option: --graph
# ------------------------------------------------------------------

_graph_option = click.option(
    "--graph",
    default=None,
    help="Graph ID (defaults to the active library).",
)


# ------------------------------------------------------------------
# Click group
# ------------------------------------------------------------------

@click.group()
def episode() -> None:
    """Episode mapping and inspection helpers."""
    pass


# ------------------------------------------------------------------
# episode from-file
# ------------------------------------------------------------------

@episode.command("from-file")
@click.argument("path", type=click.Path(exists=False))
@click.option("--line", type=int, default=None, help="Filter to episodes overlapping this line number.")
@click.option("--limit", type=int, default=50, show_default=True, help="Maximum episodes to return.")
@_graph_option
@click.pass_context
def from_file(ctx: click.Context, path: str, line: Optional[int], limit: int, graph: Optional[str]) -> None:
    """Map a file path (and optional line) to episodes."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    graph_id = obj.get_active_graph(graph)

    with obj.get_storage(graph_id) as storage:
        docs_data = [document_file_payload(storage, d) for d in map_path_to_documents(storage, path, limit=5)]
        episodes_list: list[dict] = []

        for doc in docs_data:
            params: Dict[str, Any] = {"doc": doc["document_version_id"]}
            line_clause = ""
            if line is not None:
                params["line"] = int(line)
                params["offset"] = -1
                resolved_path = doc.get("resolved_path")
                if resolved_path:
                    try:
                        content_lines = Path(resolved_path).read_text(encoding="utf-8").splitlines(keepends=True)
                        if 1 <= line <= len(content_lines):
                            params["offset"] = sum(len(ln) for ln in content_lines[: line - 1])
                    except OSError:
                        pass
                line_clause = """
                  AND (
                    (
                      line_start IS NOT NULL
                      AND COALESCE(CAST(line_start AS INTEGER), -1) <= :line
                      AND COALESCE(CAST(line_end AS INTEGER), 2147483647) >= :line
                    )
                    OR (
                      :offset >= 0
                      AND COALESCE(CAST(start_offset AS INTEGER), -1) <= :offset
                      AND COALESCE(CAST(end_offset AS INTEGER), -1) >= :offset
                    )
                  )
                """
            episodes_list.extend(read_sql(
                storage,
                f"""
                SELECT version_id AS episode_version_id, document_version_id,
                       heading_path, start_offset, end_offset, line_start,
                       line_end, source_path, source_text
                FROM v_episodes
                WHERE document_version_id = :doc {line_clause}
                ORDER BY start_offset
                """,
                params,
                limit=limit,
            ))

    data = {
        "documents": docs_data,
        "episodes": episodes_list,
        "total": len(episodes_list),
    }

    if out.is_json:
        payload = {"success": True, "command": "episode from-file", "graph_id": graph_id, "data": data}
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    columns = ("Episode ID", "Heading", "Lines", "Offset")
    table_rows: list[list[str]] = []
    for ep in episodes_list:
        table_rows.append((
            ep.get("episode_version_id", ""),
            ep.get("heading_path", ""),
            f"{ep.get('line_start', '?')}-{ep.get('line_end', '?')}",
            f"{ep.get('start_offset', '')}-{ep.get('end_offset', '')}",
        ))
    out.table(f"Episodes from {path} ({len(episodes_list)})", columns, table_rows)


# ------------------------------------------------------------------
# episode concepts
# ------------------------------------------------------------------

@episode.command()
@click.argument("episode_id")
@click.option("--limit", type=int, default=100, show_default=True, help="Maximum concepts to return.")
@_graph_option
@click.pass_context
def concepts(ctx: click.Context, episode_id: str, limit: int, graph: Optional[str]) -> None:
    """List concepts mentioned by an episode."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    graph_id = obj.get_active_graph(graph)

    with obj.get_storage(graph_id) as storage:
        concept_rows = read_sql(
            storage,
            """
            SELECT m.target_family_id AS family_id,
                   COALESCE(NULLIF(MAX(m.target_name), ''), lc.name, '') AS name,
                   lc.role, lc.content, lc.confidence
            FROM v_mentions m
            JOIN v_latest_concept lc
              ON lc.family_id = m.target_family_id
            WHERE m.episode_version_id = :episode
            GROUP BY m.target_family_id
            ORDER BY name
            """,
            {"episode": episode_id},
            limit=limit,
        )

    data = {
        "episode_id": episode_id,
        "concepts": concept_rows,
        "total": len(concept_rows),
    }

    if out.is_json:
        payload = {"success": True, "command": "episode concepts", "graph_id": graph_id, "data": data}
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    columns = ("Family ID", "Name", "Role", "Confidence")
    table_rows: list[list[str]] = []
    for c in concept_rows:
        table_rows.append((
            c.get("family_id", ""),
            c.get("name", ""),
            c.get("role", ""),
            f"{float(c.get('confidence', 0)) * 100:.1f}%" if c.get("confidence") is not None else "",
        ))
    out.table(f"Concepts in Episode {episode_id[:20]} ({len(concept_rows)})", columns, table_rows)


# ------------------------------------------------------------------
# episode get  (NEW)
# ------------------------------------------------------------------

@episode.command()
@click.argument("episode_id")
@_graph_option
@click.pass_context
def get(ctx: click.Context, episode_id: str, graph: Optional[str]) -> None:
    """Get episode details."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    graph_id = obj.get_active_graph(graph)

    with obj.get_storage(graph_id) as storage:
        rows = read_sql(
            storage,
            """
            SELECT version_id AS episode_version_id,
                   family_id,
                   name,
                   heading_path,
                   document_version_id,
                   document_family_id,
                   start_offset,
                   end_offset,
                   line_start,
                   line_end,
                   chunk_index,
                   event_time,
                   processed_time
            FROM v_episodes
            WHERE version_id = :eid
            LIMIT 1
            """,
            {"eid": episode_id},
            limit=1,
        )

    if not rows:
        out.error(f"Episode not found: {episode_id}", code=NOT_FOUND)

    ep = rows[0]

    if out.is_json:
        payload = {"success": True, "command": "episode get", "graph_id": graph_id, "data": ep}
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    # Rich panel
    lines = [
        f"Name:         {_rich_escape(ep.get('name', ''))}",
        f"Family ID:    {_rich_escape(ep.get('family_id', ''))}",
        f"Version ID:   {_rich_escape(ep.get('episode_version_id', ''))}",
        f"Document:     {_rich_escape(ep.get('document_version_id', ''))}",
        f"Heading:      {_rich_escape(ep.get('heading_path', ''))}",
        f"Lines:        {ep.get('line_start', '')} - {ep.get('line_end', '')}",
        f"Offset:       {ep.get('start_offset', '')} - {ep.get('end_offset', '')}",
        f"Chunk Index:  {ep.get('chunk_index', '')}",
        f"Event Time:   {format_timestamp(ep.get('event_time'))}",
        f"Processed:    {format_timestamp(ep.get('processed_time'))}",
    ]
    out.console.print(_Panel(
        "\n".join(lines),
        title=f"Episode: {_rich_escape(episode_id)}",
    ))


# ------------------------------------------------------------------
# episode content  (NEW)
# ------------------------------------------------------------------

@episode.command()
@click.argument("episode_id")
@_graph_option
@click.pass_context
def content(ctx: click.Context, episode_id: str, graph: Optional[str]) -> None:
    """Read episode source content."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    graph_id = obj.get_active_graph(graph)

    with obj.get_storage(graph_id) as storage:
        rows = read_sql(
            storage,
            """
            SELECT version_id AS episode_version_id,
                   name,
                   heading_path,
                   document_version_id,
                   line_start,
                   line_end,
                   source_text
            FROM v_episodes
            WHERE version_id = :eid
            LIMIT 1
            """,
            {"eid": episode_id},
            limit=1,
        )

    if not rows:
        out.error(f"Episode not found: {episode_id}", code=NOT_FOUND)

    ep = rows[0]
    source_text = ep.get("source_text", "")

    data = {
        "episode_version_id": ep.get("episode_version_id", ""),
        "name": ep.get("name", ""),
        "heading_path": ep.get("heading_path", ""),
        "document_version_id": ep.get("document_version_id", ""),
        "line_start": ep.get("line_start"),
        "line_end": ep.get("line_end"),
        "source_text": source_text,
        "length": len(source_text),
    }

    if out.is_json:
        payload = {"success": True, "command": "episode content", "graph_id": graph_id, "data": data}
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        click.echo(source_text)
        return

    # Rich panel with metadata header
    header_lines = [
        f"Name:     {_rich_escape(ep.get('name', ''))}",
        f"Heading:  {_rich_escape(ep.get('heading_path', ''))}",
        f"Document: {_rich_escape(ep.get('document_version_id', ''))}",
        f"Lines:    {ep.get('line_start', '')} - {ep.get('line_end', '')}",
        f"Length:   {len(source_text)} chars",
    ]
    header = "\n".join(header_lines)

    out.console.print(_Panel(
        f"{header}\n\n{_rich_escape(source_text)}",
        title=f"Episode Content: {_rich_escape(episode_id)}",
    ))
