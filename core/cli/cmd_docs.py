"""``docs`` command group — document discovery, search, and management.

Subcommands
-----------
roots     List searchable document root directories.
list      List indexed documents.
path      Resolve a document ID to a readable file path.
search    Literal text search over readable document files.
grep      Regex text search over readable document files.
map       Map a file-system path to Deep-Dream document records.
content   Read document content.
delete    Delete a document version (DANGEROUS).

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
from ._exit_codes import ARGS, ERROR, NOT_FOUND, OK
from ._helpers import (
    document_file_payload,
    document_rows,
    map_path_to_documents,
    read_sql,
    search_document_files,
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
def docs() -> None:
    """Document-first file discovery and search."""
    pass


# ------------------------------------------------------------------
# docs roots
# ------------------------------------------------------------------

@docs.command()
@_graph_option
@click.pass_context
def roots(ctx: click.Context, graph: Optional[str]) -> None:
    """List searchable document root directories."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    graph_id = obj.get_active_graph(graph)

    with obj.get_storage(graph_id) as storage:
        rows = document_rows(storage, limit=5000)
        roots_set: set[str] = set()
        for doc in rows:
            vault_root = doc.get("vault_root")
            if vault_root:
                roots_set.add(str(Path(vault_root).resolve()))
            if doc.get("source_mode") == "external" and doc.get("absolute_path"):
                roots_set.add(str(Path(doc["absolute_path"]).resolve().parent))
        roots_set.add(str(Path(storage.storage_path).resolve() / "content"))

        data = {
            "roots": sorted(roots_set),
            "document_count": len(rows),
        }

    if out.is_json:
        payload = {"success": True, "command": "docs roots", "graph_id": graph_id, "data": data}
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    columns = ("Root Path",)
    table_rows = [(r,) for r in sorted(roots_set)]
    out.table(f"Document Roots ({len(roots_set)} roots, {len(rows)} docs)", columns, table_rows)


# ------------------------------------------------------------------
# docs list
# ------------------------------------------------------------------

@docs.command("list")
@click.option("--limit", type=int, default=100, show_default=True, help="Maximum documents to list.")
@_graph_option
@click.pass_context
def list_docs(ctx: click.Context, limit: int, graph: Optional[str]) -> None:
    """List indexed documents."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    graph_id = obj.get_active_graph(graph)

    with obj.get_storage(graph_id) as storage:
        rows = document_rows(storage, limit=limit)
        docs_data = [document_file_payload(storage, d) for d in rows]

    data = {"documents": docs_data, "total": len(docs_data)}

    if out.is_json:
        payload = {"success": True, "command": "docs list", "graph_id": graph_id, "data": data}
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    columns = ("ID", "Title", "Source", "Verification", "Processed")
    table_rows: list[list[str]] = []
    for d in docs_data:
        table_rows.append((
            d.get("document_family_id", d.get("document_version_id", "")),
            d.get("title", ""),
            d.get("source_mode", ""),
            d.get("verification", ""),
            format_timestamp(d.get("processed_time")),
        ))
    out.table(f"Documents ({len(docs_data)})", columns, table_rows)


# ------------------------------------------------------------------
# docs path
# ------------------------------------------------------------------

@docs.command()
@click.argument("document_id")
@_graph_option
@click.pass_context
def path(ctx: click.Context, document_id: str, graph: Optional[str]) -> None:
    """Resolve a document ID to a readable file path."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    graph_id = obj.get_active_graph(graph)

    with obj.get_storage(graph_id) as storage:
        info = storage.get_document_file_info(document_id)
        if not info:
            out.error(f"Document not found: {document_id}", code=NOT_FOUND)
        payload = document_file_payload(storage, info)

    if out.is_json:
        result = {"success": True, "command": "docs path", "graph_id": graph_id, "data": payload}
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        resolved = payload.get("resolved_path", "")
        if resolved:
            click.echo(resolved)
        return

    # Rich panel
    lines = [
        f"Title:       {_rich_escape(payload.get('title', ''))}",
        f"Source Mode: {_rich_escape(payload.get('source_mode', ''))}",
        f"Path:        {_rich_escape(payload.get('resolved_path', '') or '(not found)')}",
        f"Verification:{_rich_escape(payload.get('verification', ''))}",
        f"Hash:        {_rich_escape(payload.get('content_hash', ''))}",
        f"Size:        {payload.get('char_count', 0)} chars / {payload.get('line_count', 0)} lines",
    ]
    out.console.print(_Panel(
        "\n".join(lines),
        title=f"Document: {_rich_escape(document_id)}",
    ))


# ------------------------------------------------------------------
# docs search
# ------------------------------------------------------------------

@docs.command()
@click.argument("pattern")
@click.option("--limit", type=int, default=50, show_default=True, help="Maximum hits.")
@_graph_option
@click.pass_context
def search(ctx: click.Context, pattern: str, limit: int, graph: Optional[str]) -> None:
    """Literal text search over readable document files."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    graph_id = obj.get_active_graph(graph)

    with obj.get_storage(graph_id) as storage:
        try:
            hits = search_document_files(storage, pattern, regex=False, limit=limit)
        except ValueError as exc:
            out.error(str(exc), code=ARGS)

    data = {
        "hits": hits,
        "total": len(hits),
        "used": {
            "raw_files": True,
            "sqlite": True,
            "semantic": False,
            "graph_traversal": False,
            "api": False,
        },
    }

    if out.is_json:
        payload = {"success": True, "command": "docs search", "graph_id": graph_id, "data": data}
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    columns = ("Title", "Line", "Text")
    table_rows: list[list[str]] = []
    for h in hits:
        doc = h.get("document") or {}
        text = h.get("text", "")
        if len(text) > 120:
            text = text[:117] + "..."
        table_rows.append((
            doc.get("title", ""),
            str(doc.get("line_start", "")),
            text,
        ))
    out.table(f"Search: {pattern!r} ({len(hits)} hits)", columns, table_rows)


# ------------------------------------------------------------------
# docs grep
# ------------------------------------------------------------------

@docs.command()
@click.argument("pattern")
@click.option("--limit", type=int, default=50, show_default=True, help="Maximum hits.")
@_graph_option
@click.pass_context
def grep(ctx: click.Context, pattern: str, limit: int, graph: Optional[str]) -> None:
    """Regex text search over readable document files."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    graph_id = obj.get_active_graph(graph)

    with obj.get_storage(graph_id) as storage:
        try:
            hits = search_document_files(storage, pattern, regex=True, limit=limit)
        except ValueError as exc:
            out.error(
                str(exc),
                hint="Use docs search for literal text matching.",
                code=ARGS,
            )

    data = {
        "hits": hits,
        "total": len(hits),
        "used": {
            "raw_files": True,
            "sqlite": True,
            "semantic": False,
            "graph_traversal": False,
            "api": False,
        },
    }

    if out.is_json:
        payload = {"success": True, "command": "docs grep", "graph_id": graph_id, "data": data}
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    columns = ("Title", "Line", "Match")
    table_rows: list[list[str]] = []
    for h in hits:
        doc = h.get("document") or {}
        text = h.get("text", "")
        if len(text) > 120:
            text = text[:117] + "..."
        table_rows.append((
            doc.get("title", ""),
            str(doc.get("line_start", "")),
            text,
        ))
    out.table(f"Grep: /{pattern}/ ({len(hits)} hits)", columns, table_rows)


# ------------------------------------------------------------------
# docs map
# ------------------------------------------------------------------

@docs.command()
@click.argument("path", type=click.Path(exists=False))
@_graph_option
@click.pass_context
def map_cmd(ctx: click.Context, path: str, graph: Optional[str]) -> None:
    """Map a file-system path to Deep-Dream document records."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    graph_id = obj.get_active_graph(graph)

    with obj.get_storage(graph_id) as storage:
        docs_data = [document_file_payload(storage, d) for d in map_path_to_documents(storage, path)]

    data = {"path": path, "documents": docs_data, "total": len(docs_data)}

    if out.is_json:
        payload = {"success": True, "command": "docs map", "graph_id": graph_id, "data": data}
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    columns = ("ID", "Title", "Source", "Verification")
    table_rows: list[list[str]] = []
    for d in docs_data:
        table_rows.append((
            d.get("document_family_id", d.get("document_version_id", "")),
            d.get("title", ""),
            d.get("source_mode", ""),
            d.get("verification", ""),
        ))
    out.table(f"Path Map: {path} ({len(docs_data)} docs)", columns, table_rows)


# ------------------------------------------------------------------
# docs content  (NEW)
# ------------------------------------------------------------------

@docs.command()
@click.argument("document_id")
@click.option(
    "--full",
    is_flag=True,
    default=False,
    help="Show the entire document content (no truncation).",
)
@click.option(
    "--lines",
    type=str,
    default=None,
    help="Line range to display, e.g. '10-30'.",
)
@_graph_option
@click.pass_context
def content(ctx: click.Context, document_id: str, full: bool, lines: Optional[str], graph: Optional[str]) -> None:
    """Read document content.

    By default shows the first 200 lines.  Use --full to show everything,
    or --lines 10-30 to show a specific range.
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    graph_id = obj.get_active_graph(graph)

    # Parse --lines range if given
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    if lines:
        try:
            parts = lines.split("-", 1)
            line_start = int(parts[0])
            line_end = int(parts[1]) if len(parts) > 1 else line_start
        except (ValueError, IndexError):
            out.error(
                f"Invalid --lines range: {lines!r}.  Expected format: '10-30' or '50'.",
                hint="Use --full to show the entire document.",
                code=ARGS,
            )

    with obj.get_storage(graph_id) as storage:
        try:
            result = storage.get_document_content(document_id)
        except (KeyError, LookupError):
            out.error(
                f"Document not found: {document_id}",
                hint="Use 'deep-dream docs list' to see valid document IDs.",
                code=NOT_FOUND,
            )

    raw_content = result.get("content", "")
    read_path = result.get("read_path", "")
    title = result.get("title", "")
    source_mode = result.get("source_mode", "")

    if not raw_content and not read_path:
        out.error(
            f"No content available for document: {document_id}",
            hint="The document may not have a managed file or snapshot on disk.",
            code=NOT_FOUND,
        )

    # Apply line slicing
    content_lines = raw_content.splitlines()
    total_lines = len(content_lines)

    if line_start is not None or line_end is not None:
        s = max(1, line_start or 1)
        e = min(total_lines, line_end or total_lines)
        content_lines = content_lines[s - 1:e]
    elif not full:
        show_max = 200
        if total_lines > show_max:
            content_lines = content_lines[:show_max]

    displayed = "\n".join(content_lines)
    truncated = (not full and total_lines > 200) or (line_start is not None or line_end is not None)

    data = {
        "document_version_id": document_id,
        "title": title,
        "source_mode": source_mode,
        "read_path": read_path,
        "total_lines": total_lines,
        "displayed_lines": len(content_lines),
        "truncated": truncated,
        "content": displayed,
    }

    if out.is_json:
        payload = {"success": True, "command": "docs content", "graph_id": graph_id, "data": data}
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        click.echo(displayed)
        return

    # Rich panel with metadata header
    header_lines = [
        f"Title:  {_rich_escape(title)}",
        f"Source: {_rich_escape(source_mode)}",
        f"Path:   {_rich_escape(read_path)}",
        f"Lines:  {len(content_lines)}/{total_lines}",
    ]
    if truncated:
        header_lines.append("[dim](Use --full or --lines to see more)[/dim]")
    header = "\n".join(header_lines)

    out.console.print(_Panel(
        f"{header}\n\n{_rich_escape(displayed)}",
        title=f"Document: {_rich_escape(document_id)}",
    ))


# ------------------------------------------------------------------
# docs delete  (NEW)
# ------------------------------------------------------------------

@docs.command()
@click.argument("document_id")
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Confirm deletion. This is DANGEROUS and irreversible.",
)
@_graph_option
@click.pass_context
def delete(ctx: click.Context, document_id: str, yes: bool, graph: Optional[str]) -> None:
    """Delete a document version and its associated graph data.

    \b
    DANGER: This permanently removes the document along with all linked
    episodes, concepts, relations, and embeddings.  Use --yes to confirm.

    \b
    Cascade behaviour:
    - All episodes belonging to the document are deleted.
    - Concept and relation families that become orphaned are removed.
    - Embeddings linked to deleted episodes are purged.
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    graph_id = obj.get_active_graph(graph)

    with obj.get_storage(graph_id) as storage:
        # Pre-flight: confirm the document exists
        info = storage.get_document_file_info(document_id)
        if not info:
            out.error(f"Document not found: {document_id}", code=NOT_FOUND)

        if not yes:
            if out.is_json:
                click.echo(json.dumps({
                    "success": False,
                    "error": "Deletion requires --yes confirmation.",
                    "document_id": document_id,
                    "title": info.get("title", ""),
                }, ensure_ascii=False, indent=2))
                raise SystemExit(1)

            out.console.print(
                f"[bold red]DANGER:[/bold red] About to delete document "
                f"[bold]{_rich_escape(info.get('title', document_id))}[/bold]"
            )
            out.console.print(
                "This will cascade-delete episodes, concepts, relations, and embeddings."
            )
            if not click.confirm("Proceed?", default=False):
                out.console.print("[dim]Cancelled.[/dim]")
                raise SystemExit(0)

        result = storage.delete_document_version(document_id)

    data = {
        "document_version_id": document_id,
        "title": info.get("title", ""),
        **result,
    }

    if out.is_json:
        payload = {"success": True, "command": "docs delete", "graph_id": graph_id, "data": data}
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    if result.get("deleted"):
        out.success(f"Deleted document: {info.get('title', document_id)}")
    else:
        out.console.print(
            f"[bold yellow]Delete returned:[/bold yellow] {_rich_escape(result.get('reason', 'unknown'))}"
        )
