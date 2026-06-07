"""``deep-dream graph`` — graph management commands.

Subcommands
-----------
list      List all graphs and their status.
create    Create a new graph.
use       Set the active graph.
stats     Show statistics for a graph.
rebuild   DANGEROUS — clear graph data for rebuild (requires --yes).

All commands respect the global ``--json`` / ``--quiet`` / ``--no-color``
flags handled by :class:`OutputManager`.
"""
from __future__ import annotations

import json as _json
from typing import Any, Dict, Optional

import click

from rich.markup import escape as _rich_escape
from rich.panel import Panel as _Panel

from ._ctx import CliContext
from ._exit_codes import ERROR, NOT_FOUND, OK
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
def graph() -> None:
    """Graph management — list, create, switch, and rebuild graphs."""
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
# graph list
# ------------------------------------------------------------------

@graph.command("list")
@click.pass_context
def list_graphs(ctx: click.Context) -> None:
    """List all graphs and their status."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)

    registry = obj.get_registry()
    graphs = registry.list_graphs()
    graphs_info = registry.list_graphs_info()

    data = {
        "graphs": graphs,
        "graphs_info": graphs_info,
    }

    if out.is_json:
        from ._output import json_result
        payload = json_result("graph list", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    if not graphs_info:
        out.console.print("[dim]No graphs found.[/dim]")
        return

    columns = ("Graph ID", "Documents", "Entities", "Relations", "Updated")
    rows = []
    for info in graphs_info:
        rows.append([
            info.get("graph_id", ""),
            str(info.get("document_count", 0)),
            str(info.get("entity_count", 0)),
            str(info.get("relation_count", 0)),
            format_timestamp(info.get("updated_at")),
        ])
    out.table("Graphs", columns, rows)


# ------------------------------------------------------------------
# graph create
# ------------------------------------------------------------------

@graph.command()
@click.argument("graph_id")
@click.pass_context
def create(ctx: click.Context, graph_id: str) -> None:
    """Create a new graph.

    GRAPH_ID is the identifier for the new graph.  The graph directory
    and metadata are created if they do not already exist.
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)

    with obj.get_storage(graph_id, ensure=True) as storage:
        stats = storage.get_stats()

    data = {
        "graph_id": graph_id,
        "created": True,
        "stats": stats,
    }

    if out.is_json:
        from ._output import json_result
        payload = json_result("graph create", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    out.success(f"Created graph {graph_id}")
    if stats:
        out.console.print(
            f"  [dim]Documents: {stats.get('document_count', 0)}, "
            f"Families: {stats.get('concept_family_count', 0)}, "
            f"Edges: {stats.get('concept_edge_count', 0)}[/dim]"
        )


# ------------------------------------------------------------------
# graph use
# ------------------------------------------------------------------

@graph.command()
@click.argument("graph_id")
@click.pass_context
def use(ctx: click.Context, graph_id: str) -> None:
    """Set the active graph.

    GRAPH_ID is the identifier of the graph to activate.
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)

    registry = obj.get_registry()
    registry.set_graph_metadata(graph_id)

    # Write the active graph to library.json so it persists.
    from core.server.registry import LIBRARY_ID
    from pathlib import Path
    import json

    registry_json_path = Path(obj.config.get("storage_path", "./library")) / "library.json"
    data: Dict[str, Any] = {"library": {"id": LIBRARY_ID, "graph_id": LIBRARY_ID}}
    if registry_json_path.exists():
        try:
            data = json.loads(registry_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"library": {"id": LIBRARY_ID, "graph_id": LIBRARY_ID}}
    data.setdefault("library", {"id": LIBRARY_ID, "graph_id": LIBRARY_ID})
    data["library"]["graph_id"] = LIBRARY_ID
    registry_json_path.parent.mkdir(parents=True, exist_ok=True)
    registry_json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    result = {
        "graph_id": graph_id,
        "active_graph_id": graph_id,
    }

    if out.is_json:
        from ._output import json_result
        payload = json_result("graph use", result)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    out.success(f"Active graph set to {graph_id}")


# ------------------------------------------------------------------
# graph stats
# ------------------------------------------------------------------

@graph.command()
@_graph_option
@click.pass_context
def stats(ctx: click.Context, graph: Optional[str]) -> None:
    """Show statistics for a graph."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)
    graph_id = _get_graph_id(ctx, graph)

    registry = obj.get_registry()
    info = registry.get_graph_info(graph_id)

    if info is None:
        out.error(
            f"Graph not found: {graph_id}",
            hint="Use 'deep-dream graph list' to see available graphs.",
            code=NOT_FOUND,
        )
        return  # unreachable

    data = {
        "graph_id": graph_id,
        **info,
    }

    if out.is_json:
        from ._output import json_result
        payload = json_result("graph stats", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    lines = [
        f"[bold]Graph ID:[/bold]          {_rich_escape(graph_id)}",
        f"[bold]Documents:[/bold]         {info.get('document_count', 0)}",
        f"[bold]Entities:[/bold]          {info.get('entity_count', 0)}",
        f"[bold]Relations:[/bold]         {info.get('relation_count', 0)}",
        f"[bold]Episodes:[/bold]          {info.get('episode_count', 0)}",
        f"[bold]Created:[/bold]           {format_timestamp(info.get('created_at'))}",
        f"[bold]Updated:[/bold]           {format_timestamp(info.get('updated_at'))}",
    ]
    out.console.print(_Panel(
        "\n".join(lines),
        title=f"Graph Stats: {_rich_escape(graph_id)}",
    ))


# ------------------------------------------------------------------
# graph rebuild  (DANGEROUS)
# ------------------------------------------------------------------

@graph.command()
@_graph_option
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Confirm the destructive rebuild operation.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be affected without clearing data.",
)
@click.pass_context
def rebuild(ctx: click.Context, graph: Optional[str], yes: bool, dry_run: bool) -> None:
    """DANGEROUS: Clear graph data for rebuild.

    Clears all concept families, versions, edges, and episode data
    from the graph so it can be rebuilt from scratch.  Documents are
    preserved.  Requires explicit --yes confirmation.

    Use --dry-run to preview what would be affected without making
    changes.
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config_path = _resolve_config_path(ctx)
    obj.load_config(config_path)
    graph_id = _get_graph_id(ctx, graph)

    # Gather stats before any action.
    registry = obj.get_registry()
    before = registry.get_graph_info(graph_id) or {}
    previous_stats = {
        "families": before.get("concept_family_count", 0),
        "versions": before.get("concept_version_count", 0),
        "edges": before.get("concept_edge_count", 0),
        "documents": before.get("document_count", 0),
    }

    if dry_run:
        data = {
            "graph_id": graph_id,
            "dry_run": True,
            "would_clear": True,
            "previous_stats": previous_stats,
            "message": "Dry run — no data was modified.",
        }
        if out.is_json:
            from ._output import json_result
            payload = json_result("graph rebuild", data)
            click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
            return
        if out.is_quiet:
            return
        lines = [
            "[bold yellow]DRY RUN — no data will be modified.[/bold yellow]",
            "",
            f"[bold]Graph:[/bold]    {_rich_escape(graph_id)}",
            f"[bold]Families:[/bold] {previous_stats['families']} (would be cleared)",
            f"[bold]Versions:[/bold] {previous_stats['versions']} (would be cleared)",
            f"[bold]Edges:[/bold]    {previous_stats['edges']} (would be cleared)",
            f"[bold]Documents:[/bold] {previous_stats['documents']} (preserved)",
            "",
            "Use --yes (without --dry-run) to proceed.",
        ]
        out.console.print(_Panel(
            "\n".join(lines),
            title="Graph Rebuild (dry-run)",
        ))
        return

    # Safety: require --yes to proceed.
    if not yes:
        out.error(
            "Rebuild requires explicit confirmation.  Use --yes to confirm.",
            hint=(
                "This operation clears all concept, relation, and episode "
                "data.  Documents are preserved.  This cannot be undone."
            ),
            code=ERROR,
        )
        return  # unreachable

    # Perform the clear.
    with obj.get_storage(graph_id) as storage:
        storage.clear_graph_data()

    data = {
        "graph_id": graph_id,
        "cleared": True,
        "previous_stats": previous_stats,
        "message": "Graph data cleared. Re-run remember or vault index to rebuild.",
    }

    if out.is_json:
        from ._output import json_result
        payload = json_result("graph rebuild", data)
        click.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    out.success(f"Graph data cleared for {graph_id}")
    out.console.print(
        f"  [dim]Previous — families: {previous_stats['families']}, "
        f"versions: {previous_stats['versions']}, "
        f"edges: {previous_stats['edges']}, "
        f"documents: {previous_stats['documents']}[/dim]"
    )
    out.console.print(
        "  [dim]Re-run 'deep-dream remember' or 'deep-dream vault index' to rebuild.[/dim]"
    )
