"""``library`` command group — library-level operations.

Subcommands
-----------
migrate    Migrate legacy multi-graph data into the single-library layout.

All commands respect the global ``--json`` / ``--quiet`` / ``--no-color``
flags handled by :class:`OutputManager`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from ._ctx import CliContext
from ._exit_codes import ARGS, CONFLICT, ERROR, NOT_FOUND, OK
from ._output import OutputManager


# ------------------------------------------------------------------
# Lazy import helper — avoids pulling the heavy import chain at
# module-load time so that ``--version`` / ``--help`` stay fast.
# ------------------------------------------------------------------

def _migrate_legacy_graphs(**kwargs: Any) -> dict:
    """Lazy wrapper around :func:`core.library.migrate_legacy_graphs`."""
    from core.library import migrate_legacy_graphs
    return migrate_legacy_graphs(**kwargs)


def _library_id() -> str:
    """Lazy accessor for the canonical LIBRARY_ID constant."""
    from core.server.registry import LIBRARY_ID
    return LIBRARY_ID


# ------------------------------------------------------------------
# Click group
# ------------------------------------------------------------------

@click.group()
def library() -> None:
    """Library-level operations (migrate, inspect, maintain)."""
    pass


# ------------------------------------------------------------------
# library migrate
# ------------------------------------------------------------------

@library.command()
@click.option(
    "--legacy-root",
    type=click.Path(exists=False),
    default=None,
    help="Root directory that contains the legacy ``graphs/`` folder.  "
         "Defaults to the directory of the service config.",
)
@click.option(
    "--target-root",
    type=click.Path(exists=False),
    default=None,
    help="Target directory for the single-library layout.  "
         "Defaults to ``storage_path`` from the active config (usually ``./library``).",
)
@click.option(
    "--source",
    multiple=True,
    help="Specific graph IDs to migrate.  May be given more than once.  "
         "When omitted, all graphs under ``legacy-root/graphs/`` are migrated.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Replace the target ``graph.db`` if it already exists.",
)
@click.option(
    "--no-backup",
    is_flag=True,
    default=False,
    help="Skip creating a timestamped backup of the legacy ``graphs/`` directory.",
)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Skip the interactive confirmation prompt.",
)
@click.pass_context
def migrate(
    ctx: click.Context,
    legacy_root: Optional[str],
    target_root: Optional[str],
    source: Tuple[str, ...],
    force: bool,
    no_backup: bool,
    yes: bool,
) -> None:
    """Migrate legacy multi-graph data into the single-library layout.

    \b
    What it does:
      1. Scans ``<legacy-root>/graphs/`` for per-graph ``graph.db`` files.
      2. Merges their contents into a single ``graph.db`` under
         ``<target-root>``.
      3. Copies assets (documents, blobs, snapshots, etc.) into the
         target directory.
      4. Rewrites all ``graph_id`` columns to the canonical LIBRARY_ID.
      5. Optionally backs up the original ``graphs/`` directory.

    \b
    Safety:
      - Unless ``--yes`` is given, an interactive confirmation prompt is
        shown listing the graphs that will be migrated.
      - Unless ``--no-backup`` is given, the legacy directory is moved to
        ``legacy_graphs_backup_<timestamp>`` after a successful merge.
      - ``--force`` is required when the target ``graph.db`` already exists.
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)

    # Resolve paths --------------------------------------------------
    config = obj.config

    legacy_path = Path(legacy_root).resolve() if legacy_root else None
    if legacy_path is None:
        # Default: directory that holds the service config
        config_path_str = obj._config_path or "service_config.json"
        legacy_path = Path(config_path_str).resolve().parent

    target_path = Path(target_root).resolve() if target_root else None
    if target_path is None:
        target_path = Path(
            config.get("storage_path") or "./library"
        ).resolve()

    source_ids: Optional[List[str]] = list(source) if source else None

    # Pre-flight validation ------------------------------------------
    graphs_dir = legacy_path / "graphs"
    if not graphs_dir.is_dir():
        out.error(
            f"Legacy graphs directory not found: {graphs_dir}",
            hint="Pass --legacy-root pointing at a directory that contains a graphs/ subdirectory.",
            code=NOT_FOUND,
        )

    graph_dirs = sorted(
        p for p in graphs_dir.iterdir()
        if p.is_dir() and (p / "graph.db").exists()
    )
    if source_ids:
        wanted = set(source_ids)
        graph_dirs = [p for p in graph_dirs if p.name in wanted]
        missing = wanted - {p.name for p in graph_dirs}
        if missing:
            out.error(
                f"Requested graph IDs not found in {graphs_dir}: {', '.join(sorted(missing))}",
                hint="Check the graph IDs under the legacy-root/graphs/ directory.",
                code=NOT_FOUND,
            )
    if not graph_dirs:
        out.error(
            f"No migratable graph.db files found under {graphs_dir}",
            hint="Each subdirectory must contain a graph.db file.",
            code=NOT_FOUND,
        )

    lib_id = _library_id()

    # Interactive confirmation ----------------------------------------
    if not yes:
        if out.is_json:
            # In JSON mode, still require --yes — no interactive prompt.
            click.echo(json.dumps({
                "success": False,
                "error": "Migration requires --yes confirmation in JSON mode.",
                "graphs": [p.name for p in graph_dirs],
                "target_root": str(target_path),
            }, ensure_ascii=False, indent=2))
            raise SystemExit(ARGS)

        # Rich / plain display
        graph_names = [p.name for p in graph_dirs]
        out.console.print(
            f"[bold]Library Migration[/bold]  (library_id={lib_id})"
        )
        out.console.print(f"  Legacy root : {legacy_path}")
        out.console.print(f"  Target root : {target_path}")
        out.console.print(f"  Graphs      : {len(graph_names)}")
        for name in graph_names:
            out.console.print(f"    - {name}")
        if no_backup:
            out.console.print("  [yellow]Backup: disabled (--no-backup)[/yellow]")
        else:
            out.console.print("  Backup      : enabled (legacy dir will be moved)")
        if force:
            out.console.print("  [yellow]Force : enabled (existing target DB will be replaced)[/yellow]")

        out.console.print("")
        if not click.confirm("Proceed with migration?", default=False):
            out.console.print("[dim]Cancelled.[/dim]")
            raise SystemExit(0)

    # Run migration ---------------------------------------------------
    try:
        with out.spinner("Migrating legacy graphs..."):
            result = _migrate_legacy_graphs(
                legacy_root=legacy_path,
                target_root=target_path,
                source_ids=source_ids,
                backup=not no_backup,
                force=force,
            )
    except FileNotFoundError as exc:
        out.error(str(exc), code=NOT_FOUND)
    except FileExistsError as exc:
        out.error(
            str(exc),
            hint="Use --force to overwrite the existing target database.",
            code=CONFLICT,
        )
    except Exception as exc:
        out.error(f"Migration failed: {exc}", code=ERROR)

    # Output ----------------------------------------------------------
    data = {
        "library_id": lib_id,
        **result,
    }

    if out.is_json:
        payload = {
            "success": True,
            "command": "library migrate",
            "data": data,
        }
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    # Rich display
    migrated = result.get("migrated_graphs", [])
    out.success(
        f"Migrated {len(migrated)} graph(s) into {result.get('target_root', str(target_path))}"
    )
    for name in migrated:
        out.console.print(f"  - {name}")
    backup_path = result.get("backup_path", "")
    if backup_path:
        out.console.print(f"  Backup: {backup_path}")
