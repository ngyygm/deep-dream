"""``vault`` command group -- Obsidian/Markdown vault indexing and inspection.

Subcommands
-----------
index     Index a Markdown/Obsidian vault directory or single file.
tree      Show indexed vault files as a directory tree.

All commands respect the global ``--json`` / ``--quiet`` / ``--no-color``
flags handled by :class:`OutputManager`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

import click

try:
    from rich.tree import Tree

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False

from rich.markup import escape as _rich_escape
from rich.panel import Panel as _Panel

from ._ctx import CliContext
from ._exit_codes import NOT_FOUND
from ._helpers import read_sql
from ._output import OutputManager


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
def vault() -> None:
    """Obsidian/Markdown vault indexing and inspection."""
    pass


# ------------------------------------------------------------------
# vault index
# ------------------------------------------------------------------

@vault.command()
@click.argument(
    "path",
    type=click.Path(exists=True),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-index files even if content hash is unchanged.",
)
@_graph_option
@click.pass_context
def index(
    ctx: click.Context,
    path: str,
    force: bool,
    graph: Optional[str],
) -> None:
    """Index a Markdown/Obsidian vault directory or single file.

    Scans the directory tree for ``.md``, ``.markdown``, ``.txt``, and
    ``.text`` files, creates document records, and splits each file into
    episodes.

    \b
    Examples:
      deep-dream vault index ~/my-vault
      deep-dream vault index ~/my-vault --force
      deep-dream vault index ./notes/project-plan.md
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    graph_id = obj.get_active_graph(graph)

    resolved = str(Path(path).resolve())

    # ----------------------------------------------------------
    # Run indexing
    # ----------------------------------------------------------
    with obj.get_storage(graph_id, ensure=True) as storage:
        if out.is_json:
            result = storage.index_vault(resolved, force=force)
        else:
            with out.spinner(f"Indexing {path}..."):
                result = storage.index_vault(resolved, force=force)

    # Surface errors from the indexer itself
    if "error" in result:
        out.error(
            f"Vault indexing failed: {result['error']}",
            hint="Check that the path exists and contains supported files.",
            code=NOT_FOUND,
        )

    files_scanned = result.get("files", 0)
    files_indexed = result.get("indexed", 0)
    errors = result.get("errors", 0)
    episodes_created = result.get("episodes", 0)

    data = {
        "path": resolved,
        "files_scanned": files_scanned,
        "files_indexed": files_indexed,
        "episodes_created": episodes_created,
        "errors": errors,
    }

    # ----------------------------------------------------------
    # Output
    # ----------------------------------------------------------
    if out.is_json:
        payload = {
            "success": True,
            "command": "vault index",
            "graph_id": graph_id,
            "data": data,
        }
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    # Rich summary
    summary_lines = [
        f"Files scanned:   {files_scanned}",
        f"Files indexed:   {files_indexed}",
    ]
    if episodes_created:
        summary_lines.append(f"Episodes created: {episodes_created}")
    if errors:
        summary_lines.append(f"Errors:           {errors}")

    out.console.print(_Panel(
        "\n".join(summary_lines),
        title=f"Vault Index: {_rich_escape(path)}",
    ))

    if errors:
        out.console.print(
            f"[bold yellow]![/bold yellow] {errors} file(s) could not be indexed."
        )

    out.success(f"Indexed {files_indexed}/{files_scanned} files from {path}")


# ------------------------------------------------------------------
# vault tree
# ------------------------------------------------------------------

@vault.command()
@_graph_option
@click.pass_context
def tree(ctx: click.Context, graph: Optional[str]) -> None:
    """Show indexed vault files as a directory tree.

    Queries all indexed documents grouped by ``vault_root`` and renders
    a Rich Tree showing the directory structure.

    \b
    Examples:
      deep-dream vault tree
      deep-dream vault tree --json
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    graph_id = obj.get_active_graph(graph)

    with obj.get_storage(graph_id) as storage:
        rows = read_sql(
            storage,
            """
            SELECT document_version_id,
                   title,
                   vault_root,
                   relative_path,
                   absolute_path,
                   source_mode,
                   char_count,
                   processed_time
            FROM v_document_files
            WHERE COALESCE(vault_root, '') != ''
            ORDER BY vault_root, relative_path
            """,
            limit=10000,
        )

    if not rows:
        if out.is_json:
            payload = {
                "success": True,
                "command": "vault tree",
                "graph_id": graph_id,
                "data": {"vaults": [], "document_count": 0},
            }
            click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        if out.is_quiet:
            return

        out.console.print("[dim]No indexed vaults found.[/dim]")
        out.console.print(
            "[dim]Use [bold]deep-dream vault index <path>[/bold] to index a vault.[/dim]"
        )
        return

    # ----------------------------------------------------------
    # Group documents by vault_root
    # ----------------------------------------------------------
    vaults: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        vr = row.get("vault_root", "")
        if vr:
            vaults[vr].append(row)

    data = {
        "vaults": [
            {
                "root": root,
                "document_count": len(docs),
                "documents": [
                    {
                        "id": d.get("document_version_id", ""),
                        "title": d.get("title", ""),
                        "relative_path": d.get("relative_path", ""),
                        "char_count": d.get("char_count"),
                    }
                    for d in docs
                ],
            }
            for root, docs in sorted(vaults.items())
        ],
        "document_count": len(rows),
    }

    if out.is_json:
        payload = {
            "success": True,
            "command": "vault tree",
            "graph_id": graph_id,
            "data": data,
        }
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if out.is_quiet:
        return

    # ----------------------------------------------------------
    # Build Rich Tree
    # ----------------------------------------------------------
    if not _HAS_RICH:
        # Fallback: plain-text indented tree
        for root, docs in sorted(vaults.items()):
            click.echo(f"{root}/", err=True)
            for d in docs:
                rel = d.get("relative_path", "") or d.get("title", "")
                click.echo(f"  {rel}", err=True)
        return

    total_docs = len(rows)
    root_tree = Tree(
        f"[bold]Indexed Vaults[/bold] ({total_docs} documents, "
        f"{len(vaults)} vault{'s' if len(vaults) != 1 else ''})"
    )

    for vault_root, docs in sorted(vaults.items()):
        vault_label = Path(vault_root).name or vault_root
        vault_branch = root_tree.add(
            f"[bold cyan]{vault_label}[/bold cyan] "
            f"[dim]({len(docs)} docs)[/dim]"
        )

        # Build a nested path structure for this vault
        path_tree: Any = {}  # nested dict: segment -> subtree
        for d in docs:
            rel = d.get("relative_path", "")
            if not rel:
                # File at vault root — use title or id
                label = d.get("title", d.get("document_version_id", "")[:12])
                vault_branch.add(label)
                continue

            # Walk the relative path segments
            parts = PurePosixPath(rel).parts
            node = path_tree
            for i, segment in enumerate(parts[:-1]):
                if segment not in node:
                    node[segment] = {}
                node = node[segment]

            # Leaf: the filename with metadata
            filename = parts[-1] if parts else rel
            node[filename] = d  # leaf value is the document row

        # Render the nested dict into the Rich Tree
        _render_nested_tree(vault_branch, path_tree)

    out.console.print(root_tree)


# ------------------------------------------------------------------
# Tree rendering helper
# ------------------------------------------------------------------

def _render_nested_tree(parent: Any, node: Any) -> None:
    """Recursively render a nested dict into a Rich Tree.

    Intermediate dicts are rendered as directory nodes (directories first),
    leaf document rows are rendered as file nodes.
    """
    # Separate directories (dict values) from files (non-dict leaf values)
    dirs: List[tuple] = []
    files: List[tuple] = []

    for key, value in node.items():
        if isinstance(value, dict) and not isinstance(value.get("document_version_id", None), str):
            # It's a subtree (directory)
            dirs.append((key, value))
        else:
            files.append((key, value))

    # Directories first, sorted alphabetically
    for name, subtree in sorted(dirs, key=lambda x: x[0].lower()):
        dir_branch = parent.add(f"[bold]{name}/[/bold]")
        _render_nested_tree(dir_branch, subtree)

    # Files next, sorted alphabetically
    for name, doc_row in sorted(files, key=lambda x: x[0].lower()):
        title = ""
        if isinstance(doc_row, dict):
            title = doc_row.get("title", "")
        label = name
        if title and title != Path(name).stem:
            label = f"{name} [dim]- {title}[/dim]"
        parent.add(label)
