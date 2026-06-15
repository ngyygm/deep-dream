"""``vault`` command group -- Obsidian/Markdown vault indexing and inspection.

Subcommands
-----------
index     Index a Markdown/Obsidian vault directory or single file.
tree      Show indexed vault files as a directory tree.

All commands respect the global ``--json`` / ``--quiet`` / ``--no-color``
flags handled by :class:`OutputManager`.
"""
from __future__ import annotations

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
from ._exit_codes import ARGS, NOT_FOUND, OK
from ._helpers import read_sql
from ._output import OutputManager


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
@click.pass_context
def index(
    ctx: click.Context,
    path: str,
    force: bool,
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
    graph_id = obj.get_active_graph()

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
    # Older indexers fold "unchanged" into "indexed"; tolerate a missing key.
    files_unchanged = result.get("unchanged", 0)
    errors = result.get("errors", 0)
    error_details = result.get("error_details", []) or []

    data = {
        "path": resolved,
        "files_scanned": files_scanned,
        "files_indexed": files_indexed,
        "files_unchanged": files_unchanged,
        "errors": errors,
        "error_details": error_details,
    }

    # ----------------------------------------------------------
    # Output
    # ----------------------------------------------------------
    if out.is_json:
        out.result(data, meta={"graph_id": graph_id})
        return

    if out.is_quiet:
        return

    # Rich summary
    summary_lines = [
        f"Files scanned:   {files_scanned}",
        f"Files indexed:   {files_indexed}",
        f"Unchanged:       {files_unchanged}",
    ]
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
        for entry in error_details:
            file_name = _rich_escape(str(entry.get("file", "")))
            reason = _rich_escape(str(entry.get("reason", "")))
            out.console.print(
                f"  [dim]{file_name}[/dim] — [red]{reason}[/red]"
            )

    out.success(
        f"Indexed {files_indexed}, unchanged {files_unchanged}, "
        f"of {files_scanned} files from {path}"
    )


# ------------------------------------------------------------------
# vault tree
# ------------------------------------------------------------------

@vault.command()
@click.pass_context
def tree(ctx: click.Context) -> None:
    """Show indexed vault files as a directory tree.

    Queries all indexed documents grouped by ``vault_root`` and renders
    a Rich Tree showing the directory structure.

    \b
    Examples:
      deep-dream vault tree
      deep-dream --json vault tree
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    graph_id = obj.get_active_graph()

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
            out.result(
                {"vaults": [], "document_count": 0},
                meta={"graph_id": graph_id},
            )
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
        out.result(data, meta={"graph_id": graph_id})
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
        vault_label = _vault_root_label(vault_root)
        vault_branch = root_tree.add(
            f"[bold cyan]{_rich_escape(vault_label)}[/bold cyan] "
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
            char_count = d.get("char_count")
            size_tag = f" [dim]({char_count:,} chars)[/dim]" if char_count else ""
            node[filename] = d  # leaf value is the document row

        # Render the nested dict into the Rich Tree
        _render_nested_tree(vault_branch, path_tree)

    out.console.print(root_tree)


# ------------------------------------------------------------------
# Tree rendering helper
# ------------------------------------------------------------------

def _vault_root_label(vault_root: str) -> str:
    """Extract a display label from a vault_root path string.

    Uses separator-aware stripping so a Windows-style path string stored on a
    POSIX host (``C:\\Users\\me\\vault``) yields ``vault`` rather than the whole
    raw string that ``pathlib.Path(...).name`` would return.
    """
    if not vault_root:
        return vault_root
    # Strip trailing separators (both POSIX and Windows) then split on the last
    # separator of either kind.
    stripped = vault_root.rstrip("/\\")
    if not stripped:
        return vault_root
    for sep in ("\\", "/"):
        idx = stripped.rfind(sep)
        if idx != -1:
            return stripped[idx + 1:]
    return stripped


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
        dir_branch = parent.add(f"[bold]{_rich_escape(name)}/[/bold]")
        _render_nested_tree(dir_branch, subtree)

    # Files next, sorted alphabetically
    for name, doc_row in sorted(files, key=lambda x: x[0].lower()):
        title = ""
        if isinstance(doc_row, dict):
            title = doc_row.get("title", "") or ""
        label = _rich_escape(name)
        if title and title != Path(name).stem:
            label = f"{_rich_escape(name)} [dim]- {_rich_escape(title)}[/dim]"
        parent.add(label)
