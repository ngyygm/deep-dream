"""``deep-dream scope`` -- 图限定文档沙箱。

用"检索 + 图回溯"圈出与查询相关的有界文档范围，可选物化成沙箱目录
（symlink + manifest），供 agent 在范围内用原生检索（rg/grep）精读。
实现见 :func:`core.find.scope.build_document_scope`。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click


@click.command()
@click.argument("query")
@click.option(
    "--graph",
    default=None,
    help="Graph ID [default: library]",
)
@click.option(
    "--mode",
    type=click.Choice(["bm25", "semantic", "hybrid"]),
    default="hybrid",
    show_default=True,
    help="Seed concept retrieval mode.",
)
@click.option(
    "--max-concepts",
    default=20,
    type=int,
    show_default=True,
    help="Maximum seed concepts.",
)
@click.option(
    "--max-docs",
    default=30,
    type=int,
    show_default=True,
    help="Maximum documents in the scope.",
)
@click.option(
    "--materialize/--no-materialize",
    default=False,
    show_default=True,
    help="Materialize the scope into a sandbox directory (symlinks + manifest).",
)
@click.option(
    "--sandbox-root",
    default=None,
    type=click.Path(file_okay=False),
    help="Sandbox root directory [default: <storage>/sandboxes]",
)
@click.pass_context
def scope(
    ctx: click.Context,
    query: str,
    graph: Optional[str],
    mode: str,
    max_concepts: int,
    max_docs: int,
    materialize: bool,
    sandbox_root: Optional[str],
) -> None:
    """Bound a document scope for a query via retrieval + graph backtrace.

    \b
    Examples:
      deep-dream scope "分布式存储" --max-docs 10
      deep-dream scope "搬家 时间线" --materialize
    """
    from core.find.scope import build_document_scope, materialize_scope

    from ._output import OutputManager

    out = OutputManager(ctx)
    cli_ctx = ctx.obj
    graph_id = cli_ctx.get_active_graph(graph)

    result: dict = {}
    with cli_ctx.get_storage(graph_id) as storage:
        try:
            result = build_document_scope(
                storage, query, mode=mode,
                max_concepts=max_concepts, max_docs=max_docs,
            )
        except Exception as exc:  # noqa: BLE001 -- CLI 边界统一报错
            out.error(f"scope failed: {exc}")
            raise SystemExit(1)

        if materialize:
            root = sandbox_root or str(Path(
                cli_ctx.get_registry().graph_dir(graph_id)) / "sandboxes")
            result["sandbox"] = materialize_scope(result, root)

    docs = result.get("documents", [])
    stats = result.get("stats", {})
    sandbox = result.get("sandbox") or {}

    if out.is_json:
        from ._helpers import emit_json_result

        emit_json_result("scope", result, graph_id=graph_id)
        return

    out.panel(
        f"Scope: {query}",
        f"mode={mode} · 概念 {stats.get('seed_concepts', 0)} · "
        f"文档 {len(docs)}/{stats.get('documents_total', 0)}",
    )

    if not docs:
        out.success("No documents in scope.")
        return

    rows = []
    for i, doc in enumerate(docs, 1):
        rows.append([
            str(i),
            (doc.get("title") or "")[:40],
            f"{doc.get('score', 0):.3f}",
            str(len(doc.get("matched_concepts", []))),
            str(len(doc.get("episodes", []))),
            (doc.get("path") or "")[:60],
        ])
    out.table(
        "Documents in scope",
        ["#", "Title", "Score", "Concepts", "Episodes", "Path"],
        rows,
    )

    if sandbox:
        out.success(f"Sandbox: {sandbox.get('path')} ({len(sandbox.get('files', []))} files)")
        out.echo(f"Manifest: {sandbox.get('manifest_path')}")
