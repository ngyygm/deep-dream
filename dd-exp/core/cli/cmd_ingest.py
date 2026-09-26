"""``deep-dream ingest`` -- 统一文件入库入口。

``--profile prose``（默认）走完整 remember 管线（LLM 抽取/对齐）；
``--profile log`` 走零 LLM 快速通道：时间窗/行数窗切块、只进 FTS、
写入时模式蒸馏。实现见 :func:`core.ingest.ingest_file`。
"""
from __future__ import annotations

from typing import Optional

import click


@click.command()
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--profile",
    type=click.Choice(["prose", "log"]),
    default="prose",
    show_default=True,
    help="Ingestion profile: prose = full LLM pipeline; log = zero-LLM fast path.",
)
@click.option(
    "--graph",
    default=None,
    help="Graph ID [default: library]",
)
@click.option(
    "--name",
    default=None,
    help="Document title [default: file name]",
)
@click.option(
    "--time-window",
    default=300.0,
    type=float,
    show_default=True,
    help="[log] Time window in seconds.",
)
@click.option(
    "--line-window",
    default=400,
    type=int,
    show_default=True,
    help="[log] Line window when no timestamps.",
)
@click.option(
    "--no-distill",
    is_flag=True,
    default=False,
    help="[log] Skip pattern distillation document.",
)
@click.pass_context
def ingest(
    ctx: click.Context,
    path: str,
    profile: str,
    graph: Optional[str],
    name: Optional[str],
    time_window: float,
    line_window: int,
    no_distill: bool,
) -> None:
    """Ingest a file into the library (prose pipeline or log fast path).

    \b
    Examples:
      deep-dream ingest notes.md
      deep-dream ingest server.log --profile log
      deep-dream ingest trace.jsonl --profile log --line-window 200
    """
    from core.ingest import ingest_file, ingest_text

    from ._output import OutputManager

    out = OutputManager(ctx)
    cli_ctx = ctx.obj
    graph_id = cli_ctx.get_active_graph(graph)

    kwargs = {
        "graph_id": graph_id,
        "time_window_s": time_window,
        "line_window": line_window,
        "distill": not no_distill,
    }

    def _run(storage=None, processor=None):
        if name:
            from pathlib import Path as _P

            fp = _P(path)
            return ingest_text(
                fp.read_text(encoding="utf-8"), name, profile=profile,
                absolute_path=str(fp.resolve()),
                storage=storage, processor=processor, **kwargs,
            )
        target = {"storage": storage} if storage is not None else {"processor": processor}
        return ingest_file(path, profile, **target, **kwargs)

    if profile == "log":
        # log 快速通道零 LLM，只需要 storage。
        with cli_ctx.get_storage(graph_id) as storage:
            report = _run(storage=storage)
    else:
        # prose 走完整管线，需要 processor。
        processor = cli_ctx.get_registry().get_processor(graph_id)
        report = _run(processor=processor)

    if out.is_json:
        from ._helpers import emit_json_result

        emit_json_result("ingest", report, graph_id=graph_id)
        return

    out.panel(
        f"Ingested: {path}",
        f"profile={profile} · {report.get('duration_ms', 0):.0f}ms",
    )
    rows = [
        [key, str(report[key])]
        for key in (
            "profile", "file", "lines", "windows", "skipped_duplicate_windows",
            "documents_created", "episodes_created", "patterns_distilled",
            "duration_ms",
        )
        if key in report
    ]
    out.table("Report", ["Key", "Value"], rows)
