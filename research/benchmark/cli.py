"""Isolated dual-track evaluation harness — run via ``python -m research.benchmark.cli``."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import click
from click.core import ParameterSource


# benchmark data lives under research/ (this file is research/benchmark/cli.py)
RESEARCH_ROOT = Path(__file__).resolve().parents[1]

DATASET_CHOICE = click.Choice(["longmemeval-s", "locomo"], case_sensitive=False)
TRACK_CHOICE = click.Choice(["baseline", "skill-agent", "both"], case_sensitive=False)


def _emit(ctx: click.Context, payload: dict[str, Any], message: str) -> None:
    root = ctx.find_root()
    params = getattr(root.obj, "_click_params", {}) or {}
    if params.get("json_output"):
        click.echo(json.dumps({"success": True, "data": payload}, ensure_ascii=False, indent=2))
    else:
        click.echo(message)
        for key, value in payload.items():
            click.echo(f"  {key}: {value}")


def _root_config(ctx: click.Context) -> Path:
    params = getattr(ctx.find_root().obj, "_click_params", {}) or {}
    return Path(params.get("config") or "service_config.json")


def _default_run_dir(dataset: str) -> Path:
    return RESEARCH_ROOT / ".benchmark_runs" / f"{dataset}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _context_k(ctx: click.Context, context_k: int, top_k: int | None) -> int:
    """Resolve the deprecated --top-k alias without accepting ambiguous input."""
    context_explicit = ctx.get_parameter_source("context_k") == ParameterSource.COMMANDLINE
    top_explicit = ctx.get_parameter_source("top_k") == ParameterSource.COMMANDLINE
    if context_explicit and top_explicit:
        raise click.UsageError("Use --context-k or the deprecated --top-k alias, not both.")
    return int(top_k if top_explicit and top_k is not None else context_k)


@click.group()
def benchmark() -> None:
    """Evaluate Deep-Dream retrieval and autonomous memory use."""


@benchmark.group("runtime")
def runtime() -> None:
    """Manage isolated external Agent runtimes."""


@runtime.command("install")
@click.option("--runtime", "runtime_name", type=click.Choice(["kimi"]), default="kimi", show_default=True)
@click.option("--version", default="1.49.0", show_default=True)
@click.option("--runtime-root", type=click.Path(path_type=Path), default=None)
@click.pass_context
def runtime_install(ctx: click.Context, runtime_name: str, version: str,
                    runtime_root: Path | None) -> None:
    """Install an exact external runtime into research/.benchmark_runtime."""
    from research.benchmark.kimi_runtime import install_kimi_runtime
    if runtime_name != "kimi":  # defensive; Click currently exposes only Kimi
        raise click.UsageError(f"Unsupported runtime: {runtime_name}")
    result = install_kimi_runtime(version=version, root=runtime_root)
    _emit(ctx, result, "Benchmark Agent runtime installed")


@runtime.command("check")
@click.option("--runtime", "runtime_name", type=click.Choice(["kimi"]), default="kimi", show_default=True)
@click.option("--version", default="1.49.0", show_default=True)
@click.option("--runtime-root", type=click.Path(path_type=Path), default=None)
@click.option("--executable", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.pass_context
def runtime_check(ctx: click.Context, runtime_name: str, version: str,
                  runtime_root: Path | None, executable: Path | None) -> None:
    """Verify the pinned runtime version and executable fingerprint."""
    from research.benchmark.kimi_runtime import check_kimi_runtime
    if runtime_name != "kimi":
        raise click.UsageError(f"Unsupported runtime: {runtime_name}")
    result = check_kimi_runtime(version=version, root=runtime_root, executable=executable)
    _emit(ctx, result, "Benchmark Agent runtime is ready")


def _thinking_value(value: str) -> bool:
    return value.lower() == "on"


@benchmark.command("agent-query")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--runtime", "runtime_name", type=click.Choice(["kimi"]), default="kimi", show_default=True)
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--mode", "modes", type=click.Choice(["direct", "evidence"]), multiple=True,
              default=("direct", "evidence"), show_default=True)
@click.option("--agent-model", default="qwen3.7-plus", show_default=True)
@click.option("--agent-thinking", type=click.Choice(["on", "off"]), default="on", show_default=True)
@click.option("--max-agent-steps", type=click.IntRange(min=1, max=128), default=128, show_default=True)
@click.option("--qa-workers", type=click.IntRange(min=1, max=16), default=2, show_default=True)
@click.option("--runtime-lifecycle", type=click.Choice(["per-question", "persistent"]),
              default="per-question", show_default=True)
@click.option("--agent-step-policy", type=click.Choice(["legacy", "autonomous"]),
              default="autonomous", show_default=True)
@click.option("--result-tag", default="kimi-qwen37-thinking-on", show_default=True)
@click.option("--question-id", "question_ids", multiple=True)
@click.option("--limit", type=click.IntRange(min=1), default=None)
@click.option("--eligible-evidence-only", is_flag=True)
@click.option("--context-k", type=click.IntRange(min=1), default=5, show_default=True)
@click.option("--neighbor-turns", type=click.IntRange(min=0, max=10), default=1, show_default=True)
@click.option("--timeout-seconds", type=click.IntRange(min=30), default=600, show_default=True)
@click.option("--runtime-version", default="1.49.0", show_default=True)
@click.option("--runtime-root", type=click.Path(path_type=Path), default=None)
@click.option("--runtime-executable", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--resume", is_flag=True)
@click.pass_context
def agent_query(
    ctx: click.Context, run_dir: Path, runtime_name: str, config_path: Path | None,
    modes: tuple[str, ...], agent_model: str, agent_thinking: str, max_agent_steps: int,
    qa_workers: int, runtime_lifecycle: str, agent_step_policy: str, result_tag: str,
    question_ids: tuple[str, ...], limit: int | None,
    eligible_evidence_only: bool, context_k: int, neighbor_turns: int,
    timeout_seconds: int, runtime_version: str, runtime_root: Path | None,
    runtime_executable: Path | None, resume: bool,
) -> None:
    """Run Kimi as an autonomous, conversation-scoped Deep-Dream query Agent."""
    from research.benchmark.kimi_benchmark import agent_query_benchmark
    result = agent_query_benchmark(
        run_dir, config_path or _root_config(ctx), runtime_name=runtime_name, modes=modes,
        agent_model=agent_model, agent_thinking=_thinking_value(agent_thinking),
        max_agent_steps=max_agent_steps, qa_workers=qa_workers, result_tag=result_tag,
        runtime_lifecycle=runtime_lifecycle,
        agent_step_policy=agent_step_policy,
        question_ids=question_ids, limit=limit,
        eligible_evidence_only=eligible_evidence_only, resume=resume,
        runtime_version=runtime_version, runtime_root=runtime_root,
        executable=runtime_executable, timeout_seconds=timeout_seconds,
        context_k=context_k, neighbor_turns=neighbor_turns,
    )
    _emit(ctx, result, "Kimi Agent query completed")


@benchmark.command("agent-evaluate")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--runtime", "runtime_name", type=click.Choice(["kimi"]), default="kimi", show_default=True)
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--mode", "modes", type=click.Choice(["direct", "evidence"]), multiple=True,
              default=("direct", "evidence"), show_default=True)
@click.option("--agent-model", default="qwen3.7-plus", show_default=True)
@click.option("--agent-thinking", type=click.Choice(["on", "off"]), default="on", show_default=True)
@click.option("--max-agent-steps", type=click.IntRange(min=1, max=128), default=128, show_default=True)
@click.option("--qa-workers", type=click.IntRange(min=1, max=16), default=2, show_default=True)
@click.option("--runtime-lifecycle", type=click.Choice(["per-question", "persistent"]),
              default="per-question", show_default=True)
@click.option("--agent-step-policy", type=click.Choice(["legacy", "autonomous"]),
              default="autonomous", show_default=True)
@click.option("--result-tag", default="kimi-qwen37-thinking-on", show_default=True)
@click.option("--answer-result-tag", default="qwen37-answer-v1", show_default=True)
@click.option("--answer-profile", type=click.Choice(["legacy", "normalized-v1"]),
              default="normalized-v1", show_default=True)
@click.option("--question-id", "question_ids", multiple=True)
@click.option("--limit", type=click.IntRange(min=1), default=None)
@click.option("--eligible-evidence-only", is_flag=True)
@click.option("--context-k", type=click.IntRange(min=1), default=5, show_default=True)
@click.option("--neighbor-turns", type=click.IntRange(min=0, max=10), default=1, show_default=True)
@click.option("--timeout-seconds", type=click.IntRange(min=30), default=600, show_default=True)
@click.option("--runtime-version", default="1.49.0", show_default=True)
@click.option("--runtime-root", type=click.Path(path_type=Path), default=None)
@click.option("--runtime-executable", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--resume", is_flag=True)
@click.pass_context
def agent_evaluate(
    ctx: click.Context, run_dir: Path, runtime_name: str, config_path: Path | None,
    modes: tuple[str, ...], agent_model: str, agent_thinking: str, max_agent_steps: int,
    qa_workers: int, runtime_lifecycle: str, agent_step_policy: str, result_tag: str,
    answer_result_tag: str, answer_profile: str,
    question_ids: tuple[str, ...], limit: int | None, eligible_evidence_only: bool,
    context_k: int, neighbor_turns: int, timeout_seconds: int, runtime_version: str,
    runtime_root: Path | None, runtime_executable: Path | None, resume: bool,
) -> None:
    """Run Kimi query and then replay the evidence through the shared answerer."""
    from research.benchmark.kimi_benchmark import agent_evaluate_benchmark
    result = agent_evaluate_benchmark(
        run_dir, config_path or _root_config(ctx), runtime_name=runtime_name, modes=modes,
        agent_model=agent_model, agent_thinking=_thinking_value(agent_thinking),
        max_agent_steps=max_agent_steps, qa_workers=qa_workers, result_tag=result_tag,
        runtime_lifecycle=runtime_lifecycle,
        agent_step_policy=agent_step_policy,
        question_ids=question_ids, limit=limit,
        eligible_evidence_only=eligible_evidence_only, resume=resume,
        runtime_version=runtime_version, runtime_root=runtime_root,
        executable=runtime_executable, timeout_seconds=timeout_seconds,
        context_k=context_k, neighbor_turns=neighbor_turns,
        answer_result_tag=answer_result_tag, answer_profile=answer_profile,
    )
    _emit(ctx, result, "Kimi Agent evaluation completed")


@benchmark.command("prepare")
@click.option("--dataset", type=click.Choice(["longmemeval-s", "locomo", "all"]), default="all", show_default=True)
@click.option("--data-dir", type=click.Path(path_type=Path), default=RESEARCH_ROOT / ".benchmark_data", show_default=True)
@click.option("--force", is_flag=True, help="Re-download existing files.")
@click.pass_context
def prepare(ctx: click.Context, dataset: str, data_dir: Path, force: bool) -> None:
    """Download official data and record source hashes."""
    from research.benchmark.datasets import prepare_dataset
    names = ["longmemeval-s", "locomo"] if dataset == "all" else [dataset]
    records = [prepare_dataset(name, data_dir, force=force) for name in names]
    _emit(ctx, {"datasets": records, "data_dir": str(data_dir.resolve())}, "Benchmark data prepared")


@benchmark.command("ingest")
@click.option("--dataset", type=DATASET_CHOICE, required=True)
@click.option("--run-dir", type=click.Path(path_type=Path), required=True)
@click.option("--data-dir", type=click.Path(path_type=Path), default=RESEARCH_ROOT / ".benchmark_data", show_default=True)
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--scope-id", "scope_ids", multiple=True, help="Conversation/question scope to ingest.")
@click.option("--session-limit", type=click.IntRange(min=1), default=None)
@click.option("--remember-profile", type=click.Choice(["strong-v1"]), default="strong-v1", show_default=True)
@click.option("--ingest-workers", type=click.IntRange(min=1, max=8), default=1,
              show_default=True, help="文档级并发 ingest；>1 时每文档独立 processor（共享信号量/judge）。")
@click.option("--resume", is_flag=True)
@click.pass_context
def ingest(ctx: click.Context, dataset: str, run_dir: Path, data_dir: Path,
           config_path: Path | None, scope_ids: tuple[str, ...], session_limit: int | None,
           remember_profile: str, ingest_workers: int, resume: bool) -> None:
    """Remember sessions into one isolated library per conversation scope."""
    from research.benchmark.runner import ingest_benchmark
    result = ingest_benchmark(
        dataset, data_dir, run_dir, config_path or _root_config(ctx),
        scope_ids=list(scope_ids), session_limit=session_limit,
        remember_profile=remember_profile, resume=resume,
        ingest_workers=ingest_workers,
    )
    _emit(ctx, result, "Benchmark ingestion completed")


@benchmark.command("evaluate")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--track", "tracks", type=TRACK_CHOICE, multiple=True, default=("both",), show_default=True)
@click.option("--eligible-evidence-only", is_flag=True)
@click.option("--question-id", "question_ids", multiple=True)
@click.option("--limit", type=click.IntRange(min=1), default=None)
@click.option("--max-agent-steps", type=click.IntRange(min=1, max=30), default=8, show_default=True)
@click.option("--candidate-k", type=click.IntRange(min=1), default=30, show_default=True)
@click.option("--context-k", type=click.IntRange(min=1), default=5, show_default=True)
@click.option("--top-k", type=click.IntRange(min=1), default=None,
              help="Deprecated alias for --context-k.")
@click.option("--retrieval-profile", type=click.Choice(["legacy", "hybrid-v2"]),
              default="legacy", show_default=True)
@click.option("--evidence-token-budget", type=click.IntRange(min=1), default=1600, show_default=True)
@click.option("--neighbor-turns", type=click.IntRange(min=0, max=10), default=1, show_default=True)
@click.option("--semantic-threshold", type=click.FloatRange(min=0, max=1), default=0.3, show_default=True)
@click.option("--resume", is_flag=True)
@click.option("--answer-profile", type=click.Choice(["legacy", "normalized-v1"]),
              default="normalized-v1", show_default=True)
@click.option("--agent-thinking/--no-agent-thinking", default=None,
              help="Override thinking only for Agent retrieval decisions.")
@click.option("--result-tag", default=None,
              help="Write a separate tagged track instead of replacing the base track.")
@click.pass_context
def evaluate(ctx: click.Context, run_dir: Path, config_path: Path | None,
             tracks: tuple[str, ...], eligible_evidence_only: bool,
             question_ids: tuple[str, ...], limit: int | None,
             max_agent_steps: int, candidate_k: int, context_k: int, top_k: int | None,
             retrieval_profile: str, evidence_token_budget: int, neighbor_turns: int,
             semantic_threshold: float, resume: bool, answer_profile: str,
             agent_thinking: bool | None, result_tag: str | None) -> None:
    """Run fixed baseline and/or autonomous skill-agent over a frozen library."""
    from research.benchmark.runner import evaluate_benchmark
    resolved_context_k = _context_k(ctx, context_k, top_k)
    result = evaluate_benchmark(
        run_dir, config_path or _root_config(ctx), tracks=tracks,
        eligible_evidence_only=eligible_evidence_only, question_ids=question_ids,
        limit=limit, max_agent_steps=max_agent_steps, answer_top_k=resolved_context_k,
        semantic_threshold=semantic_threshold, resume=resume,
        answer_profile=answer_profile, agent_thinking=agent_thinking, result_tag=result_tag,
        retrieval_profile=retrieval_profile, candidate_k=candidate_k,
        context_k=resolved_context_k, evidence_token_budget=evidence_token_budget,
        neighbor_turns=neighbor_turns,
    )
    _emit(ctx, result, "Benchmark evaluation completed")


@benchmark.command("retrieve")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--track", "tracks", type=TRACK_CHOICE, multiple=True,
              default=("baseline",), show_default=True)
@click.option("--eligible-evidence-only", is_flag=True)
@click.option("--question-id", "question_ids", multiple=True)
@click.option("--limit", type=click.IntRange(min=1), default=None)
@click.option("--max-agent-steps", type=click.IntRange(min=1, max=30), default=8, show_default=True)
@click.option("--candidate-k", type=click.IntRange(min=1), default=30, show_default=True)
@click.option("--context-k", type=click.IntRange(min=1), default=5, show_default=True)
@click.option("--top-k", type=click.IntRange(min=1), default=None,
              help="Deprecated alias for --context-k.")
@click.option("--retrieval-profile", type=click.Choice(["legacy", "hybrid-v2"]),
              default="hybrid-v2", show_default=True)
@click.option("--evidence-token-budget", type=click.IntRange(min=1), default=1600, show_default=True)
@click.option("--neighbor-turns", type=click.IntRange(min=0, max=10), default=1, show_default=True)
@click.option("--semantic-threshold", type=click.FloatRange(min=0, max=1), default=0.3, show_default=True)
@click.option("--agent-thinking/--no-agent-thinking", default=None)
@click.option("--result-tag", default=None)
@click.option("--resume", is_flag=True)
@click.pass_context
def retrieve(ctx: click.Context, run_dir: Path, config_path: Path | None,
             tracks: tuple[str, ...], eligible_evidence_only: bool,
             question_ids: tuple[str, ...], limit: int | None, max_agent_steps: int,
             candidate_k: int, context_k: int, top_k: int | None,
             retrieval_profile: str, evidence_token_budget: int, neighbor_turns: int,
             semantic_threshold: float, agent_thinking: bool | None,
             result_tag: str | None, resume: bool) -> None:
    """Persist retrieval evidence without invoking the answer model."""
    from research.benchmark.runner import retrieve_benchmark
    resolved_context_k = _context_k(ctx, context_k, top_k)
    result = retrieve_benchmark(
        run_dir, config_path or _root_config(ctx), tracks=tracks,
        eligible_evidence_only=eligible_evidence_only, question_ids=question_ids,
        limit=limit, max_agent_steps=max_agent_steps,
        semantic_threshold=semantic_threshold, retrieval_profile=retrieval_profile,
        candidate_k=candidate_k, context_k=resolved_context_k,
        evidence_token_budget=evidence_token_budget, neighbor_turns=neighbor_turns,
        resume=resume, agent_thinking=agent_thinking, result_tag=result_tag,
    )
    _emit(ctx, result, "Benchmark retrieval completed")


@benchmark.command("answer")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--track", "tracks", type=TRACK_CHOICE, multiple=True, default=("both",), show_default=True)
@click.option(
    "--source-track", "source_tracks", multiple=True,
    help="Replay an exact named track/variant (for example skill-agent-query-on-v3).",
)
@click.option("--question-id", "question_ids", multiple=True)
@click.option("--result-tag", default="answer-normalized-v1", show_default=True)
@click.option("--answer-profile", type=click.Choice(["legacy", "normalized-v1"]),
              default="normalized-v1", show_default=True)
@click.option("--resume", is_flag=True)
@click.pass_context
def answer(ctx: click.Context, run_dir: Path, config_path: Path | None,
           tracks: tuple[str, ...], source_tracks: tuple[str, ...],
           question_ids: tuple[str, ...], result_tag: str,
           answer_profile: str, resume: bool) -> None:
    """Replay only the answer stage from persisted retrieval/Agent evidence."""
    from research.benchmark.runner import replay_answers
    selected_tracks = source_tracks or tracks
    result = replay_answers(
        run_dir, config_path or _root_config(ctx), tracks=selected_tracks,
        result_tag=result_tag, answer_profile=answer_profile,
        question_ids=question_ids, resume=resume,
    )
    _emit(ctx, result, "Benchmark answers replayed")


@benchmark.command("run")
@click.option("--dataset", type=DATASET_CHOICE, required=True)
@click.option("--data-dir", type=click.Path(path_type=Path), default=RESEARCH_ROOT / ".benchmark_data", show_default=True)
@click.option("--run-dir", type=click.Path(path_type=Path), default=None)
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--scope-id", "scope_ids", multiple=True)
@click.option("--session-limit", type=click.IntRange(min=1), default=None)
@click.option("--remember-profile", type=click.Choice(["strong-v1"]), default="strong-v1", show_default=True)
@click.option("--track", "tracks", type=TRACK_CHOICE, multiple=True, default=("both",), show_default=True)
@click.option("--eligible-evidence-only", is_flag=True)
@click.option("--limit", type=click.IntRange(min=1), default=None)
@click.option("--question-id", "question_ids", multiple=True)
@click.option("--resume", is_flag=True)
@click.option("--top-k", type=click.IntRange(min=1), default=5, show_default=True)
@click.option("--semantic-threshold", type=click.FloatRange(min=0, max=1), default=0.3, show_default=True)
@click.option("--max-agent-steps", type=click.IntRange(min=1, max=30), default=8, show_default=True)
@click.option("--retrieval-mode", type=click.Choice(["agentic", "single-pass"]), default=None,
              help="Deprecated compatibility alias for a single track.")
@click.pass_context
def run(ctx: click.Context, dataset: str, data_dir: Path, run_dir: Path | None,
        config_path: Path | None, scope_ids: tuple[str, ...], session_limit: int | None,
        remember_profile: str, tracks: tuple[str, ...], eligible_evidence_only: bool,
        limit: int | None, question_ids: tuple[str, ...], resume: bool, top_k: int,
        semantic_threshold: float, max_agent_steps: int, retrieval_mode: str | None) -> None:
    """Shortcut for ingest followed by dual-track evaluation."""
    from research.benchmark.runner import run_benchmark
    run_dir = run_dir or _default_run_dir(dataset)
    result = run_benchmark(
        dataset, data_dir, run_dir, config_path or _root_config(ctx),
        limit=limit, question_ids=list(question_ids), resume=resume,
        answer_top_k=top_k, semantic_threshold=semantic_threshold,
        retrieval_mode=retrieval_mode, max_agent_steps=max_agent_steps,
        tracks=tracks, scope_ids=list(scope_ids), session_limit=session_limit,
        remember_profile=remember_profile, eligible_evidence_only=eligible_evidence_only,
    )
    _emit(ctx, result, "Benchmark run completed")


@benchmark.command("score")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--no-judge", is_flag=True)
@click.pass_context
def score(ctx: click.Context, run_dir: Path, no_judge: bool) -> None:
    """Calculate official-compatible metrics for every track."""
    from research.benchmark.scoring import score_run
    _emit(ctx, score_run(run_dir, judge=not no_judge), "Benchmark scoring completed")


@benchmark.command("judge")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--track", "tracks", multiple=True, default=("baseline", "baseline-hybrid-v2-answer-v1"))
@click.option("--protocol", type=click.Choice(["locomo-1540", "locomo-1986"]), default="locomo-1540", show_default=True)
@click.option("--judge-tag", default="qwen37", show_default=True)
@click.option(
    "--judge-profile",
    type=click.Choice(["legacy-mirror", "mem0-current-exact"]),
    default="legacy-mirror",
    show_default=True,
)
@click.option("--max-workers", type=click.IntRange(min=1, max=32), default=4, show_default=True)
@click.option("--resume/--no-resume", default=True, show_default=True)
@click.pass_context
def judge(
    ctx: click.Context,
    run_dir: Path,
    config_path: Path | None,
    tracks: tuple[str, ...],
    protocol: str,
    judge_tag: str,
    judge_profile: str,
    max_workers: int,
    resume: bool,
) -> None:
    """Run a versioned, resumable semantic judge over existing answers."""
    from research.benchmark.judging import judge_run
    result = judge_run(
        run_dir,
        config_path or _root_config(ctx),
        tracks=list(tracks),
        protocol=protocol,
        judge_tag=judge_tag,
        judge_profile=judge_profile,
        max_workers=max_workers,
        resume=resume,
    )
    _emit(ctx, result, "Benchmark semantic judging completed")


@benchmark.command("judge-batch")
@click.argument("action", type=click.Choice(["submit", "status", "collect"]))
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--track", "tracks", multiple=True, default=("baseline", "baseline-hybrid-v2-answer-v1"))
@click.option("--protocol", type=click.Choice(["locomo-1540", "locomo-1986"]), default="locomo-1540", show_default=True)
@click.option("--judge-tag", default="qwen37-batch", show_default=True)
@click.pass_context
def judge_batch(ctx: click.Context, action: str, run_dir: Path, config_path: Path | None,
                tracks: tuple[str, ...], protocol: str, judge_tag: str) -> None:
    """Submit, inspect, or collect resumable semantic judging batch jobs."""
    from research.benchmark.judging import batch_judge
    result = batch_judge(
        run_dir, config_path or _root_config(ctx), action=action, tracks=list(tracks),
        protocol=protocol, judge_tag=judge_tag,
    )
    _emit(ctx, result, f"Benchmark batch judging {action} completed")


@benchmark.command("report")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.pass_context
def report(ctx: click.Context, run_dir: Path) -> None:
    """Regenerate per-track and comparison reports."""
    from research.benchmark.scoring import report_run
    _emit(ctx, report_run(run_dir), "Benchmark report generated")


@benchmark.command("diagnose")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.pass_context
def diagnose(ctx: click.Context, run_dir: Path) -> None:
    """Attribute failures and audit conversation/evidence isolation."""
    from research.benchmark.scoring import diagnose_run
    _emit(ctx, diagnose_run(run_dir), "Benchmark diagnosis completed")


@benchmark.command("compare")
@click.argument("run_a", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("run_b", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.pass_context
def compare(ctx: click.Context, run_a: Path, run_b: Path) -> None:
    """Compare matching tracks from two scored runs."""
    from research.benchmark.scoring import compare_runs
    _emit(ctx, compare_runs(run_a, run_b), "Benchmark runs compared")
