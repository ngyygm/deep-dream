"""Result persistence and human-readable benchmark reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def latest_by_question(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse an append-only result journal to the latest record per question."""
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id = str(record.get("question_id", ""))
        if question_id:
            latest[question_id] = record
    return list(latest.values())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def render_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [f"# {summary['dataset']} benchmark report", "", f"- Total: {summary['total']}", f"- Scored: {summary['scored']}"]
    if summary.get("overall") is not None:
        lines.append(f"- Overall: {summary['overall']:.4f}")
    lines.extend(["", "## Scores by type", "", "| Type | Score | Count |", "|---|---:|---:|"])
    for key, value in summary.get("by_type", {}).items():
        score = value.get("score")
        lines.append(f"| {key} | {'n/a' if score is None else f'{score:.4f}'} | {value['count']} |")
    lines.extend(["", "## Retrieval", "", "| Metric | Score |", "|---|---:|"])
    for key, value in summary.get("retrieval", {}).items():
        lines.append(f"| {key} | {value:.4f} |")
    runtime = summary.get("runtime") or {}
    lines.extend(["", "## Runtime", "", f"- Completed: {runtime.get('completed', 0)}", f"- Failed: {runtime.get('failed', 0)}"])
    if runtime.get("average_latency_seconds") is not None:
        lines.append(f"- Average latency: {runtime['average_latency_seconds']:.3f}s")
        lines.append(f"- Median latency: {runtime['median_latency_seconds']:.3f}s")
        lines.append(f"- P95 latency: {runtime['p95_latency_seconds']:.3f}s")
    agent = summary.get("agent")
    if agent:
        lines.extend([
            "", "## Agent trajectory", "",
            f"- Questions: {agent['questions']}",
            f"- Average steps: {agent['average_steps']:.2f}",
            f"- Tool counts: `{json.dumps(agent['tool_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- Stop reasons: `{json.dumps(agent['stop_reasons'], ensure_ascii=False, sort_keys=True)}`",
        ])
    run = summary.get("run") or {}
    if run:
        lines.extend([
            "", "## Run configuration", "",
            f"- Track: `{summary.get('track') or run.get('retrieval_mode')}`",
            f"- Remember profile: `{run.get('remember_profile')}`",
            f"- Max agent steps: `{run.get('max_agent_steps')}`",
            f"- Answer top-k: `{run.get('answer_top_k')}`",
            f"- Git commit: `{run.get('git_commit')}`",
            f"- Dataset SHA-256: `{run.get('dataset_sha256')}`",
            f"- Models/chunking: `{json.dumps(run.get('config') or {}, ensure_ascii=False, sort_keys=True)}`",
        ])
    if summary.get("failure_attribution"):
        lines.extend([
            "", "## Failure attribution", "",
            f"`{json.dumps(summary['failure_attribution'], ensure_ascii=False, sort_keys=True)}`",
        ])
    failures = [r for r in records if r.get("score") == 0][:20]
    if failures:
        lines.extend(["", "## Failure samples", ""])
        for row in failures:
            lines.append(
                f"- `{row['question_id']}` [{row.get('failure_attribution') or 'unknown'}] "
                f"{row.get('question', '')} — {row.get('hypothesis', '')}"
            )
    return "\n".join(lines) + "\n"
