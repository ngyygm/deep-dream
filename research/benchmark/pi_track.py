"""pi agent 轨道：用 Deep-Dream pi harness（headless）答题。

每个问题独立 workdir，经 ``harness/pi/extensions/deep-dream.ts`` 注册的
dd_scope/dd_search 工具访问该 scope 的冻结库；产物 ``results.{tag}.jsonl``
与内置 baseline/skill-agent 轨道同格式，judge/score 直接可用。

用法::

    python -m research.benchmark.pi_track research/.benchmark_runs/<run> \
        --config research/service_config.kimi-par.json [--limit N] [--scope-id S]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_PATH = REPO_ROOT / "harness" / "pi" / "extensions" / "deep-dream.ts"

# 答案归一化规则：与 runner.py normalized-v1 答案阶段同源，压缩为单段，
# 证据来源从"Submitted evidence"换成 agent 自主检索的记忆库。
_PROMPT_TEMPLATE = """You are answering a question about past conversations stored in a Deep-Dream memory library.

Workflow: first call dd_scope(query, materialize=true) to bound a document scope for the question (the graph backtrace returns matching concepts, documents, and episode offsets, and materializes a sandbox directory of symlinks), then read/verify inside the sandbox with bash (grep/sed) before answering. Use dd_search only for lightweight concept lookup. Do not answer from memory without checking the library.

Answer rules (from the benchmark protocol):
- Use only what the library evidence supports. If the asked fact is not supported, answer exactly "No information available." If the evidence attributes the event to a different person than asked about (false premise), also answer "No information available."
- Resolve indirect references across sessions; respect later updates over earlier ones.
- Convert relative dates to absolute dates using the session timestamps ("last week" -> the week before that session's date; "yesterday" -> the absolute date).
- List answers must name every distinct item found in any relevant session, not just the first.
- Keep each fact to its minimal phrase — one tight clause, no trailing explanation after ';' or 'which'. Multiple requested facts are separated by commas.
- Direct factual Yes/No questions: answer exactly "Yes" or "No". Hypothetical/likely Yes/No judgments may use the compact form "Yes, since <short evidence>" or "No, since <short evidence>". A question starting with What/Who/Where/When/Which/How is never a Yes/No answer.
- Image captions describe photographs and must not replace narrated facts.
{date_line}
Question: {question}

End your final message with exactly one line in the form:
ANSWER: <concise answer>

The ANSWER line must contain ONLY the answer itself — a bare fact span like "May 7, 2023" or "Counseling, mental health". No parentheses, no explanations, no evidence notes, no quotes around the whole answer; reasoning belongs in earlier lines, not on the ANSWER line."""


def _safe_dirname(scope_id: str) -> str:
    """与 runner._safe_id 一致的确定命名（readable 前缀 + sha256[:10]）。"""
    readable = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in scope_id)[:48]
    return f"{readable}_{hashlib.sha256(scope_id.encode()).hexdigest()[:10]}"


def _find_scope_library(run_dir: Path, scope_id: str) -> Path:
    # 精确匹配优先：LongMemEval 存在 0862e8bf / 0862e8bf_abs 这类前缀嵌套
    # scope，旧的 {scope_id}_* glob 会解析到兄弟 scope 的库（必定答错）。
    exact = run_dir / "libraries" / _safe_dirname(scope_id) / "library.db"
    if exact.exists():
        return exact.parent
    hits = sorted((run_dir / "libraries").glob(f"{scope_id}_*/library.db"))
    if not hits:
        raise FileNotFoundError(f"Scope library not found for {scope_id}: {run_dir}/libraries/{scope_id}_*/library.db")
    return hits[0].parent


def _write_scope_config(base_config: Path, scope_dir: Path, out_path: Path) -> None:
    cfg = json.loads(base_config.read_text(encoding="utf-8"))
    cfg["storage_path"] = str(scope_dir)
    out_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


_ANSWER_RE = re.compile(r"^\s*\*?\*?ANSWER\*?\*?\s*[:：]\s*(.+?)\s*$", re.MULTILINE)


def _extract_answer(text: str) -> str:
    hits = _ANSWER_RE.findall(text or "")
    return hits[-1].strip().strip("*") if hits else (text or "").strip()


def _parse_pi_events(raw: str) -> dict[str, Any]:
    """从 headless JSON 事件流提取末答/用量/工具轨迹摘要。"""
    final_text = ""
    tool_calls: list[dict[str, Any]] = []
    usage_total = {"input": 0, "output": 0}
    settled = False
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        if etype == "message_end":
            message = event.get("message") or {}
            if message.get("role") == "assistant":
                texts = [
                    block.get("text", "")
                    for block in message.get("content", [])
                    if block.get("type") == "text"
                ]
                joined = "\n".join(t for t in texts if t)
                if joined.strip():
                    final_text = joined
                u = message.get("usage") or {}
                usage_total["input"] += int(u.get("input") or 0)
                usage_total["output"] += int(u.get("output") or 0)
                for block in message.get("content", []):
                    if block.get("type") == "toolCall":
                        tool_calls.append({
                            "name": block.get("name"),
                            "arguments": block.get("arguments"),
                        })
        elif etype == "agent_settled":
            settled = True
    return {
        "final_text": final_text,
        "answer": _extract_answer(final_text),
        "settled": settled,
        "tool_calls": tool_calls,
        "usage": usage_total,
    }


def _run_pi(
    workdir: Path,
    prompt: str,
    *,
    provider: str,
    model: str,
    scope_config: Path,
    cli_cmd: str,
    timeout_s: int,
    dd_timeout_s: int,
) -> dict[str, Any]:
    env = dict(os.environ)
    env["DD_CLI"] = cli_cmd
    env["DD_CONFIG"] = str(scope_config)
    env["DD_TIMEOUT"] = str(dd_timeout_s)
    cmd = [
        "pi",
        "-e", str(EXTENSION_PATH),
        "--provider", provider,
        "--model", model,
        "--mode", "json",
        "--no-session",
        "-p", prompt,
    ]
    started = time.monotonic()
    proc = subprocess.run(  # noqa: S603 - 固定参数列表
        cmd, cwd=str(workdir), env=env,
        capture_output=True, text=True, timeout=timeout_s,
    )
    latency = time.monotonic() - started
    parsed = _parse_pi_events(proc.stdout)
    parsed["latency_seconds"] = latency
    parsed["exit_code"] = proc.returncode
    parsed["stderr_tail"] = proc.stderr[-2000:]
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pi agent track over a frozen benchmark run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--track-tag", default="pi", help="Output track tag (results.<tag>.jsonl)")
    parser.add_argument("--provider", default="kimi-sz")
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scope-id", default=None)
    parser.add_argument("--question-id", default=None)
    parser.add_argument("--timeout", type=int, default=900, help="Per-question wall-clock seconds")
    parser.add_argument("--dd-timeout", type=int, default=300, help="Per-CLI-call seconds inside the agent")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent pi processes (scope 库只读并发安全)")
    parser.add_argument("--cli", default=None, help="Deep-Dream CLI command (default: <repo>/.venv/bin/python -m core.cli)")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args(argv)

    from research.benchmark.datasets import data_dir_for_dataset_path, load_benchmark

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    dataset = manifest["dataset"]
    data_dir = data_dir_for_dataset_path(dataset, Path(manifest["dataset_path"]))
    items, _ = load_benchmark(dataset, data_dir)

    if args.question_id:
        items = [i for i in items if i.question_id == args.question_id]
    if args.scope_id:
        items = [i for i in items if i.scope_id == args.scope_id]
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("No matching questions.", file=sys.stderr)
        return 2

    results_path = run_dir / f"results.{args.track_tag}.jsonl"
    done: set[str] = set()
    if args.resume and results_path.exists():
        for row in results_path.read_text(encoding="utf-8").split("\n"):  # 只按\n切，记录内可能含\u2028 等
            try:
                rec = json.loads(row)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "completed":
                done.add(rec["question_id"])

    cli_cmd = args.cli or f"{REPO_ROOT}/.venv/bin/python -m core.cli"
    events_dir = run_dir / f"pi_events.{args.track_tag}"
    work_root = run_dir / f"pi_work.{args.track_tag}"
    scope_configs: dict[str, Path] = {}

    print(f"pi track: {len(items)} questions ({len(done)} already done) -> {results_path}")
    events_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    # 预解析 scope 库与配置：主线程一次性完成，worker 内只读（config 文件
    # 并发写同一路径会互相截断）。
    pending: list[Any] = []
    for item in items:
        if item.question_id in done:
            continue
        try:
            _find_scope_library(run_dir, item.scope_id)
        except FileNotFoundError as exc:
            print(f"SKIP {item.question_id}: {exc}", file=sys.stderr)
            continue
        pending.append(item)
    for scope_id in sorted({i.scope_id for i in pending}):
        if scope_id in scope_configs:
            continue
        scope_dir = _find_scope_library(run_dir, scope_id)
        cfg_path = work_root / f"{scope_id}.service_config.json"
        _write_scope_config(args.config.resolve(), scope_dir, cfg_path)
        scope_configs[scope_id] = cfg_path

    write_lock = threading.Lock()
    counters = {"completed": 0, "errors": 0, "finished": 0}

    def _answer_one(item: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "dataset": dataset,
            "scope_id": item.scope_id,
            "question_id": item.question_id,
            "question_type": item.question_type,
            "question": item.question,
            "question_date": item.question_date,
            "track": args.track_tag,
            "status": "error",
            "model": args.model,
            "provider": args.provider,
        }
        workdir = work_root / item.question_id.replace(":", "_")
        workdir.mkdir(parents=True, exist_ok=True)
        date_line = f"\nQuestion date: {item.question_date}" if item.question_date else ""
        prompt = _PROMPT_TEMPLATE.format(date_line=date_line, question=item.question)
        try:
            parsed = _run_pi(
                workdir, prompt,
                provider=args.provider, model=args.model,
                scope_config=scope_configs[item.scope_id],
                cli_cmd=cli_cmd,
                timeout_s=args.timeout,
                dd_timeout_s=args.dd_timeout,
            )
            record.update({
                # judge/scoring 契约：answer=gold（dataset 提供），hypothesis=生成答案。
                "answer": item.answer,
                "hypothesis": parsed["answer"],
                "answer_payload": {"answer": parsed["answer"], "support": "agent_verified"},
                "status": "completed" if parsed["settled"] and parsed["answer"] else "error",
                "total_latency_seconds": round(parsed["latency_seconds"], 2),
                "prompt_tokens": parsed["usage"]["input"],
                "completion_tokens": parsed["usage"]["output"],
                "agent_steps": len(parsed["tool_calls"]),
                "tool_trajectory": parsed["tool_calls"],
                "raw_answer_response": parsed["final_text"][-4000:],
                "answer_profile": "pi-headless-v1",
                "answer_attempts": 1,
                "result_tag": None,
                "source_track": args.track_tag,
                "evidence_session_ids": item.evidence_session_ids,
                "evidence_turn_ids": item.evidence_turn_ids,
                "retrieved": [],
                "ranked_session_ids": [],
                "ranked_turn_ids": [],
                "retrieval_model": args.model,
            })
            (events_dir / f"{item.question_id.replace(':', '_')}.json").write_text(
                json.dumps({
                    "exit_code": parsed["exit_code"],
                    "stderr_tail": parsed["stderr_tail"],
                    "final_text": parsed["final_text"],
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except subprocess.TimeoutExpired:
            record["status"] = "timeout"
        except Exception as exc:  # noqa: BLE001 - 单题失败不终止轨道
            record["error"] = str(exc)[:2000]
        return record

    def _record_result(record: dict[str, Any]) -> None:
        with write_lock:
            with results_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["status"] == "completed":
                counters["completed"] += 1
            else:
                counters["errors"] += 1
            counters["finished"] += 1
            preview = (record.get("answer") or record.get("error") or "")[:80]
            print(
                f"[{counters['finished']}/{len(pending)}] {record['question_id']} {record['status']}"
                f" ({record.get('agent_steps', 0)} steps, {record.get('total_latency_seconds', 0)}s): {preview}",
                flush=True,
            )

    if args.workers <= 1:
        for item in pending:
            _record_result(_answer_one(item))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_answer_one, item) for item in pending]
            for future in as_completed(futures):
                _record_result(future.result())

    print(f"Done: {counters['completed']} completed, {counters['errors']} errors -> {results_path}")
    return 0 if counters["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
