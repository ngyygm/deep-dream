"""Official-compatible per-track scoring, diagnosis, and comparisons."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from .metrics import aggregate_records, locomo_f1
from .reporting import append_jsonl, latest_by_question, read_jsonl, render_markdown, write_json


# 官方 judge 默认按 gpt-4o 协议；换 OpenAI 兼容端点（如 kimi-k3）时用环境变量
# 覆盖。kimi-k3 等 reasoning 模型必须抬高 max_tokens，否则预算全花在
# reasoning_content 上、content 为空，全部判 no（实测 10 -> 1024 可用）。
JUDGE_MODEL = os.environ.get("LONGMEMEVAL_JUDGE_MODEL", "gpt-4o-2024-08-06")
JUDGE_MAX_TOKENS = int(os.environ.get("LONGMEMEVAL_JUDGE_MAX_TOKENS", "10"))
JUDGE_BASE_URL = os.environ.get("LONGMEMEVAL_JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")


def longmemeval_judge_prompt(task: str, question: str, answer: str, response: str,
                             abstention: bool = False) -> str:
    if abstention:
        return ("I will give you an unanswerable question, an explanation, and a response from a model. "
                "Please answer yes if the model correctly identifies the question as unanswerable. The model "
                "could say that the information is incomplete, or some other information is given but the asked "
                f"information is not.\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: "
                f"{response}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.")
    common = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if "
              "the response contains the correct answer. Otherwise, answer no. If the response is equivalent to "
              "the correct answer or contains all the intermediate steps to get the correct answer, you should "
              "also answer yes. ")
    if task == "single-session-preference":
        return ("I will give you a question, a rubric for desired personalized response, and a response from a "
                "model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                "The model does not need to reflect all the points in the rubric. The response is correct as long "
                f"as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {question}"
                f"\n\nRubric: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only.")
    if task == "temporal-reasoning":
        common += ("If the response only contains a subset of the information required by the answer, answer no. "
                   "In addition, do not penalize off-by-one errors for the number of days. If the question asks "
                   "for the number of days/weeks/months, etc., and the model makes off-by-one errors, the model's "
                   "response is still correct. ")
    elif task == "knowledge-update":
        common = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if "
                  "the response contains the correct answer. Otherwise, answer no. If the response contains some "
                  "previous information along with an updated answer, the response should be considered as correct "
                  "as long as the updated answer is the required answer. ")
    else:
        common += "If the response only contains a subset of the information required by the answer, answer no. "
    return (f"{common}\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
            "\n\nIs the model response correct? Answer yes or no only.")


def _judge(record: dict[str, Any], api_key: str) -> tuple[bool, str]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=JUDGE_BASE_URL)
    prompt = longmemeval_judge_prompt(
        str(record["question_type"]), record["question"], record["answer"], record["hypothesis"],
        abstention=str(record["question_id"]).endswith("_abs"),
    )
    last_error = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=JUDGE_MAX_TOKENS,
            )
            text = response.choices[0].message.content.strip()
            return "yes" in text.lower(), text
        except Exception as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LongMemEval judge failed: {last_error}")


def _result_paths(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    if int(manifest.get("schema_version") or 2) < 3:
        return {str(manifest.get("retrieval_mode") or "default"): run_dir / "results.jsonl"}
    tracks = manifest.get("tracks") or [
        path.name[len("results."):-len(".jsonl")]
        for path in sorted(run_dir.glob("results.*.jsonl"))
    ]
    return {track: run_dir / f"results.{track}.jsonl" for track in tracks
            if (run_dir / f"results.{track}.jsonl").exists()}


def _temporal_equivalent(answer: str, hypothesis: str) -> bool:
    """Small diagnostic only; never alters official LoCoMo F1."""
    months = {name.lower(): index for index, name in enumerate(
        ["January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"], 1)}
    reference = re.search(r"week before\s+(\d{1,2})\s+([A-Za-z]+)", answer, re.I)
    explicit = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\b", hypothesis, re.I)
    if not reference or not explicit:
        return False
    month = months.get(reference.group(2).lower())
    other_month = months.get(explicit.group(2).lower())
    if not month or not other_month:
        return False
    expected = datetime(2000, month, int(reference.group(1))) - timedelta(days=7)
    return (expected.day, expected.month) == (int(explicit.group(1)), other_month)


def _final_attribution(record: dict[str, Any]) -> str | None:
    if record.get("status") == "error":
        return record.get("failure_attribution") or "agent_stopped_early"
    prior = record.get("failure_attribution")
    if prior in {"remember_missing", "retrieval_miss", "agent_stopped_early"}:
        return prior
    score = record.get("score")
    if score is None or float(score) >= 0.999999:
        return None
    if _temporal_equivalent(str(record.get("answer") or ""), str(record.get("hypothesis") or "")):
        return "scoring_mismatch"
    return "answer_reasoning"


def _score_track(
    run_dir: Path,
    manifest: dict[str, Any],
    track: str,
    path: Path,
    *,
    judge: bool,
) -> dict[str, Any]:
    dataset = manifest["dataset"]
    records = latest_by_question(read_jsonl(path))
    if dataset == "locomo":
        for record in records:
            record["score"] = 0.0 if record.get("status") == "error" else locomo_f1(
                record["hypothesis"], record["answer"], record["question_type"]
            )
    elif dataset == "longmemeval-s":
        if not judge:
            for record in records:
                record["score"] = None
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for the official LongMemEval judge")
            judge_path = run_dir / f"judge_results.{track}.jsonl"
            prior = {row["question_id"]: row for row in read_jsonl(judge_path)}
            for record in records:
                if record.get("status") == "error":
                    record["score"] = 0.0
                    continue
                judged = prior.get(record["question_id"])
                if judged is None:
                    label, raw = _judge(record, api_key)
                    judged = {"question_id": record["question_id"], "autoeval_label": {
                        "model": JUDGE_MODEL, "label": label, "raw": raw,
                    }}
                    append_jsonl(judge_path, judged)
                record["score"] = float(judged["autoeval_label"]["label"])
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    for record in records:
        record["failure_attribution"] = _final_attribution(record)

    summary = aggregate_records(records, dataset)
    summary["track"] = track
    summary["variant"] = (manifest.get("track_variants") or {}).get(track)
    summary["failure_attribution"] = dict(Counter(
        row["failure_attribution"] for row in records if row.get("failure_attribution")
    ))
    summary["run"] = {key: manifest.get(key) for key in (
        "remember_profile", "max_agent_steps", "answer_top_k", "git_commit",
        "dataset_sha256", "runtime_policy", "skill", "config",
    )}
    if dataset == "longmemeval-s":
        type_scores = [value["score"] for value in summary["by_type"].values() if value["score"] is not None]
        summary["task_averaged"] = sum(type_scores) / len(type_scores) if type_scores else None
        abstention = [row["score"] for row in records if row.get("score") is not None
                      and str(row["question_id"]).endswith("_abs")]
        summary["abstention"] = sum(abstention) / len(abstention) if abstention else None
        predictions = [{"question_id": row["question_id"], "hypothesis": row["hypothesis"]}
                       for row in records]
        (run_dir / f"predictions.{track}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8"
        )
    (run_dir / f"scored_results.{track}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    write_json(run_dir / f"summary.{track}.json", summary)
    (run_dir / f"report.{track}.md").write_text(render_markdown(summary, records), encoding="utf-8")
    return summary


def _comparison(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparison: dict[str, Any] = {"tracks": summaries}
    if "baseline" in summaries and "skill-agent" in summaries:
        left, right = summaries["baseline"], summaries["skill-agent"]
        comparison["skill_agent_minus_baseline"] = {
            "overall": None if left.get("overall") is None or right.get("overall") is None
            else right["overall"] - left["overall"],
            "retrieval": {
                key: right.get("retrieval", {}).get(key, 0) - value
                for key, value in left.get("retrieval", {}).items()
                if key in right.get("retrieval", {})
            },
        }
    return comparison


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = ["# Benchmark Track Comparison", "", "| Track | Samples | Overall |", "|---|---:|---:|"]
    for track, summary in comparison.get("tracks", {}).items():
        overall = summary.get("overall")
        lines.append(f"| {track} | {summary.get('total', 0)} | {'n/a' if overall is None else f'{overall:.4f}'} |")
    delta = comparison.get("skill_agent_minus_baseline", {}).get("overall")
    if delta is not None:
        lines.extend(["", f"Skill-agent − baseline overall: **{delta:+.4f}**"])
    return "\n".join(lines) + "\n"


def score_run(run_dir: Path, *, judge: bool = True) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    paths = _result_paths(run_dir, manifest)
    summaries = {track: _score_track(run_dir, manifest, track, path, judge=judge)
                 for track, path in paths.items()}
    if int(manifest.get("schema_version") or 2) < 3:
        summary = next(iter(summaries.values()))
        # Preserve v2 filenames for existing runs and integrations.
        track = next(iter(summaries))
        (run_dir / "scored_results.jsonl").write_bytes((run_dir / f"scored_results.{track}.jsonl").read_bytes())
        write_json(run_dir / "summary.json", summary)
        (run_dir / "report.md").write_bytes((run_dir / f"report.{track}.md").read_bytes())
        return summary
    comparison = _comparison(summaries)
    write_json(run_dir / "comparison.json", comparison)
    (run_dir / "comparison.md").write_text(_comparison_markdown(comparison), encoding="utf-8")
    return comparison


def report_run(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("schema_version") or 2) < 3:
        if not (run_dir / "summary.json").exists():
            return score_run(run_dir, judge=False)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        records = latest_by_question(read_jsonl(
            run_dir / ("scored_results.jsonl" if (run_dir / "scored_results.jsonl").exists() else "results.jsonl")
        ))
        (run_dir / "report.md").write_text(render_markdown(summary, records), encoding="utf-8")
        return summary
    summaries = {}
    for track in _result_paths(run_dir, manifest):
        path = run_dir / f"summary.{track}.json"
        if not path.exists():
            return score_run(run_dir, judge=False)
        summary = json.loads(path.read_text(encoding="utf-8"))
        records = latest_by_question(read_jsonl(run_dir / f"scored_results.{track}.jsonl"))
        (run_dir / f"report.{track}.md").write_text(render_markdown(summary, records), encoding="utf-8")
        summaries[track] = summary
    comparison = _comparison(summaries)
    write_json(run_dir / "comparison.json", comparison)
    (run_dir / "comparison.md").write_text(_comparison_markdown(comparison), encoding="utf-8")
    return comparison


def diagnose_run(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    findings: dict[str, Any] = {"schema_version": manifest.get("schema_version"), "tracks": {}}
    for track, path in _result_paths(run_dir, manifest).items():
        scored_path = run_dir / f"scored_results.{track}.jsonl"
        records = latest_by_question(read_jsonl(scored_path if scored_path.exists() else path))
        allowed = {
            scope: set(info.get("visible_sessions") or [])
            for scope, info in (manifest.get("scopes") or {}).items()
        }
        leaks = []
        for record in records:
            permitted = allowed.get(record.get("scope_id"), set())
            exposed = {
                row.get("session_id") for row in record.get("retrieved") or [] if row.get("session_id")
            }
            if not exposed.issubset(permitted):
                leaks.append(record.get("question_id"))
        findings["tracks"][track] = {
            "records": len(records),
            "errors": sum(row.get("status") == "error" for row in records),
            "failure_attribution": dict(Counter(
                row.get("failure_attribution") for row in records if row.get("failure_attribution")
            )),
            "leakage_question_ids": leaks,
            "traceable_evidence": all(
                all(ev.get("session_id") or row.get("session_id")
                    for row in record.get("retrieved") or [] for ev in row.get("evidence") or [{}])
                for record in records
            ),
        }
    write_json(run_dir / "diagnosis.json", findings)
    return findings


def compare_runs(run_a: Path, run_b: Path) -> dict[str, Any]:
    def summaries(path: Path) -> dict[str, Any]:
        manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
        result = {}
        for track in _result_paths(path, manifest):
            summary_path = path / f"summary.{track}.json"
            if summary_path.exists():
                result[track] = json.loads(summary_path.read_text(encoding="utf-8"))
        if not result and (path / "summary.json").exists():
            result["default"] = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        return result

    left, right = summaries(run_a), summaries(run_b)
    result = {"run_a": str(run_a), "run_b": str(run_b), "tracks": {}}
    for track in sorted(set(left) & set(right)):
        a, b = left[track], right[track]
        result["tracks"][track] = {
            "overall_a": a.get("overall"), "overall_b": b.get("overall"),
            "overall_delta": None if a.get("overall") is None or b.get("overall") is None
            else b["overall"] - a["overall"],
            "retrieval_delta": {
                key: b.get("retrieval", {}).get(key, 0) - value
                for key, value in a.get("retrieval", {}).items()
                if key in b.get("retrieval", {})
            },
        }
    return result
