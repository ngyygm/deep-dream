"""Resumable LoCoMo semantic judging with an OpenAI-compatible model.

The prompt mirrors the no-evidence judge protocol in mem0ai/memory-benchmarks
commit 4b61c5d31b9c668a12b4f5e78064248a02c82d2b (Apache-2.0).  The model is
always recorded because scores from different judges are not interchangeable.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from core.llm.chat_api import openai_compatible_chat

from .reporting import append_jsonl, latest_by_question, read_jsonl, write_json


MEM0_JUDGE_SOURCE_COMMIT = "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"
MEM0_JUDGE_SYSTEM = (
    "You are evaluating conversational AI memory recall. "
    "Return JSON only with the format requested."
)
MEM0_JUDGE_PROMPT = """Label the generated answer as CORRECT or WRONG.

Rules:
1. Partial credit: if at least one correct item from a gold-answer list is present, mark CORRECT.
2. Paraphrases and semantically equivalent emotions count as CORRECT.
3. Extra detail is allowed when the core gold fact remains present.
4. Dates within 14 days and durations within 50 percent count as equivalent.
5. Judge semantic overlap and the underlying concept, not exact wording.
6. Identifying the same named referent counts even when descriptions differ.
7. Only mark WRONG when the answer contains no correct gold item or addresses a different topic.

Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return one JSON object with "reasoning" (one sentence) and "label" (CORRECT or WRONG)."""

MEM0_CURRENT_EXACT_JUDGE_PROMPT = """Label the generated answer as CORRECT or WRONG.

## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. "Chocolate raspberry tart" = "chocolate cake with raspberries". "Shelter meal service" = "volunteering at a homeless shelter". Emotions and sentiments in the same positive/negative family count as paraphrases: "proud" = "fulfilled" = "accomplished"; "huge success" = "relieved" = "thrilled" (all express positive achievement). Judge semantic meaning, not exact wording.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed or specific. If the generated answer adds extra descriptive details beyond the gold answer while still referencing the same core entity or concept, mark CORRECT.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT (e.g., "5 months" matches "six months"; "19 days" matches "two weeks"). Relative dates ("few days before November") match specific dates in the same window. A specific date (e.g., "February 2020") that is consistent with a vague reference (e.g., "a few years ago" relative to 2023) is CORRECT. Converting "last year" to the actual year (e.g., "2022" when conversations are in 2023) is CORRECT.

5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches. For EMOTIONS and FEELINGS questions, answers expressing sentiments in the same valence (positive/negative) about the same event are CORRECT — do not require the exact same emotion word.

6. **SAME REFERENT**: If the generated answer mentions or references the same named entity, character, person, or concept as the gold answer, mark CORRECT — even if the generated answer provides a different physical description or includes additional details. The key question is: does the generated answer identify the same core entity? If yes, it is CORRECT.

7. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact. Minor differences in specificity, phrasing, or scope should not result in WRONG. Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer
- The answer addresses a completely different topic

## Question
Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels."""


def _judge_prompt(record: dict[str, Any], judge_profile: str = "legacy-mirror") -> str:
    answer = str(record.get("answer") or "")
    if str(record.get("question_type")) == "3" and ";" in answer:
        answer = answer.split(";", 1)[0].strip()
    template = (
        MEM0_CURRENT_EXACT_JUDGE_PROMPT
        if judge_profile == "mem0-current-exact"
        else MEM0_JUDGE_PROMPT
    )
    return template.format(
        question=record.get("question") or "",
        answer=answer,
        response=record.get("hypothesis") or "",
    )


def _track_records(run_dir: Path, track: str, protocol: str) -> list[dict[str, Any]]:
    source_path = run_dir / f"results.{track}.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"Result track not found: {source_path}")
    records = latest_by_question(read_jsonl(source_path))
    if protocol == "locomo-1540":
        records = [row for row in records if str(row.get("question_type")) != "5"]
    return records


def _summary(
    run_dir: Path,
    track: str,
    records: list[dict[str, Any]],
    *,
    protocol: str,
    judge_tag: str,
    model: str,
    judge_profile: str = "legacy-mirror",
) -> dict[str, Any]:
    output_path = run_dir / f"judge_results.{track}.{judge_tag}.jsonl"
    latest = {row["question_id"]: row for row in read_jsonl(output_path)}
    completed = [
        latest[row["question_id"]] for row in records
        if latest.get(row["question_id"], {}).get("status") == "completed"
    ]
    errors = sum(
        latest.get(row["question_id"], {}).get("status") == "error" for row in records
    )
    by_type = {}
    for question_type in sorted({str(row["question_type"]) for row in completed}):
        values = [row["score"] for row in completed if str(row["question_type"]) == question_type]
        by_type[question_type] = {"score": sum(values) / len(values), "count": len(values)}
    summary = {
        "dataset": "locomo",
        "track": track,
        "protocol": protocol,
        "judge_tag": judge_tag,
        "judge_model": model,
        "prompt_source": {
            "repository": "mem0ai/memory-benchmarks",
            "commit": MEM0_JUDGE_SOURCE_COMMIT,
            "mode": (
                "unified-no-evidence-exact"
                if judge_profile == "mem0-current-exact"
                else "unified-no-evidence"
            ),
        },
        "total": len(records),
        "completed": len(completed),
        "pending": len(records) - len(completed) - errors,
        "errors": errors,
        "overall": (sum(row["score"] for row in completed) / len(completed)) if completed else None,
        "by_type": by_type,
        "average_latency_seconds": (
            sum(float(row.get("latency_seconds") or 0) for row in completed) / len(completed)
            if completed and any(row.get("latency_seconds") is not None for row in completed) else None
        ),
    }
    write_json(run_dir / f"judge_summary.{track}.{judge_tag}.json", summary)
    return summary


def _comparison(run_dir: Path, tracks: list[str], summaries: dict[str, Any], judge_tag: str,
                protocol: str, model: str) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "protocol": protocol, "judge_model": model, "tracks": summaries,
    }
    if len(tracks) == 2 and all(summaries[name].get("overall") is not None for name in tracks):
        comparison["delta"] = summaries[tracks[1]]["overall"] - summaries[tracks[0]]["overall"]
    write_json(run_dir / f"judge_comparison.{judge_tag}.json", comparison)
    return comparison


def _api_key(llm: dict[str, Any]) -> str:
    value = str(llm.get("api_key") or "")
    env_name = str(llm.get("api_key_env") or "").strip()
    if not value and env_name:
        value = str(os.getenv(env_name) or "")
        if not value:
            raise RuntimeError(f"Required API key environment variable is not set: {env_name}")
    if not value:
        value = str(os.getenv("OPENAI_API_KEY") or "")
    if not value:
        raise RuntimeError("No judge API key configured")
    return value


def _parse_label(text: str) -> tuple[bool, str]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I)
    try:
        payload = json.loads(cleaned)
        label = str(payload.get("label") or "").upper()
        reasoning = str(payload.get("reasoning") or "")
    except (json.JSONDecodeError, AttributeError):
        match = re.search(r"\b(CORRECT|WRONG)\b", cleaned, flags=re.I)
        label = match.group(1).upper() if match else ""
        reasoning = cleaned
    if label not in {"CORRECT", "WRONG"}:
        raise ValueError(f"Judge returned no valid label: {text[:200]!r}")
    return label == "CORRECT", reasoning


def _judge_one(
    record: dict[str, Any],
    llm: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    judge_profile = str(llm.get("_judge_profile") or "legacy-mirror")
    prompt = _judge_prompt(record, judge_profile)
    extra_body = dict(llm.get("judge_extra_body") or llm.get("answer_extra_body") or {})
    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = openai_compatible_chat(
                [{"role": "system", "content": MEM0_JUDGE_SYSTEM},
                 {"role": "user", "content": prompt}],
                model=str(llm.get("judge_model") or llm.get("model") or "gpt-4o"),
                base_url=str(llm.get("judge_base_url") or llm.get("base_url") or "https://api.openai.com/v1"),
                api_key=api_key,
                timeout=int(llm.get("timeout_seconds") or 180),
                max_tokens=int(llm.get("judge_max_tokens") or 128),
                temperature=0,
                extra_body=extra_body,
            )
            correct, reasoning = _parse_label(response.content)
            return {
                "question_id": record["question_id"],
                "question_type": str(record.get("question_type") or ""),
                "label": "CORRECT" if correct else "WRONG",
                "score": float(correct),
                "reasoning": reasoning,
                "raw": response.content,
                "latency_seconds": round(time.monotonic() - started, 3),
                "status": "completed",
            }
        except Exception as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(min(2 ** attempt, 8))
    return {
        "question_id": record["question_id"],
        "question_type": str(record.get("question_type") or ""),
        "score": 0.0,
        "status": "error",
        "error": str(last_error),
        "latency_seconds": round(time.monotonic() - started, 3),
    }


def judge_run(
    run_dir: Path,
    config_path: Path,
    *,
    tracks: list[str],
    protocol: str = "locomo-1540",
    judge_tag: str = "qwen37",
    judge_profile: str = "legacy-mirror",
    max_workers: int = 4,
    resume: bool = True,
) -> dict[str, Any]:
    from core.server.config import load_config

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != "locomo":
        raise ValueError("The semantic judge command currently supports LoCoMo only")
    if protocol not in {"locomo-1540", "locomo-1986"}:
        raise ValueError("protocol must be locomo-1540 or locomo-1986")
    if judge_profile not in {"legacy-mirror", "mem0-current-exact"}:
        raise ValueError("judge_profile must be legacy-mirror or mem0-current-exact")
    config = load_config(str(config_path))
    llm = config.get("llm") or {}
    llm = dict(llm)
    llm["_judge_profile"] = judge_profile
    key = _api_key(llm)
    model = str(llm.get("judge_model") or llm.get("model") or "")
    summaries: dict[str, Any] = {}

    for track in tracks:
        records = _track_records(run_dir, track, protocol)
        output_path = run_dir / f"judge_results.{track}.{judge_tag}.jsonl"
        prior = {
            row["question_id"]: row for row in read_jsonl(output_path)
            if row.get("status") == "completed"
        } if resume else {}
        pending = [row for row in records if row["question_id"] not in prior]
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
            futures = {
                pool.submit(_judge_one, row, llm, key): row
                for row in pending
            }
            for future in as_completed(futures):
                judged = future.result()
                append_jsonl(output_path, judged)
                if judged.get("status") == "completed":
                    prior[judged["question_id"]] = judged

        summaries[track] = _summary(
            run_dir,
            track,
            records,
            protocol=protocol,
            judge_tag=judge_tag,
            model=model,
            judge_profile=judge_profile,
        )

    return _comparison(run_dir, tracks, summaries, judge_tag, protocol, model)


def batch_judge(
    run_dir: Path,
    config_path: Path,
    *,
    action: str,
    tracks: list[str],
    protocol: str = "locomo-1540",
    judge_tag: str = "qwen37-batch",
) -> dict[str, Any]:
    """Submit, inspect, or collect OpenAI-compatible Batch API judge jobs."""
    from openai import OpenAI
    from core.server.config import load_config

    if action not in {"submit", "status", "collect"}:
        raise ValueError("action must be submit, status, or collect")
    if protocol not in {"locomo-1540", "locomo-1986"}:
        raise ValueError("protocol must be locomo-1540 or locomo-1986")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != "locomo":
        raise ValueError("The semantic judge command currently supports LoCoMo only")
    llm = (load_config(str(config_path)).get("llm") or {})
    model = str(llm.get("judge_model") or llm.get("model") or "")
    client = OpenAI(
        api_key=_api_key(llm),
        base_url=str(llm.get("judge_base_url") or llm.get("base_url")),
        timeout=float(llm.get("timeout_seconds") or 180),
        max_retries=2,
    )
    state_path = run_dir / f"judge_batch.{judge_tag}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "schema_version": 1,
        "protocol": protocol,
        "judge_tag": judge_tag,
        "judge_model": model,
        "prompt_source": {"repository": "mem0ai/memory-benchmarks",
                          "commit": MEM0_JUDGE_SOURCE_COMMIT, "mode": "unified-no-evidence"},
        "tracks": {},
    }
    if state["protocol"] != protocol or state["judge_model"] != model:
        raise ValueError("Existing batch state uses a different protocol or judge model")

    for track_index, track in enumerate(tracks):
        records = _track_records(run_dir, track, protocol)
        entry = state["tracks"].get(track)
        if action == "submit" and not entry:
            completed = {
                row["question_id"] for row in read_jsonl(
                    run_dir / f"judge_results.{track}.{judge_tag}.jsonl"
                ) if row.get("status") == "completed"
            }
            pending = [row for row in records if row["question_id"] not in completed]
            input_path = run_dir / f"judge_batch_input.{track}.{judge_tag}.jsonl"
            mappings: dict[str, str] = {}
            with input_path.open("w", encoding="utf-8") as stream:
                for index, row in enumerate(pending):
                    custom_id = f"t{track_index}-q{index}"
                    mappings[custom_id] = str(row["question_id"])
                    body: dict[str, Any] = {
                        "model": model,
                        "temperature": 0,
                        "max_tokens": int(llm.get("judge_max_tokens") or 128),
                        "messages": [
                            {"role": "system", "content": MEM0_JUDGE_SYSTEM},
                            {"role": "user", "content": _judge_prompt(row)},
                        ],
                    }
                    body.update(dict(llm.get("judge_extra_body") or llm.get("answer_extra_body") or {}))
                    request = {"custom_id": custom_id, "method": "POST",
                               "url": "/v1/chat/completions", "body": body}
                    stream.write(json.dumps(request, ensure_ascii=False) + "\n")
            if not pending:
                state["tracks"][track] = {"status": "completed", "total": 0, "mappings": {}}
            else:
                with input_path.open("rb") as source:
                    uploaded = client.files.create(file=source, purpose="batch")
                batch = client.batches.create(
                    input_file_id=uploaded.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"benchmark": "locomo", "track": track, "judge_tag": judge_tag},
                )
                state["tracks"][track] = {
                    "batch_id": batch.id, "input_file_id": uploaded.id,
                    "input_path": str(input_path), "status": batch.status,
                    "total": len(pending), "mappings": mappings,
                }
                write_json(state_path, state)
        entry = state["tracks"].get(track)
        if not entry:
            continue
        if entry.get("batch_id"):
            batch = client.batches.retrieve(entry["batch_id"])
            for key in ("status", "output_file_id", "error_file_id", "created_at",
                        "completed_at", "failed_at", "expired_at"):
                value = getattr(batch, key, None)
                if value is not None:
                    entry[key] = value
            counts = getattr(batch, "request_counts", None)
            if counts is not None:
                entry["request_counts"] = counts.model_dump()
        if action == "collect" and entry.get("output_file_id") and not entry.get("collected"):
            content = client.files.content(entry["output_file_id"]).content.decode("utf-8")
            source_by_id = {str(row["question_id"]): row for row in records}
            output_path = run_dir / f"judge_results.{track}.{judge_tag}.jsonl"
            for line in content.splitlines():
                result = json.loads(line)
                question_id = entry["mappings"].get(result.get("custom_id"))
                source = source_by_id.get(str(question_id))
                response = result.get("response") or {}
                body = response.get("body") or {}
                if not source or int(response.get("status_code") or 0) != 200:
                    append_jsonl(output_path, {
                        "question_id": question_id, "question_type": str((source or {}).get("question_type") or ""),
                        "status": "error", "score": 0.0,
                        "error": result.get("error") or body,
                    })
                    continue
                raw = str((((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""))
                try:
                    correct, reasoning = _parse_label(raw)
                    append_jsonl(output_path, {
                        "question_id": question_id, "question_type": str(source.get("question_type") or ""),
                        "label": "CORRECT" if correct else "WRONG", "score": float(correct),
                        "reasoning": reasoning, "raw": raw, "usage": body.get("usage"),
                        "request_id": response.get("request_id"), "status": "completed",
                    })
                except ValueError as exc:
                    append_jsonl(output_path, {
                        "question_id": question_id, "question_type": str(source.get("question_type") or ""),
                        "status": "error", "score": 0.0, "raw": raw, "error": str(exc),
                    })
            entry["collected"] = True
            entry["collected_at"] = int(time.time())
        state["tracks"][track] = entry
        write_json(state_path, state)

    summaries = {
        track: _summary(run_dir, track, _track_records(run_dir, track, protocol),
                        protocol=protocol, judge_tag=judge_tag, model=model)
        for track in tracks
    }
    comparison = _comparison(run_dir, tracks, summaries, judge_tag, protocol, model)
    comparison["batch_state"] = str(state_path)
    comparison["batch_status"] = {
        track: {key: value for key, value in entry.items()
                if key not in {"mappings"}}
        for track, entry in state["tracks"].items()
    }
    return comparison
