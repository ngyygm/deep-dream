"""Full-context anchor track — same answerer, no memory system, whole visible corpus.

This adds the same-model full-context baseline that the paper's cross-system table
lacks: every question is answered by the configured LLM with *all* sessions visible
to that question concatenated into the prompt (dataset order), using the same answer
profile and JSON normalization as the replayed evidence tracks, so that only the
context source differs between the anchor and the memory paths.

Integrity rules:
- the corpus must fit the configured context window; oversize scopes are excluded
  up front and recorded in the manifest (``--strict-fit`` refuses to run instead),
  because a silently truncated "full context" would void the anchor claim;
- the full prompt is not embedded in per-question records (it is the whole corpus);
  a SHA-256 plus the contributing session IDs identify it instead;
- ``--answer-profile neutral-v1`` is the recommended anchor contract: it follows the
  question's own instructions instead of the conversational-QA heuristics, which
  structurally break pattern-continuation tasks (e.g. in-context-learning MCC).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

from .datasets import DATASETS, BenchmarkItem, sha256_file
from .reporting import append_jsonl, latest_by_question, read_jsonl, write_json
from .runner import (
    AnswerGenerator, _artifact_track, _base_record, _load_config,
    _public_config, _selected_items,
)

CHARS_PER_TOKEN = 4          # same approximation AnswerGenerator uses for its budget
CONTEXT_RESERVE_TOKENS = 1500


def _visible_contexts(item: BenchmarkItem) -> list[dict[str, Any]]:
    """Sessions visible to the question, dataset order, full text.

    Falls back to every session of the scope when the item carries no explicit
    visibility whitelist, mirroring what the memory system could have ingested.
    """
    visible = set(item.visible_session_ids or [])
    return [
        {"session_id": row.session_id, "timestamp": row.timestamp, "text": row.text}
        for row in item.sessions
        if not visible or row.session_id in visible
    ]


def _prompt_chars(contexts: list[dict[str, Any]]) -> int:
    return sum(
        len(f"Session {row['session_id']} ({row.get('timestamp') or 'unknown time'}):\n{row['text']}")
        for row in contexts
    )


def fullctx_evaluate_benchmark(
    run_dir: Path, config_path: Path, *, result_tag: str = "kimik3-v1",
    answer_profile: str = "normalized-v1", question_ids: Iterable[str] = (),
    scope_ids: Iterable[str] = (),
    limit: int | None = None, resume: bool = False, qa_workers: int = 2,
    strict_fit: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    config_path = config_path.resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = _load_config(config_path)
    # dataset_path may sit at the data root (flat files) or one level down
    # (memoryagentbench/manifest.json); walk up until the registry filename
    # resolves, so the data dir is derived instead of assumed.
    dataset_rel = DATASETS[manifest["dataset"]]["filename"]
    data_dir = Path(manifest["dataset_path"]).resolve().parent
    while data_dir != data_dir.parent and not (data_dir / dataset_rel).exists():
        data_dir = data_dir.parent
    items, dataset_path = _selected_items(
        manifest["dataset"], data_dir, question_ids=question_ids,
        scope_ids=scope_ids,
    )
    if sha256_file(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("Dataset hash changed; refusing full-context evaluation")

    context_window = int((config.get("llm") or {}).get("context_window_tokens") or 8000)
    max_chars = max(4000, (context_window - CONTEXT_RESERVE_TOKENS) * CHARS_PER_TOKEN)
    # build_prompt bounds only the evidence blocks, so the fixed instruction text
    # must be reserved here too — otherwise an item could pass pre-flight and still
    # overflow the real prompt.
    answerer = AnswerGenerator(config, profile=answer_profile, full_context=True)
    prompt_overhead = len(answerer.build_prompt(items[0], [])) if items else 0
    evidence_budget = max(0, max_chars - prompt_overhead)

    fitting: list[tuple[BenchmarkItem, list[dict[str, Any]], int]] = []
    excluded_scopes: dict[str, int] = {}
    for item in items:
        contexts = _visible_contexts(item)
        chars = _prompt_chars(contexts)
        if chars > evidence_budget:
            excluded_scopes[item.scope_id] = max(excluded_scopes.get(item.scope_id, 0), chars)
            continue
        fitting.append((item, contexts, chars))
    if strict_fit and excluded_scopes:
        detail = ", ".join(
            f"{scope}: {chars} chars > {evidence_budget}"
            for scope, chars in sorted(excluded_scopes.items())
        )
        raise ValueError(f"strict fit: oversize scopes refuse to run — {detail}")
    if limit is not None:
        fitting = fitting[:limit]

    output_track = _artifact_track("full-context", result_tag)
    results_path = run_dir / f"results.{output_track}.jsonl"
    completed = {
        row["question_id"] for row in latest_by_question(read_jsonl(results_path))
        if row.get("status") != "error"
    } if resume else set()
    pending = [entry for entry in fitting if entry[0].question_id not in completed]

    def _answer_one(item: BenchmarkItem, contexts: list[dict[str, Any]], chars: int) -> dict[str, Any]:
        base = _base_record(item, output_track)
        started = time.monotonic()
        try:
            answer = answerer.answer(item, contexts)
            prompt = answer.pop("prompt", "")
            latency = float(answer.get("answer_latency_seconds") or 0.0)
            return {
                **base, **answer,
                "status": "completed",
                "source_track": "full-context",
                "full_context": True,
                "context_session_ids": [row["session_id"] for row in contexts],
                "context_chars": chars,
                "context_window_tokens": context_window,
                "prompt_chars": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_omitted": (
                    "full-context prompt is the whole corpus; identified by dataset "
                    "sha256 plus context_session_ids instead of being embedded"
                ),
                "total_latency_seconds": round(latency, 3),
                "latency_seconds": round(latency, 3),
                "answer_wall_latency_seconds": round(time.monotonic() - started, 3),
            }
        except Exception as exc:
            return {
                **base,
                "status": "error", "hypothesis": "",
                "source_track": "full-context",
                "full_context": True,
                "context_session_ids": [row["session_id"] for row in contexts],
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "total_latency_seconds": round(time.monotonic() - started, 3),
            }

    processed = errors = 0
    with ThreadPoolExecutor(max_workers=max(1, qa_workers)) as pool:
        futures = {
            pool.submit(_answer_one, item, contexts, chars): item.question_id
            for item, contexts, chars in pending
        }
        for future in as_completed(futures):
            record = future.result()
            append_jsonl(results_path, record)
            errors += record.get("status") == "error"
            processed += 1

    manifest["tracks"] = list(dict.fromkeys([
        *(manifest.get("tracks") or []), output_track,
    ]))
    manifest.setdefault("track_variants", {})[output_track] = {
        "source_track": "full-context",
        "result_tag": result_tag,
        "answer_profile": answer_profile,
        "full_context": {
            "profile": "fullctx-v1",
            "context_window_tokens": context_window,
            "chars_per_token": CHARS_PER_TOKEN,
            "context_reserve_tokens": CONTEXT_RESERVE_TOKENS,
            "max_context_chars": max_chars,
            "evidence_char_budget": evidence_budget,
            "prompt_overhead_chars": prompt_overhead,
            "excluded_scopes": excluded_scopes,
            "session_visibility": (
                "per-question visible_session_ids; falls back to all scope sessions"
            ),
            "session_order": "dataset order",
            "memory_system_used": False,
        },
        "config": _public_config(config),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest["fullctx_completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    return {
        "run_dir": str(run_dir), "processed": processed, "errors": int(errors),
        "track": output_track, "excluded_scopes": excluded_scopes,
        "max_context_chars": max_chars, "evidence_char_budget": evidence_budget,
    }
