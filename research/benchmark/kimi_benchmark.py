"""Kimi-backed autonomous query stage for frozen Deep-Dream benchmarks."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

from research.benchmark import runtime_policy_metadata

from .datasets import BenchmarkItem, group_by_scope, sha256_file
from .kimi_runtime import (
    KIMI_RUNTIME_VERSION, KimiAgentRuntime, check_kimi_runtime, runtime_executable,
)
from .reporting import append_jsonl, latest_by_question, read_jsonl, write_json
from .runner import (
    _base_record, _cache_digest, _file_sha256, _load_config, _preliminary_failure,
    _public_config, _retrieval_metrics, _selected_items, _skill_metadata, replay_answers,
)


AGENT_MODES = ("direct", "evidence")


class _ManifestLock:
    """Serialize manifest read-merge-write across independent evaluator processes."""

    def __init__(self, path: Path):
        self.path = path.with_suffix(path.suffix + ".lock")
        self.stream = None

    def __enter__(self):
        import fcntl

        self.stream = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, traceback):
        import fcntl

        if self.stream is not None:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()


def _track_names(result_tag: str) -> tuple[str, str]:
    suffix = result_tag.strip()
    if suffix.startswith("kimi-"):
        suffix = suffix[len("kimi-"):]
    suffix = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in suffix).strip("-_")
    if not suffix:
        raise ValueError("result_tag must contain at least one letter or digit")
    return f"kimi-agent-direct-{suffix}", f"kimi-agent-evidence-{suffix}"


def _resolve_api_key(config: dict[str, Any]) -> tuple[str, str]:
    llm = config.get("llm") or {}
    key_env = str(llm.get("api_key_env") or "OPENAI_API_KEY")
    key = str(os.getenv(key_env) or llm.get("api_key") or "")
    if key:
        return key_env, key
    if sys_platform() == "darwin":
        result = subprocess.run(
            ["security", "find-generic-password", "-a", "deep-dream", "-s", key_env, "-w"],
            text=True, capture_output=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return key_env, result.stdout.strip()
    raise RuntimeError(f"Required API key is unavailable: {key_env}")


def sys_platform() -> str:
    import sys
    return sys.platform


def _contexts(item: BenchmarkItem, submitted: dict[str, list[str]], *, neighbor_turns: int = 1,
              context_k: int = 5) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    session_map = {row.session_id: row for row in item.sessions}
    turn_to_session = {
        turn_id: row.session_id for row in item.sessions for turn_id in row.turn_ids
    }
    selected_turns = list(dict.fromkeys(submitted.get("turn_ids") or []))
    selected_sessions = list(dict.fromkeys(submitted.get("session_ids") or []))
    for turn_id in selected_turns:
        session_id = turn_to_session.get(turn_id)
        if session_id and session_id not in selected_sessions:
            selected_sessions.append(session_id)
    contexts = []
    ranked_turns = []
    for rank, session_id in enumerate(selected_sessions[:context_k], 1):
        session = session_map.get(session_id)
        if not session:
            continue
        lines = session.text.splitlines()
        anchors = [turn for turn in selected_turns if turn in set(session.turn_ids)]
        indices: list[int] = []
        for anchor in anchors:
            index = session.turn_ids.index(anchor)
            indices.extend(range(
                max(0, index - neighbor_turns), min(len(lines), index + neighbor_turns + 1),
            ))
            if anchor not in ranked_turns:
                ranked_turns.append(anchor)
        if not indices:
            indices = list(range(min(len(lines), 20)))
        indices = list(dict.fromkeys(indices))
        contexts.append({
            "session_id": session_id,
            "timestamp": session.timestamp,
            "text": "\n".join(lines[index] for index in indices),
            "turn_ids": [session.turn_ids[index] for index in indices if index < len(session.turn_ids)],
            "matched_turn_ids": anchors,
            "score": 1.0 / rank,
            "evidence": [{
                "episode_id": episode_id,
                "session_id": session_id,
                "turn_ids": anchors,
                "retrieval_channel": "kimi-agent-submitted",
            } for episode_id in submitted.get("episode_ids") or []] or [{
                "retrieval_channel": "kimi-agent-submitted",
            }],
        })
    return contexts, [row["session_id"] for row in contexts], ranked_turns


def _query_one(
    *, runtime: KimiAgentRuntime, item: BenchmarkItem, direct_track: str,
    evidence_track: str, visible_sessions: set[str], trajectory_dir: Path,
    modes: set[str], context_k: int, neighbor_turns: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    last_error: Exception | None = None
    result = None
    for attempt in range(1, 4):
        try:
            result = runtime.run(
                scope_id=item.scope_id, question_id=item.question_id,
                question=item.question, question_date=item.question_date,
            )
            break
        except Exception as exc:
            last_error = exc
            if not bool(getattr(exc, "retryable", False)) or attempt == 3:
                break
            time.sleep(2 ** attempt)
    base_evidence = _base_record(item, evidence_track)
    base_direct = _base_record(item, direct_track)
    if result is None:
        error = {"type": type(last_error).__name__, "message": str(last_error)}
        common = {
            "status": "error", "hypothesis": "", "retrieved": [],
            "ranked_session_ids": [], "ranked_turn_ids": [], "retrieval_metrics": {},
            "failure_attribution": "agent_stopped_early", "error": error,
        }
        return (
            {**base_direct, **common} if "direct" in modes else None,
            {**base_evidence, **common} if "evidence" in modes else None,
        )
    submitted = {
        key: list(result.final.get(key) or [])
        for key in ("session_ids", "episode_ids", "turn_ids")
    }
    contexts, ranked_sessions, ranked_turns = _contexts(
        item, submitted, neighbor_turns=neighbor_turns, context_k=context_k,
    )
    safe_question = hashlib.sha256(item.question_id.encode()).hexdigest()[:20]
    trajectory_path = trajectory_dir / f"{safe_question}.json"
    write_json(trajectory_path, {
        "question_id": item.question_id,
        "scope_id": item.scope_id,
        "events": result.events,
        "tool_counts": result.tool_counts,
        "agent_steps": result.steps,
        "agent_stop_reason": str(result.final.get("stop_reason") or "final_answer"),
    })
    common = {
        "status": "completed",
        "source_track": "kimi-agent",
        "retrieved": contexts,
        "ranked_session_ids": ranked_sessions,
        "ranked_turn_ids": ranked_turns,
        "retrieval_metrics": _retrieval_metrics(item, ranked_sessions, ranked_turns),
        "retrieval_latency_seconds": result.latency_seconds,
        "retrieval_prompt_tokens": result.prompt_tokens,
        "retrieval_completion_tokens": result.completion_tokens,
        "retrieval_model": runtime.model,
        "retrieval_profile": (
            "kimi-agent-v2-persistent"
            if runtime.lifecycle == "persistent" else "kimi-agent-v1"
        ),
        "submitted_evidence": submitted,
        "agent_steps": result.steps,
        "agent_stop_reason": str(result.final.get("stop_reason") or "final_answer"),
        "agent_tool_counts": result.tool_counts,
        "trajectory_path": str(trajectory_path.relative_to(runtime.run_dir)),
        "trajectory_sha256": _file_sha256(trajectory_path),
    }
    common["failure_attribution"] = _preliminary_failure(
        {**base_evidence, **common}, visible_sessions,
    )
    evidence = None
    if "evidence" in modes:
        evidence = {**base_evidence, **copy.deepcopy(common)}
        evidence["retrieval_cache_schema"] = 3
        evidence["retrieval_cache_sha256"] = _cache_digest(evidence)
    direct = None
    if "direct" in modes:
        direct = {
            **base_direct, **copy.deepcopy(common),
            "hypothesis": result.final["answer"],
            "prompt": "Kimi CLI Deep-Dream agent prompt (fingerprinted in manifest)",
            "answer_latency_seconds": 0.0,
            "answer_profile": "kimi-agent-direct-v1",
            "answer_payload": result.final,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_latency_seconds": result.latency_seconds,
            "latency_seconds": result.latency_seconds,
            "model": runtime.model,
        }
    return direct, evidence


def agent_query_benchmark(
    run_dir: Path, config_path: Path, *, runtime_name: str = "kimi",
    modes: Iterable[str] = AGENT_MODES, agent_model: str = "qwen3.7-plus",
    agent_thinking: bool = True, max_agent_steps: int = 12, qa_workers: int = 2,
    result_tag: str = "kimi-qwen37-thinking-on", question_ids: Iterable[str] = (),
    limit: int | None = None, eligible_evidence_only: bool = False,
    resume: bool = False, runtime_version: str = KIMI_RUNTIME_VERSION,
    runtime_root: Path | None = None, executable: Path | None = None,
    timeout_seconds: int = 600, context_k: int = 5, neighbor_turns: int = 1,
    runtime_lifecycle: str = "per-question", agent_step_policy: str = "autonomous",
) -> dict[str, Any]:
    if runtime_name != "kimi":
        raise ValueError("Only the pinned Kimi runtime is supported")
    selected_modes = set(modes)
    if not selected_modes or selected_modes - set(AGENT_MODES):
        raise ValueError(f"modes must be drawn from {AGENT_MODES}")
    run_dir = run_dir.resolve()
    config_path = config_path.resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = _load_config(config_path)
    key_env, api_key = _resolve_api_key(config)
    binary = executable or runtime_executable(runtime_version, runtime_root)
    runtime_info = check_kimi_runtime(
        version=runtime_version, root=runtime_root, executable=binary,
    )
    items, dataset_path = _selected_items(
        manifest["dataset"], Path(manifest["dataset_path"]).parent,
        question_ids=question_ids,
    )
    if sha256_file(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("Dataset hash changed; refusing Agent evaluation")
    selected: list[BenchmarkItem] = []
    for scope_id, scope_items in group_by_scope(items).items():
        scope_info = (manifest.get("scopes") or {}).get(scope_id)
        if not scope_info:
            continue
        visible = set(scope_info.get("visible_sessions") or [])
        selected.extend(
            item for item in scope_items
            if not eligible_evidence_only or (
                bool(item.evidence_session_ids) and set(item.evidence_session_ids).issubset(visible)
            )
        )
    if limit is not None:
        selected = selected[:limit]
    direct_track, evidence_track = _track_names(result_tag)
    direct_path = run_dir / f"results.{direct_track}.jsonl"
    evidence_path = run_dir / f"retrieval.{evidence_track}.jsonl"
    direct_done = {
        row["question_id"] for row in latest_by_question(read_jsonl(direct_path))
        if row.get("status") != "error"
    } if resume and "direct" in selected_modes else set()
    evidence_done = {
        row["question_id"] for row in latest_by_question(read_jsonl(evidence_path))
        if row.get("status") != "error"
    } if resume and "evidence" in selected_modes else set()
    pending = [item for item in selected if not (
        ("direct" not in selected_modes or item.question_id in direct_done)
        and ("evidence" not in selected_modes or item.question_id in evidence_done)
    )]
    runtime = KimiAgentRuntime(
        executable=binary, run_dir=run_dir, config_path=config_path,
        model=agent_model, thinking=agent_thinking, max_steps=max_agent_steps,
        timeout_seconds=timeout_seconds,
        context_window=int((config.get("llm") or {}).get("context_window_tokens") or 32000),
        api_key=api_key, lifecycle=runtime_lifecycle,
        agent_step_policy=agent_step_policy,
    )
    trajectory_dir = run_dir / "trajectories" / result_tag
    processed = errors = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, qa_workers)) as pool:
            futures = {}
            for item in pending:
                visible = set((manifest.get("scopes") or {}).get(item.scope_id, {}).get(
                    "visible_sessions", []
                ))
                item_modes = set(selected_modes)
                if item.question_id in direct_done:
                    item_modes.discard("direct")
                if item.question_id in evidence_done:
                    item_modes.discard("evidence")
                future = pool.submit(
                    _query_one, runtime=runtime, item=item, direct_track=direct_track,
                    evidence_track=evidence_track, visible_sessions=visible,
                    trajectory_dir=trajectory_dir, modes=item_modes,
                    context_k=context_k, neighbor_turns=neighbor_turns,
                )
                futures[future] = item.question_id
            for future in as_completed(futures):
                direct, evidence = future.result()
                if direct is not None:
                    append_jsonl(direct_path, direct)
                    errors += direct.get("status") == "error"
                if evidence is not None:
                    append_jsonl(evidence_path, evidence)
                    errors += evidence.get("status") == "error"
                processed += 1
    finally:
        runtime.close()

    agent_config = {
        "runtime": runtime_info,
        "model": agent_model,
        "thinking": agent_thinking,
        # Kimi 1.49.0's OpenAI-compatible provider does not expose a temperature
        # override. Record the effective setting honestly instead of claiming the
        # answerer's temperature=0 was also applied to the Agent.
        "temperature": "provider-default",
        "max_agent_steps": max_agent_steps,
        "qa_workers": qa_workers,
        "runtime_lifecycle": runtime_lifecycle,
        "agent_step_policy": agent_step_policy,
        "fresh_context_per_question": True,
        "context_k": context_k,
        "neighbor_turns": neighbor_turns,
        "runtime_policy": runtime_policy_metadata(),
        "skill": _skill_metadata(),
        "mcp_schema_sha256": _file_sha256(Path(__file__).with_name("mcp_server.py")),
        "adapter_sha256": _file_sha256(Path(__file__).with_name("kimi_runtime.py")),
        "api_key_env": key_env,
        "config": _public_config(config),
    }
    with _ManifestLock(manifest_path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = max(5, int(manifest.get("schema_version") or 1))
        tracks = manifest.setdefault("tracks", [])
        if "direct" in selected_modes and direct_track not in tracks:
            tracks.append(direct_track)
        if "evidence" in selected_modes and evidence_track not in tracks:
            tracks.append(evidence_track)
        manifest.setdefault("agent_runtimes", {})[result_tag] = agent_config
        variants = manifest.setdefault("track_variants", {})
        if "direct" in selected_modes:
            variants[direct_track] = {
                "source_track": "kimi-agent-direct", "result_tag": result_tag,
                "answer_profile": "kimi-agent-direct-v1", "agent": agent_config,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        if "evidence" in selected_modes:
            variants[evidence_track] = {
                "source_track": "kimi-agent-evidence", "result_tag": result_tag,
                "retrieval_only": True, "retrieval_reused": False,
                "retrieval": {
                    "profile": (
                        "kimi-agent-v2-persistent"
                        if runtime_lifecycle == "persistent" else "kimi-agent-v1"
                    ),
                    "context_k": context_k,
                },
                "agent": agent_config, "created_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest.setdefault("retrieval_profiles", {})[evidence_track] = {
                "profile": "kimi-agent-v1", "context_k": context_k,
                "neighbor_turns": neighbor_turns,
            }
        manifest["agent_query_completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
    return {
        "run_dir": str(run_dir), "processed": processed, "errors": int(errors),
        "tracks": [track for mode, track in (
            ("direct", direct_track), ("evidence", evidence_track)
        ) if mode in selected_modes],
        "direct_track": direct_track if "direct" in selected_modes else None,
        "evidence_track": evidence_track if "evidence" in selected_modes else None,
        "runtime": runtime_info,
    }


def agent_evaluate_benchmark(
    run_dir: Path, config_path: Path, *, answer_result_tag: str = "qwen37-answer-v1",
    answer_profile: str = "normalized-v1", **query_kwargs: Any,
) -> dict[str, Any]:
    query = agent_query_benchmark(run_dir, config_path, **query_kwargs)
    evidence_track = query.get("evidence_track")
    answer = None
    if evidence_track:
        answer = replay_answers(
            run_dir, config_path, tracks=[evidence_track], result_tag=answer_result_tag,
            answer_profile=answer_profile,
            question_ids=query_kwargs.get("question_ids") or (),
            resume=bool(query_kwargs.get("resume")),
        )
    return {"query": query, "answer": answer}
