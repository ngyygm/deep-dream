"""Isolated ingestion and dual-track LoCoMo/LongMemEval evaluation."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable

from research.benchmark import runtime_policy_metadata

from .datasets import BenchmarkItem, MemorySession, data_dir_for_dataset_path, group_by_scope, load_benchmark, parse_timestamp, sha256_file
from .metrics import retrieval_at_k
from .reporting import append_jsonl, latest_by_question, read_jsonl, write_json
from .retrieval import UnifiedRetriever


RETRIEVAL_KS = (1, 3, 5, 10, 30, 50)
TRACKS = ("baseline", "skill-agent")
DeepDreamRetriever = UnifiedRetriever  # public compatibility alias


def _safe_id(value: str) -> str:
    readable = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)[:48]
    return f"{readable}_{hashlib.sha256(value.encode()).hexdigest()[:10]}"


def _document_id(dataset: str, scope_id: str, session_id: str) -> str:
    value = f"{dataset}\0{scope_id}\0{session_id}"
    return "doc_bench_" + hashlib.sha256(value.encode()).hexdigest()[:20]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def _skill_metadata() -> dict[str, str]:
    path = Path(__file__).resolve().parents[2] / ".claude" / "skills" / "deep-dream" / "SKILL.md"
    return {"path": str(path), "sha256": _file_sha256(path)}


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    llm = config.get("llm") or {}
    embedding = config.get("embedding") or {}
    public_llm = {key: llm.get(key) for key in (
            "model", "base_url", "think", "extra_body", "temperature", "max_tokens",
            "agent_max_tokens", "agent_thinking_max_tokens", "context_window_tokens",
            "agent_think", "answer_think",
            "answer_validation_retries", "api_key_env",
        )}
    public_llm["temperature"] = float(llm.get("temperature", 0) or 0)
    public_llm["think"] = bool(llm.get("think", False))
    return {
        "llm": public_llm,
        "embedding": {key: embedding.get(key) for key in ("model", "model_path", "device")},
        "chunking": config.get("chunking") or {},
        "pipeline": config.get("pipeline") or {},
    }


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}. Copy service_config.example.json and configure the LLM first."
        )
    from core.server.config import load_config
    config = load_config(str(config_path))
    llm = config.get("llm") or {}
    key_env = str(llm.get("api_key_env") or "").strip()
    if key_env and not llm.get("api_key") and not os.getenv(key_env) and sys.platform == "darwin":
        result = subprocess.run(
            ["security", "find-generic-password", "-a", "deep-dream", "-s", key_env, "-w"],
            text=True, capture_output=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            config.setdefault("llm", {})["api_key"] = result.stdout.strip()
    return config


def _quality_config(config: dict[str, Any], profile: str) -> dict[str, Any]:
    scoped = copy.deepcopy(config)
    remember = scoped.setdefault("pipeline", {}).setdefault("remember", {})
    remember["profile"] = profile
    if profile == "strong-v1":
        remember.update({
            "window_size_chars": 6000,
            "overlap_chars": 300,
            "max_entities_per_window": 24,
            "max_relations_per_window": 36,
            # 大窗口 episode 在 BM25 长度归一化下吃亏：追加 ~800 字薄检索切片行，
            # 窗口 episode（实体锚定/版本链/溯源）原样保留
            "episode_slice_chars": 800,
        })
    return scoped


def create_manifest(
    dataset: str,
    dataset_path: Path,
    config: dict[str, Any],
    run_dir: Path,
    *,
    answer_top_k: int = 5,
    limit: int | None = None,
    question_ids: list[str] | None = None,
    retrieval_mode: str | None = None,
    max_agent_steps: int = 8,
    remember_profile: str = "strong-v1",
) -> dict[str, Any]:
    """Create schema-v3 manifest; old arguments remain source compatible."""
    return {
        "schema_version": 4,
        "dataset": dataset,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "answer_top_k": answer_top_k,
        "max_agent_steps": max_agent_steps,
        "retrieval_k": list(RETRIEVAL_KS),
        "limit": limit,
        "question_ids": list(question_ids or []),
        "random_seed": 0,
        "remember_profile": remember_profile,
        "runtime_policy": runtime_policy_metadata(),
        "skill": _skill_metadata(),
        "config": _public_config(config),
        "formal_defaults": {
            "model": "qwen3.6-27b-awq",
            "thinking": False,
            "temperature": 0,
            "answer_top_k": answer_top_k,
            "max_agent_steps": max_agent_steps,
        },
        "scopes": {},
        "tracks": [],
        "retrieval_profiles": {},
        **({"retrieval_mode": retrieval_mode} if retrieval_mode else {}),
    }


def _selected_items(
    dataset: str,
    data_dir: Path,
    *,
    limit: int | None = None,
    question_ids: Iterable[str] = (),
    scope_ids: Iterable[str] = (),
) -> tuple[list[BenchmarkItem], Path]:
    items, dataset_path = load_benchmark(dataset, data_dir)
    wanted_questions = set(question_ids)
    wanted_scopes = set(scope_ids)
    if wanted_questions:
        items = [item for item in items if item.question_id in wanted_questions]
    if wanted_scopes:
        items = [item for item in items if item.scope_id in wanted_scopes]
    if limit is not None:
        items = items[:limit]
    if not items:
        raise ValueError("No benchmark items selected")
    return items, dataset_path


def _ensure_ingestion_state(storage: Any) -> None:
    if not hasattr(storage, "_conn"):
        if not hasattr(storage, "_benchmark_ingestion_state"):
            storage._benchmark_ingestion_state = {}
        return
    storage._conn().execute(
        """CREATE TABLE IF NOT EXISTS document_ingestion_state (
               document_id TEXT PRIMARY KEY,
               state TEXT NOT NULL CHECK(state IN ('processing','active','failed','incomplete')),
               total_windows INTEGER NOT NULL DEFAULT 0,
               complete_windows INTEGER NOT NULL DEFAULT 0,
               missing_windows TEXT NOT NULL DEFAULT '[]',
               updated_at TEXT NOT NULL
           )"""
    )
    storage._conn().commit()


def _set_ingestion_state(
    storage: Any,
    document_id: str,
    state: str,
    *,
    total_windows: int = 0,
    complete_windows: int = 0,
    missing_windows: Iterable[int] = (),
) -> None:
    _ensure_ingestion_state(storage)
    if not hasattr(storage, "_conn"):
        storage._benchmark_ingestion_state[document_id] = {
            "state": state, "total_windows": total_windows,
            "complete_windows": complete_windows,
        }
        return
    storage._conn().execute(
        """INSERT INTO document_ingestion_state
           (document_id, state, total_windows, complete_windows, missing_windows, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(document_id) DO UPDATE SET
             state=excluded.state, total_windows=excluded.total_windows,
             complete_windows=excluded.complete_windows, missing_windows=excluded.missing_windows,
             updated_at=excluded.updated_at""",
        (
            document_id, state, int(total_windows), int(complete_windows),
            json.dumps(sorted(set(int(value) for value in missing_windows))),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    storage._conn().commit()


def _duplicate_family_report(storage: Any) -> dict:
    """同名实体分布到多个 family 的统计（对齐质量/并发正确性指标）。

    duplicate_family_rate = 同名多 family 的 family 行数 / 实体 family 总数。
    """
    if not hasattr(storage, "_conn"):
        return {"error": "storage has no _conn"}
    try:
        conn = storage._conn()
        rows = conn.execute(
            "SELECT LOWER(TRIM(canonical_name)) AS n, COUNT(*) AS fids "
            "FROM entity_families WHERE TRIM(canonical_name) <> '' "
            "GROUP BY n HAVING fids > 1"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM entity_families").fetchone()[0]
        dup_rows = sum(int(r[1]) for r in rows)
        return {
            "total_entity_families": int(total),
            "duplicate_names": len(rows),
            "duplicate_family_rows": dup_rows,
            "duplicate_family_rate": round(dup_rows / total, 4) if total else 0.0,
        }
    except Exception as exc:  # noqa: BLE001 - 指标采集失败不应中断 ingest
        return {"error": f"duplicate family query failed: {exc}"}


def _aggregate_llm_usage(manifest: dict) -> dict:
    """把 manifest 中各文档的 llm_call_stats 汇总为一份全局用量。"""
    agg = {
        "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "estimated_calls": 0, "by_step": {}, "document_count": 0,
    }
    for scope in (manifest.get("scopes") or {}).values():
        for doc in (scope.get("documents") or {}).values():
            stats = doc.get("llm_call_stats") or {}
            if not stats:
                continue
            agg["document_count"] += 1
            agg["calls"] += int(stats.get("calls") or 0)
            agg["prompt_tokens"] += int(stats.get("prompt_tokens") or 0)
            agg["completion_tokens"] += int(stats.get("completion_tokens") or 0)
            agg["estimated_calls"] += int(stats.get("estimated_calls") or 0)
            for step, bucket in (stats.get("by_step") or {}).items():
                target = agg["by_step"].setdefault(
                    step, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
                )
                target["calls"] += int(bucket.get("calls") or 0)
                target["prompt_tokens"] += int(bucket.get("prompt_tokens") or 0)
                target["completion_tokens"] += int(bucket.get("completion_tokens") or 0)
    return agg


def _database_complete(storage: Any, document_id: str) -> bool:
    _ensure_ingestion_state(storage)
    if not hasattr(storage, "_conn"):
        row = storage._benchmark_ingestion_state.get(document_id, {})
        return row.get("state") == "active" and row.get("total_windows") == row.get("complete_windows")
    row = storage._conn().execute(
        """SELECT s.state, s.total_windows, s.complete_windows, d.status,
                  COUNT(ep.episode_id)
           FROM document_ingestion_state s
           LEFT JOIN documents d ON d.document_id = s.document_id
           LEFT JOIN episodes ep ON ep.document_id = s.document_id AND ep.status = 'active'
           WHERE s.document_id = ? GROUP BY s.document_id""",
        (document_id,),
    ).fetchone()
    if not row:
        return False
    total, complete = int(row[1] or 0), int(row[2] or 0)
    return row[0] == "active" and row[3] == "active" and int(row[4] or 0) > 0 and total > 0 and total == complete


def _reset_incomplete_document(storage: Any, document_id: str) -> None:
    """Remove a partial isolated-benchmark document before stable-ID retry."""
    if not hasattr(storage, "_conn"):
        storage._benchmark_ingestion_state.pop(document_id, None)
        return
    row = storage._conn().execute(
        "SELECT current_version_id FROM documents WHERE document_id = ?", (document_id,)
    ).fetchone()
    if not row or not row[0]:
        return
    version_id = str(row[0])
    storage.delete_document_version(version_id)
    # delete_document_version deliberately leaves tombstones for normal user
    # history. Benchmark retry is an isolated library and needs the same stable
    # version/document identity to be insertable again.
    storage._conn().execute(
        "DELETE FROM document_versions WHERE document_version_id = ?", (version_id,)
    )
    storage._conn().execute(
        "UPDATE documents SET status = 'active', current_version_id = NULL WHERE document_id = ?",
        (document_id,),
    )
    storage._conn().commit()


def _write_raw_session(run_dir: Path, scope_id: str, session: MemorySession) -> Path:
    from core.storage.sqlite.content_fs import _atomic_write
    path = run_dir / "source_sessions" / _safe_id(scope_id) / f"{_safe_id(session.session_id)}.md"
    _atomic_write(str(path), session.text)
    return path


def _ingest_scope_concurrent(
    registry: Any,
    main_processor: Any,
    sessions: list[Any],
    dataset: str,
    scope_id: str,
    scope_dir: Path,
    scope_manifest: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    resume: bool,
    workers: int,
) -> tuple[int, int]:
    """文档级并发 ingest：每文档独立 processor（共享库级信号量/JudgeService/embedding）。

    manifest/ingestion_state 的写入全部收在主线程（as_completed 循环），
    worker 线程只跑 remember_text。任一文档失败：记录 failed 状态后取消剩余任务并抛出
    （与串行路径的 fail-fast 语义一致）。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pending: list[tuple[Any, str, int]] = []  # (session, document_id, total_windows)
    skipped_here = 0
    for session in sessions:
        document_id = _document_id(dataset, scope_id, session.session_id)
        if resume and _database_complete(main_processor.storage, document_id):
            if session.session_id not in scope_manifest["visible_sessions"]:
                scope_manifest["visible_sessions"].append(session.session_id)
            skipped_here += 1
            continue
        if resume:
            _reset_incomplete_document(main_processor.storage, document_id)
        total_windows = len(main_processor.document_processor.chunk_text(session.text)) \
            if hasattr(main_processor, "document_processor") else 1
        pending.append((session, document_id, total_windows))
    if not pending:
        return 0, skipped_here

    def _run_one(session: Any, document_id: str) -> dict[str, Any]:
        proc = registry.create_task_processor("library")
        source_path = _write_raw_session(scope_dir.parent.parent, scope_id, session)
        try:
            return {"result": proc.remember_text(
                session.text,
                doc_name=session.session_id,
                source_document=session.session_id,
                event_time=parse_timestamp(session.timestamp),
                document_path=str(source_path),
                override_doc_id=document_id,
                verbose=False,
                verbose_steps=False,
            )}
        finally:
            try:
                proc.storage.close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响结果
                pass

    ingested_here = 0
    first_error: BaseException | None = None
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bm-ingest") as pool:
        futures = {}
        for session, document_id, total_windows in pending:
            _set_ingestion_state(
                main_processor.storage, document_id, "processing", total_windows=total_windows)
            futures[pool.submit(_run_one, session, document_id)] = (
                session, document_id, total_windows)
        try:
            for future in as_completed(futures):
                session, document_id, total_windows = futures[future]
                result = future.result()["result"]  # 失败直接抛到 except 分支
                failed_indices = list(result.get("failed_window_indices") or [])
                complete = total_windows - len(failed_indices)
                state = "active" if complete == total_windows and result.get("episode_id") else "incomplete"
                _set_ingestion_state(
                    main_processor.storage, document_id, state,
                    total_windows=total_windows, complete_windows=complete,
                    missing_windows=failed_indices,
                )
                if state != "active":
                    raise RuntimeError(f"Incomplete remember result: {result}")
                scope_manifest["documents"][session.session_id] = {
                    "document_id": document_id,
                    "status": state,
                    "total_windows": total_windows,
                    "complete_windows": complete,
                    "missing_windows": failed_indices,
                    "entities": int(result.get("entities") or 0),
                    "relations": int(result.get("relations") or 0),
                    "latency_seconds": 0.0,
                    "llm_call_stats": result.get("llm_call_stats") or {},
                }
                if session.session_id not in scope_manifest["visible_sessions"]:
                    scope_manifest["visible_sessions"].append(session.session_id)
                ingested_here += 1
                write_json(manifest_path, manifest)
        except BaseException as exc:
            first_error = exc
            for future, (session, document_id, total_windows) in futures.items():
                if future.done() and session.session_id in scope_manifest.get("documents", {}):
                    continue
                try:
                    _set_ingestion_state(
                        main_processor.storage, document_id, "failed",
                        total_windows=total_windows, complete_windows=0,
                        missing_windows=range(total_windows))
                except Exception:  # noqa: BLE001
                    pass
            write_json(manifest_path, manifest)
            pool.shutdown(wait=True, cancel_futures=True)
    if first_error is not None:
        raise first_error
    return ingested_here, skipped_here


def ingest_benchmark(
    dataset: str,
    data_dir: Path,
    run_dir: Path,
    config_path: Path,
    *,
    scope_ids: list[str] | None = None,
    session_limit: int | None = None,
    remember_profile: str = "strong-v1",
    resume: bool = False,
    ingest_workers: int = 1,
) -> dict[str, Any]:
    if remember_profile != "strong-v1":
        raise ValueError("remember_profile must be strong-v1")
    items, dataset_path = _selected_items(dataset, data_dir, scope_ids=scope_ids or [])
    config = _quality_config(_load_config(config_path), remember_profile)
    if run_dir.exists() and not resume and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_sha256") != sha256_file(dataset_path):
            raise ValueError("Dataset hash changed; refusing to reuse this run")
        if manifest.get("remember_profile", "strong-v1") != remember_profile:
            raise ValueError("Remember profile changed; use a new run directory")
    else:
        manifest = create_manifest(
            dataset, dataset_path, config, run_dir, remember_profile=remember_profile,
        )
        write_json(manifest_path, manifest)

    from core.server.registry import GraphRegistry
    ingested = skipped = failed = 0
    for scope_id, scope_items in group_by_scope(items).items():
        all_sessions = scope_items[0].sessions
        sessions = all_sessions[:session_limit] if session_limit else all_sessions
        scope_dir = run_dir / "libraries" / _safe_id(scope_id)
        scope_config = copy.deepcopy(config)
        scope_config["storage_path"] = str(scope_dir)
        registry = GraphRegistry(str(scope_dir), scope_config)
        processor = registry.get_processor("library")
        scope_manifest = manifest.setdefault("scopes", {}).setdefault(scope_id, {
            "library_dir": str(scope_dir.relative_to(run_dir)),
            "visible_sessions": [],
            "documents": {},
        })
        try:
            if int(ingest_workers) > 1:
                _ing, _skip = _ingest_scope_concurrent(
                    registry, processor, sessions, dataset, scope_id, scope_dir,
                    scope_manifest, manifest_path, manifest,
                    resume=resume, workers=int(ingest_workers),
                )
                ingested += _ing
                skipped += _skip
                # ALIGN-V2：scope 收尾串行收敛（ingest 期间只收集等价组不合并，
                # 避免与并行文档写入竞争 FK/锁）。收敛成功后落 manifest 标记；
                # resume 重启时凭标记跳过——否则每次重启都把已有 scope 的收敛
                # 全量重跑（重启税 ∝ scope 数：16 个 scope 一次 ~2.8h 纯重复）。
                if not scope_manifest.get("converged"):
                    try:
                        from core.remember.align_v2 import align_v2_enabled, final_convergence_flush
                        if align_v2_enabled():
                            _flush = final_convergence_flush(processor, verbose=True)
                            print(f"[align-v2] scope={scope_id} 收尾收敛完成: {_flush}", flush=True)
                            scope_manifest["converged"] = True
                            write_json(manifest_path, manifest)
                    except Exception as _exc:
                        print(f"[align-v2] scope={scope_id} 收尾收敛失败(不阻断): {_exc}", flush=True)
                continue
            for session in sessions:
                document_id = _document_id(dataset, scope_id, session.session_id)
                if resume and _database_complete(processor.storage, document_id):
                    if session.session_id not in scope_manifest["visible_sessions"]:
                        scope_manifest["visible_sessions"].append(session.session_id)
                    skipped += 1
                    continue
                if resume:
                    _reset_incomplete_document(processor.storage, document_id)
                total_windows = len(processor.document_processor.chunk_text(session.text)) \
                    if hasattr(processor, "document_processor") else 1
                _set_ingestion_state(
                    processor.storage, document_id, "processing", total_windows=total_windows,
                )
                source_path = _write_raw_session(run_dir, scope_id, session)
                started = time.monotonic()
                try:
                    result = processor.remember_text(
                        session.text,
                        doc_name=session.session_id,
                        source_document=session.session_id,
                        event_time=parse_timestamp(session.timestamp),
                        document_path=str(source_path),
                        override_doc_id=document_id,
                        verbose=False,
                        verbose_steps=False,
                    )
                    failed_indices = list(result.get("failed_window_indices") or [])
                    complete = total_windows - len(failed_indices)
                    state = "active" if complete == total_windows and result.get("episode_id") else "incomplete"
                    _set_ingestion_state(
                        processor.storage, document_id, state,
                        total_windows=total_windows, complete_windows=complete,
                        missing_windows=failed_indices,
                    )
                    if state != "active":
                        raise RuntimeError(f"Incomplete remember result: {result}")
                    scope_manifest["documents"][session.session_id] = {
                        "document_id": document_id,
                        "status": state,
                        "total_windows": total_windows,
                        "complete_windows": complete,
                        "missing_windows": failed_indices,
                        "entities": int(result.get("entities") or 0),
                        "relations": int(result.get("relations") or 0),
                        "latency_seconds": round(time.monotonic() - started, 3),
                        "llm_call_stats": result.get("llm_call_stats") or {},
                    }
                    if session.session_id not in scope_manifest["visible_sessions"]:
                        scope_manifest["visible_sessions"].append(session.session_id)
                    ingested += 1
                except BaseException:
                    _set_ingestion_state(
                        processor.storage, document_id, "failed",
                        total_windows=total_windows, complete_windows=0,
                        missing_windows=range(total_windows),
                    )
                    failed += 1
                    write_json(manifest_path, manifest)
                    raise
                write_json(manifest_path, manifest)
        finally:
            try:
                scope_manifest["duplicate_family_report"] = _duplicate_family_report(processor.storage)
            except Exception:  # noqa: BLE001 - 指标采集失败不应中断 ingest
                scope_manifest["duplicate_family_report"] = {"error": "unavailable"}
            processor.storage.close()
    manifest["ingestion_completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["library_integrity"] = {
        "visible_session_count": sum(len(row.get("visible_sessions", [])) for row in manifest["scopes"].values()),
        "failed_documents": failed,
    }
    manifest["llm_usage"] = _aggregate_llm_usage(manifest)
    write_json(manifest_path, manifest)
    return {"run_dir": str(run_dir), "ingested": ingested, "skipped": skipped, "failed": failed}


class AnswerGenerator:
    """Track-independent answer model; no benchmark category enters the prompt."""

    PROFILES = {"legacy", "normalized-v1"}
    SUPPORT_VALUES = {"supported", "unsupported", "false_premise"}
    ANSWER_TYPES = {"boolean", "date", "duration", "list", "span", "likely"}

    def __init__(self, config: dict[str, Any], *, profile: str = "normalized-v1"):
        if profile not in self.PROFILES:
            raise ValueError(f"Unknown answer profile: {profile}")
        self.config = config.get("llm") or {}
        self.profile = profile

    @staticmethod
    def _context_blocks(contexts: list[dict[str, Any]], max_chars: int) -> list[str]:
        blocks, used = [], 0
        for row in contexts:
            block = f"Session {row['session_id']} ({row.get('timestamp') or 'unknown time'}):\n{row['text']}"
            if used + len(block) > max_chars:
                if max_chars - used > 200:
                    blocks.append(block[:max_chars - used])
                break
            blocks.append(block)
            used += len(block)
        return blocks

    def build_prompt(self, item: BenchmarkItem, contexts: list[dict[str, Any]]) -> str:
        context_window = int(self.config.get("context_window_tokens") or 8000)
        max_chars = max(4000, min(120000, (context_window - 1500) * 4))
        blocks = self._context_blocks(contexts, max_chars)
        date_line = f"Question date: {item.question_date}\n" if item.question_date else ""
        if self.profile == "normalized-v1":
            return (
                "Answer using only the submitted conversation evidence. Do not use a benchmark label or a hidden "
                "reference answer. First decide whether the exact proposition about the named person is directly "
                "supported. If the evidence only attributes the event or object to a different person, classify it "
                "as false_premise instead of answering No; this is how false premises are rejected. If the fact is "
                "absent, classify it as unsupported; unsupported and false_premise results are serialized as "
                "'No information available.'. "
                "Resolve indirect references across sessions, respect later updates, and convert relative time using "
                "the timestamp of the session containing the event; resolve relative dates before returning. For "
                "'last week', use 'The week before <session "
                "date>'; for 'yesterday', output the absolute calendar date. Collect every distinct requested field "
                "and preserve concise wording from the evidence, with multiple requested facts separated by commas. "
                "Scan every submitted session before answering and aggregate all instances of the requested fact across "
                "sessions: a list answer names each distinct item found in any session, not only the first. When the exact "
                "label is never stated but the evidence entails it (for example, a mentioned breakup or moving alone "
                "entails a relationship status), give the entailed label rather than abstaining. When several sessions "
                "describe similar but different events, first identify which event the question asks about (matching the "
                "named people, place, and activity) and answer only from that event; image captions describe photographs "
                "and must not replace narrated facts. Keep each fact to its "
                "minimal phrase — one tight clause per fact, with no trailing clause after ';', 'which', or 'because': "
                "extra true context is still wrong for a concise answer. "
                "For identity labels, use the full conventional label rather than a shortened form (for example, "
                "write 'transgender' rather than 'trans'); do not invent an identity absent from the evidence. "
                "Direct factual Yes/No questions must use answer_type boolean and answer exactly Yes or No. "
                "Hypothetical/likely/would questions that ask for a Yes/No judgment may use ordinary category "
                "knowledge and must use the compact form 'Yes, since <short evidence>' or 'No, since <short "
                "evidence>'. A question that begins with What, Who, Where, When, Which, or How is never a likely "
                "Yes/No answer even if it contains would or likely. Category evidence is sufficient for a likely "
                "judgment: for example, collecting classic children's books makes owning a well-known children's "
                "author such as Dr. Seuss likely. "
                "Do not add explanations to boolean, date, duration, list, or span answers.\n\n"
                "Return exactly one JSON object and no prose:\n"
                '{"support":"supported|unsupported|false_premise",'
                '"answer_type":"boolean|date|duration|list|span|likely","answer":"concise answer"}\n\n'
                f"{date_line}Question: {item.question}\n\nSubmitted evidence:\n" + "\n\n".join(blocks)
            )
        return (
            "Answer using only the submitted conversation evidence. Apply these rules to every question: "
            "resolve indirect references across sessions before answering (for example, 'moved from my home "
            "country' plus another source naming the home country as Sweden entails 'moved from Sweden'); "
            "resolve relative dates from session timestamps; respect later knowledge updates; detect swapped "
            "people and false premises; do not answer a corrected question. If the exact asked fact is not "
            "supported, answer exactly 'No information available.' Ordinary semantic/category knowledge may be "
            "used to interpret evidence. For hypothetical/likely questions, the named item need not be explicitly "
            "listed: infer from a stated category or preference using ordinary knowledge (for example, a collection "
            "of classic children's books makes a Dr. Seuss book likely), but never invent personal facts. For "
            "likely/would questions that ask for a Yes/No judgment, use the compact form 'Yes, since <short "
            "evidence>' or 'No, since <short evidence>' and do not restate the queried item. A question that "
            "begins with What, Who, Where, When, Which, or How is never a likely Yes/No answer even if it "
            "contains words such as would or likely. Category evidence is sufficient for a likely judgment: "
            "for example, collecting classic children's books makes owning a well-known children's author "
            "such as Dr. Seuss likely. Otherwise output only a concise answer span, with "
            "multiple requested facts separated by commas.\n\n"
            f"{date_line}Question: {item.question}\n\nSubmitted evidence:\n" + "\n\n".join(blocks)
        )

    @staticmethod
    def _format_date(value: datetime) -> str:
        return f"{value.day} {value.strftime('%B')} {value.year}"

    @staticmethod
    def _direct_boolean(question: str) -> bool:
        lowered = question.strip().lower()
        return lowered.startswith((
            "did ", "does ", "do ", "is ", "was ", "were ", "are ",
            "has ", "have ", "had ", "can ",
        ))

    @staticmethod
    def _has_explicit_negative(contexts: list[dict[str, Any]]) -> bool:
        text = "\n".join(str(row.get("text") or "") for row in contexts)
        return bool(re.search(r"\b(?:no|not|never|didn't|doesn't|isn't|wasn't|cannot|can't)\b", text, re.I))

    @staticmethod
    def _relevant_evidence(item: BenchmarkItem, contexts: list[dict[str, Any]]) -> tuple[str, datetime | None]:
        stop = {
            "what", "when", "where", "which", "who", "whose", "how", "would", "could", "did",
            "does", "have", "has", "with", "from", "into", "about", "likely", "their", "there",
            "this", "that", "photo", "ago", "long", "make", "made", "give", "gave",
        }
        needle = {
            token.lower() for token in re.findall(r"[A-Za-z0-9']+", item.question)
            if len(token) > 2 and token.lower() not in stop
        }
        best: tuple[int, str, datetime | None] = (-1, "", None)
        for row in contexts:
            parsed = parse_timestamp(str(row.get("timestamp") or ""))
            for line in str(row.get("text") or "").splitlines():
                haystack = {token.lower() for token in re.findall(r"[A-Za-z0-9']+", line)}
                score = len(needle & haystack)
                if score > best[0]:
                    best = (score, line, parsed)
        return best[1], best[2]

    def _normalize_payload(
        self,
        item: BenchmarkItem,
        contexts: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        support = str(payload.get("support") or "").strip().lower()
        answer_type = str(payload.get("answer_type") or "").strip().lower()
        answer = str(payload.get("answer") or "").strip()
        if support not in self.SUPPORT_VALUES:
            raise ValueError("support must be supported, unsupported, or false_premise")
        if answer_type not in self.ANSWER_TYPES:
            raise ValueError("answer_type is invalid")
        normalized_payload = {"support": support, "answer_type": answer_type, "answer": answer}
        if support != "supported":
            normalized_payload["answer"] = "No information available."
            return "No information available.", normalized_payload
        if not answer:
            raise ValueError("supported answers require a non-empty answer")

        wh_question = item.question.strip().lower().startswith((
            "what ", "who ", "where ", "when ", "which ", "how ",
        ))
        if wh_question and answer_type == "likely":
            raise ValueError("WH questions require a list, span, date, or duration rather than a likely Yes/No answer")
        if self._direct_boolean(item.question):
            match = re.match(r"^(yes|no)\b", answer, re.I)
            if not match:
                raise ValueError("direct factual boolean answers must start with Yes or No")
            answer = match.group(1).capitalize()
            # Evidence that says somebody else did it does not directly prove a
            # negative proposition about the queried person.
            if answer == "No" and not self._has_explicit_negative(contexts):
                normalized_payload.update({"support": "false_premise", "answer": "No information available."})
                return "No information available.", normalized_payload
        elif answer_type == "likely":
            if not re.match(r"^(yes|no),?\s+since\b", answer, re.I):
                raise ValueError("likely answers must use 'Yes, since ...' or 'No, since ...'")

        relevant_text, timestamp = self._relevant_evidence(item, contexts)
        relevant_lower = relevant_text.lower()
        if answer_type == "date" and "last week" in relevant_lower and timestamp:
            answer = f"The week before {self._format_date(timestamp)}"
        elif answer_type == "date" and "yesterday" in relevant_lower and timestamp:
            answer = self._format_date(timestamp - timedelta(days=1))
        elif answer_type == "date" and "last friday" in relevant_lower and timestamp:
            days = (timestamp.weekday() - 4) % 7 or 7
            answer = self._format_date(timestamp - timedelta(days=days))
        elif item.question.strip().lower().startswith("how long ago") and not re.search(r"\bago\b", answer, re.I):
            answer = answer.rstrip(" .") + " ago"
        normalized_payload["answer"] = answer
        return answer, normalized_payload

    def _chat(self, messages: list[dict[str, str]]) -> Any:
        from core.llm.chat_api import ollama_chat, openai_compatible_chat
        base_url = str(self.config.get("base_url") or "https://api.openai.com/v1")
        if "11434" in base_url and not base_url.rstrip("/").endswith("/v1"):
            return ollama_chat(
                messages, model=str(self.config.get("model") or "qwen3.5:4b"), base_url=base_url,
                think=bool(self.config.get("answer_think", False)),
                timeout=int(self.config.get("timeout_seconds") or 300),
                num_predict=int(self.config.get("max_tokens") or 1000),
                json_format=self.profile == "normalized-v1",
            )
        extra_body = copy.deepcopy(self.config.get("answer_extra_body") or self.config.get("extra_body") or {})
        if "enable_thinking" not in extra_body and "reasoning" not in extra_body:
            extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        key_env = str(self.config.get("api_key_env") or "").strip()
        api_key = str(self.config.get("api_key") or "")
        if not api_key and key_env:
            api_key = str(os.getenv(key_env) or "")
            if not api_key:
                raise RuntimeError(f"Required API key environment variable is not set: {key_env}")
        if not api_key:
            api_key = str(os.getenv("OPENAI_API_KEY") or "")
        return openai_compatible_chat(
            messages, model=str(self.config.get("model") or "gpt-4o-mini"), base_url=base_url,
            api_key=api_key,
            timeout=int(self.config.get("timeout_seconds") or 300),
            max_tokens=int(self.config.get("max_tokens") or 1000),
            temperature=float(self.config.get("temperature", 0) or 0),
            extra_body=extra_body,
        )

    def answer(self, item: BenchmarkItem, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = self.build_prompt(item, contexts)
        messages = [
            {"role": "system", "content": "You are a precise long-term conversational memory assistant."},
            {"role": "user", "content": prompt},
        ]
        started = time.monotonic()
        responses, validation_errors = [], []
        total_prompt_tokens = total_completion_tokens = 0
        max_attempts = 1 if self.profile == "legacy" else 1 + int(
            self.config.get("answer_validation_retries", 2) or 0
        )
        hypothesis = ""
        normalized_payload = None
        response = None
        for attempt in range(1, max_attempts + 1):
            response = self._chat(messages)
            raw = response.content.strip()
            responses.append(raw)
            total_prompt_tokens += int(response.prompt_eval_count or 0)
            total_completion_tokens += int(response.eval_count or 0)
            if self.profile == "legacy":
                hypothesis = raw
                break
            try:
                from .agentic import _parse_json_object
                payload = _parse_json_object(raw)
                hypothesis, normalized_payload = self._normalize_payload(item, contexts, payload)
                break
            except ValueError as exc:
                validation_errors.append(str(exc))
                if attempt >= max_attempts:
                    raise
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": (
                        f"The response is invalid: {exc}. Return a corrected JSON object only. "
                        "Do not add facts or explanations."
                    )},
                ])
        return {
            "hypothesis": hypothesis,
            "prompt": prompt,
            "answer_latency_seconds": round(time.monotonic() - started, 3),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "model": (response.model if response else None) or self.config.get("model"),
            "answer_profile": self.profile,
            "answer_attempts": len(responses),
            "raw_answer_response": responses[-1] if responses else "",
            "answer_payload": normalized_payload,
            "answer_validation_errors": validation_errors,
        }


def _normalize_tracks(tracks: Iterable[str]) -> list[str]:
    expanded = []
    for track in tracks:
        if track in {"both", "all"}:
            expanded.extend(TRACKS)
        else:
            expanded.append(track)
    result = list(dict.fromkeys(expanded or TRACKS))
    unknown = set(result) - set(TRACKS)
    if unknown:
        raise ValueError(f"Unknown tracks: {sorted(unknown)}")
    return result


def _base_record(item: BenchmarkItem, track: str) -> dict[str, Any]:
    return {
        "dataset": item.dataset, "scope_id": item.scope_id,
        "question_id": item.question_id, "question_type": item.question_type,
        "question": item.question, "answer": item.answer,
        "question_date": item.question_date,
        "evidence_session_ids": item.evidence_session_ids,
        "evidence_turn_ids": item.evidence_turn_ids,
        "track": track,
    }


def _retrieval_metrics(item: BenchmarkItem, sessions: list[str], turns: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    if item.evidence_session_ids:
        for k in RETRIEVAL_KS:
            result.update({f"session_{key}": value for key, value in
                           retrieval_at_k(sessions, item.evidence_session_ids, k).items()})
    if item.evidence_turn_ids:
        for k in RETRIEVAL_KS:
            result.update({f"turn_{key}": value for key, value in
                           retrieval_at_k(turns, item.evidence_turn_ids, k).items()})
    return result


def _preliminary_failure(record: dict[str, Any], visible_sessions: set[str]) -> str | None:
    gold = set(record.get("evidence_session_ids") or [])
    if gold and not gold.issubset(visible_sessions):
        return "remember_missing"
    if gold and not record.get("retrieval_metrics", {}).get("session_recall_any@5", 0):
        return "retrieval_miss"
    if record.get("track") == "skill-agent" and str(record.get("agent_stop_reason", "")).startswith("max_steps"):
        return "agent_stopped_early"
    return None


_ANSWER_ONLY_FIELDS = {
    "hypothesis", "prompt", "answer_latency_seconds", "answer_attempts",
    "raw_answer_response", "answer_payload", "answer_validation_errors", "answer_profile",
    "score", "model", "prompt_tokens", "completion_tokens", "total_latency_seconds",
    "latency_seconds",
}


def _artifact_track(track: str, result_tag: str | None) -> str:
    if not result_tag:
        return track
    tag = re.sub(r"[^A-Za-z0-9_-]+", "-", result_tag.strip()).strip("-_")
    if not tag:
        raise ValueError("result_tag must contain at least one letter or digit")
    return f"{track}-{tag}"


def _cache_digest(record: dict[str, Any]) -> str:
    payload = {
        key: record.get(key) for key in (
            "question_id", "retrieved", "ranked_session_ids", "ranked_turn_ids",
            "submitted_evidence", "trajectory", "retrieval_profile", "retrieval_audit",
        )
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _snapshot_from_result(record: dict[str, Any], *, source_track: str | None = None) -> dict[str, Any]:
    """Strip answer-only fields from an old result so its retrieval can be replayed."""
    snapshot = {key: copy.deepcopy(value) for key, value in record.items()
                if key not in _ANSWER_ONLY_FIELDS}
    trajectory = list(snapshot.get("trajectory") or [])
    retrieval_prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in trajectory)
    retrieval_completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in trajectory)
    if not trajectory:
        retrieval_prompt_tokens = int(record.get("retrieval_prompt_tokens") or 0)
        retrieval_completion_tokens = int(record.get("retrieval_completion_tokens") or 0)
    total_latency = float(record.get("total_latency_seconds") or record.get("latency_seconds") or 0)
    answer_latency = float(record.get("answer_latency_seconds") or 0)
    snapshot.update({
        "status": "completed" if record.get("status") != "error" else "error",
        "source_track": source_track or str(record.get("source_track") or record.get("track") or ""),
        "retrieval_cache_schema": 1,
        "retrieval_latency_seconds": float(record.get("retrieval_latency_seconds") or max(
            0.0, total_latency - answer_latency
        )),
        "retrieval_prompt_tokens": retrieval_prompt_tokens,
        "retrieval_completion_tokens": retrieval_completion_tokens,
        "retrieval_model": record.get("retrieval_model") or (
            record.get("model") if trajectory else None
        ),
    })
    if snapshot.get("failure_attribution") not in {
        "remember_missing", "retrieval_miss", "agent_stopped_early",
    }:
        snapshot["failure_attribution"] = None
    snapshot["retrieval_cache_sha256"] = _cache_digest(snapshot)
    return snapshot


def _materialize_retrieval_cache(run_dir: Path, track: str) -> Path:
    cache_path = run_dir / f"retrieval.{track}.jsonl"
    existing = {
        row["question_id"] for row in latest_by_question(read_jsonl(cache_path))
    }
    source_path = run_dir / f"results.{track}.jsonl"
    if not source_path.exists():
        if existing:
            return cache_path
        raise FileNotFoundError(f"No result or retrieval cache for track: {track}")
    for record in latest_by_question(read_jsonl(source_path)):
        if record.get("status") == "error" or record["question_id"] in existing:
            continue
        append_jsonl(cache_path, _snapshot_from_result(record, source_track=track))
    return cache_path


def _combine_answer(
    snapshot: dict[str, Any],
    answer: dict[str, Any],
    *,
    artifact_track: str,
    replay_wall_latency_seconds: float | None = None,
) -> dict[str, Any]:
    record = copy.deepcopy(snapshot)
    record.update(answer)
    retrieval_prompt_tokens = int(snapshot.get("retrieval_prompt_tokens") or 0)
    retrieval_completion_tokens = int(snapshot.get("retrieval_completion_tokens") or 0)
    answer_latency = float(answer.get("answer_latency_seconds") or 0)
    retrieval_latency = float(snapshot.get("retrieval_latency_seconds") or 0)
    record.update({
        "track": artifact_track,
        "status": "completed",
        "prompt_tokens": retrieval_prompt_tokens + int(answer.get("prompt_tokens") or 0),
        "completion_tokens": retrieval_completion_tokens + int(answer.get("completion_tokens") or 0),
        "total_latency_seconds": round(retrieval_latency + answer_latency, 3),
        "model": answer.get("model") or snapshot.get("retrieval_model"),
    })
    if snapshot.get("source_track") == "skill-agent" or snapshot.get("trajectory"):
        record["latency_seconds"] = record["total_latency_seconds"]
    if replay_wall_latency_seconds is not None:
        record["answer_replay"] = {
            "retrieval_reused": True,
            "source_track": snapshot.get("source_track"),
            "retrieval_cache_sha256": snapshot.get("retrieval_cache_sha256"),
            "wall_latency_seconds": round(replay_wall_latency_seconds, 3),
        }
    return record


def _reassemble_turn_only_contexts(
    item: BenchmarkItem,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Repair cached turn-only submissions without rerunning the Agent or retrieval tools."""
    submitted = snapshot.get("submitted_evidence") or {}
    selected_turns = list(dict.fromkeys(str(value) for value in submitted.get("turn_ids") or []))
    selected_sessions = list(dict.fromkeys(str(value) for value in submitted.get("session_ids") or []))
    if not selected_turns:
        return snapshot
    turn_to_session = {
        turn_id: session.session_id for session in item.sessions for turn_id in session.turn_ids
    }
    for turn_id in selected_turns:
        session_id = turn_to_session.get(turn_id)
        if session_id and session_id not in selected_sessions:
            selected_sessions.append(session_id)
    existing_sessions = {
        str(row.get("session_id") or "") for row in snapshot.get("retrieved") or []
    }
    missing_sessions = [value for value in selected_sessions if value not in existing_sessions]
    if not missing_sessions:
        return snapshot
    session_map = {session.session_id: session for session in item.sessions}
    contexts = list(snapshot.get("retrieved") or [])
    for rank, session_id in enumerate(missing_sessions, len(contexts) + 1):
        session = session_map.get(session_id)
        if not session:
            continue
        contexts.append({
            "session_id": session_id,
            "timestamp": session.timestamp,
            "text": session.text,
            "turn_ids": session.turn_ids,
            "matched_turn_ids": [turn for turn in selected_turns if turn in set(session.turn_ids)],
            "score": 1.0 / rank,
            "evidence": [{"retrieval_channel": "cached-turn-reassembly"}],
        })
    repaired = copy.deepcopy(snapshot)
    repaired["retrieved"] = contexts
    repaired["ranked_session_ids"] = list(dict.fromkeys([
        *[str(row.get("session_id") or "") for row in contexts if row.get("session_id")],
    ]))
    repaired["ranked_turn_ids"] = selected_turns
    repaired["retrieval_metrics"] = _retrieval_metrics(
        item, repaired["ranked_session_ids"], repaired["ranked_turn_ids"]
    )
    repaired["evidence_reassembled_from_cache"] = True
    repaired["retrieval_cache_sha256"] = _cache_digest(repaired)
    return repaired


def _retrieval_audit_payload(retrieval: dict[str, Any]) -> dict[str, Any]:
    """Persist compact, recomputable ranking diagnostics without the full explore payload."""
    return {
        key: copy.deepcopy(retrieval.get(key))
        for key in (
            "retrieval_profile", "query_terms", "turn_scores", "session_scores",
            "referential_bridges", "budget", "profile",
        )
        if retrieval.get(key) is not None
    } | {
        "coverage": copy.deepcopy((retrieval.get("explore") or {}).get("coverage") or {}),
    }


def retrieve_benchmark(
    run_dir: Path,
    config_path: Path,
    *,
    tracks: Iterable[str] = ("baseline",),
    eligible_evidence_only: bool = False,
    question_ids: Iterable[str] = (),
    limit: int | None = None,
    max_agent_steps: int = 8,
    semantic_threshold: float = 0.3,
    retrieval_profile: str = "legacy",
    candidate_k: int = 30,
    context_k: int = 5,
    evidence_token_budget: int = 1600,
    neighbor_turns: int = 1,
    resume: bool = False,
    agent_thinking: bool | None = None,
    result_tag: str | None = None,
) -> dict[str, Any]:
    """Run and persist retrieval independently from answer generation."""
    if retrieval_profile not in {"legacy", "hybrid-v2"}:
        raise ValueError("retrieval_profile must be legacy or hybrid-v2")
    source_tracks = _normalize_tracks(tracks)
    if retrieval_profile != "legacy" and any(track != "baseline" for track in source_tracks):
        raise ValueError("hybrid-v2 currently supports the deterministic baseline track only")

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = _load_config(config_path)
    if agent_thinking is not None:
        config = copy.deepcopy(config)
        config.setdefault("llm", {})["agent_think"] = bool(agent_thinking)
    items, dataset_path = _selected_items(
        manifest["dataset"], data_dir_for_dataset_path(manifest["dataset"], Path(manifest["dataset_path"])),
        question_ids=question_ids, limit=limit,
    )
    if sha256_file(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("Dataset hash changed; refusing evaluation")

    default_tag = retrieval_profile if retrieval_profile != "legacy" else None
    effective_tag = result_tag or default_tag
    artifact_tracks = {track: _artifact_track(track, effective_tag) for track in source_tracks}
    processed = errors = 0

    from core.server.registry import GraphRegistry
    for scope_id, scope_items in group_by_scope(items).items():
        scope_info = (manifest.get("scopes") or {}).get(scope_id)
        if not scope_info:
            continue
        visible = list(scope_info.get("visible_sessions") or [])
        visible_set = set(visible)
        selected = [
            item for item in scope_items
            if not eligible_evidence_only or (
                bool(item.evidence_session_ids)
                and set(item.evidence_session_ids).issubset(visible_set)
            )
        ]
        if not selected:
            continue
        sessions = [session for session in scope_items[0].sessions if session.session_id in visible_set]
        documents = scope_info.get("documents") or {}
        document_to_session = {
            str(documents[session.session_id]["document_id"]): session.session_id
            for session in sessions if session.session_id in documents
            and documents[session.session_id].get("status") == "active"
        }
        scope_dir = run_dir / scope_info["library_dir"]
        scope_config = copy.deepcopy(config)
        scope_config["storage_path"] = str(scope_dir)
        registry = GraphRegistry(str(scope_dir), scope_config)
        processor = registry.get_processor("library")
        try:
            try:
                retriever = DeepDreamRetriever(
                    processor.storage, document_to_session, sessions,
                    allowed_document_ids=document_to_session,
                )
            except TypeError:  # compatibility with small third-party/fake adapters
                retriever = DeepDreamRetriever(processor.storage, document_to_session, sessions)
            agents: dict[str, Any] = {}
            if "skill-agent" in source_tracks:
                from .agentic import AgentDecisionModel, AgenticMemoryRunner, AgenticMemoryTools
                answerer = AnswerGenerator(config, profile="normalized-v1")
                tools = AgenticMemoryTools(processor.storage, retriever, document_to_session, sessions)
                agents["skill-agent"] = AgenticMemoryRunner(
                    tools, AgentDecisionModel(config), answerer,
                    max_steps=max_agent_steps, answer_top_k=context_k,
                )
            for track in source_tracks:
                artifact_track = artifact_tracks[track]
                retrieval_path = run_dir / f"retrieval.{artifact_track}.jsonl"
                completed = {
                    row["question_id"] for row in latest_by_question(read_jsonl(retrieval_path))
                    if row.get("status") != "error"
                } if resume else set()
                for item in selected:
                    if item.question_id in completed:
                        continue
                    started = time.monotonic()
                    base = _base_record(item, artifact_track)
                    try:
                        if track == "baseline":
                            retrieval = retriever.explore(
                                item.question,
                                limit=max(max(RETRIEVAL_KS), context_k),
                                threshold=semantic_threshold,
                                retrieval_profile=retrieval_profile,
                                candidate_k=candidate_k,
                                context_k=context_k,
                                evidence_token_budget=evidence_token_budget,
                                neighbor_turns=neighbor_turns,
                            )
                            contexts = retrieval["contexts"]
                            ranked_sessions = retrieval["ranked_session_ids"]
                            ranked_turns = retrieval["ranked_turn_ids"]
                            retrieval_payload = {
                                "retrieval_latency_seconds": round(time.monotonic() - started, 3),
                                "retrieval_prompt_tokens": 0,
                                "retrieval_completion_tokens": 0,
                                "retrieval_model": None,
                                "retrieval_profile": retrieval_profile,
                                "retrieval_audit": _retrieval_audit_payload(retrieval),
                            }
                        else:
                            retrieval_payload = agents[track].retrieve(item)
                            contexts = retrieval_payload["retrieved"]
                            ranked_sessions = [row["session_id"] for row in contexts]
                            ranked_turns = retrieval_payload["retrieved_turn_ids"]
                            retrieval_payload["retrieval_profile"] = "skill-agent"
                        snapshot = {
                            **base,
                            "status": "completed",
                            "source_track": track,
                            "result_tag": effective_tag,
                            "retrieved": contexts,
                            "ranked_session_ids": ranked_sessions,
                            "ranked_turn_ids": ranked_turns,
                            "retrieval_metrics": _retrieval_metrics(item, ranked_sessions, ranked_turns),
                            **retrieval_payload,
                        }
                        snapshot["failure_attribution"] = _preliminary_failure(snapshot, visible_set)
                        snapshot["retrieval_cache_schema"] = 2
                        snapshot["retrieval_cache_sha256"] = _cache_digest(snapshot)
                    except Exception as exc:
                        snapshot = {
                            **base, "status": "error", "source_track": track,
                            "result_tag": effective_tag, "retrieved": [],
                            "ranked_session_ids": [], "ranked_turn_ids": [],
                            "retrieval_metrics": {}, "retrieval_profile": retrieval_profile,
                            "failure_attribution": "retrieval_miss",
                            "retrieval_latency_seconds": round(time.monotonic() - started, 3),
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        }
                        errors += 1
                    append_jsonl(retrieval_path, snapshot)
                    processed += 1
        finally:
            processor.storage.close()

    manifest["schema_version"] = max(4, int(manifest.get("schema_version") or 1))
    manifest["tracks"] = list(dict.fromkeys([
        *(manifest.get("tracks") or []), *artifact_tracks.values(),
    ]))
    retrieval_config = {
        "profile": retrieval_profile,
        "candidate_k": candidate_k,
        "context_k": context_k,
        "evidence_token_budget": evidence_token_budget,
        "neighbor_turns": neighbor_turns,
        "semantic_threshold": semantic_threshold,
        "embedding": _public_config(config).get("embedding"),
        "retrieval_code_sha256": _file_sha256(Path(__file__).with_name("retrieval.py")),
    }
    profiles = manifest.setdefault("retrieval_profiles", {})
    variants = manifest.setdefault("track_variants", {})
    for source_track, artifact_track in artifact_tracks.items():
        profiles[artifact_track] = retrieval_config
        variants[artifact_track] = {
            "source_track": source_track,
            "result_tag": effective_tag,
            "retrieval_only": True,
            "retrieval_reused": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "retrieval": retrieval_config,
            "config": _public_config(config),
        }
    manifest["retrieval_completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    return {
        "run_dir": str(run_dir), "processed": processed, "errors": errors,
        "tracks": list(artifact_tracks.values()), "source_tracks": source_tracks,
        "retrieval_profile": retrieval_profile,
    }


def evaluate_benchmark(
    run_dir: Path,
    config_path: Path,
    *,
    tracks: Iterable[str] = TRACKS,
    eligible_evidence_only: bool = False,
    question_ids: Iterable[str] = (),
    limit: int | None = None,
    max_agent_steps: int = 8,
    answer_top_k: int = 5,
    semantic_threshold: float = 0.3,
    resume: bool = False,
    answer_profile: str = "normalized-v1",
    agent_thinking: bool | None = None,
    result_tag: str | None = None,
    retrieval_profile: str = "legacy",
    candidate_k: int = 30,
    context_k: int | None = None,
    evidence_token_budget: int = 1600,
    neighbor_turns: int = 1,
    answer_workers: int = 1,
) -> dict[str, Any]:
    if retrieval_profile != "legacy":
        context_k = int(context_k or answer_top_k)
        retrieval_result = retrieve_benchmark(
            run_dir, config_path, tracks=tracks,
            eligible_evidence_only=eligible_evidence_only,
            question_ids=question_ids, limit=limit,
            max_agent_steps=max_agent_steps, semantic_threshold=semantic_threshold,
            retrieval_profile=retrieval_profile, candidate_k=candidate_k,
            context_k=context_k, evidence_token_budget=evidence_token_budget,
            neighbor_turns=neighbor_turns, resume=resume,
            agent_thinking=agent_thinking, result_tag=result_tag,
        )
        answer_result = replay_answers(
            run_dir, config_path, tracks=retrieval_result["tracks"],
            result_tag="answer-v1", answer_profile=answer_profile,
            question_ids=question_ids, resume=resume,
        )
        return {
            "run_dir": str(run_dir),
            "processed": answer_result["processed"],
            "errors": retrieval_result["errors"] + answer_result["errors"],
            "tracks": answer_result["tracks"],
            "retrieval_tracks": retrieval_result["tracks"],
            "source_tracks": retrieval_result["source_tracks"],
        }
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = _load_config(config_path)
    if agent_thinking is not None:
        config = copy.deepcopy(config)
        config.setdefault("llm", {})["agent_think"] = bool(agent_thinking)
    items, dataset_path = _selected_items(
        manifest["dataset"], data_dir_for_dataset_path(manifest["dataset"], Path(manifest["dataset_path"])),
        question_ids=question_ids,
    )
    if sha256_file(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("Dataset hash changed; refusing evaluation")
    source_tracks = _normalize_tracks(tracks)
    artifact_tracks = {track: _artifact_track(track, result_tag) for track in source_tracks}
    answerer = AnswerGenerator(config, profile=answer_profile)
    processed = errors = 0
    # 题级并行时保护 jsonl 追加与共享计数器；串行（默认）路径无锁。
    _io_lock = threading.Lock()

    from core.server.registry import GraphRegistry
    for scope_id, scope_items in group_by_scope(items).items():
        scope_info = (manifest.get("scopes") or {}).get(scope_id)
        if not scope_info:
            continue
        visible = list(scope_info.get("visible_sessions") or [])
        visible_set = set(visible)
        selected = [
            item for item in scope_items
            if not eligible_evidence_only or (
                bool(item.evidence_session_ids)
                and set(item.evidence_session_ids).issubset(visible_set)
            )
        ]
        if limit is not None:
            selected = selected[:limit]
        if not selected:
            continue
        sessions = [session for session in scope_items[0].sessions if session.session_id in visible_set]
        documents = scope_info.get("documents") or {}
        document_to_session = {
            str(documents[session.session_id]["document_id"]): session.session_id
            for session in sessions if session.session_id in documents
            and documents[session.session_id].get("status") == "active"
        }
        scope_dir = run_dir / scope_info["library_dir"]
        scope_config = copy.deepcopy(config)
        scope_config["storage_path"] = str(scope_dir)
        registry = GraphRegistry(str(scope_dir), scope_config)
        processor = registry.get_processor("library")
        try:
            try:
                retriever = DeepDreamRetriever(
                    processor.storage, document_to_session, sessions,
                    allowed_document_ids=document_to_session,
                )
            except TypeError:  # compatibility with small third-party/fake adapters
                retriever = DeepDreamRetriever(processor.storage, document_to_session, sessions)
            agents: dict[str, Any] = {}
            if "skill-agent" in source_tracks:
                from .agentic import AgentDecisionModel, AgenticMemoryRunner, AgenticMemoryTools
                tools = AgenticMemoryTools(processor.storage, retriever, document_to_session, sessions)
                agents["skill-agent"] = AgenticMemoryRunner(
                    tools, AgentDecisionModel(config), answerer,
                    max_steps=max_agent_steps, answer_top_k=answer_top_k,
                )
            for track in source_tracks:
                artifact_track = artifact_tracks[track]
                results_path = run_dir / f"results.{artifact_track}.jsonl"
                retrieval_path = run_dir / f"retrieval.{artifact_track}.jsonl"
                completed = {
                    row["question_id"] for row in latest_by_question(read_jsonl(results_path))
                    if row.get("status") != "error"
                } if resume else set()
                cached_retrieval = {
                    row["question_id"]: row
                    for row in latest_by_question(read_jsonl(retrieval_path))
                    if row.get("status") != "error"
                } if resume else {}
                def _one(item, _track=track, _artifact_track=artifact_track,
                         _retrieval_path=retrieval_path):
                    if item.question_id in completed:
                        return None
                    started = time.monotonic()
                    base = _base_record(item, _artifact_track)
                    snapshot = cached_retrieval.get(item.question_id)
                    try:
                        if snapshot is None:
                            retrieval_started = time.monotonic()
                            if _track == "baseline":
                                if hasattr(retriever, "explore"):
                                    retrieval = retriever.explore(
                                        item.question,
                                        limit=max(max(RETRIEVAL_KS), answer_top_k),
                                        threshold=semantic_threshold,
                                    )
                                    contexts = retrieval["contexts"]
                                    ranked_sessions = retrieval["ranked_session_ids"]
                                    ranked_turns = retrieval["ranked_turn_ids"]
                                else:
                                    contexts = retriever.search(
                                        item.question, top_k=max(max(RETRIEVAL_KS), answer_top_k),
                                        threshold=semantic_threshold,
                                    )
                                    ranked_sessions = [row["session_id"] for row in contexts]
                                    ranked_turns = [
                                        turn for row in contexts for turn in row.get("matched_turn_ids", [])
                                    ]
                                retrieval_payload = {
                                    "retrieval_latency_seconds": round(
                                        time.monotonic() - retrieval_started, 3
                                    ),
                                    "retrieval_prompt_tokens": 0,
                                    "retrieval_completion_tokens": 0,
                                    "retrieval_model": None,
                                }
                            else:
                                retrieval_payload = agents[_track].retrieve(item)
                                contexts = retrieval_payload["retrieved"]
                                ranked_sessions = [row["session_id"] for row in contexts]
                                ranked_turns = retrieval_payload["retrieved_turn_ids"]
                            metrics = _retrieval_metrics(item, ranked_sessions, ranked_turns)
                            snapshot = {
                                **base,
                                "status": "completed",
                                "source_track": _track,
                                "result_tag": result_tag,
                                "retrieved": contexts,
                                "ranked_session_ids": ranked_sessions,
                                "ranked_turn_ids": ranked_turns,
                                "retrieval_metrics": metrics,
                                **retrieval_payload,
                            }
                            snapshot["failure_attribution"] = _preliminary_failure(snapshot, visible_set)
                            snapshot["retrieval_cache_schema"] = 1
                            snapshot["retrieval_cache_sha256"] = _cache_digest(snapshot)
                            with _io_lock:
                                append_jsonl(_retrieval_path, snapshot)
                        contexts = list(snapshot.get("retrieved") or [])
                        answer = answerer.answer(item, contexts[:answer_top_k])
                        record = _combine_answer(snapshot, answer, artifact_track=_artifact_track)
                        return record, None
                    except Exception as exc:
                        return None, {
                            **base,
                            **(snapshot or {}),
                            "track": _artifact_track,
                            "status": "error",
                            "retrieved": list((snapshot or {}).get("retrieved") or []),
                            "ranked_session_ids": list((snapshot or {}).get("ranked_session_ids") or []),
                            "ranked_turn_ids": list((snapshot or {}).get("ranked_turn_ids") or []),
                            "retrieval_metrics": dict((snapshot or {}).get("retrieval_metrics") or {}),
                            "hypothesis": "", "prompt": "",
                            "failure_attribution": "agent_stopped_early" if _track == "skill-agent" else "retrieval_miss",
                            "total_latency_seconds": round(time.monotonic() - started, 3),
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        }

                if answer_workers <= 1:
                    for item in selected:
                        outcome = _one(item)
                        if outcome is None:
                            continue
                        record, err = outcome
                        if err is not None:
                            record = err
                            errors += 1
                        append_jsonl(results_path, record)
                        processed += 1
                else:
                    with ThreadPoolExecutor(max_workers=answer_workers) as pool:
                        futures = {pool.submit(_one, item): item for item in selected}
                        for future in as_completed(futures):
                            outcome = future.result()
                            if outcome is None:
                                continue
                            record, err = outcome
                            if err is not None:
                                record = err
                                errors += 1
                            with _io_lock:
                                append_jsonl(results_path, record)
                            processed += 1
        finally:
            processor.storage.close()
    manifest["tracks"] = list(dict.fromkeys([
        *(manifest.get("tracks") or []), *artifact_tracks.values(),
    ]))
    variants = manifest.setdefault("track_variants", {})
    for source_track, artifact_track in artifact_tracks.items():
        if artifact_track != source_track:
            variants[artifact_track] = {
                "source_track": source_track,
                "result_tag": result_tag,
                "agent_thinking": bool(config.get("llm", {}).get(
                    "agent_think", config.get("llm", {}).get("think", False)
                )),
                "answer_profile": answer_profile,
                "retrieval_reused": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": _public_config(config),
            }
    manifest["answer_top_k"] = answer_top_k
    manifest["max_agent_steps"] = max_agent_steps
    manifest["answer_profile"] = answer_profile
    manifest["eligible_evidence_only"] = eligible_evidence_only
    manifest["evaluation_completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["runtime_policy"] = runtime_policy_metadata()
    manifest["skill"] = _skill_metadata()
    write_json(manifest_path, manifest)
    return {
        "run_dir": str(run_dir), "processed": processed, "errors": errors,
        "tracks": list(artifact_tracks.values()), "source_tracks": source_tracks,
    }


def replay_answers(
    run_dir: Path,
    config_path: Path,
    *,
    tracks: Iterable[str] = TRACKS,
    result_tag: str = "answer-normalized-v1",
    answer_profile: str = "normalized-v1",
    question_ids: Iterable[str] = (),
    resume: bool = False,
) -> dict[str, Any]:
    """Regenerate answers from persisted evidence without opening a library or rerunning tools."""
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = _load_config(config_path)
    items, dataset_path = _selected_items(
        manifest["dataset"], data_dir_for_dataset_path(manifest["dataset"], Path(manifest["dataset_path"])),
        question_ids=question_ids,
    )
    if sha256_file(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("Dataset hash changed; refusing answer replay")
    by_question = {item.question_id: item for item in items}
    requested_tracks = list(tracks)
    if any(track in {"both", "all"} for track in requested_tracks):
        requested_tracks = [*TRACKS, *[
            track for track in requested_tracks if track not in {"both", "all"}
        ]]
    source_tracks = list(dict.fromkeys(requested_tracks or TRACKS))
    available_tracks = set(manifest.get("tracks") or [])
    unknown = [track for track in source_tracks if track not in available_tracks]
    if unknown:
        raise ValueError(f"Unknown source tracks for answer replay: {unknown}")
    answerer = AnswerGenerator(config, profile=answer_profile)
    processed = errors = 0
    output_tracks = []

    for source_track in source_tracks:
        cache_path = _materialize_retrieval_cache(run_dir, source_track)
        snapshots = [
            row for row in latest_by_question(read_jsonl(cache_path))
            if row.get("status") != "error" and row.get("question_id") in by_question
        ]
        output_track = _artifact_track(source_track, result_tag)
        output_tracks.append(output_track)
        results_path = run_dir / f"results.{output_track}.jsonl"
        source_variant = (manifest.get("track_variants") or {}).get(source_track, {})
        source_retrieval = source_variant.get("retrieval") or (
            manifest.get("retrieval_profiles") or {}
        ).get(source_track, {})
        source_context_k = int(
            source_retrieval.get("context_k") or manifest.get("answer_top_k") or 5
        )
        completed = {
            row["question_id"] for row in latest_by_question(read_jsonl(results_path))
            if row.get("status") != "error"
        } if resume else set()
        for snapshot in snapshots:
            question_id = snapshot["question_id"]
            if question_id in completed:
                continue
            item = by_question[question_id]
            started = time.monotonic()
            base = _base_record(item, output_track)
            try:
                snapshot = _reassemble_turn_only_contexts(item, snapshot)
                contexts = list(snapshot.get("retrieved") or [])
                answer = answerer.answer(item, contexts[:source_context_k])
                record = _combine_answer(
                    {**base, **snapshot, "source_track": source_track},
                    answer,
                    artifact_track=output_track,
                    replay_wall_latency_seconds=time.monotonic() - started,
                )
            except Exception as exc:
                record = {
                    **base, **snapshot, "track": output_track, "status": "error",
                    "hypothesis": "", "prompt": "",
                    "total_latency_seconds": round(time.monotonic() - started, 3),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
                errors += 1
            append_jsonl(results_path, record)
            processed += 1

        manifest["tracks"] = list(dict.fromkeys([
            *(manifest.get("tracks") or []), output_track,
        ]))
        manifest.setdefault("track_variants", {})[output_track] = {
            "source_track": source_track,
            "result_tag": result_tag,
            "agent_thinking": bool((manifest.get("track_variants") or {}).get(
                source_track, {}
            ).get("agent_thinking", manifest.get("config", {}).get("llm", {}).get("think", False))),
            "answer_profile": answer_profile,
            "retrieval_reused": True,
            "retrieval_source_sha256": _file_sha256(cache_path),
            "context_k": source_context_k,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": _public_config(config),
        }

    manifest["answer_replay_completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    return {
        "run_dir": str(run_dir), "processed": processed, "errors": errors,
        "tracks": output_tracks, "retrieval_reused": True,
    }


def run_benchmark(
    dataset: str,
    data_dir: Path,
    run_dir: Path,
    config_path: Path,
    *,
    limit: int | None = None,
    question_ids: list[str] | None = None,
    resume: bool = False,
    answer_top_k: int | None = None,
    semantic_threshold: float = 0.3,
    retrieval_mode: str | None = None,
    max_agent_steps: int = 8,
    tracks: Iterable[str] | None = None,
    scope_ids: list[str] | None = None,
    session_limit: int | None = None,
    remember_profile: str = "strong-v1",
    eligible_evidence_only: bool = False,
) -> dict[str, Any]:
    """Convenience ingest + evaluate, with v2 retrieval-mode compatibility."""
    if retrieval_mode and retrieval_mode not in {"agentic", "single-pass"}:
        raise ValueError("retrieval_mode must be agentic or single-pass")
    selected_tracks = list(tracks or ([] if retrieval_mode else TRACKS))
    if retrieval_mode:
        selected_tracks = ["skill-agent" if retrieval_mode == "agentic" else "baseline"]
    else:
        selected_tracks = _normalize_tracks(selected_tracks)
    ingest_result = ingest_benchmark(
        dataset, data_dir, run_dir, config_path,
        scope_ids=scope_ids, session_limit=session_limit,
        remember_profile=remember_profile, resume=resume,
    )
    evaluate_result = evaluate_benchmark(
        run_dir, config_path, tracks=selected_tracks,
        eligible_evidence_only=eligible_evidence_only,
        question_ids=question_ids or [], limit=limit,
        max_agent_steps=max_agent_steps,
        answer_top_k=answer_top_k or (10 if dataset == "longmemeval-s" else 5),
        semantic_threshold=semantic_threshold, resume=resume,
    )
    # A single-track legacy invocation keeps the old artifact name readable.
    if retrieval_mode:
        source = run_dir / f"results.{selected_tracks[0]}.jsonl"
        (run_dir / "results.jsonl").write_bytes(source.read_bytes())
    per_track = {}
    for track in selected_tracks:
        records = latest_by_question(read_jsonl(run_dir / f"results.{track}.jsonl"))
        per_track[track] = {
            "completed": sum(row.get("status") != "error" for row in records),
            "failed": sum(row.get("status") == "error" for row in records),
        }
    completed = per_track[selected_tracks[0]]["completed"]
    failed = per_track[selected_tracks[0]]["failed"]
    return {
        **ingest_result, **evaluate_result,
        "completed": completed, "failed": failed, "per_track": per_track,
    }
