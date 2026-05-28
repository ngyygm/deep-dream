"""
Remember 任务工作线程、历史修剪与磁盘恢复。
"""
from __future__ import annotations

import logging
import queue as _queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.server.task_progress import (
    _DONE_STATUSES,
    _TERMINAL_STATUSES,
    _RE_MAIN_1_8_DONE,
    estimate_chunk_count,
    remember_callback_ui_fields,
)
from core.server.task_journal import (
    RememberTask,
    RememberJournal,
    remember_task_from_record,
    short_task_id,
)
from core.log import info as _log_info_fn, warn as _log_warn_fn, error as _log_error_fn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Disk recovery
# ---------------------------------------------------------------------------

def recover_from_disk(
    *,
    journal: RememberJournal,
    tasks: Dict[str, RememberTask],
    task_queue: "_queue.Queue[RememberTask]",
    lock: threading.Lock,
    window_size: int,
    overlap: int,
    persist_fn,
    log_info_fn,
) -> int:
    """Restore unfinished tasks from journal.  Returns number of resumed tasks."""
    n_resume = 0
    records = journal.iter_records()
    # 同一个 task_id 取最后一条记录（JSONL 追加写入，后面的覆盖前面的）
    latest_by_tid: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        tid = rec.get("task_id")
        if tid:
            latest_by_tid[str(tid)] = rec
    records = sorted(
        latest_by_tid.values(),
        key=lambda rec: (
            float(rec.get("created_at") or 0.0),
            str(rec.get("task_id") or ""),
        ),
    )
    for rec in records:
        tid = rec.get("task_id")
        if not tid:
            continue
        st = rec.get("status")
        if st in _TERMINAL_STATUSES:
            continue
        if st == "paused":
            try:
                text = ""
                op = rec.get("original_path")
                if op and Path(op).exists():
                    text = Path(op).read_text(encoding="utf-8")
                task = remember_task_from_record(rec, text=text)
                task.status = "paused"
                task.phase = "paused"
                task.phase_label = "服务重启后保持暂停"
                task.message = "任务在服务重启后保持暂停，可手动继续"
                task.last_update = time.time()
                with lock:
                    tasks[tid] = task
                persist_fn(task)
            except Exception as e:
                logger.debug("恢复暂停任务 %s 失败: %s", tid, e)
            continue  # Paused tasks don't auto-resume
        # Non-paused pending/processing tasks: attempt recovery
        op = rec.get("original_path")
        if not op or not Path(op).exists():
            rec2 = dict(rec)
            rec2["status"] = "failed"
            rec2["error"] = "重启恢复失败：原始文本文件不存在"
            rec2["finished_at"] = time.time()
            try:
                tdead = remember_task_from_record(rec2, text="")
                journal.write(tdead)
            except Exception as e:
                logger.debug("写入恢复失败记录 %s: %s", tid, e)
            log_info_fn(
                "[Remember] 恢复跳过 task_id=%s: 原文缺失" % short_task_id(str(tid))
            )
            continue
        try:
            text = Path(op).read_text(encoding="utf-8")
        except Exception as e:
            rec2 = dict(rec)
            rec2["status"] = "failed"
            rec2["error"] = "重启恢复失败：无法读取原文: %s" % e
            rec2["finished_at"] = time.time()
            try:
                tdead = remember_task_from_record(rec2, text="")
                journal.write(tdead)
            except Exception as _journal_err:
                logger.warning("写入恢复失败记录到日志失败: %s", _journal_err)
            continue
        task = remember_task_from_record(rec, text=text)
        task.status = "queued"
        task.started_at = None
        task.finished_at = None
        task.error = None
        task.result = None
        task.phase = "queued"
        task.phase_label = "恢复后等待处理"
        task.phase_current = 0
        task.phase_total = 0
        task.total_chunks = max(
            task.total_chunks,
            estimate_chunk_count(len(task.text), window_size, overlap),
        )
        # 三条链的断点分别恢复；processed_chunks 继续兼容为 step10 已完成窗口数。
        _tc = max(0, int(task.total_chunks or 0))
        _step10_done = min(_tc, max(0, int(task.step10_done_chunks or task.processed_chunks or 0)))
        _step9_done = min(_tc, max(_step10_done, int(task.step9_done_chunks or task.processed_chunks or 0)))
        _main_done = min(_tc, max(_step9_done, int(task.main_done_chunks or task.processed_chunks or 0)))
        task.main_done_chunks = _main_done
        task.step9_done_chunks = _step9_done
        task.step10_done_chunks = _step10_done
        task.processed_chunks = _step10_done
        # 根据关系链已完成窗口数恢复总进度
        if task.total_chunks > 0 and task.step10_done_chunks > 0:
            task.progress = task.step10_done_chunks / task.total_chunks
        else:
            task.progress = 0.0
        task.main_progress = (_main_done / task.total_chunks) if task.total_chunks > 0 else 0.0
        if task.step10_done_chunks > 0 or task.step9_done_chunks > 0 or task.main_done_chunks > 0:
            task.message = (
                "服务重启后已恢复入队（"
                "主链 %d/%d · "
                "实体 %d/%d · "
                "关系 %d/%d）" % (
                    task.main_done_chunks, task.total_chunks,
                    task.step9_done_chunks, task.total_chunks,
                    task.step10_done_chunks, task.total_chunks,
                )
            )
        else:
            task.message = "服务重启后已恢复入队"
        task.last_update = time.time()
        with lock:
            tasks[tid] = task
        task_queue.put(task)
        persist_fn(task)
        n_resume += 1
        log_info_fn(
            "[Remember] 恢复未完成任务并入队: task_id=%s, source_name=%r"
            % (short_task_id(tid), task.source_name)
        )
    if n_resume:
        log_info_fn(
            "[Remember] 启动恢复：重新入队 %d 个未完成任务"
            "（已完成/失败仅保留在 journal，按需通过 status 查询）" % n_resume
        )
    return n_resume


# ---------------------------------------------------------------------------
# History trimming
# ---------------------------------------------------------------------------

def trim_history(
    tasks: Dict[str, RememberTask],
    max_history: int,
    lock: threading.Lock,
) -> None:
    """Remove oldest completed/failed tasks when the tracked set exceeds *max_history*."""
    if len(tasks) <= max_history:
        return
    items = sorted(tasks.values(), key=lambda t: t.created_at)
    to_remove = len(tasks) - max_history
    removed = 0
    for t in items:
        if t.status in _DONE_STATUSES and removed < to_remove:
            del tasks[t.task_id]
            removed += 1


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

def worker_loop(q: "RememberTaskQueue") -> None:  # noqa: C901 — legacy complexity
    """串行执行滑窗处理：从数据库加载最新缓存续写，或从空缓存开始。"""
    while True:
        task = q._queue.get()
        try:
            if task.status == "cancelled":
                q._log_info("[Remember] 跳过已删除任务: task_id=%s" % short_task_id(task.task_id))
                continue
            # Double-check under lock: delete_pending_task may have popped the
            # task from self._tasks after the cancelled check above but before
            # _update_task_progress writes status="queued" back to the task
            # object and _persist re-appends it to the journal.
            with q._lock:
                if task.task_id not in q._tasks:
                    q._log_info(
                        "[Remember] 跳过已删除任务 (不在任务表): "
                        "task_id=%s" % short_task_id(task.task_id)
                    )
                    continue
            _existing_main_chunks = task.main_done_chunks or 0
            _existing_step9_chunks = task.step9_done_chunks or 0
            _existing_step10_chunks = task.step10_done_chunks or task.processed_chunks or 0
            if not task.failed_window_indices and (
                _existing_main_chunks > 0 or _existing_step9_chunks > 0 or _existing_step10_chunks > 0
            ):
                try:
                    _missing = q.detect_repair_windows(task)
                    if _missing:
                        task.failed_window_indices = list(_missing)
                        task.failed_window_errors = [
                            {
                                "phase": s.get("missing_phase") or "missing",
                                "window_index": s.get("window_index"),
                                "error": "窗口缺失或落库不完整",
                            }
                            for s in (task.repair_window_statuses or [])
                        ]
                        _existing_main_chunks = max(0, task.total_chunks - len(_missing))
                        _existing_step9_chunks = _existing_main_chunks
                        _existing_step10_chunks = _existing_main_chunks
                except Exception as exc:
                    q._log_warn("[Remember] 补缺检测失败，将按断点继续: %s" % exc)
            _init_progress = _existing_step10_chunks / task.total_chunks if task.total_chunks > 0 else 0.0
            _resume_hint = (
                "断点续传："
                "主链 %d/%d · "
                "实体 %d/%d · "
                "关系 %d/%d" % (
                    _existing_main_chunks, task.total_chunks,
                    _existing_step9_chunks, task.total_chunks,
                    _existing_step10_chunks, task.total_chunks,
                )
                if (_existing_main_chunks > 0 or _existing_step9_chunks > 0 or _existing_step10_chunks > 0)
                and task.total_chunks > 0
                else ("断点续传" if (_existing_main_chunks > 0 or _existing_step9_chunks > 0 or _existing_step10_chunks > 0) else "开始处理")
            )
            _start_chunk = task.step10_done_chunks or task.processed_chunks or 0
            _is_targeted_retry = bool(task.failed_window_indices)
            if _is_targeted_retry:
                _start_chunk = 0
                _target_indices = sorted(task.failed_window_indices)
                _init_progress = max(0, (task.total_chunks - len(_target_indices))) / task.total_chunks if task.total_chunks > 0 else 0.0
            else:
                _target_indices = None
            _uses_external_cache = q._task_uses_external_cache(task)
            _task_processor = q._processor if _uses_external_cache else q._processor_factory()
            q._update_task_progress(
                task,
                status="queued",
                phase="waiting_cache_chain" if _uses_external_cache else "queued",
                phase_label="等待前序缓存链" if _uses_external_cache else "等待开始",
                phase_current=_existing_step10_chunks,
                phase_total=max(1, task.total_chunks),
                main_done_chunks=_existing_main_chunks,
                step9_done_chunks=_existing_step9_chunks,
                step10_done_chunks=_existing_step10_chunks,
                processed_chunks=_existing_step10_chunks,
                total_chunks=task.total_chunks,
                run_start_chunks=_existing_step10_chunks,
                progress=_init_progress,
                step9_progress=0.0,
                step9_label="",
                step10_progress=0.0,
                step10_label="",
                main_progress=0.0,
                main_label="",
                message=("等待前一个接续缓存链的任务完成后开始" if _uses_external_cache else "等待工作线程开始"),
                started_at=None,
                finished_at=None,
                error=None,
            )
            q._persist(task)
            if _uses_external_cache and q._phase2_lock.locked():
                q._log_info(
                    "[Remember] 等待串行执行: task_id=%s, source_name=%r"
                    % (short_task_id(task.task_id), task.source_name)
                )

            last_exc = None
            for attempt in range(q._max_retries + 1):
                try:
                    # 构建进度回调：将处理器的进度更新转发到任务跟踪
                    _task_ref = task  # 闭包引用

                    def _on_progress(progress: float, phase_label: str, message: str, chain_id: str = "step9", _t=_task_ref):
                        _fields = remember_callback_ui_fields(
                            _t, progress, phase_label, message, chain_id,
                        )
                        q._update_task_progress(_t, **_fields)
                        q._persist(_t)

                    def _on_main_chunk_done(processed_count: int, _t=_task_ref):
                        _tc = max(1, int(_t.total_chunks or 1))
                        _pc = max(0, int(processed_count))
                        if _is_targeted_retry:
                            _pc = min(_tc, max(0, _tc - len(_target_indices) + _pc))
                        _pg = min(1.0, float(_pc) / float(_tc))
                        _ml = _t.main_label or ""
                        # 当所有窗口的步骤1-5完成时，更新 main_label 显示完成状态
                        if _pc >= _tc and not _RE_MAIN_1_8_DONE.search(_ml):
                            _ml = "步骤1–8/10 已完成"
                        q._update_task_progress(
                            _t,
                            main_done_chunks=max(_pc, int(_t.main_done_chunks or 0)),
                            main_progress=max(_pg, float(_t.main_progress or 0.0)),
                            main_label=_ml,
                        )
                        q._persist(_t)

                    def _on_step9_chunk_done(processed_count: int, _t=_task_ref):
                        _tc = max(1, int(_t.total_chunks or 1))
                        _pc = max(0, int(processed_count))
                        if _is_targeted_retry:
                            _pc = min(_tc, max(0, _tc - len(_target_indices) + _pc))
                        q._update_task_progress(
                            _t,
                            step9_done_chunks=max(_pc, int(_t.step9_done_chunks or 0)),
                        )
                        q._persist(_t)

                    def _on_chunk_done(processed_count: int, _t=_task_ref):
                        """窗口 step10 完成后更新 processed_chunks；总进度与已完成窗数一致（单调递增）。"""
                        _tc = max(1, int(_t.total_chunks or 1))
                        _pc = max(0, int(processed_count))
                        if _is_targeted_retry:
                            _pc = min(_tc, max(0, _tc - len(_target_indices) + _pc))
                        _pg = min(1.0, float(_pc) / float(_tc))
                        q._update_task_progress(
                            _t,
                            step10_done_chunks=max(_pc, int(_t.step10_done_chunks or 0)),
                            processed_chunks=max(_pc, int(_t.processed_chunks or 0)),
                            progress=max(_pg, float(_t.progress or 0.0)),
                        )
                        q._persist(_t)

                    def _run_task():
                        q._set_active_processor(task.task_id, _task_processor)
                        try:
                            _document_path = task.cache_document_path or task.original_path
                            _kwargs = dict(
                                text=task.text,
                                source_document=task.source_name,
                                verbose=q._detail_logs,
                                verbose_steps=not q._detail_logs,
                                load_cache_memory=_uses_external_cache,
                                event_time=task.event_time,
                                document_path=_document_path,
                                progress_callback=_on_progress,
                                control_callback=lambda _t=task: _t.control_action,
                                start_chunk=_start_chunk,
                                main_chunk_done_callback=_on_main_chunk_done,
                                step9_chunk_done_callback=_on_step9_chunk_done,
                                chunk_done_callback=_on_chunk_done,
                            )
                            if _is_targeted_retry:
                                _kwargs["target_window_indices"] = _target_indices
                            return _task_processor.remember_text(**_kwargs)
                        finally:
                            q._clear_active_processor(task.task_id, _task_processor)

                    def _run_task_with_targets(target_indices):
                        """Run pipeline targeting specific window indices (for retry)."""
                        q._set_active_processor(task.task_id, _task_processor)
                        try:
                            _document_path = task.cache_document_path or task.original_path
                            return _task_processor.remember_text(
                                text=task.text,
                                source_document=task.source_name,
                                verbose=q._detail_logs,
                                verbose_steps=not q._detail_logs,
                                load_cache_memory=False,
                                event_time=task.event_time,
                                document_path=_document_path,
                                progress_callback=_on_progress,
                                control_callback=lambda _t=task: _t.control_action,
                                start_chunk=0,
                                target_window_indices=target_indices,
                                main_chunk_done_callback=_on_main_chunk_done,
                                step9_chunk_done_callback=_on_step9_chunk_done,
                                chunk_done_callback=_on_chunk_done,
                            )
                        finally:
                            q._clear_active_processor(task.task_id, _task_processor)

                    def _mark_task_running():
                        started_at = task.started_at or time.time()
                        task.chain_started_at = {}
                        task.chain_run_start_chunks = {
                            "main": int(_existing_main_chunks or 0),
                            "step9": int(_existing_step9_chunks or 0),
                            "step10": int(_existing_step10_chunks or 0),
                        }
                        _phase_label = _resume_hint
                        if _is_targeted_retry:
                            _phase_label = f"补跑 {len(_target_indices)} 个缺失/失败窗口"
                        q._update_task_progress(
                            task,
                            status="running",
                            phase="processing",
                            phase_label=_phase_label,
                            phase_current=_existing_step10_chunks,
                            phase_total=max(1, task.total_chunks),
                            main_done_chunks=_existing_main_chunks,
                            step9_done_chunks=_existing_step9_chunks,
                            step10_done_chunks=_existing_step10_chunks,
                            processed_chunks=_existing_step10_chunks,
                            total_chunks=task.total_chunks,
                            run_start_chunks=_existing_step10_chunks,
                            progress=_init_progress,
                            step9_progress=0.0,
                            step9_label="",
                            step10_progress=0.0,
                            step10_label="",
                            main_progress=0.0,
                            main_label="",
                            message=_resume_hint,
                            started_at=started_at,
                            finished_at=None,
                            error=None,
                        )
                        q._persist(task)
                        if _is_targeted_retry:
                            q._log_info(
                                "[Remember] 目标补跑: task_id=%s, source_name=%r, 窗口=%s"
                                % (short_task_id(task.task_id), task.source_name, _target_indices)
                            )
                        else:
                            q._log_info(
                                "[Remember] 开始处理: task_id=%s, source_name=%r, 文本长度=%d 字符, load_cache_memory=%s"
                                % (short_task_id(task.task_id), task.source_name, len(task.text), _uses_external_cache)
                            )

                    if _uses_external_cache:
                        with q._phase2_lock:
                            if task.status == "cancelled":
                                q._log_info(
                                    "[Remember] 跳过已删除任务: task_id=%s" % short_task_id(task.task_id)
                                )
                                break
                            if attempt == 0:
                                _mark_task_running()
                            result = _run_task()
                    else:
                        if task.status == "cancelled":
                            q._log_info(
                                "[Remember] 跳过已删除任务: task_id=%s" % short_task_id(task.task_id)
                            )
                            break
                        if attempt == 0:
                            _mark_task_running()
                        result = _run_task()

                    if task.control_action == "cancel" or task.status == "cancelled":
                        q._log_info(
                            "[Remember] 任务已删除，忽略后续完成状态: task_id=%s"
                            % short_task_id(task.task_id)
                        )
                        last_exc = None
                        break

                    # Warn on zero extractions (possible LLM issue)
                    if isinstance(result, dict):
                        entities = result.get("entities", 0)
                        relations = result.get("relations", 0)
                        if entities == 0 and relations == 0:
                            result["warning"] = (
                                "Extraction completed with 0 entities and 0 relations. "
                                "This may indicate an LLM connectivity issue — verify with health_check_llm."
                            )
                    result["original_path"] = task.original_path
                    finished_at = time.time()
                    _tc_done = max(1, int(task.total_chunks))
                    _cp_done = int(result.get("chunks_processed") or 0)
                    if _is_targeted_retry and int(result.get("failed_windows") or 0) == 0:
                        _cp_done = _tc_done
                    _cp_done = max(0, min(_cp_done, _tc_done))
                    _failed_windows = int(result.get("failed_windows") or 0) if isinstance(result, dict) else 0

                    # ── Targeted auto-retry for failed windows ──
                    if _failed_windows > 0 and task.max_retries > 0:
                        _failed_indices = result.get("failed_window_indices", [])
                        _failed_errors = result.get("failed_window_errors", [])
                        task.failed_window_indices = _failed_indices
                        task.failed_window_errors = _failed_errors
                        q._log_warn(
                            "[Remember] %d 窗口失败，启动自动重试: task_id=%s, windows=%s"
                            % (_failed_windows, short_task_id(task.task_id), _failed_indices)
                        )
                        _retry_success = False
                        _retry_delays = [5.0, 15.0, 45.0]
                        for _retry_round in range(task.max_retries):
                            task.retry_attempt = _retry_round + 1
                            _delay = _retry_delays[min(_retry_round, len(_retry_delays) - 1)]
                            q._update_task_progress(
                                task,
                                status="running",
                                phase="retrying",
                                phase_label=f"自动补跑失败窗口 (第{_retry_round + 1}/{task.max_retries}次)",
                                message=f"等待 {_delay:.0f}s 后补跑 {len(_failed_indices)} 个失败窗口...",
                            )
                            q._persist(task)
                            time.sleep(_delay)
                            if task.control_action == "cancel" or task.status == "cancelled":
                                break
                            try:
                                _retry_result = _run_task_with_targets(_failed_indices)
                                _new_failures = int(_retry_result.get("failed_windows") or 0)
                                if _new_failures == 0:
                                    _retry_success = True
                                    result["failed_windows"] = 0
                                    result["failed_window_indices"] = []
                                    result["chunks_processed"] = task.total_chunks
                                    result["entities"] = (result.get("entities") or 0) + (_retry_result.get("entities") or 0)
                                    result["relations"] = (result.get("relations") or 0) + (_retry_result.get("relations") or 0)
                                    task.failed_window_indices = []
                                    task.failed_window_errors = []
                                    q._log_info(
                                        "[Remember] 重试成功: task_id=%s, 第%d次" % (short_task_id(task.task_id), _retry_round + 1)
                                    )
                                    break
                                else:
                                    _failed_indices = _retry_result.get("failed_window_indices", [])
                                    task.failed_window_indices = _failed_indices
                                    task.failed_window_errors = _retry_result.get("failed_window_errors", [])
                                    q._log_warn(
                                        "[Remember] 重试仍有 %d 窗口失败: task_id=%s, round=%d"
                                        % (_new_failures, short_task_id(task.task_id), _retry_round + 1)
                                    )
                            except Exception as _retry_exc:
                                q._log_error(
                                    "[Remember] 重试异常: task_id=%s, round=%d, error=%s"
                                    % (short_task_id(task.task_id), _retry_round + 1, _retry_exc)
                                )

                        if not _retry_success and task.failed_window_indices:
                            _remaining = len(task.failed_window_indices)
                            q._update_task_progress(
                                task,
                                status="paused",
                                phase="paused",
                                phase_label=f"部分窗口未完整落库 (重试{task.retry_attempt}次后)",
                                phase_current=_cp_done,
                                phase_total=_tc_done,
                                main_done_chunks=max(int(task.main_done_chunks or 0), _cp_done),
                                step9_done_chunks=max(int(task.step9_done_chunks or 0), _cp_done),
                                step10_done_chunks=_cp_done,
                                processed_chunks=_cp_done,
                                progress=(_cp_done / _tc_done) if _tc_done > 0 else task.progress,
                                message="自动重试已耗尽，已暂停",
                                result=result,
                                error=f"有 {_remaining} 个窗口在 {task.max_retries} 次自动重试后仍失败，请手动重试或重新导入",
                                finished_at=finished_at,
                                step9_progress=task.step9_progress,
                                step10_progress=task.step10_progress,
                                main_progress=task.main_progress,
                            )
                            q._persist(task)
                            q._log_error(
                                "[Remember] 自动重试耗尽: task_id=%s, %d 窗口仍失败: %s"
                                % (short_task_id(task.task_id), _remaining, task.failed_window_indices)
                            )
                            last_exc = None
                            break

                    if task.control_action == "cancel" or task.status == "cancelled":
                        q._log_info(
                            "[Remember] 任务已删除，忽略后续完成状态: task_id=%s"
                            % short_task_id(task.task_id)
                        )
                        last_exc = None
                        break

                    if int(result.get("failed_windows") or 0) > 0:
                        # Failed windows remain after retry — already handled above
                        pass
                    else:
                        _cp_done = int(result.get("chunks_processed") or 0)
                        if _is_targeted_retry:
                            _cp_done = _tc_done
                        _cp_done = max(0, min(_cp_done, _tc_done))
                        task.failed_window_indices = []
                        task.failed_window_errors = []
                        task.repair_window_indices = []
                        task.repair_window_statuses = []
                        q._update_task_progress(
                        task,
                        status="completed",
                        phase="completed",
                        phase_label="已完成",
                        phase_current=_tc_done * 10,
                        phase_total=_tc_done * 10,
                        main_done_chunks=_cp_done,
                        step9_done_chunks=_cp_done,
                        step10_done_chunks=_cp_done,
                        processed_chunks=_cp_done,
                        progress=1.0,
                        message="处理完成",
                        result=result,
                        error=None,
                        finished_at=finished_at,
                        step9_progress=1.0,
                        step10_progress=1.0,
                        step9_label="",
                        step10_label="",
                        main_progress=1.0,
                        main_label="",
                    )
                    q._persist(task)
                    elapsed = (task.finished_at or 0) - (task.started_at or 0)
                    q._log_info(
                        "[Remember] 完成: task_id=%s, chunks_processed=%s, 耗时=%.1fs"
                        % (short_task_id(task.task_id), result.get("chunks_processed"), elapsed)
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    _control_action = getattr(exc, "remember_control_action", None)
                    if _control_action == "pause":
                        q._update_task_progress(
                            task,
                            status="paused",
                            phase="paused",
                            phase_label="已暂停",
                            progress=task.progress,
                            message="任务已暂停，可继续",
                            error=None,
                            finished_at=None,
                        )
                        task.control_action = None
                        q._persist(task)
                        q._log_info(
                            "[Remember] 已暂停: task_id=%s, source_name=%r"
                            % (short_task_id(task.task_id), task.source_name)
                        )
                        last_exc = None
                        break
                    if _control_action == "cancel":
                        q._update_task_progress(
                            task,
                            status="cancelled",
                            phase="cancelled",
                            phase_label="已删除",
                            progress=task.progress,
                            message="运行中任务已删除",
                            error=None,
                            finished_at=time.time(),
                        )
                        task.control_action = None
                        q._persist(task)
                        with q._lock:
                            q._tasks.pop(task.task_id, None)
                        q._log_info(
                            "[Remember] 已删除运行中任务: task_id=%s, source_name=%r"
                            % (short_task_id(task.task_id), task.source_name)
                        )
                        last_exc = None
                        break
                    last_exc = exc
                    if attempt < q._max_retries:
                        delay = q._retry_delay
                        q._update_task_progress(
                            task,
                            status="running",
                            phase=task.phase,
                            phase_label=task.phase_label,
                            progress=task.progress,
                            message="失败后重试中，第 %d 次，%ss 后继续" % (attempt + 1, delay),
                            error=str(exc),
                        )
                        q._persist(task)
                        q._log_warn(
                            "[Remember] 失败将重试: task_id=%s, attempt=%d, error=%r, %ss 后重试"
                            % (short_task_id(task.task_id), attempt + 1, exc, delay)
                        )
                        time.sleep(delay)
                    else:
                        q._update_task_progress(
                            task,
                            status="failed",
                            phase="failed",
                            phase_label="失败",
                            progress=task.progress,
                            message="处理失败",
                            error=str(exc),
                            finished_at=time.time(),
                        )
                        q._persist(task)
                        q._log_error(
                            "[Remember] 失败: task_id=%s, error=%r" % (short_task_id(task.task_id), exc)
                        )
        except Exception as exc:
            q._update_task_progress(
                task,
                status="failed",
                phase="failed",
                phase_label="失败",
                progress=task.progress,
                message="处理失败",
                error=str(exc),
                finished_at=time.time(),
            )
            q._persist(task)
            q._log_error("[Remember] 失败: task_id=%s, error=%r" % (short_task_id(task.task_id), exc))
        finally:
            # 任务结束（无论成功失败），清理入队时保存的临时原文
            if task.original_path and task.status in _TERMINAL_STATUSES:
                try:
                    p = Path(task.original_path)
                    if p.exists() and "originals" in p.parts:
                        p.unlink(missing_ok=True)
                except Exception as e:
                    logger.debug("清理原文文件失败: %s", e)
            q._queue.task_done()
