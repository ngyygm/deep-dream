"""
Remember 任务队列：异步记忆写入任务队列（串行滑窗处理）。

从 server/api.py 提取，消除循环依赖。

子模块：
- task_progress.py  进度计算纯函数
- task_journal.py   RememberTask 数据模型 & RememberJournal 持久化
- task_worker.py    worker 循环、历史修剪、磁盘恢复
"""
from __future__ import annotations

import logging
import queue as _queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.server.monitor import LOG_MODE_DETAIL
from core.log import info as _log_info_fn, warn as _log_warn_fn, error as _log_error_fn

# Re-export from sub-modules so that the public import paths stay unchanged:
#   from core.server.task_queue import RememberTaskQueue, RememberTask
from core.server.task_progress import (
    _TERMINAL_STATUSES,
    _DONE_STATUSES,
    estimate_chunk_count as _estimate_chunk_count,
    remember_callback_ui_fields as _remember_callback_ui_fields,
)
from core.server.task_journal import (
    RememberTask,
    RememberJournal,
    remember_task_from_record as _remember_task_from_record,
    short_task_id as _short_task_id,
)
from core.server.task_worker import (
    worker_loop as _worker_loop,
    trim_history as _trim_history,
    recover_from_disk as _recover_from_disk,
)

logger = logging.getLogger(__name__)


class RememberTaskQueue:
    """异步记忆写入任务队列。
    - load_cache_memory=True：接续图谱中已有缓存链，任务需串行执行。
    - load_cache_memory=False：从空起点开始，但任务内部滑窗仍续写本任务自己的 cache 链；
      若 max_workers > 1，可与其他独立任务并行。
    任务状态写入 tasks/，异常退出后重启会重新入队未完成任务（从 docs/ 原文重跑完整流水线）。"""

    def __init__(
        self,
        processor,
        storage_path: Path,
        *,
        processor_factory,
        max_workers: int = 1,
        max_history: int = 200,
        max_retries: int = 2,
        retry_delay_seconds: float = 2,
        event_log=None,
        stall_timeout_seconds: float = 600,
    ):
        self._processor = processor
        self._processor_factory = processor_factory
        self._journal = RememberJournal(storage_path)
        self._queue: "_queue.Queue[RememberTask]" = _queue.Queue()
        self._tasks: Dict[str, RememberTask] = {}
        self._active_processors: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._seq_counter = 0
        self._max_history = max_history
        self._max_retries = max(0, max_retries)
        self._retry_delay = max(0.0, retry_delay_seconds)
        self._phase2_lock = threading.Lock()
        self._stall_timeout = max(60.0, stall_timeout_seconds)
        self._workers: List[threading.Thread] = []
        self._event_log = event_log
        self._last_persist_ts: Dict[str, float] = {}  # debounce: task_id → last disk write timestamp
        self._detail_logs = event_log is not None and event_log.mode == LOG_MODE_DETAIL
        self._window_size = max(1, int(getattr(self._processor.document_processor, "window_size", 1000)))
        self._overlap = max(0, int(getattr(self._processor.document_processor, "overlap", 200)))
        _recover_from_disk(
            journal=self._journal,
            tasks=self._tasks,
            task_queue=self._queue,
            lock=self._lock,
            window_size=self._window_size,
            overlap=self._overlap,
            persist_fn=self._persist,
            log_info_fn=self._log_info,
        )
        for i in range(max(1, max_workers)):
            t = threading.Thread(target=_worker_loop, args=(self,), name=f"remember-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="remember-watchdog", daemon=True,
        )
        self._watchdog.start()

    def _watchdog_loop(self):
        """Periodically check for stalled running tasks and mark them failed."""
        while True:
            time.sleep(60)
            try:
                now = time.time()
                stalled_ids = []
                with self._lock:
                    for task in list(self._tasks.values()):
                        if task.status != "running":
                            continue
                        if now - task.last_update > self._stall_timeout:
                            stalled_ids.append(task.task_id)
                            task.status = "failed"
                            task.phase = "failed"
                            task.phase_label = "超时失败（看门狗）"
                            task.error = (
                                f"任务停滞超过 {self._stall_timeout:.0f}s "
                                f"无进度更新，看门狗自动标记失败"
                            )
                            task.finished_at = now
                            task.last_update = now
                            task.done_event.set()
                            self._persist(task)
                            self._tasks.pop(task.task_id, None)
                for tid in stalled_ids:
                    self._log_warn(
                        f"[Remember] 看门狗: 标记停滞任务失败: "
                        f"task_id={_short_task_id(tid)}"
                    )
            except Exception as e:
                logger.error("watchdog error: %s", e)

    def _task_uses_external_cache(self, task: RememberTask) -> bool:
        """None 表示沿用 processor 默认配置；False 时仅禁用外部链接续，不影响任务内部滑窗 cache 链。"""
        if task.load_cache is None:
            return bool(getattr(self._processor, "load_cache_memory", False))
        return bool(task.load_cache)

    def _log_info(self, message: str) -> None:
        _log_info_fn("Remember", message)

    def _log_warn(self, message: str) -> None:
        _log_warn_fn("Remember", message)

    def _log_error(self, message: str) -> None:
        _log_error_fn("Remember", message)

    def _update_task_progress(
        self,
        task: RememberTask,
        *,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        phase_label: Optional[str] = None,
        phase_current: Optional[int] = None,
        phase_total: Optional[int] = None,
        main_done_chunks: Optional[int] = None,
        step9_done_chunks: Optional[int] = None,
        step10_done_chunks: Optional[int] = None,
        processed_chunks: Optional[int] = None,
        total_chunks: Optional[int] = None,
        run_start_chunks: Optional[int] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        step9_progress: Optional[float] = None,
        step9_label: Optional[str] = None,
        step10_progress: Optional[float] = None,
        step10_label: Optional[str] = None,
        main_progress: Optional[float] = None,
        main_label: Optional[str] = None,
        started_at: Optional[float] = None,
        finished_at: Optional[float] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            if status is not None:
                task.status = status
            if phase is not None:
                task.phase = phase
            if phase_label is not None:
                task.phase_label = phase_label
            if phase_current is not None:
                task.phase_current = max(0, int(phase_current))
            if phase_total is not None:
                task.phase_total = max(0, int(phase_total))
            if main_done_chunks is not None:
                task.main_done_chunks = max(0, int(main_done_chunks))
            if step9_done_chunks is not None:
                task.step9_done_chunks = max(0, int(step9_done_chunks))
            if step10_done_chunks is not None:
                task.step10_done_chunks = max(0, int(step10_done_chunks))
            if processed_chunks is not None:
                task.processed_chunks = max(0, int(processed_chunks))
            if total_chunks is not None:
                task.total_chunks = max(0, int(total_chunks))
            if run_start_chunks is not None:
                task.run_start_chunks = max(0, int(run_start_chunks))
            if progress is not None:
                new_p = max(0.0, min(1.0, float(progress)))
                # 运行中回调可能乱序：总进度只增不减（完成/失败状态仍写入明确值）
                if status is not None and status != "running":
                    task.progress = new_p
                elif status == "running":
                    task.progress = max(task.progress, new_p)
                else:
                    if task.status == "running":
                        task.progress = max(task.progress, new_p)
                    else:
                        task.progress = new_p
            if message is not None:
                task.message = message
            if step9_progress is not None:
                new_s6 = max(0.0, min(1.0, float(step9_progress)))
                # 运行中回调可能乱序：进度只增不减（终态时写入明确值）
                if status is not None and status != "running":
                    task.step9_progress = new_s6
                elif task.status == "running":
                    task.step9_progress = max(task.step9_progress, new_s6)
                else:
                    task.step9_progress = new_s6
            if step9_label is not None:
                task.step9_label = step9_label
            if step10_progress is not None:
                new_s7 = max(0.0, min(1.0, float(step10_progress)))
                if status is not None and status != "running":
                    task.step10_progress = new_s7
                elif task.status == "running":
                    task.step10_progress = max(task.step10_progress, new_s7)
                else:
                    task.step10_progress = new_s7
            if step10_label is not None:
                task.step10_label = step10_label
            if main_progress is not None:
                new_m = max(0.0, min(1.0, float(main_progress)))
                if status is not None and status != "running":
                    task.main_progress = new_m
                elif task.status == "running":
                    task.main_progress = max(task.main_progress, new_m)
                else:
                    if task.status == "running":
                        task.main_progress = max(task.main_progress, new_m)
                    else:
                        task.main_progress = new_m
            if main_label is not None:
                task.main_label = main_label
            if started_at is not None:
                task.started_at = started_at
            if finished_at is not None:
                task.finished_at = finished_at
            if error is not None:
                task.error = error
            if result is not None:
                task.result = result
            task.last_update = time.time()
            # Signal synchronous waiters when task reaches terminal state
            if status in _DONE_STATUSES:
                task.done_event.set()

    def _task_to_dict(self, t: RememberTask) -> Dict[str, Any]:
        now = time.time()
        anchor = t.started_at or t.created_at or now
        return {
            "task_id": t.task_id,
            "task_seq": t.task_seq,
            "source_name": t.source_name,
            "load_cache_memory": t.load_cache,
            "status": t.status,
            "phase": t.phase,
            "phase_label": t.phase_label,
            "phase_current": t.phase_current,
            "phase_total": t.phase_total,
            "main_done_chunks": t.main_done_chunks,
            "step9_done_chunks": t.step9_done_chunks,
            "step10_done_chunks": t.step10_done_chunks,
            "processed_chunks": t.processed_chunks,
            "total_chunks": t.total_chunks,
            "run_start_chunks": t.run_start_chunks,
            "progress": t.progress,
            "message": t.message,
            "step9_progress": t.step9_progress,
            "step9_label": t.step9_label,
            "step10_progress": t.step10_progress,
            "step10_label": t.step10_label,
            "main_progress": t.main_progress,
            "main_label": t.main_label,
            "event_time": t.event_time.isoformat() if t.event_time else None,
            "created_at": t.created_at,
            "started_at": t.started_at,
            "finished_at": t.finished_at,
            "last_update": t.last_update,
            "error": t.error,
            "elapsed_seconds": max(0.0, (t.finished_at or now) - anchor),
        }

    # Minimum interval (seconds) between consecutive disk writes per task during progress updates.
    # Terminal state transitions always bypass this throttle.
    _PERSIST_DEBOUNCE_S = 2.0

    def _persist(self, task: RememberTask, *, _now: float = 0.0) -> None:
        """Debounced persist: skip write if last write for this task was <2s ago and not terminal."""
        tid = task.task_id
        if task.status not in _TERMINAL_STATUSES:
            now = _now or time.monotonic()
            last = self._last_persist_ts.get(tid, 0.0)
            if now - last < self._PERSIST_DEBOUNCE_S:
                return  # throttled — progress update will trigger another persist soon
            self._last_persist_ts[tid] = now
        else:
            # Terminal state: always write and clean up tracking
            self._last_persist_ts.pop(tid, None)
        try:
            self._journal.write(task)
        except Exception as e:
            self._log_warn("[Remember] journal 写入失败 task_id=%s: %s" % (_short_task_id(task.task_id), e))

    def submit(self, task: RememberTask) -> str:
        # 立即将原文保存到磁盘，确保崩溃重启后可恢复
        if task.text and not task.original_path:
            originals_dir = self._journal.dir / "originals"
            originals_dir.mkdir(parents=True, exist_ok=True)
            original_path = originals_dir / ("%s.txt" % task.task_id)
            try:
                original_path.write_text(task.text, encoding="utf-8")
                task.original_path = str(original_path)
            except Exception as e:
                self._log_warn("[Remember] 原文保存失败 task_id=%s: %s" % (_short_task_id(task.task_id), e))
        task.total_chunks = max(
            task.total_chunks,
            _estimate_chunk_count(len(task.text), self._window_size, self._overlap),
        )
        task.phase = "queued"
        task.phase_label = "等待处理"
        task.phase_current = 0
        task.phase_total = 0
        task.processed_chunks = 0
        task.progress = 0.0
        task.message = "已入队，预计 %d 个窗口" % task.total_chunks
        task.last_update = time.time()
        with self._lock:
            self._seq_counter += 1
            task.task_seq = self._seq_counter
            self._tasks[task.task_id] = task
            _trim_history(self._tasks, self._max_history, self._lock)
        self._persist(task)
        self._queue.put(task)
        self._log_info("[Remember] 任务入队: task_id=%s, source_name=%r" % (_short_task_id(task.task_id), task.source_name))
        return task.task_id

    def wait_for_task(self, task_id: str, timeout: float = 300) -> Optional[RememberTask]:
        """Block until a task reaches completed/failed state, or timeout expires.

        Returns the RememberTask (final or current state), or None if task_id not found.
        """
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            return None
        # Already done?
        if task.status in _DONE_STATUSES:
            return task
        task.done_event.wait(timeout=timeout)
        return task

    def _resolve_task_id(self, task_id_or_seq: str) -> Optional[str]:
        """Resolve a task_seq (e.g. '1', '2') or full task_id to a real task_id."""
        # Try as seq number first
        try:
            seq = int(task_id_or_seq)
            if seq > 0:
                with self._lock:
                    for t in self._tasks.values():
                        if t.task_seq == seq:
                            return t.task_id
                # Also check journal
                for rec in self._journal.iter_records():
                    if rec.get("task_seq") == seq:
                        return rec.get("task_id")
        except (ValueError, TypeError):
            pass
        # Fall back: treat as full task_id
        return task_id_or_seq

    def get_status(self, task_id: str) -> Optional[RememberTask]:
        resolved = self._resolve_task_id(task_id)
        with self._lock:
            t = self._tasks.get(resolved)
        if t is not None:
            return t
        rec = self._journal.read_record(resolved)
        if rec is None:
            return None
        text = ""
        op = rec.get("original_path")
        if op and Path(op).exists():
            try:
                text = Path(op).read_text(encoding="utf-8")
            except Exception as e:
                logger.debug("读取任务原文失败 %s: %s", op, e)
        return _remember_task_from_record(rec, text=text)

    def list_tasks(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            items = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
        out = []
        for t in items[:limit]:
            out.append(self._task_to_dict(t))
        return out

    def delete_pending_task(self, task_id: str) -> tuple[bool, str]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False, "任务不存在"
            if task.status != "queued":
                return False, "仅未开始运行的任务可以删除"
            task.status = "cancelled"
            task.phase = "cancelled"
            task.phase_label = "已删除"
            task.message = "任务已从队列删除"
            _now = time.time()
            task.finished_at = _now
            task.last_update = _now
            self._tasks.pop(task_id, None)

        removed_from_queue = False
        with self._queue.mutex:
            try:
                self._queue.queue.remove(task)
                removed_from_queue = True
                if self._queue.unfinished_tasks > 0:
                    self._queue.unfinished_tasks -= 1
                    if self._queue.unfinished_tasks == 0:
                        self._queue.all_tasks_done.notify_all()
                self._queue.not_full.notify()
            except ValueError:
                # 任务可能已被 worker 取走，但只要还没 running，后续也会被跳过。
                pass

        self._persist(task)
        if task.original_path:
            try:
                p = Path(task.original_path)
                if p.exists() and "originals" in p.parts:
                    p.unlink(missing_ok=True)
            except Exception as e:
                logger.debug("删除原文文件失败 %s: %s", task.original_path, e)

        detail = "（已从待处理队列移除）" if removed_from_queue else "（已标记删除，待 worker 跳过）"
        self._log_info(
            "[Remember] 删除待执行任务: task_id=%s, source_name=%r%s"
            % (_short_task_id(task_id), task.source_name, detail)
        )
        return True, "已删除"

    def request_pause_task(self, task_id_or_seq: str) -> tuple[bool, str, str]:
        task_id = self._resolve_task_id(task_id_or_seq)
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False, "任务不存在", "missing"
            if task.status == "paused":
                return False, "任务已暂停", "paused"
            if task.status != "running":
                return False, "仅运行中的任务可以暂停", task.status
            task.control_action = "pause"
            task.phase = "pausing"
            task.phase_label = "暂停中"
            task.message = "已收到暂停请求，将在当前安全点暂停"
            task.last_update = time.time()
        self._persist(task)
        self._log_info(
            "[Remember] 请求暂停任务: task_id=%s, source_name=%r"
            % (_short_task_id(task_id), task.source_name)
        )
        return True, "已请求暂停", "pausing"

    def resume_task(self, task_id_or_seq: str) -> tuple[bool, str, str]:
        task_id = self._resolve_task_id(task_id_or_seq)
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False, "任务不存在", "missing"
            if task.status != "paused":
                return False, "仅已暂停的任务可以继续", task.status
            task.control_action = None
            task.status = "queued"
            task.phase = "queued"
            task.phase_label = "恢复后等待处理"
            task.message = "已继续，等待工作线程开始"
            task.started_at = None
            task.finished_at = None
            task.last_update = time.time()
            self._queue.put(task)
        self._persist(task)
        self._log_info(
            "[Remember] 恢复暂停任务: task_id=%s, source_name=%r"
            % (_short_task_id(task_id), task.source_name)
        )
        return True, "已继续", "queued"

    def request_delete_task(self, task_id_or_seq: str) -> tuple[bool, str, str]:
        task_id = self._resolve_task_id(task_id_or_seq)
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False, "任务不存在", "missing"
            status = task.status
        if status == "queued":
            ok, message = self.delete_pending_task(task_id)
            return ok, message, "deleted" if ok else "queued"
        if status == "paused":
            with self._lock:
                task = self._tasks.get(task_id)
                if task is None:
                    return False, "任务不存在", "missing"
                task.status = "cancelled"
                task.phase = "cancelled"
                task.phase_label = "已删除"
                task.message = "暂停任务已删除"
                _now = time.time()
                task.finished_at = _now
                task.last_update = _now
                self._tasks.pop(task_id, None)
            self._persist(task)
            if task.original_path:
                try:
                    p = Path(task.original_path)
                    if p.exists() and "originals" in p.parts:
                        p.unlink(missing_ok=True)
                except Exception as e:
                    logger.debug("删除原文文件失败 %s: %s", task.original_path, e)
            self._log_info(
                "[Remember] 删除暂停任务: task_id=%s, source_name=%r"
                % (_short_task_id(task_id), task.source_name)
            )
            return True, "已删除", "deleted"
        if status != "running":
            return False, "仅排队中、运行中或已暂停的任务可以删除", status
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False, "任务不存在", "missing"
            task.control_action = "cancel"
            task.phase = "cancelling"
            task.phase_label = "删除中"
            task.message = "已收到删除请求，将在当前安全点停止并删除"
            task.last_update = time.time()
        self._persist(task)
        self._log_info(
            "[Remember] 请求删除运行中任务: task_id=%s, source_name=%r"
            % (_short_task_id(task_id), task.source_name)
        )
        return True, "已请求删除", "cancelling"

    def get_monitor_snapshot(self, limit: int = 6) -> Dict[str, Any]:
        with self._lock:
            items = list(self._tasks.values())
        queued: List[RememberTask] = []
        running: List[RememberTask] = []
        for t in items:
            if t.status == "queued":
                queued.append(t)
            elif t.status == "running":
                running.append(t)
        active = sorted(
            queued + running,
            key=lambda t: (0 if t.status == "running" else 1, t.created_at),
        )
        return {
            "queued_count": len(queued),
            "running_count": len(running),
            "backlog": self._queue.qsize(),
            "tracked_count": len(items),
            "active_tasks": [self._task_to_dict(t) for t in active[:limit]],
        }

    def _set_active_processor(self, task_id: str, processor: Any) -> None:
        with self._lock:
            self._active_processors[task_id] = processor

    def _clear_active_processor(self, task_id: str, processor: Optional[Any] = None) -> None:
        with self._lock:
            current = self._active_processors.get(task_id)
            if current is None:
                return
            if processor is None or current is processor:
                self._active_processors.pop(task_id, None)

    def get_runtime_stats_snapshot(self) -> Dict[str, int]:
        with self._lock:
            processors = list(self._active_processors.values())
        if not processors:
            processors = [self._processor]

        unique_processors = []
        seen_ids = set()
        for processor in processors:
            if processor is None:
                continue
            pid = id(processor)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            unique_processors.append(processor)

        totals = {
            "configured_window_workers": 0,
            "configured_llm_threads": 0,
            "active_window_extractions": 0,
            "active_main_pipeline_windows": 0,
            "peak_window_extractions": 0,
            "active_step9": 0,
            "active_step10": 0,
            "llm_semaphore_active": 0,
            "llm_semaphore_max": 0,
            "llm_upstream_active": 0,
            "llm_upstream_max": 0,
            "llm_downstream_active": 0,
            "llm_downstream_max": 0,
        }
        for processor in unique_processors:
            if not hasattr(processor, "get_runtime_stats"):
                continue
            try:
                stats = processor.get_runtime_stats() or {}
            except Exception as e:
                logger.debug("获取 processor runtime stats 失败: %s", e)
                continue
            for key in totals:
                totals[key] += int(stats.get(key, 0) or 0)
        return totals

    def get_pipeline_snapshot(self) -> Optional[Dict]:
        """返回当前正在运行的 remember 流水线逐窗口快照，无任务时返回 None。"""
        with self._lock:
            processors = list(self._active_processors.values())
        for processor in processors:
            if processor is None:
                continue
            if hasattr(processor, "get_pipeline_snapshot"):
                snap = processor.get_pipeline_snapshot()
                if snap is not None:
                    return snap
        if hasattr(self._processor, "get_pipeline_snapshot"):
            return self._processor.get_pipeline_snapshot()
        return None
