"""
Remember 任务队列：异步记忆写入任务队列（串行滑窗处理）。

从 server/api.py 提取，消除循环依赖。
"""
from __future__ import annotations

import json
import queue as _queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from server.monitor import LOG_MODE_DETAIL


def _estimate_chunk_count(text_length: int, window_size: int, overlap: int) -> int:
    if text_length <= 0:
        return 1
    stride = max(1, window_size - overlap)
    if text_length <= window_size:
        return 1
    return 1 + (max(text_length - window_size, 0) + stride - 1) // stride


def _short_task_id(task_id: str) -> str:
    return task_id[:8]


@dataclass
class RememberTask:
    task_id: str
    text: str
    source_name: str
    load_cache: Optional[bool]
    event_time: Optional[datetime]
    original_path: str
    status: str = "queued"          # queued | running | completed | failed
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    phase: str = "queued"
    phase_label: str = "等待处理"
    phase_current: int = 0
    phase_total: int = 0
    processed_chunks: int = 0
    total_chunks: int = 0
    run_start_chunks: int = 0      # 本轮开始时已有的 chunk 数（用于断点续传预估）
    progress: float = 0.0
    message: str = "等待进入处理队列"
    step6_progress: float = 0.0
    step6_label: str = ""
    step7_progress: float = 0.0
    step7_label: str = ""
    last_update: float = field(default_factory=time.time)


class RememberJournal:
    """将 remember 任务落盘到 storage_path/tasks/queue.jsonl，单文件管理。
    - 活跃任务（queued/running）始终保留在文件中
    - 已完成/失败的任务在最终持久化后从文件中移除
    - 进程崩溃重启后从文件中恢复未完成任务
    """

    def __init__(self, storage_root: Path):
        self.dir = Path(storage_root) / "tasks"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._file = self.dir / "queue.jsonl"
        self._lock = threading.Lock()

    def _task_to_dict(self, task: RememberTask) -> Dict[str, Any]:
        return {
            "task_id": task.task_id,
            "source_name": task.source_name,
            "original_path": task.original_path,
            "status": task.status,
            "event_time": task.event_time.isoformat() if task.event_time else None,
            "load_cache": task.load_cache,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "finished_at": task.finished_at,
            "error": task.error,
            "result": task.result,
            "phase": task.phase,
            "phase_label": task.phase_label,
            "phase_current": task.phase_current,
            "phase_total": task.phase_total,
            "processed_chunks": task.processed_chunks,
            "total_chunks": task.total_chunks,
            "run_start_chunks": task.run_start_chunks,
            "progress": task.progress,
            "message": task.message,
            "step6_progress": task.step6_progress,
            "step6_label": task.step6_label,
            "step7_progress": task.step7_progress,
            "step7_label": task.step7_label,
            "last_update": task.last_update,
        }

    def write(self, task: RememberTask) -> None:
        """写入/更新任务：如果已完成/失败则从文件中移除，否则更新行。"""
        with self._lock:
            self._write_unlocked(task)

    def _write_unlocked(self, task: RememberTask) -> None:
        """内部方法，不加锁（由调用方保证线程安全）。"""
        d = self._task_to_dict(task)
        line = json.dumps(d, ensure_ascii=False)
        tid = task.task_id

        # 读取现有内容，更新或移除该任务
        lines: List[str] = []
        if self._file.exists():
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    for raw_line in f:
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        try:
                            rec = json.loads(raw_line)
                            if rec.get("task_id") == tid:
                                continue  # 移除旧行
                        except Exception:
                            pass  # 保留无法解析的行
                        lines.append(raw_line)
            except Exception:
                lines = []

        # 活跃任务写回，终态任务不写（从队列中移除）
        if task.status not in ("completed", "failed"):
            lines.append(line)

        # 原子写入
        tmp = self._file.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
        tmp.replace(self._file)

    def read_record(self, task_id: str) -> Optional[Dict[str, Any]]:
        if not self._file.exists():
            return None
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        rec = json.loads(raw_line)
                        if rec.get("task_id") == task_id:
                            return rec
                    except Exception:
                        continue
        except Exception:
            pass
        return None

    def iter_records(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self._file.exists():
            return out
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        out.append(json.loads(raw_line))
                    except Exception:
                        continue
        except Exception:
            pass
        return out


def _remember_task_from_record(rec: Dict[str, Any], text: str) -> RememberTask:
    et_raw = rec.get("event_time")
    event_time: Optional[datetime] = None
    if et_raw:
        try:
            event_time = datetime.fromisoformat(str(et_raw).replace("Z", "+00:00"))
        except ValueError:
            event_time = None
    return RememberTask(
        task_id=str(rec["task_id"]),
        text=text,
        source_name=str(rec.get("source_name") or "api_input"),
        load_cache=rec.get("load_cache"),
        event_time=event_time,
        original_path=str(rec.get("original_path") or ""),
        status=str(rec.get("status") or "queued"),
        result=rec.get("result"),
        error=rec.get("error"),
        created_at=float(rec.get("created_at") or time.time()),
        started_at=rec.get("started_at"),
        finished_at=rec.get("finished_at"),
        phase=str(rec.get("phase") or "queued"),
        phase_label=str(rec.get("phase_label") or "等待处理"),
        phase_current=int(rec.get("phase_current") or 0),
        phase_total=int(rec.get("phase_total") or 0),
        processed_chunks=int(rec.get("processed_chunks") or 0),
        total_chunks=int(rec.get("total_chunks") or 0),
        run_start_chunks=int(rec.get("run_start_chunks") or 0),
        progress=float(rec.get("progress") or 0.0),
        message=str(rec.get("message") or "等待进入处理队列"),
        step6_progress=float(rec.get("step6_progress") or 0.0),
        step6_label=str(rec.get("step6_label") or ""),
        step7_progress=float(rec.get("step7_progress") or 0.0),
        step7_label=str(rec.get("step7_label") or ""),
        last_update=float(rec.get("last_update") or time.time()),
    )


class RememberTaskQueue:
    """异步记忆写入任务队列（串行滑窗处理）。
    任务按入队顺序串行执行，保证 memory_cache 链时序一致。
    可通过 load_cache_memory 从数据库加载最新缓存续写，或从空缓存开始。
    任务状态写入 tasks/，异常退出后重启会重新入队未完成任务（从 docs/ 原文重跑完整流水线）。"""

    def __init__(
        self,
        processor,
        storage_path: Path,
        max_workers: int = 1,
        max_history: int = 200,
        max_retries: int = 2,
        retry_delay_seconds: float = 2,
        event_log=None,
    ):
        self._processor = processor
        self._journal = RememberJournal(storage_path)
        self._queue: "_queue.Queue[RememberTask]" = _queue.Queue()
        self._tasks: Dict[str, RememberTask] = {}
        self._lock = threading.Lock()
        self._max_history = max_history
        self._max_retries = max(0, max_retries)
        self._retry_delay = max(0.0, retry_delay_seconds)
        self._phase2_lock = threading.Lock()
        self._workers: List[threading.Thread] = []
        self._event_log = event_log
        self._detail_logs = event_log is not None and event_log.mode == LOG_MODE_DETAIL
        self._window_size = max(1, int(getattr(self._processor.document_processor, "window_size", 1000)))
        self._overlap = max(0, int(getattr(self._processor.document_processor, "overlap", 200)))
        self._recover_from_disk()
        for i in range(max(1, max_workers)):
            t = threading.Thread(target=self._worker, name=f"remember-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def _log_info(self, message: str) -> None:
        if self._event_log is not None:
            self._event_log.info("Remember", message)
        else:
            print(message)

    def _log_warn(self, message: str) -> None:
        if self._event_log is not None:
            self._event_log.warn("Remember", message)
        else:
            print(f"[WARN] {message}")

    def _log_error(self, message: str) -> None:
        if self._event_log is not None:
            self._event_log.error("Remember", message)
        else:
            print(f"[ERROR] {message}", file=sys.stderr)

    def _update_task_progress(
        self,
        task: RememberTask,
        *,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        phase_label: Optional[str] = None,
        phase_current: Optional[int] = None,
        phase_total: Optional[int] = None,
        processed_chunks: Optional[int] = None,
        total_chunks: Optional[int] = None,
        run_start_chunks: Optional[int] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        step6_progress: Optional[float] = None,
        step6_label: Optional[str] = None,
        step7_progress: Optional[float] = None,
        step7_label: Optional[str] = None,
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
            if processed_chunks is not None:
                task.processed_chunks = max(0, int(processed_chunks))
            if total_chunks is not None:
                task.total_chunks = max(0, int(total_chunks))
            if run_start_chunks is not None:
                task.run_start_chunks = max(0, int(run_start_chunks))
            if progress is not None:
                task.progress = max(0.0, min(1.0, float(progress)))
            if message is not None:
                task.message = message
            if step6_progress is not None:
                task.step6_progress = max(0.0, min(1.0, float(step6_progress)))
            if step6_label is not None:
                task.step6_label = step6_label
            if step7_progress is not None:
                task.step7_progress = max(0.0, min(1.0, float(step7_progress)))
            if step7_label is not None:
                task.step7_label = step7_label
            if started_at is not None:
                task.started_at = started_at
            if finished_at is not None:
                task.finished_at = finished_at
            if error is not None:
                task.error = error
            if result is not None:
                task.result = result
            task.last_update = time.time()

    def _task_to_dict(self, t: RememberTask) -> Dict[str, Any]:
        now = time.time()
        anchor = t.started_at or t.created_at or now
        return {
            "task_id": t.task_id,
            "source_name": t.source_name,
            "status": t.status,
            "phase": t.phase,
            "phase_label": t.phase_label,
            "phase_current": t.phase_current,
            "phase_total": t.phase_total,
            "processed_chunks": t.processed_chunks,
            "total_chunks": t.total_chunks,
            "progress": t.progress,
            "message": t.message,
            "step6_progress": t.step6_progress,
            "step6_label": t.step6_label,
            "step7_progress": t.step7_progress,
            "step7_label": t.step7_label,
            "created_at": t.created_at,
            "started_at": t.started_at,
            "finished_at": t.finished_at,
            "last_update": t.last_update,
            "error": t.error,
            "elapsed_seconds": max(0.0, (t.finished_at or now) - anchor),
        }

    def _persist(self, task: RememberTask) -> None:
        try:
            self._journal.write(task)
        except Exception as e:
            self._log_warn(f"[Remember] journal 写入失败 task_id={_short_task_id(task.task_id)}: {e}")

    def _recover_from_disk(self) -> None:
        n_resume = 0
        records = self._journal.iter_records()
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
            if st in ("completed", "failed"):
                continue
            if st in ("queued", "running"):
                op = rec.get("original_path")
                if not op or not Path(op).exists():
                    rec2 = dict(rec)
                    rec2["status"] = "failed"
                    rec2["error"] = "重启恢复失败：原始文本文件不存在"
                    rec2["finished_at"] = time.time()
                    try:
                        tdead = _remember_task_from_record(rec2, text="")
                        self._journal.write(tdead)
                    except Exception:
                        pass
                    self._log_warn(f"[Remember] 恢复跳过 task_id={_short_task_id(str(tid))}: 原文缺失")
                    continue
                try:
                    text = Path(op).read_text(encoding="utf-8")
                except Exception as e:
                    rec2 = dict(rec)
                    rec2["status"] = "failed"
                    rec2["error"] = f"重启恢复失败：无法读取原文: {e}"
                    rec2["finished_at"] = time.time()
                    try:
                        tdead = _remember_task_from_record(rec2, text="")
                        self._journal.write(tdead)
                    except Exception:
                        pass
                    continue
                task = _remember_task_from_record(rec, text=text)
                task.status = "queued"
                task.started_at = None
                task.finished_at = None
                task.error = None
                task.result = None
                task.phase = "queued"
                task.phase_label = "恢复后等待处理"
                task.phase_current = 0
                task.phase_total = 0
                # 保留 processed_chunks（已完成的窗口数），不再重置为 0
                task.total_chunks = max(
                    task.total_chunks,
                    _estimate_chunk_count(len(task.text), self._window_size, self._overlap),
                )
                # 断点续传安全策略：回退一个窗口，宁可多跑一个也不漏
                if task.processed_chunks > 0:
                    original = task.processed_chunks
                    task.processed_chunks = max(0, task.processed_chunks - 1)
                    self._log_warn(
                        f"[Remember] task_id={_short_task_id(str(tid))}: "
                        f"断点续传回退: {original} → {task.processed_chunks}"
                    )
                # 根据已完成窗口数恢复进度
                if task.total_chunks > 0 and task.processed_chunks > 0:
                    task.progress = task.processed_chunks / task.total_chunks
                else:
                    task.progress = 0.0
                if task.processed_chunks > 0:
                    task.message = f"服务重启后已恢复入队（已完成 {task.processed_chunks}/{task.total_chunks} 窗口）"
                else:
                    task.message = "服务重启后已恢复入队"
                task.last_update = time.time()
                with self._lock:
                    self._tasks[tid] = task
                self._queue.put(task)
                self._persist(task)
                n_resume += 1
                self._log_info(
                    f"[Remember] 恢复未完成任务并入队: task_id={_short_task_id(tid)}, "
                    f"source_name={task.source_name!r}"
                )
        if n_resume:
            self._log_info(
                f"[Remember] 启动恢复：重新入队 {n_resume} 个未完成任务"
                "（已完成/失败仅保留在 journal，按需通过 status 查询）"
            )

    def submit(self, task: RememberTask) -> str:
        # 立即将原文保存到磁盘，确保崩溃重启后可恢复
        if task.text and not task.original_path:
            originals_dir = self._journal.dir / "originals"
            originals_dir.mkdir(parents=True, exist_ok=True)
            original_path = originals_dir / f"{task.task_id}.txt"
            try:
                original_path.write_text(task.text, encoding="utf-8")
                task.original_path = str(original_path)
            except Exception as e:
                self._log_warn(f"[Remember] 原文保存失败 task_id={_short_task_id(task.task_id)}: {e}")
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
        task.message = f"已入队，预计 {task.total_chunks} 个窗口"
        task.last_update = time.time()
        with self._lock:
            self._tasks[task.task_id] = task
            self._trim_history()
        self._persist(task)
        self._queue.put(task)
        self._log_info(f"[Remember] 任务入队: task_id={_short_task_id(task.task_id)}, source_name={task.source_name!r}")
        return task.task_id

    def get_status(self, task_id: str) -> Optional[RememberTask]:
        with self._lock:
            t = self._tasks.get(task_id)
        if t is not None:
            return t
        rec = self._journal.read_record(task_id)
        if rec is None:
            return None
        text = ""
        op = rec.get("original_path")
        if op and Path(op).exists():
            try:
                text = Path(op).read_text(encoding="utf-8")
            except Exception:
                pass
        return _remember_task_from_record(rec, text=text)

    def list_tasks(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            items = sorted(self._tasks.values(), key=lambda t: t.created_at)
        out = []
        for t in items[:limit]:
            out.append(self._task_to_dict(t))
        return out

    def get_monitor_snapshot(self, limit: int = 6) -> Dict[str, Any]:
        with self._lock:
            items = list(self._tasks.values())
        queued = [t for t in items if t.status == "queued"]
        running = [t for t in items if t.status == "running"]
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

    def _worker(self):
        """串行执行滑窗处理：从数据库加载最新缓存续写，或从空缓存开始。"""
        while True:
            task = self._queue.get()
            try:
                started_at = time.time()
                # 保留断点续传的 processed_chunks（恢复任务时已从磁盘加载）
                _existing_chunks = task.processed_chunks or 0
                _init_progress = _existing_chunks / task.total_chunks if task.total_chunks > 0 else 0.0
                _init_label = f"窗口 {_existing_chunks + 1}/{task.total_chunks}" if task.total_chunks > 0 else ""
                self._update_task_progress(
                    task,
                    status="running",
                    phase="processing",
                    phase_label="滑窗处理中",
                    phase_current=_existing_chunks,
                    phase_total=max(1, task.total_chunks),
                    processed_chunks=_existing_chunks,
                    total_chunks=task.total_chunks,
                    run_start_chunks=_existing_chunks,
                    progress=_init_progress,
                    step6_progress=_init_progress,
                    step6_label=_init_label if _existing_chunks > 0 else "",
                    step7_progress=_init_progress,
                    step7_label=_init_label if _existing_chunks > 0 else "",
                    message="断点续传" if _existing_chunks > 0 else "开始处理",
                    started_at=started_at,
                    finished_at=None,
                    error=None,
                )
                self._persist(task)
                self._log_info(
                    f"[Remember] 开始处理: task_id={_short_task_id(task.task_id)}, "
                    f"source_name={task.source_name!r}, 文本长度={len(task.text)} 字符"
                )

                last_exc = None
                for attempt in range(self._max_retries + 1):
                    try:
                        # 构建进度回调：将处理器的进度更新转发到任务跟踪
                        _task_ref = task  # 闭包引用

                        def _on_progress(progress: float, phase_label: str, message: str, chain_id: str = "step6", _t=_task_ref):
                            if chain_id == "phase_ab":
                                # Phase A+B 进度只更新总体 phase_label 和 message，不写入双链字段
                                import re as _re
                                _m = _re.match(r'窗口\s*(\d+)/(\d+)\s*·\s*步骤(\d+)/(\d+)', phase_label or '')
                                if _m:
                                    _win_cur, _win_tot = int(_m.group(1)), int(_m.group(2))
                                    _step_cur = int(_m.group(3))
                                    _pc = (_win_cur - 1) * 7 + _step_cur
                                    _pt = _win_tot * 7
                                else:
                                    _pc, _pt = None, None
                                self._update_task_progress(
                                    _t,
                                    phase_label=phase_label,
                                    message=message,
                                    phase_current=_pc, phase_total=_pt,
                                )
                                self._persist(_t)
                                return
                            # 从 phase_label 解析窗口/步骤编号
                            import re as _re
                            _m = _re.match(r'窗口\s*(\d+)/(\d+)\s*·\s*步骤(\d+)/(\d+)', phase_label or '')
                            if _m:
                                _win_cur, _win_tot = int(_m.group(1)), int(_m.group(2))
                                _step_cur = int(_m.group(3))
                                _pc = (_win_cur - 1) * 7 + _step_cur
                                _pt = _win_tot * 7
                            else:
                                _pc, _pt = None, None
                            if chain_id == "step7":
                                self._update_task_progress(
                                    _t,
                                    step7_progress=progress, step7_label=phase_label,
                                    message=message,
                                    phase_current=_pc, phase_total=_pt,
                                )
                            else:
                                self._update_task_progress(
                                    _t,
                                    step6_progress=progress, step6_label=phase_label,
                                    message=message,
                                    phase_current=_pc, phase_total=_pt,
                                )
                            self._persist(_t)

                        def _on_chunk_done(processed_count: int, _t=_task_ref):
                            """窗口 step6+step7 完成后更新 processed_chunks 并持久化。"""
                            self._update_task_progress(_t, processed_chunks=processed_count)
                            self._persist(_t)

                        _start_chunk = task.processed_chunks or 0

                        with self._phase2_lock:
                            result = self._processor.remember_text(
                                text=task.text,
                                source_document=task.source_name,
                                verbose=self._detail_logs,
                                load_cache_memory=task.load_cache,
                                event_time=task.event_time,
                                document_path=task.original_path,
                                progress_callback=_on_progress,
                                start_chunk=_start_chunk,
                                chunk_done_callback=_on_chunk_done,
                            )

                        result["original_path"] = task.original_path
                        finished_at = time.time()
                        self._update_task_progress(
                            task,
                            status="completed",
                            phase="completed",
                            phase_label="已完成",
                            phase_current=max(1, int(result.get("chunks_processed") or task.total_chunks)),
                            phase_total=max(1, task.total_chunks),
                            processed_chunks=max(1, int(result.get("chunks_processed") or task.total_chunks)),
                            progress=1.0,
                            message="处理完成",
                            result=result,
                            finished_at=finished_at,
                        )
                        self._persist(task)
                        elapsed = (task.finished_at or 0) - (task.started_at or 0)
                        self._log_info(
                            f"[Remember] 完成: task_id={_short_task_id(task.task_id)}, "
                            f"chunks_processed={result.get('chunks_processed')}, 耗时={elapsed:.1f}s"
                        )
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        if attempt < self._max_retries:
                            delay = self._retry_delay
                            self._update_task_progress(
                                task,
                                status="running",
                                phase=task.phase,
                                phase_label=task.phase_label,
                                progress=task.progress,
                                message=f"失败后重试中，第 {attempt + 1} 次，{delay}s 后继续",
                                error=str(exc),
                            )
                            self._persist(task)
                            self._log_warn(
                                f"[Remember] 失败将重试: task_id={_short_task_id(task.task_id)}, "
                                f"attempt={attempt + 1}, error={exc!r}, {delay}s 后重试"
                            )
                            time.sleep(delay)
                        else:
                            self._update_task_progress(
                                task,
                                status="failed",
                                phase="failed",
                                phase_label="失败",
                                progress=task.progress,
                                message="处理失败",
                                error=str(exc),
                                finished_at=time.time(),
                            )
                            self._persist(task)
                            self._log_error(
                                f"[Remember] 失败: task_id={_short_task_id(task.task_id)}, error={exc!r}"
                            )
            except Exception as exc:
                self._update_task_progress(
                    task,
                    status="failed",
                    phase="failed",
                    phase_label="失败",
                    progress=task.progress,
                    message="处理失败",
                    error=str(exc),
                    finished_at=time.time(),
                )
                self._persist(task)
                self._log_error(f"[Remember] 失败: task_id={_short_task_id(task.task_id)}, error={exc!r}")
            finally:
                # 任务结束（无论成功失败），清理入队时保存的临时原文
                if task.original_path and task.status in ("completed", "failed"):
                    try:
                        p = Path(task.original_path)
                        if p.exists() and "originals" in p.parts:
                            p.unlink(missing_ok=True)
                    except Exception:
                        pass
                self._queue.task_done()

    def _trim_history(self):
        if len(self._tasks) <= self._max_history:
            return
        items = sorted(self._tasks.values(), key=lambda t: t.created_at)
        to_remove = len(self._tasks) - self._max_history
        removed = 0
        for t in items:
            if t.status in ("completed", "failed") and removed < to_remove:
                del self._tasks[t.task_id]
                removed += 1
