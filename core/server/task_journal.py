"""
Remember 任务数据模型与日志持久化。
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.server.task_progress import _TERMINAL_STATUSES, estimate_chunk_count

logger = logging.getLogger(__name__)


def short_task_id(task_id: str) -> str:
    return task_id[:8]


@dataclass(slots=True)
class RememberTask:
    task_id: str
    text: str
    source_name: str
    load_cache: Optional[bool]
    control_action: Optional[str]
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
    main_done_chunks: int = 0
    step9_done_chunks: int = 0
    step10_done_chunks: int = 0
    processed_chunks: int = 0
    total_chunks: int = 0
    run_start_chunks: int = 0      # 本轮开始时已有的 chunk 数（用于断点续传预估）
    task_seq: int = 0
    progress: float = 0.0
    message: str = "等待进入处理队列"
    step9_progress: float = 0.0
    step9_label: str = ""
    step10_progress: float = 0.0
    step10_label: str = ""
    main_progress: float = 0.0
    main_label: str = ""
    last_update: float = field(default_factory=time.time)
    done_event: threading.Event = field(default_factory=threading.Event)


def task_to_dict(task: RememberTask) -> Dict[str, Any]:
    """Serialize a RememberTask to a dict suitable for JSON persistence."""
    return {
        "task_id": task.task_id,
        "source_name": task.source_name,
        "original_path": task.original_path,
        "status": task.status,
        "event_time": task.event_time.isoformat() if task.event_time else None,
        "load_cache": task.load_cache,
        "control_action": task.control_action,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
        "error": task.error,
        "result": task.result,
        "phase": task.phase,
        "phase_label": task.phase_label,
        "phase_current": task.phase_current,
        "phase_total": task.phase_total,
        "main_done_chunks": task.main_done_chunks,
        "step9_done_chunks": task.step9_done_chunks,
        "step10_done_chunks": task.step10_done_chunks,
        "processed_chunks": task.processed_chunks,
        "total_chunks": task.total_chunks,
        "run_start_chunks": task.run_start_chunks,
        "task_seq": task.task_seq,
        "progress": task.progress,
        "message": task.message,
        "step9_progress": task.step9_progress,
        "step9_label": task.step9_label,
        "step10_progress": task.step10_progress,
        "step10_label": task.step10_label,
        "main_progress": task.main_progress,
        "main_label": task.main_label,
        "last_update": task.last_update,
    }


def remember_task_from_record(rec: Dict[str, Any], text: str) -> RememberTask:
    """Reconstruct a RememberTask from a journal record dict."""
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
        control_action=rec.get("control_action"),
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
        main_done_chunks=int(rec.get("main_done_chunks") or rec.get("processed_chunks") or 0),
        step9_done_chunks=int(rec.get("step9_done_chunks") or rec.get("processed_chunks") or 0),
        step10_done_chunks=int(rec.get("step10_done_chunks") or rec.get("processed_chunks") or 0),
        processed_chunks=int(rec.get("processed_chunks") or 0),
        total_chunks=int(rec.get("total_chunks") or 0),
        run_start_chunks=int(rec.get("run_start_chunks") or 0),
        task_seq=int(rec.get("task_seq") or 0),
        progress=float(rec.get("progress") or 0.0),
        message=str(rec.get("message") or "等待进入处理队列"),
        step9_progress=float(rec.get("step9_progress") or 0.0),
        step9_label=str(rec.get("step9_label") or ""),
        step10_progress=float(rec.get("step10_progress") or 0.0),
        step10_label=str(rec.get("step10_label") or ""),
        main_progress=float(rec.get("main_progress") or 0.0),
        main_label=str(rec.get("main_label") or ""),
        last_update=float(rec.get("last_update") or time.time()),
    )


class RememberJournal:
    """将 remember 任务落盘到 storage_path/tasks/queue.jsonl，单文件管理。
    - 活跃任务（queued/running）始终保留在文件中
    - 已完成/失败/已取消的任务在最终持久化后从文件中移除
    - 进程崩溃重启后从文件中恢复未完成任务
    """

    def __init__(self, storage_root: Path):
        self.dir = Path(storage_root) / "tasks"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._file = self.dir / "queue.jsonl"
        self._lock = threading.Lock()

    def write(self, task: RememberTask) -> None:
        """写入/更新任务：如果已完成/失败/已取消则从文件中移除，否则更新行。"""
        with self._lock:
            self._write_unlocked(task)

    def _write_unlocked(self, task: RememberTask) -> None:
        """内部方法，不加锁（由调用方保证线程安全）。"""
        d = task_to_dict(task)
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
                        except Exception as _json_err:
                            logger.debug("任务日志行 JSON 解析失败: %s", _json_err)
                            pass  # 保留无法解析的行（ corrupted JSON ）
                        lines.append(raw_line)
            except Exception as e:
                logger.warning("读取任务日志失败: %s", e)
                lines = []

        # 活跃任务写回，终态任务不写（从队列中移除）
        if task.status not in _TERMINAL_STATUSES:
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
                    except Exception as _json_err:
                        logger.debug("跳过损坏的 JSON 行: %s", _json_err)
                        continue
        except Exception as e:
            logger.debug("查找任务记录失败 %s: %s", task_id, e)
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
                    except Exception as _json_err:
                        logger.debug("跳过损坏的 JSON 行: %s", _json_err)
                        continue
        except Exception as e:
            logger.debug("遍历任务记录失败: %s", e)
        return out
