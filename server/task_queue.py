"""
Remember 任务队列：异步记忆写入任务队列（串行滑窗处理）。

从 server/api.py 提取，消除循环依赖。
"""
from __future__ import annotations

import json
import logging
import queue as _queue
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from server.monitor import LOG_MODE_DETAIL

logger = logging.getLogger(__name__)
def _estimate_chunk_count(text_length: int, window_size: int, overlap: int) -> int:
    if text_length <= 0:
        return 1
    stride = max(1, window_size - overlap)
    if text_length <= window_size:
        return 1
    return 1 + (max(text_length - window_size, 0) + stride - 1) // stride


_RE_WINDOW_STEP = re.compile(r"窗口\s*(\d+)/(\d+)\s*·\s*步骤(\d+)/(\d+)")
_RE_MAIN_1_5_DONE = re.compile(r"步骤\s*1\s*[–-]\s*5\s*/\s*7")


def _parse_window_phase_label(phase_label: str) -> Optional[tuple]:
    m = _RE_WINDOW_STEP.match(phase_label or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def _intra_in_window_slice(global_p: float, g_lo: float, g_hi: float) -> float:
    span = g_hi - g_lo
    if span <= 1e-15:
        return 0.0
    return max(0.0, min(1.0, (global_p - g_lo) / span))


def _intra_step6_step7(global_p: float, g_lo: float, g_hi: float, chain_id: str) -> float:
    """步骤6/7 各占单窗的 1/7（与 orchestrator 传入 extraction 的 progress_range 一致），链内 0–1。"""
    span = g_hi - g_lo
    if span <= 1e-15:
        return 0.0
    if chain_id == "step6":
        s_lo = g_lo + span * (5.0 / 7.0)
        s_hi = g_lo + span * (6.0 / 7.0)
    elif chain_id == "step7":
        s_lo = g_lo + span * (6.0 / 7.0)
        s_hi = g_hi
    else:
        return _intra_in_window_slice(global_p, g_lo, g_hi)
    ss = s_hi - s_lo
    if ss <= 1e-15:
        return 0.0
    return max(0.0, min(1.0, (global_p - s_lo) / ss))


def _wf_for_chain(chain_id: str, intra: float) -> float:
    """单窗内流水线权重：步骤2–5 占 5/7，步骤6 占 1/7，步骤7 占 1/7（与 extraction 中分窗一致）。"""
    intra = max(0.0, min(1.0, intra))
    if chain_id == "phase_ab":
        return (5.0 / 7.0) * intra
    if chain_id == "step6":
        return (5.0 / 7.0) + (1.0 / 7.0) * intra
    if chain_id == "step7":
        return (6.0 / 7.0) + (1.0 / 7.0) * intra
    return (5.0 / 7.0) * intra


def _wf_win_steps_1_5(global_p: float, g_lo: float, g_hi: float) -> float:
    """单窗内步骤1–5 占窗口宽度的前 5/7；返回 [0, 5/7] 的窗口内占比（相对整窗 0–1 的片段）。"""
    span = g_hi - g_lo
    if span <= 1e-15:
        return 0.0
    return max(0.0, min(5.0 / 7.0, (global_p - g_lo) / span))


def _overall_from_window_wf(win_cur: int, win_tot: int, wf: float) -> float:
    if win_tot <= 0:
        return 0.0
    wf = max(0.0, min(1.0, wf))
    return max(0.0, min(1.0, (win_cur - 1 + wf) / float(win_tot)))


def _overall_chain_from_window_intra(win_cur: int, win_tot: int, intra: float) -> float:
    """链级进度条位置：按窗口累计，当前窗口内按 intra 细分。"""
    if win_tot <= 0:
        return 0.0
    intra = max(0.0, min(1.0, intra))
    return max(0.0, min(1.0, (win_cur - 1 + intra) / float(win_tot)))


def _completed_chunk_fraction(done_chunks: int, total_chunks: int) -> float:
    if total_chunks <= 0:
        return 0.0
    done_chunks = max(0, min(int(done_chunks), int(total_chunks)))
    return done_chunks / float(total_chunks)


def _main_chain_anchor_rank(phase_label: str, tc: int) -> tuple:
    """主滑窗 1–5 的 UI 锚点优先级：越大越应作为展示锚点（抽取步骤 2–5 / 本窗 1–5 完成 优先于 步骤1 进行中）。

    并行时主线程可能在后序窗跑步骤1，而前序窗已在步骤2–5；此时应用「更靠前」的链上位置为锚点，而非总是跟主线程窗。
    """
    pl = (phase_label or "").strip()
    if not pl:
        return (-1, 0)
    if _RE_MAIN_1_5_DONE.search(pl) and ("已完成" in pl or "缓存" in pl):
        m = re.search(r"窗口\s*(\d+)/(\d+)", pl)
        w = int(m.group(1)) if m else 0
        if tc > 0:
            w = max(1, min(w, tc))
        return (6, w)
    parsed = _parse_window_phase_label(pl)
    if not parsed:
        return (-1, 0)
    win_cur, _wt, step_cur, _st = parsed
    if tc > 0:
        win_cur = max(1, min(win_cur, tc))
    if 2 <= step_cur <= 5:
        return (step_cur, win_cur)
    if step_cur != 1:
        return (min(step_cur, 6), win_cur)
    if "进行中" in pl:
        return (0, win_cur)
    if "完成" in pl:
        return (1, win_cur)
    return (0, win_cur)


def _remember_callback_ui_fields(
    task: RememberTask,
    progress: float,
    phase_label: str,
    message: str,
    chain_id: str,
) -> Dict[str, Any]:
    """推导总进度：主滑窗链 main（步骤1–5）、步骤6/7 链各自独立进度（0–1 为链内细粒度）。"""
    parsed = _parse_window_phase_label(phase_label)
    tc = max(1, int(task.total_chunks or 1))
    pc = max(0, int(task.processed_chunks or 0))
    pc_f = pc / float(tc)

    if not parsed:
        new_o = max(pc_f, max(0.0, min(1.0, float(progress))))
        pl = phase_label or ""
        if chain_id in ("main", "phase_ab") and _RE_MAIN_1_5_DONE.search(pl) and (
            "已完成" in pl or "缓存" in pl
        ):
            m = re.search(r"窗口\s*(\d+)/(\d+)", pl)
            if m:
                win_cur = max(1, min(int(m.group(1)), tc))
                wf_main = 5.0 / 7.0
                main_global = min(1.0, (win_cur - 1 + wf_main) / float(tc))
                merged_p = max(new_o, main_global, pc_f)
                new_rank = _main_chain_anchor_rank(pl, tc)
                old_rank = _main_chain_anchor_rank(task.main_label or "", tc)
                if task.main_label and new_rank < old_rank:
                    return {"progress": merged_p}
                _pc = (win_cur - 1) * 7 + 5
                _pt = tc * 7
                return {
                    "progress": merged_p,
                    "phase_label": phase_label,
                    "message": message,
                    "phase_current": _pc,
                    "phase_total": _pt,
                    "main_progress": main_global,
                    "main_label": phase_label or message or "",
                }
        return {
            "progress": new_o,
            "phase_label": phase_label,
            "message": message,
        }

    # 仅用标签解析「当前第几窗」；分母必须与 task.total_chunks 一致，否则与 orchestrator
    # 传入的 progress（按 total_chunks 切片的全局坐标）错位，实体/关系条会不按本窗比例显示。
    win_cur, _win_tot_label, step_cur, _step_tot = parsed
    win_cur = max(1, min(win_cur, tc))
    win_tot_eff = tc
    g_lo = (win_cur - 1) / float(win_tot_eff)
    g_hi = win_cur / float(win_tot_eff)
    if chain_id in ("step6", "step7"):
        intra = _intra_step6_step7(float(progress), g_lo, g_hi, chain_id)
    else:
        intra = _intra_in_window_slice(progress, g_lo, g_hi)
    wf = _wf_for_chain(chain_id, intra)
    new_o = _overall_from_window_wf(win_cur, win_tot_eff, wf)

    _pc = (win_cur - 1) * 7 + step_cur
    _pt = win_tot_eff * 7

    wf_main = _wf_win_steps_1_5(float(progress), g_lo, g_hi)
    main_global = min(1.0, (win_cur - 1 + wf_main) / float(win_tot_eff))

    base: Dict[str, Any] = {
        "progress": max(pc_f, new_o),
        "phase_label": phase_label,
        "message": message,
        "phase_current": _pc,
        "phase_total": _pt,
    }

    # 主滑窗（步骤1–5）：chain main 或历史 phase_ab（锚点优先抽取链上位置，而非主线程步骤1）
    if chain_id in ("main", "phase_ab"):
        base["progress"] = max(base["progress"], main_global)
        new_rank = _main_chain_anchor_rank(phase_label or "", tc)
        old_rank = _main_chain_anchor_rank(task.main_label or "", tc)
        merged_p = base["progress"]
        if task.main_label and new_rank < old_rank:
            return {"progress": merged_p}
        base.update(
            main_progress=main_global,
            main_label=phase_label or message or "",
        )
        return base
    if chain_id == "step7":
        step6_global = max(
            _completed_chunk_fraction(task.step6_done_chunks or 0, tc),
            _completed_chunk_fraction(win_cur, tc),
            float(getattr(task, "step6_progress", 0.0) or 0.0),
        )
        step7_global = max(
            _completed_chunk_fraction(task.step7_done_chunks or 0, tc),
            _overall_chain_from_window_intra(win_cur, win_tot_eff, intra),
        )
        base.update(
            step6_progress=step6_global,
            step7_progress=step7_global,
            step7_label=phase_label or message or "",
        )
        return base
    # step6：实体链进度按窗口累计；不要把关系链已完成进度清零。
    base.update(
        step6_progress=max(
            _completed_chunk_fraction(task.step6_done_chunks or 0, tc),
            _overall_chain_from_window_intra(win_cur, win_tot_eff, intra),
        ),
        step6_label=phase_label or message or "",
    )
    return base


def _short_task_id(task_id: str) -> str:
    return task_id[:8]


@dataclass
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
    step6_done_chunks: int = 0
    step7_done_chunks: int = 0
    processed_chunks: int = 0
    total_chunks: int = 0
    run_start_chunks: int = 0      # 本轮开始时已有的 chunk 数（用于断点续传预估）
    progress: float = 0.0
    message: str = "等待进入处理队列"
    step6_progress: float = 0.0
    step6_label: str = ""
    step7_progress: float = 0.0
    step7_label: str = ""
    main_progress: float = 0.0
    main_label: str = ""
    last_update: float = field(default_factory=time.time)
    done_event: threading.Event = field(default_factory=threading.Event)


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

    def _task_to_dict(self, task: RememberTask) -> Dict[str, Any]:
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
            "step6_done_chunks": task.step6_done_chunks,
            "step7_done_chunks": task.step7_done_chunks,
            "processed_chunks": task.processed_chunks,
            "total_chunks": task.total_chunks,
            "run_start_chunks": task.run_start_chunks,
            "progress": task.progress,
            "message": task.message,
            "step6_progress": task.step6_progress,
            "step6_label": task.step6_label,
            "step7_progress": task.step7_progress,
            "step7_label": task.step7_label,
            "main_progress": task.main_progress,
            "main_label": task.main_label,
            "last_update": task.last_update,
        }

    def write(self, task: RememberTask) -> None:
        """写入/更新任务：如果已完成/失败/已取消则从文件中移除，否则更新行。"""
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
                            pass  # 保留无法解析的行（ corrupted JSON ）
                        lines.append(raw_line)
            except Exception as e:
                logger.warning("读取任务日志失败: %s", e)
                lines = []

        # 活跃任务写回，终态任务不写（从队列中移除）
        if task.status not in ("completed", "failed", "cancelled"):
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
                        continue  # 跳过损坏的 JSON 行
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
                    except Exception:
                        continue  # 跳过损坏的 JSON 行
        except Exception as e:
            logger.debug("遍历任务记录失败: %s", e)
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
        step6_done_chunks=int(rec.get("step6_done_chunks") or rec.get("processed_chunks") or 0),
        step7_done_chunks=int(rec.get("step7_done_chunks") or rec.get("processed_chunks") or 0),
        processed_chunks=int(rec.get("processed_chunks") or 0),
        total_chunks=int(rec.get("total_chunks") or 0),
        run_start_chunks=int(rec.get("run_start_chunks") or 0),
        progress=float(rec.get("progress") or 0.0),
        message=str(rec.get("message") or "等待进入处理队列"),
        step6_progress=float(rec.get("step6_progress") or 0.0),
        step6_label=str(rec.get("step6_label") or ""),
        step7_progress=float(rec.get("step7_progress") or 0.0),
        step7_label=str(rec.get("step7_label") or ""),
        main_progress=float(rec.get("main_progress") or 0.0),
        main_label=str(rec.get("main_label") or ""),
        last_update=float(rec.get("last_update") or time.time()),
    )


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
    ):
        self._processor = processor
        self._processor_factory = processor_factory
        self._journal = RememberJournal(storage_path)
        self._queue: "_queue.Queue[RememberTask]" = _queue.Queue()
        self._tasks: Dict[str, RememberTask] = {}
        self._active_processors: Dict[str, Any] = {}
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

    def _task_uses_external_cache(self, task: RememberTask) -> bool:
        """None 表示沿用 processor 默认配置；False 时仅禁用外部链接续，不影响任务内部滑窗 cache 链。"""
        if task.load_cache is None:
            return bool(getattr(self._processor, "load_cache_memory", False))
        return bool(task.load_cache)

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
        main_done_chunks: Optional[int] = None,
        step6_done_chunks: Optional[int] = None,
        step7_done_chunks: Optional[int] = None,
        processed_chunks: Optional[int] = None,
        total_chunks: Optional[int] = None,
        run_start_chunks: Optional[int] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        step6_progress: Optional[float] = None,
        step6_label: Optional[str] = None,
        step7_progress: Optional[float] = None,
        step7_label: Optional[str] = None,
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
            if step6_done_chunks is not None:
                task.step6_done_chunks = max(0, int(step6_done_chunks))
            if step7_done_chunks is not None:
                task.step7_done_chunks = max(0, int(step7_done_chunks))
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
            if step6_progress is not None:
                task.step6_progress = max(0.0, min(1.0, float(step6_progress)))
            if step6_label is not None:
                task.step6_label = step6_label
            if step7_progress is not None:
                task.step7_progress = max(0.0, min(1.0, float(step7_progress)))
            if step7_label is not None:
                task.step7_label = step7_label
            if main_progress is not None:
                new_m = max(0.0, min(1.0, float(main_progress)))
                if status is not None and status != "running":
                    task.main_progress = new_m
                elif status == "running":
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
            if status in ("completed", "failed"):
                task.done_event.set()

    def _task_to_dict(self, t: RememberTask) -> Dict[str, Any]:
        now = time.time()
        anchor = t.started_at or t.created_at or now
        return {
            "task_id": t.task_id,
            "source_name": t.source_name,
            "load_cache_memory": t.load_cache,
            "status": t.status,
            "phase": t.phase,
            "phase_label": t.phase_label,
            "phase_current": t.phase_current,
            "phase_total": t.phase_total,
            "main_done_chunks": t.main_done_chunks,
            "step6_done_chunks": t.step6_done_chunks,
            "step7_done_chunks": t.step7_done_chunks,
            "processed_chunks": t.processed_chunks,
            "total_chunks": t.total_chunks,
            "progress": t.progress,
            "message": t.message,
            "step6_progress": t.step6_progress,
            "step6_label": t.step6_label,
            "step7_progress": t.step7_progress,
            "step7_label": t.step7_label,
            "main_progress": t.main_progress,
            "main_label": t.main_label,
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
            if st in ("completed", "failed", "cancelled"):
                continue
            if st == "paused":
                try:
                    text = ""
                    op = rec.get("original_path")
                    if op and Path(op).exists():
                        text = Path(op).read_text(encoding="utf-8")
                    task = _remember_task_from_record(rec, text=text)
                    task.status = "paused"
                    task.phase = "paused"
                    task.phase_label = "服务重启后保持暂停"
                    task.message = "任务在服务重启后保持暂停，可手动继续"
                    task.last_update = time.time()
                    with self._lock:
                        self._tasks[tid] = task
                    self._persist(task)
                except Exception as e:
                    logger.debug("恢复暂停任务 %s 失败: %s", tid, e)
                    continue
                continue
                op = rec.get("original_path")
                if not op or not Path(op).exists():
                    rec2 = dict(rec)
                    rec2["status"] = "failed"
                    rec2["error"] = "重启恢复失败：原始文本文件不存在"
                    rec2["finished_at"] = time.time()
                    try:
                        tdead = _remember_task_from_record(rec2, text="")
                        self._journal.write(tdead)
                    except Exception as e:
                        logger.debug("写入恢复失败记录 %s: %s", tid, e)
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
                task.total_chunks = max(
                    task.total_chunks,
                    _estimate_chunk_count(len(task.text), self._window_size, self._overlap),
                )
                # 三条链的断点分别恢复；processed_chunks 继续兼容为 step7 已完成窗口数。
                _tc = max(0, int(task.total_chunks or 0))
                _step7_done = min(_tc, max(0, int(task.step7_done_chunks or task.processed_chunks or 0)))
                _step6_done = min(_tc, max(_step7_done, int(task.step6_done_chunks or task.processed_chunks or 0)))
                _main_done = min(_tc, max(_step6_done, int(task.main_done_chunks or task.processed_chunks or 0)))
                task.main_done_chunks = _main_done
                task.step6_done_chunks = _step6_done
                task.step7_done_chunks = _step7_done
                task.processed_chunks = _step7_done
                # 根据关系链已完成窗口数恢复总进度
                if task.total_chunks > 0 and task.step7_done_chunks > 0:
                    task.progress = task.step7_done_chunks / task.total_chunks
                else:
                    task.progress = 0.0
                task.main_progress = (_main_done / task.total_chunks) if task.total_chunks > 0 else 0.0
                if task.step7_done_chunks > 0 or task.step6_done_chunks > 0 or task.main_done_chunks > 0:
                    task.message = (
                        "服务重启后已恢复入队（"
                        f"主链 {task.main_done_chunks}/{task.total_chunks} · "
                        f"实体 {task.step6_done_chunks}/{task.total_chunks} · "
                        f"关系 {task.step7_done_chunks}/{task.total_chunks}）"
                    )
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

    def wait_for_task(self, task_id: str, timeout: float = 300) -> Optional[RememberTask]:
        """Block until a task reaches completed/failed state, or timeout expires.

        Returns the RememberTask (final or current state), or None if task_id not found.
        """
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            return None
        # Already done?
        if task.status in ("completed", "failed"):
            return task
        task.done_event.wait(timeout=timeout)
        return task

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
            except Exception as e:
                logger.debug("读取任务原文失败 %s: %s", op, e)
        return _remember_task_from_record(rec, text=text)

    def list_tasks(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            items = sorted(self._tasks.values(), key=lambda t: t.created_at)
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
            task.finished_at = time.time()
            task.last_update = time.time()
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
            f"[Remember] 删除待执行任务: task_id={_short_task_id(task_id)}, "
            f"source_name={task.source_name!r}{detail}"
        )
        return True, "已删除"

    def request_pause_task(self, task_id: str) -> tuple[bool, str, str]:
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
            f"[Remember] 请求暂停任务: task_id={_short_task_id(task_id)}, "
            f"source_name={task.source_name!r}"
        )
        return True, "已请求暂停", "pausing"

    def resume_task(self, task_id: str) -> tuple[bool, str, str]:
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
            f"[Remember] 恢复暂停任务: task_id={_short_task_id(task_id)}, "
            f"source_name={task.source_name!r}"
        )
        return True, "已继续", "queued"

    def request_delete_task(self, task_id: str) -> tuple[bool, str, str]:
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
                task.finished_at = time.time()
                task.last_update = time.time()
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
                f"[Remember] 删除暂停任务: task_id={_short_task_id(task_id)}, "
                f"source_name={task.source_name!r}"
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
            f"[Remember] 请求删除运行中任务: task_id={_short_task_id(task_id)}, "
            f"source_name={task.source_name!r}"
        )
        return True, "已请求删除", "cancelling"

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
            "active_step6": 0,
            "active_step7": 0,
            "llm_semaphore_active": 0,
            "llm_semaphore_max": 0,
        }
        for processor in unique_processors:
            if not hasattr(processor, "get_runtime_stats"):
                continue
            try:
                stats = processor.get_runtime_stats() or {}
            except Exception as e:
                logger.debug("获取 processor %s runtime stats 失败: %s", gid, e)
                continue
            for key in totals:
                totals[key] += int(stats.get(key, 0) or 0)
        return totals

    def _worker(self):
        """串行执行滑窗处理：从数据库加载最新缓存续写，或从空缓存开始。"""
        while True:
            task = self._queue.get()
            try:
                if task.status == "cancelled":
                    self._log_info(f"[Remember] 跳过已删除任务: task_id={_short_task_id(task.task_id)}")
                    continue
                _existing_main_chunks = task.main_done_chunks or 0
                _existing_step6_chunks = task.step6_done_chunks or 0
                _existing_step7_chunks = task.step7_done_chunks or task.processed_chunks or 0
                _init_progress = _existing_step7_chunks / task.total_chunks if task.total_chunks > 0 else 0.0
                _resume_hint = (
                    "断点续传："
                    f"主链 {_existing_main_chunks}/{task.total_chunks} · "
                    f"实体 {_existing_step6_chunks}/{task.total_chunks} · "
                    f"关系 {_existing_step7_chunks}/{task.total_chunks}"
                    if (_existing_main_chunks > 0 or _existing_step6_chunks > 0 or _existing_step7_chunks > 0)
                    and task.total_chunks > 0
                    else ("断点续传" if (_existing_main_chunks > 0 or _existing_step6_chunks > 0 or _existing_step7_chunks > 0) else "开始处理")
                )
                _start_chunk = task.step7_done_chunks or task.processed_chunks or 0
                _uses_external_cache = self._task_uses_external_cache(task)
                _task_processor = self._processor if _uses_external_cache else self._processor_factory()
                self._update_task_progress(
                    task,
                    status="queued",
                    phase="waiting_cache_chain" if _uses_external_cache else "queued",
                    phase_label="等待前序缓存链" if _uses_external_cache else "等待开始",
                    phase_current=_existing_step7_chunks,
                    phase_total=max(1, task.total_chunks),
                    main_done_chunks=_existing_main_chunks,
                    step6_done_chunks=_existing_step6_chunks,
                    step7_done_chunks=_existing_step7_chunks,
                    processed_chunks=_existing_step7_chunks,
                    total_chunks=task.total_chunks,
                    run_start_chunks=_existing_step7_chunks,
                    progress=_init_progress,
                    # 勿用「下一窗编号」预填 step6/7，否则 Web 端会误以为已在做实体/关系对齐；
                    # 主滑窗进度由 chain main 回调写入 main_*；实体/关系链写入 step6_* / step7_*。
                    step6_progress=0.0,
                    step6_label="",
                    step7_progress=0.0,
                    step7_label="",
                    main_progress=0.0,
                    main_label="",
                    message=("等待前一个接续缓存链的任务完成后开始" if _uses_external_cache else "等待工作线程开始"),
                    started_at=None,
                    finished_at=None,
                    error=None,
                )
                self._persist(task)
                if _uses_external_cache and self._phase2_lock.locked():
                    self._log_info(
                        f"[Remember] 等待串行执行: task_id={_short_task_id(task.task_id)}, "
                        f"source_name={task.source_name!r}"
                    )

                last_exc = None
                for attempt in range(self._max_retries + 1):
                    try:
                        # 构建进度回调：将处理器的进度更新转发到任务跟踪
                        _task_ref = task  # 闭包引用

                        def _on_progress(progress: float, phase_label: str, message: str, chain_id: str = "step6", _t=_task_ref):
                            _fields = _remember_callback_ui_fields(
                                _t, progress, phase_label, message, chain_id,
                            )
                            self._update_task_progress(_t, **_fields)
                            self._persist(_t)

                        def _on_main_chunk_done(processed_count: int, _t=_task_ref):
                            _tc = max(1, int(_t.total_chunks or 1))
                            _pc = max(0, int(processed_count))
                            _pg = min(1.0, float(_pc) / float(_tc))
                            self._update_task_progress(
                                _t,
                                main_done_chunks=_pc,
                                main_progress=_pg,
                            )
                            self._persist(_t)

                        def _on_step6_chunk_done(processed_count: int, _t=_task_ref):
                            _pc = max(0, int(processed_count))
                            self._update_task_progress(
                                _t,
                                step6_done_chunks=_pc,
                            )
                            self._persist(_t)

                        def _on_chunk_done(processed_count: int, _t=_task_ref):
                            """窗口 step7 完成后更新 processed_chunks；总进度与已完成窗数一致（单调递增）。"""
                            _tc = max(1, int(_t.total_chunks or 1))
                            _pc = max(0, int(processed_count))
                            _pg = min(1.0, float(_pc) / float(_tc))
                            self._update_task_progress(
                                _t,
                                step7_done_chunks=_pc,
                                processed_chunks=_pc,
                                progress=_pg,
                            )
                            self._persist(_t)

                        def _run_task():
                            self._set_active_processor(task.task_id, _task_processor)
                            try:
                                return _task_processor.remember_text(
                                    text=task.text,
                                    source_document=task.source_name,
                                    verbose=self._detail_logs,
                                    verbose_steps=not self._detail_logs,
                                    load_cache_memory=_uses_external_cache,
                                    event_time=task.event_time,
                                    document_path=task.original_path,
                                    progress_callback=_on_progress,
                                    control_callback=lambda _t=task: _t.control_action,
                                    start_chunk=_start_chunk,
                                    main_chunk_done_callback=_on_main_chunk_done,
                                    step6_chunk_done_callback=_on_step6_chunk_done,
                                    chunk_done_callback=_on_chunk_done,
                                )
                            finally:
                                self._clear_active_processor(task.task_id, _task_processor)

                        def _mark_task_running():
                            started_at = task.started_at or time.time()
                            self._update_task_progress(
                                task,
                                status="running",
                                phase="processing",
                                phase_label=_resume_hint,
                                phase_current=_existing_step7_chunks,
                                phase_total=max(1, task.total_chunks),
                                main_done_chunks=_existing_main_chunks,
                                step6_done_chunks=_existing_step6_chunks,
                                step7_done_chunks=_existing_step7_chunks,
                                processed_chunks=_existing_step7_chunks,
                                total_chunks=task.total_chunks,
                                run_start_chunks=_existing_step7_chunks,
                                progress=_init_progress,
                                step6_progress=0.0,
                                step6_label="",
                                step7_progress=0.0,
                                step7_label="",
                                main_progress=0.0,
                                main_label="",
                                message=_resume_hint,
                                started_at=started_at,
                                finished_at=None,
                                error=None,
                            )
                            self._persist(task)
                            self._log_info(
                                f"[Remember] 开始处理: task_id={_short_task_id(task.task_id)}, "
                                f"source_name={task.source_name!r}, 文本长度={len(task.text)} 字符, "
                                f"load_cache_memory={_uses_external_cache}"
                            )

                        if _uses_external_cache:
                            with self._phase2_lock:
                                if task.status == "cancelled":
                                    self._log_info(
                                        f"[Remember] 跳过已删除任务: task_id={_short_task_id(task.task_id)}"
                                    )
                                    break
                                if attempt == 0:
                                    _mark_task_running()
                                result = _run_task()
                        else:
                            if task.status == "cancelled":
                                self._log_info(
                                    f"[Remember] 跳过已删除任务: task_id={_short_task_id(task.task_id)}"
                                )
                                break
                            if attempt == 0:
                                _mark_task_running()
                            result = _run_task()

                        result["original_path"] = task.original_path
                        finished_at = time.time()
                        _tc_done = max(1, int(task.total_chunks))
                        _cp_done = int(result.get("chunks_processed") or 0)
                        _cp_done = max(0, min(_cp_done, _tc_done))
                        self._update_task_progress(
                            task,
                            status="completed",
                            phase="completed",
                            phase_label="已完成",
                            phase_current=_tc_done * 7,
                            phase_total=_tc_done * 7,
                            main_done_chunks=_cp_done,
                            step6_done_chunks=_cp_done,
                            step7_done_chunks=_cp_done,
                            processed_chunks=_cp_done,
                            progress=1.0,
                            message="处理完成",
                            result=result,
                            finished_at=finished_at,
                            step6_progress=1.0,
                            step7_progress=1.0,
                            step6_label="",
                            step7_label="",
                            main_progress=1.0,
                            main_label="",
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
                        _control_action = getattr(exc, "remember_control_action", None)
                        if _control_action == "pause":
                            self._update_task_progress(
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
                            self._persist(task)
                            self._log_info(
                                f"[Remember] 已暂停: task_id={_short_task_id(task.task_id)}, "
                                f"source_name={task.source_name!r}"
                            )
                            last_exc = None
                            break
                        if _control_action == "cancel":
                            self._update_task_progress(
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
                            self._persist(task)
                            with self._lock:
                                self._tasks.pop(task.task_id, None)
                            self._log_info(
                                f"[Remember] 已删除运行中任务: task_id={_short_task_id(task.task_id)}, "
                                f"source_name={task.source_name!r}"
                            )
                            last_exc = None
                            break
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
                if task.original_path and task.status in ("completed", "failed", "cancelled"):
                    try:
                        p = Path(task.original_path)
                        if p.exists() and "originals" in p.parts:
                            p.unlink(missing_ok=True)
                    except Exception as e:
                        logger.debug("清理原文文件失败: %s", e)
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
