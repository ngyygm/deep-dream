"""
监控系统：与处理流程解耦，独立采集系统状态。

核心组件：
  - EventLog: 双模式日志（detail 直接打印 / monitor 环形缓冲）
  - AccessTracker: API 访问统计
  - GraphMonitor: 单图谱状态采集
  - SystemMonitor: 系统级监控中心（统一入口）
"""
from __future__ import annotations

import threading
import time
import logging
from collections import Counter, deque
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 日志模式常量
# ---------------------------------------------------------------------------

LOG_MODE_DETAIL = "detail"
LOG_MODE_MONITOR = "monitor"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


# ---------------------------------------------------------------------------
# EventLog — 双模式事件日志
# ---------------------------------------------------------------------------

class EventLog:
    """双模式日志：
    - detail 模式：直接 print 到终端（适合开发调试）
    - monitor 模式：存入环形缓冲区，供仪表盘读取（适合生产运行）
    """

    def __init__(self, mode: str = LOG_MODE_DETAIL, max_events: int = 200):
        self.mode = mode if mode in (LOG_MODE_DETAIL, LOG_MODE_MONITOR) else LOG_MODE_DETAIL
        self._events: Deque[dict] = deque(maxlen=max(10, max_events))
        self._lock = threading.Lock()

    def info(self, source: str, message: str) -> None:
        self._emit("INFO", source, message)

    def warn(self, source: str, message: str) -> None:
        self._emit("WARN", source, message)

    def error(self, source: str, message: str) -> None:
        self._emit("ERROR", source, message, stderr=True)

    def _emit(self, level: str, source: str, message: str, *, stderr: bool = False) -> None:
        ts = time.strftime("%H:%M:%S")
        text = f"[{ts}] [{source}] {message}"
        if self.mode == LOG_MODE_DETAIL:
            import sys
            print(text, file=sys.stderr if stderr else sys.stdout)
        with self._lock:
            self._events.appendleft({
                "time": ts,
                "level": level,
                "source": source,
                "message": message,
            })

    def get_recent(self, limit: int = 50, level: Optional[str] = None, source: Optional[str] = None) -> List[dict]:
        with self._lock:
            items = list(self._events)
        if level:
            level = level.upper()
            items = [e for e in items if e["level"] == level]
        if source:
            source_lower = source.lower()
            items = [e for e in items if e["source"].lower() == source_lower]
        return items[:limit]


# ---------------------------------------------------------------------------
# AccessTracker — API 访问统计
# ---------------------------------------------------------------------------

class AccessTracker:
    """记录每个 API 请求，提供聚合统计。线程安全。"""

    def __init__(self, max_history: int = 5000):
        self._records: Deque[dict] = deque(maxlen=max(100, max_history))
        self._lock = threading.Lock()

    def record(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        graph_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._records.append({
                "time": time.time(),
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
                "graph_id": graph_id,
            })

    def get_stats(self, since_seconds: float = 300) -> dict:
        now = time.time()
        cutoff = now - since_seconds
        with self._lock:
            records = [r for r in self._records if r["time"] >= cutoff]

        if not records:
            return {
                "total_requests": 0,
                "error_count": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "max_duration_ms": 0.0,
                "requests_per_minute": 0.0,
                "top_endpoints": [],
                "recent_errors": [],
                "period_seconds": since_seconds,
            }

        total = len(records)
        error_count = sum(1 for r in records if r["status_code"] >= 400)
        durations = [r["duration_ms"] for r in records]
        avg_duration = sum(durations) / total
        max_duration = max(durations)

        elapsed = max(1.0, now - records[0]["time"])
        rpm = total / (elapsed / 60.0)

        # Top endpoints by hit count
        path_counter = Counter(r["path"] for r in records)
        top_endpoints = [
            {"path": path, "count": count}
            for path, count in path_counter.most_common(10)
        ]

        # Recent errors (last 5)
        recent_errors = []
        for r in reversed(records):
            if r["status_code"] >= 400:
                recent_errors.append({
                    "time": time.strftime("%H:%M:%S", time.localtime(r["time"])),
                    "method": r["method"],
                    "path": r["path"],
                    "status_code": r["status_code"],
                })
                if len(recent_errors) >= 5:
                    break

        return {
            "total_requests": total,
            "error_count": error_count,
            "success_rate": round((total - error_count) / total * 100, 1),
            "avg_duration_ms": round(avg_duration, 2),
            "max_duration_ms": round(max_duration, 2),
            "requests_per_minute": round(rpm, 1),
            "top_endpoints": top_endpoints,
            "recent_errors": recent_errors,
            "period_seconds": since_seconds,
        }


# ---------------------------------------------------------------------------
# GraphMonitor — 单图谱监控
# ---------------------------------------------------------------------------

class GraphMonitor:
    """采集单个图谱的存储统计、队列状态、线程状态。"""

    def __init__(self, graph_id: str, processor, queue):
        self.graph_id = graph_id
        self._processor = processor
        self._queue = queue

    def snapshot(self) -> dict:
        """返回图谱快照。"""
        storage = self._collect_storage_stats()
        queue = self._collect_queue_stats()
        threads = self._collect_thread_stats()
        return {
            "graph_id": self.graph_id,
            "storage": storage,
            "queue": queue,
            "threads": threads,
        }

    def _collect_storage_stats(self) -> dict:
        try:
            # 优先使用 count 方法（O(1) 查询），避免加载全部实体/关系
            storage = self._processor.storage
            total_entities = storage.count_unique_entities()
            total_relations = storage.count_unique_relations()
            cache_json_dir = storage.cache_json_dir
            cache_dir = storage.cache_dir
            # 优先从 docs/ 新结构计数
            docs_meta_files = list(storage.docs_dir.glob("*/meta.json")) if storage.docs_dir.is_dir() else []
            if docs_meta_files:
                total_episodes = len(docs_meta_files)
            else:
                json_files = list(cache_json_dir.glob("*.json"))
                total_episodes = len(json_files) if json_files else len(list(cache_dir.glob("*.json")))
            return {
                "entities": total_entities,
                "relations": total_relations,
                "episodes": total_episodes,
            }
        except Exception as e:
            logger.warning("收集图谱统计失败: %s", e)
            return {"entities": 0, "relations": 0, "episodes": 0}

    def _collect_queue_stats(self) -> dict:
        try:
            snapshot = self._queue.get_monitor_snapshot(limit=10)
            return {
                "queued_count": snapshot["queued_count"],
                "running_count": snapshot["running_count"],
                "backlog": snapshot["backlog"],
                "tracked_count": snapshot["tracked_count"],
                "active_tasks": snapshot.get("active_tasks", []),
            }
        except Exception as e:
            logger.warning("收集队列统计失败: %s", e)
            return {
                "queued_count": 0,
                "running_count": 0,
                "backlog": 0,
                "tracked_count": 0,
                "active_tasks": [],
            }

    def _collect_thread_stats(self) -> dict:
        try:
            import threading as _threading
            threads = list(_threading.enumerate())
            names = [t.name for t in threads]
            processor_stats = {}
            if hasattr(self._queue, "get_runtime_stats_snapshot"):
                try:
                    processor_stats = self._queue.get_runtime_stats_snapshot()
                except Exception as e:
                    logger.debug("获取 queue runtime stats 失败: %s", e)
                    processor_stats = {}
            if not processor_stats and hasattr(self._processor, "get_runtime_stats"):
                try:
                    processor_stats = self._processor.get_runtime_stats()
                except Exception as e:
                    logger.debug("获取 processor runtime stats 失败: %s", e)
            remember_alive = sum(1 for n in names if n.startswith("remember-worker-"))
            window_alive = sum(1 for n in names if n.startswith("tmg-window"))
            llm_alive = sum(1 for n in names if n.startswith("tmg-llm"))
            queue_snapshot = self._queue.get_monitor_snapshot(limit=0)
            return {
                "python_threads_total": len(threads),
                "remember_worker_threads_alive": remember_alive,
                "remember_worker_threads_busy": queue_snapshot["running_count"],
                "window_threads_alive": window_alive,
                "window_threads_busy": int(
                    processor_stats.get("active_main_pipeline_windows", processor_stats.get("active_window_extractions", 0))
                ),
                "window_threads_peak": int(processor_stats.get("peak_window_extractions", 0)),
                "window_threads_configured": int(processor_stats.get("configured_window_workers", 0)),
                "step6_active": int(processor_stats.get("active_step6", 0)),
                "step7_active": int(processor_stats.get("active_step7", 0)),
                "llm_threads_alive": llm_alive,
                "llm_threads_busy": int(processor_stats.get("llm_semaphore_active", 0)),
                "llm_threads_max": int(processor_stats.get("llm_semaphore_max", 0)),
            }
        except Exception as e:
            logger.warning("收集线程统计失败: %s", e)
            return {
                "python_threads_total": 0,
                "remember_worker_threads_alive": 0,
                "remember_worker_threads_busy": 0,
                "window_threads_alive": 0,
                "window_threads_busy": 0,
                "window_threads_peak": 0,
                "window_threads_configured": 0,
                "step6_active": 0,
                "step7_active": 0,
                "llm_threads_alive": 0,
                "llm_threads_busy": 0,
                "llm_threads_max": 0,
            }


# ---------------------------------------------------------------------------
# SystemMonitor — 系统级监控中心
# ---------------------------------------------------------------------------

class SystemMonitor:
    """所有监控数据的统一入口，与处理流程完全解耦。"""

    def __init__(self, config: dict, mode: str = LOG_MODE_DETAIL):
        self.config = config
        self.event_log = EventLog(mode=mode)
        self.access_tracker = AccessTracker()
        self._graphs: Dict[str, GraphMonitor] = {}
        self._graph_order: List[str] = []
        self._lock = threading.Lock()
        self._start_time = time.time()

    @property
    def start_time(self) -> float:
        return self._start_time

    @property
    def mode(self) -> str:
        return self.event_log.mode

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def attach_graph(self, graph_id: str, processor, queue) -> None:
        """注册一个图谱到监控。"""
        with self._lock:
            if graph_id not in self._graphs:
                self._graphs[graph_id] = GraphMonitor(graph_id, processor, queue)
                self._graph_order.append(graph_id)

    def overview(self) -> dict:
        """系统总览。"""
        import threading
        with self._lock:
            graph_count = len(self._graphs)
        return {
            "graph_count": graph_count,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "uptime_display": _format_seconds(self.uptime_seconds),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._start_time)),
            "python_threads_total": len(threading.enumerate()),
            "mode": self.mode,
        }

    def all_graphs(self) -> List[dict]:
        """所有图谱摘要列表。"""
        with self._lock:
            graph_ids = list(self._graph_order)
        results = []
        for gid in graph_ids:
            with self._lock:
                gm = self._graphs.get(gid)
            if gm is None:
                continue
            snap = gm.snapshot()
            results.append({
                "graph_id": gid,
                "storage": snap["storage"],
                "queue": {
                    "queued_count": snap["queue"]["queued_count"],
                    "running_count": snap["queue"]["running_count"],
                    "backlog": snap["queue"]["backlog"],
                },
                "threads": snap["threads"],
            })
        return results

    def graph_detail(self, graph_id: str) -> Optional[dict]:
        """单图谱详细状态。"""
        with self._lock:
            gm = self._graphs.get(graph_id)
        if gm is None:
            return None
        return gm.snapshot()

    def all_tasks(self, limit: int = 50) -> List[dict]:
        """所有图谱的任务列表。"""
        with self._lock:
            graph_ids = list(self._graph_order)
        all_tasks = []
        for gid in graph_ids:
            with self._lock:
                gm = self._graphs.get(gid)
            if gm is None:
                continue
            try:
                tasks = gm._queue.list_tasks(limit=limit)
                for t in tasks:
                    t["graph_id"] = gid
                all_tasks.extend(tasks)
            except Exception as e:
                logger.debug("列图 %s 任务失败: %s", gid, e)
        # 按创建时间降序排列
        all_tasks.sort(key=lambda t: t.get("created_at", 0), reverse=True)
        return all_tasks[:limit]

    def recent_logs(self, limit: int = 50, level: Optional[str] = None, source: Optional[str] = None) -> List[dict]:
        """最近系统日志。"""
        return self.event_log.get_recent(limit=limit, level=level, source=source)

    def access_stats(self, since_seconds: float = 300) -> dict:
        """API 访问统计。"""
        return self.access_tracker.get_stats(since_seconds=since_seconds)

    def dashboard_snapshot(self, task_limit: int = 50, log_limit: int = 100,
                           log_level: Optional[str] = None, log_source: Optional[str] = None,
                           access_since: float = 300) -> dict:
        """一次采集仪表盘所需的全部数据，避免多次遍历图谱。"""
        import threading

        # 1. overview
        with self._lock:
            graph_ids = list(self._graph_order)

        overview = {
            "graph_count": len(graph_ids),
            "uptime_seconds": round(self.uptime_seconds, 1),
            "uptime_display": _format_seconds(self.uptime_seconds),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._start_time)),
            "python_threads_total": len(threading.enumerate()),
            "mode": self.mode,
        }

        # 2. graphs + tasks 一次遍历
        graphs = []
        all_tasks = []
        for gid in graph_ids:
            with self._lock:
                gm = self._graphs.get(gid)
            if gm is None:
                continue
            snap = gm.snapshot()
            graphs.append({
                "graph_id": gid,
                "storage": snap["storage"],
                "queue": {
                    "queued_count": snap["queue"]["queued_count"],
                    "running_count": snap["queue"]["running_count"],
                    "backlog": snap["queue"]["backlog"],
                },
                "threads": snap["threads"],
            })
            # 同时收集 tasks
            try:
                tasks = gm._queue.list_tasks(limit=task_limit)
                for t in tasks:
                    t["graph_id"] = gid
                all_tasks.extend(tasks)
            except Exception as e:
                logger.debug("列图 %s 任务失败(2): %s", gid, e)
        all_tasks.sort(key=lambda t: t.get("created_at", 0), reverse=True)

        # 3. logs
        logs = self.event_log.get_recent(limit=log_limit, level=log_level, source=log_source)

        # 4. access stats
        access = self.access_tracker.get_stats(since_seconds=access_since)

        return {
            "overview": overview,
            "graphs": graphs,
            "tasks": all_tasks[:task_limit],
            "logs": logs,
            "access_stats": access,
        }
