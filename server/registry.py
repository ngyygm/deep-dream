"""
多图谱注册表：按 graph_id 管理多个 Processor + Queue 实例。
每个 graph_id 对应 {storage_path}/{graph_id}/ 下的独立数据库。
"""
from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from server.config import merge_llm_alignment, resolve_embedding_model  # noqa: F401
from processor.storage.embedding import EmbeddingClient
from processor.storage import create_storage_manager
from processor.pipeline.orchestrator import TemporalMemoryGraphProcessor
from processor.dream import DreamOrchestrator, DreamConfig

if TYPE_CHECKING:
    from server.monitor import SystemMonitor

# 图谱 ID 只允许：字母、数字、下划线、连字符（禁止路径穿越）
_GRAPH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class GraphRegistry:
    """多图谱注册表：按 graph_id 管理多个 Processor + RememberTaskQueue 实例。

    - 不同图谱完全隔离，各自独立的 processor + DB + queue
    - 所有图谱共享一个 EmbeddingClient 实例（线程安全，省内存）
    - Processor 延迟初始化：首次访问某 graph_id 时才创建
    - 首次创建图谱时自动注册到 SystemMonitor
    """

    def __init__(
        self,
        base_storage_path: str,
        config: dict,
        system_monitor: Optional["SystemMonitor"] = None,
    ):
        self._base_path = Path(base_storage_path)
        self._config = config
        self._system_monitor = system_monitor
        self._embedding_client: Optional[EmbeddingClient] = None
        self._processors: Dict[str, TemporalMemoryGraphProcessor] = {}
        self._queues: Dict[str, object] = {}
        self._orchestrators: Dict[str, DreamOrchestrator] = {}
        self._dream_locks: Dict[str, threading.Lock] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 共享 EmbeddingClient（延迟初始化，所有图谱共用）
    # ------------------------------------------------------------------

    def _get_embedding_client(self) -> EmbeddingClient:
        if self._embedding_client is None:
            embedding = self._config.get("embedding") or {}
            model_path, model_name, use_local = resolve_embedding_model(embedding)
            self._embedding_client = EmbeddingClient(
                model_path=model_path,
                model_name=model_name,
                device=embedding.get("device", "cpu"),
                use_local=use_local,
            )
        return self._embedding_client

    # ------------------------------------------------------------------
    # Processor 管理
    # ------------------------------------------------------------------

    def get_processor(self, graph_id: str) -> TemporalMemoryGraphProcessor:
        """获取或创建指定图谱的 Processor（线程安全）。"""
        with self._lock:
            if graph_id not in self._processors:
                storage_path = str(self._base_path / graph_id)
                self._processors[graph_id] = self._build_processor(storage_path)
            return self._processors[graph_id]

    def create_task_processor(self, graph_id: str) -> TemporalMemoryGraphProcessor:
        """为单个 remember task 创建独立 Processor 实例。

        用于 load_cache_memory=False 的独立任务并行执行，避免共享
        current_episode 等运行时状态。
        """
        with self._lock:
            storage_path = str(self._base_path / graph_id)
        return self._build_processor(storage_path)

    def _build_processor(self, storage_path: str) -> TemporalMemoryGraphProcessor:
        """根据 config 构建一个 Processor 实例，使用共享的 EmbeddingClient。"""
        config = self._config
        chunking = config.get("chunking") or {}
        window_size = chunking.get("window_size", 1000)
        overlap = chunking.get("overlap", 200)
        llm = config.get("llm") or {}
        pipeline = config.get("pipeline") or {}
        runtime = config.get("runtime") or {}
        runtime_concurrency = runtime.get("concurrency") or {}
        runtime_task = runtime.get("task") or {}
        pipeline_search = pipeline.get("search") or {}
        pipeline_alignment = pipeline.get("alignment") or {}
        pipeline_extraction = pipeline.get("extraction") or {}
        pipeline_debug = pipeline.get("debug") or {}
        max_concurrency = llm.get("max_concurrency")
        # 从 storage_path 提取 graph_id（路径格式: {base_path}/{graph_id}/）
        graph_id = Path(storage_path).name
        kwargs: dict = {
            "storage_path": storage_path,
            "config": config,
            "window_size": window_size,
            "overlap": overlap,
            "llm_api_key": llm.get("api_key"),
            "llm_model": llm.get("model", "gpt-4"),
            "llm_base_url": llm.get("base_url"),
            "alignment_llm": merge_llm_alignment(llm),
            "llm_think_mode": bool(llm.get("think", llm.get("think_mode", False))),
            "embedding_client": self._get_embedding_client(),
            "llm_max_tokens": llm.get("max_tokens"),
            "llm_context_window_tokens": llm.get("context_window_tokens"),
            "max_llm_concurrency": max_concurrency,
            "load_cache_memory": runtime_task.get("load_cache_memory", pipeline.get("load_cache_memory")),
            "max_concurrent_windows": runtime_concurrency.get("window_workers", pipeline.get("max_concurrent_windows")),
        }
        for key in (
            "similarity_threshold", "max_similar_entities", "content_snippet_length",
            "relation_content_snippet_length", "relation_endpoint_jaccard_threshold",
            "relation_endpoint_embedding_threshold",
            "jaccard_search_threshold",
            "embedding_name_search_threshold", "embedding_full_search_threshold",
        ):
            if key in pipeline_search:
                kwargs[key] = pipeline_search[key]
        if "max_alignment_candidates" in pipeline_alignment:
            kwargs["max_alignment_candidates"] = pipeline_alignment["max_alignment_candidates"]
        for key in (
            "extraction_rounds", "entity_extraction_rounds", "relation_extraction_rounds",
            "entity_post_enhancement", "prompt_episode_max_chars",
            "compress_multi_round_extraction",
        ):
            if key in pipeline_extraction:
                kwargs[key] = pipeline_extraction[key]
        if "distill_data_dir" in pipeline_debug:
            kwargs["distill_data_dir"] = pipeline_debug["distill_data_dir"]
        return TemporalMemoryGraphProcessor(**kwargs)

    # ------------------------------------------------------------------
    # DreamOrchestrator 管理（持久化跨周期历史）
    # ------------------------------------------------------------------

    def get_dream_orchestrator(
        self,
        graph_id: str,
        config: Optional["DreamConfig"] = None,
    ) -> DreamOrchestrator:
        """获取或创建指定图谱的持久化 DreamOrchestrator。

        每个 graph_id 持有一个 orchestrator 实例，保留跨周期的 LRU 历史。
        每次调用可传入新的 config 覆盖参数（如 strategy、seed_count），
        但 _history 和 _cycle_count 在实例生命周期内持久保留。
        """
        with self._lock:
            if graph_id not in self._orchestrators:
                processor = self.get_processor(graph_id)
                self._orchestrators[graph_id] = DreamOrchestrator(
                    processor.storage, processor.llm_client, config
                )
            orch = self._orchestrators[graph_id]
            if config is not None:
                orch.config = config
            return orch

    def get_dream_lock(self, graph_id: str) -> threading.Lock:
        """获取指定图谱的 dream cycle 互斥锁。

        防止同一图谱上并发执行 dream cycle，避免重复关系创建。
        """
        with self._lock:
            if graph_id not in self._dream_locks:
                self._dream_locks[graph_id] = threading.Lock()
            return self._dream_locks[graph_id]

    # ------------------------------------------------------------------
    # Queue 管理（延迟导入避免循环依赖）
    # ------------------------------------------------------------------

    def get_queue(self, graph_id: str):
        """获取或创建指定图谱的 RememberTaskQueue。"""
        # 快速路径：已存在则直接返回
        with self._lock:
            if graph_id in self._queues:
                return self._queues[graph_id]

        # 构造 RememberTaskQueue（会启动 worker 线程），必须在锁外执行，
        # 否则 worker 调用 create_task_processor() 时会死锁。
        from server.task_queue import RememberTaskQueue

        processor = self.get_processor(graph_id)
        storage_path = Path(processor.storage.storage_path)
        event_log = None
        if self._system_monitor is not None:
            event_log = self._system_monitor.event_log
        queue = RememberTaskQueue(
            processor,
            storage_path,
            processor_factory=lambda gid=graph_id: self.create_task_processor(gid),
            max_workers=self._config.get("remember_workers", 1),
            max_retries=self._config.get("remember_max_retries", 2),
            retry_delay_seconds=self._config.get("remember_retry_delay_seconds", 2),
            event_log=event_log,
        )

        with self._lock:
            # 双重检查：防止并发重复创建
            if graph_id not in self._queues:
                self._queues[graph_id] = queue
                if self._system_monitor is not None:
                    self._system_monitor.attach_graph(graph_id, processor, queue)
            return self._queues[graph_id]

    # ------------------------------------------------------------------
    # 图谱元数据（metadata.json）
    # ------------------------------------------------------------------

    def _metadata_path(self, graph_id: str) -> Path:
        """Return the path to a graph's metadata.json."""
        return self._base_path / graph_id / "metadata.json"

    def get_graph_metadata(self, graph_id: str) -> Dict[str, Any]:
        """Read metadata for a graph. Returns empty dict if not stored."""
        mp = self._metadata_path(graph_id)
        if mp.is_file():
            try:
                return json.loads(mp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def set_graph_metadata(self, graph_id: str, **kwargs) -> Dict[str, Any]:
        """Write or merge metadata for a graph. Returns the full metadata after write."""
        mp = self._metadata_path(graph_id)
        existing = self.get_graph_metadata(graph_id)
        # Only update keys that are provided and non-None
        for k, v in kwargs.items():
            if v is not None:
                existing[k] = v
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        return existing

    def get_graph_info(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """Get full info for a single graph: metadata + entity/relation counts.

        Returns None if the graph directory does not exist.
        """
        graph_dir = self._base_path / graph_id
        if not graph_dir.is_dir():
            return None
        metadata = self.get_graph_metadata(graph_id)
        # Ensure graph_id is in metadata
        metadata.setdefault("graph_id", graph_id)
        # Try to get counts from processor (lazy — won't trigger init if not loaded)
        entity_count = 0
        relation_count = 0
        processor = self._processors.get(graph_id)
        if processor and hasattr(processor, "storage"):
            try:
                stats = processor.storage.get_stats()
                entity_count = stats.get("entities", 0)
                relation_count = stats.get("relations", 0)
            except Exception:
                pass
        metadata["entity_count"] = entity_count
        metadata["relation_count"] = relation_count
        return metadata

    # ------------------------------------------------------------------
    # 图谱列表
    # ------------------------------------------------------------------

    def list_graphs(self) -> List[str]:
        """列出基础目录下所有已有图谱（支持 SQLite 和 Neo4j 后端）。"""
        result: List[str] = []
        if not self._base_path.is_dir():
            return result
        for child in sorted(self._base_path.iterdir()):
            if not child.is_dir():
                continue
            # SQLite 后端：有 graph.db
            # Neo4j 后端：有 vectors.db 或 docs/ 目录
            if ((child / "graph.db").exists() or
                    (child / "vectors.db").exists() or
                    (child / "docs").is_dir()):
                result.append(child.name)
        return result

    def list_graphs_info(self) -> List[Dict[str, Any]]:
        """列出所有图谱及其元数据和统计信息。"""
        return [info for gid in self.list_graphs() if (info := self.get_graph_info(gid)) is not None]

    # ------------------------------------------------------------------
    # 图谱删除
    # ------------------------------------------------------------------

    def delete_graph(self, graph_id: str) -> None:
        """删除指定图谱：停止任务队列、关闭 processor 连接、删除数据目录。

        Raises:
            KeyError: 如果 graph_id 不存在。
        """
        with self._lock:
            # 1. 停止并移除任务队列
            queue = self._queues.pop(graph_id, None)
            if queue and hasattr(queue, "shutdown"):
                try:
                    queue.shutdown()
                except Exception as _e:
                    logging.getLogger(__name__).warning("关闭 graph %s 任务队列失败: %s", graph_id, _e)

            # 2. 移除 dream orchestrator
            self._orchestrators.pop(graph_id, None)
            self._dream_locks.pop(graph_id, None)

            # 3. 移除 processor（关闭 DB 连接）
            processor = self._processors.pop(graph_id, None)
            if processor and hasattr(processor.storage, "close"):
                try:
                    processor.storage.close()
                except Exception as _e:
                    logging.getLogger(__name__).warning("关闭 graph %s 存储连接失败: %s", graph_id, _e)

            # 3. 删除数据目录
            graph_dir = self._base_path / graph_id
            if graph_dir.is_dir():
                import shutil
                shutil.rmtree(graph_dir)

    # ------------------------------------------------------------------
    # graph_id 校验
    # ------------------------------------------------------------------

    @staticmethod
    def validate_graph_id(graph_id: str) -> None:
        """校验 graph_id：只允许字母数字、下划线、连字符，禁止路径穿越。

        Raises:
            ValueError: 如果 graph_id 不合法。
        """
        if not isinstance(graph_id, str) or not graph_id.strip():
            raise ValueError("graph_id 为必填参数")
        graph_id = graph_id.strip()
        if not _GRAPH_ID_RE.match(graph_id):
            raise ValueError(
                f"graph_id 不合法: {graph_id!r}（只允许字母、数字、下划线、连字符，"
                "长度 1-128，以字母或数字开头）"
            )
