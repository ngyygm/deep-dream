"""Graph registry for physically isolated SQLite concept graphs."""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.remember.orchestrator import TemporalMemoryGraphProcessor
from core.server.config import merge_llm_alignment, merge_llm_extraction, resolve_embedding_model  # noqa: F401
from core.storage.embedding import EmbeddingClient

if TYPE_CHECKING:
    from core.server.monitor import SystemMonitor

logger = logging.getLogger(__name__)

_GRAPH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class GraphRegistry:
    """Owns graph-local processors, queues, and metadata.

    Business graph data is physically isolated under:
        {storage_root}/graphs/{graph_id}/

    The root-level registry.json only stores graph metadata. It never stores
    concept families, versions, edges, documents, blobs, or vector indexes.
    """

    def __init__(
        self,
        base_storage_path: str,
        config: dict,
        system_monitor: Optional["SystemMonitor"] = None,
    ):
        self._base_path = Path(base_storage_path)
        self._graphs_path = self._base_path / "graphs"
        self._registry_path = self._base_path / "registry.json"
        self._config = config
        self._system_monitor = system_monitor
        self._embedding_client: Optional[EmbeddingClient] = None
        self._processors: Dict[str, TemporalMemoryGraphProcessor] = {}
        self._queues: Dict[str, object] = {}
        self._lock = threading.RLock()

        self._base_path.mkdir(parents=True, exist_ok=True)
        self._graphs_path.mkdir(parents=True, exist_ok=True)
        if not self._registry_path.exists():
            self._write_registry({"graphs": {}})

    # ------------------------------------------------------------------
    # Paths and registry metadata
    # ------------------------------------------------------------------

    def graph_dir(self, graph_id: str) -> Path:
        self.validate_graph_id(graph_id)
        return self._graphs_path / graph_id

    def _read_registry(self) -> Dict[str, Any]:
        try:
            data = json.loads(self._registry_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("graphs", {})
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return {"graphs": {}}

    def _write_registry(self, data: Dict[str, Any]) -> None:
        data.setdefault("graphs", {})
        self._base_path.mkdir(parents=True, exist_ok=True)
        tmp = self._registry_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._registry_path)

    def get_graph_metadata(self, graph_id: str) -> Dict[str, Any]:
        self.validate_graph_id(graph_id)
        registry = self._read_registry()
        meta = dict((registry.get("graphs") or {}).get(graph_id) or {})
        return meta

    def set_graph_metadata(self, graph_id: str, **kwargs) -> Dict[str, Any]:
        self.validate_graph_id(graph_id)
        registry = self._read_registry()
        graphs = registry.setdefault("graphs", {})
        existing = dict(graphs.get(graph_id) or {})
        existing.setdefault("graph_id", graph_id)
        existing.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        for key, value in kwargs.items():
            if value is not None:
                existing[key] = value
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        graphs[graph_id] = existing
        self._write_registry(registry)
        return dict(existing)

    def _remove_graph_metadata(self, graph_id: str) -> None:
        registry = self._read_registry()
        graphs = registry.setdefault("graphs", {})
        graphs.pop(graph_id, None)
        self._write_registry(registry)

    # ------------------------------------------------------------------
    # Shared EmbeddingClient
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
    # Processor lifecycle
    # ------------------------------------------------------------------

    def get_processor(self, graph_id: str) -> TemporalMemoryGraphProcessor:
        self.validate_graph_id(graph_id)
        with self._lock:
            if graph_id not in self._processors:
                graph_dir = self.graph_dir(graph_id)
                graph_dir.mkdir(parents=True, exist_ok=True)
                self.set_graph_metadata(graph_id)
                self._processors[graph_id] = self._build_processor(str(graph_dir), graph_id)
            return self._processors[graph_id]

    def get_processor_with_retry(self, graph_id: str, max_retries: int = 2) -> TemporalMemoryGraphProcessor:
        for attempt in range(max_retries + 1):
            try:
                return self.get_processor(graph_id)
            except Exception:
                if attempt == max_retries:
                    raise
                import time

                time.sleep(0.5 * (attempt + 1))

    def create_task_processor(self, graph_id: str) -> TemporalMemoryGraphProcessor:
        self.validate_graph_id(graph_id)
        graph_dir = self.graph_dir(graph_id)
        graph_dir.mkdir(parents=True, exist_ok=True)
        return self._build_processor(str(graph_dir), graph_id)

    def _build_processor(self, storage_path: str, graph_id: str) -> TemporalMemoryGraphProcessor:
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
        pipeline_remember = pipeline.get("remember") or {}
        pipeline_debug = pipeline.get("debug") or {}

        kwargs: dict = {
            "storage_path": storage_path,
            "config": config,
            "graph_id": graph_id,
            "window_size": window_size,
            "overlap": overlap,
            "llm_api_key": llm.get("api_key"),
            "llm_model": llm.get("model", "gpt-4"),
            "llm_base_url": llm.get("base_url"),
            "alignment_llm": merge_llm_alignment(llm),
            "extraction_llm": merge_llm_extraction(llm),
            "llm_think_mode": bool(llm.get("think", llm.get("think_mode", False))),
            "embedding_client": self._get_embedding_client(),
            "llm_max_tokens": llm.get("max_tokens"),
            "llm_context_window_tokens": llm.get("context_window_tokens"),
            "max_llm_concurrency": llm.get("max_concurrency"),
            "load_cache_memory": runtime_task.get("load_cache_memory", pipeline.get("load_cache_memory")),
            "max_concurrent_windows": runtime_concurrency.get("window_workers", pipeline.get("max_concurrent_windows")),
        }
        for key in (
            "similarity_threshold",
            "max_similar_entities",
            "content_snippet_length",
            "relation_content_snippet_length",
            "relation_endpoint_jaccard_threshold",
            "relation_endpoint_embedding_threshold",
            "jaccard_search_threshold",
            "embedding_name_search_threshold",
            "embedding_full_search_threshold",
        ):
            if key in pipeline_search:
                kwargs[key] = pipeline_search[key]
        if "max_alignment_candidates" in pipeline_alignment:
            kwargs["max_alignment_candidates"] = pipeline_alignment["max_alignment_candidates"]
        for key in (
            "prompt_episode_max_chars",
            "entity_rounds",
            "relation_rounds",
            "entity_refine_rounds",
            "relation_refine_rounds",
        ):
            if key in pipeline_extraction:
                kwargs[key] = pipeline_extraction[key]
        if pipeline_remember:
            kwargs["remember_config"] = pipeline_remember
        if "distill_data_dir" in pipeline_debug:
            kwargs["distill_data_dir"] = pipeline_debug["distill_data_dir"]
        return TemporalMemoryGraphProcessor(**kwargs)

    # ------------------------------------------------------------------
    # Queue lifecycle
    # ------------------------------------------------------------------

    def get_queue(self, graph_id: str):
        self.validate_graph_id(graph_id)
        with self._lock:
            if graph_id in self._queues:
                return self._queues[graph_id]

        from core.server.task_queue import RememberTaskQueue

        processor = self.get_processor(graph_id)
        event_log = self._system_monitor.event_log if self._system_monitor is not None else None
        queue = RememberTaskQueue(
            processor,
            Path(processor.storage.storage_path),
            processor_factory=lambda gid=graph_id: self.create_task_processor(gid),
            max_workers=self._config.get("remember_workers", 1),
            max_retries=self._config.get("remember_max_retries", 2),
            retry_delay_seconds=self._config.get("remember_retry_delay_seconds", 2),
            event_log=event_log,
            stall_timeout_seconds=self._config.get("remember_stall_timeout_seconds", 600),
        )

        with self._lock:
            if graph_id not in self._queues:
                self._queues[graph_id] = queue
                if self._system_monitor is not None:
                    self._system_monitor.attach_graph(graph_id, processor, queue)
            return self._queues[graph_id]

    # ------------------------------------------------------------------
    # Graph list/info
    # ------------------------------------------------------------------

    def list_graphs(self) -> List[str]:
        ids: set[str] = set()
        registry_graphs = self._read_registry().get("graphs") or {}
        ids.update(registry_graphs.keys())
        if self._graphs_path.is_dir():
            for child in self._graphs_path.iterdir():
                if child.is_dir() and ((child / "graph.db").exists() or child.name in registry_graphs):
                    ids.add(child.name)
        with self._lock:
            ids.update(self._processors.keys())
        return sorted(ids)

    def get_graph_info(self, graph_id: str) -> Optional[Dict[str, Any]]:
        self.validate_graph_id(graph_id)
        graph_dir = self.graph_dir(graph_id)
        metadata = self.get_graph_metadata(graph_id)
        if not graph_dir.is_dir() and not metadata and graph_id not in self._processors:
            return None
        metadata.setdefault("graph_id", graph_id)
        metadata.setdefault("path", str(graph_dir))

        stats = {}
        processor = self._processors.get(graph_id)
        try:
            if processor and hasattr(processor, "storage"):
                stats = processor.storage.get_stats()
            elif (graph_dir / "graph.db").exists():
                from core.storage import create_storage_manager

                storage = create_storage_manager(self._config, embedding_client=None, storage_path=str(graph_dir), graph_id=graph_id)
                try:
                    stats = storage.get_stats()
                finally:
                    storage.close()
        except Exception as exc:
            logger.debug("Failed to read graph stats for %s: %s", graph_id, exc)

        metadata["entity_count"] = int(stats.get("entities", 0) or 0)
        metadata["relation_count"] = int(stats.get("relations", 0) or 0)
        metadata["document_count"] = int(stats.get("documents", 0) or 0)
        metadata["episode_count"] = int(stats.get("episodes", 0) or 0)
        return metadata

    def list_graphs_info(self) -> List[Dict[str, Any]]:
        return [info for gid in self.list_graphs() if (info := self.get_graph_info(gid)) is not None]

    # ------------------------------------------------------------------
    # Graph deletion/clear
    # ------------------------------------------------------------------

    def clear_graph(self, graph_id: str) -> None:
        self.validate_graph_id(graph_id)
        processor = self.get_processor(graph_id)
        if hasattr(processor.storage, "clear_graph_data"):
            processor.storage.clear_graph_data()
        self.set_graph_metadata(graph_id, cleared_at=datetime.now(timezone.utc).isoformat())
        logger.info("Cleared graph '%s'", graph_id)

    def delete_graph(self, graph_id: str) -> None:
        self.validate_graph_id(graph_id)
        with self._lock:
            queue = self._queues.pop(graph_id, None)
            if queue and hasattr(queue, "shutdown"):
                try:
                    queue.shutdown()
                except Exception as exc:
                    logger.warning("Failed to shut down graph %s queue: %s", graph_id, exc)

            processor = self._processors.pop(graph_id, None)
            if processor and hasattr(processor.storage, "close"):
                try:
                    processor.storage.close()
                except Exception as exc:
                    logger.warning("Failed to close graph %s storage: %s", graph_id, exc)

            graph_dir = self.graph_dir(graph_id)
            if graph_dir.is_dir():
                shutil.rmtree(graph_dir)

            self._remove_graph_metadata(graph_id)
            if self._system_monitor is not None:
                self._system_monitor.detach_graph(graph_id)
            logger.info("Deleted graph '%s'", graph_id)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_graph_id(graph_id: str) -> None:
        if not isinstance(graph_id, str) or not graph_id.strip():
            raise ValueError("graph_id is required")
        graph_id = graph_id.strip()
        if graph_id in (".", ".."):
            raise ValueError(f"invalid graph_id: {graph_id!r}")
        if "/" in graph_id or "\\" in graph_id:
            raise ValueError(f"invalid graph_id: {graph_id!r}")
        if "\x00" in graph_id:
            raise ValueError("graph_id contains illegal characters")
        if not _GRAPH_ID_RE.match(graph_id):
            raise ValueError(
                f"invalid graph_id: {graph_id!r} "
                "(allowed: letters, numbers, underscore, hyphen; length 1-128; starts with letter/number)"
            )
