"""
主处理流程：整合所有模块，实现完整的文档处理pipeline
"""
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import logging
import threading

# Static defaults — computed once, not per call
from .document import DocumentProcessor
from core.llm.client import LLMClient, LLMCallStats
from core.storage.embedding import EmbeddingClient
from core.storage import create_storage_manager
from .entity import EntityProcessor
from .relation import RelationProcessor
from core.models import Episode
from .alignment import _PipelineExtractionMixin
from .cross_window import _CrossWindowDedupMixin
from .orchestrator_pipeline import _PipelineMixin

# Sub-modules (no circular import — they do NOT import from orchestrator.py)
from . import pipeline_state as _ps
from . import pipeline_workers as _pw

_REMEMBER_DEFAULTS = {
    "alignment_policy": "conservative",
}


logger = logging.getLogger(__name__)


class RememberControlFlow(Exception):
    def __init__(self, action: str):
        super().__init__(action)
        self.remember_control_action = action


class TemporalMemoryGraphProcessor(_PipelineMixin, _PipelineExtractionMixin, _CrossWindowDedupMixin):
    """时序记忆图谱处理器 - 主处理流程"""

    # ------------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------------

    def _resolve_remember_config(self, config, remember_config,
                                  content_snippet_length, relation_content_snippet_length,
                                  relation_endpoint_jaccard_threshold,
                                  relation_endpoint_embedding_threshold,
                                  max_similar_entities, llm_context_window_tokens):
        """解析 remember 配置：合并默认值 → config → remember_config 覆盖。"""
        _content_snippet_length = content_snippet_length if content_snippet_length is not None else 300
        _remember_from_config = (((config or {}).get("pipeline") or {}).get("remember") or {})
        _remember_overrides = remember_config or {}
        _remember_cfg = dict(_REMEMBER_DEFAULTS)
        _remember_cfg.update(_remember_from_config)
        if remember_config:
            _remember_cfg.update(remember_config)
        self.remember_config = _remember_cfg
        self.remember_alignment_policy = str(_remember_cfg.get("alignment_policy") or "conservative").strip() or "conservative"
        self.remember_alignment_conservative = self.remember_alignment_policy == "conservative"
        self.remember_profile = str(_remember_cfg.get("profile") or "current")
        # P2：孤立实体兜底共现关系（confidence 0.3 轮询配对）默认关闭——
        # 见 alignment_relations.py 兜底阶段注释。
        self.remember_fallback_cooccurrence = bool(_remember_cfg.get("fallback_cooccurrence_relations", False))
        # strong-v1：窗口级批量对齐裁决（step9 实体一次批量、step10 有已有关系的实体对一次批量）
        self.window_batch_alignment_enabled = bool(
            _remember_cfg.get("window_batch_alignment", self.remember_profile == "strong-v1"))
        self.remember_preserve_source_language = bool(_remember_cfg.get("preserve_source_language", False))
        self.remember_max_entities_per_window = max(1, int(_remember_cfg.get("max_entities_per_window") or 16))
        self.remember_max_relations_per_window = max(1, int(_remember_cfg.get("max_relations_per_window") or 24))
        # strong-v1 检索切片：窗口 episode 之外按 ~N 字在行边界追加薄 FTS 切片行（0=关闭）
        self.remember_episode_slice_chars = max(0, int(_remember_cfg.get("episode_slice_chars") or 0))
        _relation_content_snippet_length = relation_content_snippet_length if relation_content_snippet_length is not None else 200
        _relation_endpoint_jaccard_threshold = (
            float(relation_endpoint_jaccard_threshold)
            if relation_endpoint_jaccard_threshold is not None else 0.9
        )
        _rel_emb_thr = relation_endpoint_embedding_threshold
        if _rel_emb_thr is None:
            _relation_endpoint_embedding_threshold = 0.85
        else:
            v = float(_rel_emb_thr)
            _relation_endpoint_embedding_threshold = None if v <= 0 else v
        _max_similar_entities = max_similar_entities if max_similar_entities is not None else 10

        _ctx_win = llm_context_window_tokens
        if _ctx_win is None:
            _ctx_win = 8000
        _ctx_win = max(256, int(_ctx_win))

        # Return derived values needed by _create_components
        return {
            "content_snippet_length": _content_snippet_length,
            "relation_content_snippet_length": _relation_content_snippet_length,
            "relation_endpoint_jaccard_threshold": _relation_endpoint_jaccard_threshold,
            "relation_endpoint_embedding_threshold": _relation_endpoint_embedding_threshold,
            "max_similar_entities": _max_similar_entities,
            "context_window_tokens": _ctx_win,
        }

    # ------------------------------------------------------------------
    # Threading initialization
    # ------------------------------------------------------------------

    def _init_threading(self, max_concurrent_windows, max_llm_concurrency):
        """初始化流水线并行：cache 更新串行锁 + 抽取/处理线程池。"""
        if max_concurrent_windows is not None:
            if isinstance(max_concurrent_windows, str) and max_concurrent_windows.lower() == "auto":
                _max_concurrent_windows = max(2, min(max_llm_concurrency or 1, 8))
            else:
                _max_concurrent_windows = int(max_concurrent_windows)
        else:
            _max_concurrent_windows = max(2, min(max_llm_concurrency or 1, 8))
        _max_concurrent_windows = max(1, min(_max_concurrent_windows, 64))

        self._cache_lock = threading.Lock()
        self._max_concurrent_windows = _max_concurrent_windows
        self._window_slot = threading.Semaphore(_max_concurrent_windows)
        self._runtime_lock = threading.Lock()
        self._active_window_extractions = 0
        self._peak_window_extractions = 0
        self._active_main_pipeline_windows = 0
        self._active_step9 = 0
        self._active_step10 = 0
        self._extraction_executor = ThreadPoolExecutor(
            max_workers=_max_concurrent_windows,
            thread_name_prefix="tmg-window",
        )
        self._current_state = None
        self._current_state_lock = threading.Lock()

    def __init__(self, storage_path: str, window_size: int = 1000, overlap: int = 200,
                 llm_api_key: Optional[str] = None, llm_model: str = "gpt-4",
                 config: Optional[Dict[str, Any]] = None,
                 storage_manager=None,
                 llm_base_url: Optional[str] = None,
                 alignment_llm: Optional[Dict[str, Any]] = None,
                 embedding_model_path: Optional[str] = None,
                 embedding_model_name: Optional[str] = None,
                 embedding_device: str = "cpu",
                 embedding_use_local: bool = True,
                 embedding_client: Optional[EmbeddingClient] = None,
                 llm_think_mode: bool = False,
                 llm_max_tokens: Optional[int] = None,
                 llm_context_window_tokens: Optional[int] = None,
                 llm_timeout_seconds: Optional[int] = None,
                 llm_connect_timeout_seconds: Optional[int] = None,
                 prompt_episode_max_chars: Optional[int] = None,
                 max_llm_concurrency: Optional[int] = None,
                 # pipeline 可选配置（可从 config.pipeline 传入）
                 similarity_threshold: Optional[float] = None,
                 max_similar_entities: Optional[int] = None,
                 content_snippet_length: Optional[int] = None,
                 relation_content_snippet_length: Optional[int] = None,
                 relation_endpoint_jaccard_threshold: Optional[float] = None,
                 relation_endpoint_embedding_threshold: Optional[float] = None,
                 load_cache_memory: Optional[bool] = None,
                 jaccard_search_threshold: Optional[float] = None,
                 embedding_name_search_threshold: Optional[float] = None,
                 embedding_full_search_threshold: Optional[float] = None,
                 max_concurrent_windows: Optional[int] = None,
                 max_alignment_candidates: Optional[int] = None,
                 distill_data_dir: Optional[str] = None,
                 remember_config: Optional[Dict[str, Any]] = None,
                 graph_id: Optional[str] = None,
                 embedding_cache_max_size: Optional[int] = None,
                 embedding_cache_ttl: Optional[float] = None,
                 shared_llm_semaphore=None,
                 judge_service=None,
                 family_write_gate=None):
        """
        初始化处理器

        Args:
            storage_path: 存储路径
            window_size: 窗口大小（字符数）
            overlap: 重叠大小（字符数）
            llm_api_key: LLM API密钥
            llm_model: LLM模型名称
            llm_base_url: LLM API基础URL（步骤1–5）
            alignment_llm: 可选 dict（由配置 merge_llm_alignment 生成）。含 enabled、max_concurrency（对齐阶段 LLM 并发，与 max_llm_concurrency 解耦）及 api_key、base_url、model 等；enabled 为 false 时不使用独立对齐模型
            embedding_model_path: Embedding模型本地路径（优先使用）
            embedding_model_name: Embedding模型名称（HuggingFace模型名）
            embedding_device: Embedding计算设备 ("cpu" 或 "cuda")
            embedding_use_local: 是否优先使用本地 embedding 模型
            llm_think_mode: LLM 是否开启思维链/think 模式（默认 False）。仅 Ollama 原生 `/api/chat` 支持通过 API 参数 think 控制；非 Ollama 后端忽略
            similarity_threshold: 实体相似度阈值（默认 0.7）
            max_similar_entities: 语义搜索返回的最大相似实体数（默认 10）
            content_snippet_length: 实体 content 截取长度（默认 300）
            relation_content_snippet_length: 关系 content 截取长度（默认 200）
            load_cache_memory: 是否加载缓存记忆续写（默认 False）
            jaccard_search_threshold: Jaccard 搜索阈值（可选，不设则用 similarity_threshold）
            embedding_name_search_threshold: Embedding 名称搜索阈值（可选）
            embedding_full_search_threshold: Embedding 全文搜索阈值（可选）
            max_concurrent_windows: 同时处理的滑窗数上限（默认 1）；满员时不唤醒下一窗口，避免窗口内实体/关系并行导致线程爆炸
            llm_context_window_tokens: 请求输入 prompt 的本地预检上限；未传时读 server 默认
            prompt_episode_max_chars: 注入抽取 prompt 的记忆缓存最大字符数；超长时自动截断，默认 2000
            embedding_cache_max_size: Embedding缓存最大条目数（默认8192，可从config.embedding.cache_max_size读取）
            embedding_cache_ttl: Embedding缓存TTL秒数（默认300秒，可从config.embedding.cache_ttl读取）
        """
        # --- Phase 1: Resolve remember config ---
        _derived = self._resolve_remember_config(
            config, remember_config,
            content_snippet_length, relation_content_snippet_length,
            relation_endpoint_jaccard_threshold, relation_endpoint_embedding_threshold,
            max_similar_entities, llm_context_window_tokens,
        )
        _content_snippet_length = _derived["content_snippet_length"]
        _relation_content_snippet_length = _derived["relation_content_snippet_length"]
        _relation_endpoint_jaccard_threshold = _derived["relation_endpoint_jaccard_threshold"]
        _relation_endpoint_embedding_threshold = _derived["relation_endpoint_embedding_threshold"]
        _max_similar_entities = _derived["max_similar_entities"]
        _ctx_win = _derived["context_window_tokens"]

        self.embedding_client = embedding_client or EmbeddingClient(
            model_path=embedding_model_path,
            model_name=embedding_model_name,
            device=embedding_device,
            use_local=embedding_use_local,
            cache_max_size=embedding_cache_max_size or 8192,
            cache_ttl=embedding_cache_ttl or 3600.0,
            max_concurrency=1,
        )

        if storage_manager is not None:
            self.storage = storage_manager
        elif config is not None:
            self.storage = create_storage_manager(
                config,
                embedding_client=self.embedding_client,
                storage_path=storage_path,
                entity_content_snippet_length=_content_snippet_length,
                relation_content_snippet_length=_relation_content_snippet_length,
                graph_id=graph_id,
            )
        else:
            self.storage = create_storage_manager(
                {"storage": {"backend": "sqlite"}},
                embedding_client=self.embedding_client,
                storage_path=storage_path,
                entity_content_snippet_length=_content_snippet_length,
                relation_content_snippet_length=_relation_content_snippet_length,
                graph_id=graph_id,
            )
        self.document_processor = DocumentProcessor(window_size, overlap)
        # strong-v1 单遍抽取：remember 配置可覆盖窗口尺寸（默认 6000/300 大窗口）
        _strong_ws = int(self.remember_config.get("window_size_chars") or 0)
        if _strong_ws >= 500:
            _strong_ov = int(self.remember_config.get("overlap_chars") or 0) or 300
            self.document_processor = DocumentProcessor(_strong_ws, _strong_ov)
        _al = alignment_llm or {}
        _main_openai_extra_body = ((config or {}).get("llm") or {}).get("extra_body") or {}
        _llm_protocol = ((config or {}).get("llm") or {}).get("protocol")
        # 主 client 与抽取 client 共享同一份调用计数，processor 层汇总输出
        self.llm_call_stats = LLMCallStats()
        self.llm_client = LLMClient(
            llm_api_key,
            llm_model,
            llm_base_url,
            content_snippet_length=_content_snippet_length,
            relation_content_snippet_length=_relation_content_snippet_length,
            relation_endpoint_jaccard_threshold=_relation_endpoint_jaccard_threshold,
            embedding_client=self.embedding_client,
            relation_endpoint_embedding_threshold=_relation_endpoint_embedding_threshold,
            think_mode=llm_think_mode,
            max_tokens=llm_max_tokens,
            context_window_tokens=_ctx_win,
            timeout_seconds=llm_timeout_seconds,
            connect_timeout_seconds=llm_connect_timeout_seconds,
            prompt_episode_max_chars=prompt_episode_max_chars,
            max_llm_concurrency=max_llm_concurrency,
            openai_extra_body=_main_openai_extra_body,
            protocol=_llm_protocol,
            distill_data_dir=distill_data_dir,
            alignment_enabled=bool(_al.get("enabled", False)),
            alignment_base_url=_al.get("base_url"),
            alignment_api_key=_al.get("api_key"),
            alignment_model=_al.get("model"),
            alignment_max_tokens=_al.get("max_tokens"),
            alignment_think_mode=_al.get("think_mode"),
            alignment_content_snippet_length=_al.get("content_snippet_length"),
            alignment_relation_content_snippet_length=_al.get("relation_content_snippet_length"),
            alignment_openai_extra_body=_al.get("extra_body", _main_openai_extra_body),
            call_stats=self.llm_call_stats,
            shared_llm_semaphore=shared_llm_semaphore,
            judge_service=judge_service,
        )
        self.llm_client.preserve_source_language = self.remember_preserve_source_language
        # 库级共享信号量：registry 传入时所有 processor 共用一个闸门（真实端点容量）；
        # 未传时退回旧行为——主 client 自建，抽取 client 复用主 client 的
        _shared_llm_semaphore = shared_llm_semaphore or getattr(self.llm_client, "_llm_semaphore", None)
        _shared_llm_slot_max = self.llm_client.get_llm_semaphore_max() if hasattr(self.llm_client, "get_llm_semaphore_max") else None
        self.entity_processor = EntityProcessor(
            self.storage,
            self.llm_client,
            max_similar_entities=_max_similar_entities,
            content_snippet_length=_content_snippet_length
        )
        self.relation_processor = RelationProcessor(self.storage, self.llm_client)
        # strong-v1 窗口级批量对齐开关传递给步骤9/10处理器
        self.entity_processor.window_batch_alignment_enabled = self.window_batch_alignment_enabled
        self.relation_processor.window_batch_alignment_enabled = self.window_batch_alignment_enabled
        # FamilyWriteGate：并发创建同名 family 的竞态兜底（库级共享，registry 注入）
        self.family_write_gate = family_write_gate
        self.entity_processor.family_write_gate = family_write_gate
        if self.remember_alignment_conservative:
            self.entity_processor.batch_resolution_confidence_threshold = 0.9
            self.entity_processor.merge_safe_embedding_threshold = max(self.entity_processor.merge_safe_embedding_threshold, 0.7)
            self.entity_processor.merge_safe_jaccard_threshold = max(self.entity_processor.merge_safe_jaccard_threshold, 0.55)
            self.relation_processor.batch_resolution_confidence_threshold = 0.9
            self.relation_processor.preserve_distinct_relations_per_pair = True
        else:
            self.relation_processor.preserve_distinct_relations_per_pair = False

        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else 0.7
        self.max_similar_entities = _max_similar_entities
        self.content_snippet_length = _content_snippet_length
        self.relation_content_snippet_length = _relation_content_snippet_length

        self.llm_threads = max(1, max_llm_concurrency) if max_llm_concurrency else 3
        self.load_cache_memory = load_cache_memory if load_cache_memory is not None else False

        self.jaccard_search_threshold = jaccard_search_threshold
        self.embedding_name_search_threshold = embedding_name_search_threshold
        self.embedding_full_search_threshold = embedding_full_search_threshold
        self.max_alignment_candidates = max_alignment_candidates

        # --- Phase 3: Threading ---
        self.current_episode: Optional[Episode] = None
        self._init_threading(max_concurrent_windows, max_llm_concurrency)

    def reset_llm_call_stats(self) -> None:
        """清零本 processor 的 LLM 调用计数（主 client + 抽取 client 一并清零）。"""
        self.llm_call_stats.reset()

    def get_llm_call_stats(self) -> Dict[str, Any]:
        """返回本 processor 的 LLM 调用计数快照（含 by_step/by_model 分桶）。"""
        return self.llm_call_stats.snapshot()

    def get_runtime_stats(self) -> Dict[str, int]:
        with self._runtime_lock:
            stats = {
                "configured_window_workers": self._max_concurrent_windows,
                "configured_llm_threads": self.llm_threads,
                "active_window_extractions": self._active_window_extractions,
                "active_main_pipeline_windows": self._active_main_pipeline_windows,
                "peak_window_extractions": self._peak_window_extractions,
                "active_step9": self._active_step9,
                "active_step10": self._active_step10,
            }
        # LLM 信号量活跃数（不需要 runtime_lock；支持上游/下游分池）
        if self.llm_client and hasattr(self.llm_client, "get_llm_semaphore_active_count"):
            stats["llm_semaphore_active"] = self.llm_client.get_llm_semaphore_active_count()
            stats["llm_semaphore_max"] = self.llm_client.get_llm_semaphore_max()
            if hasattr(self.llm_client, "get_llm_semaphore_detail"):
                det = self.llm_client.get_llm_semaphore_detail()
                stats["llm_upstream_active"] = det["upstream_active"]
                stats["llm_upstream_max"] = det["upstream_max"]
                stats["llm_downstream_active"] = det["downstream_active"]
                stats["llm_downstream_max"] = det["downstream_max"]
        elif self.llm_client and hasattr(self.llm_client, "_llm_semaphore") and self.llm_client._llm_semaphore:
            sem = self.llm_client._llm_semaphore
            stats["llm_semaphore_active"] = sem.active_count
            stats["llm_semaphore_max"] = sem.max_value
        return stats

    def get_pipeline_snapshot(self) -> Optional[Dict]:
        """返回当前 remember 流水线的逐窗口状态快照，无任务时返回 None。"""
        with self._current_state_lock:
            state = self._current_state
        if state is None:
            return None

        windows = []
        for i in range(state.N):
            windows.append({
                "index": i,
                "extract_done": state.extract_done[i].is_set(),
                "step9_done": state.step9_done_ev[i].is_set(),
                "step10_done": state.step10_done_ev[i].is_set(),
                "has_episode": state.episodes[i] is not None,
                "has_extract_result": state.extract_results[i] is not None,
                "has_align_result": state.align_results[i] is not None,
                "has_step10_result": state.step10_results[i] is not None,
                "failed": state.window_failures[i] is not None if i < len(state.window_failures) else False,
                "timings": dict(state.window_timings[i]) if state.window_timings[i] else {},
            })

        with state.errors_lock:
            errors = [
                {"phase": ph, "window": idx, "error": str(e)}
                for ph, idx, e in state.errors
            ]

        return {
            "total_windows": state.N,
            "errors": errors,
            "windows": windows,
        }

    # ------------------------------------------------------------------
    # Delegated methods — thin wrappers calling sub-module functions
    # ------------------------------------------------------------------

    def _acquire_window_slot(self) -> None:
        _pw.acquire_window_slot(self)

    def _release_window_slot(self) -> None:
        _pw.release_window_slot(self)

    # Pipeline state helpers (delegated to pipeline_state module)
    def _init_remember_shared_state(self, N):
        return _ps.init_remember_shared_state(N)

    @staticmethod
    def _record_window_error(state, stage, idx, exc) -> bool:
        return _ps.record_window_error(state, stage, idx, exc)

    @staticmethod
    def _signal_control_stop(state, action, from_index, **kwargs):
        return _ps.signal_control_stop(state, action, from_index, **kwargs)

    @staticmethod
    def _poll_control(state, control_callback):
        return _ps.poll_control(state, control_callback)

    @staticmethod
    def _safe_progress(progress_callback, progress, label, message, chain_id="step9"):
        return _ps.safe_progress(progress_callback, progress, label, message, chain_id)

    def _run_with_progress_heartbeat(self, *args, **kwargs):
        return _ps.run_with_progress_heartbeat(*args, **kwargs)

    @staticmethod
    def _safe_prefetch_submit(state, fn, *args, **kwargs):
        return _ps.safe_prefetch_submit(state, fn, *args, **kwargs)

    def _run_step9_worker(self, state, window_abs_indices, total_chunks, doc_name,
                          verbose, verbose_steps, event_time, progress_callback,
                          step9_chunk_done_callback, control_callback=None):
        return _pw.run_step9_worker(
            self, state, window_abs_indices, total_chunks, doc_name,
            verbose, verbose_steps, event_time, progress_callback,
            step9_chunk_done_callback, control_callback, RememberControlFlow,
        )

    def _run_step10_worker(self, state, window_abs_indices, total_chunks, doc_name,
                           verbose, verbose_steps, event_time, progress_callback,
                           chunk_done_callback, control_callback=None):
        return _pw.run_step10_worker(
            self, state, window_abs_indices, total_chunks, doc_name,
            verbose, verbose_steps, event_time, progress_callback,
            chunk_done_callback, control_callback, RememberControlFlow,
        )

    @staticmethod
    def _summarize_window_timings(window_timings):
        return _pw.summarize_window_timings(window_timings)

    @staticmethod
    def _build_timing_summary(window_timings):
        return _pw.build_timing_summary(window_timings)

