"""
主处理流程：整合所有模块，实现完整的文档处理pipeline
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
import sys
import logging
import threading
import time

# Static defaults — computed once, not per call
_REMEMBER_DEFAULTS = {
    "mode": "multi_step",
    "anchor_recall_rounds": 1,
    "named_entity_recall_rounds": 1,
    "concrete_recall_rounds": 1,
    "abstract_recall_rounds": 1,
    "coverage_gap_rounds": 1,
    "missing_concept_rounds": 1,
    "entity_write_batch_size": 6,
    "entity_content_batch_size": 6,
    "relation_hint_rounds": 1,
    "relation_candidate_rounds": 1,
    "relation_expand_rounds": 1,
    "relation_write_rounds": 1,
    "pre_alignment_validation_retries": 2,
    "validation_retries": 2,
    "min_relation_candidates_per_window": 0,
    "min_entities_per_100_chars_soft_target": 0.0,
    "alignment_policy": "conservative",
}
import uuid

from .document import DocumentProcessor
from core.llm.client import (
    LLMClient,
    LLM_PRIORITY_STEP1, LLM_PRIORITY_STEP2, LLM_PRIORITY_STEP3,
    LLM_PRIORITY_STEP4, LLM_PRIORITY_STEP5, LLM_PRIORITY_STEP6, LLM_PRIORITY_STEP7,
)
from core.storage.embedding import EmbeddingClient
from core.storage.neo4j import Neo4jStorageManager as StorageManager
from core.storage import create_storage_manager
from .entity import EntityProcessor
from .relation import RelationProcessor
from core.models import Episode, Entity
from core.utils import (
    clear_parallel_log_context,
    compute_doc_hash,
    remember_log,
    set_pipeline_role,
    set_window_label,
    wprint,
    wprint_info,
)
from .alignment import _PipelineExtractionMixin, _AlignResult
from .helpers import dedupe_extraction_lists
from .steps import _ExtractionStepsMixin
from .cross_window import _CrossWindowDedupMixin

logger = logging.getLogger(__name__)


class RememberControlFlow(Exception):
    def __init__(self, action: str):
        super().__init__(action)
        self.remember_control_action = action


class TemporalMemoryGraphProcessor(_PipelineExtractionMixin, _ExtractionStepsMixin, _CrossWindowDedupMixin):
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

        def _remember_pick(primary_key: str, fallback_key: Optional[str] = None):
            if primary_key in _remember_overrides:
                return _remember_overrides.get(primary_key)
            if primary_key in _remember_from_config:
                return _remember_from_config.get(primary_key)
            if fallback_key:
                if fallback_key in _remember_overrides:
                    return _remember_overrides.get(fallback_key)
                if fallback_key in _remember_from_config:
                    return _remember_from_config.get(fallback_key)
            return _remember_cfg.get(primary_key)

        self.remember_mode = str(_remember_cfg.get("mode") or "dual_model").strip() or "dual_model"
        if self.remember_mode not in {"standard", "dual_model"}:
            self.remember_mode = "dual_model"
        self.remember_anchor_recall_rounds = max(1, int(_remember_pick("anchor_recall_rounds") or 1))
        _named_rounds = _remember_pick("named_entity_recall_rounds", "concrete_recall_rounds")
        self.remember_named_entity_recall_rounds = max(1, int(_named_rounds or 1))
        self.remember_concrete_recall_rounds = self.remember_named_entity_recall_rounds
        self.remember_abstract_recall_rounds = max(1, int(_remember_pick("abstract_recall_rounds") or 1))
        _coverage_gap_rounds = _remember_pick("coverage_gap_rounds", "missing_concept_rounds")
        self.remember_coverage_gap_rounds = max(1, int(_coverage_gap_rounds or 1))
        self.remember_missing_concept_rounds = self.remember_coverage_gap_rounds
        _entity_write_batch_size = _remember_pick("entity_write_batch_size", "entity_content_batch_size")
        self.remember_entity_write_batch_size = max(1, int(_entity_write_batch_size or 6))
        self.remember_entity_content_batch_size = self.remember_entity_write_batch_size
        _relation_hint_rounds = _remember_pick("relation_hint_rounds", "relation_candidate_rounds")
        self.remember_relation_hint_rounds = max(1, int(_relation_hint_rounds or 1))
        self.remember_relation_candidate_rounds = self.remember_relation_hint_rounds
        self.remember_relation_expand_rounds = max(1, int(_remember_pick("relation_expand_rounds") or 1))
        self.remember_relation_write_rounds = max(1, int(_remember_pick("relation_write_rounds") or 1))
        _pre_validation_retries = _remember_pick("pre_alignment_validation_retries", "validation_retries")
        self.remember_pre_alignment_validation_retries = max(0, int(_pre_validation_retries or 0))
        self.remember_validation_retries = self.remember_pre_alignment_validation_retries
        self.remember_min_relation_candidates_per_window = max(
            0, int(_remember_pick("min_relation_candidates_per_window") or 0)
        )
        self.remember_min_entities_per_100_chars_soft_target = max(
            0.0, float(_remember_pick("min_entities_per_100_chars_soft_target") or 0.0)
        )
        self.remember_alignment_policy = str(_remember_cfg.get("alignment_policy") or "conservative").strip() or "conservative"
        self.remember_alignment_conservative = self.remember_alignment_policy == "conservative"
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
            _max_concurrent_windows = max_concurrent_windows
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
                 entity_rounds: Optional[int] = None,
                 relation_rounds: Optional[int] = None,
                 entity_refine_rounds: Optional[int] = None,
                 relation_refine_rounds: Optional[int] = None,
                 remember_config: Optional[Dict[str, Any]] = None,
                 extraction_llm: Optional[Dict[str, Any]] = None,
                 graph_id: Optional[str] = None,
                 embedding_cache_max_size: Optional[int] = None,
                 embedding_cache_ttl: Optional[float] = None):
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
            entity_refine_rounds: 实体精炼轮次（默认 2）
            relation_refine_rounds: 关系精炼轮次（默认 1）
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
            cache_ttl=embedding_cache_ttl or 300.0
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
            self.storage = StorageManager(
                storage_path,
                embedding_client=self.embedding_client,
                entity_content_snippet_length=_content_snippet_length,
                relation_content_snippet_length=_relation_content_snippet_length
            )
        self.document_processor = DocumentProcessor(window_size, overlap)
        _al = alignment_llm or {}
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
            distill_data_dir=distill_data_dir,
            alignment_enabled=bool(_al.get("enabled", False)),
            alignment_max_llm_concurrency=_al.get("max_concurrency"),
            alignment_base_url=_al.get("base_url"),
            alignment_api_key=_al.get("api_key"),
            alignment_model=_al.get("model"),
            alignment_max_tokens=_al.get("max_tokens"),
            alignment_think_mode=_al.get("think_mode"),
            alignment_content_snippet_length=_al.get("content_snippet_length"),
            alignment_relation_content_snippet_length=_al.get("relation_content_snippet_length"),
        )
        self.entity_processor = EntityProcessor(
            self.storage, 
            self.llm_client,
            max_similar_entities=_max_similar_entities,
            content_snippet_length=_content_snippet_length
        )
        self.relation_processor = RelationProcessor(self.storage, self.llm_client)
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
        
        # Pipeline rounds (new name with old name fallback)
        _er = entity_rounds if entity_rounds is not None else entity_refine_rounds
        self.entity_rounds = _er if _er is not None else 2
        _rr = relation_rounds if relation_rounds is not None else relation_refine_rounds
        self.relation_rounds = _rr if _rr is not None else 1

        # Extraction client (dual-model pipeline)
        _el = extraction_llm or {}
        self.extraction_client = None
        self.extraction_client_enabled = False
        if _el.get("enabled", False):
            self.extraction_client = LLMClient(
                _el.get("api_key", llm_api_key),
                _el.get("model", llm_model),
                _el.get("base_url", llm_base_url),
                content_snippet_length=_content_snippet_length,
                relation_content_snippet_length=_relation_content_snippet_length,
                embedding_client=self.embedding_client,
                think_mode=bool(_el.get("think_mode", False)),
                max_tokens=_el.get("max_tokens"),
                context_window_tokens=int(_el.get("context_window_tokens", _ctx_win)),
                max_llm_concurrency=_el.get("max_concurrency"),
                alignment_enabled=False,
            )
            self.extraction_client_enabled = True
            if self.remember_mode not in ("standard", "legacy"):
                self.remember_mode = "dual_model"

        self.llm_threads = max_llm_concurrency if max_llm_concurrency else 1
        self.load_cache_memory = load_cache_memory if load_cache_memory is not None else False

        self.jaccard_search_threshold = jaccard_search_threshold
        self.embedding_name_search_threshold = embedding_name_search_threshold
        self.embedding_full_search_threshold = embedding_full_search_threshold
        self.max_alignment_candidates = max_alignment_candidates

        # --- Phase 3: Threading ---
        self.current_episode: Optional[Episode] = None
        self._init_threading(max_concurrent_windows, max_llm_concurrency)

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
        elif self.llm_client and hasattr(self.llm_client, "_llm_semaphore") and self.llm_client._llm_semaphore:
            sem = self.llm_client._llm_semaphore
            stats["llm_semaphore_active"] = sem.active_count
            stats["llm_semaphore_max"] = sem.max_value
        return stats

    def _acquire_window_slot(self) -> None:
        """与 _release_window_slot 成对；占用槽即计入主链窗口（步骤1–5 阶段可见）。"""
        self._window_slot.acquire()
        with self._runtime_lock:
            self._active_main_pipeline_windows += 1

    def _release_window_slot(self) -> None:
        self._window_slot.release()
        with self._runtime_lock:
            self._active_main_pipeline_windows = max(0, self._active_main_pipeline_windows - 1)

    def _run_extraction_job(
        self,
        new_episode: Episode,
        input_text: str,
        document_name: str,
        verbose: bool = True,
        verbose_steps: bool = True,
        event_time: Optional[datetime] = None,
    ):
        with self._runtime_lock:
            self._active_window_extractions += 1
            self._peak_window_extractions = max(
                self._peak_window_extractions,
                self._active_window_extractions,
            )
        try:
            return self._process_extraction(
                new_episode,
                input_text,
                document_name,
                verbose=verbose,
                verbose_steps=verbose_steps,
                event_time=event_time,
            )
        finally:
            with self._runtime_lock:
                self._active_window_extractions = max(0, self._active_window_extractions - 1)
            self._release_window_slot()

    def process_documents(self, document_paths: List[str], verbose: bool = True,
                         entity_progress_verbose: Optional[bool] = None,
                         similarity_threshold: Optional[float] = None,
                         max_similar_entities: Optional[int] = None,
                         content_snippet_length: Optional[int] = None,
                         relation_content_snippet_length: Optional[int] = None,
                         load_cache_memory: Optional[bool] = None,
                         jaccard_search_threshold: Optional[float] = None,
                         embedding_name_search_threshold: Optional[float] = None,
                         embedding_full_search_threshold: Optional[float] = None):
        """
        处理多个文档

        Args:
            document_paths: 文档路径列表
            verbose: 是否输出详细信息
            entity_progress_verbose: 是否输出实体对齐的逐条树状进度（默认与 verbose 相同；服务场景可传 False）
            similarity_threshold: 实体搜索相似度阈值（可选，覆盖初始化时的设置）
            max_similar_entities: 语义向量初筛后返回的最大相似实体数量（可选，覆盖初始化时的设置）
            content_snippet_length: 用于相似度搜索的实体content截取长度（可选，覆盖初始化时的设置）
            relation_content_snippet_length: 用于embedding计算的关系content截取长度（可选，覆盖初始化时的设置）
            load_cache_memory: 是否加载缓存记忆（可选，覆盖初始化时的设置）
            jaccard_search_threshold: Jaccard搜索（name_only）的相似度阈值（可选，默认使用similarity_threshold）
            embedding_name_search_threshold: Embedding搜索（name_only）的相似度阈值（可选，默认使用similarity_threshold）
            embedding_full_search_threshold: Embedding搜索（name+content）的相似度阈值（可选，默认使用similarity_threshold）
        """
        # 保存原始值，以便在方法结束时恢复
        original_values = {}
        original_components = {}
        # 子对象属性（storage/llm_client 的属性被就地修改，setattr 恢复组件引用不会还原它们）
        _original_sub_attrs = {}
        
        # 如果提供了参数，临时覆盖实例属性
        if similarity_threshold is not None:
            original_values['similarity_threshold'] = self.similarity_threshold
            self.similarity_threshold = similarity_threshold
        
        # 处理三种搜索方法的独立阈值
        if jaccard_search_threshold is not None:
            original_values['jaccard_search_threshold'] = self.jaccard_search_threshold
            self.jaccard_search_threshold = jaccard_search_threshold
        if embedding_name_search_threshold is not None:
            original_values['embedding_name_search_threshold'] = self.embedding_name_search_threshold
            self.embedding_name_search_threshold = embedding_name_search_threshold
        if embedding_full_search_threshold is not None:
            original_values['embedding_full_search_threshold'] = self.embedding_full_search_threshold
            self.embedding_full_search_threshold = embedding_full_search_threshold
        
        # 先更新属性值，然后统一更新组件
        need_update_entity_processor = False
        final_max_similar_entities = self.max_similar_entities
        final_content_snippet_length = self.content_snippet_length
        
        if max_similar_entities is not None:
            original_values['max_similar_entities'] = self.max_similar_entities
            self.max_similar_entities = max_similar_entities
            final_max_similar_entities = max_similar_entities
            need_update_entity_processor = True
        
        if content_snippet_length is not None:
            original_values['content_snippet_length'] = self.content_snippet_length
            self.content_snippet_length = content_snippet_length
            final_content_snippet_length = content_snippet_length
            # 保存子对象原始属性值（setattr 恢复同一对象引用不会还原这些修改）
            if 'storage.entity_content_snippet_length' not in _original_sub_attrs:
                _original_sub_attrs['storage.entity_content_snippet_length'] = self.storage.entity_content_snippet_length
            if 'llm_client.content_snippet_length' not in _original_sub_attrs:
                _original_sub_attrs['llm_client.content_snippet_length'] = self.llm_client.content_snippet_length
            self.storage.entity_content_snippet_length = content_snippet_length
            self.llm_client.content_snippet_length = content_snippet_length
            need_update_entity_processor = True
        
        # 统一更新 EntityProcessor（如果需要）
        if need_update_entity_processor:
            if 'entity_processor' not in original_components:
                original_components['entity_processor'] = self.entity_processor
            self.entity_processor = EntityProcessor(
                self.storage,
                self.llm_client,
                max_similar_entities=final_max_similar_entities,
                content_snippet_length=final_content_snippet_length
            )
        if relation_content_snippet_length is not None:
            original_values['relation_content_snippet_length'] = self.relation_content_snippet_length
            self.relation_content_snippet_length = relation_content_snippet_length
            # 保存子对象原始属性值
            if 'storage.relation_content_snippet_length' not in _original_sub_attrs:
                _original_sub_attrs['storage.relation_content_snippet_length'] = self.storage.relation_content_snippet_length
            self.storage.relation_content_snippet_length = relation_content_snippet_length
        if load_cache_memory is not None:
            original_values['load_cache_memory'] = self.load_cache_memory
            self.load_cache_memory = load_cache_memory

        _saved_entity_progress_verbose = self.entity_processor.entity_progress_verbose
        _epv = entity_progress_verbose if entity_progress_verbose is not None else verbose
        try:
            self.entity_processor.entity_progress_verbose = _epv
            if verbose:
                wprint_info(f"开始处理 {len(document_paths)} 个文档...")
            
            # 断点续传相关变量
            resume_document_path = None
            resume_text = None
            
            # 根据配置决定是否加载最新的记忆缓存并支持断点续传
            if self.load_cache_memory:
                if verbose:
                    wprint_info("正在加载最新的缓存记忆...")

                # 获取最新缓存的元数据（包含 text 和 document_path）
                # 只查找"文档处理"类型的缓存，避免使用知识图谱整理产生的缓存（其text字段是整理后的实体信息，不是原始文档文本）
                latest_metadata = self.storage.get_latest_episode_metadata(activity_type="文档处理")
                
                if latest_metadata:
                    # 加载缓存记忆
                    self.current_episode = self.storage.load_episode(latest_metadata['absolute_id'])
                    
                    if self.current_episode:
                        if verbose:
                            wprint_info(f"已加载缓存记忆: {self.current_episode.absolute_id} (时间: {self.current_episode.event_time})")
                        
                        # 提取断点续传信息
                        resume_document_path = latest_metadata.get('document_path', '')
                        resume_text = latest_metadata.get('text', '')
                        
                        if verbose:
                            if resume_document_path:
                                wprint_info(f"[断点续传] 上次处理的文档: {resume_document_path}")
                            if resume_text:
                                text_preview = resume_text[:100].replace('\n', ' ')
                                wprint_info(f"[断点续传] 上次处理的文本片段: {text_preview}...")
                else:
                    if verbose:
                        wprint_info("未找到缓存记忆，将从头开始处理")
                    self.current_episode = None
            else:
                if verbose:
                    wprint_info("不加载缓存记忆，将从头开始处理")
                self.current_episode = None
            
            # 遍历所有文档的滑动窗口（支持断点续传）
            for chunk_idx, (input_text, document_name, is_new_document, text_start_pos, text_end_pos, total_text_length, document_path) in enumerate(
                self.document_processor.process_documents(
                    document_paths,
                    resume_document_path=resume_document_path,
                    resume_text=resume_text
                )
            ):
                if verbose:
                    wprint_info(f"\n处理窗口 {chunk_idx + 1} (文档: {document_name}, 位置: {text_start_pos}-{text_end_pos}/{total_text_length})")
                elif _epv:
                    wprint_info(f"窗口 {chunk_idx + 1} 开始 · {document_name}")
                
                # 处理当前窗口
                self._process_window(input_text, document_name, is_new_document, 
                                    text_start_pos, text_end_pos, total_text_length, verbose,
                                    verbose_steps=_epv, document_path=document_path)
        finally:
            # 恢复原始值
            for key, value in original_values.items():
                setattr(self, key, value)
            # 恢复原始组件
            for key, value in original_components.items():
                setattr(self, key, value)
            # 恢复子对象属性（storage/llm_client 的属性被就地修改，组件引用还原不会覆盖它们）
            for attr_path, value in _original_sub_attrs.items():
                obj_name, attr_name = attr_path.split('.', 1)
                setattr(getattr(self, obj_name), attr_name, value)
            self.entity_processor.entity_progress_verbose = _saved_entity_progress_verbose

    # ------------------------------------------------------------------
    # remember_text helpers (shared state, control flow, workers)
    # ------------------------------------------------------------------

    def _init_remember_shared_state(self, N):
        """Pre-allocate arrays, events, and error collectors for N windows."""
        import types
        s = types.SimpleNamespace()
        s.N = N
        s.episodes = [None] * N
        s.input_texts = [None] * N
        s.extract_results = [None] * N
        s.align_results = [None] * N
        s.step10_results = [None] * N
        s.window_timings = [{} for _ in range(N)]
        s.extract_done = [threading.Event() for _ in range(N)]
        s.step9_done_ev = [threading.Event() for _ in range(N)]
        s.step10_done_ev = [threading.Event() for _ in range(N)]
        s.errors = []
        s.errors_lock = threading.Lock()
        s.window_failures = [None] * N
        s.control_lock = threading.Lock()
        s.control_state = {"action": None}
        s.prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tmg-chain-prefetch")
        return s

    @staticmethod
    def _record_window_error(state, stage, idx, exc) -> bool:
        with state.errors_lock:
            if state.window_failures[idx] is None:
                state.window_failures[idx] = (stage, exc)
                state.errors.append((stage, idx, exc))
                return True
        return False

    @staticmethod
    def _signal_control_stop(state, action, from_index, *,
                              set_extract=True, set_step9=True, set_step10=True):
        with state.control_lock:
            if state.control_state["action"] is None:
                state.control_state["action"] = action
            _from = max(0, min(from_index, state.N))
            for j in range(_from, state.N):
                if set_extract:
                    state.extract_done[j].set()
                if set_step9:
                    state.step9_done_ev[j].set()
                if set_step10:
                    state.step10_done_ev[j].set()

    @staticmethod
    def _poll_control(state, control_callback):
        action = state.control_state["action"]
        if action:
            return action
        if control_callback is None:
            return None
        action = control_callback()
        if action in ("pause", "cancel"):
            with state.control_lock:
                if state.control_state["action"] is None:
                    state.control_state["action"] = action
                return state.control_state["action"]
        return None

    @staticmethod
    def _safe_progress(progress_callback, progress, label, message, chain_id="step9"):
        if not progress_callback:
            return
        progress_callback(progress, label, message, chain_id)

    def _run_with_progress_heartbeat(
        self,
        run_fn: Callable[[], Any],
        *,
        chain_id: str,
        base_progress: float,
        phase_label: str,
        message: str,
        window_label: str,
        pipeline_role: str,
        progress_callback=None,
        heartbeat_seconds: float = 5.0,
        log_interval_seconds: float = 30.0,
    ) -> Any:
        """为长耗时步骤补充心跳，避免前端/日志长时间停在同一标签像"卡死"。"""
        stop_ev = threading.Event()
        started = time.time()

        def _heartbeat() -> None:
            last_log_elapsed = 0.0
            set_window_label(window_label)
            set_pipeline_role(pipeline_role)
            try:
                while not stop_ev.wait(heartbeat_seconds):
                    elapsed = max(1, int(time.time() - started))
                    hb_label = f"{phase_label} · 已等待 {elapsed}s"
                    hb_message = f"{message}（已等待 {elapsed}s）"
                    self._safe_progress(progress_callback, base_progress, hb_label, hb_message, chain_id)
                    if elapsed - last_log_elapsed >= log_interval_seconds:
                        wprint_info(f"{phase_label} · 长调用进行中（已等待 {elapsed}s）")
                        last_log_elapsed = float(elapsed)
            finally:
                clear_parallel_log_context()

        hb = threading.Thread(
            target=_heartbeat,
            name=f"tmg-heartbeat-{chain_id}",
            daemon=True,
        )
        hb.start()
        try:
            return run_fn()
        finally:
            stop_ev.set()
            hb.join(timeout=0.2)

    @staticmethod
    def _safe_prefetch_submit(state, fn, *args, **kwargs):
        """解释器收尾或 Executor 已 shutdown 时 submit 会失败；返回 None 表示跳过预取。"""
        try:
            if sys.is_finalizing():
                return None
        except Exception:
            pass
        try:
            return state.prefetch_executor.submit(fn, *args, **kwargs)
        except RuntimeError:
            return None

    def _run_step9_worker(self, state, start_chunk, total_chunks, doc_name,
                          verbose, verbose_steps, event_time, progress_callback,
                          step9_chunk_done_callback):
        """Step6 worker thread: entity alignment, chained across windows."""
        _already_versioned = set()
        _emb_available = bool(self.storage.embedding_client and self.storage.embedding_client.is_available())
        for i in range(state.N):
            state.extract_done[i].wait()
            _action = self._poll_control(state, None)
            if _action:
                self._signal_control_stop(state, _action, i, set_extract=False, set_step9=True, set_step10=True)
                break
            set_window_label(f"W{start_chunk + i + 1}/{total_chunks}")
            set_pipeline_role("步骤9")
            _er = state.extract_results[i]
            emb_prefetch_future = None
            if _er is not None:
                _ents, _ = _er
                if _ents and _emb_available:
                    emb_prefetch_future = self._safe_prefetch_submit(
                        state,
                        self.entity_processor.encode_entities_for_candidate_table,
                        _ents,
                    )
            if i > 0:
                state.step9_done_ev[i - 1].wait()
            _action = self._poll_control(state, None)
            if _action:
                self._signal_control_stop(state, _action, i, set_extract=False, set_step9=True, set_step10=True)
                break
            with self._runtime_lock:
                self._active_step9 += 1
            _t_step9_start = time.time()
            try:
                mc = state.episodes[i]
                _success = False
                if _er is None:
                    _upstream = state.window_failures[i]
                    if _upstream is not None:
                        _stage, _exc = _upstream
                        if verbose or verbose_steps:
                            wprint_info(f"【步骤9】跳过｜上游｜{_stage} {_exc}")
                        continue
                    raise RuntimeError(
                        f"step9 skipped for window {start_chunk + i}: extract result is None (extraction failed)"
                    )
                ents, rels = _er
                if verbose:
                    wprint_info("【步骤9】实体｜就绪｜本窗1–5完成或缓存")
                elif verbose_steps:
                    wprint_info("【步骤9】实体｜开始｜前置1–5已就绪")
                _wi = start_chunk + i
                _g_lo = _wi / total_chunks
                _g_hi = (_wi + 1) / total_chunks
                _span = _g_hi - _g_lo
                _pr_step9 = (_g_lo + _span * (8.0 / 10.0), _g_lo + _span * (9.0 / 10.0))
                ar = self._align_entities(
                    ents, rels, mc, state.input_texts[i], doc_name,
                    verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
                    progress_callback=lambda p, l, m: self._safe_progress(progress_callback, p, l, m, "step9"),
                    progress_range=_pr_step9,
                    window_index=start_chunk + i, total_windows=total_chunks,
                    entity_embedding_prefetch=emb_prefetch_future,
                    already_versioned_family_ids=_already_versioned,
                    window_timings_ref=state.window_timings[i],
                )
                state.align_results[i] = ar
                _success = True
                _step9_elapsed = time.time() - _t_step9_start
                state.window_timings[i]["step9"] = _step9_elapsed
                if verbose or verbose_steps:
                    wprint_info(f"【步骤9】完成｜{_step9_elapsed:.1f}s")
            except Exception as e:
                if self._record_window_error(state, "step9", i, e):
                    logger.error("step9 window %d error: %s", i, e, exc_info=True)
            finally:
                with self._runtime_lock:
                    self._active_step9 = max(0, self._active_step9 - 1)
                state.step9_done_ev[i].set()
                # Free raw extraction data now that step9 has consumed it
                # NOTE: Do NOT nullify state.input_texts[i] here — step10 still needs it
                if _success:
                    state.extract_results[i] = None
                if _success and step9_chunk_done_callback:
                    step9_chunk_done_callback(start_chunk + i + 1)
                clear_parallel_log_context()

    def _run_step10_worker(self, state, start_chunk, total_chunks, doc_name,
                          verbose, verbose_steps, event_time, progress_callback,
                          chunk_done_callback):
        """Step7 worker thread: relation alignment, chained across windows."""
        for i in range(state.N):
            state.step9_done_ev[i].wait()
            _action = self._poll_control(state, None)
            if _action:
                self._signal_control_stop(state, _action, i, set_extract=False, set_step9=False, set_step10=True)
                break
            set_window_label(f"W{start_chunk + i + 1}/{total_chunks}")
            set_pipeline_role("步骤10")
            ar = state.align_results[i]
            step10_inputs_cache = None
            rel_prefetch_future = None
            if ar is not None:
                try:
                    step10_inputs_cache = self._build_step10_relation_inputs_from_align_result(ar)
                    _ri, _eid, _, _ = step10_inputs_cache
                    if i > 0 and _ri:
                        rel_prefetch_future = self._safe_prefetch_submit(
                            state,
                            self.relation_processor.build_relations_by_pair_from_inputs,
                            _ri,
                            _eid,
                        )
                except Exception as exc:
                    wprint_info(f"  │  step10 输入构建失败: {exc}")
                    step10_inputs_cache = None
                    rel_prefetch_future = None
            if i > 0:
                state.step10_done_ev[i - 1].wait()
            _action = self._poll_control(state, None)
            if _action:
                self._signal_control_stop(state, _action, i, set_extract=False, set_step9=False, set_step10=True)
                break
            prepared_relations_by_pair = None
            if rel_prefetch_future is not None:
                try:
                    prepared_relations_by_pair, _ = rel_prefetch_future.result()
                except Exception as exc:
                    wprint_info(f"  │  关系预取结果获取失败: {exc}")
                    prepared_relations_by_pair = None
            with self._runtime_lock:
                self._active_step10 += 1
            _t_step10_start = time.time()
            _success = False
            _window_has_entities = False
            try:
                if ar is None:
                    _upstream = state.window_failures[i]
                    if _upstream is not None:
                        _stage, _exc = _upstream
                        if verbose or verbose_steps:
                            wprint_info(f"【步骤10】跳过｜上游｜{_stage} {_exc}")
                        continue
                    raise RuntimeError(
                        f"step9 result for window {start_chunk + i} is None"
                    )
                mc = state.episodes[i]
                _wi = start_chunk + i
                _g_lo = _wi / total_chunks
                _g_hi = (_wi + 1) / total_chunks
                _span = _g_hi - _g_lo
                _pr_step10 = (_g_lo + _span * (9.0 / 10.0), _g_hi)
                processed_rels = self._align_relations(
                    ar, mc, state.input_texts[i], doc_name,
                    verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
                    progress_callback=lambda p, l, m: self._safe_progress(progress_callback, p, l, m, "step10"),
                    progress_range=_pr_step10,
                    window_index=start_chunk + i, total_windows=total_chunks,
                    prepared_relations_by_pair=prepared_relations_by_pair,
                    step10_inputs_cache=step10_inputs_cache,
                    window_timings_ref=state.window_timings[i],
                )
                state.step10_results[i] = processed_rels
                _success = True
                _window_has_entities = bool(ar.unique_entities)
                _step10_elapsed = time.time() - _t_step10_start
                state.window_timings[i]["step10"] = _step10_elapsed
                if verbose or verbose_steps:
                    wprint_info(f"【步骤10】完成｜{_step10_elapsed:.1f}s")

                # Phase C-2: Record Episode → Relation MENTIONS
                if processed_rels:
                    try:
                        _rel_abs_ids = list(set(
                            r.absolute_id for r in processed_rels if r.absolute_id
                        ))
                        if _rel_abs_ids:
                            self.storage.save_episode_mentions(
                                mc.absolute_id, _rel_abs_ids,
                                target_type="relation",
                            )
                            if verbose or verbose_steps:
                                wprint_info(f"【步骤10】MENTIONS｜Relation｜{len(_rel_abs_ids)}条")
                    except Exception as _me:
                        logger.warning("Relation MENTIONS 记录失败: %s", _me)

                if _window_has_entities:
                    try:
                        _orphan_count = self._cleanup_orphaned_entities(
                            ar.unique_entities,
                            verbose=verbose or verbose_steps,
                            window_text=state.input_texts[i],
                            all_entity_names=[e.name for e in ar.unique_entities] if ar.unique_entities else [],
                            episode_id=getattr(mc, 'cache_id', ''),
                            source_document=doc_name,
                        )
                        if _orphan_count > 0:
                            _window_has_entities = bool(ar.unique_entities) and _orphan_count < len(ar.unique_entities)
                    except Exception as _oe:
                        logger.warning("孤立实体清理失败: %s", _oe)
            except Exception as e:
                if self._record_window_error(state, "step10", i, e):
                    logger.error("step10 window %d error: %s", i, e, exc_info=True)
            finally:
                with self._runtime_lock:
                    self._active_step10 = max(0, self._active_step10 - 1)
                state.step10_done_ev[i].set()
                # Free alignment data now that step10 has consumed it
                if _success:
                    state.align_results[i] = None
                    state.input_texts[i] = None
                    state.episodes[i] = None
                if _success and chunk_done_callback:
                    chunk_done_callback(start_chunk + i + 1)
                if _success and not _window_has_entities:
                    wprint_info("提示: step10 完成但本窗无实体，仍已计入进度（避免断点卡死）")
                clear_parallel_log_context()

    @staticmethod
    def _summarize_window_timings(window_timings):
        """Log timing summary across all windows."""
        _all_steps = ["step1", "step2-8", "step9", "step10"]
        _step_labels = {"step1": "1-缓存", "step2-8": "2-8-抽取", "step9": "9-实体对齐", "step10": "10-关系对齐"}
        _sub_step_labels = {
            "step2_entity_extract": "2-实体提取",
            "step3_entity_dedup": "3-实体去重",
            "step4_entity_content": "4-实体内容",
            "step5_entity_quality": "5-实体质量门",
            "step6_relation_discovery": "6-关系发现",
            "step7_relation_content": "7-关系内容",
            "step8_relation_quality": "8-关系质量门",
            "step9-process_entities": "9-实体处理",
            "step9-dedup_merge": "9-同名去重",
            "step10-process_relations": "10-关系处理",
        }
        _step_totals = {s: 0.0 for s in _all_steps}
        _sub_totals = {k: 0.0 for k in _sub_step_labels}
        for _wt in window_timings:
            for _s in _all_steps:
                _step_totals[_s] += _wt.get(_s, 0.0)
            for _sk in _sub_step_labels:
                _sub_totals[_sk] += _wt.get(_sk, 0.0)
        _total_elapsed = sum(_step_totals.values())
        if _total_elapsed > 0:
            _timing_detail = " | ".join(
                f"{_step_labels[s]}:{_step_totals[s]:.1f}s"
                for s in _all_steps if _step_totals[s] > 0
            )
            remember_log(f"计时汇总｜共{_total_elapsed:.1f}s｜{_timing_detail}")
            _active_subs = {k: v for k, v in _sub_totals.items() if v > 0.01}
            if _active_subs:
                _sub_detail = " | ".join(
                    f"{_sub_step_labels[k]}:{v:.1f}s"
                    for k, v in sorted(_active_subs.items(), key=lambda x: -x[1])
                )
                remember_log(f"子步骤明细｜{_sub_detail}")

    def remember_text(self, text: str, doc_name: str = "", verbose: bool = False,
                      verbose_steps: bool = True,
                      load_cache_memory: Optional[bool] = None,
                      event_time: Optional[datetime] = None,
                      document_path: str = "",
                      progress_callback: Optional[Callable] = None,
                      control_callback: Optional[Callable[[], Optional[str]]] = None,
                      start_chunk: int = 0,
                      main_chunk_done_callback: Optional[Callable] = None,
                      step9_chunk_done_callback: Optional[Callable] = None,
                      chunk_done_callback: Optional[Callable] = None,
                      source_document: Optional[str] = None) -> Dict:
        """
        将一段文本作为记忆入库：流水线式并行处理 step9（实体对齐）和 step10（关系对齐）。

        流水线架构：
        - 主线程：Phase A（step1 串行更新缓存）+ 提交 Phase B（step2-8 并行抽取）
        - step9 线程：等待当前窗口 step2-8 完成 + 前一窗口 step9 完成 → 实体对齐
        - step10 线程：等待当前窗口 step9 完成 + 前一窗口 step10 完成 → 关系对齐
        - step9 W(i+1) 可与 step10 W(i) 并行执行

        Args:
            text: 原始文本内容
            doc_name: 文档/来源名称
            verbose: 是否打印详细处理日志（步骤内细节、LLM 提示等）
            verbose_steps: 是否在控制台输出步骤级「开始/结束」汇报（verbose=True 时仍生效，但以详细日志为准）
                并行时控制台行格式为 [窗号][角色] 正文；角色为 主线程 / 抽取 / 步骤9 / 步骤10 之一。
            load_cache_memory: 是否在开始前加载最新缓存记忆再追加
            event_time: 事件实际发生时间
            document_path: 原文文件路径
            progress_callback: 进度回调 fn(progress, phase_label, message, chain_id)
            control_callback: 控制回调 fn() -> {"pause","cancel",None}，在窗口级安全点生效
            start_chunk: 从第几个窗口开始（关系链断点续传）
            main_chunk_done_callback: 步骤1–5 完成一个窗口后的回调 fn(processed_count)
            step9_chunk_done_callback: 步骤9 完成一个窗口后的回调 fn(processed_count)
            chunk_done_callback: 步骤10 完成一个窗口后的回调 fn(processed_count)
            source_document: 来源文档名称（优先于 doc_name）

        Returns:
            dict: episode_id, chunks_processed, storage_path
        """
        doc_name = source_document or doc_name

        # Input validation: reject empty or whitespace-only text early.
        if not text or not text.strip():
            return {
                "episode_id": None,
                "chunks_processed": 0,
                "storage_path": str(self.storage.storage_path),
                "entities": 0,
                "relations": 0,
                "warnings": [{"phase": "input_validation", "error": "text is empty or whitespace-only"}],
            }

        use_load_cache = load_cache_memory if load_cache_memory is not None else self.load_cache_memory
        # 仅在真正的断点续传（start_chunk > 0）时加载已有缓存链；
        # start_chunk == 0 表示从头开始，加载旧缓存会导致 step1 重复处理已有内容
        if use_load_cache and start_chunk > 0:
            latest_metadata = self.storage.get_latest_episode_metadata(activity_type="文档处理")
            if latest_metadata:
                self.current_episode = self.storage.load_episode(latest_metadata["absolute_id"])
                if verbose and self.current_episode:
                    remember_log(
                        f"已加载缓存记忆: {self.current_episode.absolute_id}，"
                        f"将在此链上追加（断点续传 start_chunk={start_chunk}）"
                    )
                elif verbose_steps and self.current_episode:
                    remember_log("已加载缓存记忆（断点续传）")
            else:
                self.current_episode = None
        else:
            self.current_episode = None
            if start_chunk == 0 and use_load_cache:
                if verbose:
                    remember_log("start_chunk=0，从头开始处理，不加载旧缓存链")
                elif verbose_steps:
                    remember_log("从头开始处理（不加载旧缓存链）")

        if not document_path:
            document_path = f"api://{uuid.uuid4().hex}"
        window_size = self.document_processor.window_size
        overlap = self.document_processor.overlap
        total_length = len(text)

        # 计算总窗口数
        stride = max(1, window_size - overlap)
        if total_length <= window_size:
            total_chunks = 1
        else:
            total_chunks = 1 + (max(total_length - window_size, 0) + stride - 1) // stride

        # 所有窗口已处理完毕（断点续传恢复后无需重跑）
        if start_chunk >= total_chunks:
            return {
                "episode_id": getattr(self.current_episode, 'absolute_id', None),
                "chunks_processed": total_chunks,
                "storage_path": str(self.storage.storage_path),
            }

        N = total_chunks - start_chunk  # 待处理窗口数
        last_episode_id = None
        clear_parallel_log_context()

        # 预分配共享状态
        state = self._init_remember_shared_state(N)

        # 启动 step9 / step10 线程
        t9 = threading.Thread(target=self._run_step9_worker, name="tmg-step9-chain", daemon=True,
                              args=(state, start_chunk, total_chunks, doc_name, verbose, verbose_steps,
                                    event_time, progress_callback, step9_chunk_done_callback))
        t10 = threading.Thread(target=self._run_step10_worker, name="tmg-step10-chain", daemon=True,
                              args=(state, start_chunk, total_chunks, doc_name, verbose, verbose_steps,
                                    event_time, progress_callback, chunk_done_callback))
        t9.start()
        t10.start()

        if verbose or verbose_steps:
            remember_log(
                "并行流水线 · 日志前缀 [窗号][角色]："
                "主线程=步骤1+提交抽取；抽取=步骤2–5；步骤9/7=链式线程。"
                "不同窗会交错，属正常。"
            )

        # ========== 主线程：Phase A（step1 串行）+ 提交 Phase B（step2-8）==========
        try:
            start = start_chunk * stride

            for ci in range(N):
                _action = self._poll_control(state, control_callback)
                if _action:
                    self._signal_control_stop(state, _action, ci)
                    break
                self._acquire_window_slot()
                _slot_acquired = True

                try:
                    _action = self._poll_control(state, control_callback)
                    if _action:
                        self._signal_control_stop(state, _action, ci)
                        self._release_window_slot()
                        _slot_acquired = False
                        break

                    end = min(start + window_size, total_length)
                    chunk = text[start:end]
                    if start == 0:
                        chunk = f"[文档元数据] 文档名：{doc_name} [/文档元数据]\n\n{chunk}"

                    _wlabel = f"W{start_chunk + ci + 1}/{total_chunks}"
                    if verbose:
                        set_window_label(_wlabel)
                        set_pipeline_role("主线程")
                        wprint_info(
                            f"【窗口】{_wlabel}｜{doc_name}｜[{start}-{end}/{total_length}] {len(chunk)}字"
                        )
                    elif verbose_steps:
                        set_window_label(_wlabel)
                        set_pipeline_role("主线程")
                        wprint_info(
                            f"【窗口】{_wlabel}｜{doc_name}｜[{start}-{end}/{total_length}]"
                        )

                    _wi = start_chunk + ci
                    _g_lo = _wi / total_chunks
                    _g_hi = (_wi + 1) / total_chunks
                    _span = _g_hi - _g_lo
                    _p_after_step1 = _g_lo + _span * (1.0 / 7.0)
                    _p_end_main = _g_lo + _span * (8.0 / 10.0)
                    if progress_callback:
                        self._safe_progress(progress_callback,
                            _g_lo + _span * 0.02,
                            f"窗口 {start_chunk + ci + 1}/{total_chunks} · 步骤1/7 进行中",
                            "", "main",
                        )

                    # Step1: 更新缓存
                    _t_step1_start = time.time()
                    _chunk_hash = compute_doc_hash(chunk)
                    existing_mc, _saved_extraction = (
                        self.storage.find_cache_and_extraction_by_doc_hash(_chunk_hash, document_path=document_path)
                        if _chunk_hash else (None, None)
                    )
                    if existing_mc:
                        new_mc = existing_mc
                        self.current_episode = existing_mc
                        if _saved_extraction is None:
                            if verbose:
                                wprint_info("【步骤1】缓存｜命中｜跳过生成")
                            elif verbose_steps:
                                wprint_info("【步骤1】缓存｜命中｜跳过生成")
                    else:
                        with self._cache_lock:
                            def _run_step1():
                                return self._update_cache(
                                    chunk, doc_name,
                                    text_start_pos=start, text_end_pos=end,
                                    total_text_length=total_length, verbose=verbose,
                                    verbose_steps=verbose_steps,
                                    document_path=document_path, event_time=event_time,
                                    window_index=_wi + 1, total_windows=total_chunks,
                                    doc_hash=_chunk_hash,
                                )

                            new_mc = self._run_with_progress_heartbeat(
                                _run_step1,
                                chain_id="main",
                                base_progress=_g_lo + _span * 0.02,
                                phase_label=f"窗口 {_wi + 1}/{total_chunks} · 步骤1/7 进行中",
                                message="步骤1 更新记忆缓存",
                                window_label=_wlabel,
                                pipeline_role="主线程",
                                progress_callback=progress_callback,
                            )
                    _step1_elapsed = time.time() - _t_step1_start
                    state.window_timings[ci]["step1"] = _step1_elapsed
                    if verbose or verbose_steps:
                        wprint_info(f"【步骤1】完成｜{_step1_elapsed:.1f}s")
                    state.episodes[ci] = new_mc
                    state.input_texts[ci] = chunk
                    last_episode_id = new_mc.absolute_id

                    _action = self._poll_control(state, control_callback)
                    if _action:
                        self._signal_control_stop(state, _action, ci + 1)
                        state.extract_done[ci].set()
                        state.step9_done_ev[ci].set()
                        state.step10_done_ev[ci].set()
                        self._release_window_slot()
                        _slot_acquired = False
                        break

                    # 提交 step2-5
                    if _saved_extraction is not None:
                        _dedup_ents, _dedup_rels = dedupe_extraction_lists(
                            _saved_extraction[0], _saved_extraction[1]
                        )
                        state.extract_results[ci] = (_dedup_ents, _dedup_rels)
                        state.window_timings[ci]["step2-8"] = 0.0
                        state.extract_done[ci].set()
                        if main_chunk_done_callback:
                            main_chunk_done_callback(start_chunk + ci + 1)
                        self._release_window_slot()
                        _slot_acquired = False
                        if progress_callback:
                            self._safe_progress(progress_callback,
                                _p_end_main,
                                f"窗口 {_wi + 1}/{total_chunks} · 步骤1–5/7 已完成(缓存)",
                                "", "main",
                            )
                        if verbose:
                            _ents_count = len(_dedup_ents)
                            _rels_count = len(_dedup_rels)
                            if existing_mc:
                                wprint_info(
                                    f"【步骤1–5】缓存｜命中｜实体{_ents_count} 关系{_rels_count}→步骤9"
                                )
                            else:
                                wprint_info(
                                    f"【步骤2–5】缓存｜命中｜实体{_ents_count} 关系{_rels_count}"
                                )
                        elif verbose_steps:
                            if existing_mc:
                                wprint_info(
                                    f"窗口 {start_chunk + ci + 1}/{total_chunks} · 步骤1–5 已缓存跳过 → 步骤9/7"
                                )
                            else:
                                wprint_info("【步骤2–5】缓存｜跳过｜抽取已存在")
                    else:
                        if progress_callback:
                            self._safe_progress(progress_callback,
                                _p_after_step1,
                                f"窗口 {_wi + 1}/{total_chunks} · 步骤1/7 完成",
                                "", "main",
                            )

                        def _do_extract(idx=ci, mc=new_mc, chunk_text=chunk, __hash=_chunk_hash):
                            _wlabel = f"W{start_chunk + idx + 1}/{total_chunks}"
                            set_window_label(_wlabel)
                            set_pipeline_role("抽取")
                            _success_main = False
                            _t_extract_start = time.time()
                            with self._runtime_lock:
                                self._active_window_extractions += 1
                                self._peak_window_extractions = max(
                                    self._peak_window_extractions,
                                    self._active_window_extractions,
                                )
                            try:
                                _idx_lo = (start_chunk + idx) / total_chunks
                                _idx_hi = (start_chunk + idx + 1) / total_chunks
                                _idx_span = _idx_hi - _idx_lo
                                ents, rels = self._extract_only(
                                    mc, chunk_text, doc_name,
                                    verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
                                    progress_callback=lambda p, l, m: self._safe_progress(progress_callback, p, l, m, "main"),
                                    progress_range=(
                                        _idx_lo + _idx_span * (1.0 / 7.0),
                                        _idx_lo + _idx_span * (8.0 / 10.0),
                                    ),
                                    window_index=start_chunk + idx, total_windows=total_chunks,
                                    window_timings_ref=state.window_timings[idx],
                                )
                                state.extract_results[idx] = (ents, rels)
                                self.storage.save_extraction_result(__hash, ents, rels, document_path=document_path)
                                _success_main = True
                                _extract_elapsed = time.time() - _t_extract_start
                                state.window_timings[idx]["step2-8"] = _extract_elapsed
                                if verbose or verbose_steps:
                                    wprint_info(f"【步骤2–5】完成｜{_extract_elapsed:.1f}s")
                            except Exception as e:
                                if self._record_window_error(state, "extract", idx, e):
                                    logger.error("extract window %d error: %s", idx, e, exc_info=True)
                            finally:
                                with self._runtime_lock:
                                    self._active_window_extractions = max(0, self._active_window_extractions - 1)
                                state.extract_done[idx].set()
                                if _success_main and main_chunk_done_callback:
                                    main_chunk_done_callback(start_chunk + idx + 1)
                                self._release_window_slot()
                                clear_parallel_log_context()

                        try:
                            self._extraction_executor.submit(_do_extract)
                        except RuntimeError:
                            _do_extract()
                        _slot_acquired = False

                    if end >= total_length:
                        break
                    start = end - overlap
                finally:
                    if _slot_acquired:
                        self._release_window_slot()
        except Exception as e:
            with state.errors_lock:
                state.errors.append(("main", 0, e))
            logger.error("main pipeline error: %s", e, exc_info=True)
        finally:
            clear_parallel_log_context()

        # 等待所有窗口 step10 完成
        for i in range(N):
            state.step10_done_ev[i].wait()

        # Clean shutdown of prefetch executor with proper timeout
        try:
            state.prefetch_executor.shutdown(wait=True, timeout=5)
        except Exception as e:
            logger.warning("Prefetch executor shutdown failed: %s", e)
            try:
                state.prefetch_executor.shutdown(wait=False)
            except Exception:
                pass

        t9.join(timeout=60)
        if t9.is_alive():
            remember_log("警告: step9 线程在 join(60s) 超时后仍在运行")

        t10.join(timeout=60)
        if t10.is_alive():
            remember_log("警告: step10 线程在 join(60s) 超时后仍在运行")

        if state.control_state["action"] is not None:
            raise RememberControlFlow(state.control_state["action"])

        # ========== Post-window cross-window dedup (always runs, even for N=1) ==========
        # Run cross-window dedup even when some windows failed -- partial results are valuable.
        _dedup_exc = None
        try:
            self._cross_window_dedup(state.align_results, verbose=verbose)
        except Exception as e:
            _dedup_exc = e
            logger.error("Cross-window dedup failed: %s", e, exc_info=True)
            remember_log(f"后处理｜跨窗口去重失败: {e}")

        # ========== 计时汇总 ==========
        self._summarize_window_timings(state.window_timings)

        storage_path = str(self.storage.storage_path)
        total_entities = sum(
            len(ar.unique_entities) for ar in state.align_results if ar is not None
        )
        total_relations = sum(
            len(rl) for rl in state.step10_results if rl is not None
        )

        # Collect partial results even when some windows failed.
        _successful_windows = sum(
            1 for i in range(N)
            if state.align_results[i] is not None or state.step10_results[i] is not None
        )
        _failed_windows = len(state.errors)
        _window_errors_detail = [
            {"phase": phase, "window_index": idx, "error": str(exc)}
            for phase, idx, exc in state.errors
        ]

        result = {
            "episode_id": last_episode_id,
            "chunks_processed": total_chunks,
            "storage_path": storage_path,
            "entities": total_entities,
            "relations": total_relations,
        }

        if _failed_windows > 0:
            # Graceful degradation: log errors but return partial results instead of raising.
            # This ensures successful windows are persisted even when some fail.
            _error_summary = "; ".join(
                f"{phase}[W{idx}]: {exc}" for phase, idx, exc in state.errors[:5]
            )
            logger.error(
                "remember_text completed with %d/%d window failures: %s%s",
                _failed_windows, N, _error_summary,
                " (+ cross-window dedup failed)" if _dedup_exc else "",
            )
            remember_log(
                f"完成｜成功{_successful_windows}/{N}窗 "
                f"实体{total_entities} 关系{total_relations} "
                f"| {_failed_windows}窗失败: {_error_summary}"
            )
            result["warnings"] = _window_errors_detail
            result["failed_windows"] = _failed_windows
            result["successful_windows"] = _successful_windows
        elif _dedup_exc:
            result["warnings"] = [{"phase": "cross_window_dedup", "error": str(_dedup_exc)}]

        # Only raise if ALL windows failed -- partial results are still valuable.
        if _failed_windows >= N:
            _phase, _idx, exc = state.errors[0]
            raise exc

        return result

    def remember_phase1_overall(self, text: str, doc_name: str = "api_input",
                                event_time: Optional[datetime] = None,
                                document_path: str = "",
                                previous_overall_cache: Optional[Episode] = None,
                                verbose: bool = False,
                                progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Episode:
        """
        阶段1：仅生成文档整体记忆（描述即将处理的内容）。
        生成后即可作为下一文档 B 的初始记忆，无需等本文档最后一窗。
        """
        text_preview = (text[:2000] + "…") if len(text) > 2000 else text
        prev_content = previous_overall_cache.content if previous_overall_cache else None
        overall = self.llm_client.create_document_overall_memory(
            text_preview=text_preview,
            document_name=doc_name,
            event_time=event_time,
            previous_overall_content=prev_content,
        )
        if progress_callback is not None:
            progress_callback({
                "phase": "phase1",
                "phase_label": "整体记忆已生成",
                "completed": 1,
                "total": 1,
                "message": f"文档整体记忆已生成: {doc_name}",
            })
        if verbose:
            wprint_info(f"[Phase1] 文档整体记忆已生成: {overall.absolute_id[:20]}…, doc_name={doc_name!r}")
        return overall

    def remember_phase2_windows(self, text: str, doc_name: str = "api_input", verbose: bool = False,
                                verbose_steps: bool = True,
                                event_time: Optional[datetime] = None, document_path: str = "",
                                overall_cache: Optional[Episode] = None,
                                progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict:
        """
        阶段2：以整体记忆为起点，跑完所有滑窗（更新缓存 + 抽取实体/关系并写入）。
        overall_cache 即 phase1 返回的文档整体记忆，作为第一窗的 current_cache。
        """
        if not document_path:
            document_path = f"api://{uuid.uuid4().hex}"
        self.current_episode = overall_cache  # 第一窗的 _update_cache 会在此基础上续写
        window_size = self.document_processor.window_size
        overlap = self.document_processor.overlap
        total_length = len(text)
        start = 0
        chunk_idx = 0
        last_episode_id = None
        futures: List[Future] = []
        total_chunks = 1
        if total_length > 0:
            stride = max(1, window_size - overlap)
            total_chunks = 1 + max(0, (max(total_length - window_size, 0) + stride - 1) // stride)
        if progress_callback is not None:
            progress_callback({
                "phase": "phase2",
                "phase_label": "准备滑窗处理",
                "completed": 0,
                "total": total_chunks,
                "message": f"准备处理 {total_chunks} 个窗口",
            })

        while start < total_length:
            # 等待并发槽位：与 remember_text 一致，占用即计入主链窗口直至抽取任务 release
            self._acquire_window_slot()

            end = min(start + window_size, total_length)
            chunk = text[start:end]
            if start == 0:
                chunk = f"[文档元数据] 文档名：{doc_name} [/文档元数据]\n\n{chunk}"

            if verbose:
                wprint_info(f"\n{'='*60}")
                wprint_info(f"处理窗口 (文档: {doc_name}, 位置: {start}-{end}/{total_length})")
                wprint_info(f"输入文本长度: {len(chunk)} 字符")
                wprint_info(f"{'='*60}\n")
            elif verbose_steps:
                wprint_info(f"窗口 {chunk_idx + 1}/{total_chunks} 开始 · {doc_name} [{start}-{end}/{total_length}]")

            with self._cache_lock:
                new_mc = self._update_cache(
                    chunk, doc_name,
                    text_start_pos=start, text_end_pos=end,
                    total_text_length=total_length, verbose=verbose,
                    verbose_steps=verbose_steps,
                    document_path=document_path, event_time=event_time,
                )

            fut = self._extraction_executor.submit(
                self._run_extraction_job,
                new_mc, chunk, doc_name,
                verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
            )
            futures.append(fut)
            last_episode_id = new_mc.absolute_id
            chunk_idx += 1
            if progress_callback is not None:
                progress_callback({
                    "phase": "phase2",
                    "phase_label": "滑窗处理进行中",
                    "completed": chunk_idx,
                    "total": total_chunks,
                    "message": f"窗口 {chunk_idx}/{total_chunks} ({start}-{end}/{total_length})",
                    "window_start": start,
                    "window_end": end,
                    "text_length": total_length,
                })
            if end >= total_length:
                break
            start = end - overlap

        for fut in futures:
            fut.result()

        return {
            "episode_id": last_episode_id,
            "chunks_processed": chunk_idx,
            "storage_path": str(self.storage.storage_path),
        }

    def get_statistics(self) -> dict:
        """获取处理统计信息"""
        stats = self.storage.get_stats()
        return {
            "episodes": stats.get("episodes", 0),
            "entities": stats.get("entities", 0),
            "relations": stats.get("relations", 0),
            "storage_path": str(self.storage.storage_path)
        }

    def close(self):
        """释放资源：关闭线程池和存储连接。"""
        if hasattr(self, '_extraction_executor') and self._extraction_executor:
            self._extraction_executor.shutdown(wait=False)
        if hasattr(self, 'storage') and self.storage and hasattr(self.storage, 'close'):
            self.storage.close()

    def __del__(self):
        try:
            import sys
            if sys.is_finalizing():
                # Interpreter shutting down — don't touch executor, just close storage
                if hasattr(self, 'storage') and self.storage and hasattr(self.storage, 'close'):
                    try:
                        self.storage.close()
                    except Exception:
                        pass
                return
            self.close()
        except Exception:
            pass




def main():
    """示例使用"""
    
    # 配置
    storage_path = "./tmg_storage"
    document_paths = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if not document_paths:
        wprint_info("用法: python -m Temporal_Memory_Graph.processor <文档路径1> [文档路径2] ...")
        wprint_info("示例: python -m Temporal_Memory_Graph.processor doc1.txt doc2.txt")
        return
    
    # 创建处理器
    processor = TemporalMemoryGraphProcessor(
        storage_path=storage_path,
        window_size=1000,
        overlap=200,
        # llm_api_key="your-api-key",  # 如果需要，取消注释并填入
        # llm_model="gpt-4",
        # llm_base_url="https://api.openai.com/v1",  # 可自定义LLM API URL
        # embedding_model_path="/path/to/local/model",  # 本地embedding模型路径
        # embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",  # 或使用HuggingFace模型
    )
    
    # 处理文档
    processor.process_documents(document_paths, verbose=True)
    
    # 输出统计信息
    stats = processor.get_statistics()
    wprint_info("\n处理完成！")
    wprint_info(f"统计信息: {stats}")


if __name__ == "__main__":
    main()
