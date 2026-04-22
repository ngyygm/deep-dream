"""
主处理流程：整合所有模块，实现完整的文档处理pipeline
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import sys
import logging
import threading
import time
import uuid

from .document import DocumentProcessor
from ..llm.client import (
    LLMClient,
    LLM_PRIORITY_STEP1, LLM_PRIORITY_STEP2, LLM_PRIORITY_STEP3,
    LLM_PRIORITY_STEP4, LLM_PRIORITY_STEP5, LLM_PRIORITY_STEP6, LLM_PRIORITY_STEP7,
)
from ..storage.embedding import EmbeddingClient
from ..storage.manager import StorageManager
from ..storage import create_storage_manager
from .entity import EntityProcessor
from .relation import RelationProcessor
from ..models import Episode
from ..utils import (
    clear_parallel_log_context,
    compute_doc_hash,
    remember_log,
    set_pipeline_role,
    set_window_label,
    wprint,
    wprint_info,
)
from .extraction import _ExtractionMixin, _AlignResult
from .extraction_utils import dedupe_extraction_lists
from .new_extraction import _NewExtractionMixin

logger = logging.getLogger(__name__)


class RememberControlFlow(Exception):
    def __init__(self, action: str):
        super().__init__(action)
        self.remember_control_action = action


class TemporalMemoryGraphProcessor(_ExtractionMixin, _NewExtractionMixin):
    """时序记忆图谱处理器 - 主处理流程"""
    
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
                 prompt_episode_max_chars: Optional[int] = None,
                 max_llm_concurrency: Optional[int] = None,
                 # pipeline 可选配置（可从 config.pipeline 传入）
                 similarity_threshold: Optional[float] = None,
                 max_similar_entities: Optional[int] = None,
                 content_snippet_length: Optional[int] = None,
                 relation_content_snippet_length: Optional[int] = None,
                 relation_endpoint_jaccard_threshold: Optional[float] = None,
                 relation_endpoint_embedding_threshold: Optional[float] = None,
                 entity_extraction_max_iterations: Optional[int] = None,
                 entity_extraction_iterative: Optional[bool] = None,
                 entity_post_enhancement: Optional[bool] = None,
                 relation_extraction_max_iterations: Optional[int] = None,
                 relation_extraction_absolute_max_iterations: Optional[int] = None,
                 relation_extraction_iterative: Optional[bool] = None,
                 load_cache_memory: Optional[bool] = None,
                 jaccard_search_threshold: Optional[float] = None,
                 embedding_name_search_threshold: Optional[float] = None,
                 embedding_full_search_threshold: Optional[float] = None,
                 max_concurrent_windows: Optional[int] = None,
                 max_alignment_candidates: Optional[int] = None,
                 distill_data_dir: Optional[str] = None,
                 extraction_rounds: Optional[int] = None,
                 entity_extraction_rounds: Optional[int] = None,
                 relation_extraction_rounds: Optional[int] = None,
                 compress_multi_round_extraction: Optional[bool] = None,
                 v2_enable_reflection: Optional[bool] = None,
                 v2_enable_orphan_recovery: Optional[bool] = None,
                 v3_entity_refine_rounds: Optional[int] = None,
                 v3_relation_refine_rounds: Optional[int] = None,
                 remember_config: Optional[Dict[str, Any]] = None,
                 extraction_llm: Optional[Dict[str, Any]] = None,
                 graph_id: Optional[str] = None):
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
            entity_extraction_max_iterations: 实体抽取最大轮次（默认 3）
            entity_extraction_iterative: 是否迭代实体抽取（默认 True）
            entity_post_enhancement: 是否实体后验增强（默认 False）
            relation_extraction_max_iterations: 关系抽取最大轮次（默认 3）
            relation_extraction_absolute_max_iterations: 关系抽取绝对最大轮次（默认 10）
            relation_extraction_iterative: 是否迭代关系抽取（默认 True）
            load_cache_memory: 是否加载缓存记忆续写（默认 False）
            jaccard_search_threshold: Jaccard 搜索阈值（可选，不设则用 similarity_threshold）
            embedding_name_search_threshold: Embedding 名称搜索阈值（可选）
            embedding_full_search_threshold: Embedding 全文搜索阈值（可选）
            max_concurrent_windows: 同时处理的滑窗数上限（默认 1）；满员时不唤醒下一窗口，避免窗口内实体/关系并行导致线程爆炸
            compress_multi_round_extraction: 多轮实体/关系抽取是否使用压缩对话（不累积各轮 assistant 全文，默认 False）
            llm_context_window_tokens: 请求输入 prompt 的本地预检上限；未传时读 server 默认
            prompt_episode_max_chars: 注入抽取 prompt 的记忆缓存最大字符数；超长时自动截断，默认 2000
        """
        _content_snippet_length = content_snippet_length if content_snippet_length is not None else 300
        _remember_defaults = {
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
        _remember_from_config = (((config or {}).get("pipeline") or {}).get("remember") or {})
        _remember_overrides = dict(remember_config or {})
        _remember_cfg = dict(_remember_defaults)
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

        self.remember_mode = str(_remember_cfg.get("mode") or "v3").strip() or "v3"
        if self.remember_mode not in {"v2", "v3"}:
            self.remember_mode = "v3"
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
            # ≤0：关闭关系端点向量对齐（仅用 Jaccard/精确匹配）
            _relation_endpoint_embedding_threshold = None if v <= 0 else v
        _max_similar_entities = max_similar_entities if max_similar_entities is not None else 10

        _ctx_win = llm_context_window_tokens
        if _ctx_win is None:
            _ctx_win = 8000  # default context window tokens
        _ctx_win = max(256, int(_ctx_win))

        self.embedding_client = embedding_client or EmbeddingClient(
            model_path=embedding_model_path,
            model_name=embedding_model_name,
            device=embedding_device,
            use_local=embedding_use_local
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
        
        # extraction_rounds / *_rounds / *_max_iterations 三种命名都支持
        _entity_rounds = entity_extraction_max_iterations or entity_extraction_rounds or extraction_rounds or 3
        _relation_rounds = relation_extraction_max_iterations or relation_extraction_rounds or extraction_rounds or 3
        self.relation_extraction_max_iterations = _relation_rounds
        self.relation_extraction_absolute_max_iterations = relation_extraction_absolute_max_iterations if relation_extraction_absolute_max_iterations is not None else 10
        self.relation_extraction_iterative = relation_extraction_iterative if relation_extraction_iterative is not None else True

        self.entity_extraction_max_iterations = _entity_rounds
        self.entity_extraction_iterative = entity_extraction_iterative if entity_extraction_iterative is not None else True
        self.entity_post_enhancement = entity_post_enhancement if entity_post_enhancement is not None else False
        self.compress_multi_round_extraction = (
            compress_multi_round_extraction if compress_multi_round_extraction is not None else False
        )

        # V2 pipeline feature flags
        self.v2_enable_reflection = v2_enable_reflection if v2_enable_reflection is not None else True
        self.v2_enable_orphan_recovery = v2_enable_orphan_recovery if v2_enable_orphan_recovery is not None else True

        # V3 pipeline refine rounds
        self.v3_entity_refine_rounds = v3_entity_refine_rounds if v3_entity_refine_rounds is not None else 2
        self.v3_relation_refine_rounds = v3_relation_refine_rounds if v3_relation_refine_rounds is not None else 1

        # V3 extraction client (dual-model pipeline)
        _el = extraction_llm or {}
        self.extraction_client = None
        self.v3_extraction_enabled = False
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
            self.v3_extraction_enabled = True
            if self.remember_mode not in ("v2", "legacy"):
                self.remember_mode = "v3"

        self.llm_threads = max_llm_concurrency if max_llm_concurrency else 1
        self.load_cache_memory = load_cache_memory if load_cache_memory is not None else False

        # 别名：mixin 的 _extract_only 使用 rounds 参数名
        self.entity_extraction_rounds = self.entity_extraction_max_iterations
        self.relation_extraction_rounds = self.relation_extraction_max_iterations
        
        self.jaccard_search_threshold = jaccard_search_threshold
        self.embedding_name_search_threshold = embedding_name_search_threshold
        self.embedding_full_search_threshold = embedding_full_search_threshold
        self.max_alignment_candidates = max_alignment_candidates
        
        # 同时处理的滑窗数上限（满员时不唤醒下一窗口，避免窗口内实体/关系并行导致线程爆炸）
        if max_concurrent_windows is not None:
            _max_concurrent_windows = max_concurrent_windows
        else:
            # 自动推导：step2-5 约 4 个 LLM 步骤，step6+7 约 2 个，比值约 2:1
            # 至少 2 个窗口并行才能让 step6/7 不空闲；跟随 LLM 并发数，上限 8
            _max_concurrent_windows = max(2, min(max_llm_concurrency or 1, 8))
        _max_concurrent_windows = max(1, min(_max_concurrent_windows, 64))  # 合理范围 [1, 64]
        
        # 当前状态
        self.current_episode: Optional[Episode] = None
        
        # 流水线并行：cache 更新串行锁 + 抽取/处理线程池（max_workers 限制同时处理的窗口数）
        self._cache_lock = threading.Lock()
        self._max_concurrent_windows = _max_concurrent_windows
        self._window_slot = threading.Semaphore(_max_concurrent_windows)
        self._runtime_lock = threading.Lock()
        self._active_window_extractions = 0
        self._peak_window_extractions = 0
        # 已占用滑窗槽位、尚未 release 的窗口数（含主线程步骤1 + 步骤2–5 抽取等，与 Semaphore 成对）
        self._active_main_pipeline_windows = 0
        self._active_step6 = 0
        self._active_step7 = 0
        self._extraction_executor = ThreadPoolExecutor(
            max_workers=_max_concurrent_windows,
            thread_name_prefix="tmg-window",
        )

    def get_runtime_stats(self) -> Dict[str, int]:
        with self._runtime_lock:
            stats = {
                "configured_window_workers": self._max_concurrent_windows,
                "configured_llm_threads": self.llm_threads,
                "active_window_extractions": self._active_window_extractions,
                "active_main_pipeline_windows": self._active_main_pipeline_windows,
                "peak_window_extractions": self._peak_window_extractions,
                "active_step6": self._active_step6,
                "active_step7": self._active_step7,
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
                         entity_extraction_max_iterations: Optional[int] = None,
                         relation_extraction_absolute_max_iterations: Optional[int] = None,
                         entity_extraction_iterative: Optional[bool] = None,
                         entity_post_enhancement: Optional[bool] = None,
                         relation_extraction_max_iterations: Optional[int] = None,
                         relation_extraction_iterative: Optional[bool] = None,
                         load_cache_memory: Optional[bool] = None,
                         jaccard_search_threshold: Optional[float] = None,
                         embedding_name_search_threshold: Optional[float] = None,
                         embedding_full_search_threshold: Optional[float] = None,
                         compress_multi_round_extraction: Optional[bool] = None):
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
            entity_extraction_max_iterations: 实体抽取最大迭代次数（可选，覆盖初始化时的设置）
            relation_extraction_absolute_max_iterations: 关系抽取绝对最大迭代次数（可选，覆盖初始化时的设置）
            entity_extraction_iterative: 是否启用迭代实体抽取（可选，覆盖初始化时的设置）
            entity_post_enhancement: 是否启用实体后验增强（可选，覆盖初始化时的设置）
            relation_extraction_max_iterations: 关系抽取最大迭代次数（可选，覆盖初始化时的设置）
            relation_extraction_iterative: 是否启用迭代关系抽取（可选，覆盖初始化时的设置）
            load_cache_memory: 是否加载缓存记忆（可选，覆盖初始化时的设置）
            jaccard_search_threshold: Jaccard搜索（name_only）的相似度阈值（可选，默认使用similarity_threshold）
            embedding_name_search_threshold: Embedding搜索（name_only）的相似度阈值（可选，默认使用similarity_threshold）
            embedding_full_search_threshold: Embedding搜索（name+content）的相似度阈值（可选，默认使用similarity_threshold）
            compress_multi_round_extraction: 多轮实体/关系抽取是否使用压缩对话（可选，覆盖初始化时的设置）
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
        if entity_extraction_max_iterations is not None:
            original_values['entity_extraction_max_iterations'] = self.entity_extraction_max_iterations
            self.entity_extraction_max_iterations = entity_extraction_max_iterations
        if relation_extraction_absolute_max_iterations is not None:
            original_values['relation_extraction_absolute_max_iterations'] = self.relation_extraction_absolute_max_iterations
            self.relation_extraction_absolute_max_iterations = relation_extraction_absolute_max_iterations
        if entity_extraction_iterative is not None:
            original_values['entity_extraction_iterative'] = self.entity_extraction_iterative
            self.entity_extraction_iterative = entity_extraction_iterative
        if entity_post_enhancement is not None:
            original_values['entity_post_enhancement'] = self.entity_post_enhancement
            self.entity_post_enhancement = entity_post_enhancement
        if relation_extraction_max_iterations is not None:
            original_values['relation_extraction_max_iterations'] = self.relation_extraction_max_iterations
            self.relation_extraction_max_iterations = relation_extraction_max_iterations
        if relation_extraction_iterative is not None:
            original_values['relation_extraction_iterative'] = self.relation_extraction_iterative
            self.relation_extraction_iterative = relation_extraction_iterative
        if load_cache_memory is not None:
            original_values['load_cache_memory'] = self.load_cache_memory
            self.load_cache_memory = load_cache_memory
        if compress_multi_round_extraction is not None:
            original_values['compress_multi_round_extraction'] = self.compress_multi_round_extraction
            self.compress_multi_round_extraction = compress_multi_round_extraction

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

    def remember_text(self, text: str, doc_name: str = "api_input", verbose: bool = False,
                      verbose_steps: bool = True,
                      load_cache_memory: Optional[bool] = None,
                      event_time: Optional[datetime] = None,
                      document_path: str = "",
                      progress_callback: Optional[Callable] = None,
                      control_callback: Optional[Callable[[], Optional[str]]] = None,
                      start_chunk: int = 0,
                      main_chunk_done_callback: Optional[Callable] = None,
                      step6_chunk_done_callback: Optional[Callable] = None,
                      chunk_done_callback: Optional[Callable] = None,
                      source_document: Optional[str] = None) -> Dict:
        """
        将一段文本作为记忆入库：流水线式并行处理 step6（实体对齐）和 step7（关系对齐）。

        流水线架构：
        - 主线程：Phase A（step1 串行更新缓存）+ 提交 Phase B（step2-5 并行抽取）
        - step6 线程：等待当前窗口 step2-5 完成 + 前一窗口 step6 完成 → 实体对齐
        - step7 线程：等待当前窗口 step6 完成 + 前一窗口 step7 完成 → 关系对齐
        - step6 W(i+1) 可与 step7 W(i) 并行执行

        Args:
            text: 原始文本内容
            doc_name: 文档/来源名称
            verbose: 是否打印详细处理日志（步骤内细节、LLM 提示等）
            verbose_steps: 是否在控制台输出步骤级「开始/结束」汇报（verbose=True 时仍生效，但以详细日志为准）
                并行时控制台行格式为 [窗号][角色] 正文；角色为 主线程 / 抽取 / 步骤6 / 步骤7 之一。
            load_cache_memory: 是否在开始前加载最新缓存记忆再追加
            event_time: 事件实际发生时间
            document_path: 原文文件路径
            progress_callback: 进度回调 fn(progress, phase_label, message, chain_id)
            control_callback: 控制回调 fn() -> {"pause","cancel",None}，在窗口级安全点生效
            start_chunk: 从第几个窗口开始（关系链断点续传）
            main_chunk_done_callback: 步骤1–5 完成一个窗口后的回调 fn(processed_count)
            step6_chunk_done_callback: 步骤6 完成一个窗口后的回调 fn(processed_count)
            chunk_done_callback: 步骤7 完成一个窗口后的回调 fn(processed_count)
            source_document: 来源文档名称（优先于 doc_name）

        Returns:
            dict: episode_id, chunks_processed, storage_path
        """
        doc_name = source_document or doc_name
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

        # 预分配数组，用于线程间共享数据
        episodes = [None] * N
        input_texts = [None] * N
        extract_results = [None] * N   # (entities, relations) 元组
        align_results = [None] * N     # _AlignResult
        step7_results = [None] * N     # List[Relation] from _align_relations

        # 每窗口计时（线程安全：各线程只写自己的 index）
        window_timings: List[Dict[str, float]] = [{} for _ in range(N)]

        # 同步事件
        extract_done = [threading.Event() for _ in range(N)]
        step6_done_ev = [threading.Event() for _ in range(N)]
        step7_done_ev = [threading.Event() for _ in range(N)]

        # 错误收集
        errors = []
        errors_lock = threading.Lock()
        window_failures = [None] * N
        _control_lock = threading.Lock()
        _control_state = {"action": None}

        def _record_window_error(stage: str, idx: int, exc: Exception) -> bool:
            """只记录每个窗口的首个真实错误，避免 step6/7 级联噪音覆盖根因。"""
            with errors_lock:
                if window_failures[idx] is None:
                    window_failures[idx] = (stage, exc)
                    errors.append((stage, idx, exc))
                    return True
            return False

        def _signal_control_stop(action: str, from_index: int, *, set_extract: bool = True,
                                 set_step6: bool = True, set_step7: bool = True) -> None:
            with _control_lock:
                if _control_state["action"] is None:
                    _control_state["action"] = action
                _from = max(0, min(from_index, N))
                for j in range(_from, N):
                    if set_extract:
                        extract_done[j].set()
                    if set_step6:
                        step6_done_ev[j].set()
                    if set_step7:
                        step7_done_ev[j].set()

        def _poll_control() -> Optional[str]:
            action = _control_state["action"]
            if action:
                return action
            if control_callback is None:
                return None
            action = control_callback()
            if action in ("pause", "cancel"):
                with _control_lock:
                    if _control_state["action"] is None:
                        _control_state["action"] = action
                    return _control_state["action"]
            return None

        # 进度回调包装
        _progress_lock = threading.Lock()

        def _safe_progress(progress, label, message, chain_id="step6"):
            if not progress_callback:
                return
            progress_callback(progress, label, message, chain_id)

        def _run_with_progress_heartbeat(
            run_fn: Callable[[], Any],
            *,
            chain_id: str,
            base_progress: float,
            phase_label: str,
            message: str,
            window_label: str,
            pipeline_role: str,
            heartbeat_seconds: float = 5.0,
            log_interval_seconds: float = 30.0,
        ) -> Any:
            """为长耗时步骤补充心跳，避免前端/日志长时间停在同一标签像“卡死”。

            不伪造进度，仅重复上报当前 phase，并附加已等待秒数。
            """
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
                        _safe_progress(base_progress, hb_label, hb_message, chain_id)
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

        # 跨窗预取：单线程避免与 embedding 客户端并发冲突；在「等上一窗 step6/7」期间跑本窗可并行工作
        _prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tmg-chain-prefetch")

        def _safe_prefetch_submit(fn, *args, **kwargs):
            """解释器收尾或 Executor 已 shutdown 时 submit 会失败；返回 None 表示跳过预取（现场编码）。"""
            try:
                if sys.is_finalizing():
                    return None
            except Exception:
                pass
            try:
                return _prefetch_executor.submit(fn, *args, **kwargs)
            except RuntimeError:
                return None

        # ========== step6 工作线程 ==========
        def step6_worker():
            _already_versioned = set()  # 跨窗口共享：同一次 remember 中同一实体只创建一次版本
            for i in range(N):
                extract_done[i].wait()          # 等待当前窗口 step2-5 完成
                _action = _poll_control()
                if _action:
                    _signal_control_stop(_action, i, set_extract=False, set_step6=True, set_step7=True)
                    break
                set_window_label(f"W{start_chunk + i + 1}/{total_chunks}")
                set_pipeline_role("步骤6")
                _er = extract_results[i]
                emb_prefetch_future: Optional[Future] = None
                if _er is not None:
                    _ents, _ = _er
                    if _ents and self.storage.embedding_client and self.storage.embedding_client.is_available():
                        emb_prefetch_future = _safe_prefetch_submit(
                            self.entity_processor.encode_entities_for_candidate_table,
                            _ents,
                        )
                if i > 0:
                    step6_done_ev[i - 1].wait() # 等待前一窗口 step6 完成
                _action = _poll_control()
                if _action:
                    _signal_control_stop(_action, i, set_extract=False, set_step6=True, set_step7=True)
                    break
                with self._runtime_lock:
                    self._active_step6 += 1
                _t_step6_start = time.time()
                try:
                    mc = episodes[i]
                    _success = False
                    if _er is None:
                        # 抽取阶段已经失败时，只跳过后续链路，保留首个真实异常。
                        _upstream = window_failures[i]
                        if _upstream is not None:
                            _stage, _exc = _upstream
                            if verbose or verbose_steps:
                                wprint_info(f"【步骤6】跳过｜上游｜{_stage} {_exc}")
                            continue
                        raise RuntimeError(
                            f"step6 skipped for window {start_chunk + i}: extract result is None (extraction failed)"
                        )
                    ents, rels = _er
                    if verbose:
                        wprint_info("【步骤6】实体｜就绪｜本窗1–5完成或缓存")
                    elif verbose_steps:
                        wprint_info("【步骤6】实体｜开始｜前置1–5已就绪")
                    _wi = start_chunk + i
                    _g_lo = _wi / total_chunks
                    _g_hi = (_wi + 1) / total_chunks
                    _span = _g_hi - _g_lo
                    # 与 queue._wf_for_chain / 主链 7 分窗一致：步骤6、7 各占本窗的 1/7（非整窗 [g_lo,g_hi]）
                    _pr_step6 = (_g_lo + _span * (5.0 / 7.0), _g_lo + _span * (6.0 / 7.0))
                    ar = self._align_entities(
                        ents, rels, mc, input_texts[i], doc_name,
                        verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
                        progress_callback=lambda p, l, m: _safe_progress(p, l, m, "step6"),
                        progress_range=_pr_step6,
                        window_index=start_chunk + i, total_windows=total_chunks,
                        entity_embedding_prefetch=emb_prefetch_future,
                        already_versioned_family_ids=_already_versioned,
                        window_timings_ref=window_timings[i],
                    )
                    align_results[i] = ar
                    _success = True
                    _step6_elapsed = time.time() - _t_step6_start
                    window_timings[i]["step6"] = _step6_elapsed
                    if verbose or verbose_steps:
                        wprint_info(f"【步骤6】完成｜{_step6_elapsed:.1f}s")
                except Exception as e:
                    if _record_window_error("step6", i, e):
                        logger.error("step6 window %d error: %s", i, e, exc_info=True)
                finally:
                    with self._runtime_lock:
                        self._active_step6 = max(0, self._active_step6 - 1)
                    step6_done_ev[i].set()
                    if _success and step6_chunk_done_callback:
                        step6_chunk_done_callback(start_chunk + i + 1)
                    clear_parallel_log_context()

        # ========== step7 工作线程 ==========
        def step7_worker():
            for i in range(N):
                step6_done_ev[i].wait()          # 等待当前窗口 step6 完成
                _action = _poll_control()
                if _action:
                    _signal_control_stop(_action, i, set_extract=False, set_step6=False, set_step7=True)
                    break
                set_window_label(f"W{start_chunk + i + 1}/{total_chunks}")
                set_pipeline_role("步骤7")
                ar = align_results[i]
                step7_inputs_cache: Optional[Tuple[List[Dict[str, str]], Dict[str, str], List[Dict], List[Dict]]] = None
                rel_prefetch_future: Optional[Future] = None
                if ar is not None:
                    try:
                        step7_inputs_cache = self._build_step7_relation_inputs_from_align_result(ar)
                        _ri, _eid, _, _ = step7_inputs_cache
                        if i > 0 and _ri:
                            rel_prefetch_future = _safe_prefetch_submit(
                                self.relation_processor.build_relations_by_pair_from_inputs,
                                _ri,
                                _eid,
                            )
                    except Exception as exc:
                        wprint_info(f"  │  step7 输入构建失败: {exc}")
                        step7_inputs_cache = None
                        rel_prefetch_future = None
                if i > 0:
                    step7_done_ev[i - 1].wait()  # 等待前一窗口 step7 完成
                _action = _poll_control()
                if _action:
                    _signal_control_stop(_action, i, set_extract=False, set_step6=False, set_step7=True)
                    break
                prepared_relations_by_pair = None
                if rel_prefetch_future is not None:
                    try:
                        prepared_relations_by_pair, _ = rel_prefetch_future.result()
                    except Exception as exc:
                        wprint_info(f"  │  关系预取结果获取失败: {exc}")
                        prepared_relations_by_pair = None
                with self._runtime_lock:
                    self._active_step7 += 1
                _t_step7_start = time.time()
                _success = False
                _window_has_entities = False
                try:
                    if ar is None:
                        _upstream = window_failures[i]
                        if _upstream is not None:
                            _stage, _exc = _upstream
                            if verbose or verbose_steps:
                                wprint_info(f"【步骤7】跳过｜上游｜{_stage} {_exc}")
                            continue
                        raise RuntimeError(
                            f"step6 result for window {start_chunk + i} is None"
                        )
                    mc = episodes[i]
                    _wi = start_chunk + i
                    _g_lo = _wi / total_chunks
                    _g_hi = (_wi + 1) / total_chunks
                    _span = _g_hi - _g_lo
                    _pr_step7 = (_g_lo + _span * (6.0 / 7.0), _g_hi)
                    processed_rels = self._align_relations(
                        ar, mc, input_texts[i], doc_name,
                        verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
                        progress_callback=lambda p, l, m: _safe_progress(p, l, m, "step7"),
                        progress_range=_pr_step7,
                        window_index=start_chunk + i, total_windows=total_chunks,
                        prepared_relations_by_pair=prepared_relations_by_pair,
                        step7_inputs_cache=step7_inputs_cache,
                        window_timings_ref=window_timings[i],
                    )
                    step7_results[i] = processed_rels
                    _success = True
                    _window_has_entities = bool(ar.unique_entities)
                    _step7_elapsed = time.time() - _t_step7_start
                    window_timings[i]["step7"] = _step7_elapsed
                    if verbose or verbose_steps:
                        wprint_info(f"【步骤7】完成｜{_step7_elapsed:.1f}s")

                    # step7 成功后清理孤立实体
                    if _window_has_entities:
                        try:
                            _orphan_count = self._cleanup_orphaned_entities(
                                ar.unique_entities,
                                verbose=verbose or verbose_steps,
                            )
                            if _orphan_count > 0:
                                _window_has_entities = bool(ar.unique_entities) and _orphan_count < len(ar.unique_entities)
                        except Exception as _oe:
                            logger.warning("孤立实体清理失败: %s", _oe)
                except Exception as e:
                    if _record_window_error("step7", i, e):
                        logger.error("step7 window %d error: %s", i, e, exc_info=True)
                finally:
                    with self._runtime_lock:
                        self._active_step7 = max(0, self._active_step7 - 1)
                    step7_done_ev[i].set()
                    # step7 成功即推进断点：无实体窗口也应计数，否则断点永不前进、重启后会反复重跑同一窗
                    if _success and chunk_done_callback:
                        chunk_done_callback(start_chunk + i + 1)
                    if _success and not _window_has_entities:
                        wprint_info("提示: step7 完成但本窗无实体，仍已计入进度（避免断点卡死）")
                    clear_parallel_log_context()

        # 启动 step6 / step7 线程
        t6 = threading.Thread(target=step6_worker, name="tmg-step6-chain", daemon=True)
        t7 = threading.Thread(target=step7_worker, name="tmg-step7-chain", daemon=True)
        t6.start()
        t7.start()

        if verbose or verbose_steps:
            remember_log(
                "并行流水线 · 日志前缀 [窗号][角色]："
                "主线程=步骤1+提交抽取；抽取=步骤2–5；步骤6/7=链式线程。"
                "不同窗会交错，属正常。"
            )

        # ========== 主线程：Phase A（step1 串行）+ 提交 Phase B（step2-5）==========
        try:
            # 跳到 start_chunk 对应的文本位置
            start = start_chunk * stride

            for ci in range(N):
                _action = _poll_control()
                if _action:
                    _signal_control_stop(_action, ci, set_extract=True, set_step6=True, set_step7=True)
                    break
                # 等待并发槽位：如果已有 max_concurrent_windows 个窗口在 step1–5，暂缓下一窗
                self._acquire_window_slot()
                _slot_acquired = True

                try:
                    _action = _poll_control()
                    if _action:
                        _signal_control_stop(_action, ci, set_extract=True, set_step6=True, set_step7=True)
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

                    # 本窗全局区间（步骤1–5 占前 5/7）：一进入窗口就上报 main，避免长时间 step1(LLM) 无回调导致 Web/日志仍显示上一窗
                    _wi = start_chunk + ci
                    _g_lo = _wi / total_chunks
                    _g_hi = (_wi + 1) / total_chunks
                    _span = _g_hi - _g_lo
                    _p_after_step1 = _g_lo + _span * (1.0 / 7.0)
                    _p_end_main = _g_lo + _span * (5.0 / 7.0)
                    if progress_callback:
                        _safe_progress(
                            _g_lo + _span * 0.02,
                            f"窗口 {start_chunk + ci + 1}/{total_chunks} · 步骤1/7 进行中",
                            "",
                            "main",
                        )

                    # Step1: 更新缓存（串行，需加锁）
                    # 断点续传优化：如果该窗口的缓存已存在，直接复用，跳过 LLM 调用
                    _t_step1_start = time.time()
                    _chunk_hash = compute_doc_hash(chunk)
                    _saved_extraction = (
                        self.storage.load_extraction_result(_chunk_hash, document_path=document_path)
                        if _chunk_hash
                        else None
                    )
                    existing_mc = self.storage.find_cache_by_doc_hash(
                        _chunk_hash, document_path=document_path,
                    )
                    if existing_mc:
                        new_mc = existing_mc
                        self.current_episode = existing_mc
                        # 若步骤2–5 也有缓存，稍后在下方合并打印，避免看起来像「没跑步骤1–5 就直接步骤6」
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
                                )

                            new_mc = _run_with_progress_heartbeat(
                                _run_step1,
                                chain_id="main",
                                base_progress=_g_lo + _span * 0.02,
                                phase_label=f"窗口 {_wi + 1}/{total_chunks} · 步骤1/7 进行中",
                                message="步骤1 更新记忆缓存",
                                window_label=_wlabel,
                                pipeline_role="主线程",
                            )
                    _step1_elapsed = time.time() - _t_step1_start
                    window_timings[ci]["step1"] = _step1_elapsed
                    if verbose or verbose_steps:
                        wprint_info(f"【步骤1】完成｜{_step1_elapsed:.1f}s")
                    episodes[ci] = new_mc
                    input_texts[ci] = chunk
                    last_episode_id = new_mc.absolute_id

                    _action = _poll_control()
                    if _action:
                        _signal_control_stop(_action, ci + 1, set_extract=True, set_step6=True, set_step7=True)
                        extract_done[ci].set()
                        step6_done_ev[ci].set()
                        step7_done_ev[ci].set()
                        self._release_window_slot()
                        _slot_acquired = False
                        break

                    # 提交 step2-5 到线程池
                    # 断点续传优化：如果抽取结果已存在，直接加载，跳过 LLM 调用
                    if _saved_extraction is not None:
                        # 抽取结果已存在，跳过步骤2-5，立即释放槽位（与在线抽取一致：实体/关系列表去重）
                        _dedup_ents, _dedup_rels = dedupe_extraction_lists(
                            _saved_extraction[0], _saved_extraction[1]
                        )
                        extract_results[ci] = (_dedup_ents, _dedup_rels)
                        window_timings[ci]["step2-5"] = 0.0  # 缓存命中
                        extract_done[ci].set()
                        if main_chunk_done_callback:
                            main_chunk_done_callback(start_chunk + ci + 1)
                        self._release_window_slot()
                        _slot_acquired = False
                        if progress_callback:
                            _safe_progress(
                                _p_end_main,
                                f"窗口 {_wi + 1}/{total_chunks} · 步骤1–5/7 已完成(缓存)",
                                "",
                                "main",
                            )
                        if verbose:
                            _ents_count = len(_dedup_ents)
                            _rels_count = len(_dedup_rels)
                            if existing_mc:
                                wprint_info(
                                    f"【步骤1–5】缓存｜命中｜实体{_ents_count} 关系{_rels_count}→步骤6"
                                )
                            else:
                                wprint_info(
                                    f"【步骤2–5】缓存｜命中｜实体{_ents_count} 关系{_rels_count}"
                                )
                        elif verbose_steps:
                            if existing_mc:
                                wprint_info(
                                    f"窗口 {start_chunk + ci + 1}/{total_chunks} · 步骤1–5 已缓存跳过 → 步骤6/7"
                                )
                            else:
                                wprint_info("【步骤2–5】缓存｜跳过｜抽取已存在")
                    else:
                        if progress_callback:
                            _safe_progress(
                                _p_after_step1,
                                f"窗口 {_wi + 1}/{total_chunks} · 步骤1/7 完成",
                                "",
                                "main",
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
                                    progress_callback=lambda p, l, m: _safe_progress(p, l, m, "main"),
                                    progress_range=(
                                        _idx_lo + _idx_span * (1.0 / 7.0),
                                        _idx_lo + _idx_span * (5.0 / 7.0),
                                    ),
                                    window_index=start_chunk + idx, total_windows=total_chunks,
                                    window_timings_ref=window_timings[idx],
                                )
                                extract_results[idx] = (ents, rels)
                                # 保存抽取结果供断点续传复用
                                self.storage.save_extraction_result(__hash, ents, rels, document_path=document_path)
                                _success_main = True
                                _extract_elapsed = time.time() - _t_extract_start
                                window_timings[idx]["step2-5"] = _extract_elapsed
                                if verbose or verbose_steps:
                                    wprint_info(f"【步骤2–5】完成｜{_extract_elapsed:.1f}s")
                            except Exception as e:
                                if _record_window_error("extract", idx, e):
                                    logger.error("extract window %d error: %s", idx, e, exc_info=True)
                            finally:
                                with self._runtime_lock:
                                    self._active_window_extractions = max(0, self._active_window_extractions - 1)
                                extract_done[idx].set()
                                if _success_main and main_chunk_done_callback:
                                    main_chunk_done_callback(start_chunk + idx + 1)
                                self._release_window_slot()
                                clear_parallel_log_context()

                        self._extraction_executor.submit(_do_extract)
                        _slot_acquired = False

                    if end >= total_length:
                        break
                    start = end - overlap
                finally:
                    if _slot_acquired:
                        self._release_window_slot()
        except Exception as e:
            with errors_lock:
                errors.append(("main", 0, e))
            logger.error("main pipeline error: %s", e, exc_info=True)
        finally:
            clear_parallel_log_context()

        # 等待所有窗口 step7 完成
        for i in range(N):
            step7_done_ev[i].wait()

        t6.join(timeout=60)
        if t6.is_alive():
            remember_log("警告: step6 线程在 join(60s) 超时后仍在运行")

        t7.join(timeout=60)
        if t7.is_alive():
            remember_log("警告: step7 线程在 join(60s) 超时后仍在运行")

        _prefetch_executor.shutdown(wait=False)

        if _control_state["action"] is not None:
            raise RememberControlFlow(_control_state["action"])

        if errors:
            _phase, _idx, exc = errors[0]
            raise exc

        # ========== Post-window cross-window same-name dedup ==========
        # After all windows complete, find same-name entities with different family_ids
        # and merge those that are genuinely the same concept (high embedding similarity).
        if N > 1 and verbose:
            self._cross_window_dedup(align_results, verbose=verbose)

        # ========== 计时汇总 ==========
        _all_steps = ["step1", "step2-5", "step6", "step7"]
        _step_labels = {"step1": "步骤1-缓存", "step2-5": "步骤2-5-抽取", "step6": "步骤6-实体对齐", "step7": "步骤7-关系对齐"}
        # Sub-step keys (accumulated across all windows)
        _sub_step_labels = {
            "v3-1_entity_extract": "V3-1-实体提取",
            "v3-2_entity_dedup": "V3-2-实体去重",
            "v3-3_entity_content": "V3-3-实体内容",
            "v3-4_entity_quality": "V3-4-实体质量门",
            "v3-5_relation_discovery": "V3-5-关系发现",
            "v3-6_relation_content": "V3-6-关系内容",
            "v3-7_relation_quality": "V3-7-关系质量门",
            "step6-process_entities": "步骤6-实体处理",
            "step6-dedup_merge": "步骤6-同名去重",
            "step7-process_relations": "步骤7-关系处理",
        }
        _step_totals = {s: 0.0 for s in _all_steps}
        _sub_totals = {k: 0.0 for k in _sub_step_labels}
        for _i, _wt in enumerate(window_timings):
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

        storage_path = str(self.storage.storage_path)
        # Aggregate entity/relation counts from all windows
        total_entities = sum(
            len(ar.unique_entities) for ar in align_results if ar is not None
        )
        total_relations = sum(
            len(rl) for rl in step7_results if rl is not None
        )
        return {
            "episode_id": last_episode_id,
            "chunks_processed": total_chunks,
            "storage_path": storage_path,
            "entities": total_entities,
            "relations": total_relations,
        }

    def _cross_window_dedup(self, align_results, verbose=True):
        """After all windows complete, find and merge same-name entities with different family_ids.

        Uses embedding similarity to distinguish genuine duplicates from same-name-different-meaning entities.
        Only merges when embedding similarity is above threshold (default 0.75).
        """
        from collections import defaultdict

        # Collect all unique entities across windows
        all_entities = []
        for ar in align_results:
            if ar is None:
                continue
            all_entities.extend(ar.unique_entities)

        # Group by name
        name_to_entities = defaultdict(list)
        for entity in all_entities:
            name_to_entities[entity.name.strip()].append(entity)

        # Find same-name duplicates with different family_ids
        dupes = {name: ents for name, ents in name_to_entities.items()
                 if len(set(e.family_id for e in ents)) > 1}

        if not dupes:
            return

        if verbose:
            wprint_info(f"【后处理】同名检查｜{len(dupes)}组")

        for name, ents in dupes.items():
            # Group by family_id (in case same fid appears multiple times)
            fid_groups = defaultdict(list)
            for e in ents:
                fid_groups[e.family_id].append(e)

            fids = list(fid_groups.keys())
            if len(fids) < 2:
                continue

            # Check embedding similarity between pairs
            primary_fid = fids[0]
            primary = fid_groups[primary_fid][0]

            for other_fid in fids[1:]:
                # Compute content similarity using embeddings
                try:
                    sim = self._compute_entity_content_similarity(primary, fid_groups[other_fid][0])
                except Exception:
                    sim = 0.0

                if sim >= 0.75:
                    # High similarity: redirect relations to primary, then delete duplicate
                    self.storage.register_entity_redirect(other_fid, primary_fid)
                    try:
                        # Re-point relations that reference the duplicate entity
                        self.storage.redirect_entity_relations(other_fid, primary_fid)
                        deleted = self.storage.delete_entity_all_versions(other_fid)
                        if verbose:
                            wprint_info(f"【后处理】同名合并｜{name} sim={sim:.2f} {other_fid}→{primary_fid} (deleted {deleted}v)")
                    except Exception as e:
                        if verbose:
                            wprint_info(f"【后处理】同名合并｜{name} sim={sim:.2f} {other_fid}→{primary_fid} (redirect only, merge failed: {e})")
                else:
                    if verbose:
                        wprint_info(f"【后处理】同名保留｜{name} sim={sim:.2f} (不同概念)")

    def _compute_entity_content_similarity(self, entity1, entity2):
        """Compute content similarity between two entities using embeddings.

        Fetches content from DB if Entity.content is empty (alignment result
        objects may not have content populated).
        """
        c1 = (entity1.content or "")[:500]
        c2 = (entity2.content or "")[:500]

        # Fetch from DB if content is empty
        if not c1:
            try:
                versions = self.storage.get_entity_versions(entity1.family_id)
                if versions:
                    c1 = (versions[0].content or "")[:500]
            except Exception:
                pass
        if not c2:
            try:
                versions = self.storage.get_entity_versions(entity2.family_id)
                if versions:
                    c2 = (versions[0].content or "")[:500]
            except Exception:
                pass

        if not c1 or not c2:
            # Fall back to name-only comparison
            return 1.0 if entity1.name == entity2.name else 0.0

        try:
            import numpy as np
            if not self.storage.embedding_client or not self.storage.embedding_client.is_available():
                # No embedding available, use name match only
                return 1.0 if entity1.name == entity2.name else 0.0
            emb1 = np.asarray(self.storage.embedding_client.encode(c1)).flatten()
            emb2 = np.asarray(self.storage.embedding_client.encode(c2)).flatten()
            sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
            return max(0.0, min(1.0, sim))
        except Exception as e:
            wprint_info(f"【后处理】相似度计算失败｜{entity1.name} {type(e).__name__}: {e}")
            return 0.0

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
        # 这里可以添加统计逻辑
        return {
            "episodes": len(list(self.storage.cache_dir.glob("*.json"))),
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
