"""
主处理流程：整合所有模块，实现完整的文档处理pipeline
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, Future, wait, FIRST_COMPLETED
import re
import sys
import threading
import time
import traceback
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
from ..models import Episode, Entity
from ..utils import (
    clear_parallel_log_context,
    compute_doc_hash,
    remember_log,
    set_pipeline_role,
    set_window_label,
    wprint,
)
from .extraction import _ExtractionMixin, _AlignResult, dedupe_extraction_lists


class RememberControlFlow(Exception):
    def __init__(self, action: str):
        super().__init__(action)
        self.remember_control_action = action


class TemporalMemoryGraphProcessor(_ExtractionMixin):
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
                 compress_multi_round_extraction: Optional[bool] = None):
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
            from server.config import DEFAULTS
            _llm_d = DEFAULTS.get("llm") or {}
            if "context_window_tokens" not in _llm_d:
                raise RuntimeError("server.config.DEFAULTS['llm'] 缺少 context_window_tokens")
            _ctx_win = int(_llm_d["context_window_tokens"])
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
            # 更新 StorageManager
            if 'storage' not in original_components:
                original_components['storage'] = self.storage
            self.storage.entity_content_snippet_length = content_snippet_length
            # 更新 LLMClient
            if 'llm_client' not in original_components:
                original_components['llm_client'] = self.llm_client
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
            # 更新 StorageManager
            if 'storage' not in original_components:
                original_components['storage'] = self.storage
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
                wprint(f"开始处理 {len(document_paths)} 个文档...")
            
            # 断点续传相关变量
            resume_document_path = None
            resume_text = None
            
            # 根据配置决定是否加载最新的记忆缓存并支持断点续传
            if self.load_cache_memory:
                if verbose:
                    wprint("正在加载最新的缓存记忆...")

                # 获取最新缓存的元数据（包含 text 和 document_path）
                # 只查找"文档处理"类型的缓存，避免使用知识图谱整理产生的缓存（其text字段是整理后的实体信息，不是原始文档文本）
                latest_metadata = self.storage.get_latest_episode_metadata(activity_type="文档处理")
                
                if latest_metadata:
                    # 加载缓存记忆
                    self.current_episode = self.storage.load_episode(latest_metadata['absolute_id'])
                    
                    if self.current_episode:
                        if verbose:
                            wprint(f"已加载缓存记忆: {self.current_episode.absolute_id} (时间: {self.current_episode.event_time})")
                        
                        # 提取断点续传信息
                        resume_document_path = latest_metadata.get('document_path', '')
                        resume_text = latest_metadata.get('text', '')
                        
                        if verbose:
                            if resume_document_path:
                                wprint(f"[断点续传] 上次处理的文档: {resume_document_path}")
                            if resume_text:
                                text_preview = resume_text[:100].replace('\n', ' ')
                                wprint(f"[断点续传] 上次处理的文本片段: {text_preview}...")
                else:
                    if verbose:
                        wprint("未找到缓存记忆，将从头开始处理")
                    self.current_episode = None
            else:
                if verbose:
                    wprint("不加载缓存记忆，将从头开始处理")
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
                    wprint(f"\n处理窗口 {chunk_idx + 1} (文档: {document_name}, 位置: {text_start_pos}-{text_end_pos}/{total_text_length})")
                elif _epv:
                    wprint(f"窗口 {chunk_idx + 1} 开始 · {document_name}")
                
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
                            wprint(f"{phase_label} · 长调用进行中（已等待 {elapsed}s）")
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
                try:
                    mc = episodes[i]
                    _success = False
                    if _er is None:
                        # 抽取阶段已经失败时，只跳过后续链路，保留首个真实异常。
                        _upstream = window_failures[i]
                        if _upstream is not None:
                            _stage, _exc = _upstream
                            if verbose or verbose_steps:
                                wprint(f"【步骤6】跳过｜上游｜{_stage} {_exc}")
                            continue
                        raise RuntimeError(
                            f"step6 skipped for window {start_chunk + i}: extract result is None (extraction failed)"
                        )
                    ents, rels = _er
                    if verbose:
                        wprint("【步骤6】实体｜就绪｜本窗1–5完成或缓存")
                    elif verbose_steps:
                        wprint("【步骤6】实体｜开始｜前置1–5已就绪")
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
                    )
                    align_results[i] = ar
                    _success = True
                except Exception as e:
                    if _record_window_error("step6", i, e):
                        traceback.print_exc()
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
                        wprint(f"  │  step7 输入构建失败: {exc}")
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
                        wprint(f"  │  关系预取结果获取失败: {exc}")
                        prepared_relations_by_pair = None
                with self._runtime_lock:
                    self._active_step7 += 1
                _success = False
                _window_has_entities = False
                try:
                    if ar is None:
                        _upstream = window_failures[i]
                        if _upstream is not None:
                            _stage, _exc = _upstream
                            if verbose or verbose_steps:
                                wprint(f"【步骤7】跳过｜上游｜{_stage} {_exc}")
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
                    self._align_relations(
                        ar, mc, input_texts[i], doc_name,
                        verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
                        progress_callback=lambda p, l, m: _safe_progress(p, l, m, "step7"),
                        progress_range=_pr_step7,
                        window_index=start_chunk + i, total_windows=total_chunks,
                        prepared_relations_by_pair=prepared_relations_by_pair,
                        step7_inputs_cache=step7_inputs_cache,
                    )
                    _success = True
                    _window_has_entities = bool(ar.unique_entities)
                except Exception as e:
                    if _record_window_error("step7", i, e):
                        traceback.print_exc()
                finally:
                    with self._runtime_lock:
                        self._active_step7 = max(0, self._active_step7 - 1)
                    step7_done_ev[i].set()
                    # step7 成功即推进断点：无实体窗口也应计数，否则断点永不前进、重启后会反复重跑同一窗
                    if _success and chunk_done_callback:
                        chunk_done_callback(start_chunk + i + 1)
                    if _success and not _window_has_entities:
                        wprint("提示: step7 完成但本窗无实体，仍已计入进度（避免断点卡死）")
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
                        chunk = f"开始阅读新的文档，文件名是：{doc_name}\n\n{chunk}"

                    _wlabel = f"W{start_chunk + ci + 1}/{total_chunks}"
                    if verbose:
                        set_window_label(_wlabel)
                        set_pipeline_role("主线程")
                        wprint(
                            f"【窗口】{_wlabel}｜{doc_name}｜[{start}-{end}/{total_length}] {len(chunk)}字"
                        )
                    elif verbose_steps:
                        set_window_label(_wlabel)
                        set_pipeline_role("主线程")
                        wprint(
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
                                wprint("【步骤1】缓存｜命中｜跳过生成")
                            elif verbose_steps:
                                wprint("【步骤1】缓存｜命中｜跳过生成")
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
                                wprint(
                                    f"【步骤1–5】缓存｜命中｜实体{_ents_count} 关系{_rels_count}→步骤6"
                                )
                            else:
                                wprint(
                                    f"【步骤2–5】缓存｜命中｜实体{_ents_count} 关系{_rels_count}"
                                )
                        elif verbose_steps:
                            if existing_mc:
                                wprint(
                                    f"窗口 {start_chunk + ci + 1}/{total_chunks} · 步骤1–5 已缓存跳过 → 步骤6/7"
                                )
                            else:
                                wprint("【步骤2–5】缓存｜跳过｜抽取已存在")
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
                                )
                                extract_results[idx] = (ents, rels)
                                # 保存抽取结果供断点续传复用
                                self.storage.save_extraction_result(__hash, ents, rels, document_path=document_path)
                                _success_main = True
                            except Exception as e:
                                if _record_window_error("extract", idx, e):
                                    traceback.print_exc()
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
            traceback.print_exc()
        finally:
            clear_parallel_log_context()

        # 等待所有窗口 step7 完成
        for i in range(N):
            step7_done_ev[i].wait()

        if t6.is_alive():
            remember_log("警告: step6 线程在 join 超时后仍在运行")
        t6.join(timeout=60)

        if t7.is_alive():
            remember_log("警告: step7 线程在 join 超时后仍在运行")
        t7.join(timeout=60)

        _prefetch_executor.shutdown(wait=False)

        if _control_state["action"] is not None:
            raise RememberControlFlow(_control_state["action"])

        if errors:
            _phase, _idx, exc = errors[0]
            raise exc

        storage_path = str(self.storage.storage_path)
        return {
            "episode_id": last_episode_id,
            "chunks_processed": total_chunks,
            "storage_path": storage_path,
        }

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
            wprint(f"[Phase1] 文档整体记忆已生成: {overall.absolute_id[:20]}…, doc_name={doc_name!r}")
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
                chunk = f"开始阅读新的文档，文件名是：{doc_name}\n\n{chunk}"

            if verbose:
                wprint(f"\n{'='*60}")
                wprint(f"处理窗口 (文档: {doc_name}, 位置: {start}-{end}/{total_length})")
                wprint(f"输入文本长度: {len(chunk)} 字符")
                wprint(f"{'='*60}\n")
            elif verbose_steps:
                wprint(f"窗口 {chunk_idx + 1}/{total_chunks} 开始 · {doc_name} [{start}-{end}/{total_length}]")

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
    
    def _get_existing_relations_between_entities(self, family_ids: List[str]) -> Dict[str, List[Dict]]:
        """
        检查一组实体之间两两是否存在已有关系
        
        Args:
            family_ids: 实体ID列表
        
        Returns:
            已有关系字典，key为 "entity1_id|entity2_id" 格式（按字母序排序），
            value为该实体对之间的关系列表，每个关系包含:
                - family_id: 关系家族ID
                - content: 关系描述
        """
        # 生成所有实体对并一次性批量查询，替代 O(K^2) 逐对查询
        all_pairs = [
            tuple(sorted((family_ids[i], family_ids[j])))
            for i in range(len(family_ids))
            for j in range(i + 1, len(family_ids))
        ]
        if not all_pairs:
            return {}

        batch_result = self.storage.get_relations_by_entity_pairs(all_pairs)

        existing_relations = {}
        for pair_key, relations in batch_result.items():
            if not relations:
                continue
            str_key = f"{pair_key[0]}|{pair_key[1]}"

            # 按 family_id 分组，每个 family_id 只保留最新版本
            relation_dict = {}
            for rel in relations:
                if rel.family_id not in relation_dict:
                    relation_dict[rel.family_id] = rel
                elif rel.processed_time > relation_dict[rel.family_id].processed_time:
                    relation_dict[rel.family_id] = rel

            existing_relations[str_key] = [
                {
                    'family_id': r.family_id,
                    'content': r.content
                }
                for r in relation_dict.values()
            ]

        return existing_relations
    
    def _is_relation_indicating_same_entity(self, relation_content: str) -> bool:
        """
        判断关系是否表示两个实体是同一实体
        
        Args:
            relation_content: 关系的content描述
        
        Returns:
            如果关系表示同一实体，返回True；否则返回False
        """
        if not relation_content:
            return False

        content = relation_content

        # 高置信度关键词（严格匹配，避免 "就是" 等极常见词误触发）
        high_confidence_phrases = [
            "同一实体", "同一个", "同一人", "同一物", "同一对象",
            r"是同一(?:个|人|物|实体|对象)",
            r"与.{0,4}是同一(?:个|人|物|实体|对象)",
            r"等同于",
            r"指的是同一",
            r"指向的是(?:同一)?(?:实体|对象|人|物)",
            r"(?:也)?(?:就是|即是)(?:同一|别名|简称|同|即)",
        ]
        for pattern in high_confidence_phrases:
            if re.search(pattern, content):
                return True

        # 别名类关键词（需要以特定句式出现，而非单独出现）
        alias_patterns = [
            r"(?:又名|又称|亦称|也叫|也叫做|别称)[是为]?",
            r"(?:别名|简称|全称|昵称|绰号|外号|本名|原名|真名|实名)[是为]",
            r"是(?:.{0,4})(?:的别名|的别称|的简称|的昵称|的绰号|的全称|的本名|的原名|的真名)",
        ]
        for pattern in alias_patterns:
            if re.search(pattern, content):
                return True

        return False
    
    def _check_and_merge_entities_from_relations(self, family_ids: List[str], 
                                                  entities_info: List[Dict],
                                                  version_counts: Dict[str, int],
                                                  merged_family_ids: set,
                                                  merge_mapping: Dict[str, str],
                                                  result: Dict,
                                                  verbose: bool = True) -> Dict[str, List[Dict]]:
        """
        检查实体之间的关系，如果关系表示同一实体，则直接合并
        
        Args:
            family_ids: 实体ID列表
            entities_info: 实体信息列表（包含name等）
            version_counts: 实体版本数统计
            merged_family_ids: 已合并的实体ID集合
            merge_mapping: 合并映射字典
            result: 结果统计字典
            verbose: 是否输出详细信息
        
        Returns:
            过滤后的已有关系字典（已排除表示同一实体的关系）
        """
        existing_relations_between = self._get_existing_relations_between_entities(family_ids)
        
        # 检查是否有关系表示同一实体，如果有则直接合并
        entities_to_merge_from_relations = []
        
        for pair_key, relations in existing_relations_between.items():
            family_ids_pair = pair_key.split("|")
            if len(family_ids_pair) != 2:
                continue
            
            entity1_id, entity2_id = family_ids_pair
            
            # 检查是否有关系表示同一实体
            for rel in relations:
                if self._is_relation_indicating_same_entity(rel['content']):
                    # 找到表示同一实体的关系，准备合并
                    # 选择版本数多的作为target
                    entity1_version_count = version_counts.get(entity1_id, 0)
                    entity2_version_count = version_counts.get(entity2_id, 0)
                    
                    if entity1_version_count >= entity2_version_count:
                        target_id = entity1_id
                        source_id = entity2_id
                    else:
                        target_id = entity2_id
                        source_id = entity1_id
                    
                    # 检查是否已被合并
                    if source_id not in merged_family_ids and target_id not in merged_family_ids:
                        entities_to_merge_from_relations.append({
                            'target_id': target_id,
                            'source_id': source_id,
                            'family_id': rel['family_id'],
                            'relation_content': rel['content']
                        })
                    
                    break  # 只要有一个关系表示同一实体，就合并
        
        # 执行从关系判断出的合并
        if entities_to_merge_from_relations:
            if verbose:
                wprint(f"    发现 {len(entities_to_merge_from_relations)} 对实体通过关系判断为同一实体，直接合并")
            
            for merge_info in entities_to_merge_from_relations:
                target_id = merge_info['target_id']
                source_id = merge_info['source_id']
                relation_content = merge_info['relation_content']
                
                # 执行合并
                merge_result = self.storage.merge_entity_families(target_id, [source_id])
                merge_result["reason"] = f"关系表示同一实体: {relation_content}"
                
                if verbose:
                    target_name = next((e.get('name', '') for e in entities_info if e.get('family_id') == target_id), target_id)
                    source_name = next((e.get('name', '') for e in entities_info if e.get('family_id') == source_id), source_id)
                    wprint(f"      合并实体（基于关系）: {target_name} ({target_id}) <- {source_name} ({source_id})")
                    wprint(f"        原因: {relation_content}")
                
                # 处理合并后产生的自指向关系
                self._handle_self_referential_relations_after_merge(target_id, verbose)
                
                # 记录已合并的实体和合并映射
                merged_family_ids.add(source_id)
                merge_mapping[source_id] = target_id
                
                # 更新结果统计
                result["merge_details"].append(merge_result)
                result["entities_merged"] += merge_result.get("entities_updated", 0)
        
        # 过滤掉已通过关系合并的实体对，只保留非同一实体的关系
        filtered_existing_relations = {}
        for pair_key, relations in existing_relations_between.items():
            family_ids_pair = pair_key.split("|")
            if len(family_ids_pair) != 2:
                continue
            
            entity1_id, entity2_id = family_ids_pair
            
            # 如果这对实体已经通过关系合并了，跳过
            if (entity1_id in merged_family_ids and merge_mapping.get(entity1_id) == entity2_id) or \
               (entity2_id in merged_family_ids and merge_mapping.get(entity2_id) == entity1_id):
                continue
            
            # 过滤掉表示同一实体的关系
            filtered_relations = [
                rel for rel in relations 
                if not self._is_relation_indicating_same_entity(rel['content'])
            ]
            
            if filtered_relations:
                filtered_existing_relations[pair_key] = filtered_relations
        
        return filtered_existing_relations
    
    def _handle_self_referential_relations_after_merge(self, target_family_id: str, verbose: bool = True) -> int:
        """
        处理合并后产生的自指向关系
        
        合并操作会将源实体的family_id更新为目标实体的family_id，这可能导致原本不是自指向的关系变成自指向关系。
        例如：实体A(ent_001)和实体B(ent_002)之间有关系，合并后B的family_id变为ent_001，这个关系就变成了自指向关系。
        
        此方法会：
        1. 检查目标实体是否有自指向关系
        2. 如果有，将这些关系的内容总结到实体的content中
        3. 删除这些自指向关系
        
        Args:
            target_family_id: 合并后的目标实体ID
            verbose: 是否输出详细信息
        
        Returns:
            处理的自指向关系数量
        """
        # 检查是否有自指向关系
        self_ref_relations = self.storage.get_self_referential_relations_for_entity(target_family_id)
        
        if not self_ref_relations:
            return 0
        
        if verbose:
            wprint(f"        检测到合并后产生 {len(self_ref_relations)} 个自指向关系，正在处理...")
        
        # 获取实体的最新版本
        entity = self.storage.get_entity_by_family_id(target_family_id)
        if not entity:
            if verbose:
                wprint(f"        警告：无法获取实体 {target_family_id}")
            return 0
        
        # 收集所有自指向关系的content
        self_ref_contents = [rel['content'] for rel in self_ref_relations if rel.get('content')]
        
        if self_ref_contents:
            # 用LLM总结这些关系内容到实体的content中
            summarized_content = self.llm_client.merge_entity_content(
                old_content=entity.content,
                new_content="\n\n".join([f"属性信息：{content}" for content in self_ref_contents])
            )

            # 内容未变化则跳过版本创建
            if summarized_content.replace(' ', '').replace('\n', '') == entity.content.replace(' ', '').replace('\n', ''):
                if verbose:
                    wprint(f"        内容未变化，跳过版本创建")
                # 仍需删除自指向关系
                actual_deleted = self.storage.delete_self_referential_relations()
                if verbose:
                    wprint(f"      已删除 {actual_deleted} 个自指向关系")
                return len(self_ref_relations)

            # 创建实体的新版本
            new_family_id = f"entity_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            new_entity = Entity(
                absolute_id=new_family_id,
                family_id=entity.family_id,
                name=entity.name,
                content=summarized_content,
                event_time=datetime.now(),
                processed_time=datetime.now(),
                episode_id=entity.episode_id,
                source_document=entity.source_document
            )
            self.storage.save_entity(new_entity)
            
            if verbose:
                wprint(f"        已将 {len(self_ref_contents)} 个自指向关系的内容总结到实体content中")
        
        # 删除这些自指向关系
        deleted_count = self.storage.delete_self_referential_relations_for_entity(target_family_id)
        
        if verbose:
            wprint(f"        已删除 {deleted_count} 个自指向关系")
        
        return deleted_count
    
    def _process_single_alias_relation(self, rel_info: Dict, verbose: bool = True) -> Optional[Dict]:
        """
        处理单个别名关系（可并行调用）
        
        Args:
            rel_info: 关系信息字典，包含：
                - entity1_id, entity2_id: 原始family_id
                - actual_entity1_id, actual_entity2_id: 实际使用的family_id（可能已合并）
                - entity1_name, entity2_name: 实体名称
                - content: 初步的关系content（可选，如果提供则用于初步判断）
            verbose: 是否输出详细信息
        
        Returns:
            处理结果字典，包含：
                - entity1_id, entity2_id: 实体ID
                - entity1_name, entity2_name: 实体名称
                - content: 关系content
                - family_id: 关系家族ID
                - is_new: 是否新创建
                - is_updated: 是否更新
            如果处理失败或跳过，返回None
        """
        actual_entity1_id = rel_info["actual_entity1_id"]
        actual_entity2_id = rel_info["actual_entity2_id"]
        entity1_name = rel_info["entity1_name"]
        entity2_name = rel_info["entity2_name"]
        preliminary_content = rel_info.get("content")  # 初步的content（从分析阶段生成）
        
        if verbose:
            wprint(f"      处理关系: {entity1_name} -> {entity2_name}")
        
        try:
            # 获取两个实体的完整信息
            entity1 = self.storage.get_entity_by_family_id(actual_entity1_id)
            entity2 = self.storage.get_entity_by_family_id(actual_entity2_id)
            
            if not entity1 or not entity2:
                if verbose:
                    wprint(f"        错误：无法获取实体信息")
                return None
            
            # 步骤0：如果有初步content，先用它判断关系是否存在和是否需要更新
            if preliminary_content:
                if verbose:
                    wprint(f"        使用初步content进行预判断: {preliminary_content[:100]}...")
                
                # 检查是否存在关系
                existing_relations_before = self.storage.get_relations_by_entities(
                    actual_entity1_id,
                    actual_entity2_id
                )
                
                if existing_relations_before:
                    # 按 family_id 分组，每个 family_id 只保留最新版本
                    relation_dict = {}
                    for rel in existing_relations_before:
                        if rel.family_id not in relation_dict:
                            relation_dict[rel.family_id] = rel
                        else:
                            if rel.processed_time > relation_dict[rel.family_id].processed_time:
                                relation_dict[rel.family_id] = rel

                    unique_relations = list(relation_dict.values())
                    existing_relations_info = [
                        {
                            'family_id': r.family_id,
                            'content': r.content
                        }
                        for r in unique_relations
                    ]

                    # 构建初步的extracted_relation格式
                    preliminary_extracted_relation = {
                        "entity1_name": entity1.name,
                        "entity2_name": entity2.name,
                        "content": preliminary_content
                    }
                    
                    # 用LLM判断是否匹配已有关系
                    match_result = self.llm_client.judge_relation_match(
                        preliminary_extracted_relation,
                        existing_relations_info
                    )
                    if isinstance(match_result, list) and len(match_result) > 0:
                        match_result = match_result[0] if isinstance(match_result[0], dict) else None
                    elif not isinstance(match_result, dict):
                        match_result = None

                    if match_result and match_result.get('family_id'):
                        # 匹配到已有关系，判断是否需要更新
                        family_id = match_result['family_id']
                        latest_relation = relation_dict.get(family_id)

                        if latest_relation:
                            need_update = self.llm_client.judge_content_need_update(
                                latest_relation.content,
                                preliminary_content
                            )

                            if not need_update:
                                # 不需要更新，直接返回，跳过后续详细生成
                                if verbose:
                                    wprint(f"        关系已存在且无需更新（使用初步content判断），跳过详细生成: {family_id}")
                                return {
                                    "entity1_id": actual_entity1_id,
                                    "entity2_id": actual_entity2_id,
                                    "entity1_name": entity1_name,
                                    "entity2_name": entity2_name,
                                    "content": latest_relation.content,
                                    "family_id": family_id,
                                    "is_new": False,
                                    "is_updated": False
                                }
                            else:
                                if verbose:
                                    wprint(f"        关系已存在但需要更新（使用初步content判断），继续生成详细content: {family_id}")
            
            # 获取实体的episode（只有在需要详细生成时才获取）
            entity1_episode = None
            entity2_episode = None
            if entity1.episode_id:
                from_cache = self.storage.load_episode(entity1.episode_id)
                if from_cache:
                    entity1_episode = from_cache.content

            if entity2.episode_id:
                to_cache = self.storage.load_episode(entity2.episode_id)
                if to_cache:
                    entity2_episode = to_cache.content
            
            # 步骤1：先判断是否真的需要创建关系边（使用完整的实体信息）
            need_create_relation = self.llm_client.judge_need_create_relation(
                entity1_name=entity1.name,
                entity1_content=entity1.content,
                entity2_name=entity2.name,
                entity2_content=entity2.content,
                entity1_episode=entity1_episode,
                entity2_episode=entity2_episode
            )
            
            if not need_create_relation:
                if verbose:
                    wprint(f"        判断结果：两个实体之间没有明确的、有意义的关联，跳过创建关系边")
                return None
            
            if verbose:
                wprint(f"        判断结果：两个实体之间存在明确的、有意义的关联，需要创建关系边")
            
            # 步骤2：生成关系的episode（临时，不保存）
            relation_episode_content = self.llm_client.generate_relation_episode(
                [],  # 关系列表为空，因为还没有生成关系content
                [
                    {"family_id": actual_entity1_id, "name": entity1.name, "content": entity1.content},
                    {"family_id": actual_entity2_id, "name": entity2.name, "content": entity2.content}
                ],
                {
                    actual_entity1_id: entity1_episode or "",
                    actual_entity2_id: entity2_episode or ""
                }
            )
            
            # 步骤3：根据episode和两个实体，生成关系的content
            relation_content = self.llm_client.generate_relation_content(
                entity1_name=entity1.name,
                entity1_content=entity1.content,
                entity2_name=entity2.name,
                entity2_content=entity2.content,
                relation_episode=relation_episode_content,
                preliminary_content=preliminary_content
            )
            
            if verbose:
                wprint(f"        生成关系content: {relation_content}")
            
            # 步骤4：检查是否存在关系
            existing_relations_before = self.storage.get_relations_by_entities(
                actual_entity1_id,
                actual_entity2_id
            )
            
            # 构建extracted_relation格式，用于判断是否需要更新
            extracted_relation = {
                "entity1_name": entity1.name,
                "entity2_name": entity2.name,
                "content": relation_content
            }
            
            # 判断是否需要创建或更新关系
            need_create_or_update = False
            is_new_relation = False
            is_updated = False
            relation = None
            
            if not existing_relations_before:
                # 4a. 如果不存在关系，需要创建新关系
                need_create_or_update = True
                is_new_relation = True
                if verbose:
                    wprint(f"        不存在关系，需要创建新关系")
            else:
                # 4b. 如果存在关系，判断是否需要更新
                # 按 family_id 分组，每个 family_id 只保留最新版本
                relation_dict = {}
                for rel in existing_relations_before:
                    if rel.family_id not in relation_dict:
                        relation_dict[rel.family_id] = rel
                    else:
                        if rel.processed_time > relation_dict[rel.family_id].processed_time:
                            relation_dict[rel.family_id] = rel

                unique_relations = list(relation_dict.values())
                existing_relations_info = [
                    {
                        'family_id': r.family_id,
                        'content': r.content
                    }
                    for r in unique_relations
                ]

                # 用LLM判断是否匹配已有关系
                match_result = self.llm_client.judge_relation_match(
                    extracted_relation,
                    existing_relations_info
                )
                if isinstance(match_result, list) and len(match_result) > 0:
                    match_result = match_result[0] if isinstance(match_result[0], dict) else None
                elif not isinstance(match_result, dict):
                    match_result = None

                if match_result and match_result.get('family_id'):
                    # 匹配到已有关系，判断是否需要更新
                    family_id = match_result['family_id']
                    latest_relation = relation_dict.get(family_id)

                    if latest_relation:
                        need_update = self.llm_client.judge_content_need_update(
                            latest_relation.content,
                            relation_content
                        )

                        if need_update:
                            # 需要更新
                            need_create_or_update = True
                            is_updated = True
                            if verbose:
                                wprint(f"        关系已存在，需要更新: {family_id}")
                        else:
                            # 不需要更新
                            if verbose:
                                wprint(f"        关系已存在，无需更新: {family_id}")
                            relation = latest_relation
                    else:
                        # 找不到匹配的关系，创建新关系
                        need_create_or_update = True
                        is_new_relation = True
                        if verbose:
                            wprint(f"        未找到匹配的关系，创建新关系")
                else:
                    # 没有匹配到已有关系，创建新关系
                    need_create_or_update = True
                    is_new_relation = True
                    if verbose:
                        wprint(f"        未匹配到已有关系，创建新关系")
            
            # 只有在需要创建或更新时，才保存episode并创建/更新关系
            if need_create_or_update:
                # 生成总结的episode（用于json的text字段）
                cache_text_content = f"""实体1:
- name: {entity1.name}
- content: {entity1.content}
- episode: {entity1_episode if entity1_episode else '无'}

实体2:
- name: {entity2.name}
- content: {entity2.content}
- episode: {entity2_episode if entity2_episode else '无'}
"""
                
                # 保存episode（md和json）
                # 从实体中获取文档名
                doc_name_from_entity = entity1.source_document if entity1.source_document else ""
                
                relation_episode = Episode(
                    absolute_id=f"cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                    content=relation_episode_content,
                    event_time=datetime.now(),
                    source_document=doc_name_from_entity,
                    activity_type="知识图谱整理-关系生成"
                )
                # 保存episode，json的text是两个实体的name+content+episode
                self.storage.save_episode(relation_episode, text=cache_text_content)
                
                if verbose:
                    wprint(f"        保存关系episode: {relation_episode.absolute_id}")
                
                relation = self.relation_processor._process_single_relation(
                    extracted_relation,
                    actual_entity1_id,
                    actual_entity2_id,
                    relation_episode.absolute_id,
                    entity1.name,
                    entity2.name,
                    verbose_relation=verbose,
                    source_document=doc_name_from_entity,
                    base_time=relation_episode.event_time,
                )
            
            if relation:
                # 返回关系信息（用于后续统计）
                alias_detail = {
                    "entity1_id": actual_entity1_id,
                    "entity2_id": actual_entity2_id,
                    "entity1_name": entity1.name,
                    "entity2_name": entity2.name,
                    "content": relation_content,
                    "family_id": relation.family_id,
                    "is_new": is_new_relation,
                    "is_updated": is_updated
                }
                
                if is_new_relation:
                    if verbose:
                        wprint(f"        成功创建新关系: {relation.family_id}")
                elif is_updated:
                    if verbose:
                        wprint(f"        关系已存在，已更新: {relation.family_id}")
                else:
                    if verbose:
                        wprint(f"        关系已存在，无需更新: {relation.family_id}")
                
                return alias_detail
            else:
                if verbose:
                    wprint(f"        创建关系失败")
                return None
                    
        except Exception as e:
            if verbose:
                wprint(f"        处理失败: {e}")
            if verbose:
                traceback.print_exc()
            return None
    
    def _finalize_consolidation(self, result: Dict, all_analyzed_entities_text: List[str], verbose: bool = True):
        """
        完成知识图谱整理的收尾工作（创建总结记忆缓存）
        
        Args:
            result: 整理结果字典
            all_analyzed_entities_text: 所有分析过的实体文本列表
            verbose: 是否输出详细信息
        """
        # 步骤5：创建整理总结记忆缓存
        if verbose:
            wprint(f"\n步骤5: 创建整理总结记忆缓存...")
        
        consolidation_summary = self.llm_client.generate_consolidation_summary(
            result["merge_details"],
            result["alias_details"],
            result["entities_analyzed"]
        )
        
        # 构建整理结果摘要文本（用于保存到JSON的text字段）
        # 包含所有分析过的实体列表 + 整理结果摘要
        consolidation_text = f"""知识图谱整理完成

整理结果摘要：
- 分析的实体数: {result['entities_analyzed']}
- 合并的实体记录数: {result['entities_merged']}
- 创建的关联关系数: {result['alias_relations_created']}

合并详情:
"""
        for merge_detail in result.get("merge_details", []):
            target_name = merge_detail.get("target_name", "未知")
            source_names = merge_detail.get("source_names", [])
            consolidation_text += f"  - {target_name} <- {', '.join(source_names)}\n"
        
        consolidation_text += "\n关联关系详情:\n"
        for alias_detail in result.get("alias_details", []):
            entity1_name = alias_detail.get("entity1_name", "未知")
            entity2_name = alias_detail.get("entity2_name", "未知")
            is_new = alias_detail.get("is_new", False)
            is_updated = alias_detail.get("is_updated", False)
            status = "新建" if is_new else ("更新" if is_updated else "已存在")
            consolidation_text += f"  - {entity1_name} -> {entity2_name} ({status})\n"
        
        # 添加所有分析过的实体列表信息
        if all_analyzed_entities_text:
            consolidation_text += "\n\n" + "="*80
            consolidation_text += "\n所有传入LLM进行判断的实体列表\n"
            consolidation_text += "="*80
            consolidation_text += "".join(all_analyzed_entities_text)
        
        # 创建总结性的记忆缓存
        summary_cache = Episode(
            absolute_id=f"cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            content=f"""# 知识图谱整理总结

## 整理总结

{consolidation_summary}
""",
            event_time=datetime.now(),
            source_document="",  # 知识图谱整理总结不关联特定文档
            activity_type="知识图谱整理总结"
        )
        
        # 保存总结记忆缓存
        self.storage.save_episode(
            summary_cache, 
            text=consolidation_text
        )
        
        if verbose:
            wprint(f"  已创建整理总结记忆缓存: {summary_cache.absolute_id}")
    
    def _build_entity_list_text(self, entities_for_analysis: List[Dict]) -> str:
        """
        构建包含完整entity信息的文本（用于保存到JSON的text字段）
        
        Args:
            entities_for_analysis: 传入LLM分析的实体列表
            
        Returns:
            包含完整实体信息的文本（包括family_id, name, content等）
        """
        text_lines = []
        text_lines.append(f"知识图谱整理 - 传入LLM进行判断的实体列表（共 {len(entities_for_analysis)} 个实体）\n")
        text_lines.append("=" * 80)
        text_lines.append("")
        
        for idx, entity_info in enumerate(entities_for_analysis, 1):
            family_id = entity_info.get("family_id", "未知")
            name = entity_info.get("name", "未知")
            content = entity_info.get("content", "")
            version_count = entity_info.get("version_count", 0)
            
            text_lines.append(f"{idx}. 实体名称: {name}")
            text_lines.append(f"   family_id: {family_id}")
            text_lines.append(f"   版本数: {version_count}")
            text_lines.append(f"   完整内容:")
            text_lines.append(f"   {content}")
            text_lines.append("")
            text_lines.append("-" * 80)
            text_lines.append("")
        
        return "\n".join(text_lines)


def main():
    """示例使用"""
    
    # 配置
    storage_path = "./tmg_storage"
    document_paths = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if not document_paths:
        wprint("用法: python -m Temporal_Memory_Graph.processor <文档路径1> [文档路径2] ...")
        wprint("示例: python -m Temporal_Memory_Graph.processor doc1.txt doc2.txt")
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
    wprint("\n处理完成！")
    wprint(f"统计信息: {stats}")


if __name__ == "__main__":
    main()
