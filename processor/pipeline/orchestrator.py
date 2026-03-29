"""
主处理流程：整合所有模块，实现完整的文档处理pipeline
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import sys
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
from .entity import EntityProcessor
from .relation import RelationProcessor
from ..models import MemoryCache, Entity
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
                 prompt_memory_cache_max_chars: Optional[int] = None,
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
            content_snippet_length: 实体 content 截取长度（默认 50）
            relation_content_snippet_length: 关系 content 截取长度（默认 50）
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
            llm_context_window_tokens: 模型总上下文 token 上限（输入+输出），与 service_config.llm.context_window_tokens 一致；未传时读 server 默认
            prompt_memory_cache_max_chars: 注入抽取 prompt 的记忆缓存最大字符数；超长时自动截断，默认 2000
        """
        _content_snippet_length = content_snippet_length if content_snippet_length is not None else 50
        _relation_content_snippet_length = relation_content_snippet_length if relation_content_snippet_length is not None else 50
        _relation_endpoint_jaccard_threshold = (
            float(relation_endpoint_jaccard_threshold)
            if relation_endpoint_jaccard_threshold is not None else 0.9
        )
        _rel_emb_thr = relation_endpoint_embedding_threshold
        if _rel_emb_thr is None:
            _relation_endpoint_embedding_threshold = 0.72
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
            prompt_memory_cache_max_chars=prompt_memory_cache_max_chars,
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
        self.current_memory_cache: Optional[MemoryCache] = None
        
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
        new_memory_cache: MemoryCache,
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
                new_memory_cache,
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
                latest_metadata = self.storage.get_latest_memory_cache_metadata(activity_type="文档处理")
                
                if latest_metadata:
                    # 加载缓存记忆
                    self.current_memory_cache = self.storage.load_memory_cache(latest_metadata['absolute_id'])
                    
                    if self.current_memory_cache:
                        if verbose:
                            wprint(f"已加载缓存记忆: {self.current_memory_cache.absolute_id} (时间: {self.current_memory_cache.event_time})")
                        
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
                    self.current_memory_cache = None
            else:
                if verbose:
                    wprint("不加载缓存记忆，将从头开始处理")
                self.current_memory_cache = None
            
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
            dict: memory_cache_id, chunks_processed, storage_path
        """
        doc_name = source_document or doc_name
        use_load_cache = load_cache_memory if load_cache_memory is not None else self.load_cache_memory
        # 仅在真正的断点续传（start_chunk > 0）时加载已有缓存链；
        # start_chunk == 0 表示从头开始，加载旧缓存会导致 step1 重复处理已有内容
        if use_load_cache and start_chunk > 0:
            latest_metadata = self.storage.get_latest_memory_cache_metadata(activity_type="文档处理")
            if latest_metadata:
                self.current_memory_cache = self.storage.load_memory_cache(latest_metadata["absolute_id"])
                if verbose and self.current_memory_cache:
                    remember_log(
                        f"已加载缓存记忆: {self.current_memory_cache.absolute_id}，"
                        f"将在此链上追加（断点续传 start_chunk={start_chunk}）"
                    )
                elif verbose_steps and self.current_memory_cache:
                    remember_log("已加载缓存记忆（断点续传）")
            else:
                self.current_memory_cache = None
        else:
            self.current_memory_cache = None
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
                "memory_cache_id": getattr(self.current_memory_cache, 'absolute_id', None),
                "chunks_processed": total_chunks,
                "storage_path": str(self.storage.storage_path),
            }

        N = total_chunks - start_chunk  # 待处理窗口数
        last_memory_cache_id = None
        clear_parallel_log_context()

        # 预分配数组，用于线程间共享数据
        memory_caches = [None] * N
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
            with _progress_lock:
                pass
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
                    mc = memory_caches[i]
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
                        import traceback
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
                    except Exception:
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
                    except Exception:
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
                    mc = memory_caches[i]
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
                        import traceback
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
                        self.current_memory_cache = existing_mc
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
                    memory_caches[ci] = new_mc
                    input_texts[ci] = chunk
                    last_memory_cache_id = new_mc.absolute_id

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
                                    import traceback
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
            import traceback
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
            "memory_cache_id": last_memory_cache_id,
            "chunks_processed": total_chunks,
            "storage_path": storage_path,
        }

    def remember_phase1_overall(self, text: str, doc_name: str = "api_input",
                                event_time: Optional[datetime] = None,
                                document_path: str = "",
                                previous_overall_cache: Optional[MemoryCache] = None,
                                verbose: bool = False,
                                progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> MemoryCache:
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
                                overall_cache: Optional[MemoryCache] = None,
                                progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict:
        """
        阶段2：以整体记忆为起点，跑完所有滑窗（更新缓存 + 抽取实体/关系并写入）。
        overall_cache 即 phase1 返回的文档整体记忆，作为第一窗的 current_cache。
        """
        if not document_path:
            document_path = f"api://{uuid.uuid4().hex}"
        self.current_memory_cache = overall_cache  # 第一窗的 _update_cache 会在此基础上续写
        window_size = self.document_processor.window_size
        overlap = self.document_processor.overlap
        total_length = len(text)
        start = 0
        chunk_idx = 0
        last_memory_cache_id = None
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
            last_memory_cache_id = new_mc.absolute_id
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
            "memory_cache_id": last_memory_cache_id,
            "chunks_processed": chunk_idx,
            "storage_path": str(self.storage.storage_path),
        }

    def get_statistics(self) -> dict:
        """获取处理统计信息"""
        # 这里可以添加统计逻辑
        return {
            "memory_caches": len(list(self.storage.cache_dir.glob("*.json"))),
            "storage_path": str(self.storage.storage_path)
        }
    
    def consolidate_knowledge_graph_entity(self, verbose: bool = True,
                                    similarity_threshold: float = 0.6,
                                    max_candidates: int = 5,
                                    batch_candidates: Optional[int] = None,
                                    content_snippet_length: int = 64,
                                    parallel: bool = False,
                                    enable_name_match_step: bool = True,
                                    enable_pre_search: Optional[bool] = None) -> dict:
        # NOTE: 此方法与 consolidation.py _ConsolidationMixin 中为重复实现。
        # 如需修改 consolidate 逻辑，两处均需同步更新。
        """
        整理知识图谱：识别并合并重复实体，创建关联关系
        
        对每个实体，分别按name和name+content搜索相似实体，使用LLM判断是否需要合并或创建关系边。
        
        Args:
            verbose: 是否输出详细信息
            similarity_threshold: 相似度搜索阈值（默认0.6）
            max_candidates: 每次搜索返回的最大候选实体数（默认5）
            batch_candidates: 每次批量处理的候选实体数（默认None，表示不限制，一次性处理所有max_candidates个）
                            如果设置了且小于max_candidates，则分批处理，每批处理batch_candidates个
                            如果大于等于max_candidates，则按max_candidates的值处理
            content_snippet_length: 传入LLM的实体content最大长度（默认64字符）
            parallel: 是否启用多线程并行处理（默认False）
            enable_name_match_step: 是否启用步骤1.5（按名称完全匹配进行初步整理，默认True）
            enable_pre_search: 是否启用预搜索（步骤2）。如果为None，则根据parallel自动决定：
                              - parallel=True时，必须启用（强制为True）
                              - parallel=False时，默认启用（True），但可以设置为False改为按需搜索
        
        Returns:
            整理结果统计，包含:
            - entities_analyzed: 分析的实体数量
            - entities_merged: 合并的实体数量
            - alias_relations_created: 创建的关联关系数量
            - merge_details: 合并操作详情列表
            - alias_details: 关联关系详情列表
        """
        # 如果启用并行处理且线程数大于1，使用多线程版本
        if parallel and self.llm_threads > 1:
            return self._consolidate_knowledge_graph_parallel(
                verbose=verbose,
                similarity_threshold=similarity_threshold,
                max_candidates=max_candidates,
                batch_candidates=batch_candidates,
                content_snippet_length=content_snippet_length
            )
        
        # 确定是否启用预搜索
        # 如果parallel=True，必须启用预搜索（但这种情况应该已经进入上面的并行版本）
        # 如果parallel=False，根据enable_pre_search参数决定
        if enable_pre_search is None:
            # 默认启用预搜索（批量计算更高效）
            use_pre_search = True
        else:
            use_pre_search = enable_pre_search

        if verbose:
            wprint("=" * 60)
            wprint("开始知识图谱整理...")
            wprint("=" * 60)
        
        # 步骤0：处理自指向的关系，将其总结到实体的content中
        if verbose:
            wprint(f"\n步骤0: 处理自指向的关系（总结到实体content）...")
        
        self_ref_relations = self.storage.get_self_referential_relations()
        entities_updated_from_self_ref = 0
        deleted_self_ref_count = 0
        
        if self_ref_relations:
            if verbose:
                wprint(f"  发现 {len(self_ref_relations)} 个实体有自指向关系，共 {sum(len(rels) for rels in self_ref_relations.values())} 个关系")
            
            for entity_id, relations in self_ref_relations.items():
                # 获取实体的最新版本
                entity = self.storage.get_entity_by_id(entity_id)
                if not entity:
                    continue
                
                # 收集所有自指向关系的content
                self_ref_contents = [rel['content'] for rel in relations]
                
                if verbose:
                    wprint(f"    处理实体 {entity.name} ({entity_id})，有 {len(relations)} 个自指向关系")
                
                # 用LLM总结这些关系内容到实体的content中
                # 将自指向关系的内容视为实体的属性信息
                summarized_content = self.llm_client.merge_entity_content(
                    old_content=entity.content,
                    new_content="\n\n".join([f"属性信息：{content}" for content in self_ref_contents])
                )
                
                # 更新实体的最新版本（创建新版本）
                from datetime import datetime
                new_entity_id = f"entity_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                new_entity = Entity(
                    absolute_id=new_entity_id,
                    entity_id=entity.entity_id,
                    name=entity.name,
                    content=summarized_content,
                    event_time=datetime.now(),
                    processed_time=datetime.now(),
                    memory_cache_id=entity.memory_cache_id,
                    source_document=entity.source_document
                )
                self.storage.save_entity(new_entity)

                entities_updated_from_self_ref += 1
                deleted_self_ref_count += len(relations)
                
                if verbose:
                    wprint(f"      已将 {len(relations)} 个自指向关系总结到实体content中")
            
            # 删除所有自指向的关系
            actual_deleted = self.storage.delete_self_referential_relations()
            if verbose:
                wprint(f"  已删除 {actual_deleted} 个自指向的关系")
        else:
            if verbose:
                wprint(f"  未发现自指向的关系")
        
        # 结果统计
        result = {
            "entities_analyzed": 0,
            "entities_merged": 0,
            "alias_relations_created": 0,
            "alias_relations_updated": 0,  # 新增：更新的关系数量
            "self_referential_relations_processed": deleted_self_ref_count,  # 处理的自指向关系数量
            "entities_updated_from_self_ref": entities_updated_from_self_ref,  # 因自指向关系而更新的实体数量
            "merge_details": [],
            "alias_details": []
        }
        
        # 步骤1：获取所有实体
        if verbose:
            wprint(f"\n步骤1: 获取所有实体...")
        
        all_entities = self.storage.get_all_entities()
        
        if not all_entities:
            if verbose:
                wprint("  知识库中没有实体。")
            return result
        
        # 按版本数量从大到小排序
        entity_ids = [entity.entity_id for entity in all_entities]
        version_counts = self.storage.get_entity_version_counts(entity_ids)
        all_entities.sort(key=lambda e: version_counts.get(e.entity_id, 0), reverse=True)
        
        # 记录整理前的实体总数
        initial_entity_count = len(all_entities)
        if verbose:
            wprint(f"  整理前共有 {initial_entity_count} 个实体")
        
        # 记录已合并的实体ID（用于后续embedding搜索时排除）
        merged_entity_ids = set()
        # 记录合并映射：source_entity_id -> target_entity_id
        merge_mapping = {}
        
        # 步骤1.5：先按名称完全匹配进行整理
        if enable_name_match_step:
            if verbose:
                wprint(f"\n步骤1.5: 按名称完全匹配进行初步整理...")
            
            # 构建名称到实体列表的映射
            name_to_entities = {}
            for entity in all_entities:
                name = entity.name
                if name not in name_to_entities:
                    name_to_entities[name] = []
                name_to_entities[name].append(entity)
            
            # 对每个名称组内的实体按版本数排序（从大到小）
            for name in name_to_entities:
                name_to_entities[name].sort(
                    key=lambda e: version_counts.get(e.entity_id, 0), 
                    reverse=True
                )
            
            # 按照每个名称组中实体的最大版本数排序（从大到小），然后按顺序处理
            name_groups_sorted = sorted(
                name_to_entities.items(),
                key=lambda item: max(
                    (version_counts.get(e.entity_id, 0) for e in item[1]),
                    default=0
                ),
                reverse=True
            )
            
            # 处理名称完全一致的实体组
            name_match_count = 0
            for name, entities_with_same_name in name_groups_sorted:
                # 只处理有多个实体的名称组
                if len(entities_with_same_name) <= 1:
                    continue
                
                name_match_count += 1
                if verbose:
                    wprint(f"  发现名称完全一致的实体组: {name} (共 {len(entities_with_same_name)} 个实体)")
                
                # 准备实体信息用于LLM判断
                entities_info = []
                for entity in entities_with_same_name:
                    # 跳过已合并的实体
                    if entity.entity_id in merged_entity_ids:
                        continue
                    
                    version_count = version_counts.get(entity.entity_id, 0)
                    entities_info.append({
                        "entity_id": entity.entity_id,
                        "name": entity.name,
                        "content": entity.content,
                        "version_count": version_count
                    })
                
                # 如果过滤后只剩一个或没有实体，跳过
                if len(entities_info) <= 1:
                    continue
                
                # 获取记忆上下文
                memory_contexts = {}
                for entity in entities_with_same_name:
                    if entity.entity_id in merged_entity_ids:
                        continue
                    cache_text = self.storage.get_memory_cache_text(entity.memory_cache_id)
                    if cache_text:
                        memory_contexts[entity.entity_id] = cache_text
                
                # 检查实体对之间是否已有关系，如果关系表示同一实体则直接合并
                entity_ids_for_check = [info['entity_id'] for info in entities_info]
                existing_relations_between = self._check_and_merge_entities_from_relations(
                    entity_ids_for_check,
                    entities_info,
                    version_counts,
                    merged_entity_ids,
                    merge_mapping,
                    result,
                    verbose
                )
                
                if verbose and existing_relations_between:
                    wprint(f"    发现 {len(existing_relations_between)} 对实体之间已有关系，将交由LLM判断是否应该合并")
                
                # 调用LLM分析：判断是合并还是关联关系
                analysis_result = self.llm_client.analyze_entity_duplicates(
                    entities_info,
                    memory_contexts,
                    content_snippet_length=content_snippet_length,
                    existing_relations_between_entities=existing_relations_between
                )
                
                if "error" in analysis_result:
                    if verbose:
                        wprint(f"    分析失败，跳过该组")
                    continue
            
            # 处理合并（过滤掉已有关系的实体对）
            merge_groups = analysis_result.get("merge_groups", [])
            for merge_group in merge_groups:
                target_entity_id = merge_group.get("target_entity_id")
                source_entity_ids = merge_group.get("source_entity_ids", [])
                reason = merge_group.get("reason", "")
                
                if not target_entity_id or not source_entity_ids:
                    continue
                
                # 检查是否已被合并
                if any(sid in merged_entity_ids for sid in source_entity_ids):
                    continue
                
                # 不再过滤已有关系的实体对，让LLM判断是否应该合并
                # 即使有关系，如果关系表示同一实体，也应该合并
                
                # 执行合并
                merge_result = self.storage.merge_entity_ids(target_entity_id, source_entity_ids)
                merge_result["reason"] = reason
                
                if verbose:
                    target_name = next((e.name for e in entities_with_same_name if e.entity_id == target_entity_id), target_entity_id)
                    wprint(f"    合并实体: {target_name} ({target_entity_id}) <- {len(source_entity_ids)} 个源实体")
                    wprint(f"      原因: {reason}")
                
                # 处理合并后产生的自指向关系
                self._handle_self_referential_relations_after_merge(target_entity_id, verbose)
                
                # 记录已合并的实体和合并映射
                for sid in source_entity_ids:
                    merged_entity_ids.add(sid)
                    merge_mapping[sid] = target_entity_id
                
                # 更新结果统计
                result["merge_details"].append(merge_result)
                result["entities_merged"] += merge_result.get("entities_updated", 0)
            
            # 处理关系（别名关系）
            alias_relations = analysis_result.get("alias_relations", [])
            for alias_info in alias_relations:
                entity1_id = alias_info.get("entity1_id")
                entity2_id = alias_info.get("entity2_id")
                entity1_name = alias_info.get("entity1_name", "")
                entity2_name = alias_info.get("entity2_name", "")
                preliminary_content = alias_info.get("content")
                
                if not entity1_id or not entity2_id:
                    continue
                
                # 检查是否已被合并（如果已合并，需要找到合并后的实际ID）
                actual_entity1_id = merge_mapping.get(entity1_id, entity1_id)
                actual_entity2_id = merge_mapping.get(entity2_id, entity2_id)
                
                # 如果实体已被合并，跳过（因为合并后的实体可能不在当前名称组中）
                if entity1_id in merged_entity_ids or entity2_id in merged_entity_ids:
                    if verbose:
                        wprint(f"    跳过关系（实体已合并）: {entity1_name} -> {entity2_name}")
                    continue
                
                # 处理关系
                rel_info = {
                    "entity1_id": entity1_id,
                    "entity2_id": entity2_id,
                    "actual_entity1_id": actual_entity1_id,
                    "actual_entity2_id": actual_entity2_id,
                    "entity1_name": entity1_name,
                    "entity2_name": entity2_name,
                    "content": preliminary_content
                }
                
                rel_result = self._process_single_alias_relation(rel_info, verbose=False)
                if rel_result:
                    result["alias_details"].append(rel_result)
                    if rel_result.get("is_new"):
                        result["alias_relations_created"] += 1
                    elif rel_result.get("is_updated"):
                        result["alias_relations_updated"] += 1
            
            if verbose:
                wprint(f"  名称匹配完成，处理了 {name_match_count} 个名称组，合并了 {len(merged_entity_ids)} 个实体")
        else:
            if verbose:
                wprint(f"\n步骤1.5: 跳过（已禁用）")
        
        # 步骤1.5之后，重新按版本数量从大到小排序（因为合并可能改变了版本数）
        entity_ids = [entity.entity_id for entity in all_entities]
        version_counts = self.storage.get_entity_version_counts(entity_ids)
        all_entities.sort(key=lambda e: version_counts.get(e.entity_id, 0), reverse=True)
        
        # 用于累积所有分析过的实体信息（用于最终保存到JSON的text字段）
        all_analyzed_entities_text = []
        
        # 记录已处理的entity_id对，避免重复分析
        processed_pairs = set()
        
        # 步骤2：使用混合检索方式一次性找到所有实体的关联实体（可选）
        entity_to_candidates = {}
        
        if use_pre_search:
            if verbose:
                wprint(f"\n步骤2: 使用混合检索方式预搜索所有实体的关联实体（阈值: {similarity_threshold}, 最大候选数: {max_candidates}）...")
                wprint(f"  使用多种检索模式：name_only(embedding) + name_and_content(embedding) + name_only(text/jaccard)")
            
            # 定义进度回调函数
            def progress_callback(current: int, total: int, entity_name: str):
                if verbose and current % max(1, total // 20) == 0 or current == total:  # 每5%或最后一个显示一次
                    percentage = (current / total) * 100
                    wprint(f"  预搜索进度: [{current}/{total}] ({percentage:.1f}%) - 当前处理: {entity_name[:30]}...")
            
            # 使用混合检索方式一次性找到所有实体的关联实体
            entity_to_candidates = self.storage.find_related_entities_by_embedding(
                similarity_threshold=similarity_threshold,
                max_candidates=max_candidates,
                use_mixed_search=True,  # 启用混合检索
                content_snippet_length=content_snippet_length,
                progress_callback=progress_callback if verbose else None
            )
            
            # 过滤掉已合并的实体（在候选列表中排除）
            for entity_id in list(entity_to_candidates.keys()):
                # 如果当前实体已合并，从候选列表中移除
                if entity_id in merged_entity_ids:
                    del entity_to_candidates[entity_id]
                    continue
                
                # 从候选列表中排除已合并的实体
                candidates = entity_to_candidates[entity_id]
                entity_to_candidates[entity_id] = candidates - merged_entity_ids
            
            if verbose:
                total_candidates = sum(len(candidates) for candidates in entity_to_candidates.values())
                wprint(f"  预搜索完成，共 {len(entity_to_candidates)} 个实体，找到 {total_candidates} 个关联实体（已排除 {len(merged_entity_ids)} 个已合并实体）")
        else:
            if verbose:
                wprint(f"\n步骤2: 跳过预搜索，将按需搜索每个实体的关联实体")
        
        if verbose:
            wprint(f"\n步骤3: 逐个实体分析并处理...")
        
        for entity_idx, entity in enumerate(all_entities, 1):
            # 跳过已被合并的实体
            if entity.entity_id in merged_entity_ids:
                continue
            
            if verbose:
                # 获取实体的版本数
                entity_version_count = version_counts.get(entity.entity_id, 0)
                wprint(f"\n  [{entity_idx}/{len(all_entities)}] 分析实体: {entity.name} (entity_id: {entity.entity_id}, 版本数: {entity_version_count})")
            
            # 获取候选实体：如果启用了预搜索，从预搜索结果中获取；否则按需搜索
            if use_pre_search:
                candidate_entity_ids = entity_to_candidates.get(entity.entity_id, set())
            else:
                # 按需搜索：使用混合检索方式搜索当前实体的关联实体
                candidate_entity_ids = set()

                # 模式1：只用name检索（使用embedding）
                candidates_name_jaccard = self.storage.search_entities_by_similarity(
                    query_name=entity.name,
                    query_content=None,
                    threshold=0.0,
                    max_results=max_candidates,
                    content_snippet_length=content_snippet_length,
                    text_mode="name_only",
                    similarity_method="jaccard"
                )
                
                # 模式1：只用name检索（使用embedding）
                candidates_name_embedding = self.storage.search_entities_by_similarity(
                    query_name=entity.name,
                    query_content=None,
                    threshold=similarity_threshold,
                    max_results=max_candidates,
                    content_snippet_length=content_snippet_length,
                    text_mode="name_only",
                    similarity_method="embedding"
                )
                
                # 模式2：使用name+content检索（使用embedding）
                candidates_full_embedding = self.storage.search_entities_by_similarity(
                    query_name=entity.name,
                    query_content=entity.content,
                    threshold=similarity_threshold,
                    max_results=max_candidates,
                    content_snippet_length=content_snippet_length,
                    text_mode="name_and_content",
                    similarity_method="embedding"
                )
                
                # 合并候选实体并去重（按entity_id去重，保留每个entity_id的最新版本）
                candidate_dict = {}
                for candidate in candidates_name_jaccard + candidates_name_embedding + candidates_full_embedding:
                    if candidate.entity_id == entity.entity_id:
                        continue  # 跳过自己
                    if candidate.entity_id not in candidate_dict:
                        candidate_dict[candidate.entity_id] = candidate
                    else:
                        # 保留处理时间最新的
                        if candidate.processed_time > candidate_dict[candidate.entity_id].processed_time:
                            candidate_dict[candidate.entity_id] = candidate
                
                # 提取entity_id到set中
                candidate_entity_ids = {cid for cid in candidate_dict.keys()}
            
            # 过滤掉已处理的配对和已合并的实体
            candidate_entity_ids = {
                cid for cid in candidate_entity_ids 
                if cid not in merged_entity_ids and 
                   (min(entity.entity_id, cid), max(entity.entity_id, cid)) not in processed_pairs
            }
            
            if not candidate_entity_ids:
                if verbose:
                    wprint(f"    未找到相似实体候选")
                continue
            
            # 确定批量处理的大小
            if batch_candidates is not None and batch_candidates < max_candidates:
                batch_size = batch_candidates
            else:
                batch_size = max_candidates
            
            # 将候选实体转换为列表并分批处理
            candidate_entity_ids_list = list(candidate_entity_ids)
            total_candidates = len(candidate_entity_ids_list)
            total_batches = (total_candidates + batch_size - 1) // batch_size  # 向上取整
            
            if verbose:
                wprint(f"    找到 {total_candidates} 个候选实体，将分 {total_batches} 批处理（每批 {batch_size} 个）")
            
            # 准备当前实体信息（所有批次共享）
            current_version_count = self.storage.get_entity_version_count(entity.entity_id)
            current_entity_info = {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "content": entity.content,
                "version_count": current_version_count
            }
            
            # ========== 阶段1: 分批初步筛选（只收集候选，不执行操作） ==========
            # 收集所有批次的候选
            all_possible_merges = []  # 所有可能需要合并的候选
            all_possible_relations = []  # 所有可能需要创建关系的候选
            all_candidates_full_info = {}  # 所有候选实体的完整信息
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_candidates)
                batch_candidate_ids = candidate_entity_ids_list[start_idx:end_idx]
                
                if verbose:
                    wprint(f"\n    [初步筛选] 第 {batch_idx + 1}/{total_batches} 批（{len(batch_candidate_ids)} 个候选实体）...")
                
                # 获取当前批次的候选实体完整信息
                candidates_info = []
                for cid in batch_candidate_ids:
                    candidate_entity = self.storage.get_entity_by_id(cid)
                    if candidate_entity:
                        version_count = self.storage.get_entity_version_count(cid)
                        info = {
                            "entity_id": cid,
                            "name": candidate_entity.name,
                            "content": candidate_entity.content,
                            "version_count": version_count
                        }
                        candidates_info.append(info)
                        all_candidates_full_info[cid] = info
                        # 记录已处理的配对
                        pair = (min(entity.entity_id, cid), max(entity.entity_id, cid))
                        processed_pairs.add(pair)
                
                if not candidates_info:
                    continue
                
                # 按版本数量从大到小排序候选实体
                candidates_info.sort(key=lambda x: x.get('version_count', 0), reverse=True)
                
                # 构建分析组：当前实体 + 当前批次的候选实体
                entities_for_analysis = [current_entity_info] + candidates_info
                
                if verbose:
                    wprint(f"      当前批次候选实体:")
                    for info in candidates_info:
                        wprint(f"        - {info['name']} (entity_id: {info['entity_id']}, versions: {info['version_count']})")
                
                # 初步筛选（使用snippet）- 只收集候选，不执行任何操作
                preliminary_result = self.llm_client.analyze_entity_candidates_preliminary(
                    entities_for_analysis,
                    content_snippet_length=content_snippet_length
                )
                
                possible_merges = preliminary_result.get("possible_merges", [])
                possible_relations = preliminary_result.get("possible_relations", [])
                no_action = preliminary_result.get("no_action", [])
                preliminary_summary = preliminary_result.get("analysis_summary", "")
                
                if verbose:
                    if preliminary_summary:
                        wprint(f"      初步筛选结果: {preliminary_summary[:100]}..." if len(preliminary_summary) > 100 else f"      初步筛选结果: {preliminary_summary}")
                    wprint(f"      可能需要合并: {len(possible_merges)} 个, 可能存在关系: {len(possible_relations)} 个, 不处理: {len(no_action)} 个")
                
                # 收集候选（记录当前实体和候选实体的配对）
                for item in possible_merges:
                    cid = item.get("entity_id") if isinstance(item, dict) else item
                    if cid and cid not in merged_entity_ids:
                        all_possible_merges.append({
                            "current_entity_id": entity.entity_id,
                            "current_entity_info": current_entity_info,
                            "candidate_entity_id": cid,
                            "reason": item.get("reason", "") if isinstance(item, dict) else ""
                        })
                
                for item in possible_relations:
                    cid = item.get("entity_id") if isinstance(item, dict) else item
                    if cid and cid not in merged_entity_ids:
                        all_possible_relations.append({
                            "current_entity_id": entity.entity_id,
                            "current_entity_info": current_entity_info,
                            "candidate_entity_id": cid,
                            "reason": item.get("reason", "") if isinstance(item, dict) else ""
                        })
            
            # ========== 阶段2: 精细化判断（所有批次完成后） ==========
            # 对于被判断为关系的候选，先检查是否已有关系，如果有则跳过精细化判断
            filtered_possible_relations = []
            skipped_relations_count = 0
            for item in all_possible_relations:
                cid = item["candidate_entity_id"]
                # 检查是否已有关系
                existing_rels = self.storage.get_relations_by_entities(
                    entity.entity_id,
                    cid
                )
                if existing_rels and len(existing_rels) > 0:
                    # 已有关系，跳过精细化判断
                    skipped_relations_count += 1
                    if verbose:
                        # 获取候选实体名称
                        candidate_name = cid
                        if cid in all_candidates_full_info:
                            candidate_name = all_candidates_full_info[cid].get('name', cid)
                        else:
                            candidate_entity = self.storage.get_entity_by_id(cid)
                            if candidate_entity:
                                candidate_name = candidate_entity.name
                        wprint(f"      跳过已有关系: {entity.name} <-> {candidate_name} (已有 {len(existing_rels)} 个关系)")
                else:
                    # 没有关系，需要精细化判断
                    filtered_possible_relations.append(item)
            
            if verbose:
                total_candidates_to_analyze = len(all_possible_merges) + len(filtered_possible_relations)
                wprint(f"\n    [精细化判断] 共 {total_candidates_to_analyze} 个候选需要精细化判断...")
                wprint(f"      可能合并: {len(all_possible_merges)} 个")
                wprint(f"      可能关系: {len(filtered_possible_relations)} 个 (跳过已有关系: {skipped_relations_count} 个)")
            
            # 合并可能合并和可能关系的候选（去重）
            all_candidates_to_analyze = {}
            for item in all_possible_merges + filtered_possible_relations:
                cid = item["candidate_entity_id"]
                if cid not in all_candidates_to_analyze:
                    all_candidates_to_analyze[cid] = item
            
            # 对每个候选进行精细化判断
            merge_decisions = []  # 精细化判断后确定要合并的
            relation_decisions = []  # 精细化判断后确定要创建关系的
            
            for cid, item in all_candidates_to_analyze.items():
                if cid not in all_candidates_full_info:
                    continue
                
                # 检查是否已被合并
                if cid in merged_entity_ids:
                    continue
                
                candidate_info = all_candidates_full_info[cid]
                
                # 获取两个实体之间的已有关系
                existing_rels = self.storage.get_relations_by_entities(
                    entity.entity_id,
                    cid
                )
                existing_relations_list = []
                if existing_rels:
                    # 去重，每个relation_id只保留最新版本
                    rel_dict = {}
                    for rel in existing_rels:
                        if rel.relation_id not in rel_dict or rel.processed_time > rel_dict[rel.relation_id].processed_time:
                            rel_dict[rel.relation_id] = rel
                    for rel in rel_dict.values():
                        existing_relations_list.append({
                            "relation_id": rel.relation_id,
                            "content": rel.content
                        })

                # 获取上下文信息（优先使用当前实体的memory_cache，如果没有则使用候选实体的）
                context_text = None
                if entity.memory_cache_id:
                    context_text = self.storage.get_memory_cache_text(entity.memory_cache_id)
                if not context_text:
                    candidate_entity = self.storage.get_entity_by_id(cid)
                    if candidate_entity and candidate_entity.memory_cache_id:
                        context_text = self.storage.get_memory_cache_text(candidate_entity.memory_cache_id)
                
                if verbose:
                    wprint(f"      精细化判断: {entity.name} vs {candidate_info['name']}")
                    if existing_relations_list:
                        wprint(f"        已有 {len(existing_relations_list)} 个关系")
                
                # 调用精细化判断（传入上下文文本）
                detailed_result = self.llm_client.analyze_entity_pair_detailed(
                    current_entity_info,
                    candidate_info,
                    existing_relations_list,
                    context_text=context_text
                )
                
                action = detailed_result.get("action", "no_action")
                reason = detailed_result.get("reason", "")
                
                if verbose:
                    wprint(f"        判断结果: {action}")
                    wprint(f"        理由: {reason[:80]}..." if len(reason) > 80 else f"        理由: {reason}")
                
                if action == "merge":
                    merge_target = detailed_result.get("merge_target", "")
                    # 确定合并方向（版本多的作为target）
                    if not merge_target:
                        if current_entity_info["version_count"] >= candidate_info["version_count"]:
                            merge_target = entity.entity_id
                        else:
                            merge_target = cid
                    
                    merge_decisions.append({
                        "target_entity_id": merge_target,
                        "source_entity_id": cid if merge_target == entity.entity_id else entity.entity_id,
                        "source_name": candidate_info["name"],
                        "target_name": entity.name if merge_target == entity.entity_id else candidate_info["name"],
                        "reason": reason
                    })
                elif action == "create_relation":
                    relation_content = detailed_result.get("relation_content", "")
                    relation_decisions.append({
                        "entity1_id": entity.entity_id,
                        "entity2_id": cid,
                        "entity1_name": entity.name,
                        "entity2_name": candidate_info["name"],
                        "content": relation_content,
                        "reason": reason
                    })
            
            result["entities_analyzed"] += 1
            
            # 构建包含完整entity信息的text
            all_entities_info = [current_entity_info] + list(all_candidates_full_info.values())
            entity_list_text = self._build_entity_list_text(all_entities_info)
            all_analyzed_entities_text.append(f"\n\n{'='*80}\n分析实体: {entity.name} ({entity.entity_id})\n{'='*80}\n")
            all_analyzed_entities_text.append(entity_list_text)
            
            if verbose:
                wprint(f"\n    [精细化判断完成]")
                wprint(f"      确定需要合并: {len(merge_decisions)} 个")
                wprint(f"      确定需要创建关系: {len(relation_decisions)} 个")
            
            # ========== 阶段3: 执行操作（精细化判断全部完成后） ==========
            if verbose and (merge_decisions or relation_decisions):
                wprint(f"\n    [执行操作]...")
            
            final_target_id = None  # 用于后续创建关联关系时使用
            all_merged_in_this_round = set()  # 本次循环中被合并的实体ID
            
            # 转换为旧格式的merge_groups以复用后续代码
            merge_groups = []
            for md in merge_decisions:
                # 检查是否已有相同target的组
                found = False
                for mg in merge_groups:
                    if mg["target_entity_id"] == md["target_entity_id"]:
                        if md["source_entity_id"] not in mg["source_entity_ids"]:
                            mg["source_entity_ids"].append(md["source_entity_id"])
                            mg["reason"] += f"; {md['reason']}"
                        found = True
                        break
                if not found:
                    merge_groups.append({
                        "target_entity_id": md["target_entity_id"],
                        "source_entity_ids": [md["source_entity_id"]],
                        "reason": md["reason"]
                    })
            
            # 转换为旧格式的alias_relations
            alias_relations = relation_decisions
            
            # 构建entities_for_analysis（用于后续关系处理）
            entities_for_analysis = [current_entity_info] + list(all_candidates_full_info.values())
            
            if merge_groups:
                if verbose:
                    wprint(f"      执行合并操作...")
                
                # 收集所有需要合并的实体ID（包括target和source）
                all_merge_entity_ids = set()
                merge_reasons = []
                
                for merge_info in merge_groups:
                    target_id = merge_info.get("target_entity_id")
                    source_ids = merge_info.get("source_entity_ids", [])
                    reason = merge_info.get("reason", "")
                    
                    if not target_id or not source_ids:
                        continue
                    
                    # 检查是否已被合并
                    if any(sid in merged_entity_ids for sid in source_ids):
                        if verbose:
                            wprint(f"        跳过已合并的实体: {source_ids}")
                        continue
                    
                    # 不再过滤已有关系的实体对，让LLM判断是否应该合并
                    # 即使有关系，如果关系表示同一实体，也应该合并
                    
                    # 收集所有需要合并的实体
                    all_merge_entity_ids.add(target_id)
                    all_merge_entity_ids.update(source_ids)
                    if reason:
                        merge_reasons.append(reason)
                
                if all_merge_entity_ids:
                    # 确定最终的target：选择版本数最多的实体
                    target_candidates = []
                    for eid in all_merge_entity_ids:
                        version_count = self.storage.get_entity_version_count(eid)
                        target_candidates.append((eid, version_count))
                    
                    # 按版本数排序，选择最多的作为target
                    target_candidates.sort(key=lambda x: x[1], reverse=True)
                    final_target_id = target_candidates[0][0]
                    final_target_versions = target_candidates[0][1]
                    
                    # 其他实体都是source
                    final_source_ids = [eid for eid, _ in target_candidates[1:]]
                    
                    if final_source_ids:
                        # 获取实体名称用于显示
                        target_entity = self.storage.get_entity_by_id(final_target_id)
                        target_name = target_entity.name if target_entity else final_target_id
                        
                        # 合并所有原因
                        combined_reason = "；".join(merge_reasons) if merge_reasons else "多个实体需要合并"
                        
                        if verbose:
                            wprint(f"      合并多个实体到目标实体:")
                            wprint(f"        目标: {target_name} ({final_target_id}, 版本数: {final_target_versions})")
                            merge_names = [f"{self.storage.get_entity_by_id(sid).name} ({sid})" if self.storage.get_entity_by_id(sid) else sid for sid in final_source_ids]
                            wprint(f"        源实体: {', '.join(merge_names)}")
                            wprint(f"        原因: {combined_reason}")
                        
                        # 执行合并（一次性合并所有source到target）
                        merge_result = self.storage.merge_entity_ids(final_target_id, final_source_ids)
                        merge_result["reason"] = combined_reason
                        merge_result["target_versions"] = final_target_versions
                        
                        if verbose:
                            wprint(f"        结果: 更新了 {merge_result.get('entities_updated', 0)} 条实体记录")
                        
                        # 处理合并后产生的自指向关系
                        self._handle_self_referential_relations_after_merge(final_target_id, verbose)
                        
                        # 记录已合并的entity_id
                        for sid in final_source_ids:
                            merged_entity_ids.add(sid)
                            all_merged_in_this_round.add(sid)
                        
                        result["merge_details"].append(merge_result)
                        result["entities_merged"] += merge_result.get("entities_updated", 0)
                
                # 立即创建关联关系（步骤4）
                if alias_relations:
                    if verbose:
                        wprint(f"      创建关联关系...")
                        if self.llm_threads > 1 and len(alias_relations) > 1:
                            wprint(f"      使用 {self.llm_threads} 个线程并行处理 {len(alias_relations)} 个关系...")
                    
                    # 构建有效的entity_id映射（用于验证LLM返回的ID是否有效）
                    valid_entity_ids = {e["entity_id"] for e in entities_for_analysis}
                    entity_id_entity2_name = {e["entity_id"]: e["name"] for e in entities_for_analysis}
                    
                    # 准备所有需要处理的关系信息
                    relations_to_process = []
                    
                    for alias_info in alias_relations:
                        entity1_id = alias_info.get("entity1_id")
                        entity2_id = alias_info.get("entity2_id")
                        entity1_name = alias_info.get("entity1_name", "")
                        entity2_name = alias_info.get("entity2_name", "")
                        # 注意：现在alias_info中不再包含content，需要在后续步骤中生成
                        
                        if verbose:
                            wprint(f"        处理关系: {entity1_name} ({entity1_id}) -> {entity2_name} ({entity2_id})")
                        
                        if not entity1_id or not entity2_id:
                            if verbose:
                                wprint(f"          跳过：缺少entity_id (entity1: {entity1_id}, entity2: {entity2_id})")
                            continue
                        
                        # 验证entity_id是否在传入的实体列表中
                        if entity1_id not in valid_entity_ids:
                            if verbose:
                                wprint(f"          警告：entity1_id {entity1_id} 不在分析列表中，尝试通过名称查找...")
                            # 尝试通过名称查找实体
                            found_entity = None
                            for e in entities_for_analysis:
                                if e["name"] == entity1_name:
                                    found_entity = e
                                    break
                            if found_entity:
                                entity1_id = found_entity["entity_id"]
                                if verbose:
                                    wprint(f"            通过名称找到实体: {entity1_name} -> {entity1_id}")
                            else:
                                if verbose:
                                    wprint(f"            无法找到实体: {entity1_name} ({entity1_id})")
                                continue
                        
                        if entity2_id not in valid_entity_ids:
                            if verbose:
                                wprint(f"          警告：entity2_id {entity2_id} 不在分析列表中，尝试通过名称查找...")
                            # 尝试通过名称查找实体
                            found_entity = None
                            for e in entities_for_analysis:
                                if e["name"] == entity2_name:
                                    found_entity = e
                                    break
                            if found_entity:
                                entity2_id = found_entity["entity_id"]
                                if verbose:
                                    wprint(f"            通过名称找到实体: {entity2_name} -> {entity2_id}")
                            else:
                                if verbose:
                                    wprint(f"            无法找到实体: {entity2_name} ({entity2_id})")
                                continue
                        
                        # 检查实体是否在本次循环中被合并（如果被合并，需要使用合并后的entity_id）
                        actual_entity1_id = entity1_id
                        actual_entity2_id = entity2_id
                        
                        # 如果entity1实体在本次循环中被合并，使用最终的target_id
                        if entity1_id in all_merged_in_this_round and final_target_id:
                            actual_entity1_id = final_target_id
                        
                        # 如果entity2实体在本次循环中被合并，使用最终的target_id
                        if entity2_id in all_merged_in_this_round and final_target_id:
                            actual_entity2_id = final_target_id
                        
                        # 检查实体是否在之前的循环中已被合并
                        # 如果entity_id在merged_entity_ids中，说明已经被合并，需要找到合并后的entity_id
                        if entity1_id in merged_entity_ids:
                            # 从merge_details中查找该实体被合并到哪个target
                            found_target = None
                            for merge_detail in result["merge_details"]:
                                if entity1_id in merge_detail.get("merged_source_ids", []):
                                    found_target = merge_detail.get("target_entity_id")
                                    break
                            if found_target:
                                actual_entity1_id = found_target
                                if verbose:
                                    wprint(f"            注意：entity1实体 {entity1_name} ({entity1_id}) 已被合并到 {found_target}")
                            else:
                                # 如果找不到，尝试查询数据库（可能entity_id已经更新）
                                entity1_db = self.storage.get_entity_by_id(entity1_id)
                                if entity1_db:
                                    actual_entity1_id = entity1_db.entity_id
                        
                        if entity2_id in merged_entity_ids:
                            # 从merge_details中查找该实体被合并到哪个target
                            found_target = None
                            for merge_detail in result["merge_details"]:
                                if entity2_id in merge_detail.get("merged_source_ids", []):
                                    found_target = merge_detail.get("target_entity_id")
                                    break
                            if found_target:
                                actual_entity2_id = found_target
                                if verbose:
                                    wprint(f"            注意：entity2实体 {entity2_name} ({entity2_id}) 已被合并到 {found_target}")
                            else:
                                # 如果找不到，尝试查询数据库（可能entity_id已经更新）
                                entity2_db = self.storage.get_entity_by_id(entity2_id)
                                if entity2_db:
                                    actual_entity2_id = entity2_db.entity_id
                        
                        # 验证最终的entity_id是否有效
                        entity1_check = self.storage.get_entity_by_id(actual_entity1_id)
                        entity2_check = self.storage.get_entity_by_id(actual_entity2_id)
                        
                        if not entity1_check:
                            if verbose:
                                wprint(f"          错误：无法找到entity1实体 (entity_id: {actual_entity1_id}, name: {entity1_name})")
                            continue
                        
                        if not entity2_check:
                            if verbose:
                                wprint(f"          错误：无法找到entity2实体 (entity_id: {actual_entity2_id}, name: {entity2_name})")
                            continue
                        
                        # 如果合并后entity1和entity2是同一个实体，跳过创建关系
                        if actual_entity1_id == actual_entity2_id:
                            if verbose:
                                wprint(f"          跳过：合并后entity1和entity2是同一实体")
                            continue
                        
                        if verbose:
                            wprint(f"          准备处理关系: {entity1_name} -> {entity2_name}")
                            if actual_entity1_id != entity1_id or actual_entity2_id != entity2_id:
                                wprint(f"            注意：使用了合并后的entity_id (entity1: {entity1_id}->{actual_entity1_id}, entity2: {entity2_id}->{actual_entity2_id})")
                        
                        # 收集关系信息，准备并行处理
                        # 从alias_info中获取初步的content（如果存在）
                        preliminary_content = alias_info.get("content")
                        relations_to_process.append({
                            "entity1_id": entity1_id,
                            "entity2_id": entity2_id,
                            "actual_entity1_id": actual_entity1_id,
                            "actual_entity2_id": actual_entity2_id,
                            "entity1_name": entity1_name,
                            "entity2_name": entity2_name,
                            "content": preliminary_content  # 初步的content，用于预判断
                        })
                
                # 并行处理关系
                if verbose:
                    wprint(f"      准备处理 {len(relations_to_process)} 个关系，llm_threads={self.llm_threads}")
                if self.llm_threads > 1 and len(relations_to_process) > 1:
                    # 使用多线程并行处理
                    if verbose:
                        wprint(f"      使用 {self.llm_threads} 个线程并行处理 {len(relations_to_process)} 个关系...")
                    with ThreadPoolExecutor(max_workers=self.llm_threads, thread_name_prefix="tmg-llm") as executor:
                        # 提交所有任务（多线程模式下不显示每个关系的详细信息）
                        future_to_relation = {
                            executor.submit(
                                self._process_single_alias_relation,
                                rel_info,
                                False  # 多线程模式下不显示详细信息
                            ): rel_info
                            for rel_info in relations_to_process
                        }
                        
                        # 收集结果
                        for future in as_completed(future_to_relation):
                            rel_info = future_to_relation[future]
                            try:
                                result_data = future.result()
                                if result_data:
                                    # 更新统计信息
                                    if result_data.get("is_new"):
                                        result["alias_relations_created"] += 1
                                    elif result_data.get("is_updated"):
                                        result["alias_relations_updated"] += 1
                                    result["alias_details"].append(result_data)
                            except Exception as e:
                                if verbose:
                                    wprint(f"      处理关系 {rel_info['entity1_name']} -> {rel_info['entity2_name']} 失败: {e}")
                else:
                    # 串行处理
                    if verbose:
                        if self.llm_threads <= 1:
                            wprint(f"      串行处理 {len(relations_to_process)} 个关系（llm_threads={self.llm_threads}，未启用多线程）")
                        elif len(relations_to_process) <= 1:
                            wprint(f"      串行处理 {len(relations_to_process)} 个关系（关系数量 <= 1，无需并行）")
                    for rel_info in relations_to_process:
                        try:
                            result_data = self._process_single_alias_relation(rel_info, verbose)
                            if result_data:
                                # 更新统计信息
                                if result_data.get("is_new"):
                                    result["alias_relations_created"] += 1
                                elif result_data.get("is_updated"):
                                    result["alias_relations_updated"] += 1
                                result["alias_details"].append(result_data)
                        except Exception as e:
                            if verbose:
                                wprint(f"      处理关系 {rel_info['entity1_name']} -> {rel_info['entity2_name']} 失败: {e}")
    
    def _consolidate_knowledge_graph_parallel(self, verbose: bool = True, 
                                              similarity_threshold: float = 0.6,
                                              max_candidates: int = 5,
                                              batch_candidates: Optional[int] = None,
                                              content_snippet_length: int = 64) -> dict:
        """
        多线程并行版本的知识图谱整理
        
        通过预排除关联实体来避免并行处理时的冲突：
        1. 预先搜索所有实体的关联实体
        2. 调度器选择不冲突的实体并行处理
        3. 线程完成后，释放锁定的实体，更新合并状态
        
        Args:
            verbose: 是否输出详细信息
            similarity_threshold: 相似度搜索阈值
            max_candidates: 每次搜索返回的最大候选实体数
            content_snippet_length: 传入LLM的实体content最大长度
        
        Returns:
            整理结果统计
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
        from queue import Queue
        
        if verbose:
            wprint("=" * 60)
            wprint(f"开始知识图谱整理（多线程模式，{self.llm_threads}个线程）...")
            wprint("=" * 60)
        
        # 结果统计（线程安全）
        result = {
            "entities_analyzed": 0,
            "entities_merged": 0,
            "alias_relations_created": 0,
            "alias_relations_updated": 0,
            "merge_details": [],
            "alias_details": []
        }
        result_lock = threading.Lock()
        
        # 步骤1：获取所有实体
        if verbose:
            wprint(f"\n步骤1: 获取所有实体...")
        
        all_entities = self.storage.get_all_entities()
        
        if not all_entities:
            if verbose:
                wprint("  知识库中没有实体。")
            return result
        
        # 按版本数量从大到小排序
        entity_ids = [entity.entity_id for entity in all_entities]
        version_counts = self.storage.get_entity_version_counts(entity_ids)
        all_entities.sort(key=lambda e: version_counts.get(e.entity_id, 0), reverse=True)
        
        initial_entity_count = len(all_entities)
        if verbose:
            wprint(f"  整理前共有 {initial_entity_count} 个实体")
        
        # 步骤1.5：先按名称完全匹配进行整理
        if verbose:
            wprint(f"\n步骤1.5: 按名称完全匹配进行初步整理...")
        
        # 记录已合并的实体ID（用于后续embedding搜索时排除）
        merged_entity_ids = set()
        # 记录合并映射：source_entity_id -> target_entity_id
        merge_mapping = {}
        
        # 构建名称到实体列表的映射
        name_to_entities = {}
        for entity in all_entities:
            name = entity.name
            if name not in name_to_entities:
                name_to_entities[name] = []
            name_to_entities[name].append(entity)
        
        # 对每个名称组内的实体按版本数排序（从大到小）
        for name in name_to_entities:
            name_to_entities[name].sort(
                key=lambda e: version_counts.get(e.entity_id, 0), 
                reverse=True
            )
        
        # 按照每个名称组中实体的最大版本数排序（从大到小），然后按顺序处理
        name_groups_sorted = sorted(
            name_to_entities.items(),
            key=lambda item: max(
                (version_counts.get(e.entity_id, 0) for e in item[1]),
                default=0
            ),
            reverse=True
        )
        
        # 处理名称完全一致的实体组
        name_match_count = 0
        for name, entities_with_same_name in name_groups_sorted:
            # 只处理有多个实体的名称组
            if len(entities_with_same_name) <= 1:
                continue
            
            name_match_count += 1
            if verbose:
                wprint(f"  发现名称完全一致的实体组: {name} (共 {len(entities_with_same_name)} 个实体)")
            
            # 准备实体信息用于LLM判断
            entities_info = []
            for entity in entities_with_same_name:
                # 跳过已合并的实体
                if entity.entity_id in merged_entity_ids:
                    continue
                
                version_count = version_counts.get(entity.entity_id, 0)
                entities_info.append({
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "content": entity.content,
                    "version_count": version_count
                })
            
            # 如果过滤后只剩一个或没有实体，跳过
            if len(entities_info) <= 1:
                continue
            
            # 获取记忆上下文
            memory_contexts = {}
            for entity in entities_with_same_name:
                if entity.entity_id in merged_entity_ids:
                    continue
                cache_text = self.storage.get_memory_cache_text(entity.memory_cache_id)
                if cache_text:
                    memory_contexts[entity.entity_id] = cache_text
            
            # 检查实体对之间是否已有关系，如果关系表示同一实体则直接合并
            entity_ids_for_check = [info['entity_id'] for info in entities_info]
            existing_relations_between = self._check_and_merge_entities_from_relations(
                entity_ids_for_check,
                entities_info,
                version_counts,
                merged_entity_ids,
                merge_mapping,
                result,
                verbose
            )
            
            if verbose and existing_relations_between:
                wprint(f"    发现 {len(existing_relations_between)} 对实体之间已有关系（非同一实体关系），这些实体对不会被合并")
            
            # 调用LLM分析：判断是合并还是关联关系
            analysis_result = self.llm_client.analyze_entity_duplicates(
                entities_info,
                memory_contexts,
                content_snippet_length=content_snippet_length,
                existing_relations_between_entities=existing_relations_between
            )
            
            if "error" in analysis_result:
                if verbose:
                    wprint(f"    分析失败，跳过该组")
                continue
            
            # 处理合并（过滤掉已有关系的实体对）
            merge_groups = analysis_result.get("merge_groups", [])
            for merge_group in merge_groups:
                target_entity_id = merge_group.get("target_entity_id")
                source_entity_ids = merge_group.get("source_entity_ids", [])
                reason = merge_group.get("reason", "")
                
                if not target_entity_id or not source_entity_ids:
                    continue
                
                # 检查是否已被合并
                if any(sid in merged_entity_ids for sid in source_entity_ids):
                    continue
                
                # 不再过滤已有关系的实体对，让LLM判断是否应该合并
                # 即使有关系，如果关系表示同一实体，也应该合并
                
                # 执行合并
                merge_result = self.storage.merge_entity_ids(target_entity_id, source_entity_ids)
                merge_result["reason"] = reason
                
                if verbose:
                    target_name = next((e.name for e in entities_with_same_name if e.entity_id == target_entity_id), target_entity_id)
                    wprint(f"    合并实体: {target_name} ({target_entity_id}) <- {len(source_entity_ids)} 个源实体")
                    wprint(f"      原因: {reason}")
                
                # 处理合并后产生的自指向关系
                self._handle_self_referential_relations_after_merge(target_entity_id, verbose)
                
                # 记录已合并的实体和合并映射
                for sid in source_entity_ids:
                    merged_entity_ids.add(sid)
                    merge_mapping[sid] = target_entity_id
                
                # 更新结果统计
                result["merge_details"].append(merge_result)
                result["entities_merged"] += merge_result.get("entities_updated", 0)
            
            # 处理关系（别名关系）
            alias_relations = analysis_result.get("alias_relations", [])
            for alias_info in alias_relations:
                entity1_id = alias_info.get("entity1_id")
                entity2_id = alias_info.get("entity2_id")
                entity1_name = alias_info.get("entity1_name", "")
                entity2_name = alias_info.get("entity2_name", "")
                preliminary_content = alias_info.get("content")
                
                if not entity1_id or not entity2_id:
                    continue
                
                # 检查是否已被合并（如果已合并，需要找到合并后的实际ID）
                actual_entity1_id = merge_mapping.get(entity1_id, entity1_id)
                actual_entity2_id = merge_mapping.get(entity2_id, entity2_id)
                
                # 如果实体已被合并，跳过（因为合并后的实体可能不在当前名称组中）
                if entity1_id in merged_entity_ids or entity2_id in merged_entity_ids:
                    if verbose:
                        wprint(f"    跳过关系（实体已合并）: {entity1_name} -> {entity2_name}")
                    continue
                
                # 处理关系
                rel_info = {
                    "entity1_id": entity1_id,
                    "entity2_id": entity2_id,
                    "actual_entity1_id": actual_entity1_id,
                    "actual_entity2_id": actual_entity2_id,
                    "entity1_name": entity1_name,
                    "entity2_name": entity2_name,
                    "content": preliminary_content
                }
                
                rel_result = self._process_single_alias_relation(rel_info, verbose=False)
                if rel_result:
                    result["alias_details"].append(rel_result)
                    if rel_result.get("is_new"):
                        result["alias_relations_created"] += 1
                    elif rel_result.get("is_updated"):
                        result["alias_relations_updated"] += 1
        
        if verbose:
            wprint(f"  名称匹配完成，处理了 {name_match_count} 个名称组，合并了 {len(merged_entity_ids)} 个实体")
        
        # 步骤2：使用混合检索方式一次性找到所有实体的关联实体
        if verbose:
            wprint(f"\n步骤2: 使用混合检索方式预搜索所有实体的关联实体...")
            wprint(f"  使用多种检索模式：name_only(embedding) + name_and_content(embedding) + name_only(text/jaccard)")
        
        # 使用混合检索方式一次性找到所有实体的关联实体
        entity_to_candidates = self.storage.find_related_entities_by_embedding(
            similarity_threshold=similarity_threshold,
            max_candidates=max_candidates,
            use_mixed_search=True,  # 启用混合检索
            content_snippet_length=content_snippet_length
        )
        
        # 过滤掉已合并的实体（在候选列表中排除）
        for entity_id in list(entity_to_candidates.keys()):
            # 如果当前实体已合并，从候选列表中移除
            if entity_id in merged_entity_ids:
                del entity_to_candidates[entity_id]
                continue
            
            # 从候选列表中排除已合并的实体
            candidates = entity_to_candidates[entity_id]
            entity_to_candidates[entity_id] = candidates - merged_entity_ids
        
        if verbose:
            total_candidates = sum(len(candidates) for candidates in entity_to_candidates.values())
            wprint(f"  预搜索完成，共 {len(entity_to_candidates)} 个实体，找到 {total_candidates} 个关联实体（已排除 {len(merged_entity_ids)} 个已合并实体）")
        
        # 步骤3：并行处理实体
        if verbose:
            wprint(f"\n步骤3: 并行处理实体（{self.llm_threads}个线程）...")
        
        # 共享状态（需要加锁）
        # merged_entity_ids 已经在步骤1.5中初始化，这里只需要创建锁
        merged_ids_lock = threading.Lock()
        
        in_progress_ids = set()  # 正在处理中的实体ID（包括关联实体）
        in_progress_lock = threading.Lock()
        
        processed_pairs = set()
        processed_pairs_lock = threading.Lock()
        
        # 待处理实体列表
        pending_entities = list(all_entities)
        pending_lock = threading.Lock()
        
        # 用于累积所有分析过的实体信息
        all_analyzed_entities_text = []
        analyzed_text_lock = threading.Lock()
        
        # 计数器
        processed_count = [0]  # 使用列表以便在闭包中修改
        count_lock = threading.Lock()
        
        def get_next_entity():
            """
            获取下一个可以处理的实体（不与正在处理的实体冲突）
            返回: (entity, candidate_ids) 或 (None, None)
            """
            with pending_lock:
                for i, entity in enumerate(pending_entities):
                    # 检查是否已合并
                    with merged_ids_lock:
                        if entity.entity_id in merged_entity_ids:
                            pending_entities.pop(i)
                            continue
                    
                    # 获取关联实体
                    candidates = entity_to_candidates.get(entity.entity_id, set())
                    
                    # 过滤掉已合并的关联实体
                    with merged_ids_lock:
                        candidates = candidates - merged_entity_ids
                    
                    # 检查是否与正在处理的实体冲突
                    all_ids = {entity.entity_id} | candidates
                    with in_progress_lock:
                        if all_ids & in_progress_ids:
                            continue  # 有冲突，跳过
                        
                        # 标记为正在处理
                        in_progress_ids.update(all_ids)
                    
                    # 找到了可以处理的实体
                    pending_entities.pop(i)
                    return entity, candidates
            
            return None, None
        
        def release_entity(entity_id, candidate_ids):
            """释放实体的处理权"""
            all_ids = {entity_id} | candidate_ids
            with in_progress_lock:
                in_progress_ids.difference_update(all_ids)
        
        def process_entity_task(entity, candidate_ids):
            """
            处理单个实体及其关联实体
            返回处理结果
            """
            task_result = {
                "entities_analyzed": 0,
                "entities_merged": 0,
                "alias_relations_created": 0,
                "alias_relations_updated": 0,
                "merge_details": [],
                "alias_details": [],
                "merged_ids": set(),
                "analyzed_text": ""
            }
            
            try:
                # 过滤已处理的配对
                with processed_pairs_lock:
                    filtered_candidates = {
                        cid for cid in candidate_ids
                        if (min(entity.entity_id, cid), max(entity.entity_id, cid)) not in processed_pairs
                    }
                    # 记录配对
                    for cid in filtered_candidates:
                        processed_pairs.add((min(entity.entity_id, cid), max(entity.entity_id, cid)))
                
                if not filtered_candidates:
                    return task_result
                
                # 获取候选实体的完整信息
                candidates_info = []
                for cid in filtered_candidates:
                    candidate_entity = self.storage.get_entity_by_id(cid)
                    if candidate_entity:
                        version_count = self.storage.get_entity_version_count(cid)
                        candidates_info.append({
                            "entity_id": cid,
                            "name": candidate_entity.name,
                            "content": candidate_entity.content,
                            "version_count": version_count
                        })
                
                if not candidates_info:
                    return task_result
                
                # 准备当前实体信息
                current_version_count = self.storage.get_entity_version_count(entity.entity_id)
                current_entity_info = {
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "content": entity.content,
                    "version_count": current_version_count
                }
                
                entities_for_analysis = [current_entity_info] + candidates_info
                
                # 获取记忆上下文
                memory_contexts = {}
                cache_text = self.storage.get_memory_cache_text(entity.memory_cache_id)
                if cache_text:
                    memory_contexts[entity.entity_id] = cache_text
                
                for info in candidates_info:
                    candidate_entity = self.storage.get_entity_by_id(info["entity_id"])
                    if candidate_entity:
                        c_text = self.storage.get_memory_cache_text(candidate_entity.memory_cache_id)
                        if c_text:
                            memory_contexts[info["entity_id"]] = c_text
                
                # 检查实体对之间是否已有关系，如果关系表示同一实体则直接合并
                analysis_entity_ids = [info['entity_id'] for info in entities_for_analysis]
                existing_relations_between = self._check_and_merge_entities_from_relations(
                    analysis_entity_ids,
                    entities_for_analysis,
                    version_counts,
                    merged_entity_ids,
                    merge_mapping,
                    result,
                    verbose
                )
                
                # 调用LLM分析
                analysis_result = self.llm_client.analyze_entity_duplicates(
                    entities_for_analysis,
                    memory_contexts,
                    content_snippet_length=content_snippet_length,
                    existing_relations_between_entities=existing_relations_between
                )
                
                if "error" in analysis_result:
                    return task_result
                
                task_result["entities_analyzed"] = 1
                task_result["analyzed_text"] = self._build_entity_list_text(entities_for_analysis)
                
                # 处理合并（过滤掉已有关系的实体对）
                merge_groups = analysis_result.get("merge_groups", [])
                alias_relations = analysis_result.get("alias_relations", [])
                
                # 执行合并操作
                for merge_group in merge_groups:
                    target_entity_id = merge_group.get("target_entity_id")
                    source_entity_ids = merge_group.get("source_entity_ids", [])
                    reason = merge_group.get("reason", "")
                    
                    if not target_entity_id or not source_entity_ids:
                        continue
                    
                    # 检查是否已被合并
                    with merged_ids_lock:
                        if any(sid in merged_entity_ids for sid in source_entity_ids):
                            continue
                    
                    # 不再过滤已有关系的实体对，让LLM判断是否应该合并
                    # 即使有关系，如果关系表示同一实体，也应该合并
                    
                    # 执行合并
                    merge_result = self.storage.merge_entity_ids(target_entity_id, source_entity_ids)
                    merge_result["reason"] = reason
                    
                    # 处理合并后产生的自指向关系
                    self._handle_self_referential_relations_after_merge(target_entity_id, verbose=False)
                    
                    task_result["merge_details"].append(merge_result)
                    task_result["entities_merged"] += merge_result.get("entities_updated", 0)
                    
                    # 记录已合并的实体
                    for sid in source_entity_ids:
                        task_result["merged_ids"].add(sid)
                
                # 处理关系（简化版，只记录需要创建的关系，后续统一处理）
                for alias_info in alias_relations:
                    entity1_id = alias_info.get("entity1_id")
                    entity2_id = alias_info.get("entity2_id")
                    entity1_name = alias_info.get("entity1_name", "")
                    entity2_name = alias_info.get("entity2_name", "")
                    preliminary_content = alias_info.get("content")
                    
                    if not entity1_id or not entity2_id:
                        continue
                    
                    # 处理关系
                    rel_info = {
                        "entity1_id": entity1_id,
                        "entity2_id": entity2_id,
                        "actual_entity1_id": entity1_id,
                        "actual_entity2_id": entity2_id,
                        "entity1_name": entity1_name,
                        "entity2_name": entity2_name,
                        "content": preliminary_content
                    }
                    
                    rel_result = self._process_single_alias_relation(rel_info, verbose=False)
                    if rel_result:
                        task_result["alias_details"].append(rel_result)
                        if rel_result.get("is_new"):
                            task_result["alias_relations_created"] += 1
                        elif rel_result.get("is_updated"):
                            task_result["alias_relations_updated"] += 1
                
                return task_result
                
            except Exception as e:
                if verbose:
                    wprint(f"    处理实体 {entity.name} 失败: {e}")
                import traceback
                traceback.print_exc()
                return task_result
        
        # 主调度循环
        with ThreadPoolExecutor(max_workers=self.llm_threads, thread_name_prefix="tmg-llm") as executor:
            futures = {}
            
            while True:
                # 尝试提交新任务（直到达到线程数或没有可用实体）
                while len(futures) < self.llm_threads:
                    entity, candidates = get_next_entity()
                    if entity is None:
                        break
                    
                    future = executor.submit(process_entity_task, entity, candidates)
                    futures[future] = (entity, candidates)
                    
                    with count_lock:
                        processed_count[0] += 1
                        if verbose:
                            wprint(f"\n  [{processed_count[0]}/{initial_entity_count}] 开始处理: {entity.name}")
                
                # 如果没有正在运行的任务且没有待处理的实体，退出
                if not futures:
                    with pending_lock:
                        if not pending_entities:
                            break
                        # 还有待处理的实体但都在冲突中，等待一下
                    import time
                    time.sleep(0.1)
                    continue
                
                # 等待一个任务完成
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                
                for future in done:
                    entity, candidates = futures.pop(future)
                    
                    try:
                        task_result = future.result()
                        
                        # 更新全局结果（加锁）
                        with result_lock:
                            result["entities_analyzed"] += task_result["entities_analyzed"]
                            result["entities_merged"] += task_result["entities_merged"]
                            result["alias_relations_created"] += task_result["alias_relations_created"]
                            result["alias_relations_updated"] += task_result["alias_relations_updated"]
                            result["merge_details"].extend(task_result["merge_details"])
                            result["alias_details"].extend(task_result["alias_details"])
                        
                        # 更新合并状态
                        with merged_ids_lock:
                            merged_entity_ids.update(task_result["merged_ids"])
                        
                        # 累积分析文本
                        if task_result["analyzed_text"]:
                            with analyzed_text_lock:
                                all_analyzed_entities_text.append(
                                    f"\n\n{'='*80}\n分析实体: {entity.name}\n{'='*80}\n"
                                )
                                all_analyzed_entities_text.append(task_result["analyzed_text"])
                        
                        if verbose and task_result["entities_analyzed"] > 0:
                            wprint(f"    完成: {entity.name} "
                                  f"(合并: {task_result['entities_merged']}, "
                                  f"新建关系: {task_result['alias_relations_created']}, "
                                  f"更新关系: {task_result['alias_relations_updated']})")
                    
                    finally:
                        # 释放处理权
                        release_entity(entity.entity_id, candidates)
        
        # 调用收尾工作
        self._finalize_consolidation(result, all_analyzed_entities_text, verbose)
        
        # 获取整理后的实体总数
        final_entities = self.storage.get_all_entities()
        final_entity_count = len(final_entities) if final_entities else 0
        
        # 输出最终统计总结
        if verbose:
            wprint("\n" + "=" * 60)
            wprint("知识图谱整理完成！（多线程模式）")
            wprint("=" * 60)
            wprint(f"📊 实体统计:")
            wprint(f"  - 整理前实体数: {initial_entity_count}")
            wprint(f"  - 整理后实体数: {final_entity_count}")
            wprint(f"  - 减少的实体数: {initial_entity_count - final_entity_count}")
            wprint(f"")
            wprint(f"📈 整理操作统计:")
            wprint(f"  - 分析的实体数: {result['entities_analyzed']}")
            wprint(f"  - 合并的实体记录数: {result['entities_merged']}")
            wprint(f"")
            wprint(f"🔗 关系边统计:")
            wprint(f"  - 新建的关系边数: {result['alias_relations_created']}")
            wprint(f"  - 更新的关系边数: {result['alias_relations_updated']}")
            wprint(f"  - 总处理的关系边数: {result['alias_relations_created'] + result['alias_relations_updated']}")
            wprint("=" * 60)
        
        return result
    
    def _get_existing_relations_between_entities(self, entity_ids: List[str]) -> Dict[str, List[Dict]]:
        """
        检查一组实体之间两两是否存在已有关系
        
        Args:
            entity_ids: 实体ID列表
        
        Returns:
            已有关系字典，key为 "entity1_id|entity2_id" 格式（按字母序排序），
            value为该实体对之间的关系列表，每个关系包含:
                - relation_id: 关系ID
                - content: 关系描述
        """
        existing_relations = {}
        
        # 遍历所有实体对
        for i, entity1_id in enumerate(entity_ids):
            for entity2_id in entity_ids[i+1:]:
                # 检查两个实体之间是否存在关系
                relations = self.storage.get_relations_by_entities(entity1_id, entity2_id)
                
                if relations:
                    # 按字母序排序实体ID作为key
                    sorted_ids = sorted([entity1_id, entity2_id])
                    pair_key = f"{sorted_ids[0]}|{sorted_ids[1]}"
                    
                    # 按relation_id分组，每个relation_id只保留最新版本
                    relation_dict = {}
                    for rel in relations:
                        if rel.relation_id not in relation_dict:
                            relation_dict[rel.relation_id] = rel
                        else:
                            if rel.processed_time > relation_dict[rel.relation_id].processed_time:
                                relation_dict[rel.relation_id] = rel

                    # 提取关系信息
                    existing_relations[pair_key] = [
                        {
                            'relation_id': r.relation_id,
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
        
        content_lower = relation_content.lower()
        
        # 关键词列表：表示同一实体的关系描述
        same_entity_keywords = [
            "同一实体", "同一个", "同一人", "同一物", "同一对象",
            "别名", "别称", "又称", "也叫", "亦称",
            "是", "就是", "即是", "等于", "等同于",
            "指", "指的是", "指向", "表示",
            "简称", "全称", "昵称", "绰号", "外号",
            "本名", "原名", "真名", "实名"
        ]
        
        # 检查是否包含关键词
        for keyword in same_entity_keywords:
            if keyword in content_lower:
                return True
        
        return False
    
    def _check_and_merge_entities_from_relations(self, entity_ids: List[str], 
                                                  entities_info: List[Dict],
                                                  version_counts: Dict[str, int],
                                                  merged_entity_ids: set,
                                                  merge_mapping: Dict[str, str],
                                                  result: Dict,
                                                  verbose: bool = True) -> Dict[str, List[Dict]]:
        """
        检查实体之间的关系，如果关系表示同一实体，则直接合并
        
        Args:
            entity_ids: 实体ID列表
            entities_info: 实体信息列表（包含name等）
            version_counts: 实体版本数统计
            merged_entity_ids: 已合并的实体ID集合
            merge_mapping: 合并映射字典
            result: 结果统计字典
            verbose: 是否输出详细信息
        
        Returns:
            过滤后的已有关系字典（已排除表示同一实体的关系）
        """
        existing_relations_between = self._get_existing_relations_between_entities(entity_ids)
        
        # 检查是否有关系表示同一实体，如果有则直接合并
        entities_to_merge_from_relations = []
        
        for pair_key, relations in existing_relations_between.items():
            entity_ids_pair = pair_key.split("|")
            if len(entity_ids_pair) != 2:
                continue
            
            entity1_id, entity2_id = entity_ids_pair
            
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
                    if source_id not in merged_entity_ids and target_id not in merged_entity_ids:
                        entities_to_merge_from_relations.append({
                            'target_id': target_id,
                            'source_id': source_id,
                            'relation_id': rel['relation_id'],
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
                merge_result = self.storage.merge_entity_ids(target_id, [source_id])
                merge_result["reason"] = f"关系表示同一实体: {relation_content}"
                
                if verbose:
                    target_name = next((e.get('name', '') for e in entities_info if e.get('entity_id') == target_id), target_id)
                    source_name = next((e.get('name', '') for e in entities_info if e.get('entity_id') == source_id), source_id)
                    wprint(f"      合并实体（基于关系）: {target_name} ({target_id}) <- {source_name} ({source_id})")
                    wprint(f"        原因: {relation_content}")
                
                # 处理合并后产生的自指向关系
                self._handle_self_referential_relations_after_merge(target_id, verbose)
                
                # 记录已合并的实体和合并映射
                merged_entity_ids.add(source_id)
                merge_mapping[source_id] = target_id
                
                # 更新结果统计
                result["merge_details"].append(merge_result)
                result["entities_merged"] += merge_result.get("entities_updated", 0)
        
        # 过滤掉已通过关系合并的实体对，只保留非同一实体的关系
        filtered_existing_relations = {}
        for pair_key, relations in existing_relations_between.items():
            entity_ids_pair = pair_key.split("|")
            if len(entity_ids_pair) != 2:
                continue
            
            entity1_id, entity2_id = entity_ids_pair
            
            # 如果这对实体已经通过关系合并了，跳过
            if (entity1_id in merged_entity_ids and merge_mapping.get(entity1_id) == entity2_id) or \
               (entity2_id in merged_entity_ids and merge_mapping.get(entity2_id) == entity1_id):
                continue
            
            # 过滤掉表示同一实体的关系
            filtered_relations = [
                rel for rel in relations 
                if not self._is_relation_indicating_same_entity(rel['content'])
            ]
            
            if filtered_relations:
                filtered_existing_relations[pair_key] = filtered_relations
        
        return filtered_existing_relations
    
    def _handle_self_referential_relations_after_merge(self, target_entity_id: str, verbose: bool = True) -> int:
        """
        处理合并后产生的自指向关系
        
        合并操作会将源实体的entity_id更新为目标实体的entity_id，这可能导致原本不是自指向的关系变成自指向关系。
        例如：实体A(ent_001)和实体B(ent_002)之间有关系，合并后B的entity_id变为ent_001，这个关系就变成了自指向关系。
        
        此方法会：
        1. 检查目标实体是否有自指向关系
        2. 如果有，将这些关系的内容总结到实体的content中
        3. 删除这些自指向关系
        
        Args:
            target_entity_id: 合并后的目标实体ID
            verbose: 是否输出详细信息
        
        Returns:
            处理的自指向关系数量
        """
        # 检查是否有自指向关系
        self_ref_relations = self.storage.get_self_referential_relations_for_entity(target_entity_id)
        
        if not self_ref_relations:
            return 0
        
        if verbose:
            wprint(f"        检测到合并后产生 {len(self_ref_relations)} 个自指向关系，正在处理...")
        
        # 获取实体的最新版本
        entity = self.storage.get_entity_by_id(target_entity_id)
        if not entity:
            if verbose:
                wprint(f"        警告：无法获取实体 {target_entity_id}")
            return 0
        
        # 收集所有自指向关系的content
        self_ref_contents = [rel['content'] for rel in self_ref_relations if rel.get('content')]
        
        if self_ref_contents:
            # 用LLM总结这些关系内容到实体的content中
            summarized_content = self.llm_client.merge_entity_content(
                old_content=entity.content,
                new_content="\n\n".join([f"属性信息：{content}" for content in self_ref_contents])
            )
            
            # 创建实体的新版本
            new_entity_id = f"entity_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            new_entity = Entity(
                absolute_id=new_entity_id,
                entity_id=entity.entity_id,
                name=entity.name,
                content=summarized_content,
                event_time=datetime.now(),
                processed_time=datetime.now(),
                memory_cache_id=entity.memory_cache_id,
                source_document=entity.source_document
            )
            self.storage.save_entity(new_entity)
            
            if verbose:
                wprint(f"        已将 {len(self_ref_contents)} 个自指向关系的内容总结到实体content中")
        
        # 删除这些自指向关系
        deleted_count = self.storage.delete_self_referential_relations_for_entity(target_entity_id)
        
        if verbose:
            wprint(f"        已删除 {deleted_count} 个自指向关系")
        
        return deleted_count
    
    def _process_single_alias_relation(self, rel_info: Dict, verbose: bool = True) -> Optional[Dict]:
        """
        处理单个别名关系（可并行调用）
        
        Args:
            rel_info: 关系信息字典，包含：
                - entity1_id, entity2_id: 原始entity_id
                - actual_entity1_id, actual_entity2_id: 实际使用的entity_id（可能已合并）
                - entity1_name, entity2_name: 实体名称
                - content: 初步的关系content（可选，如果提供则用于初步判断）
            verbose: 是否输出详细信息
        
        Returns:
            处理结果字典，包含：
                - entity1_id, entity2_id: 实体ID
                - entity1_name, entity2_name: 实体名称
                - content: 关系content
                - relation_id: 关系ID
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
            entity1 = self.storage.get_entity_by_id(actual_entity1_id)
            entity2 = self.storage.get_entity_by_id(actual_entity2_id)
            
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
                    # 按relation_id分组，每个relation_id只保留最新版本
                    relation_dict = {}
                    for rel in existing_relations_before:
                        if rel.relation_id not in relation_dict:
                            relation_dict[rel.relation_id] = rel
                        else:
                            if rel.processed_time > relation_dict[rel.relation_id].processed_time:
                                relation_dict[rel.relation_id] = rel

                    unique_relations = list(relation_dict.values())
                    existing_relations_info = [
                        {
                            'relation_id': r.relation_id,
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

                    if match_result and match_result.get('relation_id'):
                        # 匹配到已有关系，判断是否需要更新
                        relation_id = match_result['relation_id']
                        latest_relation = relation_dict.get(relation_id)
                        
                        if latest_relation:
                            need_update = self.llm_client.judge_content_need_update(
                                latest_relation.content,
                                preliminary_content
                            )
                            
                            if not need_update:
                                # 不需要更新，直接返回，跳过后续详细生成
                                if verbose:
                                    wprint(f"        关系已存在且无需更新（使用初步content判断），跳过详细生成: {relation_id}")
                                return {
                                    "entity1_id": actual_entity1_id,
                                    "entity2_id": actual_entity2_id,
                                    "entity1_name": entity1_name,
                                    "entity2_name": entity2_name,
                                    "content": latest_relation.content,
                                    "relation_id": relation_id,
                                    "is_new": False,
                                    "is_updated": False
                                }
                            else:
                                if verbose:
                                    wprint(f"        关系已存在但需要更新（使用初步content判断），继续生成详细content: {relation_id}")
            
            # 获取实体的memory_cache（只有在需要详细生成时才获取）
            entity1_memory_cache = None
            entity2_memory_cache = None
            if entity1.memory_cache_id:
                from_cache = self.storage.load_memory_cache(entity1.memory_cache_id)
                if from_cache:
                    entity1_memory_cache = from_cache.content
            
            if entity2.memory_cache_id:
                to_cache = self.storage.load_memory_cache(entity2.memory_cache_id)
                if to_cache:
                    entity2_memory_cache = to_cache.content
            
            # 步骤1：先判断是否真的需要创建关系边（使用完整的实体信息）
            need_create_relation = self.llm_client.judge_need_create_relation(
                entity1_name=entity1.name,
                entity1_content=entity1.content,
                entity2_name=entity2.name,
                entity2_content=entity2.content,
                entity1_memory_cache=entity1_memory_cache,
                entity2_memory_cache=entity2_memory_cache
            )
            
            if not need_create_relation:
                if verbose:
                    wprint(f"        判断结果：两个实体之间没有明确的、有意义的关联，跳过创建关系边")
                return None
            
            if verbose:
                wprint(f"        判断结果：两个实体之间存在明确的、有意义的关联，需要创建关系边")
            
            # 步骤2：生成关系的memory_cache（临时，不保存）
            relation_memory_cache_content = self.llm_client.generate_relation_memory_cache(
                [],  # 关系列表为空，因为还没有生成关系content
                [
                    {"entity_id": actual_entity1_id, "name": entity1.name, "content": entity1.content},
                    {"entity_id": actual_entity2_id, "name": entity2.name, "content": entity2.content}
                ],
                {
                    actual_entity1_id: entity1_memory_cache or "",
                    actual_entity2_id: entity2_memory_cache or ""
                }
            )
            
            # 步骤3：根据memory_cache和两个实体，生成关系的content
            relation_content = self.llm_client.generate_relation_content(
                entity1_name=entity1.name,
                entity1_content=entity1.content,
                entity2_name=entity2.name,
                entity2_content=entity2.content,
                relation_memory_cache=relation_memory_cache_content,
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
                # 按relation_id分组，每个relation_id只保留最新版本
                relation_dict = {}
                for rel in existing_relations_before:
                    if rel.relation_id not in relation_dict:
                        relation_dict[rel.relation_id] = rel
                    else:
                        if rel.processed_time > relation_dict[rel.relation_id].processed_time:
                            relation_dict[rel.relation_id] = rel
                
                unique_relations = list(relation_dict.values())
                existing_relations_info = [
                    {
                        'relation_id': r.relation_id,
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

                if match_result and match_result.get('relation_id'):
                    # 匹配到已有关系，判断是否需要更新
                    relation_id = match_result['relation_id']
                    latest_relation = relation_dict.get(relation_id)
                    
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
                                wprint(f"        关系已存在，需要更新: {relation_id}")
                        else:
                            # 不需要更新
                            if verbose:
                                wprint(f"        关系已存在，无需更新: {relation_id}")
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
            
            # 只有在需要创建或更新时，才保存memory_cache并创建/更新关系
            if need_create_or_update:
                # 生成总结的memory_cache（用于json的text字段）
                cache_text_content = f"""实体1:
- name: {entity1.name}
- content: {entity1.content}
- memory_cache: {entity1_memory_cache if entity1_memory_cache else '无'}

实体2:
- name: {entity2.name}
- content: {entity2.content}
- memory_cache: {entity2_memory_cache if entity2_memory_cache else '无'}
"""
                
                # 保存memory_cache（md和json）
                # 从实体中获取文档名
                doc_name_from_entity = entity1.source_document if entity1.source_document else ""
                
                relation_memory_cache = MemoryCache(
                    absolute_id=f"cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                    content=relation_memory_cache_content,
                    event_time=datetime.now(),
                    source_document=doc_name_from_entity,
                    activity_type="知识图谱整理-关系生成"
                )
                # 保存memory_cache，json的text是两个实体的name+content+memory_cache
                self.storage.save_memory_cache(relation_memory_cache, text=cache_text_content)
                
                if verbose:
                    wprint(f"        保存关系memory_cache: {relation_memory_cache.absolute_id}")
                
                relation = self.relation_processor._process_single_relation(
                    extracted_relation,
                    actual_entity1_id,
                    actual_entity2_id,
                    relation_memory_cache.absolute_id,
                    entity1.name,
                    entity2.name,
                    verbose_relation=verbose,
                    source_document=doc_name_from_entity,
                    base_time=relation_memory_cache.event_time,
                )
            
            if relation:
                # 返回关系信息（用于后续统计）
                alias_detail = {
                    "entity1_id": actual_entity1_id,
                    "entity2_id": actual_entity2_id,
                    "entity1_name": entity1.name,
                    "entity2_name": entity2.name,
                    "content": relation_content,
                    "relation_id": relation.relation_id,
                    "is_new": is_new_relation,
                    "is_updated": is_updated
                }
                
                if is_new_relation:
                    if verbose:
                        wprint(f"        成功创建新关系: {relation.relation_id}")
                elif is_updated:
                    if verbose:
                        wprint(f"        关系已存在，已更新: {relation.relation_id}")
                else:
                    if verbose:
                        wprint(f"        关系已存在，无需更新: {relation.relation_id}")
                
                return alias_detail
            else:
                if verbose:
                    wprint(f"        创建关系失败")
                return None
                    
        except Exception as e:
            if verbose:
                wprint(f"        处理失败: {e}")
            import traceback
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
        summary_cache = MemoryCache(
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
        self.storage.save_memory_cache(
            summary_cache, 
            text=consolidation_text
        )
        
        if verbose:
            wprint(f"  已创建整理总结记忆缓存: {summary_cache.absolute_id}")
    
    def _build_entity_list_summary(self, entities_for_analysis: List[Dict]) -> str:
        """
        构建传入LLM的entity列表总结
        
        Args:
            entities_for_analysis: 传入LLM分析的实体列表
            
        Returns:
            Markdown格式的实体列表总结
        """
        summary_lines = []
        summary_lines.append(f"共 {len(entities_for_analysis)} 个实体：\n")
        
        for idx, entity_info in enumerate(entities_for_analysis, 1):
            entity_id = entity_info.get("entity_id", "未知")
            name = entity_info.get("name", "未知")
            content = entity_info.get("content", "")
            version_count = entity_info.get("version_count", 0)
            
            # 截取content的前100字符作为摘要
            content_snippet = content[:100] + "..." if len(content) > 100 else content
            
            summary_lines.append(f"{idx}. **{name}** (entity_id: `{entity_id}`, 版本数: {version_count})")
            summary_lines.append(f"   - 内容摘要: {content_snippet}")
            summary_lines.append("")
        
        return "\n".join(summary_lines)
    
    def _build_entity_list_text(self, entities_for_analysis: List[Dict]) -> str:
        """
        构建包含完整entity信息的文本（用于保存到JSON的text字段）
        
        Args:
            entities_for_analysis: 传入LLM分析的实体列表
            
        Returns:
            包含完整实体信息的文本（包括entity_id, name, content等）
        """
        text_lines = []
        text_lines.append(f"知识图谱整理 - 传入LLM进行判断的实体列表（共 {len(entities_for_analysis)} 个实体）\n")
        text_lines.append("=" * 80)
        text_lines.append("")
        
        for idx, entity_info in enumerate(entities_for_analysis, 1):
            entity_id = entity_info.get("entity_id", "未知")
            name = entity_info.get("name", "未知")
            content = entity_info.get("content", "")
            version_count = entity_info.get("version_count", 0)
            
            text_lines.append(f"{idx}. 实体名称: {name}")
            text_lines.append(f"   entity_id: {entity_id}")
            text_lines.append(f"   版本数: {version_count}")
            text_lines.append(f"   完整内容:")
            text_lines.append(f"   {content}")
            text_lines.append("")
            text_lines.append("-" * 80)
            text_lines.append("")
        
        return "\n".join(text_lines)


def main():
    """示例使用"""
    import sys
    
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
