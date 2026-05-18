"""
实体处理模块：实体搜索、对齐、更新/新建

This module provides EntityProcessor — the main entry point for entity
processing. Heavy logic is delegated to sub-modules:
  - entity_construction: factory/build helpers
  - entity_search: search, filtering, alignment guard
  - entity_sequential: sequential fallback processing
  - entity_parallel: parallel processing
  - entity_batch: batch candidate processing mixin
"""
from typing import List, Dict, Optional, Tuple, Any
from collections import OrderedDict, defaultdict
import re
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

import threading
import uuid
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

from core.debug_log import log as dbg, log_struct as _dbg_struct, log_section as _dbg_section
from core.models import Entity, Episode, ContentPatch
from core.llm.client import LLMClient
from core.utils import wprint_info, calculate_jaccard_similarity, cosine_similarity

# Pool refs are now in _shared
from ._shared import _doc_basename, _get_or_create_pool, _get_entity_pool, _ENTITY_POOL, _ENTITY_POOL_MAX
from core.content_schema import (
    ENTITY_SECTIONS,
    compute_content_patches,
    content_to_sections,
    compute_section_diff,
    sections_equal,
    has_any_change,
    section_hash,
)
from core.remember.entity_candidates import (
    EntityCandidateBuilder,
    normalize_entity_name_for_matching,
)
from core.remember._shared import _TITLE_SUFFIXES_RE

# Sub-module imports
from core.remember.entity_construction import (
    _extract_summary as _extract_summary_fn,
    _construct_entity as _construct_entity_fn,
    _build_new_entity as _build_new_entity_fn,
    _create_new_entity as _create_new_entity_fn,
    _build_entity_version as _build_entity_version_fn,
    _create_entity_version as _create_entity_version_fn,
    _compute_entity_patches as _compute_entity_patches_fn,
)
from core.remember.entity_search import (
    _calculate_jaccard_similarity as _calc_jaccard_fn,
    _cosine_similarity as _cosine_sim_fn,
    _alignment_guard as _alignment_guard_fn,
    _search_entity_candidates as _search_entity_candidates_fn,
    _filter_candidates_by_existing_relations as _filter_candidates_fn,
    _try_context_alias_merge as _try_context_alias_merge_fn,
)
from core.remember.entity_sequential import (
    _process_entity_sequential_fallback as _process_entity_sequential_fallback_fn,
)
from core.remember.entity_parallel import (
    _process_entities_sequential as _process_entities_sequential_fn,
    _process_entities_parallel as _process_entities_parallel_fn,
)
from core.remember.entity_batch import _EntityBatchMixin


def _preprocess_extraction_context(extracted_entities, extracted_relations):
    """Build entity name set, relation pair set, and related-entity name set from extraction results."""
    extracted_entity_names = {e['name'] for e in extracted_entities}
    extracted_relation_pairs = set()
    related_entity_names = set()
    if extracted_relations:
        for rel in extracted_relations:
            entity1_name = rel.get('entity1_name') or rel.get('from_entity_name', '').strip()
            entity2_name = rel.get('entity2_name') or rel.get('to_entity_name', '').strip()
            content = rel.get('content', '')
            content_lower = content.strip().lower()
            if entity1_name and entity2_name:
                pair_key = (entity1_name, entity2_name) if entity1_name <= entity2_name else (entity2_name, entity1_name)
                extracted_relation_pairs.add((pair_key, hash(content_lower)))
                related_entity_names.add(entity1_name)
                related_entity_names.add(entity2_name)
    return extracted_entity_names, extracted_relation_pairs, related_entity_names


class EntityProcessor(_EntityBatchMixin):
    """实体处理器 - 负责实体的搜索、对齐、更新和新建"""

    def __init__(self, storage, llm_client: LLMClient,
                 max_similar_entities: int = 10, content_snippet_length: int = 50,
                 max_alignment_candidates: Optional[int] = None,
                 verbose: bool = True,
                 entity_progress_verbose: bool = False,
                 merge_safe_embedding_threshold: float = 0.55,
                 merge_safe_jaccard_threshold: float = 0.4):
        self.storage = storage
        self.llm_client = llm_client
        self.max_similar_entities = max_similar_entities
        self.content_snippet_length = content_snippet_length
        self.max_alignment_candidates = max_alignment_candidates  # None = 不限制
        self.batch_resolution_enabled = True
        self.batch_resolution_confidence_threshold = 0.75
        self.verbose = verbose
        # 逐实体树状进度（处理实体 x/y、批量候选等）；默认关闭以免服务/API 控制台刷屏
        self.entity_progress_verbose = entity_progress_verbose
        self._entity_tree_log_result = verbose and entity_progress_verbose
        self.merge_safe_embedding_threshold = merge_safe_embedding_threshold
        self.merge_safe_jaccard_threshold = merge_safe_jaccard_threshold
        # Instance-level LRU cache for _alignment_guard (avoids repeated LLM calls for same entity pairs)
        self._alignment_guard_cache: OrderedDict[Tuple[str, ...], Optional[Tuple[str, float]]] = OrderedDict()
        # Candidate builder — encapsulates all candidate table logic
        self._candidate_builder = EntityCandidateBuilder(
            storage=self.storage,
            llm_client=self.llm_client,
            max_alignment_candidates=max_alignment_candidates,
            max_similar_entities=max_similar_entities,
            merge_safe_embedding_threshold=merge_safe_embedding_threshold,
            merge_safe_jaccard_threshold=merge_safe_jaccard_threshold,
            verbose=verbose,
            entity_progress_verbose=entity_progress_verbose,
        )

    def _entity_tree_log(self) -> bool:
        return self._entity_tree_log_result

    def encode_entities_for_candidate_table(
        self, extracted_entities: List[Dict[str, str]]
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """为本窗实体批量编码 name / name+snippet，供 _build_entity_candidate_table 使用（可异步预取）。"""
        if not extracted_entities:
            return None, None
        if not self.storage.embedding_client or not self.storage.embedding_client.is_available():
            return None, None
        snip = self.llm_client.effective_entity_snippet_length()
        N = len(extracted_entities)
        name_texts = [e["name"] for e in extracted_entities]
        full_texts = [f"# {e['name']}\n{e['content'][:snip]}" for e in extracted_entities]
        all_embeddings = self.storage.embedding_client.encode(name_texts + full_texts)
        return all_embeddings[:N], all_embeddings[N:]

    def process_entities(self, extracted_entities: List[Dict[str, str]],
                        episode_id: str, similarity_threshold: float = 0.7,
                        episode: Optional[Episode] = None, source_document: str = "",
                        context_text: Optional[str] = None,
                        extracted_relations: Optional[List[Dict[str, str]]] = None,
                        jaccard_search_threshold: Optional[float] = None,
                        embedding_name_search_threshold: Optional[float] = None,
                        embedding_full_search_threshold: Optional[float] = None,
                        on_entity_processed: Optional[callable] = None,
                        base_time: Optional[datetime] = None,
                        max_workers: Optional[int] = None,
                        verbose: Optional[bool] = None,
                        entity_embedding_prefetch: Optional[Future] = None,
                        already_versioned_family_ids: Optional[set] = None) -> Tuple[List[Entity], List[Dict], Dict[str, str]]:
        """
        处理抽取的实体：搜索、对齐、更新/新建。
        当 max_workers > 1 且实体数 > 1 时使用多线程并行；合并冲突时以数据库中已存在的 family_id 为准。

        Args:
            extracted_entities: 抽取的实体列表（每个包含name和content）
            episode_id: 当前记忆缓存的ID
            similarity_threshold: 相似度阈值（用于搜索，作为默认值）
            episode: 当前记忆缓存对象（可选，用于LLM判断时提供上下文）
            source_document: 文档名称（只保存文档名，不包含路径）
            context_text: 可选的上下文文本（当前处理的文本片段），用于精细化判断时提供场景信息
            extracted_relations: 步骤3抽取的关系列表（用于判断是否已存在关系）
            jaccard_search_threshold: Jaccard搜索（name_only）的相似度阈值（可选，默认使用similarity_threshold）
            embedding_name_search_threshold: Embedding搜索（name_only）的相似度阈值（可选，默认使用similarity_threshold）
            embedding_full_search_threshold: Embedding搜索（name+content）的相似度阈值（可选，默认使用similarity_threshold）
            on_entity_processed: 每个实体处理完的回调（可选）
            base_time: 基准时间（可选）
            max_workers: 并行线程数；>1 且实体数>1 时启用多线程，合并冲突时以数据库已有 id 为准
            entity_embedding_prefetch: 可选 Future，结果为 encode_entities_for_candidate_table 的返回值；失败时回退为现场 encode
            already_versioned_family_ids: 可选 set，当前 process_entities 调用期间已创建版本的 family_id 集合；
                防止同一窗口内多个抽取实体匹配到同一已有实体时重复创建版本。
                若为 None 则自动创建空集合。

        Returns:
            Tuple[处理后的实体列表, 待处理的关系列表, 实体名称到ID的映射]
            关系信息格式：{"entity1_name": "...", "entity2_name": "...", "content": "...", "relation_type": "alias|normal"}
            注意：关系中的实体使用名称而不是ID，因为新实体在创建前还没有ID
        """
        # 临时覆盖 verbose
        _orig_verbose = self.verbose
        if verbose is not None:
            self.verbose = verbose

        try:
            if already_versioned_family_ids is None:
                already_versioned_family_ids = set()
            prefetched_embeddings: Optional[Tuple[Optional[Any], Optional[Any]]] = None
            if entity_embedding_prefetch is not None:
                try:
                    prefetched_embeddings = entity_embedding_prefetch.result()
                except Exception as exc:
                    wprint_info(f"  │  embedding预取失败: {exc}")
                    prefetched_embeddings = None
            use_parallel = (max_workers is not None and max_workers > 1 and len(extracted_entities) > 1)
            if use_parallel:
                result = self._process_entities_parallel(
                    extracted_entities=extracted_entities,
                    episode_id=episode_id,
                    similarity_threshold=similarity_threshold,
                    episode=episode,
                    source_document=source_document,
                    context_text=context_text,
                    extracted_relations=extracted_relations,
                    jaccard_search_threshold=jaccard_search_threshold,
                    embedding_name_search_threshold=embedding_name_search_threshold,
                    embedding_full_search_threshold=embedding_full_search_threshold,
                    on_entity_processed=on_entity_processed,
                    base_time=base_time,
                    max_workers=max_workers,
                    prefetched_embeddings=prefetched_embeddings,
                    already_versioned_family_ids=already_versioned_family_ids,
                )
            else:
                result = self._process_entities_sequential(
                    extracted_entities=extracted_entities,
                    episode_id=episode_id,
                    similarity_threshold=similarity_threshold,
                    episode=episode,
                    source_document=source_document,
                    context_text=context_text,
                    extracted_relations=extracted_relations,
                    jaccard_search_threshold=jaccard_search_threshold,
                    embedding_name_search_threshold=embedding_name_search_threshold,
                    embedding_full_search_threshold=embedding_full_search_threshold,
                    on_entity_processed=on_entity_processed,
                    base_time=base_time,
                    prefetched_embeddings=prefetched_embeddings,
                    already_versioned_family_ids=already_versioned_family_ids,
                )
            return result
        finally:
            self.verbose = _orig_verbose

    # ── Thin wrappers delegating to entity_parallel sub-module ──

    def _process_entities_sequential(self, extracted_entities: List[Dict[str, str]],
                        episode_id: str, similarity_threshold: float = 0.7,
                        episode: Optional[Episode] = None, source_document: str = "",
                        context_text: Optional[str] = None,
                        extracted_relations: Optional[List[Dict[str, str]]] = None,
                        jaccard_search_threshold: Optional[float] = None,
                        embedding_name_search_threshold: Optional[float] = None,
                        embedding_full_search_threshold: Optional[float] = None,
                        on_entity_processed: Optional[callable] = None,
                        base_time: Optional[datetime] = None,
                        prefetched_embeddings: Optional[Tuple[Optional[Any], Optional[Any]]] = None,
                        already_versioned_family_ids: Optional[set] = None) -> Tuple[List[Entity], List[Dict], Dict[str, str]]:
        return _process_entities_sequential_fn(
            storage=self.storage,
            llm_client=self.llm_client,
            candidate_builder=self._candidate_builder,
            entity_tree_log=self._entity_tree_log(),
            build_entity_candidate_table_fn=self._build_entity_candidate_table,
            process_entity_with_batch_candidates_fn=self._process_entity_with_batch_candidates,
            extracted_entities=extracted_entities,
            episode_id=episode_id,
            similarity_threshold=similarity_threshold,
            episode=episode,
            source_document=source_document,
            context_text=context_text,
            extracted_relations=extracted_relations,
            jaccard_search_threshold=jaccard_search_threshold,
            embedding_name_search_threshold=embedding_name_search_threshold,
            embedding_full_search_threshold=embedding_full_search_threshold,
            on_entity_processed=on_entity_processed,
            base_time=base_time,
            prefetched_embeddings=prefetched_embeddings,
            already_versioned_family_ids=already_versioned_family_ids,
        )

    def _process_entities_parallel(self, extracted_entities: List[Dict[str, str]],
                        episode_id: str, similarity_threshold: float = 0.7,
                        episode: Optional[Episode] = None, source_document: str = "",
                        context_text: Optional[str] = None,
                        extracted_relations: Optional[List[Dict[str, str]]] = None,
                        jaccard_search_threshold: Optional[float] = None,
                        embedding_name_search_threshold: Optional[float] = None,
                        embedding_full_search_threshold: Optional[float] = None,
                        on_entity_processed: Optional[callable] = None,
                        base_time: Optional[datetime] = None,
                        max_workers: int = 1,
                        prefetched_embeddings: Optional[Tuple[Optional[Any], Optional[Any]]] = None,
                        already_versioned_family_ids: Optional[set] = None) -> Tuple[List[Entity], List[Dict], Dict[str, str]]:
        return _process_entities_parallel_fn(
            storage=self.storage,
            llm_client=self.llm_client,
            candidate_builder=self._candidate_builder,
            entity_tree_log=self._entity_tree_log(),
            build_entity_candidate_table_fn=self._build_entity_candidate_table,
            process_entity_with_batch_candidates_fn=self._process_entity_with_batch_candidates,
            get_entity_pool_fn=_get_entity_pool,
            extracted_entities=extracted_entities,
            episode_id=episode_id,
            similarity_threshold=similarity_threshold,
            episode=episode,
            source_document=source_document,
            context_text=context_text,
            extracted_relations=extracted_relations,
            jaccard_search_threshold=jaccard_search_threshold,
            embedding_name_search_threshold=embedding_name_search_threshold,
            embedding_full_search_threshold=embedding_full_search_threshold,
            on_entity_processed=on_entity_processed,
            base_time=base_time,
            max_workers=max_workers,
            prefetched_embeddings=prefetched_embeddings,
            already_versioned_family_ids=already_versioned_family_ids,
        )

    # 名称规范化：委托给共享模块
    _normalize_entity_name_for_matching = staticmethod(normalize_entity_name_for_matching)
    _TITLE_SUFFIXES_RE = _TITLE_SUFFIXES_RE  # re-export from entity_candidates module

    # ── Thin wrappers delegating to entity_search sub-module ──

    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        return _calc_jaccard_fn(text1, text2)

    def _alignment_guard(
        self, name_a: str, content_a: str, name_b: str, content_b: str,
        *, name_match_type: str = "none", require_content: bool = True,
    ) -> Optional[Tuple[str, float]]:
        return _alignment_guard_fn(
            self.llm_client, self._alignment_guard_cache,
            name_a, content_a, name_b, content_b,
            name_match_type=name_match_type, require_content=require_content,
        )

    @staticmethod
    def _cosine_similarity(embedding1, embedding2) -> float:
        return _cosine_sim_fn(embedding1, embedding2)

    def _build_entity_candidate_table(self,
                                      extracted_entities: List[Dict[str, str]],
                                      similarity_threshold: float,
                                      jaccard_search_threshold: Optional[float] = None,
                                      embedding_name_search_threshold: Optional[float] = None,
                                      embedding_full_search_threshold: Optional[float] = None,
                                      prefetched_embeddings: Optional[Tuple[Optional[Any], Optional[Any]]] = None,
                                      ) -> Dict[int, List[Dict[str, Any]]]:
        """Delegate to EntityCandidateBuilder."""
        return self._candidate_builder.build_candidate_table(
            extracted_entities=extracted_entities,
            similarity_threshold=similarity_threshold,
            jaccard_search_threshold=jaccard_search_threshold,
            embedding_name_search_threshold=embedding_name_search_threshold,
            embedding_full_search_threshold=embedding_full_search_threshold,
            prefetched_embeddings=prefetched_embeddings,
        )

    def _try_context_alias_merge(
        self,
        entity_name: str,
        entity_content: str,
        candidates: List[Dict[str, Any]],
        context_text: Optional[str],
        episode_id: str,
        source_document: str,
        base_time: Optional[datetime],
        already_versioned_family_ids: Optional[set],
        _version_lock: Optional[Any],
        entity_name_to_id: Optional[Dict[str, str]] = None,
    ) -> Optional[Tuple]:
        return _try_context_alias_merge_fn(
            storage=self.storage,
            llm_client=self.llm_client,
            alignment_guard_cache=self._alignment_guard_cache,
            merge_two_contents_fn=self._merge_two_contents,
            build_entity_version_fn=self._build_entity_version,
            mark_versioned_fn=self._mark_versioned,
            entity_tree_log=self._entity_tree_log(),
            entity_name=entity_name,
            entity_content=entity_content,
            candidates=candidates,
            context_text=context_text,
            episode_id=episode_id,
            source_document=source_document,
            base_time=base_time,
            already_versioned_family_ids=already_versioned_family_ids,
            _version_lock=_version_lock,
            entity_name_to_id=entity_name_to_id,
        )

    # ── Helpers kept on the class ──

    @staticmethod
    def _mark_versioned(family_id: str, already_versioned: Optional[set], lock: Optional[Any] = None):
        """线程安全地标记 family_id 已创建版本，防止同窗口重复版本化。"""
        if already_versioned is not None:
            if lock:
                with lock:
                    already_versioned.add(family_id)
            else:
                already_versioned.add(family_id)

    def _merge_two_contents(self, old_entity, entity_name, entity_content,
                            source_document, episode_id, base_time=None):
        """增量合并两个实体的 content，遵循 CLAUDE.md 第九条 fast-forward 策略。

        Args:
            old_entity: 已有实体（有 .content, .name, .source_document）
            entity_name: 新实体名称
            entity_content: 新实体内容
            source_document: 新实体来源文档
            episode_id: Episode ID
            base_time: 基准时间

        Returns:
            merged_content (str)
        """
        old_content = (old_entity.content or "").strip()
        new_content = entity_content.strip()
        if old_content and old_content != new_content:
            return self.llm_client.merge_multiple_entity_contents(
                [old_entity.content, entity_content],
                entity_sources=[old_entity.source_document, source_document],
                entity_names=[old_entity.name, entity_name],
            )
        elif old_content == new_content:
            return old_entity.content or entity_content
        else:
            return entity_content

    # ── Thin wrappers delegating to entity_sequential sub-module ──

    def _search_entity_candidates(
        self,
        entity_name: str,
        entity_content: str,
        similarity_threshold: float,
        jaccard_search_threshold: Optional[float] = None,
        embedding_name_search_threshold: Optional[float] = None,
        embedding_full_search_threshold: Optional[float] = None,
        extracted_entity_names: Optional[set] = None,
        extracted_relation_pairs: Optional[set] = None,
    ) -> List[Entity]:
        return _search_entity_candidates_fn(
            storage=self.storage,
            llm_client=self.llm_client,
            max_similar_entities=self.max_similar_entities,
            entity_tree_log=self._entity_tree_log(),
            entity_name=entity_name,
            entity_content=entity_content,
            similarity_threshold=similarity_threshold,
            jaccard_search_threshold=jaccard_search_threshold,
            embedding_name_search_threshold=embedding_name_search_threshold,
            embedding_full_search_threshold=embedding_full_search_threshold,
            extracted_entity_names=extracted_entity_names,
            extracted_relation_pairs=extracted_relation_pairs,
        )

    def _filter_candidates_by_existing_relations(
        self,
        candidates: List[Entity],
        entity_name: str,
        extracted_entity_names: set,
        extracted_relation_pairs: set,
    ) -> List[Entity]:
        return _filter_candidates_fn(
            candidates, entity_name,
            extracted_entity_names, extracted_relation_pairs,
            entity_tree_log=self._entity_tree_log(),
        )

    def _process_entity_sequential_fallback(self, extracted_entity: Dict[str, str],
                               episode_id: str,
                               similarity_threshold: float,
                               episode: Optional[Episode] = None,
                               source_document: str = "",
                               context_text: Optional[str] = None,
                               entity_index: int = 0,
                               total_entities: int = 0,
                               extracted_entity_names: Optional[set] = None,
                               extracted_relation_pairs: Optional[set] = None,
                               jaccard_search_threshold: Optional[float] = None,
                               embedding_name_search_threshold: Optional[float] = None,
                               embedding_full_search_threshold: Optional[float] = None,
                               base_time: Optional[datetime] = None,
                               already_versioned_family_ids: Optional[set] = None,
                               _version_lock: Optional[Any] = None,
                               prefetched_embedding: Optional[Any] = None,
                               prebuilt_candidates: Optional[List[Dict[str, Any]]] = None) -> Tuple[Optional[Entity], List[Dict], Dict[str, str]]:
        return _process_entity_sequential_fallback_fn(
            storage=self.storage,
            llm_client=self.llm_client,
            entity_tree_log=self._entity_tree_log(),
            search_entity_candidates_fn=self._search_entity_candidates,
            create_new_entity_fn=self._create_new_entity,
            build_new_entity_fn=self._build_new_entity,
            create_entity_version_fn=self._create_entity_version,
            build_entity_version_fn=self._build_entity_version,
            mark_versioned_fn=self._mark_versioned,
            alignment_guard_fn=self._alignment_guard,
            calculate_jaccard_fn=self._calculate_jaccard_similarity,
            cosine_similarity_fn=self._cosine_similarity,
            merge_two_contents_fn=self._merge_two_contents,
            extracted_entity=extracted_entity,
            episode_id=episode_id,
            similarity_threshold=similarity_threshold,
            episode=episode,
            source_document=source_document,
            context_text=context_text,
            entity_index=entity_index,
            total_entities=total_entities,
            extracted_entity_names=extracted_entity_names,
            extracted_relation_pairs=extracted_relation_pairs,
            jaccard_search_threshold=jaccard_search_threshold,
            embedding_name_search_threshold=embedding_name_search_threshold,
            embedding_full_search_threshold=embedding_full_search_threshold,
            base_time=base_time,
            already_versioned_family_ids=already_versioned_family_ids,
            _version_lock=_version_lock,
            prefetched_embedding=prefetched_embedding,
            prebuilt_candidates=prebuilt_candidates,
        )

    # ── Thin wrappers delegating to entity_construction sub-module ──

    @staticmethod
    def _extract_summary(name: str, content: str) -> str:
        return _extract_summary_fn(name, content)

    def _construct_entity(self, name: str, content: str, episode_id: str,
                          family_id: str, source_document: str = "",
                          base_time: Optional[datetime] = None,
                          confidence: Optional[float] = None) -> Entity:
        return _construct_entity_fn(name, content, episode_id, family_id,
                                    source_document=source_document, base_time=base_time,
                                    confidence=confidence)

    def _build_new_entity(self, name: str, content: str, episode_id: str,
                          source_document: str = "", base_time: Optional[datetime] = None,
                          confidence: Optional[float] = None) -> Entity:
        return _build_new_entity_fn(name, content, episode_id, source_document,
                                    base_time=base_time, confidence=confidence)

    def _create_new_entity(self, name: str, content: str, episode_id: str,
                           source_document: str = "", base_time: Optional[datetime] = None,
                           confidence: Optional[float] = None) -> Entity:
        return _create_new_entity_fn(self.storage, name, content, episode_id,
                                     source_document, base_time=base_time, confidence=confidence)

    def _build_entity_version(self, family_id: str, name: str, content: str,
                              episode_id: str, source_document: str = "",
                              base_time: Optional[datetime] = None,
                              old_content: str = "",
                              old_content_format: str = "plain") -> Entity:
        return _build_entity_version_fn(family_id, name, content, episode_id,
                                        source_document, base_time=base_time,
                                        old_content=old_content,
                                        old_content_format=old_content_format)

    def _create_entity_version(self, family_id: str, name: str, content: str,
                              episode_id: str, source_document: str = "",
                              base_time: Optional[datetime] = None,
                              old_content: str = "",
                              old_content_format: str = "plain") -> Entity:
        return _create_entity_version_fn(self.storage, family_id, name, content,
                                         episode_id, source_document, base_time=base_time,
                                         old_content=old_content,
                                         old_content_format=old_content_format)

    def _compute_entity_patches(
        self,
        family_id: str,
        old_content: str,
        old_content_format: str,
        new_content: str,
        new_absolute_id: str,
        source_document: str = "",
        event_time: Optional[datetime] = None,
    ) -> list:
        return _compute_entity_patches_fn(
            family_id=family_id,
            old_content=old_content,
            old_content_format=old_content_format,
            new_content=new_content,
            new_absolute_id=new_absolute_id,
            source_document=source_document,
            event_time=event_time,
        )
