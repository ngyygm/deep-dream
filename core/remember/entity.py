"""
实体处理模块：实体搜索、对齐、更新/新建
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
from core.storage.neo4j_store import Neo4jStorageManager
from core.llm.client import LLMClient
from core.utils import wprint_info, calculate_jaccard_similarity, cosine_similarity

# Shared thread pool — reused across entity processing calls within a session
_ENTITY_POOL: ThreadPoolExecutor | None = None
_ENTITY_POOL_MAX = 4


def _get_entity_pool(max_workers: int) -> ThreadPoolExecutor:
    """Return (and lazily create) the shared entity ThreadPoolExecutor."""
    global _ENTITY_POOL, _ENTITY_POOL_MAX
    if _ENTITY_POOL is not None:
        if max_workers > _ENTITY_POOL_MAX:
            try:
                _ENTITY_POOL.shutdown(wait=False)
            except Exception:
                pass
            _ENTITY_POOL = None
        else:
            return _ENTITY_POOL
    _ENTITY_POOL_MAX = max(max_workers, _ENTITY_POOL_MAX)
    _ENTITY_POOL = ThreadPoolExecutor(
        max_workers=_ENTITY_POOL_MAX,
        thread_name_prefix="entity",
    )
    return _ENTITY_POOL


def _doc_basename(source_document: str) -> str:
    """Extract basename from source_document path using rpartition."""
    return source_document.rpartition('/')[-1] if source_document else ""
from core.content_schema import (
    ENTITY_SECTIONS,
    content_to_sections,
    compute_section_diff,
    sections_equal,
    has_any_change,
    section_hash,
)
from core.remember.entity_candidates import (
    EntityCandidateBuilder,
    normalize_entity_name_for_matching,
    _TITLE_SUFFIXES_RE,
)


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


class EntityProcessor:
    """实体处理器 - 负责实体的搜索、对齐、更新和新建"""
    
    def __init__(self, storage: Neo4jStorageManager, llm_client: LLMClient,
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
        full_texts = [f"{e['name']} {e['content'][:snip]}" for e in extracted_entities]
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
        """串行处理实体（原逻辑）。"""
        processed_entities: List[Entity] = []
        pending_relations: List[Dict] = []
        entity_name_to_id: Dict[str, str] = {}
        entities_to_persist: List[Entity] = []

        extracted_entity_names, extracted_relation_pairs, related_entity_names = _preprocess_extraction_context(
            extracted_entities, extracted_relations,
        )

        candidate_table = self._build_entity_candidate_table(
            extracted_entities,
            similarity_threshold=similarity_threshold,
            jaccard_search_threshold=jaccard_search_threshold,
            embedding_name_search_threshold=embedding_name_search_threshold,
            embedding_full_search_threshold=embedding_full_search_threshold,
            prefetched_embeddings=prefetched_embeddings,
        )

        total_entities = len(extracted_entities)
        _skipped_orphans = 0
        # Extract per-entity full-text embeddings from prefetch for sequential path
        _prefetched_full_embs = None
        if prefetched_embeddings is not None:
            try:
                _, _full_embs = prefetched_embeddings
                if _full_embs is not None:
                    _prefetched_full_embs = _full_embs
            except Exception:
                pass
        for idx, extracted_entity in enumerate(extracted_entities, 1):
            candidates = candidate_table.get(idx - 1, [])
            _ent_emb = None
            if _prefetched_full_embs is not None and (idx - 1) < len(_prefetched_full_embs):
                try:
                    _ent_emb = np.array(_prefetched_full_embs[idx - 1], dtype=np.float32)
                except Exception:
                    pass
            entity, relations, name_mapping, to_persist = self._process_entity_with_batch_candidates(
                extracted_entity=extracted_entity,
                candidates=candidates,
                episode_id=episode_id,
                similarity_threshold=similarity_threshold,
                episode=episode,
                source_document=source_document,
                context_text=context_text,
                entity_index=idx,
                total_entities=total_entities,
                extracted_entity_names=extracted_entity_names,
                extracted_relation_pairs=extracted_relation_pairs,
                jaccard_search_threshold=jaccard_search_threshold,
                embedding_name_search_threshold=embedding_name_search_threshold,
                embedding_full_search_threshold=embedding_full_search_threshold,
                base_time=base_time,
                already_versioned_family_ids=already_versioned_family_ids,
                entity_name_to_id=entity_name_to_id,
                prefetched_embedding=_ent_emb,
            )

            if entity:
                processed_entities.append(entity)
                entity_name_to_id[entity.name] = entity.family_id
                entity_name_to_id[extracted_entity['name']] = entity.family_id
            if relations:
                pending_relations.extend(relations)
            if name_mapping:
                entity_name_to_id.update(name_mapping)
            if to_persist:
                entities_to_persist.append(to_persist)
            if on_entity_processed and entity:
                on_entity_processed(entity, entity_name_to_id, relations or [])

        if entities_to_persist:
            self.storage.bulk_save_entities(entities_to_persist)
            # 置信度演化：新版本 = 独立来源印证 → 置信度提升
            _corroborated = list({e.family_id for e in entities_to_persist if e.family_id.startswith("ent_")})
            if _corroborated:
                try:
                    self.storage.adjust_confidence_on_corroboration_batch(_corroborated, source_type="entity")
                except Exception:
                    pass

        return processed_entities, pending_relations, entity_name_to_id

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
                        max_workers: int = 2,
                        prefetched_embeddings: Optional[Tuple[Optional[Any], Optional[Any]]] = None,
                        already_versioned_family_ids: Optional[set] = None) -> Tuple[List[Entity], List[Dict], Dict[str, str]]:
        """多线程处理实体；合并冲突时以数据库中已存在的 family_id 为准。"""
        extracted_entity_names, extracted_relation_pairs, related_entity_names = _preprocess_extraction_context(
            extracted_entities, extracted_relations,
        )

        # 不再过滤孤立实体：所有通过验证的实体都应被处理
        # 孤立实体仍然有价值（如对话中提到的技术选型），丢弃会导致信息损失
        _skipped_orphans = 0
        _orig_indices = list(range(len(extracted_entities)))
        filtered_entities = extracted_entities

        candidate_table = self._build_entity_candidate_table(
            extracted_entities,
            similarity_threshold=similarity_threshold,
            jaccard_search_threshold=jaccard_search_threshold,
            embedding_name_search_threshold=embedding_name_search_threshold,
            embedding_full_search_threshold=embedding_full_search_threshold,
            prefetched_embeddings=prefetched_embeddings,
        )
        total_entities = len(extracted_entities)
        _distill_step = self.llm_client._current_distill_step
        _priority = getattr(self.llm_client._priority_local, 'priority', 5)
        _version_lock = threading.RLock()
        # Extract per-entity full-text embeddings from prefetch
        _prefetched_full_embs = None
        if prefetched_embeddings is not None:
            try:
                _, _full_embs = prefetched_embeddings
                if _full_embs is not None:
                    _prefetched_full_embs = _full_embs
            except Exception:
                pass

        def task(idx: int, extracted_entity: Dict[str, str], orig_idx: int):
            # 将主线程的 distill step 和优先级传播到工作线程（threading.local）
            self.llm_client._current_distill_step = _distill_step
            self.llm_client._priority_local.priority = _priority
            candidates = candidate_table.get(orig_idx, [])
            _ent_emb = None
            if _prefetched_full_embs is not None and orig_idx < len(_prefetched_full_embs):
                try:
                    _ent_emb = np.array(_prefetched_full_embs[orig_idx], dtype=np.float32)
                except Exception:
                    pass
            entity, relations, name_mapping, to_persist = self._process_entity_with_batch_candidates(
                extracted_entity=extracted_entity,
                candidates=candidates,
                episode_id=episode_id,
                similarity_threshold=similarity_threshold,
                episode=episode,
                source_document=source_document,
                context_text=context_text,
                entity_index=idx,
                total_entities=total_entities,
                extracted_entity_names=extracted_entity_names,
                extracted_relation_pairs=extracted_relation_pairs,
                jaccard_search_threshold=jaccard_search_threshold,
                embedding_name_search_threshold=embedding_name_search_threshold,
                embedding_full_search_threshold=embedding_full_search_threshold,
                base_time=base_time,
                already_versioned_family_ids=already_versioned_family_ids,
                _version_lock=_version_lock,
                prefetched_embedding=_ent_emb,
            )
            return (idx, entity, relations, name_mapping, to_persist)

        results: List[Tuple[int, Optional[Entity], List[Dict], Dict[str, str], Optional[Entity]]] = []
        executor = _get_entity_pool(max_workers)
        futures = {
            executor.submit(task, idx, extracted_entity, orig_idx): idx
            for idx, (extracted_entity, orig_idx) in enumerate(
                zip(filtered_entities, _orig_indices), 1
            )
        }
        for future in as_completed(futures):
            results.append(future.result())
        results.sort(key=lambda r: r[0])

        name_to_ids: Dict[str, set] = defaultdict(set)
        all_candidate_eids = set()
        for idx, entity, relations, name_mapping, to_persist in results:
            if name_mapping:
                for name, eid in name_mapping.items():
                    if name and eid:
                        name_to_ids[name].add(eid)
                        all_candidate_eids.add(eid)

        entity_name_to_id: Dict[str, str] = {}
        if all_candidate_eids:
            # resolve_family_ids 返回存在的映射；不存在的 eid 会被过滤
            try:
                _resolve_fn = getattr(self.storage, 'resolve_family_ids', None)
                if _resolve_fn:
                    resolved_map = _resolve_fn(list(all_candidate_eids)) or {}
                    existing_eids = set(resolved_map.keys()) | set(resolved_map.values())
                else:
                    _batch_result = self.storage.get_entities_by_family_ids(list(all_candidate_eids))
                    existing_eids = set(_batch_result.keys())
            except Exception:
                existing_eids = set()
        else:
            existing_eids = set()

        for name, ids in name_to_ids.items():
            # 优先使用数据库中已存在的 family_id（同名实体被多个线程分别匹配到不同候选）
            in_storage = [eid for eid in ids if eid in existing_eids]
            if in_storage:
                entity_name_to_id[name] = in_storage[0]
            else:
                entity_name_to_id[name] = min(ids)

        redirect_pairs = []
        for name, ids in name_to_ids.items():
            canonical_id = entity_name_to_id.get(name)
            if not canonical_id:
                continue
            for eid in ids:
                if eid and eid != canonical_id:
                    redirect_pairs.append((eid, canonical_id))
        if redirect_pairs:
            if hasattr(self.storage, 'register_entity_redirects_batch'):
                self.storage.register_entity_redirects_batch(redirect_pairs)
            else:
                for source_id, canonical_id in redirect_pairs:
                    self.storage.register_entity_redirect(source_id, canonical_id)

        # 对于被合并到 canonical ID 的非 canonical 实体，需要从 results 中修正
        _canonical_ids_to_fetch = set()
        for idx, entity, relations, name_mapping, to_persist in results:
            if entity and entity.family_id != entity_name_to_id.get(entity.name):
                canonical_id = entity_name_to_id.get(entity.name)
                if canonical_id:
                    _canonical_ids_to_fetch.add(canonical_id)
        if _canonical_ids_to_fetch:
            try:
                _canonical_ent_map = self.storage.get_entities_by_family_ids(
                    list(_canonical_ids_to_fetch))
            except Exception:
                _canonical_ent_map = {}
            for i, (idx, entity, relations, name_mapping, to_persist) in enumerate(results):
                if entity and entity.family_id != entity_name_to_id.get(entity.name):
                    canonical_id = entity_name_to_id.get(entity.name)
                    if canonical_id:
                        canonical_entity = _canonical_ent_map.get(canonical_id)
                        if canonical_entity:
                            results[i] = (idx, canonical_entity, relations, name_mapping, to_persist)

        canonical_ids = set(entity_name_to_id.values())
        all_to_persist: List[Entity] = [r[4] for r in results if r[4] is not None]
        entities_to_persist_final = [e for e in all_to_persist if e.family_id in canonical_ids]
        # 按 family_id 去重：同一 family_id 只保留一个待持久化实体（避免批量写入重复版本）
        if entities_to_persist_final:
            _seen_fids = set()
            _deduped = []
            for e in entities_to_persist_final:
                if e.family_id not in _seen_fids:
                    _seen_fids.add(e.family_id)
                    _deduped.append(e)
            if len(_deduped) < len(entities_to_persist_final):
                _dup_count = len(entities_to_persist_final) - len(_deduped)
                if self._entity_tree_log():
                    wprint_info(f"  │  持久化去重: 移除 {_dup_count} 个重复 family_id 的待持久化实体")
                entities_to_persist_final = _deduped
            self.storage.bulk_save_entities(entities_to_persist_final)
            # 置信度演化：新版本 = 独立来源印证 → 置信度提升
            _corro_fids = list({e.family_id for e in entities_to_persist_final})
            if _corro_fids:
                try:
                    self.storage.adjust_confidence_on_corroboration_batch(_corro_fids, source_type="entity")
                except Exception:
                    pass

        processed_entities = [r[1] for r in results if r[1] is not None]
        pending_relations: List[Dict] = []
        for r in results:
            if r[2]:
                pending_relations.extend(r[2])
        if on_entity_processed:
            for r in results:
                if r[1]:
                    on_entity_processed(r[1], entity_name_to_id, r[2] or [])

        return processed_entities, pending_relations, entity_name_to_id
    
    # 名称规范化：委托给共享模块
    _normalize_entity_name_for_matching = staticmethod(normalize_entity_name_for_matching)
    _TITLE_SUFFIXES_RE = _TITLE_SUFFIXES_RE  # re-export from entity_candidates module

    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        return calculate_jaccard_similarity(text1, text2)

    def _alignment_guard(
        self, name_a: str, content_a: str, name_b: str, content_b: str,
        *, name_match_type: str = "none", require_content: bool = True,
    ) -> Optional[Tuple[str, float]]:
        """Three-way alignment check. Returns (verdict, confidence) if reject, None if same (proceed)."""
        if not hasattr(self.llm_client, 'judge_entity_alignment'):
            return None
        if require_content and not content_b:
            return None
        # Trivial content_b (e.g. "是", "no") carries no alignment signal — skip LLM call
        if content_b is not None and len(content_b) < 3 and not require_content:
            return ("different", 0.9)
        # Check instance cache (keyed by name + content prefix for bounded size)
        _ca = content_a or ""
        _cb = content_b or ""
        _cache_key = (name_a, _ca[:200] if len(_ca) > 200 else _ca, name_b, _cb[:200] if len(_cb) > 200 else _cb)
        if _cache_key in self._alignment_guard_cache:
            return self._alignment_guard_cache[_cache_key]
        result = self.llm_client.judge_entity_alignment(
            name_a, content_a, name_b, content_b, name_match_type=name_match_type,
        )
        verdict = result.get("verdict", "uncertain")
        confidence = result.get("confidence", 0.5)
        _dbg_struct("alignment_guard",
                    name_a=name_a, name_b=name_b,
                    content_a_snippet=(content_a or "")[:80],
                    content_b_snippet=(content_b or "")[:80],
                    verdict=verdict, confidence=f"{confidence:.2f}",
                    name_match_type=name_match_type)
        if verdict in ("different", "uncertain"):
            ans = (verdict, confidence)
        else:
            ans = None
        # LRU eviction: remove oldest entry when cache exceeds limit
        if len(self._alignment_guard_cache) > 500:
            self._alignment_guard_cache.popitem(last=False)
        self._alignment_guard_cache[_cache_key] = ans
        self._alignment_guard_cache.move_to_end(_cache_key)
        return ans

    @staticmethod
    def _cosine_similarity(embedding1, embedding2) -> float:
        return cosine_similarity(embedding1, embedding2)

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
        """Check if top candidate is an alias and merge after LLM verification.

        Gate: EITHER name Jaccard >= 0.3 OR embedding(name+content) >= 0.5.
        Then checks content-mention alias evidence, and finally verifies with
        _alignment_guard before merging.

        Returns a result tuple if alias verified, None otherwise.
        """
        if not candidates or not context_text:
            return None

        top = candidates[0]
        cand_name = top.get("name", "")
        cand_content = top.get("content", "")

        # Skip if exact name match (already handled by fast path above)
        if cand_name == entity_name:
            return None

        # Gate: name Jaccard OR embedding similarity must pass threshold.
        # Either signal independently justifies trying LLM verification.
        _name_jaccard = self._calculate_jaccard_similarity(entity_name, cand_name)
        _dense_score = top.get("dense_score", 0)
        _lexical_score = top.get("lexical_score", 0)

        _jaccard_ok = _name_jaccard >= 0.3
        _embedding_ok = _dense_score >= 0.5

        if not _jaccard_ok and not _embedding_ok:
            return None

        # Check alias evidence
        is_alias = False
        alias_reason = ""

        # Check 1: Candidate content mentions the extracted name
        # e.g., 刘备 content: "刘备,字玄德" → mentions "玄德"
        if entity_name in cand_content and len(entity_name) >= 2:
            is_alias = True
            alias_reason = f"候选内容提及'{entity_name}'"

        # Check 2: Extracted content mentions the candidate name
        if not is_alias and cand_name in entity_content and len(cand_name) >= 2:
            is_alias = True
            alias_reason = f"当前内容提及'{cand_name}'"

        if not is_alias:
            return None

        # Alias evidence found — verify with _alignment_guard before committing.
        # Content-mention alone is insufficient: "打听" appearing as a verb in
        # "周瑞家的" content is not alias evidence.
        _guard = self._alignment_guard(
            entity_name, entity_content, cand_name, cand_content or "",
            name_match_type=top.get("name_match_type", "none"),
        )
        if _guard:
            _guard_verdict, _guard_conf = _guard
            _dbg_struct("alias_merge_guard_reject",
                        entity_name=entity_name, cand_name=cand_name,
                        alias_reason=alias_reason,
                        name_jaccard=f"{_name_jaccard:.3f}",
                        dense_score=f"{_dense_score:.3f}",
                        guard_verdict=_guard_verdict, guard_conf=f"{_guard_conf:.2f}")
            if self._entity_tree_log():
                wprint_info(f"  │  别名合并被 guard 拒绝: '{entity_name}' ≁ '{cand_name}' (verdict={_guard_verdict}, conf={_guard_conf:.2f})")
            return None

        # Alias verified by guard — proceed with merge.
        _combined = top.get("combined_score", 0)
        match_existing_id = top.get("family_id", "")
        if not match_existing_id:
            return None

        # Handle within-batch alias (__batch_ prefixed IDs)
        if match_existing_id.startswith("__batch_"):
            batch_name = top.get("name", "")
            if batch_name:
                # Resolve via entity_name_to_id dict (populated incrementally)
                resolved_id = (entity_name_to_id or {}).get(batch_name)
                if resolved_id:
                    match_existing_id = resolved_id
                else:
                    return None  # Not yet resolved, can't merge
            else:
                return None

        latest_entity = top.get("entity") or self.storage.get_entity_by_family_id(match_existing_id)
        if not latest_entity:
            return None

        if self._entity_tree_log():
            wprint_info(f"  │  别名合并: '{entity_name}' = '{cand_name}' ({alias_reason}, jaccard={_name_jaccard:.2f}, emb={_dense_score:.2f}, guard=passed)")

        # Use the longer/more standard name as the merged name
        merged_name = cand_name  # Default: keep existing entity's name
        # Heuristic: if the existing entity's name is a full name and the new one is an alias, keep full name
        if len(entity_name) > len(cand_name):
            merged_name = entity_name
        # If the candidate's content explicitly states the entity's name as an alias
        # (e.g., "刘备,字玄德"), keep the first name (the actual name)
        if cand_content and entity_name in cand_content:
            # The candidate is likely the full-name entity, keep its name
            merged_name = cand_name

        # Prevent same-window duplicate versioning
        if already_versioned_family_ids and latest_entity.family_id in already_versioned_family_ids:
            if self._entity_tree_log():
                wprint_info(f"  │  别名合并: 同窗口复用 {latest_entity.family_id}")
            return latest_entity, [], {
                entity_name: latest_entity.family_id,
                latest_entity.name: latest_entity.family_id,
            }, None

        # Merge content (fast-forward)
        merged_content = self._merge_two_contents(
            latest_entity, entity_name, entity_content,
            source_document, episode_id, base_time=base_time,
        )

        entity_version = self._build_entity_version(
            latest_entity.family_id,
            merged_name,
            merged_content,
            episode_id,
            source_document,
            base_time=base_time,
        )
        self._mark_versioned(latest_entity.family_id, already_versioned_family_ids, _version_lock)

        if self._entity_tree_log():
            wprint_info(f"  │  别名合并: '{entity_name}' → {latest_entity.family_id} (merged_name='{merged_name}')")

        return entity_version, [], {
            entity_name: latest_entity.family_id,
            merged_name: latest_entity.family_id,
        }, entity_version

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
            # Fast path: old is a substring of new → new already contains old
            if old_content in new_content:
                return new_content
            # Fast path: new starts with old → incremental knowledge growth
            if new_content.startswith(old_content):
                return new_content
            # Fast path: high content overlap → use the longer content (avoids LLM call)
            # For same-name entities with very similar content, the longer version
            # typically subsumes the shorter one. Character bigram Jaccard >= 0.55
            # captures "same concept, different wording" cases.
            _min_len = min(len(old_content), len(new_content))
            if _min_len > 15:
                _old_bigrams = set(zip(old_content, old_content[1:]))
                _new_bigrams = set(zip(new_content, new_content[1:]))
                if _old_bigrams and _new_bigrams:
                    _jaccard = len(_old_bigrams & _new_bigrams) / len(_old_bigrams | _new_bigrams)
                    if _jaccard >= 0.55:
                        return max(old_content, new_content, key=len)
            return self.llm_client.merge_multiple_entity_contents(
                [old_entity.content, entity_content],
                entity_sources=[old_entity.source_document, source_document],
                entity_names=[old_entity.name, entity_name],
            )
        elif old_content == new_content:
            return old_entity.content or entity_content
        else:
            return entity_content

    def _process_entity_with_batch_candidates(self,
                                     extracted_entity: Dict[str, str],
                                     candidates: List[Dict[str, Any]],
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
                                     entity_name_to_id: Optional[Dict[str, str]] = None,
                                     prefetched_embedding: Optional[Any] = None) -> Tuple[Optional[Entity], List[Dict], Dict[str, str], Optional[Entity]]:
        """批量候选 + 批量裁决主路径，低置信度时回退旧逻辑。

        Args:
            already_versioned_family_ids: 已创建版本的 family_id 集合，防止同窗口重复版本化。
            _version_lock: 可选线程锁，保护 already_versioned_family_ids 的并发访问。
        """
        entity_name = extracted_entity["name"]
        entity_content = extracted_entity["content"]
        _t_entity_start = time.monotonic()
        if self._entity_tree_log() and total_entities > 0:
            wprint_info(f"  ├─ 处理实体 [{entity_index}/{total_entities}]: {entity_name}")

        # ── Alignment trace: entity start ──
        _dbg_struct("entity_start",
                    name=entity_name,
                    content_snippet=(entity_content or "")[:120],
                    episode_id=episode_id,
                    n_candidates=len(candidates) if candidates else 0,
                    already_versioned_count=len(already_versioned_family_ids) if already_versioned_family_ids else 0)

        if not candidates:
            new_entity = self._build_new_entity(entity_name, entity_content, episode_id, source_document, base_time=base_time)
            if self._entity_tree_log():
                wprint_info(f"  │  未找到候选实体，批量路径创建新实体: {new_entity.family_id}")
            _dbg_struct("decision_no_candidates",
                        name=entity_name, new_family_id=new_entity.family_id)
            wprint_info(f"[entity_timing] '{entity_name}' no_candidates → {time.monotonic() - _t_entity_start:.1f}s")
            self._mark_versioned(new_entity.family_id, already_versioned_family_ids, _version_lock)
            return new_entity, [], {entity_name: new_entity.family_id, new_entity.name: new_entity.family_id}, new_entity

        if self._entity_tree_log():
            wprint_info(f"  │  批量候选生成: {len(candidates)} 个")

        # ── Alignment trace: candidate summary ──
        _cand_summary = "; ".join(
            f"{c.get('name','?')}(fid={c.get('family_id','?')},score={c.get('combined_score',0):.3f},safe={c.get('merge_safe',True)},type={c.get('name_match_type','?')})"
            for c in candidates[:5]
        )
        _dbg_struct("candidates_top",
                    name=entity_name, top_n=min(len(candidates), 5),
                    candidates=_cand_summary)

        # ---- Fix 2a: 精确名称匹配 + 高embedding相似度 → 同窗口复用/跨窗口创建版本，跳过LLM ----
        top = candidates[0]
        _exact_match_skip_guard = (
            top["name"] == entity_name
            and top.get("combined_score", 0) >= 0.85
            and top.get("merge_safe", True)
            and top.get("name_match_type", "none") in ("exact", "substring")
        )
        if (top["name"] == entity_name
            and top.get("combined_score", 0) >= 0.85
            and top.get("merge_safe", True)):
            # 优先使用候选中已携带的实体对象，避免重复 DB 查询
            latest = top.get("entity") or self.storage.get_entity_by_family_id(top["family_id"])
            if latest:
                # Skip alignment guard for merge_safe exact/substring matches — the candidate
                # table already confirmed strong name + embedding similarity. The guard adds
                # ~20-40s LLM call per entity with near-zero value for these high-confidence cases.
                if not _exact_match_skip_guard:
                    # ---- Three-way alignment guard for exact name matches (Phase 4) ----
                    # Even with exact name match, check if content describes a different entity
                    # This catches "张伟(教授)" vs "张伟(CEO)" cases
                    _guard = self._alignment_guard(
                        entity_name, entity_content, latest.name, latest.content or "",
                        name_match_type=top.get("name_match_type", "none"),
                    )
                    if _guard:
                        _align_verdict, _align_confidence = _guard
                        if self._entity_tree_log():
                            _label = "同名但不同实体" if _align_verdict == "different" else "保守策略"
                            wprint_info(f"  │  快捷路径三值对齐: verdict={_align_verdict} (conf={_align_confidence:.2f}), {_label}→新建")
                        _dbg_struct("decision_exact_match_guard_reject",
                                    name=entity_name, matched_name=top.get("name","?"),
                                    matched_fid=top.get("family_id","?"),
                                    verdict=_align_verdict, guard_conf=f"{_align_confidence:.2f}",
                                    action="create_new")
                        new_entity = self._build_new_entity(entity_name, entity_content, episode_id, source_document, base_time=base_time)
                        self._mark_versioned(new_entity.family_id, already_versioned_family_ids, _version_lock)
                        return new_entity, [], {entity_name: new_entity.family_id, new_entity.name: new_entity.family_id}, new_entity
                        # verdict == "same" → proceed with fast path merge

                # 同窗口内已有版本 → 直接复用，避免同窗口重复版本化（加锁防竞态）
                def _fast_path_create_version():
                    """在锁保护下检查+创建版本，防止并行线程重复版本化。"""
                    if already_versioned_family_ids and latest.family_id in already_versioned_family_ids:
                        if self._entity_tree_log():
                            wprint_info(f"  │  快捷路径：同窗口复用 {latest.family_id}")
                        _dbg_struct("decision_exact_same_window_reuse",
                                    name=entity_name, family_id=latest.family_id,
                                    action="reuse_existing_version")
                        return latest, [], {entity_name: latest.family_id, latest.name: latest.family_id}, None

                    # 内容完全相同 → 直接复用旧 content（零 LLM 开销）
                    old_content = (latest.content or "").strip()
                    new_content = entity_content.strip()
                    if old_content and old_content == new_content:
                        entity_version = self._build_entity_version(
                            latest.family_id, entity_name, latest.content,
                            episode_id, source_document, base_time=base_time,
                        )
                        self._mark_versioned(latest.family_id, already_versioned_family_ids, _version_lock)
                        if self._entity_tree_log():
                            wprint_info(f"  │  快捷路径：内容相同，直接复用 {latest.family_id}")
                        _dbg_struct("decision_exact_content_identical",
                                    name=entity_name, family_id=latest.family_id,
                                    action="reuse_content_new_version")
                        return entity_version, [], {entity_name: latest.family_id, latest.name: latest.family_id}, entity_version

                    # 内容有差异 → 增量合并（git-like editing）
                    merged_content = self._merge_two_contents(
                        latest, entity_name, entity_content,
                        source_document, episode_id, base_time=base_time,
                    )
                    final_name = entity_name

                    entity_version = self._build_entity_version(
                        latest.family_id, final_name, merged_content,
                        episode_id, source_document, base_time=base_time,
                    )
                    self._mark_versioned(latest.family_id, already_versioned_family_ids, _version_lock)
                    if self._entity_tree_log():
                        wprint_info(f"  │  快捷路径：增量合并新版本 {latest.family_id}")
                    _dbg_struct("decision_exact_incremental_merge",
                                name=entity_name, family_id=latest.family_id,
                                action="merge_and_new_version")
                    return entity_version, [], {entity_name: latest.family_id, latest.name: latest.family_id}, entity_version

                if _version_lock:
                    with _version_lock:
                        _r = _fast_path_create_version()
                        wprint_info(f"[entity_timing] '{entity_name}' exact_match_fast → {time.monotonic() - _t_entity_start:.1f}s")
                        return _r
                else:
                    _r = _fast_path_create_version()
                    wprint_info(f"[entity_timing] '{entity_name}' exact_match_fast → {time.monotonic() - _t_entity_start:.1f}s")
                    return _r

        # ---- Fix 2b: 全部候选低相似度 → 直接新建，跳过LLM ----
        if candidates[0].get("combined_score", 0) < 0.4:
            if self._entity_tree_log():
                wprint_info(f"  │  快捷路径：候选相似度过低({candidates[0].get('combined_score', 0):.2f})→新建")
            _dbg_struct("decision_low_similarity",
                        name=entity_name, best_score=f"{candidates[0].get('combined_score', 0):.3f}",
                        best_name=candidates[0].get('name', '?'), action="create_new")
            new_entity = self._build_new_entity(entity_name, entity_content, episode_id, source_document, base_time=base_time)
            if new_entity:
                self._mark_versioned(new_entity.family_id, already_versioned_family_ids, _version_lock)
            if new_entity:
                wprint_info(f"[entity_timing] '{entity_name}' low_similarity(score<{0.4}) → {time.monotonic() - _t_entity_start:.1f}s")
                return new_entity, [], {entity_name: new_entity.family_id, new_entity.name: new_entity.family_id}, new_entity

        # ---- Context-based alias bypass (skip LLM for obvious aliases) ----
        # Detects alias pairs like 玄德/刘备, 曹操/丞相, 使君/刘备 by checking:
        # 1. Candidate content mentions the extracted name (e.g., "刘备,字玄德")
        # 2. Extracted entity content mentions the candidate name
        # 3. Both names appear as dialogue subjects in context text (A曰, B曰)
        # When any condition is met AND embedding similarity >= 0.5, bypass LLM.
        alias_merged = self._try_context_alias_merge(
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
        if alias_merged is not None:
            _dbg_struct("decision_alias_merge",
                        name=entity_name, matched_name=candidates[0].get('name', '?') if candidates else '?',
                        matched_fid=candidates[0].get('family_id', '?') if candidates else '?',
                        combined_score=f"{candidates[0].get('combined_score', 0):.3f}" if candidates else "0",
                        action="alias_merge_guard_verified")
            wprint_info(f"[entity_timing] '{entity_name}' alias_merge → {time.monotonic() - _t_entity_start:.1f}s")
            return alias_merged

        batch_result = self.llm_client.resolve_entity_candidates_batch(
            {
                "family_id": "NEW_ENTITY",
                "name": entity_name,
                "content": entity_content,
                "source_document": _doc_basename(source_document),
                "version_count": 0,
            },
            candidates,
            context_text=context_text,
        )
        confidence = float(batch_result.get("confidence", 0.0) or 0.0)
        update_mode = batch_result.get("update_mode") or "reuse_existing"

        # ── Alignment trace: batch LLM decision ──
        _dbg_struct("batch_llm_decision",
                    name=entity_name, confidence=f"{confidence:.3f}",
                    update_mode=update_mode,
                    match_existing_id=batch_result.get("match_existing_id", ""),
                    merged_name=batch_result.get("merged_name", ""),
                    n_relations=len(batch_result.get("relations_to_create", []) or []))
        # Trust batch_resolve decisions directly — update_mode is the LLM's judgment,
        # confidence is just self-reported noise. create_new = no match = fast path.
        _safe_create_new = (update_mode == "create_new")

        # Lightweight path: when batch gives a merge/reuse decision with moderate confidence,
        # verify with a single _alignment_guard instead of falling back to the
        # full sequential path.
        _moderate_conf_merge = (
            update_mode in ("merge_into_latest", "reuse_existing")
            and confidence < self.batch_resolution_confidence_threshold
            and self.batch_resolution_enabled
        )
        if _moderate_conf_merge:
            match_existing_id = (batch_result.get("match_existing_id") or "").strip()
            if match_existing_id:
                _cand_by_fid = {c.get("family_id"): c for c in candidates if c.get("family_id")}
                _matched_cand = _cand_by_fid.get(match_existing_id)
                _guard = self._alignment_guard(
                    entity_name, entity_content,
                    _matched_cand.get("name", "") if _matched_cand else "",
                    _matched_cand.get("content", "") if _matched_cand else "",
                    name_match_type=_matched_cand.get("name_match_type", "none") if _matched_cand else "none",
                    require_content=False,
                )
                if _guard:
                    if self._entity_tree_log():
                        wprint_info(f"  │  中等置信合并被 alignment_guard 拒绝 (conf={confidence:.2f})→新建")
                    _dbg_struct("decision_moderate_guard_reject",
                                name=entity_name, match_fid=match_existing_id,
                                batch_conf=f"{confidence:.2f}",
                                guard_verdict=_guard[0], guard_conf=f"{_guard[1]:.2f}",
                                action="create_new")
                    new_entity = self._build_new_entity(entity_name, entity_content, episode_id, source_document, base_time=base_time, confidence=confidence)
                    self._mark_versioned(new_entity.family_id, already_versioned_family_ids, _version_lock)
                    wprint_info(f"[entity_timing] '{entity_name}' moderate_guard_reject → {time.monotonic() - _t_entity_start:.1f}s")
                    return new_entity, [], {entity_name: new_entity.family_id, new_entity.name: new_entity.family_id}, new_entity
                # Guard passed — proceed with the merge/reuse decision below
                if self._entity_tree_log():
                    wprint_info(f"  │  中等置信合并通过 guard (conf={confidence:.2f}), 跳过完整 fallback")
            else:
                # No match_existing_id — treat as create_new
                new_entity = self._build_new_entity(entity_name, entity_content, episode_id, source_document, base_time=base_time, confidence=confidence)
                self._mark_versioned(new_entity.family_id, already_versioned_family_ids, _version_lock)
                wprint_info(f"[entity_timing] '{entity_name}' moderate_no_match → {time.monotonic() - _t_entity_start:.1f}s")
                return new_entity, [], {entity_name: new_entity.family_id, new_entity.name: new_entity.family_id}, new_entity

        _need_full_fallback = (not self.batch_resolution_enabled) or update_mode == "fallback"
        if _need_full_fallback:
            _dbg_struct("decision_fallback",
                        name=entity_name, batch_conf=f"{confidence:.2f}",
                        update_mode=update_mode,
                        reason="disabled" if not self.batch_resolution_enabled else
                               "fallback_mode" if update_mode == "fallback" else "low_confidence",
                        action="sequential_fallback")
            if self._entity_tree_log():
                wprint_info(f"  │  批量裁决置信度不足，回退到旧逻辑 (confidence={confidence:.2f})")
            entity, relations, name_mapping = self._process_entity_sequential_fallback(
                extracted_entity,
                episode_id,
                similarity_threshold,
                episode,
                source_document,
                context_text,
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
                prebuilt_candidates=candidates,
            )
            wprint_info(f"[entity_timing] '{entity_name}' fallback_sequential(conf={confidence:.2f}) → {time.monotonic() - _t_entity_start:.1f}s")
            return entity, relations, name_mapping, None

        wprint_info(f"[entity_timing] '{entity_name}' batch_resolve(conf={confidence:.2f},{update_mode}) → {time.monotonic() - _t_entity_start:.1f}s (past fallback check)")

        # Pre-build family_id → candidate dict for O(1) lookups (avoids 4× linear scans)
        _cand_by_fid = {c.get("family_id"): c for c in candidates if c.get("family_id")}
        relations_to_create: List[Dict] = []
        for relation in batch_result.get("relations_to_create", []) or []:
            candidate = _cand_by_fid.get(relation.get("family_id"))
            if not candidate:
                continue
            relation_content = (relation.get("relation_content") or "").strip()
            if not relation_content:
                continue
            relations_to_create.append({
                "entity1_name": entity_name,
                "entity2_name": candidate.get("name", ""),
                "content": relation_content,
                "relation_type": "alias" if ("别名" in relation_content or "简称" in relation_content or "称呼" in relation_content) else "normal",
            })

        match_existing_id = (batch_result.get("match_existing_id") or "").strip()
        # Handle within-batch alias matches (__batch_ prefixed IDs)
        if match_existing_id.startswith("__batch_"):
            batch_idx_str = match_existing_id[len("__batch_"):]
            try:
                batch_idx = int(batch_idx_str)
            except ValueError:
                batch_idx = -1
            if batch_idx >= 0:
                matched_candidate = _cand_by_fid.get(match_existing_id)
                if matched_candidate:
                    batch_name = matched_candidate.get("name", "")
                    # Resolve via entity_name_to_id dict (populated incrementally during sequential processing)
                    if batch_name:
                        resolved_id = (entity_name_to_id or {}).get(batch_name)
                        if resolved_id:
                            match_existing_id = resolved_id
                            if self._entity_tree_log():
                                wprint_info(f"  │  Within-batch alias resolved: __batch_{batch_idx} '{batch_name}' → {match_existing_id}")
                        else:
                            # Entity not yet resolved — create new entity, let the other entity merge later
                            match_existing_id = ""
                            if self._entity_tree_log():
                                wprint_info(f"  │  Within-batch alias: '{batch_name}' not yet in entity_name_to_id, creating new entity")
                    else:
                        match_existing_id = ""
        # 合并安全检查：如果匹配的候选 merge_safe=False（仅名字字面匹配），
        # 不允许合并或复用，改为创建新实体
        # 但是：如果候选内容提及当前实体名称（别名证据），允许合并
        if match_existing_id:
            matched_candidate = _cand_by_fid.get(match_existing_id)
            if matched_candidate and not matched_candidate.get("merge_safe", True):
                # Check for content-mention evidence (alias signal)
                mc_name = matched_candidate.get("name", "")
                mc_content = matched_candidate.get("content", "")
                has_content_mention = (
                    (entity_name in mc_content and len(entity_name) >= 2)
                    or (mc_name in entity_content and len(mc_name) >= 2)
                )
                if not has_content_mention:
                    if update_mode in ("merge_into_latest", "reuse_existing"):
                        if self._entity_tree_log():
                            wprint_info("  │  批量裁决: merge_safe=False，禁止合并/复用，创建新实体")
                        _dbg_struct("decision_merge_unsafe_reject",
                                    name=entity_name, match_fid=match_existing_id,
                                    matched_name=mc_name, update_mode=update_mode,
                                    action="create_new")
                        new_entity = self._build_new_entity(entity_name, entity_content, episode_id, source_document, base_time=base_time, confidence=confidence)
                        self._mark_versioned(new_entity.family_id, already_versioned_family_ids, _version_lock)
                        wprint_info(f"[entity_timing] '{entity_name}' merge_unsafe_reject → {time.monotonic() - _t_entity_start:.1f}s")
                        return new_entity, relations_to_create, {
                            entity_name: new_entity.family_id,
                            new_entity.name: new_entity.family_id,
                        }, new_entity
            # ---- Three-way alignment verification (Phase 4) ----
            # Only verify when batch confidence is below threshold — high-confidence
            # batch results already made a well-informed same/different decision.
            _batch_conf = confidence  # reuse cached value from line 920
            if _batch_conf < 0.8:
                _matched_cand = _cand_by_fid.get(match_existing_id)
                _guard = self._alignment_guard(
                    entity_name, entity_content,
                    _matched_cand.get("name", "") if _matched_cand else "",
                    _matched_cand.get("content", "") if _matched_cand else "",
                    name_match_type=_matched_cand.get("name_match_type", "none") if _matched_cand else "none",
                    require_content=False,
                )
                if _guard:
                    _align_verdict, _align_confidence = _guard
                    if self._entity_tree_log():
                        wprint_info(f"  │  三值对齐: verdict={_align_verdict} (conf={_align_confidence:.2f}), 拒绝合并→新建")
                    _dbg_struct("decision_guard_reject",
                                name=entity_name, match_fid=match_existing_id,
                                batch_conf=f"{_batch_conf:.2f}",
                                guard_verdict=_align_verdict,
                                guard_conf=f"{_align_confidence:.2f}",
                                action="create_new")
                    new_entity = self._build_new_entity(entity_name, entity_content, episode_id, source_document, base_time=base_time, confidence=confidence)
                    self._mark_versioned(new_entity.family_id, already_versioned_family_ids, _version_lock)
                    wprint_info(f"[entity_timing] '{entity_name}' guard_reject(conf={_batch_conf:.2f}) → {time.monotonic() - _t_entity_start:.1f}s")
                    return new_entity, relations_to_create, {
                        entity_name: new_entity.family_id,
                        new_entity.name: new_entity.family_id,
                    }, new_entity
            elif self._entity_tree_log() and _batch_conf >= 0.8:
                wprint_info(f"  │  三值对齐: batch conf={_batch_conf:.2f} >= 0.8, 跳过验证")

            latest_entity = matched_candidate.get("entity") if matched_candidate else None
            if not latest_entity:
                # Try redirect resolution first
                resolved_id = self.storage.resolve_family_id(match_existing_id)
                if resolved_id and resolved_id != match_existing_id:
                    latest_entity = self.storage.get_entity_by_family_id(resolved_id)
            if not latest_entity:
                # Entity not found (merged/deleted) — create new directly instead of
                # expensive fallback. Register redirect so future lookups find the new entity.
                if self._entity_tree_log():
                    wprint_info(f"  │  批量裁决命中的实体不存在: {match_existing_id}，直接新建")
                new_entity = self._build_new_entity(entity_name, entity_content, episode_id, source_document, base_time=base_time, confidence=confidence)
                self._mark_versioned(new_entity.family_id, already_versioned_family_ids, _version_lock)
                try:
                    self.storage.register_entity_redirect(match_existing_id, new_entity.family_id)
                except Exception:
                    pass
                wprint_info(f"[entity_timing] '{entity_name}' entity_not_found→create_new(conf={confidence:.2f}) → {time.monotonic() - _t_entity_start:.1f}s")
                return new_entity, relations_to_create, {entity_name: new_entity.family_id, new_entity.name: new_entity.family_id}, new_entity

            if update_mode == "merge_into_latest":
                # 防止同窗口内重复版本化（加锁防竞态）
                def _batch_merge_create_version():
                    if already_versioned_family_ids and match_existing_id in already_versioned_family_ids:
                        if self._entity_tree_log():
                            wprint_info(f"  │  批量裁决: family_id {match_existing_id} 已在本次处理中创建版本，复用已有实体")
                        _dbg_struct("decision_batch_merge_same_window_reuse",
                                    name=entity_name, family_id=match_existing_id,
                                    action="reuse_existing_version")
                        return latest_entity, relations_to_create, {
                            entity_name: latest_entity.family_id,
                            latest_entity.name: latest_entity.family_id,
                        }, None

                    merged_name = (batch_result.get("merged_name") or latest_entity.name).strip()

                    # 增量合并：使用专用 merge 函数，而非 batch 裁决的 merged_content
                    # 确保 CONTENT_MERGE_REQUIREMENTS 的六条增量规则始终生效
                    merged_content = self._merge_two_contents(
                        latest_entity, entity_name, entity_content,
                        source_document, episode_id, base_time=base_time,
                    )

                    # 始终创建新版本（每个 episode 提及的概念都版本化）
                    entity_version = self._build_entity_version(
                        latest_entity.family_id,
                        merged_name,
                        merged_content,
                        episode_id,
                        source_document,
                        base_time=base_time,
                    )
                    self._mark_versioned(latest_entity.family_id, already_versioned_family_ids, _version_lock)
                    if self._entity_tree_log():
                        wprint_info(f"  │  批量裁决: 增量合并到已有实体 {latest_entity.family_id} 并生成新版本")
                    _dbg_struct("decision_batch_merge",
                                name=entity_name, family_id=latest_entity.family_id,
                                merged_name=merged_name,
                                confidence=f"{confidence:.2f}",
                                action="merge_incremental_new_version")
                    return entity_version, relations_to_create, {
                        entity_name: latest_entity.family_id,
                        entity_version.name: latest_entity.family_id,
                    }, entity_version

                if _version_lock:
                    with _version_lock:
                        _r = _batch_merge_create_version()
                        wprint_info(f"[entity_timing] '{entity_name}' batch_merge(conf={confidence:.2f}) → {time.monotonic() - _t_entity_start:.1f}s")
                        return _r
                else:
                    _r = _batch_merge_create_version()
                    wprint_info(f"[entity_timing] '{entity_name}' batch_merge(conf={confidence:.2f}) → {time.monotonic() - _t_entity_start:.1f}s")
                    return _r

            # reuse_existing: 跨窗口再次遇到已知实体 → 创建新版本（同窗口内已有版本则复用）
            # 使用锁保护 check+create，防止并行线程重复版本化（TOCTOU 竞态）
            def _batch_reuse_create_version():
                if already_versioned_family_ids and latest_entity.family_id in already_versioned_family_ids:
                    if self._entity_tree_log():
                        wprint_info(f"  │  批量裁决: 同窗口复用已有实体 {latest_entity.family_id}")
                    _dbg_struct("decision_batch_reuse_same_window",
                                name=entity_name, family_id=latest_entity.family_id,
                                action="reuse_existing_version")
                    return latest_entity, relations_to_create, {
                        entity_name: latest_entity.family_id,
                        latest_entity.name: latest_entity.family_id,
                    }, None
                # 始终创建新版本（每个 episode 提及的概念都版本化）
                # reuse_existing: 保留已有实体的名称和内容（新信息已被已有内容覆盖）
                entity_version = self._build_entity_version(
                    latest_entity.family_id, latest_entity.name, latest_entity.content or entity_content,
                    episode_id, source_document, base_time=base_time,
                )
                self._mark_versioned(latest_entity.family_id, already_versioned_family_ids, _version_lock)
                if self._entity_tree_log():
                    wprint_info(f"  │  批量裁决: 跨窗口创建新版本 {latest_entity.family_id}")
                _dbg_struct("decision_batch_reuse_cross_window",
                            name=entity_name, family_id=latest_entity.family_id,
                            confidence=f"{confidence:.2f}",
                            action="reuse_existing_new_version")
                return entity_version, relations_to_create, {
                    entity_name: latest_entity.family_id,
                    latest_entity.name: latest_entity.family_id,
                }, entity_version

            if _version_lock:
                with _version_lock:
                    _r = _batch_reuse_create_version()
                    wprint_info(f"[entity_timing] '{entity_name}' batch_reuse(conf={confidence:.2f}) → {time.monotonic() - _t_entity_start:.1f}s")
                    return _r
            else:
                _r = _batch_reuse_create_version()
                wprint_info(f"[entity_timing] '{entity_name}' batch_reuse(conf={confidence:.2f}) → {time.monotonic() - _t_entity_start:.1f}s")
                return _r

        merged_name = (batch_result.get("merged_name") or entity_name).strip() or entity_name
        new_entity = self._build_new_entity(merged_name, entity_content, episode_id, source_document, base_time=base_time, confidence=confidence)
        # 标记新实体的 family_id 已创建版本
        self._mark_versioned(new_entity.family_id, already_versioned_family_ids, _version_lock)
        if self._entity_tree_log():
            wprint_info(f"  │  批量裁决: 创建新实体 '{entity_name}' {new_entity.family_id} (had {len(candidates)} cands, best={candidates[0].get('name','?')} score={candidates[0].get('combined_score',0):.2f}, LLM chose create_new conf={confidence:.2f})")
        _dbg_struct("decision_batch_create_new",
                    name=entity_name, new_family_id=new_entity.family_id,
                    confidence=f"{confidence:.2f}",
                    best_candidate=candidates[0].get('name', '?'),
                    best_score=f"{candidates[0].get('combined_score', 0):.3f}",
                    action="create_new")
        wprint_info(f"[entity_timing] '{entity_name}' batch_create_new(conf={confidence:.2f}) → {time.monotonic() - _t_entity_start:.1f}s")
        return new_entity, relations_to_create, {
            entity_name: new_entity.family_id,
            new_entity.name: new_entity.family_id,
        }, new_entity

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
        """混合搜索候选实体：Jaccard + Embedding（name / name+content），去重合并后返回。

        3-4 个搜索查询并行执行，结果去重后返回。
        """
        jaccard_threshold = jaccard_search_threshold if jaccard_search_threshold is not None else min(similarity_threshold, 0.6)
        embedding_name_threshold = embedding_name_search_threshold if embedding_name_search_threshold is not None else min(similarity_threshold, 0.6)
        embedding_full_threshold = embedding_full_search_threshold if embedding_full_search_threshold is not None else min(similarity_threshold, 0.6)

        snippet_len = self.llm_client.effective_entity_snippet_length()

        # Build search tasks — all independent, can run in parallel
        def _search_jaccard():
            return self.storage.search_entities_by_similarity(
                entity_name, query_content=None, threshold=jaccard_threshold,
                max_results=self.max_similar_entities,
                content_snippet_length=snippet_len,
                text_mode="name_only", similarity_method="jaccard"
            )

        # 补充搜索：去称谓核心名称
        _core_name = self._TITLE_SUFFIXES_RE.sub('', entity_name).strip()
        _has_title_suffix = _core_name != entity_name and len(_core_name) >= 2

        def _search_core_jaccard():
            return self.storage.search_entities_by_similarity(
                _core_name, query_content=None, threshold=jaccard_threshold,
                max_results=self.max_similar_entities,
                content_snippet_length=snippet_len,
                text_mode="name_only", similarity_method="jaccard"
            )

        def _search_name_embedding():
            return self.storage.search_entities_by_similarity(
                entity_name, query_content=None, threshold=embedding_name_threshold,
                max_results=self.max_similar_entities,
                content_snippet_length=snippet_len,
                text_mode="name_only", similarity_method="embedding"
            )

        def _search_full_embedding():
            return self.storage.search_entities_by_similarity(
                entity_name, query_content=entity_content, threshold=embedding_full_threshold,
                max_results=self.max_similar_entities,
                content_snippet_length=snippet_len,
                text_mode="name_and_content", similarity_method="embedding"
            )

        # Execute searches in parallel
        search_fns = [_search_jaccard, _search_name_embedding, _search_full_embedding]
        if _has_title_suffix:
            search_fns.append(_search_core_jaccard)

        if len(search_fns) > 1:
            pool = _get_entity_pool(len(search_fns))
            futures = [pool.submit(fn) for fn in search_fns]
            search_results = [fut.result() for fut in futures]
        else:
            search_results = [fn() for fn in search_fns]

        # Unpack results (core_jaccard is last if present)
        candidates_jaccard = search_results[0]
        candidates_name_embedding = search_results[1]
        candidates_full_embedding = search_results[2]
        candidates_core_jaccard = search_results[3] if _has_title_suffix else []

        if self._entity_tree_log():
            wprint_info(f"  │  ├─ Jaccard搜索（name_only）: {len(candidates_jaccard)} 个")
            if _has_title_suffix:
                wprint_info(f"  │  ├─ 核心名称Jaccard搜索（{_core_name}）: {len(candidates_core_jaccard)} 个")
            wprint_info(f"  │  ├─ Embedding搜索（name_only）: {len(candidates_name_embedding)} 个")
            wprint_info(f"  │  ├─ Embedding搜索（name+content）: {len(candidates_full_embedding)} 个")

        # 按 family_id 去重，保留最新版本
        entity_dict: Dict[str, Entity] = {}
        all_candidates = candidates_jaccard + candidates_core_jaccard + candidates_name_embedding + candidates_full_embedding
        for entity in all_candidates:
            existing = entity_dict.get(entity.family_id)
            if existing is None or entity.processed_time > existing.processed_time:
                entity_dict[entity.family_id] = entity
        similar_entities = list(entity_dict.values())

        # 过滤：已在当前抽取列表且已有关系的候选跳过
        if extracted_entity_names and extracted_relation_pairs:
            similar_entities = self._filter_candidates_by_existing_relations(
                similar_entities, entity_name,
                extracted_entity_names, extracted_relation_pairs,
            )

        return similar_entities

    def _filter_candidates_by_existing_relations(
        self,
        candidates: List[Entity],
        entity_name: str,
        extracted_entity_names: set,
        extracted_relation_pairs: set,
    ) -> List[Entity]:
        """过滤掉已有关系的候选实体（步骤3已处理）。"""
        # Pre-extract pair keys into a set for O(1) lookup (avoids O(C*R) any() scan)
        _pair_keys = {pair[0] for pair in extracted_relation_pairs} if extracted_relation_pairs else set()
        filtered = []
        skipped = 0
        for candidate in candidates:
            if candidate.name == entity_name:
                filtered.append(candidate)
            elif candidate.name not in extracted_entity_names:
                filtered.append(candidate)
            else:
                pair_key = (entity_name, candidate.name) if entity_name <= candidate.name else (candidate.name, entity_name)
                if pair_key in _pair_keys:
                    skipped += 1
                    if self._entity_tree_log():
                        wprint_info(f"  │  │  ├─ {candidate.name}: 跳过已有关系（步骤3已处理）")
                else:
                    filtered.append(candidate)
        if self._entity_tree_log() and skipped > 0:
            wprint_info(f"  │  跳过 {skipped} 个已在当前抽取列表且已存在关系的候选实体（步骤3已处理）")
        return filtered

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
        """
        处理单个实体

        流程：
        6.1 初步筛选：判断当前抽取的实体与检索到的实体列表，是否需要合并或存在关系
        6.2 精细化判断：对需要处理的候选进行详细判断，决定合并/创建关系/新建实体
        6.3 创建新实体并分配ID，更新关系边中的实体名称到ID映射

        Returns:
            Tuple[处理后的实体, 待处理的关系列表（使用实体名称）, 实体名称到ID的映射]
        """
        entity_name = extracted_entity['name']
        entity_content = extracted_entity['content']

        # 显示进度信息
        if self._entity_tree_log():
            if total_entities > 0:
                wprint_info(f"  ├─ 处理实体 [{entity_index}/{total_entities}]: {entity_name}")
            else:
                wprint_info(f"  ├─ 处理实体: {entity_name}")

        # 步骤1：使用预构建候选或重新搜索
        if prebuilt_candidates:
            # Reuse candidates from batch path — extract Entity objects and version_counts
            similar_entities = []
            version_counts: Dict[str, int] = {}
            for c in prebuilt_candidates:
                ent = c.get("entity")
                if ent is not None:
                    similar_entities.append(ent)
                    vc = c.get("version_count", 1)
                    if vc and c.get("family_id"):
                        version_counts[c["family_id"]] = vc
        else:
            similar_entities = self._search_entity_candidates(
                entity_name, entity_content, similarity_threshold,
                jaccard_search_threshold, embedding_name_search_threshold,
                embedding_full_search_threshold,
                extracted_entity_names, extracted_relation_pairs,
            )
            version_counts = {}

        if not similar_entities:
            # 没有找到相似实体，直接新建
            new_entity = self._create_new_entity(entity_name, entity_content, episode_id, source_document, base_time=base_time)
            self._mark_versioned(new_entity.family_id, already_versioned_family_ids, _version_lock)
            if self._entity_tree_log():
                wprint_info(f"  │  未找到相似实体，创建新实体: {new_entity.family_id}")
            _dbg_struct("decision_fallback_no_candidates",
                        name=entity_name, new_family_id=new_entity.family_id,
                        action="create_new")
            # 返回实体、空关系列表、实体名称到ID的映射
            entity_name_to_id = {
                entity_name: new_entity.family_id,
                new_entity.name: new_entity.family_id
            }
            return new_entity, [], entity_name_to_id

        if self._entity_tree_log():
            wprint_info(f"  │  找到 {len(similar_entities)} 个候选实体")

        unique_entities = similar_entities  # already deduped

        # 步骤3：准备已有实体信息供LLM分析
        # 构建实体组：当前抽取的实体（作为第一个，即"当前分析的实体"）+ 候选实体
        entities_group = [
            {
                'family_id': 'NEW_ENTITY',  # 标记为新实体
                'name': entity_name,
                'content': entity_content,
                'source_document': _doc_basename(source_document),
                'version_count': 0
            }
        ]

        # 添加候选实体信息（使用预构建 version_counts 或批量查询）
        if not version_counts:
            family_ids = [e.family_id for e in unique_entities]
            version_counts = self.storage.get_entity_version_counts(family_ids)
        for e in unique_entities:
            entities_group.append({
                'family_id': e.family_id,
                'name': e.name,
                'content': e.content,
                'source_document': e.source_document,
                'version_count': version_counts.get(e.family_id, 1)
            })

        # 步骤5：直接进行精细化判断（跳过 preliminary 筛选）
        # 候选表已经通过 Jaccard + embedding + BM25 + content-mention 多重筛选，
        # preliminary analysis 是多余的 LLM 调用。直接对所有候选做 detailed analysis。
        if self._entity_tree_log():
            wprint_info(f"  │  调用LLM分析（候选数: {len(unique_entities)}）")

        # All unique entities are candidates for detailed analysis (skip preliminary)
        candidates_to_analyze = {}
        for e in unique_entities:
            candidates_to_analyze[e.family_id] = {"type": "pending", "reason": ""}

        # Pre-encode current entity embedding for merge safety checks (once, not per-candidate)
        _current_entity_emb = prefetched_embedding
        if _current_entity_emb is None and self.storage.embedding_client and self.storage.embedding_client.is_available():
            try:
                _snip = self.llm_client.effective_entity_snippet_length()
                _embs = self.storage.embedding_client.encode(
                    [f"{entity_name} {entity_content[:_snip]}"]
                )
                if _embs is not None:
                    _current_entity_emb = np.array(_embs[0], dtype=np.float32)
            except Exception:
                pass

        # 输出初步筛选结果
        if self._entity_tree_log():
            wprint_info(f"  │  ├─ 跳过 preliminary, 直接精细化判断: {len(candidates_to_analyze)} 个候选")
        
        # 准备当前实体信息（新实体）
        current_entity_info = {
            "family_id": "NEW_ENTITY",
            "name": entity_name,
            "content": entity_content,
            "source_document": _doc_basename(source_document),
            "version_count": 0
        }
        
        # 对每个候选进行精细化判断
        merge_decisions = []  # 精细化判断后确定要合并的，包含候选实体信息
        relation_decisions = []  # 精细化判断后确定要创建关系的

        # 如果有需要精细化判断的候选，先打印开始提示
        if candidates_to_analyze:
            if self._entity_tree_log():
                wprint_info(f"  │  ├─ 精细化判断开始（共 {len(candidates_to_analyze)} 个候选）")

        # Phase 1: Parallel LLM calls for detailed analysis
        # Limit to top 5 candidates to cap LLM calls (sorted by combined_score desc)
        _MAX_DETAILED_CANDIDATES = 5
        _detailed_tasks = []  # (cid, candidate_entity, candidate_info, future_or_result)
        _unique_by_fid = {e.family_id: e for e in unique_entities if hasattr(e, 'family_id') and e.family_id}
        _sorted_cids = list(candidates_to_analyze.items())
        if len(_sorted_cids) > _MAX_DETAILED_CANDIDATES:
            _sorted_cids = _sorted_cids[:_MAX_DETAILED_CANDIDATES]
            if self._entity_tree_log():
                wprint_info(f"  │  ├─ 精细化判断截断: 仅分析前 {_MAX_DETAILED_CANDIDATES}/{len(candidates_to_analyze)} 个候选")
        for cid, info in _sorted_cids:
            candidate_entity = _unique_by_fid.get(cid)
            if not candidate_entity:
                continue
            candidate_info = {
                "family_id": cid,
                "name": candidate_entity.name,
                "content": candidate_entity.content,
                "source_document": candidate_entity.source_document,
                "version_count": version_counts.get(cid, 1)
            }
            _detailed_tasks.append((cid, candidate_entity, candidate_info))

        # Execute LLM calls in parallel (3 workers to utilize concurrency budget)
        _detailed_results: Dict[str, Optional[Dict]] = {}
        if len(_detailed_tasks) > 1:
            def _call_detailed(task):
                cid, cent, cinfo = task
                try:
                    return (cid, self.llm_client.analyze_entity_pair_detailed(
                        current_entity_info, cinfo, [], context_text=context_text))
                except Exception as e:
                    logger.warning("LLM detailed analysis failed for '%s' vs '%s': %s — skipping",
                                   entity_name, cent.name, e)
                    return (cid, None)
            pool = _get_entity_pool(3)
            for cid, result in pool.map(_call_detailed, _detailed_tasks):
                if result is not None:
                    _detailed_results[cid] = result
        else:
            for cid, cent, cinfo in _detailed_tasks:
                try:
                    _detailed_results[cid] = self.llm_client.analyze_entity_pair_detailed(
                        current_entity_info, cinfo, [], context_text=context_text)
                except Exception as e:
                    logger.warning("LLM detailed analysis failed for '%s' vs '%s': %s — skipping",
                                   entity_name, cent.name, e)

        # Phase 2: Sequential result processing (merge safety checks, state mutation)
        for cid, candidate_entity, candidate_info in _detailed_tasks:
            detailed_result = _detailed_results.get(cid)
            if not detailed_result:
                continue
            
            action = detailed_result.get("action", "no_action")
            relation_content = detailed_result.get("relation_content", "")
            

            if action == "merge":
                _dbg_struct("fallback_detailed_analysis",
                            name=entity_name, candidate_name=candidate_entity.name,
                            candidate_fid=cid, action=action)
                # ---- Three-way alignment verification (Phase 4) ----
                _guard = self._alignment_guard(
                    entity_name, entity_content,
                    candidate_entity.name, candidate_entity.content or "",
                )
                if _guard:
                    _align_verdict, _align_confidence = _guard
                    if self._entity_tree_log():
                        wprint_info(f"  │  │  ├─ 三值对齐: verdict={_align_verdict} (conf={_align_confidence:.2f}), 跳过")
                    continue  # skip this candidate

                # 合并安全检查：Jaccard 名称相似度 < 0.3 或 embedding < 0.5 → 禁止合并
                _jaccard = self._calculate_jaccard_similarity(entity_name, candidate_entity.name)
                if _jaccard < 0.3:
                    if self._entity_tree_log():
                        wprint_info(f"  │  │  ├─ 合并被阻止: 名称Jaccard相似度过低 ({_jaccard:.2f})")
                    continue
                if _current_entity_emb is not None:
                    _cand_emb = getattr(candidate_entity, 'embedding', None)
                    if _cand_emb is not None:
                        # embedding 可能存储为 bytes（tobytes()），需要正确还原
                        if isinstance(_cand_emb, bytes):
                            _cand_emb = np.frombuffer(_cand_emb, dtype=np.float32)
                        elif not isinstance(_cand_emb, np.ndarray):
                            _cand_emb = np.array(_cand_emb, dtype=np.float32)
                        _sim = self._cosine_similarity(
                            _current_entity_emb,
                            _cand_emb,
                        )
                        if _sim < 0.5:
                            if self._entity_tree_log():
                                wprint_info(f"  │  │  ├─ 合并被阻止: embedding相似度过低 ({_sim:.2f})")
                            continue
                merge_target_id = cid  # 使用候选实体ID作为合并目标
                merge_decisions.append({
                    "target_family_id": merge_target_id,
                    "source_family_id": "NEW_ENTITY",
                    "candidate_family_id": cid,  # 记录候选实体ID，用于后续收集content
                    "candidate_content": candidate_entity.content,  # 记录候选实体content
                    "candidate_name": candidate_entity.name,  # 记录候选实体名称
                })
            elif action == "create_relation":
                # 确保有关系描述
                if not relation_content:
                    relation_content = f"{entity_name}与{candidate_entity.name}存在关联关系"

                relation_decisions.append({
                    "entity1_id": "NEW_ENTITY",
                    "entity2_id": cid,
                    "entity1_name": entity_name,
                    "entity2_name": candidate_entity.name,
                    "content": relation_content,
                })
            elif action == "no_action":
                pass

        # 输出最终分析结果
        if merge_decisions or relation_decisions:
            if self._entity_tree_log():
                wprint_info(f"  │  └─ 精细化判断: 合并 {len(merge_decisions)} 个, 关系 {len(relation_decisions)} 个")
        
        # 步骤9：处理分析结果（合并决策和关系决策）
        final_entity = None
        pending_relations = []  # 待处理的关系（使用实体名称，因为新实体还没有ID）
        entity_name_to_id = {}  # 实体名称到ID的映射
        other_targets_entities = {}  # 存储其他目标实体的信息（在合并前收集，合并后这些ID就不存在了）
        
        # 6.1-6.2：处理合并决策
        # 如果有多个合并决策，需要选择一个主要目标实体
        # 策略：优先选择版本数最多的实体作为目标
        if merge_decisions:
            # 收集所有目标实体ID
            target_family_ids = [d.get("target_family_id") for d in merge_decisions
                                if d.get("target_family_id") and d.get("target_family_id") != 'NEW_ENTITY']
            
            if target_family_ids:
                # 如果所有合并决策都指向同一个目标，直接使用
                _target_set = set(target_family_ids)
                if len(_target_set) == 1:
                    primary_target_id = target_family_ids[0]
                    other_targets = []  # 没有其他目标
                else:
                    # 如果有多个不同的目标，选择版本数最多的作为主要目标
                    target_version_counts = {}
                    counts = self.storage.get_entity_version_counts(target_family_ids)
                    target_version_counts = {tid: counts.get(tid, 0) for tid in target_family_ids}

                    primary_target_id = max(target_family_ids, key=lambda tid: target_version_counts.get(tid, 0))

                    # 输出多个合并目标的信息
                    other_targets = [tid for tid in _target_set if tid != primary_target_id]
                    if other_targets:
                        if self._entity_tree_log():
                            wprint_info(f"  │  ├─ 多合并目标: 选择 {primary_target_id} 为主要目标（版本数最多）")
                        
                        # 在合并之前，先收集其他目标实体的信息（合并后这些ID就不存在了）
                        other_targets_entities.clear()  # 清空之前的数据
                        try:
                            other_entities_map = self.storage.get_entities_by_family_ids(other_targets)
                            for tid, other_entity in other_entities_map.items():
                                other_targets_entities[tid] = {
                                    'entity': other_entity,
                                    'name': other_entity.name,
                                    'content': other_entity.content
                                }
                        except Exception:
                            # Fallback: individual fetch
                            for other_target_id in other_targets:
                                other_entity = self.storage.get_entity_by_family_id(other_target_id)
                                if other_entity:
                                    other_targets_entities[other_target_id] = {
                                        'entity': other_entity,
                                        'name': other_entity.name,
                                        'content': other_entity.content
                                    }
                        
                        # 如果有多个不同的目标实体ID，说明这些实体都是同一个实体
                        # 需要将其他目标实体ID合并到主要目标ID
                        merge_result = self.storage.merge_entity_families(primary_target_id, other_targets)
                        
                        # 更新映射：将所有指向旧实体ID的映射更新为新的 primary_target_id
                        # 这确保映射中不会保留指向已合并ID的失效映射
                        updated_mapping_count = 0
                        for name, eid in list(entity_name_to_id.items()):
                            if eid in other_targets:
                                entity_name_to_id[name] = primary_target_id
                                updated_mapping_count += 1
                        # 处理合并后产生的自指向关系（暂时跳过，因为entity_processor中没有这个方法）
                        # 自指向关系会在后续的consolidate_knowledge_graph_entity中处理
                
                # 合并新实体到主要目标实体
                latest_entity = self.storage.get_entity_by_family_id(primary_target_id)
                if latest_entity:
                    # 防止同窗口重复版本化：如果该 family_id 已创建过版本，复用已有实体
                    if already_versioned_family_ids and primary_target_id in already_versioned_family_ids:
                        if self._entity_tree_log():
                            wprint_info(f"  │  family_id {primary_target_id} 已在本次处理中创建版本，复用已有实体")
                        final_entity = latest_entity
                        entity_name_to_id[entity_name] = primary_target_id
                        entity_name_to_id[final_entity.name] = primary_target_id
                    else:
                        target_name = latest_entity.name

                        # 收集所有需要合并到主要目标的实体的content
                        # 包括：主要目标实体 + 新实体 + 所有指向主要目标的候选实体 + 被合并到主要目标的其他目标实体
                        contents_to_merge = [latest_entity.content, entity_content]
                        _contents_set = {latest_entity.content, entity_content}  # O(1) dedup companion
                        entities_to_merge_names = [latest_entity.name, entity_name]
                        entity_sources_to_merge = [latest_entity.source_document, source_document]

                        # 收集被合并到主要目标的其他目标实体的content（如果有多个不同的目标实体ID）
                        # 注意：这些实体ID已经在合并前被收集到 other_targets_entities 中，因为合并后这些ID就不存在了
                        if len(_target_set) > 1 and other_targets_entities:
                            for other_target_id, other_info in other_targets_entities.items():
                                other_content = other_info.get('content')
                                other_name = other_info.get('name')
                                if other_content:
                                    # 检查是否已经添加（通过内容比较，避免重复）
                                    if other_content not in _contents_set:
                                        contents_to_merge.append(other_content)
                                        _contents_set.add(other_content)
                                        entities_to_merge_names.append(other_name or f"实体{other_target_id}")
                                        other_entity = other_info.get('entity')
                                        entity_sources_to_merge.append(other_entity.source_document if other_entity else "")

                        # 收集所有指向主要目标的候选实体的content
                        for merge_decision in merge_decisions:
                            candidate_target_id = merge_decision.get("target_family_id")
                            candidate_family_id = merge_decision.get("candidate_family_id")
                            candidate_content = merge_decision.get("candidate_content")
                            candidate_name = merge_decision.get("candidate_name")

                            # 如果这个合并决策指向主要目标，且候选实体不是主要目标本身
                            if candidate_target_id == primary_target_id and candidate_family_id and candidate_family_id != primary_target_id:
                                # 添加候选实体的content（如果还没有添加，避免重复）
                                if candidate_content:
                                    # 检查是否已经添加（通过内容比较，避免重复）
                                    if candidate_content not in _contents_set:
                                        contents_to_merge.append(candidate_content)
                                        _contents_set.add(candidate_content)
                                        entities_to_merge_names.append(candidate_name or f"实体{candidate_family_id}")
                                        entity_sources_to_merge.append(merge_decision.get("source_document", ""))

                        # 快速比较：内容是否变化（始终版本化，但避免多余的合并 LLM 调用）
                        _old_content = (latest_entity.content or "").strip()
                        _new_content = entity_content.strip()
                        if _old_content == _new_content and entity_name == latest_entity.name:
                            # 内容完全相同 → 直接复制创建版本（不调 LLM）
                            final_entity = self._create_entity_version(
                                primary_target_id,
                                latest_entity.name,
                                latest_entity.content,
                                episode_id,
                                source_document,
                                base_time=base_time,
                                old_content=latest_entity.content or "",
                                old_content_format=latest_entity.content_format or "plain",
                            )
                            self._mark_versioned(primary_target_id, already_versioned_family_ids, _version_lock)
                        else:
                            # 内容有差异 → 走完整合并流程
                            if entity_name != latest_entity.name:
                                merged_name = self.llm_client.merge_entity_name(
                                    latest_entity.name,
                                    entity_name
                                )
                            else:
                                merged_name = entity_name

                            merged_content = self.llm_client.merge_multiple_entity_contents(
                                contents_to_merge,
                                entity_sources=entity_sources_to_merge,
                                entity_names=entities_to_merge_names,
                            )
                            if self._entity_tree_log():
                                wprint_info(f"  │  ├─ 合并 {len(contents_to_merge)} 个实体的content: {', '.join(entities_to_merge_names[:3])}{'...' if len(entities_to_merge_names) > 3 else ''}")

                            final_entity = self._create_entity_version(
                                primary_target_id,
                                merged_name,
                                merged_content,
                                episode_id,
                                source_document,
                                base_time=base_time,
                                old_content=latest_entity.content or "",
                                old_content_format=latest_entity.content_format or "plain",
                            )
                            self._mark_versioned(primary_target_id, already_versioned_family_ids, _version_lock)

                        # 更新映射：原始名称和目标实体名称都映射到目标实体ID
                        entity_name_to_id[entity_name] = primary_target_id
                        entity_name_to_id[final_entity.name] = primary_target_id
        
        # 6.2：处理关系决策（记录关系，但使用实体名称，因为新实体可能还没有ID）
        for rel_info in relation_decisions:
            entity1_name = rel_info.get("entity1_name", entity_name)
            entity2_name = rel_info.get("entity2_name", "")
            content = rel_info.get("content", "")
            
            # 判断关系类型
            relation_type = "normal"
            if "别名" in content or "称呼" in content or "简称" in content:
                relation_type = "alias"
            
            if self._entity_tree_log():
                wprint_info(f"  │  ├─ 关系: {entity1_name} <-> {entity2_name}")
            
            # 关系使用实体名称，ID将在步骤9中更新
            pending_relations.append({
                "entity1_name": entity1_name,  # 当前抽取的实体名称
                "entity2_name": entity2_name,  # 候选实体名称
                "content": content,
                "relation_type": relation_type
            })
        
        # 步骤9：如果没有匹配或合并，创建新实体并分配ID
        if not final_entity:
            # 检查是否有匹配的实体（通过分析结果判断）
            matched = bool(merge_decisions)

            if matched:
                # 有合并决策但未成功生成 final_entity，尝试取第一个候选作为兜底
                if self._entity_tree_log():
                    wprint_info("  │  ⚠️ 合并决策存在但未生成最终实体，使用兜底逻辑")
                first_target_id = merge_decisions[0].get("target_family_id", "")
                if first_target_id:
                    fallback_entity = self.storage.get_entity_by_family_id(first_target_id)
                    if fallback_entity:
                        # 始终创建新版本（兜底路径也要版本化）
                        final_entity = self._create_entity_version(
                            first_target_id,
                            entity_name,
                            entity_content,
                            episode_id,
                            source_document,
                            base_time=base_time,
                            old_content=fallback_entity.content or "",
                            old_content_format=fallback_entity.content_format or "plain",
                        )
                        self._mark_versioned(first_target_id, already_versioned_family_ids, _version_lock)
                        entity_name_to_id[entity_name] = final_entity.family_id
                        entity_name_to_id[final_entity.name] = final_entity.family_id

            if not final_entity:
                # 没有匹配或兜底失败，创建新实体
                final_entity = self._create_new_entity(entity_name, entity_content, episode_id, source_document, base_time=base_time)
                self._mark_versioned(final_entity.family_id, already_versioned_family_ids, _version_lock)
                # 更新映射：新创建的实体
                entity_name_to_id[entity_name] = final_entity.family_id
                entity_name_to_id[final_entity.name] = final_entity.family_id
        
        # 步骤9：更新关系边中的实体名称到ID映射
        # 对于pending_relations中的关系，如果涉及当前实体（entity1_name），更新为实际的family_id
        updated_relations = []
        for rel in pending_relations:
            if rel["entity1_name"] == entity_name:
                # 当前实体已创建，更新为family_id
                updated_rel = rel.copy()
                updated_rel["entity1_id"] = final_entity.family_id if final_entity else None
                updated_relations.append(updated_rel)
            else:
                # 保持原样（entity2_name是已有实体，将在步骤10中处理）
                updated_relations.append(rel)
        
        # 输出最终结果
        if self._entity_tree_log():
            if final_entity:
                if updated_relations:
                    wprint_info(f"  └─ 完成: {final_entity.name} ({final_entity.family_id}), 关系 {len(updated_relations)} 个")
                else:
                    wprint_info(f"  └─ 完成: {final_entity.name} ({final_entity.family_id})")
            else:
                if updated_relations:
                    wprint_info(f"  └─ 完成: 关系 {len(updated_relations)} 个")
        
        return final_entity, updated_relations, entity_name_to_id
    
    @staticmethod
    def _extract_summary(name: str, content: str) -> str:
        """从实体名称和内容中提取简短摘要（无需额外LLM调用）。"""
        # 跳过 markdown 标题行，取第一行非空正文
        if not content:
            return name[:100]
        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            return stripped[:200] if len(stripped) > 200 else stripped
        # 回退到名称
        return name[:100]

    def _construct_entity(self, name: str, content: str, episode_id: str,
                          family_id: str, source_document: str = "",
                          base_time: Optional[datetime] = None,
                          confidence: Optional[float] = None) -> Entity:
        """Shared helper: construct an Entity object with standard fields.

        Args:
            confidence: Initial confidence from LLM extraction (0.0–1.0).
                        Falls back to 0.7 if not provided.
        """
        # Guard: never create entities with empty names
        name = (name or "").strip()
        if not name:
            logger.warning("_construct_entity called with empty name — using fallback")
            name = "未命名概念"
        _now = datetime.now(timezone.utc)
        event_time = base_time if base_time is not None else _now
        processed_time = _now
        entity_record_id = f"entity_{processed_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        source_document_only = _doc_basename(source_document)
        # Use LLM-provided confidence if available, otherwise default
        initial_confidence = confidence if confidence is not None else 0.7
        initial_confidence = max(0.0, min(1.0, initial_confidence))
        return Entity(
            absolute_id=entity_record_id,
            family_id=family_id,
            name=name,
            content=content,
            event_time=event_time,
            processed_time=processed_time,
            episode_id=episode_id,
            source_document=source_document_only,
            content_format="markdown",
            summary=self._extract_summary(name, content),
            confidence=initial_confidence,
        )

    def _build_new_entity(self, name: str, content: str, episode_id: str,
                          source_document: str = "", base_time: Optional[datetime] = None,
                          confidence: Optional[float] = None) -> Entity:
        """构建新实体对象，但不立即写库。"""
        return self._construct_entity(
            name, content, episode_id,
            family_id=f"ent_{uuid.uuid4().hex[:12]}",
            source_document=source_document, base_time=base_time,
            confidence=confidence,
        )

    def _create_new_entity(self, name: str, content: str, episode_id: str,
                           source_document: str = "", base_time: Optional[datetime] = None,
                           confidence: Optional[float] = None) -> Entity:
        """创建新实体"""
        entity = self._build_new_entity(name, content, episode_id, source_document, base_time=base_time,
                                        confidence=confidence)
        self.storage.save_entity(entity)
        return entity

    def _build_entity_version(self, family_id: str, name: str, content: str,
                              episode_id: str, source_document: str = "",
                              base_time: Optional[datetime] = None) -> Entity:
        """构建实体新版本对象，但不立即写库。"""
        return self._construct_entity(
            name, content, episode_id,
            family_id=family_id,
            source_document=source_document, base_time=base_time,
        )

    def _create_entity_version(self, family_id: str, name: str, content: str,
                              episode_id: str, source_document: str = "",
                              base_time: Optional[datetime] = None,
                              old_content: str = "",
                              old_content_format: str = "plain") -> Entity:
        """创建实体的新版本，并记录 section 级 patches。"""
        # 始终创建新版本（每个 episode 提及的概念都版本化）

        entity = self._build_entity_version(family_id, name, content, episode_id, source_document, base_time=base_time)
        self.storage.save_entity(entity)

        # 注意：置信度 corroboration 在 extraction.py Phase C-1b 统一处理，不在此处重复调用

        # 计算 section patches
        _source_document_only = _doc_basename(source_document)
        if old_content:
            patches = self._compute_entity_patches(
                family_id=family_id,
                old_content=old_content,
                old_content_format=old_content_format,
                new_content=content,
                new_absolute_id=entity.absolute_id,
                source_document=_source_document_only,
                event_time=entity.event_time,
            )
            if patches:
                self.storage.save_content_patches(patches)

        return entity
    
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
        """计算新旧内容之间的 section 级变更 patches。"""
        old_sections = content_to_sections(old_content, old_content_format, ENTITY_SECTIONS)
        new_sections = content_to_sections(new_content, "markdown", ENTITY_SECTIONS)
        if sections_equal(old_sections, new_sections):
            return []
        diff = compute_section_diff(old_sections, new_sections)
        if not has_any_change(diff):
            return []
        patches = []
        _now = datetime.now()
        for key, info in diff.items():
            if not info.get("changed", False):
                continue
            patches.append(ContentPatch(
                uuid=str(uuid.uuid4()),
                target_type="Entity",
                target_absolute_id=new_absolute_id,
                target_family_id=family_id,
                section_key=key,
                change_type=info.get("change_type", "modified"),
                old_hash=section_hash(info.get("old", "") or ""),
                new_hash=section_hash(info.get("new", "") or ""),
                diff_summary=f"Section '{key}' {info.get('change_type', 'modified')}",
                source_document=source_document,
                event_time=event_time or _now,
            ))
        return patches

