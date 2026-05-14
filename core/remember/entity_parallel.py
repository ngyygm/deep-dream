"""
Parallel entity processing logic.
Extracted from EntityProcessor for modularity.
"""
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import threading
import time
import numpy as np
import logging

from core.models import Entity
from core.storage.neo4j_store import Neo4jStorageManager
from core.llm.client import LLMClient
from core.utils import wprint_info
from core.debug_log import log_struct as _dbg_struct
from core.remember._shared import _doc_basename

logger = logging.getLogger(__name__)


def _process_entities_sequential(
    storage: Neo4jStorageManager,
    llm_client: LLMClient,
    candidate_builder,  # EntityCandidateBuilder
    entity_tree_log: bool,
    build_entity_candidate_table_fn,  # callable
    process_entity_with_batch_candidates_fn,  # callable
    extracted_entities: List[Dict[str, str]],
    episode_id: str,
    similarity_threshold: float = 0.7,
    episode=None,
    source_document: str = "",
    context_text: Optional[str] = None,
    extracted_relations: Optional[List[Dict[str, str]]] = None,
    jaccard_search_threshold: Optional[float] = None,
    embedding_name_search_threshold: Optional[float] = None,
    embedding_full_search_threshold: Optional[float] = None,
    on_entity_processed: Optional[callable] = None,
    base_time=None,
    prefetched_embeddings: Optional[Tuple[Optional[Any], Optional[Any]]] = None,
    already_versioned_family_ids: Optional[set] = None,
) -> Tuple[List[Entity], List[Dict], Dict[str, str]]:
    """串行处理实体（原逻辑）。"""
    from core.remember.entity import _preprocess_extraction_context

    processed_entities: List[Entity] = []
    pending_relations: List[Dict] = []
    entity_name_to_id: Dict[str, str] = {}
    _corroborated_fids: List[str] = []

    extracted_entity_names, extracted_relation_pairs, related_entity_names = _preprocess_extraction_context(
        extracted_entities, extracted_relations,
    )

    candidate_table = build_entity_candidate_table_fn(
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
        entity, relations, name_mapping, to_persist = process_entity_with_batch_candidates_fn(
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
            storage.save_entity(to_persist)
            _ent_patches = getattr(to_persist, '_pending_patches', None) or []
            if _ent_patches:
                try:
                    storage.save_content_patches(_ent_patches)
                except Exception:
                    pass
            if to_persist.family_id:
                _corroborated_fids.append(to_persist.family_id)
        if on_entity_processed and entity:
            on_entity_processed(entity, entity_name_to_id, relations or [])

    # Batch corroboration: 独立来源印证 → 置信度提升
    if _corroborated_fids:
        _unique_fids = list(set(_corroborated_fids))
        try:
            storage.adjust_confidence_on_corroboration_batch(_unique_fids, source_type="entity")
        except Exception:
            pass

    return processed_entities, pending_relations, entity_name_to_id


def _process_entities_parallel(
    storage: Neo4jStorageManager,
    llm_client: LLMClient,
    candidate_builder,
    entity_tree_log: bool,
    build_entity_candidate_table_fn,  # callable
    process_entity_with_batch_candidates_fn,  # callable
    get_entity_pool_fn,  # callable: (max_workers) -> ThreadPoolExecutor
    extracted_entities: List[Dict[str, str]],
    episode_id: str,
    similarity_threshold: float = 0.7,
    episode=None,
    source_document: str = "",
    context_text: Optional[str] = None,
    extracted_relations: Optional[List[Dict[str, str]]] = None,
    jaccard_search_threshold: Optional[float] = None,
    embedding_name_search_threshold: Optional[float] = None,
    embedding_full_search_threshold: Optional[float] = None,
    on_entity_processed: Optional[callable] = None,
    base_time=None,
    max_workers: int = 1,
    prefetched_embeddings: Optional[Tuple[Optional[Any], Optional[Any]]] = None,
    already_versioned_family_ids: Optional[set] = None,
) -> Tuple[List[Entity], List[Dict], Dict[str, str]]:
    """多线程处理实体；合并冲突时以数据库中已存在的 family_id 为准。"""
    from core.remember.entity import _preprocess_extraction_context

    extracted_entity_names, extracted_relation_pairs, related_entity_names = _preprocess_extraction_context(
        extracted_entities, extracted_relations,
    )

    # 不再过滤孤立实体：所有通过验证的实体都应被处理
    # 孤立实体仍然有价值（如对话中提到的技术选型），丢弃会导致信息损失
    _skipped_orphans = 0
    _orig_indices = list(range(len(extracted_entities)))
    filtered_entities = extracted_entities

    candidate_table = build_entity_candidate_table_fn(
        extracted_entities,
        similarity_threshold=similarity_threshold,
        jaccard_search_threshold=jaccard_search_threshold,
        embedding_name_search_threshold=embedding_name_search_threshold,
        embedding_full_search_threshold=embedding_full_search_threshold,
        prefetched_embeddings=prefetched_embeddings,
    )
    total_entities = len(extracted_entities)
    _distill_step = llm_client._current_distill_step
    _priority = getattr(llm_client._priority_local, 'priority', 5)
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

    # Pre-seed entity name cache from candidate entities to reduce hidden reads
    for _cand_row in candidate_table.values():
        for _cand in _cand_row:
            _cand_ent = _cand.get("entity")
            if _cand_ent and hasattr(_cand_ent, 'absolute_id') and hasattr(_cand_ent, 'name'):
                storage._cache_entity_name(_cand_ent.absolute_id, _cand_ent.name)

    def task(idx: int, extracted_entity: Dict[str, str], orig_idx: int):
        # 将主线程的 distill step 和优先级传播到工作线程（threading.local）
        llm_client._current_distill_step = _distill_step
        llm_client._priority_local.priority = _priority
        candidates = candidate_table.get(orig_idx, [])
        _ent_emb = None
        if _prefetched_full_embs is not None and orig_idx < len(_prefetched_full_embs):
            try:
                _ent_emb = np.array(_prefetched_full_embs[orig_idx], dtype=np.float32)
            except Exception:
                pass
        entity, relations, name_mapping, to_persist = process_entity_with_batch_candidates_fn(
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
    executor = get_entity_pool_fn(max_workers)
    from concurrent.futures import as_completed
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
            _resolve_fn = getattr(storage, 'resolve_family_ids', None)
            if _resolve_fn:
                resolved_map = _resolve_fn(list(all_candidate_eids)) or {}
                existing_eids = set(resolved_map.keys()) | set(resolved_map.values())
            else:
                _batch_result = storage.get_entities_by_family_ids(list(all_candidate_eids))
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
        if hasattr(storage, 'register_entity_redirects_batch'):
            storage.register_entity_redirects_batch(redirect_pairs)
        else:
            for source_id, canonical_id in redirect_pairs:
                storage.register_entity_redirect(source_id, canonical_id)

    # 对于被合并到 canonical ID 的非 canonical 实体，需要从 results 中修正
    _canonical_ids_to_fetch = set()
    for idx, entity, relations, name_mapping, to_persist in results:
        if entity and entity.family_id != entity_name_to_id.get(entity.name):
            canonical_id = entity_name_to_id.get(entity.name)
            if canonical_id:
                _canonical_ids_to_fetch.add(canonical_id)
    if _canonical_ids_to_fetch:
        try:
            _canonical_ent_map = storage.get_entities_by_family_ids(
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
    # 按 family_id 去重：同一 family_id 只保留一个待持久化实体（避免重复写入）
    if entities_to_persist_final:
        _seen_fids = set()
        _deduped = []
        for e in entities_to_persist_final:
            if e.family_id not in _seen_fids:
                _seen_fids.add(e.family_id)
                _deduped.append(e)
        if len(_deduped) < len(entities_to_persist_final):
            _dup_count = len(entities_to_persist_final) - len(_deduped)
            if entity_tree_log:
                wprint_info(f"  │  持久化去重: 移除 {_dup_count} 个重复 family_id 的待持久化实体")
            entities_to_persist_final = _deduped
        # 批量保存实体（UNWIND 一次写入，减少 Neo4j 连接数）
        _corro_fids = []
        # 预计算所有 embedding（CPU 密集，不需要 Neo4j session）
        for e in entities_to_persist_final:
            try:
                _emb_result = storage._compute_entity_embedding(e)
                if _emb_result is not None:
                    e.embedding = _emb_result[0]
            except Exception:
                pass
        # 一次 UNWIND 写入所有实体
        try:
            storage.bulk_save_entities_with_embedding(entities_to_persist_final)
        except Exception:
            # Fallback: 逐条写入
            for e in entities_to_persist_final:
                try:
                    storage.save_entity(e)
                except Exception:
                    pass
        # 一次写入所有 patches
        _all_patches = []
        for e in entities_to_persist_final:
            _ent_patches = getattr(e, '_pending_patches', None) or []
            _all_patches.extend(_ent_patches)
            if e.family_id:
                _corro_fids.append(e.family_id)
        if _all_patches:
            try:
                storage.save_content_patches(_all_patches)
            except Exception:
                pass
        # Batch corroboration
        if _corro_fids:
            try:
                storage.adjust_confidence_on_corroboration_batch(list(set(_corro_fids)), source_type="entity")
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
