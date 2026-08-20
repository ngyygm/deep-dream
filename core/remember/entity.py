"""
实体处理模块：实体搜索、对齐、更新/新建

This module provides EntityProcessor — the main entry point for entity
processing. Heavy logic is delegated to sub-modules:
  - entity_alignment: batch/parallel alignment layer (in this package)
  - entity_candidates: candidate building + enrich + search filters
"""
from typing import List, Dict, Optional, Tuple, Any
from collections import OrderedDict
from datetime import datetime, timezone
from concurrent.futures import Future

import uuid
import numpy as np
import logging

from core.debug_log import log_struct as _dbg_struct
from core.models import Entity, Episode
from core.storage.sqlite.manager import SQLiteGraphStorageManager
from core.llm.client import LLMClient
from core.utils import wprint_info


from ._shared import _doc_basename, _get_entity_pool
from core.content_schema import (
    ENTITY_SECTIONS,
    compute_content_patches,
)
from core.remember.entity_candidates import (
    EntityCandidateBuilder,
    normalize_entity_name_for_matching,
)

# Sub-module imports

from core.remember.entity_candidates import (
    _calculate_jaccard_similarity as _calc_jaccard_fn,
    _cosine_similarity as _cosine_sim_fn,
    _alignment_guard as _alignment_guard_fn,
    _try_context_alias_merge as _try_context_alias_merge_fn,
)

from core.remember.entity_alignment import (
    _EntityBatchMixin,
    _process_entities_sequential as _process_entities_sequential_fn,
    _process_entities_parallel as _process_entities_parallel_fn,
)


logger = logging.getLogger(__name__)
# Pool refs are now in _shared
# ---------------------------------------------------------------------------
# Construction factories + sequential fallback (moved from
# entity_construction.py / entity_sequential.py)
# ---------------------------------------------------------------------------

def _construct_entity(name: str, content: str, episode_id: str,
                      family_id: str, source_document: str = "",
                      base_time: Optional[datetime] = None,
                      confidence: Optional[float] = None) -> Entity:
    """Shared helper: construct an Entity object with standard fields.

    Args:
        confidence: Initial confidence from LLM extraction (0.0-1.0).
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
        confidence=initial_confidence,
    )


def _build_new_entity(name: str, content: str, episode_id: str,
                      source_document: str = "", base_time: Optional[datetime] = None,
                      confidence: Optional[float] = None) -> Entity:
    """构建新实体对象，但不立即写库。"""
    return _construct_entity(
        name, content, episode_id,
        family_id=f"ent_{uuid.uuid4().hex[:12]}",
        source_document=source_document, base_time=base_time,
        confidence=confidence,
    )


def _create_new_entity(storage: SQLiteGraphStorageManager,
                       name: str, content: str, episode_id: str,
                       source_document: str = "", base_time: Optional[datetime] = None,
                       confidence: Optional[float] = None) -> Entity:
    """创建新实体"""
    entity = _build_new_entity(name, content, episode_id, source_document, base_time=base_time,
                               confidence=confidence)
    storage.save_entity(entity)
    return entity


def _compute_entity_patches(
    family_id: str,
    old_content: str,
    old_content_format: str,
    new_content: str,
    new_absolute_id: str,
    source_document: str = "",
    event_time: Optional[datetime] = None,
) -> list:
    return compute_content_patches(
        family_id=family_id,
        old_content=old_content,
        old_content_format=old_content_format,
        new_content=new_content,
        new_absolute_id=new_absolute_id,
        target_type="Entity",
        schema=ENTITY_SECTIONS,
        source_document=source_document,
        event_time=event_time,
    )


def _build_entity_version(family_id: str, name: str, content: str,
                          episode_id: str, source_document: str = "",
                          base_time: Optional[datetime] = None,
                          old_content: str = "",
                          old_content_format: str = "plain") -> Entity:
    """构建实体新版本对象，但不立即写库。附带 section patch 计算。"""
    entity = _construct_entity(
        name, content, episode_id,
        family_id=family_id,
        source_document=source_document, base_time=base_time,
    )
    if old_content:
        patches = _compute_entity_patches(
            family_id=family_id,
            old_content=old_content,
            old_content_format=old_content_format,
            new_content=content,
            new_absolute_id=entity.absolute_id,
            source_document=_doc_basename(source_document),
            event_time=entity.event_time,
        )
        if patches:
            entity._pending_patches = patches
    return entity


def _create_entity_version(storage: SQLiteGraphStorageManager,
                           family_id: str, name: str, content: str,
                           episode_id: str, source_document: str = "",
                           base_time: Optional[datetime] = None,
                           old_content: str = "",
                           old_content_format: str = "plain") -> Entity:
    """创建实体的新版本，并记录 section 级 patches。"""
    # 始终创建新版本（每个 episode 提及的概念都版本化）

    entity = _build_entity_version(family_id, name, content, episode_id, source_document, base_time=base_time)
    storage.save_entity(entity)

    # 注意：置信度 corroboration 在 extraction.py Phase C-1b 统一处理，不在此处重复调用

    # 计算 section patches
    _source_document_only = _doc_basename(source_document)
    if old_content:
        patches = _compute_entity_patches(
            family_id=family_id,
            old_content=old_content,
            old_content_format=old_content_format,
            new_content=content,
            new_absolute_id=entity.absolute_id,
            source_document=_source_document_only,
            event_time=entity.event_time,
        )
        if patches:
            storage.save_content_patches(patches)

    return entity


def _process_entity_sequential_fallback(
    storage: SQLiteGraphStorageManager,
    llm_client: LLMClient,
    entity_tree_log: bool,
    create_new_entity_fn,  # callable for _create_new_entity
    build_new_entity_fn,  # callable for _build_new_entity
    create_entity_version_fn,  # callable for _create_entity_version
    build_entity_version_fn,  # callable for _build_entity_version
    mark_versioned_fn,  # callable for _mark_versioned
    alignment_guard_fn,  # callable for _alignment_guard
    calculate_jaccard_fn,  # callable for _calculate_jaccard_similarity
    cosine_similarity_fn,  # callable for _cosine_similarity
    merge_two_contents_fn,  # callable for _merge_two_contents
    extracted_entity: Dict[str, str],
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
    prebuilt_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[Entity], List[Dict], Dict[str, str]]:
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
    entity_content = extracted_entity.get('content', '')

    # 显示进度信息
    if entity_tree_log:
        if total_entities > 0:
            wprint_info(f"  ├─ 处理实体 [{entity_index}/{total_entities}]: {entity_name}")
        else:
            wprint_info(f"  ├─ 处理实体: {entity_name}")

    # 步骤1：使用预构建候选（batch 路径已生成）
    similar_entities = []
    version_counts: Dict[str, int] = {}
    for c in (prebuilt_candidates or []):
        ent = c.get("entity")
        if ent is not None:
            similar_entities.append(ent)
            vc = c.get("version_count", 1)
            if vc and c.get("family_id"):
                version_counts[c["family_id"]] = vc

    if not similar_entities:
        # 没有找到相似实体，直接新建
        new_entity = create_new_entity_fn(entity_name, entity_content, episode_id, source_document, base_time=base_time)
        mark_versioned_fn(new_entity.family_id, already_versioned_family_ids, _version_lock)
        if entity_tree_log:
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

    if entity_tree_log:
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
        version_counts = storage.get_entity_version_counts(family_ids)
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
    if entity_tree_log:
        wprint_info(f"  │  调用LLM分析（候选数: {len(unique_entities)}）")

    # All unique entities are candidates for detailed analysis (skip preliminary)
    candidates_to_analyze = {}
    for e in unique_entities:
        candidates_to_analyze[e.family_id] = {"type": "pending", "reason": ""}

    # Pre-encode current entity embedding for merge safety checks (once, not per-candidate)
    _current_entity_emb = prefetched_embedding
    if _current_entity_emb is None and storage.embedding_client and storage.embedding_client.is_available():
        try:
            _snip = llm_client.effective_entity_snippet_length()
            _embs = storage.embedding_client.encode(
                [f"{entity_name} {entity_content[:_snip]}"]
            )
            if _embs is not None:
                _current_entity_emb = np.array(_embs[0], dtype=np.float32)
        except Exception:
            pass

    # 输出初步筛选结果
    if entity_tree_log:
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
        if entity_tree_log:
            wprint_info(f"  │  ├─ 精细化判断开始（共 {len(candidates_to_analyze)} 个候选）")

    # Phase 1: Parallel LLM calls for detailed analysis
    # Limit to top 5 candidates to cap LLM calls (sorted by combined_score desc)
    _MAX_DETAILED_CANDIDATES = 5
    _detailed_tasks = []  # (cid, candidate_entity, candidate_info, future_or_result)
    _unique_by_fid = {e.family_id: e for e in unique_entities if hasattr(e, 'family_id') and e.family_id}
    _sorted_cids = list(candidates_to_analyze.items())
    if len(_sorted_cids) > _MAX_DETAILED_CANDIDATES:
        _sorted_cids = _sorted_cids[:_MAX_DETAILED_CANDIDATES]
        if entity_tree_log:
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
    from core.remember._shared import _get_entity_pool, _ENTITY_POOL_MAX
    _detailed_results: Dict[str, Optional[Dict]] = {}
    if len(_detailed_tasks) > 1:
        def _call_detailed(task):
            cid, cent, cinfo = task
            try:
                return (cid, llm_client.analyze_entity_pair_detailed(
                    current_entity_info, cinfo, [], context_text=context_text))
            except Exception as e:
                logger.warning("LLM detailed analysis failed for '%s' vs '%s': %s — skipping",
                               entity_name, cent.name, e)
                return (cid, None)
        pool = _get_entity_pool(min(3, _ENTITY_POOL_MAX[0]))
        for cid, result in pool.map(_call_detailed, _detailed_tasks):
            if result is not None:
                _detailed_results[cid] = result
    else:
        for cid, cent, cinfo in _detailed_tasks:
            try:
                _detailed_results[cid] = llm_client.analyze_entity_pair_detailed(
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
            _guard = alignment_guard_fn(
                entity_name, entity_content,
                candidate_entity.name, candidate_entity.content or "",
            )
            if _guard:
                _align_verdict, _align_confidence = _guard
                if entity_tree_log:
                    wprint_info(f"  │  │  ├─ 三值对齐: verdict={_align_verdict} (conf={_align_confidence:.2f}), 跳过")
                continue  # skip this candidate

            # 合并安全检查：Jaccard 名称相似度 < 0.3 或 embedding < 0.5 → 禁止合并
            _jaccard = calculate_jaccard_fn(entity_name, candidate_entity.name)
            if _jaccard < 0.3:
                if entity_tree_log:
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
                    _sim = cosine_similarity_fn(
                        _current_entity_emb,
                        _cand_emb,
                    )
                    if _sim < 0.5:
                        if entity_tree_log:
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
        if entity_tree_log:
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
                counts = storage.get_entity_version_counts(target_family_ids)
                target_version_counts = {tid: counts.get(tid, 0) for tid in target_family_ids}

                primary_target_id = max(target_family_ids, key=lambda tid: target_version_counts.get(tid, 0))

                # 输出多个合并目标的信息
                other_targets = [tid for tid in _target_set if tid != primary_target_id]
                if other_targets:
                    if entity_tree_log:
                        wprint_info(f"  │  ├─ 多合并目标: 选择 {primary_target_id} 为主要目标（版本数最多）")

                    # 在合并之前，先收集其他目标实体的信息（合并后这些ID就不存在了）
                    other_targets_entities.clear()  # 清空之前的数据
                    try:
                        other_entities_map = storage.get_entities_by_family_ids(other_targets)
                        for tid, other_entity in other_entities_map.items():
                            other_targets_entities[tid] = {
                                'entity': other_entity,
                                'name': other_entity.name,
                                'content': other_entity.content
                            }
                    except Exception:
                        # Fallback: individual fetch
                        for other_target_id in other_targets:
                            other_entity = storage.get_entity_by_family_id(other_target_id)
                            if other_entity:
                                other_targets_entities[other_target_id] = {
                                    'entity': other_entity,
                                    'name': other_entity.name,
                                    'content': other_entity.content
                                }

                    # 如果有多个不同的目标实体ID，说明这些实体都是同一个实体
                    # 需要将其他目标实体ID合并到主要目标ID
                    storage.merge_entity_families(primary_target_id, other_targets)

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
            latest_entity = storage.get_entity_by_family_id(primary_target_id)
            if latest_entity:
                # 防止同窗口重复版本化：如果该 family_id 已创建过版本，复用已有实体
                if already_versioned_family_ids and primary_target_id in already_versioned_family_ids:
                    if entity_tree_log:
                        wprint_info(f"  │  family_id {primary_target_id} 已在本次处理中创建版本，复用已有实体")
                    final_entity = latest_entity
                    entity_name_to_id[entity_name] = primary_target_id
                    entity_name_to_id[final_entity.name] = primary_target_id
                else:
                    pass

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
                        final_entity = create_entity_version_fn(
                            primary_target_id,
                            latest_entity.name,
                            latest_entity.content,
                            episode_id,
                            source_document,
                            base_time=base_time,
                            old_content=latest_entity.content or "",
                            old_content_format=latest_entity.content_format or "plain",
                        )
                        mark_versioned_fn(primary_target_id, already_versioned_family_ids, _version_lock)
                    else:
                        # 内容有差异 → 走完整合并流程
                        if entity_name != latest_entity.name:
                            merged_name = llm_client.merge_entity_name(
                                latest_entity.name,
                                entity_name
                            )
                        else:
                            merged_name = entity_name

                        merged_content = llm_client.merge_multiple_entity_contents(
                            contents_to_merge,
                            entity_sources=entity_sources_to_merge,
                            entity_names=entities_to_merge_names,
                        )
                        if entity_tree_log:
                            wprint_info(f"  │  ├─ 合并 {len(contents_to_merge)} 个实体的content: {', '.join(entities_to_merge_names[:3])}{'...' if len(entities_to_merge_names) > 3 else ''}")

                        final_entity = create_entity_version_fn(
                            primary_target_id,
                            merged_name,
                            merged_content,
                            episode_id,
                            source_document,
                            base_time=base_time,
                            old_content=latest_entity.content or "",
                            old_content_format=latest_entity.content_format or "plain",
                        )
                        mark_versioned_fn(primary_target_id, already_versioned_family_ids, _version_lock)

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

        if entity_tree_log:
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
            if entity_tree_log:
                wprint_info("  │  ⚠️ 合并决策存在但未生成最终实体，使用兜底逻辑")
            first_target_id = merge_decisions[0].get("target_family_id", "")
            if first_target_id:
                fallback_entity = storage.get_entity_by_family_id(first_target_id)
                if fallback_entity:
                    # 始终创建新版本（兜底路径也要版本化）
                    final_entity = create_entity_version_fn(
                        first_target_id,
                        entity_name,
                        entity_content,
                        episode_id,
                        source_document,
                        base_time=base_time,
                        old_content=fallback_entity.content or "",
                        old_content_format=fallback_entity.content_format or "plain",
                    )
                    mark_versioned_fn(first_target_id, already_versioned_family_ids, _version_lock)
                    entity_name_to_id[entity_name] = final_entity.family_id
                    entity_name_to_id[final_entity.name] = final_entity.family_id

        if not final_entity:
            # 没有匹配或兜底失败，创建新实体
            final_entity = create_new_entity_fn(entity_name, entity_content, episode_id, source_document, base_time=base_time)
            mark_versioned_fn(final_entity.family_id, already_versioned_family_ids, _version_lock)
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
    if entity_tree_log:
        if final_entity:
            if updated_relations:
                wprint_info(f"  └─ 完成: {final_entity.name} ({final_entity.family_id}), 关系 {len(updated_relations)} 个")
            else:
                wprint_info(f"  └─ 完成: {final_entity.name} ({final_entity.family_id})")
        else:
            if updated_relations:
                wprint_info(f"  └─ 完成: 关系 {len(updated_relations)} 个")

    return final_entity, updated_relations, entity_name_to_id


def _preprocess_extraction_context(extracted_entities, extracted_relations):
    """Build entity name set, relation pair set, and related-entity name set from extraction results."""
    extracted_entity_names = {e['name'] for e in extracted_entities}
    extracted_relation_pairs = set()
    related_entity_names = set()
    if extracted_relations:
        for rel in extracted_relations:
            entity1_name = (rel.get('entity1_name') or rel.get('from_entity_name', '')).strip()
            entity2_name = (rel.get('entity2_name') or rel.get('to_entity_name', '')).strip()
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

    def release_candidate_run_cache(self) -> None:
        """run 结束释放候选表 run 级投影缓存（orchestrator_pipeline 在所有窗口完成后调用）。"""
        self._candidate_builder.release_run_cache()

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
        full_texts = [f"# {e['name']}\n{e.get('content', '')[:snip]}" for e in extracted_entities]
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
                        already_versioned_family_ids: Optional[set] = None,
                        window_timings_ref: Optional[Dict[str, float]] = None) -> Tuple[List[Entity], List[Dict], Dict[str, str]]:
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
                    window_timings_ref=window_timings_ref,
                    window_batch_alignment=getattr(self, "window_batch_alignment_enabled", False),
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
                    window_timings_ref=window_timings_ref,
                    window_batch_alignment=getattr(self, "window_batch_alignment_enabled", False),
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
                        already_versioned_family_ids: Optional[set] = None,
                        window_timings_ref: Optional[Dict[str, float]] = None,
                        window_batch_alignment: bool = False) -> Tuple[List[Entity], List[Dict], Dict[str, str]]:
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
            window_timings_ref=window_timings_ref,
            window_batch_alignment=window_batch_alignment,
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
                        already_versioned_family_ids: Optional[set] = None,
                        window_timings_ref: Optional[Dict[str, float]] = None,
                        window_batch_alignment: bool = False) -> Tuple[List[Entity], List[Dict], Dict[str, str]]:
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
            window_timings_ref=window_timings_ref,
            window_batch_alignment=window_batch_alignment,
        )

    # 名称规范化：委托给共享模块
    _normalize_entity_name_for_matching = staticmethod(normalize_entity_name_for_matching)

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
        return _process_entity_sequential_fallback(
            storage=self.storage,
            llm_client=self.llm_client,
            entity_tree_log=self._entity_tree_log(),
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

    def _construct_entity(self, name: str, content: str, episode_id: str,
                          family_id: str, source_document: str = "",
                          base_time: Optional[datetime] = None,
                          confidence: Optional[float] = None) -> Entity:
        return _construct_entity(name, content, episode_id, family_id,
                                    source_document=source_document, base_time=base_time,
                                    confidence=confidence)

    def _build_new_entity(self, name: str, content: str, episode_id: str,
                          source_document: str = "", base_time: Optional[datetime] = None,
                          confidence: Optional[float] = None) -> Entity:
        return _build_new_entity(name, content, episode_id, source_document,
                                    base_time=base_time, confidence=confidence)

    def _gate_create_entity(self, name: str, content: str, episode_id: str,
                            source_document: str = "", base_time: Optional[datetime] = None,
                            confidence: Optional[float] = None,
                            judged_candidate_names=None) -> Entity:
        """创建新 family；FamilyWriteGate 并发竞态兜底（P4）。

        在写临界区内重验名称：若其他 worker 在本次候选检索之后新建了同名
        family，则改在该 family 下创建新版本（等价于"检索时看到它并复用"）。
        judged_candidate_names 是本次已检索/已裁决过的候选名——名字命中该集合
        时跳过 gate（候选已在场却被判 create_new，说明是"同名不同概念"，
        gate 不得覆盖该裁决）。
        """
        gate = getattr(self, "family_write_gate", None)
        if gate is None:
            return self._build_new_entity(name, content, episode_id, source_document,
                                          base_time=base_time, confidence=confidence)
        from core.judge.models import norm_name
        _norm = norm_name(name)
        _judged = {norm_name(n) for n in (judged_candidate_names or ()) if n}
        with gate.write_txn():
            existing_fid = None
            if _norm and _norm not in _judged:
                existing_fid = gate.resolve_name(name)
            if existing_fid:
                latest = None
                try:
                    latest = self.storage.get_entity_by_family_id(existing_fid)
                except Exception:
                    latest = None
                if latest is not None:
                    gate.register(latest.name, latest.family_id)
                    return self._build_entity_version(
                        latest.family_id, latest.name, latest.content or content,
                        episode_id, source_document, base_time=base_time,
                        old_content=latest.content or "",
                        old_content_format=latest.content_format or "plain")
                # 缓存命中的 fid 尚未提交（并发 worker 刚 register，save 还在门外）：
                # 不得落穿另建新 family——直接在该 fid 下建版本，两 worker 收敛同一家族
                return self._build_entity_version(
                    existing_fid, name, content, episode_id, source_document,
                    base_time=base_time, old_content="", old_content_format="plain")
            entity = self._build_new_entity(name, content, episode_id, source_document,
                                            base_time=base_time, confidence=confidence)
            gate.register(name, entity.family_id)
            return entity


    def _create_new_entity(self, name: str, content: str, episode_id: str,
                           source_document: str = "", base_time: Optional[datetime] = None,
                           confidence: Optional[float] = None) -> Entity:
        return _create_new_entity(self.storage, name, content, episode_id,
                                     source_document, base_time=base_time, confidence=confidence)

    def _build_entity_version(self, family_id: str, name: str, content: str,
                              episode_id: str, source_document: str = "",
                              base_time: Optional[datetime] = None,
                              old_content: str = "",
                              old_content_format: str = "plain") -> Entity:
        return _build_entity_version(family_id, name, content, episode_id,
                                        source_document, base_time=base_time,
                                        old_content=old_content,
                                        old_content_format=old_content_format)

    def _create_entity_version(self, family_id: str, name: str, content: str,
                              episode_id: str, source_document: str = "",
                              base_time: Optional[datetime] = None,
                              old_content: str = "",
                              old_content_format: str = "plain") -> Entity:
        return _create_entity_version(self.storage, family_id, name, content,
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
        return _compute_entity_patches(
            family_id=family_id,
            old_content=old_content,
            old_content_format=old_content_format,
            new_content=new_content,
            new_absolute_id=new_absolute_id,
            source_document=source_document,
            event_time=event_time,
        )
