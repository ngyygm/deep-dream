"""
Sequential entity processing logic.
Extracted from EntityProcessor for modularity.
"""
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging
import numpy as np

from core.models import Entity, Episode
from core.storage.sqlite.manager import SQLiteGraphStorageManager as Neo4jStorageManager
from core.llm.client import LLMClient
from core.utils import wprint_info
from core.debug_log import log_struct as _dbg_struct
from core.remember._shared import _doc_basename

logger = logging.getLogger(__name__)


def _process_entity_sequential_fallback(
    storage: Neo4jStorageManager,
    llm_client: LLMClient,
    entity_tree_log: bool,
    search_entity_candidates_fn,  # callable for _search_entity_candidates
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
    entity_content = extracted_entity['content']

    # 显示进度信息
    if entity_tree_log:
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
        similar_entities = search_entity_candidates_fn(
            entity_name, entity_content, similarity_threshold,
            jaccard_search_threshold, embedding_name_search_threshold,
            embedding_full_search_threshold,
            extracted_entity_names, extracted_relation_pairs,
        )
        version_counts = {}

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
                    merge_result = storage.merge_entity_families(primary_target_id, other_targets)

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
