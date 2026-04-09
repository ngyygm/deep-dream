"""知识图谱整理 mixin：从 orchestrator.py 提取。"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import threading
import uuid

from ..models import Episode, Entity
from ..utils import wprint


class _ConsolidationMixin:
    def consolidate_knowledge_graph_entity(self, verbose: bool = True,
                                    similarity_threshold: float = 0.6,
                                    max_candidates: int = 5,
                                    batch_candidates: Optional[int] = None,
                                    content_snippet_length: int = 64,
                                    parallel: bool = False,
                                    enable_name_match_step: bool = True,
                                    enable_pre_search: Optional[bool] = None) -> dict:
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

            for family_id, relations in self_ref_relations.items():
                # 获取实体的最新版本
                entity = self.storage.get_entity_by_family_id(family_id)
                if not entity:
                    continue

                # 收集所有自指向关系的content
                self_ref_contents = [rel['content'] for rel in relations]

                if verbose:
                    wprint(f"    处理实体 {entity.name} ({family_id})，有 {len(relations)} 个自指向关系")

                # 用LLM总结这些关系内容到实体的content中
                # 将自指向关系的内容视为实体的属性信息
                summarized_content = self.llm_client.merge_entity_content(
                    old_content=entity.content,
                    new_content="\n\n".join([f"属性信息：{content}" for content in self_ref_contents])
                )

                # 内容未变化则跳过版本创建
                if summarized_content.replace(' ', '').replace('\n', '') == entity.content.replace(' ', '').replace('\n', ''):
                    if verbose:
                        wprint(f"      内容未变化，跳过版本创建")
                    # 仍需删除自指向关系
                    actual_deleted = self.storage.delete_self_referential_relations()
                    if verbose:
                        wprint(f"    已删除 {actual_deleted} 个自指向关系")
                    entities_updated_from_self_ref += 1
                    continue

                # 更新实体的最新版本（创建新版本）
                from datetime import datetime
                new_family_id = f"entity_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                new_entity = Entity(
                    absolute_id=new_family_id,
                    family_id=entity.family_id,
                    name=entity.name,
                    content=summarized_content,
                    event_time=datetime.now(),
                    processed_time=datetime.now(),
                    episode_id=entity.episode_id,
                    source_document=entity.source_document if hasattr(entity, 'source_document') else ""
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
        family_ids = [entity.family_id for entity in all_entities]
        version_counts = self.storage.get_entity_version_counts(family_ids)
        all_entities.sort(key=lambda e: version_counts.get(e.family_id, 0), reverse=True)

        # 熔断检查：扫描异常膨胀实体（版本数 > 20 或不同名称 > 5）
        _MAX_VERSIONS = 20
        _MAX_DISTINCT_NAMES = 5
        for _entity in all_entities:
            _vc = version_counts.get(_entity.family_id, 0)
            if _vc > _MAX_VERSIONS:
                if verbose:
                    wprint(
                        f"  ⚠️ 实体 {_entity.name} ({_entity.family_id}) 有 {_vc} 个版本，"
                        f"可能存在过度合并"
                    )
            if _vc > _MAX_DISTINCT_NAMES:
                versions = self.storage.get_entity_versions(_entity.family_id)
                if versions:
                    distinct_names = len({v.name for v in versions})
                    if distinct_names > _MAX_DISTINCT_NAMES:
                        if verbose:
                            wprint(
                                f"  ⚠️ 实体 {_entity.family_id} 有 {distinct_names} 个不同名称，"
                                f"可能存在过度合并：{[v.name for v in versions[:10]]}"
                            )

        # 记录整理前的实体总数
        initial_entity_count = len(all_entities)
        if verbose:
            wprint(f"  整理前共有 {initial_entity_count} 个实体")

        # 记录已合并的实体ID（用于后续embedding搜索时排除）
        merged_family_ids = set()
        # 记录合并映射：source_family_id -> target_family_id
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
                    key=lambda e: version_counts.get(e.family_id, 0),
                    reverse=True
                )

            # 按照每个名称组中实体的最大版本数排序（从大到小），然后按顺序处理
            name_groups_sorted = sorted(
                name_to_entities.items(),
                key=lambda item: max(
                    (version_counts.get(e.family_id, 0) for e in item[1]),
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
                    if entity.family_id in merged_family_ids:
                        continue

                    version_count = version_counts.get(entity.family_id, 0)
                    entities_info.append({
                        "family_id": entity.family_id,
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
                    if entity.family_id in merged_family_ids:
                        continue
                    cache_text = self.storage.get_episode_text(entity.episode_id)
                    if cache_text:
                        memory_contexts[entity.family_id] = cache_text

                # 检查实体对之间是否已有关系，如果关系表示同一实体则直接合并
                family_ids_for_check = [info['family_id'] for info in entities_info]
                existing_relations_between = self._check_and_merge_entities_from_relations(
                    family_ids_for_check,
                    entities_info,
                    version_counts,
                    merged_family_ids,
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
                target_family_id = merge_group.get("target_family_id")
                source_family_ids = merge_group.get("source_family_ids", [])
                reason = merge_group.get("reason", "")

                if not target_family_id or not source_family_ids:
                    continue

                # 检查是否已被合并
                if any(sid in merged_family_ids for sid in source_family_ids):
                    continue

                # 不再过滤已有关系的实体对，让LLM判断是否应该合并
                # 即使有关系，如果关系表示同一实体，也应该合并

                # 执行合并
                merge_result = self.storage.merge_entity_families(target_family_id, source_family_ids)
                merge_result["reason"] = reason

                if verbose:
                    target_name = next((e.name for e in entities_with_same_name if e.family_id == target_family_id), target_family_id)
                    wprint(f"    合并实体: {target_name} ({target_family_id}) <- {len(source_family_ids)} 个源实体")
                    wprint(f"      原因: {reason}")

                # 处理合并后产生的自指向关系
                self._handle_self_referential_relations_after_merge(target_family_id, verbose)

                # 记录已合并的实体和合并映射
                for sid in source_family_ids:
                    merged_family_ids.add(sid)
                    merge_mapping[sid] = target_family_id

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

                # 如果两个实体合并后指向同一个目标实体，则跳过（自指向关系无意义）
                if actual_entity1_id == actual_entity2_id:
                    if verbose:
                        wprint(f"    跳过关系（合并后为同一实体）: {entity1_name} -> {entity2_name}")
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
                wprint(f"  名称匹配完成，处理了 {name_match_count} 个名称组，合并了 {len(merged_family_ids)} 个实体")
        else:
            if verbose:
                wprint(f"\n步骤1.5: 跳过（已禁用）")

        # 步骤1.5之后，重新按版本数量从大到小排序（因为合并可能改变了版本数）
        family_ids = [entity.family_id for entity in all_entities]
        version_counts = self.storage.get_entity_version_counts(family_ids)
        all_entities.sort(key=lambda e: version_counts.get(e.family_id, 0), reverse=True)

        # 用于累积所有分析过的实体信息（用于最终保存到JSON的text字段）
        all_analyzed_entities_text = []

        # 记录已处理的family_id对，避免重复分析
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
            for family_id in list(entity_to_candidates.keys()):
                # 如果当前实体已合并，从候选列表中移除
                if family_id in merged_family_ids:
                    del entity_to_candidates[family_id]
                    continue

                # 从候选列表中排除已合并的实体
                candidates = entity_to_candidates[family_id]
                entity_to_candidates[family_id] = candidates - merged_family_ids

            if verbose:
                total_candidates = sum(len(candidates) for candidates in entity_to_candidates.values())
                wprint(f"  预搜索完成，共 {len(entity_to_candidates)} 个实体，找到 {total_candidates} 个关联实体（已排除 {len(merged_family_ids)} 个已合并实体）")
        else:
            if verbose:
                wprint(f"\n步骤2: 跳过预搜索，将按需搜索每个实体的关联实体")

        if verbose:
            wprint(f"\n步骤3: 逐个实体分析并处理...")

        for family_idx, entity in enumerate(all_entities, 1):
            # 跳过已被合并的实体
            if entity.family_id in merged_family_ids:
                continue

            if verbose:
                # 获取实体的版本数
                entity_version_count = version_counts.get(entity.family_id, 0)
                wprint(f"\n  [{family_idx}/{len(all_entities)}] 分析实体: {entity.name} (family_id: {entity.family_id}, 版本数: {entity_version_count})")

            # 获取候选实体：如果启用了预搜索，从预搜索结果中获取；否则按需搜索
            if use_pre_search:
                candidate_family_ids = entity_to_candidates.get(entity.family_id, set())
            else:
                # 按需搜索：使用混合检索方式搜索当前实体的关联实体
                candidate_family_ids = set()

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

                # 合并候选实体并去重（按family_id去重，保留每个family_id的最新版本）
                candidate_dict = {}
                for candidate in candidates_name_jaccard + candidates_name_embedding + candidates_full_embedding:
                    if candidate.family_id == entity.family_id:
                        continue  # 跳过自己
                    if candidate.family_id not in candidate_dict:
                        candidate_dict[candidate.family_id] = candidate
                    else:
                        # 保留处理时间最新的
                        if candidate.processed_time > candidate_dict[candidate.family_id].processed_time:
                            candidate_dict[candidate.family_id] = candidate

                # 提取family_id到set中
                candidate_family_ids = {cid for cid in candidate_dict.keys()}

            # 过滤掉已处理的配对和已合并的实体
            candidate_family_ids = {
                cid for cid in candidate_family_ids
                if cid not in merged_family_ids and
                   (min(entity.family_id, cid), max(entity.family_id, cid)) not in processed_pairs
            }

            if not candidate_family_ids:
                if verbose:
                    wprint(f"    未找到相似实体候选")
                continue

            # 确定批量处理的大小
            if batch_candidates is not None and batch_candidates < max_candidates:
                batch_size = batch_candidates
            else:
                batch_size = max_candidates

            # 将候选实体转换为列表并分批处理
            candidate_family_ids_list = list(candidate_family_ids)
            total_candidates = len(candidate_family_ids_list)
            total_batches = (total_candidates + batch_size - 1) // batch_size  # 向上取整

            if verbose:
                wprint(f"    找到 {total_candidates} 个候选实体，将分 {total_batches} 批处理（每批 {batch_size} 个）")

            # 准备当前实体信息（所有批次共享）
            current_version_count = self.storage.get_entity_version_count(entity.family_id)
            current_entity_info = {
                "family_id": entity.family_id,
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
                batch_candidate_ids = candidate_family_ids_list[start_idx:end_idx]

                if verbose:
                    wprint(f"\n    [初步筛选] 第 {batch_idx + 1}/{total_batches} 批（{len(batch_candidate_ids)} 个候选实体）...")

                # 获取当前批次的候选实体完整信息
                candidates_info = []
                for cid in batch_candidate_ids:
                    candidate_entity = self.storage.get_entity_by_family_id(cid)
                    if candidate_entity:
                        version_count = self.storage.get_entity_version_count(cid)
                        info = {
                            "family_id": cid,
                            "name": candidate_entity.name,
                            "content": candidate_entity.content,
                            "version_count": version_count
                        }
                        candidates_info.append(info)
                        all_candidates_full_info[cid] = info
                        # 记录已处理的配对
                        pair = (min(entity.family_id, cid), max(entity.family_id, cid))
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
                        wprint(f"        - {info['name']} (family_id: {info['family_id']}, versions: {info['version_count']})")

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
                    cid = item.get("family_id") if isinstance(item, dict) else item
                    if cid and cid not in merged_family_ids:
                        all_possible_merges.append({
                            "current_family_id": entity.family_id,
                            "current_entity_info": current_entity_info,
                            "candidate_family_id": cid,
                            "reason": item.get("reason", "") if isinstance(item, dict) else ""
                        })

                for item in possible_relations:
                    cid = item.get("family_id") if isinstance(item, dict) else item
                    if cid and cid not in merged_family_ids:
                        all_possible_relations.append({
                            "current_family_id": entity.family_id,
                            "current_entity_info": current_entity_info,
                            "candidate_family_id": cid,
                            "reason": item.get("reason", "") if isinstance(item, dict) else ""
                        })

            # ========== 阶段2: 精细化判断（所有批次完成后） ==========
            # 对于被判断为关系的候选，先检查是否已有关系，如果有则跳过精细化判断
            filtered_possible_relations = []
            skipped_relations_count = 0
            for item in all_possible_relations:
                cid = item["candidate_family_id"]
                # 检查是否已有关系
                existing_rels = self.storage.get_relations_by_entities(
                    entity.family_id,
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
                            candidate_entity = self.storage.get_entity_by_family_id(cid)
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
                cid = item["candidate_family_id"]
                if cid not in all_candidates_to_analyze:
                    all_candidates_to_analyze[cid] = item

            # 对每个候选进行精细化判断
            merge_decisions = []  # 精细化判断后确定要合并的
            relation_decisions = []  # 精细化判断后确定要创建关系的

            for cid, item in all_candidates_to_analyze.items():
                if cid not in all_candidates_full_info:
                    continue

                # 检查是否已被合并
                if cid in merged_family_ids:
                    continue

                candidate_info = all_candidates_full_info[cid]

                # 获取两个实体之间的已有关系
                existing_rels = self.storage.get_relations_by_entities(
                    entity.family_id,
                    cid
                )
                existing_relations_list = []
                if existing_rels:
                    # 去重，每个family_id只保留最新版本
                    rel_dict = {}
                    for rel in existing_rels:
                        if rel.family_id not in rel_dict or rel.processed_time > rel_dict[rel.family_id].processed_time:
                            rel_dict[rel.family_id] = rel
                    for rel in rel_dict.values():
                        existing_relations_list.append({
                            "family_id": rel.family_id,
                            "content": rel.content
                        })

                # 获取上下文信息（优先使用当前实体的memory_cache，如果没有则使用候选实体的）
                context_text = None
                if entity.episode_id:
                    context_text = self.storage.get_episode_text(entity.episode_id)
                if not context_text:
                    candidate_entity = self.storage.get_entity_by_family_id(cid)
                    if candidate_entity and candidate_entity.episode_id:
                        context_text = self.storage.get_episode_text(candidate_entity.episode_id)

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
                            merge_target = entity.family_id
                        else:
                            merge_target = cid

                    merge_decisions.append({
                        "target_family_id": merge_target,
                        "source_family_id": cid if merge_target == entity.family_id else entity.family_id,
                        "source_name": candidate_info["name"],
                        "target_name": entity.name if merge_target == entity.family_id else candidate_info["name"],
                        "reason": reason
                    })
                elif action == "create_relation":
                    relation_content = detailed_result.get("relation_content", "")
                    relation_decisions.append({
                        "entity1_id": entity.family_id,
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
            all_analyzed_entities_text.append(f"\n\n{'='*80}\n分析实体: {entity.name} ({entity.family_id})\n{'='*80}\n")
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
                    if mg["target_family_id"] == md["target_family_id"]:
                        if md["source_family_id"] not in mg["source_family_ids"]:
                            mg["source_family_ids"].append(md["source_family_id"])
                            mg["reason"] += f"; {md['reason']}"
                        found = True
                        break
                if not found:
                    merge_groups.append({
                        "target_family_id": md["target_family_id"],
                        "source_family_ids": [md["source_family_id"]],
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
                all_merge_entity_families = set()
                merge_reasons = []

                for merge_info in merge_groups:
                    target_id = merge_info.get("target_family_id")
                    source_ids = merge_info.get("source_family_ids", [])
                    reason = merge_info.get("reason", "")

                    if not target_id or not source_ids:
                        continue

                    # 检查是否已被合并
                    if any(sid in merged_family_ids for sid in source_ids):
                        if verbose:
                            wprint(f"        跳过已合并的实体: {source_ids}")
                        continue

                    # 不再过滤已有关系的实体对，让LLM判断是否应该合并
                    # 即使有关系，如果关系表示同一实体，也应该合并

                    # 收集所有需要合并的实体
                    all_merge_entity_families.add(target_id)
                    all_merge_entity_families.update(source_ids)
                    if reason:
                        merge_reasons.append(reason)

                if all_merge_entity_families:
                    # 确定最终的target：选择版本数最多的实体
                    target_candidates = []
                    for eid in all_merge_entity_families:
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
                        target_entity = self.storage.get_entity_by_family_id(final_target_id)
                        target_name = target_entity.name if target_entity else final_target_id

                        # 合并所有原因
                        combined_reason = "；".join(merge_reasons) if merge_reasons else "多个实体需要合并"

                        if verbose:
                            wprint(f"      合并多个实体到目标实体:")
                            wprint(f"        目标: {target_name} ({final_target_id}, 版本数: {final_target_versions})")
                            merge_names = [f"{self.storage.get_entity_by_family_id(sid).name} ({sid})" if self.storage.get_entity_by_family_id(sid) else sid for sid in final_source_ids]
                            wprint(f"        源实体: {', '.join(merge_names)}")
                            wprint(f"        原因: {combined_reason}")

                        # 执行合并（一次性合并所有source到target）
                        merge_result = self.storage.merge_entity_families(final_target_id, final_source_ids)
                        merge_result["reason"] = combined_reason
                        merge_result["target_versions"] = final_target_versions

                        if verbose:
                            wprint(f"        结果: 更新了 {merge_result.get('entities_updated', 0)} 条实体记录")

                        # 处理合并后产生的自指向关系
                        self._handle_self_referential_relations_after_merge(final_target_id, verbose)

                        # 记录已合并的family_id
                        for sid in final_source_ids:
                            merged_family_ids.add(sid)
                            all_merged_in_this_round.add(sid)

                        result["merge_details"].append(merge_result)
                        result["entities_merged"] += merge_result.get("entities_updated", 0)

                # 立即创建关联关系（步骤4）
                if alias_relations:
                    if verbose:
                        wprint(f"      创建关联关系...")
                        if self.llm_threads > 1 and len(alias_relations) > 1:
                            wprint(f"      使用 {self.llm_threads} 个线程并行处理 {len(alias_relations)} 个关系...")

                    # 构建有效的family_id映射（用于验证LLM返回的ID是否有效）
                    valid_family_ids = {e["family_id"] for e in entities_for_analysis}
                    family_id_entity2_name = {e["family_id"]: e["name"] for e in entities_for_analysis}

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
                                wprint(f"          跳过：缺少family_id (entity1: {entity1_id}, entity2: {entity2_id})")
                            continue

                        # 验证family_id是否在传入的实体列表中
                        if entity1_id not in valid_family_ids:
                            if verbose:
                                wprint(f"          警告：entity1_id {entity1_id} 不在分析列表中，尝试通过名称查找...")
                            # 尝试通过名称查找实体
                            found_entity = None
                            for e in entities_for_analysis:
                                if e["name"] == entity1_name:
                                    found_entity = e
                                    break
                            if found_entity:
                                entity1_id = found_entity["family_id"]
                                if verbose:
                                    wprint(f"            通过名称找到实体: {entity1_name} -> {entity1_id}")
                            else:
                                if verbose:
                                    wprint(f"            无法找到实体: {entity1_name} ({entity1_id})")
                                continue

                        if entity2_id not in valid_family_ids:
                            if verbose:
                                wprint(f"          警告：entity2_id {entity2_id} 不在分析列表中，尝试通过名称查找...")
                            # 尝试通过名称查找实体
                            found_entity = None
                            for e in entities_for_analysis:
                                if e["name"] == entity2_name:
                                    found_entity = e
                                    break
                            if found_entity:
                                entity2_id = found_entity["family_id"]
                                if verbose:
                                    wprint(f"            通过名称找到实体: {entity2_name} -> {entity2_id}")
                            else:
                                if verbose:
                                    wprint(f"            无法找到实体: {entity2_name} ({entity2_id})")
                                continue

                        # 检查实体是否在本次循环中被合并（如果被合并，需要使用合并后的family_id）
                        actual_entity1_id = entity1_id
                        actual_entity2_id = entity2_id

                        # 注：不再使用 final_target_id 直接映射（会导致不同 merge target 的实体
                        # 被错误映射到同一个 target）。统一由下方 merge_details 查找逻辑处理。

                        # 检查实体是否在之前的循环中已被合并
                        # 如果family_id在merged_family_ids中，说明已经被合并，需要找到合并后的family_id
                        if entity1_id in merged_family_ids:
                            # 从merge_details中查找该实体被合并到哪个target
                            found_target = None
                            for merge_detail in result["merge_details"]:
                                if entity1_id in merge_detail.get("merged_source_ids", []):
                                    found_target = merge_detail.get("target_family_id")
                                    break
                            if found_target:
                                actual_entity1_id = found_target
                                if verbose:
                                    wprint(f"            注意：entity1实体 {entity1_name} ({entity1_id}) 已被合并到 {found_target}")
                            else:
                                # 如果找不到，尝试查询数据库（可能family_id已经更新）
                                entity1_db = self.storage.get_entity_by_family_id(entity1_id)
                                if entity1_db:
                                    actual_entity1_id = entity1_db.family_id

                        if entity2_id in merged_family_ids:
                            # 从merge_details中查找该实体被合并到哪个target
                            found_target = None
                            for merge_detail in result["merge_details"]:
                                if entity2_id in merge_detail.get("merged_source_ids", []):
                                    found_target = merge_detail.get("target_family_id")
                                    break
                            if found_target:
                                actual_entity2_id = found_target
                                if verbose:
                                    wprint(f"            注意：entity2实体 {entity2_name} ({entity2_id}) 已被合并到 {found_target}")
                            else:
                                # 如果找不到，尝试查询数据库（可能family_id已经更新）
                                entity2_db = self.storage.get_entity_by_family_id(entity2_id)
                                if entity2_db:
                                    actual_entity2_id = entity2_db.family_id

                        # 验证最终的family_id是否有效
                        entity1_check = self.storage.get_entity_by_family_id(actual_entity1_id)
                        entity2_check = self.storage.get_entity_by_family_id(actual_entity2_id)

                        if not entity1_check:
                            if verbose:
                                wprint(f"          错误：无法找到entity1实体 (family_id: {actual_entity1_id}, name: {entity1_name})")
                            continue

                        if not entity2_check:
                            if verbose:
                                wprint(f"          错误：无法找到entity2实体 (family_id: {actual_entity2_id}, name: {entity2_name})")
                            continue

                        # 如果合并后entity1和entity2是同一个实体，跳过创建关系
                        if actual_entity1_id == actual_entity2_id:
                            if verbose:
                                wprint(f"          跳过：合并后entity1和entity2是同一实体")
                            continue

                        if verbose:
                            wprint(f"          准备处理关系: {entity1_name} -> {entity2_name}")
                            if actual_entity1_id != entity1_id or actual_entity2_id != entity2_id:
                                wprint(f"            注意：使用了合并后的family_id (entity1: {entity1_id}->{actual_entity1_id}, entity2: {entity2_id}->{actual_entity2_id})")

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
        family_ids = [entity.family_id for entity in all_entities]
        version_counts = self.storage.get_entity_version_counts(family_ids)
        all_entities.sort(key=lambda e: version_counts.get(e.family_id, 0), reverse=True)

        initial_entity_count = len(all_entities)
        if verbose:
            wprint(f"  整理前共有 {initial_entity_count} 个实体")

        # 步骤1.5：先按名称完全匹配进行整理
        if verbose:
            wprint(f"\n步骤1.5: 按名称完全匹配进行初步整理...")

        # 记录已合并的实体ID（用于后续embedding搜索时排除）
        merged_family_ids = set()
        # 记录合并映射：source_family_id -> target_family_id
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
                key=lambda e: version_counts.get(e.family_id, 0),
                reverse=True
            )

        # 按照每个名称组中实体的最大版本数排序（从大到小），然后按顺序处理
        name_groups_sorted = sorted(
            name_to_entities.items(),
            key=lambda item: max(
                (version_counts.get(e.family_id, 0) for e in item[1]),
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
                if entity.family_id in merged_family_ids:
                    continue

                version_count = version_counts.get(entity.family_id, 0)
                entities_info.append({
                    "family_id": entity.family_id,
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
                if entity.family_id in merged_family_ids:
                    continue
                cache_text = self.storage.get_episode_text(entity.episode_id)
                if cache_text:
                    memory_contexts[entity.family_id] = cache_text

            # 检查实体对之间是否已有关系，如果关系表示同一实体则直接合并
            family_ids_for_check = [info['family_id'] for info in entities_info]
            existing_relations_between = self._check_and_merge_entities_from_relations(
                family_ids_for_check,
                entities_info,
                version_counts,
                merged_family_ids,
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
                target_family_id = merge_group.get("target_family_id")
                source_family_ids = merge_group.get("source_family_ids", [])
                reason = merge_group.get("reason", "")

                if not target_family_id or not source_family_ids:
                    continue

                # 检查是否已被合并
                if any(sid in merged_family_ids for sid in source_family_ids):
                    continue

                # 不再过滤已有关系的实体对，让LLM判断是否应该合并
                # 即使有关系，如果关系表示同一实体，也应该合并

                # 执行合并
                merge_result = self.storage.merge_entity_families(target_family_id, source_family_ids)
                merge_result["reason"] = reason

                if verbose:
                    target_name = next((e.name for e in entities_with_same_name if e.family_id == target_family_id), target_family_id)
                    wprint(f"    合并实体: {target_name} ({target_family_id}) <- {len(source_family_ids)} 个源实体")
                    wprint(f"      原因: {reason}")

                # 处理合并后产生的自指向关系
                self._handle_self_referential_relations_after_merge(target_family_id, verbose)

                # 记录已合并的实体和合并映射
                for sid in source_family_ids:
                    merged_family_ids.add(sid)
                    merge_mapping[sid] = target_family_id

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

                # 如果两个实体合并后指向同一个目标实体，则跳过（自指向关系无意义）
                if actual_entity1_id == actual_entity2_id:
                    if verbose:
                        wprint(f"    跳过关系（合并后为同一实体）: {entity1_name} -> {entity2_name}")
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
            wprint(f"  名称匹配完成，处理了 {name_match_count} 个名称组，合并了 {len(merged_family_ids)} 个实体")

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
        for family_id in list(entity_to_candidates.keys()):
            # 如果当前实体已合并，从候选列表中移除
            if family_id in merged_family_ids:
                del entity_to_candidates[family_id]
                continue

            # 从候选列表中排除已合并的实体
            candidates = entity_to_candidates[family_id]
            entity_to_candidates[family_id] = candidates - merged_family_ids

        if verbose:
            total_candidates = sum(len(candidates) for candidates in entity_to_candidates.values())
            wprint(f"  预搜索完成，共 {len(entity_to_candidates)} 个实体，找到 {total_candidates} 个关联实体（已排除 {len(merged_family_ids)} 个已合并实体）")

        # 步骤3：并行处理实体
        if verbose:
            wprint(f"\n步骤3: 并行处理实体（{self.llm_threads}个线程）...")

        # 共享状态（需要加锁）
        # merged_family_ids 已经在步骤1.5中初始化，这里只需要创建锁
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
                        if entity.family_id in merged_family_ids:
                            pending_entities.pop(i)
                            continue

                    # 获取关联实体
                    candidates = entity_to_candidates.get(entity.family_id, set())

                    # 过滤掉已合并的关联实体
                    with merged_ids_lock:
                        candidates = candidates - merged_family_ids

                    # 检查是否与正在处理的实体冲突
                    all_ids = {entity.family_id} | candidates
                    with in_progress_lock:
                        if all_ids & in_progress_ids:
                            continue  # 有冲突，跳过

                        # 标记为正在处理
                        in_progress_ids.update(all_ids)

                    # 找到了可以处理的实体
                    pending_entities.pop(i)
                    return entity, candidates

            return None, None

        def release_entity(family_id, candidate_ids):
            """释放实体的处理权"""
            all_ids = {family_id} | candidate_ids
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
                        if (min(entity.family_id, cid), max(entity.family_id, cid)) not in processed_pairs
                    }
                    # 记录配对
                    for cid in filtered_candidates:
                        processed_pairs.add((min(entity.family_id, cid), max(entity.family_id, cid)))

                if not filtered_candidates:
                    return task_result

                # 获取候选实体的完整信息
                candidates_info = []
                for cid in filtered_candidates:
                    candidate_entity = self.storage.get_entity_by_family_id(cid)
                    if candidate_entity:
                        version_count = self.storage.get_entity_version_count(cid)
                        candidates_info.append({
                            "family_id": cid,
                            "name": candidate_entity.name,
                            "content": candidate_entity.content,
                            "version_count": version_count
                        })

                if not candidates_info:
                    return task_result

                # 准备当前实体信息
                current_version_count = self.storage.get_entity_version_count(entity.family_id)
                current_entity_info = {
                    "family_id": entity.family_id,
                    "name": entity.name,
                    "content": entity.content,
                    "version_count": current_version_count
                }

                entities_for_analysis = [current_entity_info] + candidates_info

                # 获取记忆上下文
                memory_contexts = {}
                cache_text = self.storage.get_episode_text(entity.episode_id)
                if cache_text:
                    memory_contexts[entity.family_id] = cache_text

                for info in candidates_info:
                    candidate_entity = self.storage.get_entity_by_family_id(info["family_id"])
                    if candidate_entity:
                        c_text = self.storage.get_episode_text(candidate_entity.episode_id)
                        if c_text:
                            memory_contexts[info["family_id"]] = c_text

                # 检查实体对之间是否已有关系，如果关系表示同一实体则直接合并
                analysis_family_ids = [info['family_id'] for info in entities_for_analysis]
                existing_relations_between = self._check_and_merge_entities_from_relations(
                    analysis_family_ids,
                    entities_for_analysis,
                    version_counts,
                    merged_family_ids,
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
                    target_family_id = merge_group.get("target_family_id")
                    source_family_ids = merge_group.get("source_family_ids", [])
                    reason = merge_group.get("reason", "")

                    if not target_family_id or not source_family_ids:
                        continue

                    # 检查是否已被合并
                    with merged_ids_lock:
                        if any(sid in merged_family_ids for sid in source_family_ids):
                            continue

                    # 不再过滤已有关系的实体对，让LLM判断是否应该合并
                    # 即使有关系，如果关系表示同一实体，也应该合并

                    # 执行合并
                    merge_result = self.storage.merge_entity_families(target_family_id, source_family_ids)
                    merge_result["reason"] = reason

                    # 处理合并后产生的自指向关系
                    self._handle_self_referential_relations_after_merge(target_family_id, verbose=False)

                    task_result["merge_details"].append(merge_result)
                    task_result["entities_merged"] += merge_result.get("entities_updated", 0)

                    # 记录已合并的实体，并同步到共享 merge_mapping
                    for sid in source_family_ids:
                        task_result["merged_ids"].add(sid)
                        with merged_ids_lock:
                            merged_family_ids.add(sid)
                            merge_mapping[sid] = target_family_id

                # 处理关系 — 使用 merge_mapping 解析正确的端点
                for alias_info in alias_relations:
                    entity1_id = alias_info.get("entity1_id")
                    entity2_id = alias_info.get("entity2_id")
                    entity1_name = alias_info.get("entity1_name", "")
                    entity2_name = alias_info.get("entity2_name", "")
                    preliminary_content = alias_info.get("content")

                    if not entity1_id or not entity2_id:
                        continue

                    # 应用 merge_mapping 解析到正确的目标实体
                    actual_entity1_id = merge_mapping.get(entity1_id, entity1_id)
                    actual_entity2_id = merge_mapping.get(entity2_id, entity2_id)

                    # 跳过自指向关系（合并后两端可能指向同一实体）
                    if actual_entity1_id == actual_entity2_id:
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
                            merged_family_ids.update(task_result["merged_ids"])

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
                        release_entity(entity.family_id, candidates)

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
