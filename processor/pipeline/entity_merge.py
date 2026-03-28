"""实体合并 mixin：从 orchestrator.py 提取。"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List
import uuid

from ..models import Entity
from ..utils import wprint


class _EntityMergeMixin:
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
                # 应用合并映射：如果关系中的实体ID已被合并，更新为新ID
                resolved_e1 = merge_mapping.get(entity1_id, entity1_id)
                resolved_e2 = merge_mapping.get(entity2_id, entity2_id)
                if resolved_e1 == resolved_e2:
                    continue  # 合并后变成自指向，跳过
                new_pair_key = f"{resolved_e1}|{resolved_e2}" if resolved_e1 < resolved_e2 else f"{resolved_e2}|{resolved_e1}"
                if new_pair_key in filtered_existing_relations:
                    filtered_existing_relations[new_pair_key].extend(filtered_relations)
                else:
                    filtered_existing_relations[new_pair_key] = filtered_relations

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
