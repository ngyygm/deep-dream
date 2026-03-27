"""关系操作 mixin：从 orchestrator.py 提取。"""
from __future__ import annotations

from typing import Dict, List, Optional
from datetime import datetime
import hashlib
import uuid

from ..models import MemoryCache
from ..utils import wprint


class _RelationOpsMixin:
    # ──────────────────────────────────────────────
    # 关系操作方法（从 orchestrator.py 提取）
    # ──────────────────────────────────────────────

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
                            if rel.physical_time > relation_dict[rel.relation_id].physical_time:
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
                        if rel.physical_time > relation_dict[rel.relation_id].physical_time:
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
                # 从实体中获取文档名（如果实体有source_document，使用第一个实体的source_document）
                source_document_from_entity = entity1.source_document if hasattr(entity1, 'source_document') and entity1.source_document else ""

                relation_memory_cache = MemoryCache(
                    absolute_id=f"cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                    content=relation_memory_cache_content,
                    physical_time=datetime.now(),
                    source_document=source_document_from_entity,
                    activity_type="知识图谱整理-关系生成"
                )
                # 保存memory_cache
                rel_doc_hash = hashlib.md5(cache_text_content.encode("utf-8")).hexdigest()[:12] if cache_text_content else ""
                self.storage.save_memory_cache(relation_memory_cache, text=cache_text_content, doc_hash=rel_doc_hash)

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
                    source_document=source_document_from_entity,
                    base_time=relation_memory_cache.physical_time,
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
            physical_time=datetime.now(),
            source_document="",  # 知识图谱整理总结不关联特定文档
            activity_type="知识图谱整理总结"
        )

        # 保存总结记忆缓存
        summary_doc_hash = hashlib.md5(consolidation_text.encode("utf-8")).hexdigest()[:12] if consolidation_text else ""
        self.storage.save_memory_cache(
            summary_cache,
            text=consolidation_text,
            doc_hash=summary_doc_hash
        )

        if verbose:
            wprint(f"  已创建整理总结记忆缓存: {summary_cache.absolute_id}")

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
