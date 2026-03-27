"""LLM客户端 - 关系抽取相关操作。"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from ..models import MemoryCache, Entity
from ..debug_log import log as dbg
from ..utils import wprint
from .prompts import (
    EXTRACT_RELATIONS_SINGLE_PASS_SYSTEM_PROMPT,
    JUDGE_NEED_CREATE_RELATION_SYSTEM_PROMPT,
    GENERATE_RELATION_CONTENT_SYSTEM_PROMPT,
)


class _RelationExtractionMixin:
    """关系抽取相关的 LLM 操作（mixin，通过 LLMClient 多继承使用）。"""

    def extract_relations(self, memory_cache: MemoryCache, input_text: str,
                         entities: Union[List[Dict[str, str]], List[Entity]],
                         rounds: int = 1,
                         verbose: bool = False,
                         on_round_done=None) -> List[Dict[str, str]]:
        """
        抽取概念关系边，支持多轮次补充抽取（利用 LLM 对话历史）。

        Args:
            memory_cache: 当前的记忆缓存
            input_text: 当前窗口的输入文本
            entities: 实体列表，可以是Dict（name, content）或Entity对象（包含entity_id）
            rounds: 抽取轮次（默认 1）；>1 时利用对话历史要求 LLM 继续补充
            verbose: 是否输出详细日志
            on_round_done: 每轮完成后的回调 fn(round_idx, total_rounds, cumulative_count)

        Returns:
            抽取的关系列表，每个关系包含 entity1_name, entity2_name, content
        """
        if not entities:
            return []

        # 统一转换为实体信息字典格式
        entity_info_list = []
        for entity in entities:
            if isinstance(entity, Entity):
                entity_info = {
                    'name': entity.name,
                    'content': entity.content,
                    'entity_id': entity.entity_id
                }
            else:
                entity_info = {
                    'name': entity['name'],
                    'content': entity['content'],
                    'entity_id': None
                }
            entity_info_list.append(entity_info)

        system_prompt = EXTRACT_RELATIONS_SINGLE_PASS_SYSTEM_PROMPT

        # 构建实体列表字符串
        entity_lines = []
        for e in entity_info_list:
            name = e.get('name', '').strip()
            content = e.get('content', '').strip()
            if content:
                snippet = content[:self.content_snippet_length] + ('...' if len(content) > self.content_snippet_length else '')
                entity_lines.append(f"- {name}：{snippet}")
            else:
                entity_lines.append(f"- {name}")
        entities_str = '\n'.join(entity_lines)

        first_prompt = f"""<记忆缓存>
{memory_cache.content}
</记忆缓存>

<输入文本>
{input_text}
</输入文本>

<概念实体列表>
{entities_str}
</概念实体列表>

请从文本中抽取所有概念实体间的关系（越多越好）："""

        all_relations: List[Dict[str, str]] = []
        seen_rel_keys: set = set()  # (entity1, entity2, content_hash)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_prompt},
        ]

        # 收集所有有效实体名称，用于规范化
        valid_entity_names = {e.get('name', '').strip() for e in entity_info_list if e.get('name', '').strip()}

        for round_idx in range(max(1, rounds)):
            if verbose:
                wprint(f"      关系抽取第 {round_idx + 1}/{rounds} 轮...")

            response = self._call_llm("", messages=messages)
            dbg(f"关系抽取第{round_idx+1}轮 LLM原始响应 ({len(response)} 字符): {response[:800]}")
            new_relations = self._parse_relations_response(response)

            # 对实体名称进行规范化
            for rel in new_relations:
                original_e1 = rel.get('entity1_name', '')
                original_e2 = rel.get('entity2_name', '')
                normalized_e1 = self._normalize_entity_name_to_original(original_e1, valid_entity_names)
                normalized_e2 = self._normalize_entity_name_to_original(original_e2, valid_entity_names)
                if normalized_e1 != original_e1 or normalized_e2 != original_e2:
                    rel['entity1_name'] = normalized_e1
                    rel['entity2_name'] = normalized_e2

            # 追加 assistant 回复到对话历史
            messages.append({"role": "assistant", "content": response})

            # 去重合并
            new_count = 0
            for rel in new_relations:
                e1 = rel.get('entity1_name', '').strip()
                e2 = rel.get('entity2_name', '').strip()
                content = rel.get('content', '').strip()
                if not e1 or not e2 or not content:
                    continue
                # 标准化实体对
                if e1 > e2:
                    e1, e2 = e2, e1
                content_hash = hash(content.strip().lower())
                key = (e1, e2, content_hash)
                if key not in seen_rel_keys:
                    seen_rel_keys.add(key)
                    all_relations.append({
                        'entity1_name': e1,
                        'entity2_name': e2,
                        'content': content,
                    })
                    new_count += 1

            if verbose:
                wprint(f"        第 {round_idx + 1} 轮完成：新增 {new_count} 关系，累计 {len(all_relations)} 关系")

            if on_round_done:
                on_round_done(round_idx + 1, rounds, len(all_relations))

            if new_count == 0:
                if verbose:
                    wprint(f"        本轮无新增关系，停止抽取")
                break

            if round_idx + 1 < rounds:
                messages.append({"role": "user", "content": "请继续从文本中补充更多概念实体间的关系，不要重复已提取的内容。"})

        # 多轮次：保存最后一轮的完整 messages（含对话历史）
        if self._distill_data_dir and self._current_distill_step and messages:
            self._save_distill_conversation(messages)

        wprint(f"[TMG] 关系抽取完成: 共 {len(all_relations)} 个关系 "
              f"(实体数: {len(entity_info_list)}, 轮次: {max(1, rounds)})")
        if len(all_relations) == 0 and len(entity_info_list) > 1:
            wprint(f"[TMG]   警告: 有 {len(entity_info_list)} 个实体但未抽取到任何关系，"
                  f"可能LLM未返回有效JSON或文本中无明确关系")
        return all_relations

    def _extract_relations_single_pass(self, memory_cache: MemoryCache,
                                      input_text: str,
                                      entities: List[Dict[str, Any]],
                                      existing_relations: Optional[Dict[tuple, List[str]]] = None,
                                      uncovered_entities: Optional[List[str]] = None,
                                      verbose: bool = False) -> List[Dict[str, str]]:
        """
        单次关系抽取

        Args:
            memory_cache: 记忆缓存
            input_text: 输入文本
            entities: 实体信息列表（包含name, content, entity_id）
            existing_relations: 已抽取的关系字典，key是(entity1, entity2)，value是关系content列表
            uncovered_entities: 未覆盖的实体名称列表（还没有任何关系，需要优先补全）
            verbose: 是否输出详细日志
        """
        system_prompt = EXTRACT_RELATIONS_SINGLE_PASS_SYSTEM_PROMPT

        # 构建实体名称集合，用于区分已覆盖和未覆盖的实体
        uncovered_set = set(uncovered_entities) if uncovered_entities else set()

        # 辅助函数：格式化实体信息
        def format_entity(e: Dict[str, Any]) -> str:
            name = e.get('name', '').strip()
            content = e.get('content', '').strip()
            if content:
                snippet = content[:self.content_snippet_length] + ('...' if len(content) > self.content_snippet_length else '')
                return f"- {name}：{snippet}"
            return f"- {name}"

        # 分离未覆盖实体和已覆盖实体
        uncovered_entities_list = []
        covered_entities_list = []
        for e in entities:
            name = e.get('name', '').strip()
            if not name:
                continue
            if name in uncovered_set:
                uncovered_entities_list.append(format_entity(e))
            else:
                covered_entities_list.append(format_entity(e))

        # 构建prompt各部分
        prompt_parts = []

        # 记忆缓存
        prompt_parts.append(f"<记忆缓存>\n{memory_cache.content}\n</记忆缓存>")

        # 已抽取的关系（如果有，且所有实体都已覆盖）
        if existing_relations:
            relations_info = []
            for (e1, e2), contents in existing_relations.items():
                if contents:
                    relations_info.append(f"- {e1} ↔ {e2}：{' / '.join(contents)}")
            if relations_info:
                prompt_parts.append(f"<已有关系>\n" + "\n".join(relations_info) + "\n</已有关系>")

        # 输入文本
        prompt_parts.append(f"<输入文本>\n{input_text}\n</输入文本>")

        # 优先实体放在最后（权重更高）
        if uncovered_entities_list:
            uncovered_names = [e.get('name', '').strip() for e in entities if e.get('name', '').strip() in uncovered_set]
            prompt_parts.append("<未覆盖实体>\n\n" + "\n".join(uncovered_names) + "\n\n</未覆盖实体>\n\n【注意】输出时entity1_name和entity2_name其中一个必须与上面的概念实体名称【完全一致】，包括括号中的说明文字，不能简化！")

        # 简洁的任务说明
        task_instruction = "请抽取概念实体间的关系。"
        if existing_relations:
            task_instruction += "不要重复已有关系。"
        prompt_parts.append(task_instruction)

        prompt = "\n\n".join(prompt_parts)

        if verbose:
            wprint(f"          正在调用LLM进行关系抽取（实体数量: {len(entities)}）...")
        response = self._call_llm(prompt, system_prompt)
        if verbose:
            wprint(f"          LLM调用完成，正在解析结果...")
        result = self._parse_relations_response(response)

        # 对实体名称进行规范化，将LLM可能简化的名称映射回原始完整名称
        valid_entity_names = {e.get('name', '').strip() for e in entities if e.get('name', '').strip()}
        normalized_count = 0
        for rel in result:
            original_e1 = rel.get('entity1_name', '')
            original_e2 = rel.get('entity2_name', '')
            normalized_e1 = self._normalize_entity_name_to_original(original_e1, valid_entity_names)
            normalized_e2 = self._normalize_entity_name_to_original(original_e2, valid_entity_names)
            if normalized_e1 != original_e1 or normalized_e2 != original_e2:
                rel['entity1_name'] = normalized_e1
                rel['entity2_name'] = normalized_e2
                normalized_count += 1

        if verbose:
            if normalized_count > 0:
                wprint(f"          解析完成，获得 {len(result)} 个关系（规范化了 {normalized_count} 个实体名称）")
            else:
                wprint(f"          解析完成，获得 {len(result)} 个关系")
        return result

    def _deduplicate_relations(self, relations: List[Dict[str, str]],
                               seen_relations: set) -> List[Dict[str, str]]:
        """
        对关系进行去重（代码层面的去重规则）
        关系是无向的，将(A,B)和(B,A)视为同一个关系

        Args:
            relations: 关系列表
            seen_relations: 已见过的关系集合，元素是 (entity1_name, entity2_name, content_hash)

        Returns:
            去重后的关系列表
        """
        deduplicated = []
        for rel in relations:
            entity1_name = rel.get('entity1_name', '').strip()
            entity2_name = rel.get('entity2_name', '').strip()
            content = rel.get('content', '').strip()

            if not entity1_name or not entity2_name or not content:
                continue

            # 标准化实体对（按字母顺序排序，使关系无向化）
            normalized_pair = self._normalize_entity_pair(entity1_name, entity2_name)
            normalized_entity1 = normalized_pair[0]
            normalized_entity2 = normalized_pair[1]

            # 使用关系内容的哈希值来去重（允许同一实体对存在多个不同关系）
            content_hash = hash(content.lower())
            # 使用标准化后的实体对作为key，使(A,B)和(B,A)被视为同一个关系
            relation_key = (normalized_entity1, normalized_entity2, content_hash)

            # 如果这个关系还没有见过，添加到结果中
            if relation_key not in seen_relations:
                # 确保输出的关系使用标准化后的实体对
                rel_copy = rel.copy()
                rel_copy['entity1_name'] = normalized_entity1
                rel_copy['entity2_name'] = normalized_entity2
                deduplicated.append(rel_copy)

        return deduplicated

    def _parse_relations_response(self, response: str) -> List[Dict[str, str]]:
        """
        解析关系抽取的LLM响应

        Args:
            response: LLM的响应文本

        Returns:
            解析后的关系列表
        """
        try:
            relations = self._parse_json_response(response)
            # 确保返回的是列表
            if not isinstance(relations, list):
                relations = [relations]

            # 验证并清理关系数据，移除ID字段（如果存在）
            # 支持两种格式：entity1_name/entity2_name（无向关系）和entity1_name/entity2_name（向后兼容）
            valid_relations = []
            for rel in relations:
                if not isinstance(rel, dict):
                    wprint(f"警告：跳过无效的关系格式: {rel}")
                    continue

                # 尝试获取实体名称（支持多种格式：英文、中文）
                entity1 = None
                entity2 = None

                # 支持英文键名
                if 'entity1_name' in rel and 'entity2_name' in rel:
                    # 新格式：无向关系
                    entity1 = str(rel['entity1_name']).strip()
                    entity2 = str(rel['entity2_name']).strip()
                elif 'entity1_name' in rel and 'entity2_name' in rel:
                    # 旧格式：有向关系（向后兼容）
                    entity1 = str(rel['entity1_name']).strip()
                    entity2 = str(rel['entity2_name']).strip()
                # 支持中文键名
                elif '实体1' in rel and '实体2' in rel:
                    entity1 = str(rel['实体1']).strip()
                    entity2 = str(rel['实体2']).strip()
                elif '实体A' in rel and '实体B' in rel:
                    entity1 = str(rel['实体A']).strip()
                    entity2 = str(rel['实体B']).strip()
                elif 'from' in rel and 'to' in rel:
                    entity1 = str(rel['from']).strip()
                    entity2 = str(rel['to']).strip()
                else:
                    wprint(f"警告：跳过无效的关系格式（缺少实体名称）: {rel}")
                    continue

                if not entity1 or not entity2:
                    wprint(f"警告：跳过无效的关系格式（实体名称为空）: {rel}")
                    continue

                # 标准化实体对（按字母顺序排序，使关系无向化）
                normalized_pair = self._normalize_entity_pair(entity1, entity2)

                # 获取content（支持英文和中文键名）
                content = ''
                if 'content' in rel:
                    content = str(rel['content']).strip()
                elif '内容' in rel:
                    content = str(rel['内容']).strip()
                elif '关系内容' in rel:
                    content = str(rel['关系内容']).strip()
                elif '描述' in rel:
                    content = str(rel['描述']).strip()

                # 只保留必需的字段，移除其他字段（如relation_id）
                cleaned_relation = {
                    'entity1_name': normalized_pair[0],  # 使用标准化后的顺序
                    'entity2_name': normalized_pair[1],
                    'content': content
                }
                valid_relations.append(cleaned_relation)
            wprint(f"[TMG] 关系解析: LLM返回 {len(relations)} 条, 有效 {len(valid_relations)} 条")
            dbg(f"关系解析: LLM返回 {len(relations)} 条, 有效 {len(valid_relations)} 条")
            if len(relations) != len(valid_relations):
                dbg(f"  无效关系详情:")
                for _rel in relations:
                    if not isinstance(_rel, dict):
                        dbg(f"    非dict类型: {_rel}")
                        continue
                    _e1 = _rel.get('entity1_name') or _rel.get('实体1') or _rel.get('实体A')
                    _e2 = _rel.get('entity2_name') or _rel.get('实体2') or _rel.get('实体B')
                    if not _e1 or not _e2:
                        dbg(f"    缺少实体名称: {_rel}")
            return valid_relations
        except (json.JSONDecodeError, Exception) as e:
            wprint(f"[TMG] 关系解析失败: JSON解析错误: {e}")
            wprint(f"[TMG]   LLM响应前500字符: {response[:500]}")
            dbg(f"关系解析失败: JSON解析错误: {e}")
            dbg(f"  LLM响应前800字符: {response[:800]}")
            return []

    def judge_need_create_relation(self, entity1_name: str, entity1_content: str,
                                    entity2_name: str, entity2_content: str,
                                    entity1_memory_cache: Optional[str] = None,
                                    entity2_memory_cache: Optional[str] = None) -> bool:
        """
        判断两个实体之间是否真的需要创建关系边

        这个方法在生成关系content之前调用，使用两个实体的完整信息（name、content、memory_cache）
        来判断是否确实存在明确的、有意义的关联。

        Args:
            entity1_name: 起始实体名称
            entity1_content: 起始实体内容描述
            entity2_name: 目标实体名称
            entity2_content: 目标实体内容描述
            entity1_memory_cache: 起始实体的记忆缓存内容（可选）
            entity2_memory_cache: 目标实体的记忆缓存内容（可选）

        Returns:
            True表示确实需要创建关系边，False表示不需要
        """
        system_prompt = JUDGE_NEED_CREATE_RELATION_SYSTEM_PROMPT

        # 构建实体信息
        entities_info = f"""
起始实体：
- 名称: {entity1_name}
- 描述: {entity1_content}
"""
        if entity1_memory_cache:
            entities_info += f"- 相关记忆缓存: {entity1_memory_cache[:500]}...\n"

        entities_info += f"""
目标实体：
- 名称: {entity2_name}
- 描述: {entity2_content}
"""
        if entity2_memory_cache:
            entities_info += f"- 相关记忆缓存: {entity2_memory_cache[:500]}...\n"

        prompt = f"""<实体信息>
{entities_info}
</实体信息>

请判断以下两个实体之间是否需要创建关系边。

输出格式：{{"need_create": true/false}}"""

        response = self._call_llm(prompt, system_prompt)

        # 尝试解析JSON响应
        try:
            result = self._parse_json_response(response)

            if isinstance(result, dict) and "need_create" in result:
                return bool(result["need_create"])
            else:
                # 如果解析失败，默认返回False（保守策略）
                return False

        except (json.JSONDecodeError, Exception) as e:
            wprint(f"解析判断关系JSON失败: {e}")
            wprint(f"响应内容: {response[:500]}...")
            # 如果修复失败，默认返回False（保守策略）
            return False

    def generate_relation_content(self, entity1_name: str, entity1_content: str,
                                  entity2_name: str, entity2_content: str,
                                  relation_memory_cache: Optional[str] = None,
                                  preliminary_content: Optional[str] = None) -> str:
        """
        根据两个实体和memory_cache生成关系的content

        Args:
            entity1_name: 起始实体名称
            entity1_content: 起始实体内容描述
            entity2_name: 目标实体名称
            entity2_content: 目标实体内容描述
            relation_memory_cache: 关系的记忆缓存内容（可选，这是总结两个实体的memory_cache）
            preliminary_content: 初步的关系content（可选，用于参考）

        Returns:
            生成的关系content（自然语言描述）
        """
        system_prompt = GENERATE_RELATION_CONTENT_SYSTEM_PROMPT

        # 构建初步关系描述信息
        preliminary_info = ""
        if preliminary_content:
            preliminary_info = f"""
初步关系描述（作为参考）：
{preliminary_content}
"""

        # 构建实体信息（简化，只提供基本信息）
        entities_info = f"""
起始实体：
- 名称: {entity1_name}
- 描述: {entity1_content[:200]}{'...' if len(entity1_content) > 200 else ''}

目标实体：
- 名称: {entity2_name}
- 描述: {entity2_content[:200]}{'...' if len(entity2_content) > 200 else ''}
"""

        # 构建关系记忆缓存信息（只关注与关系相关的部分）
        relation_context = ""
        if relation_memory_cache:
            relation_context = f"""
关系记忆缓存：
{relation_memory_cache[:500]}{'...' if len(relation_memory_cache) > 500 else ''}
"""

        prompt = f"""<实体信息>
{preliminary_info}
{entities_info}
{relation_context}
</实体信息>

输出格式：{{"content": "关系描述"}}"""

        response = self._call_llm(prompt, system_prompt)

        # 尝试解析JSON响应
        try:
            result = self._parse_json_response(response)

            if isinstance(result, dict) and "content" in result:
                return result["content"]
            else:
                # 如果解析失败，返回默认描述
                return f"'{entity1_name}'与'{entity2_name}'的关联关系"

        except (json.JSONDecodeError, Exception) as e:
            wprint(f"解析关系content JSON失败: {e}")
            # 显示更多调试信息
            error_pos = getattr(e, 'pos', None)
            if error_pos is not None:
                start = max(0, error_pos - 100)
                end = min(len(response), error_pos + 100)
                wprint(f"错误位置: {error_pos}, 附近内容: {response[start:end]}")
            else:
                wprint(f"响应内容前500字符: {response[:500]}...")

            # 尝试直接从响应中提取content字段的值（使用正则表达式）
            try:
                import re
                content_match = re.search(r'"content"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', response)
                if content_match:
                    content_value = content_match.group(1)
                    content_value = content_value.encode().decode('unicode_escape')
                    wprint(f"使用正则表达式提取的content: {content_value[:100]}...")
                    return content_value
            except Exception as regex_error:
                wprint(f"正则表达式提取失败: {regex_error}")

            # 最后回退到默认描述
            wprint(f"所有修复尝试都失败，使用默认描述")
            return f"'{entity1_name}'与'{entity2_name}'的关联关系"
