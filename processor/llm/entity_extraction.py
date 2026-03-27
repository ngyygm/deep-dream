"""LLM客户端 - 实体抽取相关操作。"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..models import MemoryCache
from ..utils import clean_markdown_code_blocks, wprint
from .prompts import (
    EXTRACT_ENTITIES_AND_RELATIONS_SYSTEM_PROMPT,
    EXTRACT_ENTITIES_SINGLE_PASS_SYSTEM_PROMPT,
    EXTRACT_ENTITIES_BY_NAMES_SYSTEM_PROMPT,
    ENHANCE_ENTITY_CONTENT_SYSTEM_PROMPT,
    DETAILED_JUDGMENT_PROCESS,
)


class _EntityExtractionMixin:
    """实体抽取相关的 LLM 操作（mixin，通过 LLMClient 多继承使用）。"""

    def extract_entities_and_relations(self, memory_cache: MemoryCache, input_text: str,
                                        rounds: int = 1, verbose: bool = False) -> tuple:
        """
        合并抽取概念实体和实体间关系，支持多轮次补充抽取。

        多轮次利用 LLM 对话历史：第 2+ 轮把上一轮的 LLM 响应作为 assistant 消息，
        再追加一条 user 消息要求继续补充，天然携带已生成内容的完整上下文。

        Args:
            memory_cache: 记忆缓存
            input_text: 输入文本
            rounds: 抽取轮次（默认 1）；>1 时每轮要求 LLM 继续补充
            verbose: 是否输出详细信息

        Returns:
            (entities, relations) — entities: List[Dict{name, content}],
                                    relations: List[Dict{entity1_name, entity2_name, content}]
        """
        system_prompt = EXTRACT_ENTITIES_AND_RELATIONS_SYSTEM_PROMPT

        first_prompt = f"""<记忆缓存>
{memory_cache.content}
</记忆缓存>

<输入文本>
{input_text}
</输入文本>

请从文本中同时抽取所有概念实体和实体间关系（越多越好）："""

        all_entities: List[Dict[str, str]] = []
        all_relations: List[Dict[str, str]] = []
        seen_names: set = set()
        seen_rel_keys: set = set()

        # 构建对话历史
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_prompt},
        ]

        for round_idx in range(max(1, rounds)):
            if verbose:
                wprint(f"      合并抽取第 {round_idx + 1}/{rounds} 轮...")

            response = self._call_llm("", messages=messages)
            new_entities, new_relations = self._parse_merged_extraction_response(response)

            # 把本轮的 user + assistant 追加到对话历史
            if round_idx == 0:
                # 第一轮：user 消息已在 messages 中，追加 assistant 回复
                messages.append({"role": "assistant", "content": response})
            else:
                # 后续轮次：user 追问 + assistant 回复 都追加
                pass  # user 消息在下面追加

            # 按名称去重合并实体
            new_entity_count = 0
            for e in new_entities:
                if e['name'] not in seen_names:
                    all_entities.append(e)
                    seen_names.add(e['name'])
                    new_entity_count += 1

            # 按 (entity1, entity2) 去重合并关系
            new_rel_count = 0
            for r in new_relations:
                key = (r['entity1_name'], r['entity2_name'])
                if key not in seen_rel_keys:
                    all_relations.append(r)
                    seen_rel_keys.add(key)
                    new_rel_count += 1

            if verbose:
                wprint(f"        第 {round_idx + 1} 轮完成：新增 {new_entity_count} 实体、{new_rel_count} 关系，"
                      f"累计 {len(all_entities)} 实体、{len(all_relations)} 关系")

            # 本轮无新增，提前退出
            if new_entity_count == 0 and new_rel_count == 0:
                if verbose:
                    wprint(f"        本轮无新增内容，停止抽取")
                break

            # 还有下一轮，追加追问消息
            if round_idx + 1 < rounds:
                messages.append({"role": "user", "content": "请继续从文本中补充更多实体和关系，不要重复已提取的内容。"})

        return all_entities, all_relations

    def _parse_merged_extraction_response(self, response: str) -> tuple:
        """
        解析合并抽取的 LLM 响应。

        Args:
            response: LLM 响应文本

        Returns:
            (entities, relations) — 解析后的实体和关系列表
        """
        try:
            result = self._parse_json_response(response)

            if not isinstance(result, dict):
                result = {"entities": [], "relations": []}

            # 解析实体
            raw_entities = result.get("entities", [])
            if not isinstance(raw_entities, list):
                raw_entities = [raw_entities]
            entities = []
            for e in raw_entities:
                if isinstance(e, dict) and 'name' in e and 'content' in e:
                    entities.append({
                        'name': str(e['name']).strip(),
                        'content': str(e['content']).strip()
                    })

            # 解析关系
            raw_relations = result.get("relations", [])
            if not isinstance(raw_relations, list):
                raw_relations = [raw_relations]
            relations = []
            for rel in raw_relations:
                if not isinstance(rel, dict):
                    continue
                entity1 = (rel.get('entity1_name') or '').strip()
                entity2 = (rel.get('entity2_name') or '').strip()
                content = (rel.get('content') or '').strip()
                if entity1 and entity2 and content:
                    # 标准化实体对（无向）
                    if entity1 > entity2:
                        entity1, entity2 = entity2, entity1
                    relations.append({
                        'entity1_name': entity1,
                        'entity2_name': entity2,
                        'content': content,
                    })

            # 过滤关系中引用了不存在实体的关系
            entity_names = {e['name'] for e in entities}
            valid_relations = []
            for rel in relations:
                if rel['entity1_name'] in entity_names and rel['entity2_name'] in entity_names:
                    valid_relations.append(rel)
                else:
                    # 尝试模糊匹配（LLM 可能简化了实体名称）
                    e1_matched = self._fuzzy_match_entity_name(rel['entity1_name'], entity_names)
                    e2_matched = self._fuzzy_match_entity_name(rel['entity2_name'], entity_names)
                    if e1_matched and e2_matched:
                        valid_relations.append({
                            'entity1_name': e1_matched,
                            'entity2_name': e2_matched,
                            'content': rel['content'],
                        })

            return entities, valid_relations

        except (json.JSONDecodeError, Exception) as e:
            wprint(f"解析合并抽取JSON失败: {e}")
            wprint(f"响应内容: {response[:500]}...")
            return [], []

    @staticmethod
    def _fuzzy_match_entity_name(name: str, valid_names: set) -> Optional[str]:
        """尝试模糊匹配实体名称到有效名称集合。"""
        if name in valid_names:
            return name
        # 去除括号内容后匹配
        base_name = name.split('（')[0].split('(')[0].strip()
        for vn in valid_names:
            vn_base = vn.split('（')[0].split('(')[0].strip()
            if base_name == vn_base:
                return vn
        return None

    def extract_entities(self, memory_cache: MemoryCache, input_text: str,
                         rounds: int = 1, verbose: bool = False,
                         on_round_done=None) -> List[Dict[str, str]]:
        """
        抽取实体，支持多轮次补充抽取（利用 LLM 对话历史）。

        Args:
            memory_cache: 当前的记忆缓存
            input_text: 当前窗口的输入文本
            rounds: 抽取轮次（默认 1）；>1 时利用对话历史要求 LLM 继续补充
            verbose: 是否输出详细信息
            on_round_done: 每轮完成后的回调 fn(round_idx, total_rounds, cumulative_count)

        Returns:
            抽取的实体列表，每个实体包含 name 和 content
        """
        system_prompt = EXTRACT_ENTITIES_SINGLE_PASS_SYSTEM_PROMPT

        first_prompt = f"""<记忆缓存>
{memory_cache.content}
</记忆缓存>

<输入文本>
{input_text}
</输入文本>

请从文本中抽取所有概念实体（越多越好）："""

        all_entities: List[Dict[str, str]] = []
        seen_names: set = set()

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_prompt},
        ]

        for round_idx in range(max(1, rounds)):
            if verbose:
                wprint(f"      实体抽取第 {round_idx + 1}/{rounds} 轮...")

            response = self._call_llm("", messages=messages)

            # 解析
            new_entities = self._parse_entities_response(response)

            # 追加 assistant 回复到对话历史
            messages.append({"role": "assistant", "content": response})

            # 去重合并
            new_count = 0
            for e in new_entities:
                if e['name'] not in seen_names:
                    all_entities.append(e)
                    seen_names.add(e['name'])
                    new_count += 1

            if verbose:
                wprint(f"        第 {round_idx + 1} 轮完成：新增 {new_count} 实体，累计 {len(all_entities)} 实体")

            if on_round_done:
                on_round_done(round_idx + 1, rounds, len(all_entities))

            if new_count == 0:
                if verbose:
                    wprint(f"        本轮无新增实体，停止抽取")
                break

            if round_idx + 1 < rounds:
                messages.append({"role": "user", "content": "请继续从文本中补充更多概念实体，不要重复已提取的内容。"})

        # 多轮次：保存最后一轮的完整 messages（含对话历史）
        if self._distill_data_dir and self._current_distill_step and messages:
            self._save_distill_conversation(messages)

        return all_entities

    def _parse_entities_response(self, response: str) -> List[Dict[str, str]]:
        """解析实体抽取的 LLM 响应。"""
        try:
            entities = self._parse_json_response(response)
            if not isinstance(entities, list):
                entities = [entities]
            cleaned = []
            for entity in entities:
                if isinstance(entity, dict) and 'name' in entity and 'content' in entity:
                    cleaned.append({
                        'name': str(entity['name']).strip(),
                        'content': str(entity['content']).strip(),
                    })
            return cleaned
        except Exception as e:
            wprint(f"解析实体JSON失败: {e}")
            wprint(f"响应内容: {response[:500]}...")
            return []

    def _extract_entities_single_pass(self, memory_cache: MemoryCache, input_text: str,
                                 existing_entities: Optional[List[Dict[str, str]]] = None,
                                 verbose: bool = False) -> List[Dict[str, str]]:
        """
        单次实体抽取

        Args:
            memory_cache: 记忆缓存
            input_text: 输入文本
            existing_entities: 已抽取的实体列表（用于查漏补缺，可选）
            verbose: 是否输出详细信息
        """
        system_prompt = EXTRACT_ENTITIES_SINGLE_PASS_SYSTEM_PROMPT

        # 简化已有实体提示
        existing_entities_str = ""
        if existing_entities:
            existing_names = [e.get('name', '') for e in existing_entities]
            # 先构建字符串，避免在f-string中使用反斜杠
            names_list = '\n'.join(existing_names)
            existing_entities_str = f"""
<已抽取实体>
{names_list}
</已抽取实体>

请基于已有实体拓展查找：
- 查找相关、相对、同类概念
- 从上下文推断隐含实体
- 只抽取新发现的实体"""

        prompt = f"""<记忆缓存>
{memory_cache.content}
</记忆缓存>

<输入文本>
{input_text}
</输入文本>
{existing_entities_str}

请从文本中抽取所有概念实体（越多越好）："""


        if verbose:
            if existing_entities:
                wprint(f"          正在调用LLM查漏补缺（已抽取 {len(existing_entities)} 个实体）...")
            else:
                wprint(f"          正在调用LLM进行实体抽取...")

        response = self._call_llm(prompt, system_prompt)

        if verbose:
            wprint(f"          LLM调用完成，正在解析结果...")

        # 解析JSON响应
        try:
            entities = self._parse_json_response(response)
            if not isinstance(entities, list):
                entities = [entities]

            # 验证并清理实体数据，移除ID字段（如果存在）
            cleaned_entities = []
            for entity in entities:
                if isinstance(entity, dict) and 'name' in entity and 'content' in entity:
                    # 只保留name和content字段，移除其他字段（如entity_id）
                    cleaned_entity = {
                        'name': str(entity['name']).strip(),
                        'content': str(entity['content']).strip()
                    }
                    cleaned_entities.append(cleaned_entity)

            if verbose:
                wprint(f"          解析完成，获得 {len(cleaned_entities)} 个实体")

            return cleaned_entities
        except (json.JSONDecodeError, Exception) as e:
            wprint(f"解析实体JSON失败: {e}")
            wprint(f"响应内容: {response[:500]}...")  # 只显示前500个字符
            return []

    def extract_entities_by_names(self, memory_cache: MemoryCache, input_text: str,
                                  entity_names: List[str],
                                  verbose: bool = False) -> List[Dict[str, str]]:
        """
        抽取指定名称的实体

        Args:
            memory_cache: 记忆缓存
            input_text: 输入文本
            entity_names: 要抽取的实体名称列表
            verbose: 是否输出详细信息

        Returns:
            抽取的实体列表，每个实体包含 name 和 content
        """
        if not entity_names:
            return []

        system_prompt = EXTRACT_ENTITIES_BY_NAMES_SYSTEM_PROMPT

        entity_names_str = "\n".join([f"- {name}" for name in entity_names])

        prompt = f"""<记忆缓存>
{memory_cache.content}
</记忆缓存>

<输入文本>
{input_text}
</输入文本>

<指定实体名称>
{entity_names_str}
</指定实体名称>

请从输入文本中抽取上述指定的实体："""

        if verbose:
            wprint(f"          正在调用LLM抽取指定实体（共 {len(entity_names)} 个）...")

        response = self._call_llm(prompt, system_prompt)

        if verbose:
            wprint(f"          LLM调用完成，正在解析结果...")

        # 解析JSON响应
        try:
            entities = self._parse_json_response(response)
            if not isinstance(entities, list):
                entities = [entities]

            cleaned_entities = []
            for entity in entities:
                if isinstance(entity, dict) and 'name' in entity and 'content' in entity:
                    cleaned_entity = {
                        'name': str(entity['name']).strip(),
                        'content': str(entity['content']).strip()
                    }
                    # 只保留在指定名称列表中的实体
                    if cleaned_entity['name'] in entity_names:
                        cleaned_entities.append(cleaned_entity)

            if verbose:
                wprint(f"          解析完成，获得 {len(cleaned_entities)} 个实体")

            return cleaned_entities
        except (json.JSONDecodeError, Exception) as e:
            if verbose:
                wprint(f"          解析实体JSON失败: {e}")
            return []

    def enhance_entity_content(self, memory_cache: MemoryCache, input_text: str,
                              entity: Dict[str, str]) -> str:
        """
        实体后验增强：结合缓存记忆和当前text对实体的content进行更细致的补全挖掘

        Args:
            memory_cache: 当前的记忆缓存
            input_text: 当前窗口的输入文本
            entity: 实体字典，包含name和content

        Returns:
            增强后的实体content
        """
        system_prompt = ENHANCE_ENTITY_CONTENT_SYSTEM_PROMPT

        prompt = f"""<记忆缓存>
{memory_cache.content}
</记忆缓存>

<输入文本>
{input_text}
</输入文本>

<已抽取实体>
- 名称：{entity['name']}
- 当前content：{entity['content']}
</已抽取实体>

请对该实体的 content 进行增强，输出格式：{{"content": "增强后的完整实体content"}}"""

        response = self._call_llm(prompt, system_prompt)

        # 尝试解析JSON响应
        try:
            result = self._parse_json_response(response)

            # 提取content字段
            if isinstance(result, dict) and 'content' in result:
                enhanced_content = str(result['content']).strip()
                # 如果content不为空，返回增强后的内容
                if enhanced_content:
                    return enhanced_content

            # 如果JSON格式不正确或content为空，回退到原始响应
            wprint(f"警告：实体后验增强返回的JSON格式不正确或content为空，使用原始响应")
            return response.strip()

        except (json.JSONDecodeError, Exception) as e:
            # JSON解析失败
            wprint(f"警告：实体后验增强JSON解析失败，使用原始响应: {e}")
            cleaned_response = clean_markdown_code_blocks(response)
            return cleaned_response.strip()
