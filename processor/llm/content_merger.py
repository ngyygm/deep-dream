"""LLM客户端 - 内容判断与合并相关操作。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..utils import clean_markdown_code_blocks, wprint
from .prompts import (
    JUDGE_CONTENT_NEED_UPDATE_SYSTEM_PROMPT,
    MERGE_ENTITY_CONTENT_SYSTEM_PROMPT,
    MERGE_ENTITY_NAME_SYSTEM_PROMPT,
    JUDGE_RELATION_MATCH_SYSTEM_PROMPT,
    MERGE_RELATION_CONTENT_SYSTEM_PROMPT,
    MERGE_MULTIPLE_RELATION_CONTENTS_SYSTEM_PROMPT,
    MERGE_MULTIPLE_ENTITY_CONTENTS_SYSTEM_PROMPT,
)


class _ContentMergerMixin:
    """内容判断与合并相关的 LLM 操作（mixin，通过 LLMClient 多继承使用）。"""

    def judge_content_need_update(self, old_content: str, new_content: str) -> bool:
        """
        判断内容是否需要更新

        比较最新版本的content和当前抽取的content，判断是否需要更新

        判断规则：
        1. 如果新内容的所有信息都已经被旧内容包含，返回False（不需要更新）
        2. 如果新内容包含新信息、修正了旧内容、或与旧内容有实质性差异，返回True（需要更新）

        Args:
            old_content: 数据库中最新版本的content
            new_content: 当前抽取的content

        Returns:
            True表示需要更新，False表示不需要更新
        """
        # 如果内容完全相同，不需要更新
        if old_content.strip() == new_content.strip():
            return False

        # 使用LLM判断新内容是否已经被旧内容包含
        system_prompt = JUDGE_CONTENT_NEED_UPDATE_SYSTEM_PROMPT

        prompt = f"""<旧内容>
{old_content}
</旧内容>

<新内容>
{new_content}
</新内容>

请判断当前抽取的内容是否已被旧版本包含："""

        response = self._call_llm(prompt, system_prompt)

        # 解析响应
        response = response.strip().lower()
        if response == "true":
            return True
        elif response == "false":
            return False
        else:
            # 如果LLM返回的不是true/false，默认返回True（保守策略，倾向于更新）
            return True

    def merge_entity_content(self, old_content: str, new_content: str) -> str:
        """
        合并实体内容（更新时使用）

        Args:
            old_content: 旧的内容
            new_content: 新的内容

        Returns:
            合并后的内容
        """
        system_prompt = MERGE_ENTITY_CONTENT_SYSTEM_PROMPT

        prompt = f"""<旧内容>
{old_content}
</旧内容>

<新内容>
{new_content}
</新内容>

请重新总结，输出格式：{{"content": "重新总结后的完整实体描述"}}"""

        response = self._call_llm(prompt, system_prompt)

        # 尝试解析JSON响应
        try:
            result = self._parse_json_response(response)

            # 提取content字段
            if isinstance(result, dict) and 'content' in result:
                merged_content = str(result['content']).strip()
                # 如果content不为空，返回合并后的内容
                if merged_content:
                    return merged_content

            # 如果JSON格式不正确或content为空，回退到原始响应
            wprint(f"警告：实体内容合并返回的JSON格式不正确或content为空，使用原始响应")
            return response.strip()

        except Exception as e:
            # JSON解析失败
            wprint(f"警告：实体内容合并JSON解析失败，使用原始响应: {e}")
            cleaned_response = clean_markdown_code_blocks(response)
            return cleaned_response.strip()

    def merge_entity_name(self, old_name: str, new_name: str) -> str:
        """
        合并实体名称（更新时使用）

        如果新名称与旧名称不同，生成一个包含别称的名称。
        例如：旧名称"科幻世界"，新名称"科幻世界出版机构"，合并后可能是"科幻世界（出版机构）"

        Args:
            old_name: 旧的名称
            new_name: 新的名称

        Returns:
            合并后的名称
        """
        # 如果名称相同，直接返回
        if old_name == new_name:
            return old_name

        # 如果一个名称包含另一个，使用较长的
        if old_name in new_name:
            return new_name
        if new_name in old_name:
            return old_name

        system_prompt = MERGE_ENTITY_NAME_SYSTEM_PROMPT

        prompt = f"""旧名称：{old_name}
新名称：{new_name}

请将这两个名称合并为一个规范名称，输出格式：{{"name": "合并后的规范名称"}}"""

        response = self._call_llm(prompt, system_prompt)

        # 尝试解析JSON响应
        try:
            result = self._parse_json_response(response)

            if isinstance(result, dict) and 'name' in result:
                merged_name = str(result['name']).strip()
                if merged_name:
                    return merged_name

            # JSON格式不正确，使用简单合并策略
            return f"{old_name}（{new_name}）"

        except Exception:
            # JSON解析失败，使用简单合并策略
            # 选择较短的作为主名称，较长的作为补充
            if len(old_name) <= len(new_name):
                return f"{old_name}（{new_name}）"
            else:
                return f"{new_name}（{old_name}）"

    def judge_relation_match(self, extracted_relation: Dict[str, str],
                            existing_relations: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """
        判断抽取的关系是否与已有关系匹配

        Args:
            extracted_relation: 抽取的关系（包含entity1_name, entity2_name, content）
            existing_relations: 已有关系列表（每个包含relation_id, content）

        Returns:
            如果匹配，返回 {"relation_id": "...", "need_update": True/False}
            如果不匹配，返回 None
        """
        if not existing_relations:
            return None

        system_prompt = JUDGE_RELATION_MATCH_SYSTEM_PROMPT

        existing_str = "\n\n".join([
            f"relation_id: {r['relation_id']}\tcontent: {r['content']}"
            for r in existing_relations
        ])

        entity1_name = extracted_relation.get('entity1_name') or extracted_relation.get('entity1_name', '')
        entity2_name = extracted_relation.get('entity2_name') or extracted_relation.get('entity2_name', '')

        prompt = f"""<新关系>
- entity1: {entity1_name}
- entity2: {entity2_name}
- content: {extracted_relation['content']}
</新关系>

<已有关系列表>
{existing_str}
</已有关系列表>

请判断新关系是否与已有关系相同或非常相似。"""

        response = self._call_llm(prompt, system_prompt)

        try:
            result = self._parse_json_response(response)
            if result is None or result == "null":
                return None
            # LLM 有时返回 list，统一转为单个 dict
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                return result[0]
            if isinstance(result, dict):
                return result
            return None
        except Exception:
            return None

    def merge_relation_content(self, old_content: str, new_content: str) -> str:
        """
        合并关系内容（更新时使用）

        Args:
            old_content: 旧的内容
            new_content: 新的内容

        Returns:
            合并后的内容
        """
        system_prompt = MERGE_RELATION_CONTENT_SYSTEM_PROMPT

        prompt = f"""<旧内容>
{old_content}
</旧内容>

<新内容>
{new_content}
</新内容>

请重新总结，输出格式：{{"content": "重新总结后的完整关系描述"}}"""

        response = self._call_llm(prompt, system_prompt)

        # 尝试解析JSON响应
        try:
            result = self._parse_json_response(response)

            # 提取content字段
            if isinstance(result, dict) and 'content' in result:
                merged_content = str(result['content']).strip()
                # 如果content不为空，返回合并后的内容
                if merged_content:
                    return merged_content

            # 如果JSON格式不正确或content为空，回退到原始响应
            wprint(f"警告：关系内容合并返回的JSON格式不正确或content为空，使用原始响应")
            return response.strip()

        except Exception as e:
            # JSON解析失败
            wprint(f"警告：关系内容合并JSON解析失败，使用原始响应: {e}")
            cleaned_response = clean_markdown_code_blocks(response)
            return cleaned_response.strip()

    def merge_multiple_relation_contents(self, contents: List[str]) -> str:
        """
        合并多个关系内容（用于去重合并）

        Args:
            contents: 多个关系内容列表

        Returns:
            合并后的内容
        """
        if not contents:
            return ""

        if len(contents) == 1:
            return contents[0]

        system_prompt = MERGE_MULTIPLE_RELATION_CONTENTS_SYSTEM_PROMPT

        contents_str = "\n\n".join([
            f"关系描述 {i+1}:\n{content}"
            for i, content in enumerate(contents)
        ])

        prompt = f"""<关系描述列表>
{contents_str}
</关系描述列表>

请重新总结，确保保留所有独特信息，输出重新总结后的完整内容："""

        return self._call_llm(prompt, system_prompt)

    def merge_multiple_entity_contents(self, contents: List[str]) -> str:
        """
        合并多个实体内容（用于多实体合并）

        Args:
            contents: 多个实体内容列表

        Returns:
            合并后的内容
        """
        if not contents:
            return ""

        if len(contents) == 1:
            return contents[0]

        system_prompt = MERGE_MULTIPLE_ENTITY_CONTENTS_SYSTEM_PROMPT

        contents_str = "\n\n".join([
            f"实体内容 {i+1}:\n{content}"
            for i, content in enumerate(contents)
        ])

        prompt = f"""<实体内容列表>
{contents_str}
</实体内容列表>

请重新总结，确保保留所有独特信息，输出格式：{{"content": "重新总结后的完整实体描述"}}"""

        response = self._call_llm(prompt, system_prompt)

        # 尝试解析JSON响应
        try:
            result = self._parse_json_response(response)

            # 提取content字段
            if isinstance(result, dict) and 'content' in result:
                merged_content = str(result['content']).strip()
                # 如果content不为空，返回合并后的内容
                if merged_content:
                    return merged_content

            # 如果JSON格式不正确或content为空，回退到原始响应
            wprint(f"警告：多实体内容合并返回的JSON格式不正确或content为空，使用原始响应")
            return response.strip()

        except Exception as e:
            # JSON解析失败
            wprint(f"警告：多实体内容合并JSON解析失败，使用原始响应: {e}")
            cleaned_response = clean_markdown_code_blocks(response)
            return cleaned_response.strip()
