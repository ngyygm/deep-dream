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
    MERGE_ENTITY_SECTION_SYSTEM_PROMPT,
    MERGE_RELATION_SECTION_SYSTEM_PROMPT,
)


class _ContentMergerMixin:
    """内容判断与合并相关的 LLM 操作（mixin，通过 LLMClient 多继承使用）。"""

    @staticmethod
    def _source_doc_label(source_document: Optional[str]) -> str:
        src = (source_document or "").strip()
        return src if src else "(未知文档)"

    def judge_content_need_update(
        self,
        old_content: str,
        new_content: str,
        *,
        old_source_document: str = "",
        new_source_document: str = "",
        old_name: str = "",
        new_name: str = "",
        object_type: str = "实体",
    ) -> bool:
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

        prompt = f"""<对象类型>
{object_type}
</对象类型>

<旧版本>
- name: {old_name or '(未提供名称)'}
- source_document: {self._source_doc_label(old_source_document)}
- content:
{old_content}
</旧版本>

<新版本>
- name: {new_name or old_name or '(未提供名称)'}
- source_document: {self._source_doc_label(new_source_document)}
- content:
{new_content}
</新版本>

请判断当前抽取的内容是否已被旧版本包含："""

        response = self._call_llm(prompt, system_prompt)

        # 解析响应
        response = response.strip().lower()
        # 宽松匹配：处理LLM返回的各种格式
        if response in ("true", "yes", "是", "需要更新", "需要"):
            return True
        elif response in ("false", "no", "否", "不需要更新", "不需要", "已包含"):
            return False
        else:
            # 如果LLM返回明确的更新指令（包含"更新"等关键词），视为需要更新
            if "更新" in response or "新信息" in response or "差异" in response:
                return True
            # 兜底：模糊响应默认不更新，避免版本膨胀
            return False

    def merge_entity_content(
        self,
        old_content: str,
        new_content: str,
        *,
        old_source_document: str = "",
        new_source_document: str = "",
        old_name: str = "",
        new_name: str = "",
    ) -> str:
        """
        合并实体内容（更新时使用）

        Args:
            old_content: 旧的内容
            new_content: 新的内容

        Returns:
            合并后的内容
        """
        system_prompt = MERGE_ENTITY_CONTENT_SYSTEM_PROMPT

        prompt = f"""<旧版本实体>
- name: {old_name or '(未提供名称)'}
- source_document: {self._source_doc_label(old_source_document)}
- content:
{old_content}
</旧版本实体>

<新版本实体>
- name: {new_name or old_name or '(未提供名称)'}
- source_document: {self._source_doc_label(new_source_document)}
- content:
{new_content}
</新版本实体>

请重新总结，只输出一个 ```json ... ``` 代码块；代码块内部格式为：{{"content": "重新总结后的完整实体描述"}}"""

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

请将这两个名称合并为一个规范名称，只输出一个 ```json ... ``` 代码块；代码块内部格式为：{{"name": "合并后的规范名称"}}"""

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

        except Exception as exc:
            # JSON解析失败，使用简单合并策略
            wprint(f"警告：名称合并JSON解析失败，使用简单策略: {exc}")
            # 选择较短的作为主名称，较长的作为补充
            if len(old_name) <= len(new_name):
                return f"{old_name}（{new_name}）"
            else:
                return f"{new_name}（{old_name}）"

    def judge_relation_match(self, extracted_relation: Dict[str, str],
                            existing_relations: List[Dict[str, str]],
                            *,
                            new_source_document: str = "") -> Optional[Dict[str, Any]]:
        """
        判断抽取的关系是否与已有关系匹配

        Args:
            extracted_relation: 抽取的关系（包含entity1_name, entity2_name, content）
            existing_relations: 已有关系列表（每个包含family_id, content）

        Returns:
            如果匹配，返回 {"family_id": "...", "need_update": True/False}
            如果不匹配，返回 None
        """
        if not existing_relations:
            return None

        system_prompt = JUDGE_RELATION_MATCH_SYSTEM_PROMPT

        existing_str = "\n\n".join([
            f"family_id: {r['family_id']}\tsource_document: {self._source_doc_label(r.get('source_document', ''))}\tcontent: {r['content']}"
            for r in existing_relations
        ])

        entity1_name = extracted_relation.get('entity1_name') or extracted_relation.get('entity1') or extracted_relation.get('from', '')
        entity2_name = extracted_relation.get('entity2_name') or extracted_relation.get('entity2') or extracted_relation.get('to', '')

        prompt = f"""<新关系>
- entity1: {entity1_name}
- entity2: {entity2_name}
- source_document: {self._source_doc_label(new_source_document)}
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
        except Exception as e:
            wprint(f"[DeepDream] 实体合并内容解析失败: {e}")
            return None

    def merge_relation_content(
        self,
        old_content: str,
        new_content: str,
        *,
        old_source_document: str = "",
        new_source_document: str = "",
        entity1_name: str = "",
        entity2_name: str = "",
    ) -> str:
        """
        合并关系内容（更新时使用）

        Args:
            old_content: 旧的内容
            new_content: 新的内容

        Returns:
            合并后的内容
        """
        system_prompt = MERGE_RELATION_CONTENT_SYSTEM_PROMPT

        prompt = f"""<旧版本关系>
- entity1: {entity1_name or '(未提供实体1)'}
- entity2: {entity2_name or '(未提供实体2)'}
- source_document: {self._source_doc_label(old_source_document)}
- content:
{old_content}
</旧版本关系>

<新版本关系>
- entity1: {entity1_name or '(未提供实体1)'}
- entity2: {entity2_name or '(未提供实体2)'}
- source_document: {self._source_doc_label(new_source_document)}
- content:
{new_content}
</新版本关系>

请重新总结，只输出一个 ```json ... ``` 代码块；代码块内部格式为：{{"content": "重新总结后的完整关系描述"}}"""

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

    def merge_multiple_relation_contents(
        self,
        contents: List[str],
        *,
        relation_sources: Optional[List[str]] = None,
        entity_pair: Optional[tuple[str, str]] = None,
    ) -> str:
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
            (
                f"关系描述 {i+1}:\n"
                f"- source_document: {self._source_doc_label(relation_sources[i] if relation_sources and i < len(relation_sources) else '')}\n"
                f"- content: {content}"
            )
            for i, content in enumerate(contents)
        ])
        pair_note = ""
        if entity_pair:
            pair_note = f"\n<关系实体对>\n- entity1: {entity_pair[0]}\n- entity2: {entity_pair[1]}\n</关系实体对>\n"

        prompt = f"""{pair_note}<关系描述列表>
{contents_str}
</关系描述列表>

请重新总结，确保保留所有独特信息，输出重新总结后的完整内容："""

        return self._call_llm(prompt, system_prompt)

    def merge_multiple_entity_contents(
        self,
        contents: List[str],
        *,
        entity_sources: Optional[List[str]] = None,
        entity_names: Optional[List[str]] = None,
    ) -> str:
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
            (
                f"实体内容 {i+1}:\n"
                f"- name: {entity_names[i] if entity_names and i < len(entity_names) else '(未提供名称)'}\n"
                f"- source_document: {self._source_doc_label(entity_sources[i] if entity_sources and i < len(entity_sources) else '')}\n"
                f"- content: {content}"
            )
            for i, content in enumerate(contents)
        ])

        prompt = f"""<实体内容列表>
{contents_str}
</实体内容列表>

请重新总结，确保保留所有独特信息；只输出一个 ```json ... ``` 代码块；代码块内部格式为：{{"content": "重新总结后的完整实体描述"}}"""

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

    # ------------------------------------------------------------------
    # Section 级合并
    # ------------------------------------------------------------------

    def merge_entity_section(
        self,
        section_key: str,
        old_section: str,
        new_section: str,
        *,
        old_source_document: str = "",
        new_source_document: str = "",
        entity_name: str = "",
    ) -> str:
        """合并实体的单个 section（增量更新）。

        只对变更的 section 调用 LLM，而非整个 content。
        """
        system_prompt = MERGE_ENTITY_SECTION_SYSTEM_PROMPT

        prompt = f"""<实体名称>{entity_name or '(未提供)'}</实体名称>
<Section>{section_key}</Section>

<旧版本>
- source_document: {self._source_doc_label(old_source_document)}
- content:
{old_section}
</旧版本>

<新版本>
- source_document: {self._source_doc_label(new_source_document)}
- content:
{new_section}
</新版本>

请合并该 section 的内容，只输出一个 ```json ... ``` 代码块；代码块内部格式为：{{"content": "合并后的 section 正文"}}"""

        response = self._call_llm(prompt, system_prompt)

        try:
            result = self._parse_json_response(response)
            if isinstance(result, dict) and 'content' in result:
                merged = str(result['content']).strip()
                if merged:
                    return merged
            wprint(f"警告：实体 section 合并返回的 JSON 格式不正确，使用原始响应")
            return response.strip()
        except Exception as e:
            wprint(f"警告：实体 section 合并 JSON 解析失败: {e}")
            return clean_markdown_code_blocks(response).strip()

    def merge_relation_section(
        self,
        section_key: str,
        old_section: str,
        new_section: str,
        *,
        old_source_document: str = "",
        new_source_document: str = "",
        entity1_name: str = "",
        entity2_name: str = "",
    ) -> str:
        """合并关系的单个 section（增量更新）。"""
        system_prompt = MERGE_RELATION_SECTION_SYSTEM_PROMPT

        prompt = f"""<关系实体对>
- entity1: {entity1_name or '(未提供)'}
- entity2: {entity2_name or '(未提供)'}
</关系实体对>
<Section>{section_key}</Section>

<旧版本>
- source_document: {self._source_doc_label(old_source_document)}
- content:
{old_section}
</旧版本>

<新版本>
- source_document: {self._source_doc_label(new_source_document)}
- content:
{new_section}
</新版本>

请合并该 section 的内容，只输出一个 ```json ... ``` 代码块；代码块内部格式为：{{"content": "合并后的 section 正文"}}"""

        response = self._call_llm(prompt, system_prompt)

        try:
            result = self._parse_json_response(response)
            if isinstance(result, dict) and 'content' in result:
                merged = str(result['content']).strip()
                if merged:
                    return merged
            wprint(f"警告：关系 section 合并返回的 JSON 格式不正确，使用原始响应")
            return response.strip()
        except Exception as e:
            wprint(f"警告：关系 section 合并 JSON 解析失败: {e}")
            return clean_markdown_code_blocks(response).strip()
