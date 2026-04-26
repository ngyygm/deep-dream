"""LLM客户端 - 内容判断与合并相关操作。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..utils import clean_markdown_code_blocks, wprint_info
from .prompts import (
    JUDGE_CONTENT_NEED_UPDATE_SYSTEM_PROMPT,
    MERGE_ENTITY_NAME_SYSTEM_PROMPT,
    JUDGE_RELATION_MATCH_SYSTEM_PROMPT,
    MERGE_MULTIPLE_RELATION_CONTENTS_SYSTEM_PROMPT,
    MERGE_MULTIPLE_ENTITY_CONTENTS_SYSTEM_PROMPT,
)

_TRUE_WORDS = frozenset(("true", "yes", "是", "需要更新", "需要"))


def _contents_fast_path(contents: List[str]) -> Optional[str]:
    """Check if all contents are identical or new is a subset. Returns winner or None."""
    if not contents:
        return ""
    if len(contents) == 1:
        return contents[0]
    if len(contents) == 2:
        # Most common case: avoid list allocation, compare directly
        s0 = contents[0].strip()
        s1 = contents[1].strip()
        if s0 == s1:
            return contents[0]
        if s1 and s1 in s0:
            return contents[0]
        return None
    # 3+ contents: generator-based check avoids list allocation
    base = contents[0].strip()
    if all(c.strip() == base for c in contents[1:]):
        return contents[0]
    return None
_FALSE_WORDS = frozenset(("false", "no", "否", "不需要更新", "不需要", "已包含"))


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
        _old_s = old_content.strip()
        _new_s = new_content.strip()
        if _old_s == _new_s:
            return False

        # 快速路径：新内容是旧内容的子串 → 旧内容已包含新内容，无需更新
        if _new_s and _new_s in _old_s:
            return False

        # 快速路径：旧内容是新内容的前缀 → 增量增长，需要更新
        if _old_s and _new_s.startswith(_old_s):
            return True

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

        # 提取 markdown 代码块内的内容（prompt 要求 LLM 输出 ```json true/false ```）
        _cleaned = clean_markdown_code_blocks(response)
        text = (_cleaned if _cleaned else response).strip().lower()
        # 宽松匹配：处理LLM返回的各种格式
        if text in _TRUE_WORDS:
            return True
        elif text in _FALSE_WORDS:
            return False
        else:
            # 如果LLM返回明确的更新指令（包含"更新"等关键词），视为需要更新
            if "更新" in text or "新信息" in text or "差异" in text:
                return True
            # 兜底：模糊响应默认不更新，避免版本膨胀
            return False

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
                merged_name = result['name'].strip()
                if merged_name:
                    return merged_name

            # JSON格式不正确，使用简单合并策略
            return f"{old_name}（{new_name}）"

        except Exception as exc:
            # JSON解析失败，使用简单合并策略
            wprint_info(f"警告：名称合并JSON解析失败，使用简单策略: {exc}")
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

        # Fast-path: check if new content matches or is substring of any existing
        _new_content = (extracted_relation.get('content') or '').strip()
        if _new_content:
            for r in existing_relations:
                _existing = (r.get('content') or '').strip()
                if _existing and (_new_content == _existing or _new_content in _existing):
                    return {"family_id": r['family_id'], "need_update": False}

        system_prompt = JUDGE_RELATION_MATCH_SYSTEM_PROMPT

        # Cap existing relations to avoid blowing up LLM context for hub entities
        _MAX_EXISTING_IN_PROMPT = 15
        _rels_to_include = existing_relations[:_MAX_EXISTING_IN_PROMPT]
        existing_str = "\n\n".join([
            f"family_id: {r['family_id']}\tsource_document: {self._source_doc_label(r.get('source_document', ''))}\tcontent: {r['content']}"
            for r in _rels_to_include
        ])
        if len(existing_relations) > _MAX_EXISTING_IN_PROMPT:
            existing_str += f"\n\n... (共 {len(existing_relations)} 条，已省略 {len(existing_relations) - _MAX_EXISTING_IN_PROMPT} 条)"

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
            if isinstance(result, list) and result and isinstance(result[0], dict):
                return result[0]
            if isinstance(result, dict):
                return result
            return None
        except Exception as e:
            wprint_info(f"[DeepDream] 实体合并内容解析失败: {e}")
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
        """合并两个关系内容（委托给 merge_multiple_relation_contents）。"""
        # Fast-path empty checks before delegating to batch method
        _old_s = old_content.strip() if old_content else ""
        _new_s = new_content.strip() if new_content else ""
        if not _old_s:
            return new_content
        if not _new_s:
            return old_content
        # Inline 2-element fast path to avoid temp list allocation
        if _old_s == _new_s or (_new_s and _new_s in _old_s):
            return old_content
        return self.merge_multiple_relation_contents(
            [old_content, new_content],
            relation_sources=[old_source_document, new_source_document],
            entity_pair=(entity1_name, entity2_name) if entity1_name else None,
        )

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
        _fast = _contents_fast_path(contents)
        if _fast is not None:
            return _fast

        system_prompt = MERGE_MULTIPLE_RELATION_CONTENTS_SYSTEM_PROMPT

        contents_str = "\n\n".join([
            f"关系描述 {i+1}:\n{content}"
            for i, content in enumerate(contents)
        ])
        pair_note = ""
        if entity_pair:
            pair_note = f"\n实体对: {entity_pair[0]} ↔ {entity_pair[1]}\n"

        prompt = f"""{pair_note}<关系描述列表>
{contents_str}
</关系描述列表>

保持第一个描述原文，仅将后续的新增信息补充进去。无新信息则返回第一个描述原文。直接输出合并后的文字，不要 JSON 包装。"""

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
        _fast = _contents_fast_path(contents)
        if _fast is not None:
            return _fast

        system_prompt = MERGE_MULTIPLE_ENTITY_CONTENTS_SYSTEM_PROMPT

        base_content = contents[0]
        new_infos = []
        for i, content in enumerate(contents[1:], 1):
            name_label = entity_names[i] if entity_names and i < len(entity_names) else ""
            prefix = f"[{name_label}] " if name_label else ""
            new_infos.append(f"新信息 {i}: {prefix}{content}")
        new_infos_str = "\n".join(new_infos)

        prompt = f"""<基础版本>
{base_content}
</基础版本>

<待融入的新信息>
{new_infos_str}
</待融入的新信息>

在基础版本上做最小修改来融入新信息。禁止重写。无新信息则返回基础版本原文。直接输出合并后的文字，不要 JSON 包装。"""

        response = self._call_llm(prompt, system_prompt)
        return response.strip()

