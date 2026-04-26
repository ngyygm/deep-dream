"""LLM客户端 - 知识图谱整理相关操作。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..utils import wprint_info


def _truncate(text: str, limit: int) -> str:
    """Truncate text to limit chars, appending '...' if truncated."""
    return text[:limit] + ("..." if len(text) > limit else "")


def _content_snippet(entity: Dict[str, Any], limit: int = 200) -> str:
    """Extract a short content snippet from an entity dict."""
    return (entity.get("content") or "")[:limit]


from .prompts import (
    ANALYZE_ENTITY_CANDIDATES_PRELIMINARY_SYSTEM_PROMPT,
    RESOLVE_ENTITY_CANDIDATES_BATCH_SYSTEM_PROMPT,
    ENTITY_PAIR_JUDGMENT_RULES,
    analyze_entity_pair_detailed_system_prompt,
    RESOLVE_RELATION_PAIR_BATCH_SYSTEM_PROMPT,
)


class _ConsolidationMixin:
    """知识图谱整理相关的 LLM 操作（mixin，通过 LLMClient 多继承使用）。"""

    def analyze_entity_candidates_preliminary(self, entities_group: List[Dict[str, Any]],
                                              context_text: Optional[str] = None) -> Dict[str, Any]:
        """
        初步筛选：分析一组候选实体，返回可能需要合并或存在关系的候选列表

        这是两步判断流程的第一步，使用完整content进行快速筛选。

        Args:
            entities_group: 候选实体组，每个实体包含:
                - family_id: 实体ID
                - name: 实体名称
                - content: 实体内容描述
                - version_count: 该实体的版本数量
            context_text: 可选的上下文文本（当前处理的文本片段或记忆缓存内容），
                          用于帮助理解实体出现的场景

        Returns:
            初步筛选结果，包含:
            - possible_merges: 可能需要合并的实体对列表
            - possible_relations: 可能存在关系的实体对列表
            - no_action: 不需要处理的实体ID列表
        """
        if not entities_group or len(entities_group) < 2:
            return {"possible_merges": [], "possible_relations": [], "no_action": []}

        system_prompt = ANALYZE_ENTITY_CANDIDATES_PRELIMINARY_SYSTEM_PROMPT

        # 构建实体信息字符串
        current_entity = entities_group[0]
        _ce_fid = current_entity.get('family_id', '')
        _ce_name = current_entity.get('name', '')
        _ce_snip = _content_snippet(current_entity)
        _parts = [f"""
【当前实体】
- family_id: {_ce_fid}
- name: {_ce_name}
- content: {_ce_snip}
"""]

        for i, entity in enumerate(entities_group[1:], 2):
            _fid = entity.get('family_id', '')
            _nm = entity.get('name', '')
            _snip = _content_snippet(entity)
            _parts.append(f"""
【候选{i}】
- family_id: {_fid}
- name: {_nm}
- content: {_snip}
""")
        entities_str = "".join(_parts)

        # 构建上下文信息
        context_note = ""
        if context_text:
            context_snippet = _truncate(context_text, 300)
            context_note = f"""
<原文片段>
{context_snippet}
</原文片段>
"""

        prompt = f"""判断以下候选实体中，哪些可能与当前实体是同一个概念（名称相似、别名、或描述高度重合）：
{context_note}
<候选实体列表>
{entities_str}
</候选实体列表>

只输出一个 ```json ... ``` 代码块，不要包含任何其他文字。"""

        # 调用LLM
        try:
            response = self._call_llm(prompt, system_prompt)

            # 解析JSON响应
            result = self._parse_json_response(response)

            if not isinstance(result, dict):
                raise ValueError("响应格式不正确")

            result.setdefault("candidates", [])
            return result

        except Exception as e:
            wprint_info(f"  初步筛选出错: {e}")
            return {
                "candidates": [],
                "error": str(e)
            }

    def analyze_entity_pair_detailed(self,
                                     current_entity: Dict[str, Any],
                                     candidate_entity: Dict[str, Any],
                                     existing_relations: List[Dict[str, Any]] = None,
                                     context_text: Optional[str] = None) -> Dict[str, Any]:
        """
        精细化判断：对一对实体进行详细分析，判断是否合并或创建关系

        这是两步判断流程的第二步，使用完整的content和已有关系进行精确判断。

        Args:
            current_entity: 当前实体，包含:
                - family_id: 实体ID
                - name: 实体名称
                - content: 完整的实体内容描述
                - version_count: 版本数量
            candidate_entity: 候选实体，格式同上
            existing_relations: 两个实体之间已存在的关系列表，每个关系包含:
                - family_id: 关系ID
                - content: 关系描述
            context_text: 可选的上下文文本（当前处理的文本片段或记忆缓存内容），
                          用于帮助理解实体出现的场景和关系

        Returns:
            判断结果，包含:
            - action: "merge" | "create_relation" | "no_action"
            - reason: 判断理由
            - relation_content: 如果action是create_relation，提供关系描述
            - merge_target: 如果action是merge，提供目标family_id
        """
        # 构建已有关系的提示
        existing_relations_note = ""
        if existing_relations:
            _rel_lines = [f"- {rel.get('content', '无描述')}" for rel in existing_relations]
            existing_relations_note = "\n已有关系（表明是不同实体，除非有明确证据否则不合并）：\n" + "\n".join(_rel_lines) + "\n"

        system_prompt = analyze_entity_pair_detailed_system_prompt(
            existing_relations_note
        )

        # 构建上下文信息
        context_note = ""
        if context_text:
            context_snippet = _truncate(context_text, 500)
            context_note = f"""
<原文片段>
{context_snippet}
</原文片段>
"""

        prompt = f"""<当前实体>
- name: {current_entity.get('name', '')}
- content: {current_entity.get('content', '')}
</当前实体>

<候选实体>
- name: {candidate_entity.get('name', '')}
- content: {candidate_entity.get('content', '')}
</候选实体>
{context_note}

只输出一个 ```json ... ``` 代码块，不要其他文字："""

        try:
            response = self._call_llm(prompt, system_prompt)

            # 解析JSON响应
            result = self._parse_json_response(response)

            if not isinstance(result, dict):
                raise ValueError("响应格式不正确")

            # 确保必需的字段存在
            if "action" not in result:
                result["action"] = "no_action"
            result.setdefault("relation_content", "")

            return result

        except Exception as e:
            wprint_info(f"  精细化判断出错: {e}")
            return {
                "action": "no_action",
                "relation_content": "",
                "error": str(e)
            }

    def resolve_entity_candidates_batch(self,
                                        current_entity: Dict[str, Any],
                                        candidates: List[Dict[str, Any]],
                                        context_text: Optional[str] = None) -> Dict[str, Any]:
        """一次性判断当前实体与多个候选的关系，减少逐候选 detailed 调用。"""
        if not candidates:
            return {
                "match_existing_id": "",
                "update_mode": "create_new",
                "merged_name": current_entity.get("name", ""),
                "relations_to_create": [],
                "confidence": 1.0,
            }

        system_prompt = RESOLVE_ENTITY_CANDIDATES_BATCH_SYSTEM_PROMPT

        context_note = ""
        if context_text:
            context_snippet = _truncate(context_text, 500)
            context_note = f"""
<原文上下文>
{context_snippet}
</原文上下文>"""

        candidates_str = []
        for idx, candidate in enumerate(candidates, 1):
            _cand_mt = candidate.get('name_match_type', 'none')
            match_type_note = ""
            if _cand_mt == "substring":
                match_type_note = "\n- name_match_type: substring（名称子串关系，可能是简称/别名）"
            elif _cand_mt == "exact":
                match_type_note = "\n- name_match_type: exact（核心名称完全相同）"
            elif _cand_mt == "within_batch_alias":
                match_type_note = "\n- name_match_type: within_batch_alias（同批次别名，极强合并信号）"

            _cand_fid = candidate.get('family_id', '')
            _cand_name = candidate.get('name', '')
            _cand_snip = _content_snippet(candidate)
            candidates_str.append(
                f"""候选{idx}:
- family_id: {_cand_fid}
- name: {_cand_name}{match_type_note}
- content: {_cand_snip}"""
            )

        _cur_name = current_entity.get('name', '')
        cur_content = _content_snippet(current_entity)
        prompt = f"""<当前实体>
- name: {_cur_name}
- content: {cur_content}
</当前实体>
{context_note}
<候选实体列表>
{chr(10).join(candidates_str)}
</候选实体列表>

请通过角色指纹对比判断对齐：当前实体与哪个候选在文本中扮演相同角色？

输出 ```json``` 代码块：
{{"match_existing_id": "", "update_mode": "reuse_existing|merge_into_latest|create_new", "merged_name": "", "relations_to_create": [{{"family_id": "", "relation_content": ""}}], "confidence": 0.0}}"""

        try:
            result = self._parse_json_response(self._call_llm(prompt, system_prompt))
            if not isinstance(result, dict):
                raise ValueError("响应格式不正确")
            result.setdefault("match_existing_id", "")
            result.setdefault("update_mode", "create_new")
            result.setdefault("merged_name", "")
            result.setdefault("relations_to_create", [])
            result.setdefault("confidence", 0.0)
            return result
        except Exception as e:
            return {
                "match_existing_id": "",
                "update_mode": "fallback",
                "merged_name": "",
                "relations_to_create": [],
                "confidence": 0.0,
                "error": str(e),
            }

    def resolve_relation_pair_batch(self,
                                    entity1_name: str,
                                    entity2_name: str,
                                    new_relation_contents: List[str],
                                    existing_relations: List[Dict[str, Any]],
                                    new_source_document: str = "") -> Dict[str, Any]:
        """对同一实体对的一批候选关系做一次性 match/update/create 判定。"""
        if not new_relation_contents:
            return {"action": "skip", "confidence": 1.0}

        if not existing_relations:
            merged_content = self.merge_multiple_relation_contents(
                new_relation_contents,
                relation_sources=[new_source_document] * len(new_relation_contents),
                entity_pair=(entity1_name, entity2_name),
            )
            return {
                "action": "create_new",
                "matched_family_id": "",
                "merged_content": merged_content,
                "confidence": 1.0,
            }

        system_prompt = RESOLVE_RELATION_PAIR_BATCH_SYSTEM_PROMPT

        new_relations_text = "\n".join(
            f"- 新关系{i+1} [source_document={new_source_document or '(当前文档)'}]: {content}"
            for i, content in enumerate(new_relation_contents)
        )
        existing_text = "\n".join(
            f"- family_id={rel.get('family_id', '')} [source_document={rel.get('source_document', '') or '(未知文档)'}]: {rel.get('content', '')}"
            for rel in existing_relations
        )
        prompt = f"""<实体对>
- entity1: {entity1_name}
- entity2: {entity2_name}
</实体对>

<新关系描述>
{new_relations_text}
</新关系描述>

<已有关系>
{existing_text}
</已有关系>

判断新关系是否与某个已有关系描述同一性质的关系。参考 source_document，跨文档时只有明确同一语义关系才可匹配。

输出 ```json``` 代码块：
{{"action": "match_existing|create_new", "matched_relation_id": "", "need_update": false, "confidence": 0.0}}"""

        try:
            result = self._parse_json_response(self._call_llm(prompt, system_prompt))
            if not isinstance(result, dict):
                raise ValueError("响应格式不正确")
            result.setdefault("action", "create_new")
            result.setdefault("matched_relation_id", result.pop("matched_family_id", ""))
            result.setdefault("need_update", result.get("action") == "create_new")
            result.setdefault("confidence", 0.0)
            return result
        except Exception as e:
            return {
                "action": "fallback",
                "matched_relation_id": "",
                "need_update": False,
                "confidence": 0.0,
                "error": str(e),
            }

