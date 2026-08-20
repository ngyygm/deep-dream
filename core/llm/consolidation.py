"""LLM客户端 - 知识图谱整理相关操作。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..utils import wprint_info
from .prompts import (
    RESOLVE_ENTITY_CANDIDATES_BATCH_SYSTEM_PROMPT,
    analyze_entity_pair_detailed_system_prompt,
    RESOLVE_RELATION_PAIR_BATCH_SYSTEM_PROMPT,
)


def _truncate(text: str, limit: int) -> str:
    """Truncate text to limit chars, appending '...' if truncated."""
    return text[:limit] + ("..." if len(text) > limit else "")


def _content_snippet(entity: Dict[str, Any], limit: int = 200) -> str:
    """Extract a short content snippet from an entity dict."""
    return (entity.get("content") or "")[:limit]


class _ConsolidationMixin:
    """知识图谱整理相关的 LLM 操作（mixin，通过 LLMClient 多继承使用）。"""

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
        return self._resolve_entity_candidates_llm(
            current_entity, candidates, context_text=context_text)

    def _resolve_entity_candidates_llm(self,
                                       current_entity: Dict[str, Any],
                                       candidates: List[Dict[str, Any]],
                                       context_text: Optional[str] = None) -> Dict[str, Any]:
        """resolve_entity_candidates_batch 的原始直调实现。"""

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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            result, _ = self.call_llm_until_json_parses(
                messages, parse_fn=self._parse_json_response, json_parse_retries=1,
            )
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
        return self._resolve_relation_pair_llm(
            entity1_name, entity2_name, new_relation_contents,
            existing_relations, new_source_document=new_source_document)

    def _resolve_relation_pair_llm(self,
                                   entity1_name: str,
                                   entity2_name: str,
                                   new_relation_contents: List[str],
                                   existing_relations: List[Dict[str, Any]],
                                   new_source_document: str = "") -> Dict[str, Any]:
        """resolve_relation_pair_batch 的原始直调实现。"""

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

    # ------------------------------------------------------------------
    # 窗口级批量裁决（strong-v1）：一次调用裁决窗口内多个实体/多个关系对
    # ------------------------------------------------------------------

    @staticmethod
    def _pair_batch_key(entity1_name: str, entity2_name: str) -> str:
        return f"{entity1_name}\x1f{entity2_name}"

    def resolve_entities_window_batch(
        self,
        entities: List[Dict[str, Any]],
        candidates_by_name: Dict[str, List[Dict[str, Any]]],
        context_text: Optional[str] = None,
        source_document: str = "",
        max_entities_per_call: int = 8,
    ) -> Dict[str, Dict[str, Any]]:
        """窗口级实体对齐：把多个待裁决实体合并进一次 LLM 调用。

        Args:
            entities: [{"name","content"}] 待裁决实体（调用方已预筛掉免 LLM 快路径）
            candidates_by_name: 每个实体的候选列表（与单实体 resolve 相同结构）
            max_entities_per_call: 单次调用实体数上限，超出自动拆批

        Returns:
            {entity_name: verdict}，verdict schema 与 resolve_entity_candidates_batch 一致；
            出错时对应实体缺省（调用方回退到逐实体调用）。
        """
        verdicts: Dict[str, Dict[str, Any]] = {}
        if not entities:
            return verdicts
        max_entities_per_call = max(1, int(max_entities_per_call))

        _prev_step = getattr(self, "_current_distill_step", None)
        self._current_distill_step = "09s_window_batch_entities"
        try:
            return self._resolve_entities_window_batch_inner(
                entities, candidates_by_name, context_text, source_document,
                max_entities_per_call)
        finally:
            self._current_distill_step = _prev_step

    def _resolve_entities_window_batch_inner(
        self,
        entities: List[Dict[str, Any]],
        candidates_by_name: Dict[str, List[Dict[str, Any]]],
        context_text: Optional[str],
        source_document: str,
        max_entities_per_call: int,
    ) -> Dict[str, Dict[str, Any]]:
        verdicts: Dict[str, Dict[str, Any]] = {}

        context_note = ""
        if context_text:
            context_note = (f"\n<原文上下文>\n{_truncate(context_text, 800)}\n</原文上下文>")

        for start in range(0, len(entities), max_entities_per_call):
            chunk = entities[start:start + max_entities_per_call]
            blocks = []
            for ent in chunk:
                name = str(ent.get("name", ""))
                cands = candidates_by_name.get(name) or []
                cand_lines = []
                for idx, cand in enumerate(cands[:6], 1):
                    _mt = cand.get('name_match_type', 'none')
                    mt_note = ""
                    if _mt == "substring":
                        mt_note = "（名称子串，可能是简称/别名）"
                    elif _mt == "exact":
                        mt_note = "（核心名称完全相同）"
                    cand_lines.append(
                        f"  候选{idx}: family_id={cand.get('family_id', '')} | "
                        f"name={cand.get('name', '')}{mt_note} | content={_content_snippet(cand, 120)}"
                    )
                blocks.append(
                    f"<待对齐实体 name=\"{name}\">\n"
                    f"- content: {_content_snippet(ent, 160)}\n"
                    + ("\n".join(cand_lines) if cand_lines else "  （无候选）")
                    + "\n</待对齐实体>"
                )
            prompt = f"""以下是对同一窗口内多个待入库实体的对齐裁决任务。{context_note}

<待对齐实体列表>
{chr(10).join(blocks)}
</待对齐实体列表>

请逐实体判断：该实体与哪个候选在文本中扮演相同角色（角色指纹对比）。各实体独立判断，互不影响。

输出一个 ```json``` 代码块：
{{"results": [{{"name": "实体名", "match_existing_id": "", "update_mode": "reuse_existing|merge_into_latest|create_new", "merged_name": "", "relations_to_create": [{{"family_id": "", "relation_content": ""}}], "confidence": 0.0}}]}}"""

            try:
                result, _ = self.call_llm_until_json_parses(
                    [{"role": "user", "content": prompt}],
                    parse_fn=self._parse_json_response,
                    json_parse_retries=1,
                )
                rows = result.get("results") if isinstance(result, dict) else None
                if not isinstance(rows, list):
                    continue
                by_name = {}
                for ent in chunk:
                    by_name[str(ent.get("name", ""))] = ent
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    name = str(row.get("name", "") or "")
                    if name not in by_name:
                        continue
                    row.setdefault("match_existing_id", "")
                    row.setdefault("update_mode", "create_new")
                    row.setdefault("merged_name", "")
                    row.setdefault("relations_to_create", [])
                    row.setdefault("confidence", 0.0)
                    verdicts[name] = row
            except Exception as e:
                wprint_info(f"[window_batch] 实体窗口批量裁决失败（{start // max_entities_per_call + 1} 批）: {e}")
                continue
        return verdicts

    def resolve_relation_pairs_window_batch(
        self,
        pairs: List[Dict[str, Any]],
        context_text: Optional[str] = None,
        source_document: str = "",
        max_pairs_per_call: int = 8,
    ) -> Dict[str, Dict[str, Any]]:
        """窗口级关系对齐：把多个待裁决实体对合并进一次 LLM 调用。

        Args:
            pairs: [{"entity1_name","entity2_name","new_relation_contents":[...],
                     "existing_relations":[{family_id,content,source_document}]}]
            max_pairs_per_call: 单次调用实体对数上限，超出自动拆批

        Returns:
            {"{entity1_name}\\x1f{entity2_name}": verdict}，verdict schema 与
            resolve_relation_pair_batch 一致；出错时对应 pair 缺省（回退逐对调用）。
        """
        verdicts: Dict[str, Dict[str, Any]] = {}
        if not pairs:
            return verdicts
        max_pairs_per_call = max(1, int(max_pairs_per_call))

        _prev_step = getattr(self, "_current_distill_step", None)
        self._current_distill_step = "10s_window_batch_relations"
        try:
            return self._resolve_relation_pairs_window_batch_inner(
                pairs, context_text, source_document, max_pairs_per_call)
        finally:
            self._current_distill_step = _prev_step

    def _resolve_relation_pairs_window_batch_inner(
        self,
        pairs: List[Dict[str, Any]],
        context_text: Optional[str],
        source_document: str,
        max_pairs_per_call: int,
    ) -> Dict[str, Dict[str, Any]]:
        verdicts: Dict[str, Dict[str, Any]] = {}

        for start in range(0, len(pairs), max_pairs_per_call):
            chunk = pairs[start:start + max_pairs_per_call]
            blocks = []
            for pair in chunk:
                e1 = str(pair.get("entity1_name", ""))
                e2 = str(pair.get("entity2_name", ""))
                new_lines = "\n".join(
                    f"  - {c}" for c in (pair.get("new_relation_contents") or [])[:4]
                )
                exist_lines = "\n".join(
                    f"  - family_id={r.get('family_id', '')} | {r.get('content', '')[:100]}"
                    for r in (pair.get("existing_relations") or [])[:6]
                )
                blocks.append(
                    f"<待裁决关系对 entity1=\"{e1}\" entity2=\"{e2}\">\n"
                    f"新关系描述:\n{new_lines or '  （无）'}\n"
                    f"已有关系:\n{exist_lines or '  （无）'}\n"
                    f"</待裁决关系对>"
                )
            prompt = f"""以下是对同一窗口内多个新关系的 match/create 裁决任务。

<待裁决关系对列表>
{chr(10).join(blocks)}
</待裁决关系对列表>

请逐对判断：新关系是否与该对某个已有关系描述同一性质的关系。参考 source_document={source_document or '(当前文档)'}，跨文档时只有明确同一语义关系才可匹配。各对独立判断，互不影响。

输出一个 ```json``` 代码块：
{{"results": [{{"entity1_name": "", "entity2_name": "", "action": "match_existing|create_new", "matched_relation_id": "", "need_update": false, "confidence": 0.0}}]}}"""

            try:
                result, _ = self.call_llm_until_json_parses(
                    [{"role": "user", "content": prompt}],
                    parse_fn=self._parse_json_response,
                    json_parse_retries=1,
                )
                rows = result.get("results") if isinstance(result, dict) else None
                if not isinstance(rows, list):
                    continue
                key_map = {}
                for pair in chunk:
                    key_map[self._pair_batch_key(
                        str(pair.get("entity1_name", "")), str(pair.get("entity2_name", "")))] = pair
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    key = self._pair_batch_key(
                        str(row.get("entity1_name", "") or ""),
                        str(row.get("entity2_name", "") or ""))
                    if key not in key_map:
                        continue
                    row.setdefault("action", "create_new")
                    row.setdefault("matched_relation_id", row.pop("matched_family_id", ""))
                    row.setdefault("need_update", row.get("action") == "create_new")
                    row.setdefault("confidence", 0.0)
                    verdicts[key] = row
            except Exception as e:
                wprint_info(f"[window_batch] 关系窗口批量裁决失败（{start // max_pairs_per_call + 1} 批）: {e}")
                continue
        return verdicts

