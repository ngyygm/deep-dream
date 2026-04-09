"""LLM客户端 - 知识图谱整理相关操作。"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..utils import wprint
from .prompts import (
    ANALYZE_ENTITY_CANDIDATES_PRELIMINARY_SYSTEM_PROMPT,
    RESOLVE_ENTITY_CANDIDATES_BATCH_SYSTEM_PROMPT,
    analyze_entity_pair_detailed_system_prompt,
    RESOLVE_RELATION_PAIR_BATCH_SYSTEM_PROMPT,
    analyze_entity_duplicates_system_prompt,
    DETAILED_JUDGMENT_PROCESS,
)


class _ConsolidationMixin:
    """知识图谱整理相关的 LLM 操作（mixin，通过 LLMClient 多继承使用）。"""

    def analyze_entity_candidates_preliminary(self, entities_group: List[Dict[str, Any]],
                                              content_snippet_length: int = 200,
                                              context_text: Optional[str] = None) -> Dict[str, Any]:
        """
        初步筛选：分析一组候选实体，返回可能需要合并或存在关系的候选列表

        这是两步判断流程的第一步，使用截断的content进行快速筛选。

        Args:
            entities_group: 候选实体组，每个实体包含:
                - family_id: 实体ID
                - name: 实体名称
                - content: 实体内容描述
                - version_count: 该实体的版本数量
            content_snippet_length: 传入LLM的实体content最大长度（默认300字符）
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
        entities_str = ""
        current_entity = entities_group[0]

        entities_str += f"""
【当前分析的实体】实体1:
- family_id: {current_entity.get('family_id', '')}
- name: {current_entity.get('name', '')}
- version_count: {current_entity.get('version_count', 1)}
- source_document: {current_entity.get('source_document', '') or '(当前文档)'}
- content: {current_entity.get('content', '')[:content_snippet_length]}{'...' if len(current_entity.get('content', '')) > content_snippet_length else ''}
"""

        for i, entity in enumerate(entities_group[1:], 2):
            content = entity.get('content', '')
            content_snippet = content[:content_snippet_length] + ('...' if len(content) > content_snippet_length else '')

            entities_str += f"""
【候选实体】实体{i}:
- family_id: {entity.get('family_id', '')}
- name: {entity.get('name', '')}
- version_count: {entity.get('version_count', 1)}
- source_document: {entity.get('source_document', '') or '(未知文档)'}
- content: {content_snippet}
"""

        # 构建上下文信息
        context_note = ""
        if context_text:
            # 限制上下文长度，避免prompt过长
            context_snippet = context_text[:800] + ('...' if len(context_text) > 800 else '')
            context_note = f"""
**📄 上下文信息**（实体出现的原始文本片段）：
{context_snippet}

**重要**：请参考上下文信息来理解实体出现的具体场景，这有助于更准确地判断实体之间的关系。
"""

        prompt = f"""请对以下实体进行初步筛选，判断哪些候选实体**可能**与当前实体有关联：
{context_note}
<候选实体列表>
{entities_str}
</候选实体列表>

**重要提示**：
- **必须基于content描述进行判断**，不能仅凭名称相似就判断为有关系
- **必须参考 source_document**：不同文档中的实体默认要更谨慎，只有在明确是同一概念或明确存在直接关系时，才允许合并/建边
- 仔细对比两个实体的content描述，判断是否为同一对象或存在明确关联
- 如果两个实体是不同对象且没有明确的、直接的、有意义的关联，应该放入no_action
- 如果关联模糊、间接或牵强，应该放入no_action，不要放入possible_relations

请分析每个候选实体，将它们分类到possible_merges、possible_relations或no_action中。

**输出要求**：
- 只需要输出family_id列表，不需要其他字段
- 每个候选实体只能出现在一个列表中（possible_merges、possible_relations或no_action中的一个）
- 只输出一个 ```json ... ``` 代码块，不要包含任何其他文字或说明

只输出一个 ```json ... ``` 代码块："""

        # 调用LLM
        try:
            response = self._call_llm(prompt, system_prompt)

            # 解析JSON响应
            result = self._parse_json_response(response)

            if not isinstance(result, dict):
                raise ValueError("响应格式不正确")

            # 确保必需的字段存在
            if "possible_merges" not in result:
                result["possible_merges"] = []
            if "possible_relations" not in result:
                result["possible_relations"] = []
            if "no_action" not in result:
                result["no_action"] = []

            return result

        except Exception as e:
            wprint(f"  初步筛选出错: {e}")
            # 出错时默认 no_action，避免误合并
            return {
                "possible_merges": [],
                "possible_relations": [],
                "no_action": [e.get("family_id") for e in entities_group[1:] if e.get("family_id")],
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
        if existing_relations and len(existing_relations) > 0:
            existing_relations_note = f"""
**⚠️ 这两个实体之间已存在以下关系**：
"""
            for rel in existing_relations:
                existing_relations_note += f"- {rel.get('content', '无描述')}\n"
            existing_relations_note += """
**重要**：已存在关系表明这两个实体之前被判断为不同实体。除非有明确证据证明它们确实是同一对象，否则不应该合并。
"""

        system_prompt = analyze_entity_pair_detailed_system_prompt(
            existing_relations_note
        )

        # 构建上下文信息
        context_note = ""
        if context_text:
            # 限制上下文长度，避免prompt过长
            context_snippet = context_text[:1000] + ('...' if len(context_text) > 1000 else '')
            context_note = f"""
**📄 上下文信息**（实体出现的原始文本片段）：
{context_snippet}

**重要**：请参考上下文信息来理解实体出现的具体场景，这有助于：
- 更准确地判断两个实体之间的关系
- 生成更准确、更具体的关系描述
- 避免基于猜测创建关系
"""

        prompt = f"""<当前实体>
- family_id: {current_entity.get('family_id', '')}
- name: {current_entity.get('name', '')}
- version_count: {current_entity.get('version_count', 1)}
- source_document: {current_entity.get('source_document', '') or '(当前文档)'}
- content: {current_entity.get('content', '')}
</当前实体>

<候选实体>
- family_id: {candidate_entity.get('family_id', '')}
- name: {candidate_entity.get('name', '')}
- version_count: {candidate_entity.get('version_count', 1)}
- source_document: {candidate_entity.get('source_document', '') or '(未知文档)'}
- content: {candidate_entity.get('content', '')}
</候选实体>
{context_note}
{DETAILED_JUDGMENT_PROCESS}

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
            if "reason" not in result:
                result["reason"] = ""
            if "relation_content" not in result:
                result["relation_content"] = ""
            if "merge_target" not in result:
                result["merge_target"] = ""

            return result

        except Exception as e:
            wprint(f"  精细化判断出错: {e}")
            return {
                "action": "no_action",
                "reason": f"判断出错: {str(e)}",
                "relation_content": "",
                "merge_target": "",
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
                "merged_content": current_entity.get("content", ""),
                "relations_to_create": [],
                "confidence": 1.0,
            }

        system_prompt = RESOLVE_ENTITY_CANDIDATES_BATCH_SYSTEM_PROMPT

        context_note = ""
        if context_text:
            context_snippet = context_text[:1200] + ("..." if len(context_text) > 1200 else "")
            context_note = f"\n上下文：\n{context_snippet}\n"

        candidates_str = []
        for idx, candidate in enumerate(candidates, 1):
            candidates_str.append(
                f"""候选{idx}:
- family_id: {candidate.get('family_id', '')}
- name: {candidate.get('name', '')}
- version_count: {candidate.get('version_count', 1)}
- source_document: {candidate.get('source_document', '') or '(未知文档)'}
- lexical_score: {candidate.get('lexical_score', 0):.4f}
- dense_score: {candidate.get('dense_score', 0):.4f}
- content: {candidate.get('content', '')}"""
            )

        prompt = f"""<当前实体>
- family_id: {current_entity.get('family_id', 'NEW_ENTITY')}
- name: {current_entity.get('name', '')}
- source_document: {current_entity.get('source_document', '') or '(当前文档)'}
- content: {current_entity.get('content', '')}
</当前实体>
{context_note}
<候选实体列表>
{chr(10).join(candidates_str)}
</候选实体列表>

请输出一个 ```json ... ``` 代码块，代码块内部为：
{{
  "match_existing_id": "若应合并到已有实体则填写 family_id，否则为空字符串",
  "update_mode": "reuse_existing | merge_into_latest | create_new",
  "merged_name": "若需要，给出最终名称，否则为空字符串",
  "merged_content": "若需要更新/合并，给出最终内容，否则为空字符串",
  "relations_to_create": [
    {{"family_id": "候选family_id", "relation_content": "与当前实体的自然语言关系"}}
  ],
  "confidence": 0.0
}}

要求：
- 只能选一个 match_existing_id
- 若不合并，但与若干候选存在明确关系，可放入 relations_to_create
- 必须参考 source_document；跨文档时只有在明确是同一概念实体时才允许合并或融合内容
- 若信息不足，confidence 降低
- 只输出一个 ```json ... ``` 代码块"""

        try:
            result = self._parse_json_response(self._call_llm(prompt, system_prompt))
            if not isinstance(result, dict):
                raise ValueError("响应格式不正确")
            result.setdefault("match_existing_id", "")
            result.setdefault("update_mode", "create_new")
            result.setdefault("merged_name", "")
            result.setdefault("merged_content", "")
            result.setdefault("relations_to_create", [])
            result.setdefault("confidence", 0.0)
            return result
        except Exception as e:
            return {
                "match_existing_id": "",
                "update_mode": "fallback",
                "merged_name": "",
                "merged_content": "",
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

请输出一个 ```json ... ``` 代码块（action 选 match_existing 或 create_new），代码块内部为：
{{
  "action": "match_existing | create_new",
  "matched_family_id": "若命中已有关系则填写 family_id，否则为空字符串",
  "need_update": true,
  "merged_content": "若需要创建或更新，给出最终关系内容；否则为空字符串",
  "confidence": 0.0
}}

要求：
- 必须参考 source_document；跨文档时只有在明确表达的是同一对概念之间的同一关系时，才允许匹配/融合。"""

        try:
            result = self._parse_json_response(self._call_llm(prompt, system_prompt))
            if not isinstance(result, dict):
                raise ValueError("响应格式不正确")
            result.setdefault("action", "create_new")
            result.setdefault("matched_family_id", "")
            result.setdefault("need_update", result.get("action") == "create_new")
            result.setdefault("merged_content", "")
            result.setdefault("confidence", 0.0)
            return result
        except Exception as e:
            return {
                "action": "fallback",
                "matched_family_id": "",
                "need_update": False,
                "merged_content": "",
                "confidence": 0.0,
                "error": str(e),
            }

    def analyze_entity_duplicates(self, entities_group: List[Dict[str, Any]],
                                  memory_contexts: Optional[Dict[str, str]] = None,
                                  content_snippet_length: int = 300,
                                  existing_relations_between_entities: Optional[Dict[str, List[Dict]]] = None) -> Dict[str, Any]:
        """
        分析一组候选实体，判断是否为同一实体或存在别名关系（保留兼容性）

        注意：这是旧版本的方法，建议使用两步判断流程：
        1. analyze_entity_candidates_preliminary - 初步筛选
        2. analyze_entity_pair_detailed - 精细化判断

        Args:
            entities_group: 候选实体组，每个实体包含:
                - family_id: 实体ID
                - name: 实体名称
                - content: 实体内容描述
                - version_count: 该实体的版本数量
            memory_contexts: 可选的记忆上下文字典，key是family_id，value是对应的缓存记忆文本
            content_snippet_length: 传入LLM的实体content最大长度（默认300字符）
            existing_relations_between_entities: 已存在关系的实体对信息字典，
                key为 "entity1_id|entity2_id" 格式（按字母序排序），
                value为该实体对之间的关系列表，每个关系包含:
                    - family_id: 关系ID
                    - content: 关系描述

        Returns:
            分析结果，包含:
            - merge_groups: 需要合并的实体组列表
            - alias_relations: 需要创建的别名关系列表
        """
        if not entities_group or len(entities_group) < 2:
            return {"merge_groups": [], "alias_relations": []}

        # 构建已有关系的提示信息
        existing_relations_note = ""
        if existing_relations_between_entities:
            existing_relations_note = """

**⚠️ 已存在关系的实体对（绝对不能合并！）**：
以下实体对之间**已经存在直接关联关系**，这表明它们是**不同的实体**，**绝对不能合并**！
对于这些实体对，你只需要判断：
- 如果有新的关系信息需要补充，放入 alias_relations 中（用于更新或新增关系）
- 如果没有新信息，则不需要处理

"""
            for pair_key, relations in existing_relations_between_entities.items():
                family_ids = pair_key.split("|")
                if len(relations) > 0:
                    rel_contents = [r.get('content', '无描述')[:50] for r in relations[:3]]
                    existing_relations_note += f"- 实体 {family_ids[0]} 和 {family_ids[1]} 之间已有关系：{'; '.join(rel_contents)}\n"

        system_prompt = analyze_entity_duplicates_system_prompt(
            existing_relations_note
        )

        # 构建实体信息字符串
        # 第一个实体是当前分析的实体，其他是候选实体
        entities_str = ""
        for i, entity in enumerate(entities_group, 1):
            family_id = entity.get('family_id', '')
            name = entity.get('name', '')
            content = entity.get('content', '')
            version_count = entity.get('version_count', 1)

            # 标注当前实体和候选实体
            if i == 1:
                entity_label = "【当前分析的实体】"
            else:
                entity_label = "【候选实体】"

            # 限制content长度
            content_snippet = content[:content_snippet_length] + ('...' if len(content) > content_snippet_length else '')

            entities_str += f"""
{entity_label} 实体 {i}:
- family_id: {family_id}
- name: {name}
- version_count: {version_count}
- content: {content_snippet}
"""

        prompt = f"""<实体列表>
{entities_str}
</实体列表>

**重要说明**：
- **实体1**是当前正在分析的实体
- **实体2-N**是通过相似度搜索找到的候选实体
- 你需要判断：**实体1与每个候选实体之间**是否需要合并或存在别名关系
- **不要判断候选实体之间的关系**，只关注实体1与候选实体之间的关系

**⚠️ 关键判断标准**：
- **只有content描述的是完全相同的对象时，才能合并**
- **如果两个实体有关联但描述的是不同的对象，不应该合并，只应该创建关系边**
- **例如：《三体》（小说）和《科幻世界》（杂志）虽然有关联，但它们是不同的实体，不应该合并**

请仔细分析实体1与每个候选实体的名称和**content描述**，判断：
1. **实体1与哪些候选实体应该合并**（content明确描述的是同一对象）
2. **实体1与哪些候选实体存在别名关系**（一个是另一个的别名/称呼，且指向同一对象）
3. **实体1与哪些候选实体确实存在明确的、有意义的关联但不是同一对象**（**必须**创建关系边，但不合并）
4. **实体1与哪些候选实体是无关的**（不应该合并或创建关系）

**⚠️ 重要规则**：
- **仔细阅读每个实体的content描述**，这是判断的关键
- **如果content描述的是不同的对象，即使名称相似或有关联，也绝对不能合并**
- **如果两个实体之间已存在关系（见上方"已存在关系的实体对"），说明它们已被确认为不同实体，绝对不能合并！**
- **关键判断标准**：只有两个实体**确实存在明确的、直接的、有意义的关联**时，才应该在alias_relations中创建关系边
- **关联必须是明确的、直接的、有意义的**（例如：使用关系、包含关系、交互关系、从属关系等）
- **不能是模糊的、间接的或牵强的关联**（例如：只是因为在同一场景中出现，但没有实际的交互或联系）
- **如果不确定两个实体之间是否确实存在明确的、有意义的关联，宁可不创建关系边**
- 版本数量多的实体应该作为合并的目标（target）
- 如果实体1和候选实体是同一对象，应该合并（保留版本数多的作为target）
- **如果不确定是否为同一对象，宁可不合并，也不创建关系边**
- 合并和别名关系可以同时存在（先合并，再创建别名关系）

只输出一个 ```json ... ``` 代码块，不要其他文字："""

        # 重试机制：最多重试3次
        max_retries = 3
        last_error = None
        last_response = None

        for attempt in range(max_retries):
            try:
                # 调用LLM
                response = self._call_llm(prompt, system_prompt)
                last_response = response

                # 解析JSON响应
                result = self._parse_json_response(response)

                # 验证结果格式
                if not isinstance(result, dict):
                    raise ValueError("响应格式不正确：不是字典类型")

                # 确保必需的字段存在
                if "merge_groups" not in result:
                    result["merge_groups"] = []
                if "alias_relations" not in result:
                    result["alias_relations"] = []

                # 验证和清理alias_relations中的数据
                valid_alias_relations = []
                for rel in result.get("alias_relations", []):
                    if isinstance(rel, dict):
                        entity1_id = rel.get("entity1_id")
                        entity2_id = rel.get("entity2_id")
                        # 如果family_id为空或None，跳过这个关系
                        if entity1_id and entity2_id:
                            valid_alias_relations.append(rel)
                        else:
                            # 记录警告但不跳过，尝试后续通过名称查找
                            wprint(f"    警告：alias_relation缺少family_id: {rel}")
                            valid_alias_relations.append(rel)
                result["alias_relations"] = valid_alias_relations

                # 解析成功，返回结果
                if attempt > 0:
                    wprint(f"  重试成功（第 {attempt + 1} 次尝试）")
                return result

            except (json.JSONDecodeError, Exception) as e:
                last_error = e

                # 如果还有重试机会，继续重试
                if attempt < max_retries - 1:
                    wprint(f"  解析实体分析JSON失败（第 {attempt + 1}/{max_retries} 次尝试）: {e}")
                    wprint(f"  响应内容: {last_response[:500] if last_response else 'N/A'}...")
                    wprint(f"  正在重试...")
                    continue
                else:
                    # 最后一次尝试也失败
                    wprint(f"  解析实体分析JSON失败（已重试 {max_retries} 次）: {e}")
                    wprint(f"  响应内容: {last_response[:500] if last_response else 'N/A'}...")

        # 所有重试都失败，返回错误
        error_msg = str(last_error) if last_error else "未知错误"
        return {"merge_groups": [], "alias_relations": [], "error": f"重试 {max_retries} 次后仍然失败: {error_msg}"}
