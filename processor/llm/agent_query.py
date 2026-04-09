"""Agent-First API - 元查询、解释、置信度、建议。"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..models import Entity
from ..utils import wprint

logger = logging.getLogger(__name__)

AGENT_QUERY_SYSTEM_PROMPT = """你是一个知识图谱查询助手。你的任务是理解用户的自然语言问题，并将其转化为结构化的查询意图。

你需要分析问题并返回：
- query_type: 搜索类型 ("semantic", "bm25", "hybrid", "traverse")
- query_text: 搜索文本
- entity_name: 如果问题针对特定实体，提取实体名称
- max_results: 建议返回数量

请只输出一个 ```json ... ``` 代码块。"""

EXPLAIN_ENTITY_SYSTEM_PROMPT = """你是一个知识图谱解释助手。你的任务是基于实体数据生成自然语言解释。

解释应该：
1. 简洁明了，适合 Agent 直接使用
2. 包含实体的关键信息
3. 使用用户提问的语言风格

请直接输出解释文本，不要使用 JSON 格式。"""

SUGGESTIONS_SYSTEM_PROMPT = """你是一个知识图谱分析助手。你的任务是分析图谱，给出改进建议。

分析维度：
1. 可能重复的实体
2. 可能过时的关系
3. 建议执行的记忆巩固

请只输出一个 ```json ... ``` 代码块，包含键 "suggestions"（值为建议列表数组，每个包含 "type"、"description" 和 "priority"）。"""


class AgentQueryMixin:
    """Agent 查询 mixin，通过 LLMClient 多继承使用。"""

    async def agent_meta_query(self, question: str, graph_id: str = "default") -> Dict[str, Any]:
        """Agent 元查询：解析自然语言问题，返回结构化查询 + 自然语言回答。

        Args:
            question: 用户的自然语言问题
            graph_id: 图谱 ID

        Returns:
            包含 query_plan 和 answer 的字典
        """
        # 步骤 1：解析查询意图
        intent = await self._parse_query_intent(question)

        # 步骤 2：执行查询
        results = await self._execute_query(intent, graph_id)

        # 步骤 3：生成自然语言回答
        answer = await self._generate_answer(question, results)

        return {
            "query_plan": intent,
            "results": results,
            "answer": answer,
        }

    async def explain_entity(
        self,
        entity: Entity,
        aspect: str = "summary",
    ) -> str:
        """基于实体数据生成自然语言解释。

        Args:
            entity: 实体对象
            aspect: 解释方面 ("summary" | "relations" | "timeline" | "contradictions")

        Returns:
            自然语言解释
        """
        entity_info = f"""实体名称: {entity.name}
实体内容: {entity.content[:500]}
摘要: {getattr(entity, 'summary', '无') or '无'}
事件时间: {entity.event_time.isoformat() if entity.event_time else '未知'}"""

        prompt = f"""<实体信息>
{entity_info}
</实体信息>

<解释方面>
{aspect}
</解释方面>

请解释该实体的{aspect}方面："""

        messages = [
            {"role": "system", "content": EXPLAIN_ENTITY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result = self._call_llm(prompt="", messages=messages)
            return (result or "").strip()
        except Exception as e:
            wprint(f"实体解释失败: {e}")
            return entity.content[:300]

    async def generate_suggestions(
        self,
        entities: List[Entity],
        entity_count: int = 0,
        relation_count: int = 0,
    ) -> List[dict]:
        """分析图谱，返回改进建议。

        Args:
            entities: 实体列表（样本）
            entity_count: 总实体数
            relation_count: 总关系数

        Returns:
            建议列表
        """
        sample = "\n".join(
            f"- {e.name}: {e.content[:100]}"
            for e in entities[:30]
        )

        prompt = f"""<图谱统计>
实体总数: {entity_count}
关系总数: {relation_count}
</图谱统计>

<实体样本>
{sample}
</实体样本>

请分析图谱并给出改进建议："""

        messages = [
            {"role": "system", "content": SUGGESTIONS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result, _ = self.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: self._parse_suggestions(r),
                json_parse_retries=2,
            )
            return result
        except Exception as e:
            wprint(f"建议生成失败: {e}")
            return []

    async def _parse_query_intent(self, question: str) -> Dict[str, Any]:
        """解析查询意图。"""
        prompt = f"""<用户问题>
{question}
</用户问题>

请分析问题并返回查询意图："""

        messages = [
            {"role": "system", "content": AGENT_QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result, _ = self.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: self._parse_json_response(r),
                json_parse_retries=2,
            )
            return result if isinstance(result, dict) else {"query_type": "hybrid", "query_text": question}
        except Exception:
            return {"query_type": "hybrid", "query_text": question}

    async def _execute_query(self, intent: Dict[str, Any], graph_id: str) -> Dict[str, Any]:
        """执行查询（由 API 层完成实际搜索，这里返回意图供 API 层使用）。"""
        return intent

    async def _generate_answer(self, question: str, results: Dict[str, Any]) -> str:
        """基于查询结果生成自然语言回答。"""
        # 简化版：返回查询文本和结果摘要
        query_text = results.get("query_text", question)
        return f"查询: {query_text}"

    def synthesize_answer(self, question: str, entity_dicts: list, relation_dicts: list) -> str:
        """基于搜索结果用 LLM 综合回答用户问题。

        同步方法，供 API 层在获取搜索结果后调用。
        """
        if not entity_dicts and not relation_dicts:
            return "未找到相关的知识图谱数据。"

        # 构建上下文
        context_parts = []
        for e in entity_dicts[:15]:
            name = e.get("name", "")
            summary = e.get("summary") or ""
            content = e.get("content", "")
            snippet = summary if summary else (content[:200] if content else "")
            context_parts.append(f"- 实体【{name}】: {snippet}")

        for r in relation_dicts[:10]:
            e1 = r.get("entity1_name", "")
            e2 = r.get("entity2_name", "")
            content = r.get("content", "")
            context_parts.append(f"- 关系【{e1} ↔ {e2}】: {content[:150]}")

        context = "\n".join(context_parts)

        prompt = f"""<用户问题>
{question}
</用户问题>

<知识图谱检索结果>
{context}
</知识图谱检索结果>

请基于以上知识图谱检索结果，简洁地回答用户的问题。如果检索结果不足以完整回答，请基于已有信息给出部分回答并指出信息缺口。直接输出回答文本，不要使用 JSON 格式。"""

        messages = [
            {"role": "system", "content": "你是一个知识图谱问答助手。基于检索到的实体和关系数据，用简洁、准确的语言回答用户问题。"},
            {"role": "user", "content": prompt},
        ]

        try:
            result = self._call_llm(prompt="", messages=messages)
            return (result or "").strip() or "无法基于检索结果生成回答。"
        except Exception as e:
            logger.warning("回答综合失败: %s", e)
            # 回退：拼接实体名称
            names = [e.get("name", "") for e in entity_dicts[:5] if e.get("name")]
            return f"找到相关实体: {', '.join(names)}" if names else "检索完成但无法生成回答。"

    def _parse_suggestions(self, response: str) -> List[dict]:
        """解析建议列表响应。"""
        result = self._parse_json_response(response)
        if not isinstance(result, dict):
            return []
        suggestions = result.get("suggestions", [])
        if not isinstance(suggestions, list):
            return []
        valid = []
        for s in suggestions:
            if isinstance(s, dict) and "description" in s:
                valid.append({
                    "type": str(s.get("type", "general")),
                    "description": str(s["description"]),
                    "priority": str(s.get("priority", "medium")),
                })
        return valid
