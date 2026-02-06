"""
Summarizer 总结器

负责筛选有用信息并生成推理总结
"""
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..llm.base import BaseLLMClient
from ..context.reasoning_cache import ReasoningCache, ReasoningState
from ..logger import AgentLogger, get_logger
from .prompts import (
    FILTER_INFO_PROMPT,
    SUMMARY_PROMPT,
    CONTEXT_GENERATION_PROMPT,
    format_entity_for_filter,
    format_relation_for_filter,
    format_sub_goals,
    format_hypotheses_for_summary
)


@dataclass
class SummaryResult:
    """总结结果"""
    # 答案
    question: str
    answer: str
    confidence: float
    answer_type: str  # "direct", "inferred", "uncertain"
    
    # 推理链路
    reasoning_chain: List[Dict[str, str]] = field(default_factory=list)
    
    # 证据
    supporting_evidence: List[str] = field(default_factory=list)
    entities_used: List[Dict[str, Any]] = field(default_factory=list)
    relations_used: List[Dict[str, Any]] = field(default_factory=list)
    
    # 局限性
    limitations: List[str] = field(default_factory=list)
    
    # 上下文文本（供外部 LLM 使用）
    context_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "answer_type": self.answer_type,
            "reasoning_chain": self.reasoning_chain,
            "evidence": {
                "supporting": self.supporting_evidence,
                "entities_used": [e.get("name", str(e)) for e in self.entities_used],
                "relations_used": [r.get("content", str(r))[:100] for r in self.relations_used]
            },
            "limitations": self.limitations,
            "context_text": self.context_text
        }


class Summarizer:
    """
    总结器
    
    主要职责：
    1. 筛选与问题相关的实体和关系
    2. 生成推理链路说明
    3. 输出最终推理结论
    4. 生成供外部 LLM 使用的上下文
    """
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        logger: Optional[AgentLogger] = None
    ):
        """
        初始化总结器
        
        Args:
            llm_client: LLM 客户端
            logger: 日志记录器
        """
        self.llm_client = llm_client
        self.logger = logger or get_logger()
    
    def summarize(
        self,
        reasoning_state: ReasoningState,
        filter_info: bool = True
    ) -> SummaryResult:
        """
        生成推理总结
        
        Args:
            reasoning_state: 推理状态
            filter_info: 是否先筛选信息
            
        Returns:
            总结结果
        """
        self.logger.debug("生成推理总结")
        
        # 1. 筛选相关信息
        if filter_info:
            filtered = self._filter_relevant_info(reasoning_state)
            relevant_entities = filtered.get("relevant_entities", [])
            relevant_relations = filtered.get("relevant_relations", [])
        else:
            relevant_entities = list(reasoning_state.entity_facts.values())
            relevant_relations = list(reasoning_state.relation_facts.values())
        
        # 2. 生成结构化总结
        summary_data = self._generate_summary(
            reasoning_state,
            relevant_entities,
            relevant_relations
        )
        
        # 3. 生成上下文文本
        context_text = self._generate_context_text(
            reasoning_state,
            relevant_entities,
            relevant_relations
        )
        
        # 4. 构建结果
        result = SummaryResult(
            question=reasoning_state.question,
            answer=summary_data.get("summary", {}).get("answer", reasoning_state.conclusion or "无法确定"),
            confidence=summary_data.get("summary", {}).get("confidence", reasoning_state.confidence),
            answer_type=summary_data.get("summary", {}).get("answer_type", "uncertain"),
            reasoning_chain=summary_data.get("reasoning_chain", []),
            supporting_evidence=summary_data.get("evidence", {}).get("supporting", []),
            entities_used=relevant_entities,
            relations_used=relevant_relations,
            limitations=summary_data.get("limitations", []),
            context_text=context_text
        )
        
        self.logger.info(f"生成总结完成，置信度: {result.confidence:.0%}")
        
        return result
    
    async def asummarize(
        self,
        reasoning_state: ReasoningState,
        filter_info: bool = True
    ) -> SummaryResult:
        """异步版本的总结生成"""
        self.logger.debug("生成推理总结（异步）")
        
        if filter_info:
            filtered = await self._afilter_relevant_info(reasoning_state)
            relevant_entities = filtered.get("relevant_entities", [])
            relevant_relations = filtered.get("relevant_relations", [])
        else:
            relevant_entities = list(reasoning_state.entity_facts.values())
            relevant_relations = list(reasoning_state.relation_facts.values())
        
        summary_data = await self._agenerate_summary(
            reasoning_state,
            relevant_entities,
            relevant_relations
        )
        
        context_text = await self._agenerate_context_text(
            reasoning_state,
            relevant_entities,
            relevant_relations
        )
        
        result = SummaryResult(
            question=reasoning_state.question,
            answer=summary_data.get("summary", {}).get("answer", reasoning_state.conclusion or "无法确定"),
            confidence=summary_data.get("summary", {}).get("confidence", reasoning_state.confidence),
            answer_type=summary_data.get("summary", {}).get("answer_type", "uncertain"),
            reasoning_chain=summary_data.get("reasoning_chain", []),
            supporting_evidence=summary_data.get("evidence", {}).get("supporting", []),
            entities_used=relevant_entities,
            relations_used=relevant_relations,
            limitations=summary_data.get("limitations", []),
            context_text=context_text
        )
        
        return result
    
    def _filter_relevant_info(self, state: ReasoningState) -> Dict[str, Any]:
        """筛选相关信息"""
        if not state.entity_facts and not state.relation_facts:
            return {"relevant_entities": [], "relevant_relations": []}
        
        prompt = FILTER_INFO_PROMPT.format(
            question=state.question,
            entity_info=format_entity_for_filter(state.entity_facts),
            relation_info=format_relation_for_filter(state.relation_facts),
            other_facts=self._format_other_facts(state.known_facts)
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat(messages)
        result = self._parse_json_response(response.content)
        
        # 将筛选结果与原始数据关联
        relevant_entities = []
        for item in result.get("relevant_entities", []):
            eid = item.get("entity_id")
            if eid and eid in state.entity_facts:
                entity = state.entity_facts[eid].copy()
                entity["relevance"] = item.get("relevance", "")
                entity["key_info"] = item.get("key_info", "")
                relevant_entities.append(entity)
        
        relevant_relations = []
        for item in result.get("relevant_relations", []):
            rid = item.get("relation_id")
            if rid and rid in state.relation_facts:
                relation = state.relation_facts[rid].copy()
                relation["relevance"] = item.get("relevance", "")
                relation["key_info"] = item.get("key_info", "")
                relevant_relations.append(relation)
        
        # 如果 LLM 没有筛选出任何内容，使用原始数据
        if not relevant_entities and state.entity_facts:
            relevant_entities = list(state.entity_facts.values())
        if not relevant_relations and state.relation_facts:
            relevant_relations = list(state.relation_facts.values())
        
        return {
            "relevant_entities": relevant_entities,
            "relevant_relations": relevant_relations,
            "filter_reasoning": result.get("filter_reasoning", "")
        }
    
    async def _afilter_relevant_info(self, state: ReasoningState) -> Dict[str, Any]:
        """异步筛选相关信息"""
        if not state.entity_facts and not state.relation_facts:
            return {"relevant_entities": [], "relevant_relations": []}
        
        prompt = FILTER_INFO_PROMPT.format(
            question=state.question,
            entity_info=format_entity_for_filter(state.entity_facts),
            relation_info=format_relation_for_filter(state.relation_facts),
            other_facts=self._format_other_facts(state.known_facts)
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm_client.achat(messages)
        result = self._parse_json_response(response.content)
        
        relevant_entities = []
        for item in result.get("relevant_entities", []):
            eid = item.get("entity_id")
            if eid and eid in state.entity_facts:
                entity = state.entity_facts[eid].copy()
                entity["relevance"] = item.get("relevance", "")
                relevant_entities.append(entity)
        
        relevant_relations = []
        for item in result.get("relevant_relations", []):
            rid = item.get("relation_id")
            if rid and rid in state.relation_facts:
                relation = state.relation_facts[rid].copy()
                relation["relevance"] = item.get("relevance", "")
                relevant_relations.append(relation)
        
        if not relevant_entities and state.entity_facts:
            relevant_entities = list(state.entity_facts.values())
        if not relevant_relations and state.relation_facts:
            relevant_relations = list(state.relation_facts.values())
        
        return {
            "relevant_entities": relevant_entities,
            "relevant_relations": relevant_relations
        }
    
    def _generate_summary(
        self,
        state: ReasoningState,
        entities: List[Dict],
        relations: List[Dict]
    ) -> Dict[str, Any]:
        """生成结构化总结"""
        # 格式化子目标
        sub_goals_str = format_sub_goals([g.to_dict() for g in state.sub_goals])
        
        # 格式化关键事实
        key_facts_str = self._format_other_facts(state.known_facts)
        
        # 格式化实体
        entities_str = "\n".join([
            f"- {e.get('name', 'Unknown')}: {e.get('content', '')[:150]}..."
            for e in entities[:10]
        ]) or "无"
        
        # 格式化关系
        relations_str = "\n".join([
            f"- [{r.get('entity1_name', '?')}] -- [{r.get('entity2_name', '?')}]: {r.get('content', '')[:100]}..."
            for r in relations[:10]
        ]) or "无"
        
        # 格式化假设
        hypotheses_str = format_hypotheses_for_summary([h.to_dict() for h in state.hypotheses])
        
        prompt = SUMMARY_PROMPT.format(
            question=state.question,
            question_type=state.question_type.value,
            sub_goals=sub_goals_str,
            key_facts=key_facts_str,
            entities=entities_str,
            relations=relations_str,
            hypotheses=hypotheses_str,
            conclusion=state.conclusion or "尚未得出结论"
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat(messages)
        
        return self._parse_json_response(response.content)
    
    async def _agenerate_summary(
        self,
        state: ReasoningState,
        entities: List[Dict],
        relations: List[Dict]
    ) -> Dict[str, Any]:
        """异步生成结构化总结"""
        sub_goals_str = format_sub_goals([g.to_dict() for g in state.sub_goals])
        key_facts_str = self._format_other_facts(state.known_facts)
        
        entities_str = "\n".join([
            f"- {e.get('name', 'Unknown')}: {e.get('content', '')[:150]}..."
            for e in entities[:10]
        ]) or "无"
        
        relations_str = "\n".join([
            f"- [{r.get('entity1_name', '?')}] -- [{r.get('entity2_name', '?')}]: {r.get('content', '')[:100]}..."
            for r in relations[:10]
        ]) or "无"
        
        hypotheses_str = format_hypotheses_for_summary([h.to_dict() for h in state.hypotheses])
        
        prompt = SUMMARY_PROMPT.format(
            question=state.question,
            question_type=state.question_type.value,
            sub_goals=sub_goals_str,
            key_facts=key_facts_str,
            entities=entities_str,
            relations=relations_str,
            hypotheses=hypotheses_str,
            conclusion=state.conclusion or "尚未得出结论"
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm_client.achat(messages)
        
        return self._parse_json_response(response.content)
    
    def _generate_context_text(
        self,
        state: ReasoningState,
        entities: List[Dict],
        relations: List[Dict]
    ) -> str:
        """生成供外部 LLM 使用的上下文文本"""
        # 格式化实体
        entities_str = "\n".join([
            f"- **{e.get('name', 'Unknown')}**: {e.get('content', '')}"
            for e in entities[:10]
        ]) or "无相关实体"
        
        # 格式化关系
        relations_str = "\n".join([
            f"- {r.get('entity1_name', '?')} 与 {r.get('entity2_name', '?')}: {r.get('content', '')}"
            for r in relations[:10]
        ]) or "无相关关系"
        
        prompt = CONTEXT_GENERATION_PROMPT.format(
            question=state.question,
            entities=entities_str,
            relations=relations_str,
            conclusion=state.conclusion or "需要进一步分析"
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat(messages)
        
        return response.content.strip()
    
    async def _agenerate_context_text(
        self,
        state: ReasoningState,
        entities: List[Dict],
        relations: List[Dict]
    ) -> str:
        """异步生成上下文文本"""
        entities_str = "\n".join([
            f"- **{e.get('name', 'Unknown')}**: {e.get('content', '')}"
            for e in entities[:10]
        ]) or "无相关实体"
        
        relations_str = "\n".join([
            f"- {r.get('entity1_name', '?')} 与 {r.get('entity2_name', '?')}: {r.get('content', '')}"
            for r in relations[:10]
        ]) or "无相关关系"
        
        prompt = CONTEXT_GENERATION_PROMPT.format(
            question=state.question,
            entities=entities_str,
            relations=relations_str,
            conclusion=state.conclusion or "需要进一步分析"
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm_client.achat(messages)
        
        return response.content.strip()
    
    def _format_other_facts(self, facts: Dict[str, Any]) -> str:
        """格式化其他已知事实"""
        if not facts:
            return "无"
        
        lines = []
        for key, value in facts.items():
            if key.startswith("reasoning_"):
                continue  # 跳过推理步骤
            lines.append(f"- {key}: {str(value)[:200]}")
        
        return "\n".join(lines) or "无"
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """解析 LLM 的 JSON 响应"""
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = content.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
                return json.loads(json_str)
            except json.JSONDecodeError:
                self.logger.warning(f"无法解析 JSON: {content[:200]}")
                return {}
    
    def quick_summary(self, state: ReasoningState) -> str:
        """
        快速生成简单总结（不调用 LLM）
        
        Args:
            state: 推理状态
            
        Returns:
            简单总结文本
        """
        lines = []
        
        # 问题
        lines.append(f"**问题**: {state.question}")
        lines.append("")
        
        # 结论
        if state.conclusion:
            lines.append(f"**结论**: {state.conclusion}")
            lines.append(f"**置信度**: {state.confidence:.0%}")
        else:
            lines.append("**结论**: 无法确定")
        lines.append("")
        
        # 关键实体
        if state.entity_facts:
            lines.append("**相关实体**:")
            for eid, facts in list(state.entity_facts.items())[:5]:
                name = facts.get("name", eid)
                content = facts.get("content", "")[:100]
                lines.append(f"- {name}: {content}...")
            lines.append("")
        
        # 关键关系
        if state.relation_facts:
            lines.append("**相关关系**:")
            for rid, facts in list(state.relation_facts.items())[:5]:
                e1 = facts.get("entity1_name", "?")
                e2 = facts.get("entity2_name", "?")
                content = facts.get("content", "")[:80]
                lines.append(f"- {e1} -- {e2}: {content}...")
        
        return "\n".join(lines)
