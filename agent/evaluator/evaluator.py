"""
Evaluator 评估器

负责判断当前收集的记忆是否足够回答问题
"""
import json
import re
from typing import List, Dict, Any, Optional

from ..llm.base import BaseLLMClient
from ..models import EvaluationResult
from ..context.reasoning_cache import ReasoningState, GoalStatus
from ..logger import AgentLogger, get_logger
from .prompts import (
    EVALUATOR_SYSTEM_PROMPT,
    EVALUATOR_REQUEST_TEMPLATE,
    EVALUATOR_REQUEST_WITH_REASONING,
    REASONING_EVALUATOR_SYSTEM_PROMPT,
    format_collected_memories
)


class Evaluator:
    """评估器 - 判断记忆是否足够"""
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        logger: Optional[AgentLogger] = None
    ):
        """
        初始化评估器
        
        Args:
            llm_client: LLM 客户端
            logger: 日志记录器
        """
        self.llm_client = llm_client
        self.logger = logger or get_logger()
    
    def evaluate(
        self,
        question: str,
        collected_memories: List[Dict[str, Any]],
        iteration: int = 1,
        reasoning_state: Optional[ReasoningState] = None
    ) -> EvaluationResult:
        """
        评估当前收集的记忆是否足够
        
        Args:
            question: 用户问题
            collected_memories: 已收集的记忆
            iteration: 当前迭代次数
            reasoning_state: 推理状态（可选）
            
        Returns:
            评估结果
        """
        # 格式化记忆
        memories_str = format_collected_memories(collected_memories)
        
        # 根据是否有推理状态选择评估方式
        if reasoning_state and reasoning_state.question_type.value != "direct":
            return self._evaluate_with_reasoning(
                question, memories_str, iteration, reasoning_state
            )
        
        # 简单评估
        request = EVALUATOR_REQUEST_TEMPLATE.format(
            question=question,
            collected_memories=memories_str,
            iteration=iteration
        )
        
        messages = [
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": request}
        ]
        
        self.logger.debug(f"Evaluator request: {request[:500]}...")
        response = self.llm_client.chat(messages)
        self.logger.debug(f"Evaluator response: {response.content[:500]}...")
        
        result = self._parse_response(response.content)
        self.logger.evaluate(result.is_sufficient, result.reasoning)
        
        return result
    
    def _evaluate_with_reasoning(
        self,
        question: str,
        memories_str: str,
        iteration: int,
        reasoning_state: ReasoningState
    ) -> EvaluationResult:
        """带推理状态的评估"""
        # 格式化子目标
        sub_goals_str = self._format_sub_goals(reasoning_state.sub_goals)
        
        # 格式化已知事实
        known_facts_str = self._format_known_facts(
            reasoning_state.known_facts,
            reasoning_state.entity_facts,
            reasoning_state.relation_facts
        )
        
        # 格式化缺失信息
        missing_str = "\n".join(f"- {info}" for info in reasoning_state.missing_info) or "无"
        
        # 格式化假设
        hypotheses_str = self._format_hypotheses(reasoning_state.hypotheses)
        
        request = EVALUATOR_REQUEST_WITH_REASONING.format(
            question=question,
            question_type=reasoning_state.question_type.value,
            sub_goals=sub_goals_str,
            known_facts=known_facts_str,
            missing_info=missing_str,
            hypotheses=hypotheses_str,
            collected_memories=memories_str,
            iteration=iteration
        )
        
        messages = [
            {"role": "system", "content": REASONING_EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": request}
        ]
        
        self.logger.debug(f"Reasoning evaluator request: {request[:500]}...")
        response = self.llm_client.chat(messages)
        self.logger.debug(f"Reasoning evaluator response: {response.content[:500]}...")
        
        result = self._parse_reasoning_response(response.content)
        self.logger.evaluate(result.is_sufficient, result.reasoning)
        
        return result
    
    async def aevaluate(
        self,
        question: str,
        collected_memories: List[Dict[str, Any]],
        iteration: int = 1,
        reasoning_state: Optional[ReasoningState] = None
    ) -> EvaluationResult:
        """异步版本的评估"""
        memories_str = format_collected_memories(collected_memories)
        
        if reasoning_state and reasoning_state.question_type.value != "direct":
            return await self._aevaluate_with_reasoning(
                question, memories_str, iteration, reasoning_state
            )
        
        request = EVALUATOR_REQUEST_TEMPLATE.format(
            question=question,
            collected_memories=memories_str,
            iteration=iteration
        )
        
        messages = [
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": request}
        ]
        
        response = await self.llm_client.achat(messages)
        result = self._parse_response(response.content)
        self.logger.evaluate(result.is_sufficient, result.reasoning)
        
        return result
    
    async def _aevaluate_with_reasoning(
        self,
        question: str,
        memories_str: str,
        iteration: int,
        reasoning_state: ReasoningState
    ) -> EvaluationResult:
        """异步带推理状态的评估"""
        sub_goals_str = self._format_sub_goals(reasoning_state.sub_goals)
        known_facts_str = self._format_known_facts(
            reasoning_state.known_facts,
            reasoning_state.entity_facts,
            reasoning_state.relation_facts
        )
        missing_str = "\n".join(f"- {info}" for info in reasoning_state.missing_info) or "无"
        hypotheses_str = self._format_hypotheses(reasoning_state.hypotheses)
        
        request = EVALUATOR_REQUEST_WITH_REASONING.format(
            question=question,
            question_type=reasoning_state.question_type.value,
            sub_goals=sub_goals_str,
            known_facts=known_facts_str,
            missing_info=missing_str,
            hypotheses=hypotheses_str,
            collected_memories=memories_str,
            iteration=iteration
        )
        
        messages = [
            {"role": "system", "content": REASONING_EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": request}
        ]
        
        response = await self.llm_client.achat(messages)
        result = self._parse_reasoning_response(response.content)
        self.logger.evaluate(result.is_sufficient, result.reasoning)
        
        return result
    
    def _format_sub_goals(self, sub_goals: List) -> str:
        """格式化子目标"""
        if not sub_goals:
            return "无子目标"
        
        lines = []
        for goal in sub_goals:
            status_icon = {
                GoalStatus.PENDING: "⏳",
                GoalStatus.IN_PROGRESS: "🔄",
                GoalStatus.COMPLETED: "✅",
                GoalStatus.FAILED: "❌"
            }.get(goal.status, "?")
            lines.append(f"{status_icon} {goal.description}")
            if goal.result:
                lines.append(f"   结果: {str(goal.result)[:100]}")
        
        return "\n".join(lines)
    
    def _format_known_facts(
        self,
        known_facts: Dict,
        entity_facts: Dict,
        relation_facts: Dict
    ) -> str:
        """格式化已知事实"""
        lines = []
        
        if known_facts:
            lines.append("**一般事实:**")
            for key, value in list(known_facts.items())[:10]:
                lines.append(f"- {key}: {str(value)[:100]}")
        
        if entity_facts:
            lines.append("\n**实体信息:**")
            for eid, facts in list(entity_facts.items())[:5]:
                name = facts.get("name", eid)
                lines.append(f"- {name}: {facts.get('content', '')[:100]}...")
        
        if relation_facts:
            lines.append("\n**关系信息:**")
            for rid, facts in list(relation_facts.items())[:5]:
                e1 = facts.get("entity1_name", "?")
                e2 = facts.get("entity2_name", "?")
                lines.append(f"- {e1} -- {e2}: {facts.get('content', '')[:80]}...")
        
        return "\n".join(lines) or "无"
    
    def _format_hypotheses(self, hypotheses: List) -> str:
        """格式化假设"""
        if not hypotheses:
            return "无"
        
        lines = []
        for hyp in hypotheses:
            verified_str = "?" if hyp.verified is None else ("✓" if hyp.verified else "✗")
            lines.append(f"- [{verified_str}] [{hyp.confidence:.0%}] {hyp.content}")
        
        return "\n".join(lines)
    
    def _parse_reasoning_response(self, content: str) -> EvaluationResult:
        """解析推理评估响应"""
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = content.strip()
        
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            # 尝试从文本推断
            is_sufficient = any(keyword in content.lower() for keyword in 
                              ["足够", "充足", "sufficient", "可以推理", "can_reason"])
            
            return EvaluationResult(
                is_sufficient=is_sufficient,
                reasoning=content[:500],
                next_action="" if is_sufficient else "继续查询相关信息"
            )
        
        # 提取推理可行性
        reasoning_feasibility = result.get("reasoning_feasibility", {})
        can_reason = reasoning_feasibility.get("can_reason", False)
        
        # 综合判断：信息充足或可以推理
        is_sufficient = result.get("is_sufficient", False) or can_reason
        
        # 提取问题类型调整建议
        adjustment_info = result.get("question_type_adjustment", {})
        question_type_adjustment = None
        if adjustment_info.get("should_adjust", False):
            from agent.models import QuestionTypeAdjustment
            question_type_adjustment = QuestionTypeAdjustment(
                should_adjust=True,
                new_type=adjustment_info.get("new_type"),
                reason=adjustment_info.get("reason", "")
            )
        
        return EvaluationResult(
            is_sufficient=is_sufficient,
            reasoning=result.get("reasoning", ""),
            memories_to_keep=result.get("memories_to_keep", []),
            next_action=result.get("next_action", ""),
            question_type_adjustment=question_type_adjustment
        )
    
    def _parse_response(self, content: str) -> EvaluationResult:
        """解析 LLM 响应"""
        # 尝试提取 JSON
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = content.strip()
        
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            # 尝试从文本中推断结果
            is_sufficient = any(keyword in content.lower() for keyword in 
                              ["足够", "充足", "sufficient", "可以回答", "enough"])
            
            return EvaluationResult(
                is_sufficient=is_sufficient,
                reasoning=content[:500],
                next_action="" if is_sufficient else "继续查询相关信息"
            )
        
        return EvaluationResult(
            is_sufficient=result.get("is_sufficient", False),
            reasoning=result.get("reasoning", ""),
            memories_to_keep=result.get("memories_to_keep", []),
            next_action=result.get("next_action", "")
        )
    
    def quick_check(self, collected_memories: List[Dict[str, Any]]) -> bool:
        """
        快速检查（不调用 LLM）
        
        用于简单场景的快速判断，如：
        - 没有收集到任何记忆
        - 收集到了明确的错误
        
        Args:
            collected_memories: 已收集的记忆
            
        Returns:
            是否需要继续查询（True = 需要继续）
        """
        if not collected_memories:
            return True  # 没有记忆，需要继续
        
        # 检查最后一个查询结果
        last_memory = collected_memories[-1]
        if isinstance(last_memory, dict):
            result = last_memory.get("result", {})
            if isinstance(result, dict):
                # 如果最后一个查询找到了实体或关系，可能需要进一步探索
                if result.get("entities") or result.get("relations"):
                    return True
                # 如果查询成功但没有结果，可能需要换一种查询方式
                if result.get("success") and result.get("count", 0) == 0:
                    return True
        
        return False
