"""
Reasoner 推理器

负责问题分析、推理规划和结论生成
"""
import json
import re
from typing import List, Dict, Any, Optional, Tuple

from ..llm.base import BaseLLMClient
from ..context.reasoning_cache import (
    ReasoningCache,
    ReasoningState,
    QuestionType,
    GoalStatus,
    SubGoal,
    Hypothesis
)
from ..logger import AgentLogger, get_logger
from .prompts import (
    QUESTION_ANALYSIS_PROMPT,
    FACT_INTEGRATION_PROMPT,
    CONCLUSION_PROMPT,
    TEMPORAL_REASONING_PROMPT,
    format_known_facts,
    format_entity_facts,
    format_relation_facts,
    format_hypotheses
)
from .strategies import (
    ReasoningStrategy,
    TemporalStrategy,
    RelationStrategy,
    DirectQueryStrategy,
    StrategyResult
)


class Reasoner:
    """
    推理器
    
    主要职责：
    1. 分析问题类型并分解子目标
    2. 整合检索到的信息为可用事实
    3. 验证假设
    4. 生成推理结论
    """
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        reasoning_cache: ReasoningCache,
        logger: Optional[AgentLogger] = None
    ):
        """
        初始化推理器
        
        Args:
            llm_client: LLM 客户端
            reasoning_cache: 推理缓存
            logger: 日志记录器
        """
        self.llm_client = llm_client
        self.cache = reasoning_cache
        self.logger = logger or get_logger()
        
        # 初始化推理策略
        self.strategies: List[ReasoningStrategy] = [
            TemporalStrategy(),
            RelationStrategy(),
            DirectQueryStrategy()
        ]
    
    def analyze_question(self, question: str) -> ReasoningState:
        """
        分析问题并初始化推理状态
        
        Args:
            question: 用户问题
            
        Returns:
            初始化的推理状态
        """
        self.logger.debug(f"分析问题: {question}")
        
        # 调用 LLM 分析问题
        prompt = QUESTION_ANALYSIS_PROMPT.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        
        response = self.llm_client.chat(messages)
        analysis = self._parse_json_response(response.content)
        
        # 确定问题类型
        question_type_str = analysis.get("question_type", "direct")
        question_type = QuestionType(question_type_str) if question_type_str in [qt.value for qt in QuestionType] else QuestionType.DIRECT
        
        # 初始化推理状态
        state = self.cache.init_state(question, question_type)
        
        # 添加子目标
        sub_goals = analysis.get("sub_goals", [])
        for i, goal in enumerate(sub_goals):
            depends_on = [f"goal_{j+1}" for j in goal.get("depends_on", [])]
            self.cache.add_sub_goal(
                description=goal.get("description", f"子目标 {i+1}"),
                depends_on=depends_on
            )
        
        # 添加关键实体和关系作为缺失信息
        key_entities = analysis.get("key_entities", [])
        for entity in key_entities:
            self.cache.add_missing_info(f"实体: {entity}")
        
        key_relations = analysis.get("key_relations", [])
        for relation in key_relations:
            self.cache.add_missing_info(f"关系: {relation}")
        
        # 记录推理提示
        hints = analysis.get("reasoning_hints", "")
        if hints:
            self.cache.add_known_fact("reasoning_hints", hints)
        
        self.logger.info(f"问题类型: {question_type.value}, 子目标数: {len(sub_goals)}")
        
        return state
    
    async def aanalyze_question(self, question: str) -> ReasoningState:
        """异步版本的问题分析"""
        self.logger.debug(f"分析问题: {question}")
        
        prompt = QUESTION_ANALYSIS_PROMPT.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        
        response = await self.llm_client.achat(messages)
        analysis = self._parse_json_response(response.content)
        
        question_type_str = analysis.get("question_type", "direct")
        question_type = QuestionType(question_type_str) if question_type_str in [qt.value for qt in QuestionType] else QuestionType.DIRECT
        
        state = self.cache.init_state(question, question_type)
        
        sub_goals = analysis.get("sub_goals", [])
        for i, goal in enumerate(sub_goals):
            depends_on = [f"goal_{j+1}" for j in goal.get("depends_on", [])]
            self.cache.add_sub_goal(
                description=goal.get("description", f"子目标 {i+1}"),
                depends_on=depends_on
            )
        
        key_entities = analysis.get("key_entities", [])
        for entity in key_entities:
            self.cache.add_missing_info(f"实体: {entity}")
        
        key_relations = analysis.get("key_relations", [])
        for relation in key_relations:
            self.cache.add_missing_info(f"关系: {relation}")
        
        return state
    
    def integrate_facts(self, collected_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        整合收集到的信息为可用事实
        
        Args:
            collected_info: 工具返回的原始信息列表
            
        Returns:
            整合后的事实
        """
        if not self.cache.state:
            return {}
        
        state = self.cache.state
        
        # 从收集的信息中提取实体和关系
        for info in collected_info:
            result = info.get("result", {})
            
            # 提取实体
            if "entities" in result:
                for entity in result["entities"]:
                    entity_id = entity.get("entity_id")
                    if entity_id:
                        self.cache.add_entity_fact(entity_id, {
                            "name": entity.get("name", ""),
                            "content": entity.get("content", ""),
                            "physical_time": entity.get("physical_time")
                        })
                        # 移除相关的缺失信息
                        name = entity.get("name", "")
                        self.cache.remove_missing_info(f"实体: {name}")
            
            # 提取关系
            if "relations" in result:
                for relation in result["relations"]:
                    relation_id = relation.get("relation_id")
                    if relation_id:
                        self.cache.add_relation_fact(relation_id, {
                            "content": relation.get("content", ""),
                            "entity1_name": relation.get("entity1_name", ""),
                            "entity2_name": relation.get("entity2_name", ""),
                            "physical_time": relation.get("physical_time")
                        })
            
            # 提取路径信息
            if "paths" in result:
                for path in result["paths"]:
                    path_desc = path.get("path_description", "")
                    if path_desc:
                        self.cache.add_known_fact(
                            f"path_{len(state.known_facts)}",
                            path_desc
                        )
                    # 提取路径中的关系
                    for edge in path.get("edges", []):
                        if edge.get("relation_id"):
                            self.cache.add_relation_fact(edge["relation_id"], edge)
            
            # 提取版本信息
            if "versions" in result:
                versions = result["versions"]
                if versions:
                    self.cache.add_known_fact("version_count", len(versions))
                    if result.get("earliest_time"):
                        self.cache.add_known_fact("earliest_time", result["earliest_time"])
                    if result.get("latest_time"):
                        self.cache.add_known_fact("latest_time", result["latest_time"])
        
        # 使用 LLM 进一步整合
        if state.entity_facts or state.relation_facts:
            integration_result = self._llm_integrate_facts(collected_info)
            
            # 处理 LLM 返回的新事实
            new_facts = integration_result.get("new_facts", {})
            for key, value in new_facts.items():
                self.cache.add_known_fact(key, value)
            
            # 处理缺失信息
            still_missing = integration_result.get("still_missing", [])
            for info in still_missing:
                self.cache.add_missing_info(info)
            
            # 处理假设
            hypotheses = integration_result.get("hypotheses", [])
            for hyp in hypotheses:
                self.cache.add_hypothesis(
                    content=hyp.get("content", ""),
                    confidence=hyp.get("confidence", 0.5)
                )
        
        return {
            "known_facts": state.known_facts,
            "entity_facts": state.entity_facts,
            "relation_facts": state.relation_facts,
            "missing_info": state.missing_info
        }
    
    def _llm_integrate_facts(self, collected_info: List[Dict]) -> Dict[str, Any]:
        """使用 LLM 整合事实"""
        state = self.cache.state
        
        # 格式化收集的信息
        info_str = self._format_collected_info(collected_info)
        facts_str = format_known_facts(state.known_facts)
        
        prompt = FACT_INTEGRATION_PROMPT.format(
            question=state.question,
            collected_info=info_str,
            known_facts=facts_str
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat(messages)
        
        return self._parse_json_response(response.content)
    
    def try_conclude(self) -> Tuple[bool, Optional[str], float]:
        """
        尝试得出结论
        
        Returns:
            (是否成功, 结论, 置信度)
        """
        if not self.cache.state:
            return False, None, 0.0
        
        state = self.cache.state
        
        # 先尝试使用策略得出结论
        for strategy in self.strategies:
            if strategy.can_handle(state.question_type.value, {"question": state.question}):
                result = strategy.analyze(
                    question=state.question,
                    known_facts=state.known_facts,
                    entity_facts=state.entity_facts,
                    relation_facts=state.relation_facts
                )
                
                if result.success and result.conclusion:
                    self.cache.set_conclusion(result.conclusion, result.confidence)
                    # 保存策略的推理链
                    if result.reasoning_chain:
                        for i, step in enumerate(result.reasoning_chain):
                            self.cache.add_known_fact(f"reasoning_step_{i}", step)
                    # 保存证据
                    if result.evidence:
                        for i, ev in enumerate(result.evidence):
                            self.cache.add_known_fact(f"evidence_{i}", ev)
                    return True, result.conclusion, result.confidence
        
        # 策略无法直接得出结论，使用 LLM 推理
        return self._llm_conclude()
    
    def _llm_conclude(self) -> Tuple[bool, Optional[str], float]:
        """使用 LLM 进行推理得出结论"""
        state = self.cache.state
        
        prompt = CONCLUSION_PROMPT.format(
            question=state.question,
            question_type=state.question_type.value,
            known_facts=format_known_facts(state.known_facts),
            entity_facts=format_entity_facts(state.entity_facts),
            relation_facts=format_relation_facts(state.relation_facts),
            hypotheses=format_hypotheses(state.hypotheses)
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat(messages)
        result = self._parse_json_response(response.content)
        
        if result.get("can_conclude", False):
            conclusion = result.get("conclusion", "")
            confidence = result.get("confidence", 0.5)
            
            self.cache.set_conclusion(conclusion, confidence)
            
            # 记录推理链
            reasoning_chain = result.get("reasoning_chain", [])
            for step in reasoning_chain:
                self.cache.add_known_fact(f"reasoning_step_{len(state.known_facts)}", step)
            
            return True, conclusion, confidence
        else:
            # 无法得出结论，添加缺失信息
            still_needed = result.get("still_needed", [])
            for info in still_needed:
                self.cache.add_missing_info(info)
            
            reason = result.get("reason", "无法确定")
            self.logger.debug(f"无法得出结论: {reason}")
            
            return False, None, 0.0
    
    async def atry_conclude(self) -> Tuple[bool, Optional[str], float]:
        """异步版本的结论生成"""
        if not self.cache.state:
            return False, None, 0.0
        
        state = self.cache.state
        
        # 先尝试策略
        for strategy in self.strategies:
            if strategy.can_handle(state.question_type.value, {"question": state.question}):
                result = strategy.analyze(
                    question=state.question,
                    known_facts=state.known_facts,
                    entity_facts=state.entity_facts,
                    relation_facts=state.relation_facts
                )
                
                if result.success and result.conclusion:
                    self.cache.set_conclusion(result.conclusion, result.confidence)
                    # 保存策略的推理链
                    if result.reasoning_chain:
                        for i, step in enumerate(result.reasoning_chain):
                            self.cache.add_known_fact(f"reasoning_step_{i}", step)
                    # 保存证据
                    if result.evidence:
                        for i, ev in enumerate(result.evidence):
                            self.cache.add_known_fact(f"evidence_{i}", ev)
                    return True, result.conclusion, result.confidence
        
        # 使用 LLM
        prompt = CONCLUSION_PROMPT.format(
            question=state.question,
            question_type=state.question_type.value,
            known_facts=format_known_facts(state.known_facts),
            entity_facts=format_entity_facts(state.entity_facts),
            relation_facts=format_relation_facts(state.relation_facts),
            hypotheses=format_hypotheses(state.hypotheses)
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm_client.achat(messages)
        result = self._parse_json_response(response.content)
        
        if result.get("can_conclude", False):
            conclusion = result.get("conclusion", "")
            confidence = result.get("confidence", 0.5)
            self.cache.set_conclusion(conclusion, confidence)
            return True, conclusion, confidence
        
        return False, None, 0.0
    
    def get_next_strategy(self) -> Optional[Dict[str, Any]]:
        """
        获取下一步推理策略建议
        
        Returns:
            策略建议，包含 strategy 和 reason
        """
        if not self.cache.state:
            return None
        
        state = self.cache.state
        tried = [q.to_dict() for q in state.tried_queries]
        
        # 根据问题类型选择策略
        for strategy in self.strategies:
            if strategy.can_handle(state.question_type.value, {"question": state.question}):
                suggestions = strategy.get_next_queries(
                    question=state.question,
                    known_facts=state.known_facts,
                    entity_facts=state.entity_facts,
                    relation_facts=state.relation_facts,
                    tried_queries=tried
                )
                
                if suggestions:
                    return suggestions[0]
        
        return None
    
    def update_goal_from_result(self, goal_id: str, result: Dict[str, Any]):
        """根据查询结果更新子目标状态"""
        success = result.get("success", False)
        
        if success:
            # 检查是否有实质性结果
            has_data = (
                result.get("entities") or
                result.get("relations") or
                result.get("paths") or
                result.get("versions")
            )
            
            if has_data:
                self.cache.update_goal_status(
                    goal_id,
                    GoalStatus.COMPLETED,
                    result
                )
            else:
                # 成功但无数据，可能需要换策略
                self.cache.update_goal_status(
                    goal_id,
                    GoalStatus.IN_PROGRESS
                )
        else:
            self.cache.update_goal_status(
                goal_id,
                GoalStatus.FAILED,
                result.get("message", "查询失败")
            )
    
    def _format_collected_info(self, collected_info: List[Dict]) -> str:
        """格式化收集的信息"""
        lines = []
        for i, info in enumerate(collected_info, 1):
            tool_name = info.get("tool_name", "unknown")
            result = info.get("result", {})
            
            lines.append(f"### 查询 {i}: {tool_name}")
            
            if "entities" in result:
                lines.append(f"找到 {len(result['entities'])} 个实体")
                for entity in result["entities"][:5]:
                    lines.append(f"  - {entity.get('name')}: {entity.get('content', '')[:100]}")
            
            if "relations" in result:
                lines.append(f"找到 {len(result['relations'])} 个关系")
                for rel in result["relations"][:5]:
                    lines.append(f"  - {rel.get('content', '')[:100]}")
            
            if "paths" in result:
                lines.append(f"找到 {len(result['paths'])} 条路径")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """解析 LLM 的 JSON 响应"""
        # 尝试提取 JSON 块
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = content.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 尝试移除注释
            try:
                json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
                return json.loads(json_str)
            except json.JSONDecodeError:
                self.logger.warning(f"无法解析 JSON: {content[:200]}")
                return {}
