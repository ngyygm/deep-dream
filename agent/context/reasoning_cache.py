"""
推理缓存

追踪推理过程中的状态，包括子目标、已知事实、假设和缺失信息
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from enum import Enum
import json


class QuestionType(Enum):
    """问题类型"""
    DIRECT = "direct"  # 直接查询（如"史强是谁"）
    REASONING = "reasoning"  # 需要推理（如"他们是什么关系"）
    TEMPORAL_REASONING = "temporal_reasoning"  # 需要时序推理（如"第几次见面"）


class GoalStatus(Enum):
    """子目标状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubGoal:
    """子目标"""
    goal_id: str
    description: str
    status: GoalStatus = GoalStatus.PENDING
    result: Any = None
    depends_on: List[str] = field(default_factory=list)  # 依赖的其他子目标
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
            "depends_on": self.depends_on
        }


@dataclass
class Hypothesis:
    """假设"""
    hypothesis_id: str
    content: str
    confidence: float = 0.0  # 0-1
    evidence: List[str] = field(default_factory=list)  # 支持的证据
    counter_evidence: List[str] = field(default_factory=list)  # 反驳的证据
    verified: Optional[bool] = None  # None=未验证, True=已验证, False=已否定
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "content": self.content,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "counter_evidence": self.counter_evidence,
            "verified": self.verified
        }


@dataclass
class TriedQuery:
    """已尝试的查询"""
    tool_name: str
    parameters: Dict[str, Any]
    iteration: int
    success: bool
    result_summary: str  # 结果摘要
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "iteration": self.iteration,
            "success": self.success,
            "result_summary": self.result_summary
        }
    
    def matches(self, tool_name: str, parameters: Dict[str, Any]) -> bool:
        """检查是否与给定的查询匹配（避免重复）"""
        if self.tool_name != tool_name:
            return False
        # 比较关键参数
        for key, value in parameters.items():
            if key in self.parameters and self.parameters[key] == value:
                continue
            return False
        return True


@dataclass
class ReasoningState:
    """推理状态"""
    question: str
    question_type: QuestionType = QuestionType.DIRECT
    
    # 推理目标分解
    sub_goals: List[SubGoal] = field(default_factory=list)
    
    # 信息追踪
    known_facts: Dict[str, Any] = field(default_factory=dict)  # key -> fact
    entity_facts: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # entity_id -> facts
    relation_facts: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # relation_id -> facts
    
    # 假设
    hypotheses: List[Hypothesis] = field(default_factory=list)
    
    # 缺失信息
    missing_info: List[str] = field(default_factory=list)
    
    # 查询历史
    tried_queries: List[TriedQuery] = field(default_factory=list)
    failed_strategies: List[str] = field(default_factory=list)
    
    # 推理结论
    conclusion: Optional[str] = None
    confidence: float = 0.0
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "question_type": self.question_type.value,
            "sub_goals": [g.to_dict() for g in self.sub_goals],
            "known_facts": self.known_facts,
            "entity_facts": self.entity_facts,
            "relation_facts": self.relation_facts,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "missing_info": self.missing_info,
            "tried_queries_count": len(self.tried_queries),
            "failed_strategies": self.failed_strategies,
            "conclusion": self.conclusion,
            "confidence": self.confidence
        }


class ReasoningCache:
    """
    推理缓存
    
    管理推理过程中的状态，提供以下功能：
    1. 追踪子目标和完成状态
    2. 记录已知事实和缺失信息
    3. 管理假设和验证状态
    4. 避免重复查询
    """
    
    def __init__(self):
        self.state: Optional[ReasoningState] = None
        self._goal_counter = 0
        self._hypothesis_counter = 0
    
    def init_state(
        self,
        question: str,
        question_type: QuestionType = QuestionType.DIRECT
    ) -> ReasoningState:
        """
        初始化推理状态
        
        Args:
            question: 用户问题
            question_type: 问题类型
            
        Returns:
            新的推理状态
        """
        self.state = ReasoningState(
            question=question,
            question_type=question_type
        )
        self._goal_counter = 0
        self._hypothesis_counter = 0
        return self.state
    
    def add_sub_goal(
        self,
        description: str,
        depends_on: List[str] = None
    ) -> SubGoal:
        """
        添加子目标
        
        Args:
            description: 目标描述
            depends_on: 依赖的其他子目标 ID
            
        Returns:
            新创建的子目标
        """
        if self.state is None:
            raise RuntimeError("推理状态未初始化")
        
        self._goal_counter += 1
        goal = SubGoal(
            goal_id=f"goal_{self._goal_counter}",
            description=description,
            depends_on=depends_on or []
        )
        self.state.sub_goals.append(goal)
        self.state.updated_at = datetime.now()
        return goal
    
    def update_goal_status(
        self,
        goal_id: str,
        status: GoalStatus,
        result: Any = None
    ):
        """更新子目标状态"""
        if self.state is None:
            return
        
        for goal in self.state.sub_goals:
            if goal.goal_id == goal_id:
                goal.status = status
                if result is not None:
                    goal.result = result
                self.state.updated_at = datetime.now()
                break
    
    def get_pending_goals(self) -> List[SubGoal]:
        """获取待处理的子目标（依赖已满足）"""
        if self.state is None:
            return []
        
        completed_ids = {
            g.goal_id for g in self.state.sub_goals
            if g.status == GoalStatus.COMPLETED
        }
        
        pending = []
        for goal in self.state.sub_goals:
            if goal.status == GoalStatus.PENDING:
                # 检查依赖是否满足
                deps_satisfied = all(
                    dep in completed_ids for dep in goal.depends_on
                )
                if deps_satisfied:
                    pending.append(goal)
        
        return pending
    
    def add_known_fact(self, key: str, value: Any):
        """添加已知事实"""
        if self.state is None:
            return
        
        self.state.known_facts[key] = value
        self.state.updated_at = datetime.now()
    
    def add_entity_fact(self, entity_id: str, facts: Dict[str, Any]):
        """添加实体相关事实"""
        if self.state is None:
            return
        
        if entity_id not in self.state.entity_facts:
            self.state.entity_facts[entity_id] = {}
        
        self.state.entity_facts[entity_id].update(facts)
        self.state.updated_at = datetime.now()
    
    def add_relation_fact(self, relation_id: str, facts: Dict[str, Any]):
        """添加关系相关事实"""
        if self.state is None:
            return
        
        if relation_id not in self.state.relation_facts:
            self.state.relation_facts[relation_id] = {}
        
        self.state.relation_facts[relation_id].update(facts)
        self.state.updated_at = datetime.now()
    
    def add_hypothesis(
        self,
        content: str,
        confidence: float = 0.5
    ) -> Hypothesis:
        """
        添加假设
        
        Args:
            content: 假设内容
            confidence: 初始置信度
            
        Returns:
            新创建的假设
        """
        if self.state is None:
            raise RuntimeError("推理状态未初始化")
        
        self._hypothesis_counter += 1
        hypothesis = Hypothesis(
            hypothesis_id=f"hyp_{self._hypothesis_counter}",
            content=content,
            confidence=confidence
        )
        self.state.hypotheses.append(hypothesis)
        self.state.updated_at = datetime.now()
        return hypothesis
    
    def update_hypothesis(
        self,
        hypothesis_id: str,
        evidence: str = None,
        counter_evidence: str = None,
        confidence_delta: float = 0.0,
        verified: bool = None
    ):
        """更新假设"""
        if self.state is None:
            return
        
        for hyp in self.state.hypotheses:
            if hyp.hypothesis_id == hypothesis_id:
                if evidence:
                    hyp.evidence.append(evidence)
                if counter_evidence:
                    hyp.counter_evidence.append(counter_evidence)
                hyp.confidence = max(0, min(1, hyp.confidence + confidence_delta))
                if verified is not None:
                    hyp.verified = verified
                self.state.updated_at = datetime.now()
                break
    
    def add_missing_info(self, info: str):
        """添加缺失信息"""
        if self.state is None:
            return
        
        if info not in self.state.missing_info:
            self.state.missing_info.append(info)
            self.state.updated_at = datetime.now()
    
    def remove_missing_info(self, info: str):
        """移除已获取的信息"""
        if self.state is None:
            return
        
        if info in self.state.missing_info:
            self.state.missing_info.remove(info)
            self.state.updated_at = datetime.now()
    
    def record_query(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        iteration: int,
        success: bool,
        result_summary: str
    ):
        """记录已尝试的查询"""
        if self.state is None:
            return
        
        query = TriedQuery(
            tool_name=tool_name,
            parameters=parameters,
            iteration=iteration,
            success=success,
            result_summary=result_summary
        )
        self.state.tried_queries.append(query)
        self.state.updated_at = datetime.now()
    
    def has_tried_query(self, tool_name: str, parameters: Dict[str, Any]) -> bool:
        """检查是否已尝试过相同的查询"""
        if self.state is None:
            return False
        
        for query in self.state.tried_queries:
            if query.matches(tool_name, parameters):
                return True
        return False
    
    def add_failed_strategy(self, strategy: str):
        """记录失败的策略"""
        if self.state is None:
            return
        
        if strategy not in self.state.failed_strategies:
            self.state.failed_strategies.append(strategy)
            self.state.updated_at = datetime.now()
    
    def set_conclusion(self, conclusion: str, confidence: float = 1.0):
        """设置推理结论"""
        if self.state is None:
            return
        
        self.state.conclusion = conclusion
        self.state.confidence = confidence
        self.state.updated_at = datetime.now()
    
    def get_state_summary(self) -> str:
        """
        获取推理状态摘要（用于传递给 Planner）
        
        Returns:
            状态摘要文本
        """
        if self.state is None:
            return "推理状态未初始化"
        
        lines = []
        
        # 问题类型
        lines.append(f"**问题类型**: {self.state.question_type.value}")
        
        # 子目标状态
        if self.state.sub_goals:
            lines.append("\n**子目标**:")
            for goal in self.state.sub_goals:
                status_icon = {
                    GoalStatus.PENDING: "⏳",
                    GoalStatus.IN_PROGRESS: "🔄",
                    GoalStatus.COMPLETED: "✅",
                    GoalStatus.FAILED: "❌"
                }.get(goal.status, "?")
                lines.append(f"  {status_icon} {goal.description}")
                if goal.result:
                    lines.append(f"      结果: {str(goal.result)[:100]}...")
        
        # 已知事实
        if self.state.known_facts or self.state.entity_facts:
            lines.append("\n**已知事实**:")
            for key, value in self.state.known_facts.items():
                lines.append(f"  - {key}: {str(value)[:100]}")
            for eid, facts in self.state.entity_facts.items():
                name = facts.get("name", eid)
                # 显示关键字段的值，而不是字段名列表
                key_info = []
                if "content" in facts:
                    content = str(facts["content"])[:80]
                    key_info.append(f"content='{content}...'")
                if "physical_time" in facts:
                    ptime = facts["physical_time"]
                    key_info.append(f"time='{ptime}'")
                if "memory_cache_id" in facts:
                    cache_id = str(facts["memory_cache_id"])[:20]
                    key_info.append(f"cache='{cache_id}...'")
                info_str = ", ".join(key_info) if key_info else "无详细信息"
                lines.append(f"  - 实体 [{name}]: {info_str}")
        
        # 假设
        active_hypotheses = [h for h in self.state.hypotheses if h.verified is None]
        if active_hypotheses:
            lines.append("\n**待验证假设**:")
            for hyp in active_hypotheses:
                lines.append(f"  - [{hyp.confidence:.1%}] {hyp.content}")
        
        # 缺失信息
        if self.state.missing_info:
            lines.append("\n**缺失信息**:")
            for info in self.state.missing_info:
                lines.append(f"  - {info}")
        
        # 失败策略
        if self.state.failed_strategies:
            lines.append("\n**已失败的策略** (避免重复):")
            for strategy in self.state.failed_strategies[-5:]:  # 只显示最近5个
                lines.append(f"  - {strategy}")
        
        # 查询历史统计
        if self.state.tried_queries:
            success_count = sum(1 for q in self.state.tried_queries if q.success)
            lines.append(f"\n**查询统计**: {len(self.state.tried_queries)} 次查询, {success_count} 次成功")
        
        return "\n".join(lines)
    
    def is_reasoning_complete(self) -> bool:
        """
        检查推理是否完成
        
        Returns:
            是否完成
        """
        if self.state is None:
            return False
        
        # 如果有结论，认为完成
        if self.state.conclusion:
            return True
        
        # 如果所有子目标都完成
        if self.state.sub_goals:
            all_completed = all(
                g.status == GoalStatus.COMPLETED
                for g in self.state.sub_goals
            )
            if all_completed:
                return True
        
        # 如果没有缺失信息且没有待处理的子目标
        if not self.state.missing_info and not self.get_pending_goals():
            return True
        
        return False
    
    def get_reasoning_progress(self) -> Dict[str, Any]:
        """
        获取推理进度
        
        Returns:
            进度信息
        """
        if self.state is None:
            return {"progress": 0, "status": "not_started"}
        
        if self.state.conclusion:
            return {"progress": 100, "status": "completed"}
        
        total_goals = len(self.state.sub_goals)
        if total_goals == 0:
            return {"progress": 50, "status": "no_goals"}
        
        completed = sum(
            1 for g in self.state.sub_goals
            if g.status == GoalStatus.COMPLETED
        )
        
        progress = int((completed / total_goals) * 100)
        
        return {
            "progress": progress,
            "status": "in_progress",
            "completed_goals": completed,
            "total_goals": total_goals,
            "missing_info_count": len(self.state.missing_info)
        }
