"""
推理策略

不同类型问题的推理策略实现
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class StrategyResult:
    """策略执行结果"""
    success: bool
    conclusion: Optional[str] = None
    confidence: float = 0.0
    reasoning_chain: List[str] = None
    evidence: List[str] = None
    next_steps: List[str] = None
    
    def __post_init__(self):
        if self.reasoning_chain is None:
            self.reasoning_chain = []
        if self.evidence is None:
            self.evidence = []
        if self.next_steps is None:
            self.next_steps = []


class ReasoningStrategy(ABC):
    """推理策略基类"""
    
    @abstractmethod
    def can_handle(self, question_type: str, context: Dict[str, Any]) -> bool:
        """判断是否可以处理该类型问题"""
        pass
    
    @abstractmethod
    def analyze(
        self,
        question: str,
        known_facts: Dict[str, Any],
        entity_facts: Dict[str, Dict],
        relation_facts: Dict[str, Dict]
    ) -> StrategyResult:
        """分析并尝试得出结论"""
        pass
    
    @abstractmethod
    def get_next_queries(
        self,
        question: str,
        known_facts: Dict[str, Any],
        entity_facts: Dict[str, Dict],
        relation_facts: Dict[str, Dict],
        tried_queries: List[Dict]
    ) -> List[Dict[str, Any]]:
        """获取下一步需要执行的查询"""
        pass


class TemporalStrategy(ReasoningStrategy):
    """
    时序推理策略
    
    处理涉及时间顺序的问题，如"第几次"、"最早"、"最晚"等
    """
    
    TEMPORAL_KEYWORDS = [
        "第一次", "第二次", "第三次", "第几次",
        "最早", "最晚", "之前", "之后",
        "首次", "上次", "下次",
        "什么时候", "何时"
    ]
    
    def can_handle(self, question_type: str, context: Dict[str, Any]) -> bool:
        if question_type == "temporal_reasoning":
            return True
        
        question = context.get("question", "")
        return any(kw in question for kw in self.TEMPORAL_KEYWORDS)
    
    def analyze(
        self,
        question: str,
        known_facts: Dict[str, Any],
        entity_facts: Dict[str, Dict],
        relation_facts: Dict[str, Dict]
    ) -> StrategyResult:
        """
        时序分析
        
        步骤：
        1. 收集所有相关事件
        2. 提取时间信息
        3. 排序
        4. 确定目标事件
        """
        reasoning_chain = []
        evidence = []
        
        # 1. 收集事件（从关系中提取）
        events = []
        for rid, facts in relation_facts.items():
            event = {
                "relation_id": rid,
                "content": facts.get("content", ""),
                "entity1": facts.get("entity1_name", ""),
                "entity2": facts.get("entity2_name", ""),
                "physical_time": facts.get("physical_time"),
                "version": facts.get("version", 1)
            }
            events.append(event)
        
        reasoning_chain.append(f"收集到 {len(events)} 个相关事件")
        
        if not events:
            return StrategyResult(
                success=False,
                next_steps=["搜索相关实体的关系", "获取关系的版本历史"]
            )
        
        # 2. 提取时间信息并排序
        events_with_time = []
        events_without_time = []
        
        for event in events:
            if event.get("physical_time"):
                events_with_time.append(event)
            else:
                events_without_time.append(event)
        
        reasoning_chain.append(
            f"其中 {len(events_with_time)} 个有明确时间，"
            f"{len(events_without_time)} 个需要从版本推断"
        )
        
        # 如果缺少时间信息，需要进一步查询
        if events_without_time and not events_with_time:
            return StrategyResult(
                success=False,
                reasoning_chain=reasoning_chain,
                next_steps=[
                    "获取关系的版本历史以确定时间顺序",
                    "查找关系内容中的时间线索"
                ]
            )
        
        # 3. 按时间排序
        if events_with_time:
            events_with_time.sort(key=lambda x: x.get("physical_time", ""))
            reasoning_chain.append("按时间顺序排列事件")
            
            for i, event in enumerate(events_with_time, 1):
                evidence.append(
                    f"第{i}个事件: {event['content'][:50]}... "
                    f"(时间: {event.get('physical_time', 'unknown')})"
                )
        
        # 4. 确定目标（如"第二次"）
        target_order = self._extract_target_order(question)
        
        if target_order and len(events_with_time) >= target_order:
            target_event = events_with_time[target_order - 1]
            conclusion = (
                f"根据时间顺序，第{target_order}次相关事件是：{target_event['content'][:100]}，"
                f"发生时间约为 {target_event.get('physical_time', '未知')}"
            )
            
            return StrategyResult(
                success=True,
                conclusion=conclusion,
                confidence=0.8 if events_with_time else 0.5,
                reasoning_chain=reasoning_chain,
                evidence=evidence
            )
        
        # 无法确定具体是第几个
        return StrategyResult(
            success=False,
            reasoning_chain=reasoning_chain,
            evidence=evidence,
            next_steps=[
                f"需要确定事件的精确时间以判断第{target_order or '?'}次",
                "获取更多版本历史信息"
            ]
        )
    
    def _extract_target_order(self, question: str) -> Optional[int]:
        """从问题中提取目标顺序"""
        order_map = {
            "第一": 1, "首次": 1, "第1": 1,
            "第二": 2, "第2": 2,
            "第三": 3, "第3": 3,
            "第四": 4, "第4": 4,
            "第五": 5, "第5": 5,
            "最早": 1, "最先": 1,
            "最晚": -1, "最后": -1
        }
        
        for keyword, order in order_map.items():
            if keyword in question:
                return order
        
        return None
    
    def get_next_queries(
        self,
        question: str,
        known_facts: Dict[str, Any],
        entity_facts: Dict[str, Dict],
        relation_facts: Dict[str, Dict],
        tried_queries: List[Dict]
    ) -> List[Dict[str, Any]]:
        """获取时序推理需要的下一步查询"""
        queries = []
        
        # 如果还没有关系，先搜索关系
        if not relation_facts:
            # 从问题中提取可能的实体
            queries.append({
                "strategy": "search_entity_relations",
                "reason": "需要获取实体之间的关系以进行时序分析"
            })
        
        # 如果有关系但缺少时间信息，获取版本历史
        if relation_facts:
            for rid in relation_facts:
                queries.append({
                    "strategy": "get_relation_versions",
                    "relation_id": rid,
                    "reason": "需要获取关系的版本历史以确定时间顺序"
                })
        
        return queries


class RelationStrategy(ReasoningStrategy):
    """
    关系推理策略
    
    处理实体之间关系的问题
    """
    
    RELATION_KEYWORDS = [
        "关系", "联系", "相关", "认识",
        "之间", "互动", "交集"
    ]
    
    def can_handle(self, question_type: str, context: Dict[str, Any]) -> bool:
        if question_type == "reasoning":
            return True
        
        question = context.get("question", "")
        return any(kw in question for kw in self.RELATION_KEYWORDS)
    
    def analyze(
        self,
        question: str,
        known_facts: Dict[str, Any],
        entity_facts: Dict[str, Dict],
        relation_facts: Dict[str, Dict]
    ) -> StrategyResult:
        """关系分析"""
        reasoning_chain = []
        evidence = []
        
        # 检查是否有足够的实体信息
        if len(entity_facts) < 2:
            return StrategyResult(
                success=False,
                reasoning_chain=["需要找到至少两个相关实体"],
                next_steps=["搜索问题中提到的实体"]
            )
        
        reasoning_chain.append(f"找到 {len(entity_facts)} 个相关实体")
        
        # 检查关系信息
        if not relation_facts:
            return StrategyResult(
                success=False,
                reasoning_chain=reasoning_chain + ["尚未找到实体之间的关系"],
                next_steps=["查询实体之间的直接关系", "尝试多跳路径搜索"]
            )
        
        reasoning_chain.append(f"找到 {len(relation_facts)} 个相关关系")
        
        # 整合关系描述
        relation_descriptions = []
        for rid, facts in relation_facts.items():
            desc = facts.get("content", "")
            e1 = facts.get("entity1_name", "")
            e2 = facts.get("entity2_name", "")
            relation_descriptions.append(f"{e1} 与 {e2}: {desc}")
            evidence.append(f"关系证据: {desc[:100]}")
        
        # 尝试总结关系
        if relation_descriptions:
            conclusion = "根据记忆库中的信息，" + "；".join(relation_descriptions[:5])
            if len(relation_descriptions) > 5:
                conclusion += f"...等共 {len(relation_descriptions)} 条关系记录"
            
            return StrategyResult(
                success=True,
                conclusion=conclusion,
                confidence=0.8,
                reasoning_chain=reasoning_chain,
                evidence=evidence
            )
        
        return StrategyResult(
            success=False,
            reasoning_chain=reasoning_chain,
            next_steps=["需要更多关系信息"]
        )
    
    def get_next_queries(
        self,
        question: str,
        known_facts: Dict[str, Any],
        entity_facts: Dict[str, Dict],
        relation_facts: Dict[str, Dict],
        tried_queries: List[Dict]
    ) -> List[Dict[str, Any]]:
        """获取关系推理需要的下一步查询"""
        queries = []
        
        # 如果实体不足，先搜索实体
        if len(entity_facts) < 2:
            queries.append({
                "strategy": "search_entities",
                "reason": "需要找到问题中提到的实体"
            })
        
        # 如果有实体但没有关系，查询关系
        if entity_facts and not relation_facts:
            entity_ids = list(entity_facts.keys())
            if len(entity_ids) >= 2:
                queries.append({
                    "strategy": "get_entity_relations",
                    "entity1_id": entity_ids[0],
                    "entity2_id": entity_ids[1],
                    "reason": "查询两个实体之间的直接关系"
                })
                queries.append({
                    "strategy": "get_relation_paths",
                    "entity1_id": entity_ids[0],
                    "entity2_id": entity_ids[1],
                    "reason": "查询两个实体之间的间接路径"
                })
        
        return queries


class DirectQueryStrategy(ReasoningStrategy):
    """
    直接查询策略
    
    处理可以直接回答的简单问题
    """
    
    def can_handle(self, question_type: str, context: Dict[str, Any]) -> bool:
        return question_type == "direct"
    
    def analyze(
        self,
        question: str,
        known_facts: Dict[str, Any],
        entity_facts: Dict[str, Dict],
        relation_facts: Dict[str, Dict]
    ) -> StrategyResult:
        """直接查询分析"""
        # 对于直接查询，如果找到了实体信息就可以回答
        if entity_facts:
            # 取第一个实体的信息作为答案
            for eid, facts in entity_facts.items():
                name = facts.get("name", "")
                content = facts.get("content", "")
                if content:
                    return StrategyResult(
                        success=True,
                        conclusion=f"{name}: {content}",
                        confidence=0.9,
                        reasoning_chain=["找到相关实体信息"],
                        evidence=[f"实体内容: {content[:200]}"]
                    )
        
        return StrategyResult(
            success=False,
            reasoning_chain=["未找到相关实体信息"],
            next_steps=["搜索问题中提到的实体"]
        )
    
    def get_next_queries(
        self,
        question: str,
        known_facts: Dict[str, Any],
        entity_facts: Dict[str, Dict],
        relation_facts: Dict[str, Dict],
        tried_queries: List[Dict]
    ) -> List[Dict[str, Any]]:
        """获取直接查询需要的下一步"""
        if not entity_facts:
            return [{
                "strategy": "search_entity",
                "reason": "搜索问题中提到的实体"
            }]
        return []
