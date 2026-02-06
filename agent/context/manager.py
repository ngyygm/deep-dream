"""
上下文管理器

管理查询过程中累积的记忆和状态
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..models import RetrievedMemory, ToolResult
from ..llm.base import BaseLLMClient


@dataclass
class QueryContext:
    """查询上下文"""
    question: str
    collected_memories: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    relevant_entities: Dict[str, Dict] = field(default_factory=dict)  # entity_id -> entity_info
    relevant_relations: Dict[str, Dict] = field(default_factory=dict)  # relation_id -> relation_info
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    start_time: datetime = field(default_factory=datetime.now)


class ContextManager:
    """
    上下文管理器
    
    负责：
    - 管理累积的查询结果
    - 选择性保留有价值的记忆
    - 避免重复查询
    """
    
    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        """
        初始化上下文管理器
        
        Args:
            llm_client: LLM 客户端（用于智能选择）
        """
        self.llm_client = llm_client
        self.context: Optional[QueryContext] = None
    
    def start_query(self, question: str) -> QueryContext:
        """开始新的查询"""
        self.context = QueryContext(question=question)
        return self.context
    
    def add_tool_result(self, tool_name: str, result: ToolResult):
        """添加工具执行结果"""
        if self.context is None:
            raise RuntimeError("No active query context")
        
        self.context.tool_results.append(result)
        
        # 提取记忆信息
        if result.is_success and result.data:
            self.context.collected_memories.append({
                "tool_name": tool_name,
                "result": result.data,
                "timestamp": datetime.now().isoformat()
            })
            
            # 提取实体
            if "entities" in result.data:
                for entity in result.data["entities"]:
                    entity_id = entity.get("entity_id")
                    if entity_id and entity_id not in self.context.relevant_entities:
                        self.context.relevant_entities[entity_id] = entity
            
            # 提取单个实体
            if "entity" in result.data and result.data["entity"]:
                entity = result.data["entity"]
                entity_id = entity.get("entity_id")
                if entity_id and entity_id not in self.context.relevant_entities:
                    self.context.relevant_entities[entity_id] = entity
            
            # 提取关系
            if "relations" in result.data:
                for relation in result.data["relations"]:
                    relation_id = relation.get("relation_id")
                    if relation_id and relation_id not in self.context.relevant_relations:
                        self.context.relevant_relations[relation_id] = relation
    
    def add_reasoning_step(self, step_type: str, content: str, data: Any = None):
        """添加推理步骤"""
        if self.context is None:
            raise RuntimeError("No active query context")
        
        self.context.reasoning_trace.append({
            "type": step_type,
            "content": content,
            "data": data,
            "iteration": self.context.iteration,
            "timestamp": datetime.now().isoformat()
        })
    
    def increment_iteration(self):
        """增加迭代计数"""
        if self.context:
            self.context.iteration += 1
    
    def get_collected_info(self) -> List[Dict[str, Any]]:
        """获取已收集的信息"""
        if self.context is None:
            return []
        return self.context.collected_memories
    
    def get_relevant_entities(self) -> List[Dict[str, Any]]:
        """获取相关实体列表"""
        if self.context is None:
            return []
        return list(self.context.relevant_entities.values())
    
    def get_relevant_relations(self) -> List[Dict[str, Any]]:
        """获取相关关系列表"""
        if self.context is None:
            return []
        return list(self.context.relevant_relations.values())
    
    def get_reasoning_trace(self) -> List[Dict[str, Any]]:
        """获取推理追踪"""
        if self.context is None:
            return []
        return self.context.reasoning_trace
    
    def prune_memories(self, memories_to_keep: List[str]):
        """
        修剪记忆（移除不需要的记忆）
        
        Args:
            memories_to_keep: 要保留的记忆标识符列表
        """
        # 简单实现：保留所有记忆
        # 更复杂的实现可以基于 LLM 的建议进行选择
        pass
    
    def build_retrieved_memories(self) -> List[RetrievedMemory]:
        """构建检索到的记忆列表"""
        memories = []
        
        # 从实体构建记忆
        for entity_id, entity in self.context.relevant_entities.items():
            memories.append(RetrievedMemory(
                memory_type="entity",
                content=f"{entity.get('name', 'Unknown')}: {entity.get('content', '')}",
                source_id=entity_id,
                physical_time=datetime.fromisoformat(entity["physical_time"]) 
                    if entity.get("physical_time") else None,
                metadata=entity
            ))
        
        # 从关系构建记忆
        for relation_id, relation in self.context.relevant_relations.items():
            e1_name = relation.get("entity1_name", "?")
            e2_name = relation.get("entity2_name", "?")
            memories.append(RetrievedMemory(
                memory_type="relation",
                content=f"[{e1_name}] -- [{e2_name}]: {relation.get('content', '')}",
                source_id=relation_id,
                physical_time=datetime.fromisoformat(relation["physical_time"])
                    if relation.get("physical_time") else None,
                metadata=relation
            ))
        
        return memories
    
    def has_entity(self, entity_name: str) -> bool:
        """检查是否已有指定名称的实体"""
        for entity in self.context.relevant_entities.values():
            if entity.get("name", "").lower() == entity_name.lower():
                return True
        return False
    
    def get_entity_by_name(self, entity_name: str) -> Optional[Dict]:
        """按名称获取实体"""
        for entity in self.context.relevant_entities.values():
            if entity.get("name", "").lower() == entity_name.lower():
                return entity
        return None
    
    def get_entity_id_by_name(self, entity_name: str) -> Optional[str]:
        """按名称获取实体 ID"""
        entity = self.get_entity_by_name(entity_name)
        return entity.get("entity_id") if entity else None
