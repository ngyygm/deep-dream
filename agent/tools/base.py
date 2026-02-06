"""
工具基类定义
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str  # "string", "integer", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Any = None
    items: Optional[Dict[str, Any]] = None  # 用于 array 类型


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """转换为 OpenAI 函数调用格式"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.items:
                prop["items"] = param.items
            if param.default is not None:
                prop["default"] = param.default
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self, storage_manager):
        """
        初始化工具
        
        Args:
            storage_manager: StorageManager 实例，用于访问记忆库
        """
        self.storage_manager = storage_manager
    
    @classmethod
    @abstractmethod
    def get_definition(cls) -> ToolDefinition:
        """
        获取工具定义
        
        Returns:
            工具定义对象
        """
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行工具
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            执行结果字典
        """
        pass
    
    async def aexecute(self, **kwargs) -> Dict[str, Any]:
        """
        异步执行工具（默认实现调用同步方法）
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            执行结果字典
        """
        return self.execute(**kwargs)
    
    def validate_parameters(self, **kwargs) -> bool:
        """
        验证参数
        
        Args:
            **kwargs: 传入的参数
            
        Returns:
            是否有效
            
        Raises:
            ValueError: 如果参数无效
        """
        definition = self.get_definition()
        
        for param in definition.parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(f"Missing required parameter: {param.name}")
        
        return True
    
    def _format_entity(self, entity) -> Dict[str, Any]:
        """格式化实体为字典"""
        return {
            "id": entity.id,
            "entity_id": entity.entity_id,
            "name": entity.name,
            "content": entity.content,
            "physical_time": entity.physical_time.isoformat() if entity.physical_time else None,
            "memory_cache_id": entity.memory_cache_id
        }
    
    def _format_relation(self, relation) -> Dict[str, Any]:
        """格式化关系为字典"""
        return {
            "id": relation.id,
            "relation_id": relation.relation_id,
            "entity1_absolute_id": relation.entity1_absolute_id,
            "entity2_absolute_id": relation.entity2_absolute_id,
            "content": relation.content,
            "physical_time": relation.physical_time.isoformat() if relation.physical_time else None,
            "memory_cache_id": relation.memory_cache_id
        }
    
    def _format_memory_cache(self, cache) -> Dict[str, Any]:
        """格式化记忆缓存为字典"""
        return {
            "id": cache.id,
            "content": cache.content,
            "physical_time": cache.physical_time.isoformat() if cache.physical_time else None,
            "activity_type": cache.activity_type
        }
