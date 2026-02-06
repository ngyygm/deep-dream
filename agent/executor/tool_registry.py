"""
工具注册表
"""
from typing import Dict, Type, Optional, List

from ..tools.base import BaseTool, ToolDefinition


class ToolRegistry:
    """工具注册表 - 管理所有可用工具"""
    
    def __init__(self):
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._instances: Dict[str, BaseTool] = {}
    
    def register(self, name: str, tool_class: Type[BaseTool]):
        """注册工具类"""
        self._tools[name] = tool_class
    
    def register_instance(self, name: str, tool_instance: BaseTool):
        """注册工具实例"""
        self._instances[name] = tool_instance
    
    def get_tool_class(self, name: str) -> Optional[Type[BaseTool]]:
        """获取工具类"""
        return self._tools.get(name)
    
    def get_tool_instance(self, name: str) -> Optional[BaseTool]:
        """获取工具实例"""
        return self._instances.get(name)
    
    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义"""
        if name in self._instances:
            return self._instances[name].get_definition()
        elif name in self._tools:
            return self._tools[name].get_definition()
        return None
    
    def get_all_definitions(self) -> Dict[str, ToolDefinition]:
        """获取所有工具定义"""
        definitions = {}
        
        # 从实例获取
        for name, instance in self._instances.items():
            definitions[name] = instance.get_definition()
        
        # 从类获取（如果实例中没有）
        for name, tool_class in self._tools.items():
            if name not in definitions:
                definitions[name] = tool_class.get_definition()
        
        return definitions
    
    def get_tool_names(self) -> List[str]:
        """获取所有工具名称"""
        names = set(self._tools.keys())
        names.update(self._instances.keys())
        return list(names)
    
    def has_tool(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self._tools or name in self._instances
    
    def clear(self):
        """清空注册表"""
        self._tools.clear()
        self._instances.clear()


def create_default_registry(storage_manager) -> ToolRegistry:
    """
    创建默认的工具注册表
    
    Args:
        storage_manager: StorageManager 实例
        
    Returns:
        包含所有默认工具的注册表
    """
    from ..tools import (
        SearchEntityTool,
        GetRelationsTool,
        GetRelationPathsTool,
        GetVersionsTool,
        GetMemoryCacheTool,
        SearchRelationsTool,
        TimeQueryTool
    )
    
    registry = ToolRegistry()
    
    # 注册工具实例
    registry.register_instance("search_entity", SearchEntityTool(storage_manager))
    registry.register_instance("get_entity_relations", GetRelationsTool(storage_manager))
    registry.register_instance("get_relation_paths", GetRelationPathsTool(storage_manager))
    registry.register_instance("get_entity_versions", GetVersionsTool(storage_manager))
    registry.register_instance("get_memory_cache", GetMemoryCacheTool(storage_manager))
    registry.register_instance("search_relations", SearchRelationsTool(storage_manager))
    registry.register_instance("query_by_time", TimeQueryTool(storage_manager))
    
    return registry
