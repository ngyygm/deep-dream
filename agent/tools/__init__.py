"""
Agent 工具层 - 7 个细粒度查询工具
"""
from .base import BaseTool, ToolDefinition
from .search_entity import SearchEntityTool
from .get_relations import GetRelationsTool
from .get_relation_paths import GetRelationPathsTool
from .get_versions import GetVersionsTool
from .get_memory_cache import GetMemoryCacheTool
from .search_relations import SearchRelationsTool
from .time_query import TimeQueryTool

__all__ = [
    "BaseTool",
    "ToolDefinition",
    "SearchEntityTool",
    "GetRelationsTool",
    "GetRelationPathsTool",
    "GetVersionsTool",
    "GetMemoryCacheTool",
    "SearchRelationsTool",
    "TimeQueryTool"
]

# 工具注册表
TOOL_REGISTRY = {
    "search_entity": SearchEntityTool,
    "get_entity_relations": GetRelationsTool,
    "get_relation_paths": GetRelationPathsTool,
    "get_entity_versions": GetVersionsTool,
    "get_memory_cache": GetMemoryCacheTool,
    "search_relations": SearchRelationsTool,
    "query_by_time": TimeQueryTool
}


def get_all_tools():
    """获取所有工具的定义"""
    return {name: cls.get_definition() for name, cls in TOOL_REGISTRY.items()}


def create_tool(tool_name: str, storage_manager) -> BaseTool:
    """创建工具实例"""
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    return TOOL_REGISTRY[tool_name](storage_manager)
