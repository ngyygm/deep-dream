"""
获取记忆缓存工具

获取记忆缓存内容（上下文摘要）
"""
from typing import Dict, Any, List, Optional

from .base import BaseTool, ToolDefinition, ToolParameter


class GetMemoryCacheTool(BaseTool):
    """获取记忆缓存工具"""
    
    @classmethod
    def get_definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name="get_memory_cache",
            description="""获取记忆缓存的内容。记忆缓存包含了处理文本时的上下文摘要、状态信息等。

返回字段说明：
- id: 缓存唯一ID
- content: Markdown格式的完整描述（上下文摘要）
- physical_time: 该缓存创建的时间点
- activity_type: 活动类型（如"阅读小说"、"处理文档"等）

用途：通过 memory_cache_id 关联可以追溯实体/关系的原始上下文来源。""",
            parameters=[
                ToolParameter(
                    name="cache_id",
                    type="string",
                    description="记忆缓存的 ID（可选，如果不提供则获取最新的缓存）",
                    required=False
                ),
                ToolParameter(
                    name="include_text",
                    type="boolean",
                    description="是否包含对应的原始文本内容",
                    required=False,
                    default=False
                )
            ]
        )
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行获取记忆缓存
        
        Args:
            cache_id: 缓存 ID（可选）
            include_text: 是否包含原始文本
            
        Returns:
            {
                "success": bool,
                "cache": Dict,
                "text": str (if include_text),
                "message": str
            }
        """
        cache_id = kwargs.get("cache_id")
        include_text = kwargs.get("include_text", False)
        
        try:
            if cache_id:
                # 获取指定的缓存
                cache = self.storage_manager.load_memory_cache(cache_id)
            else:
                # 获取最新的缓存
                cache = self.storage_manager.get_latest_memory_cache()
            
            if not cache:
                return {
                    "success": False,
                    "cache": None,
                    "message": "未找到记忆缓存"
                }
            
            result = {
                "success": True,
                "cache": self._format_memory_cache(cache),
                "message": "成功获取记忆缓存"
            }
            
            # 如果需要包含原始文本
            if include_text:
                text = self.storage_manager.get_memory_cache_text(cache.id)
                result["text"] = text[:1000] if text else None  # 限制长度
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "cache": None,
                "message": f"获取记忆缓存失败: {str(e)}"
            }
