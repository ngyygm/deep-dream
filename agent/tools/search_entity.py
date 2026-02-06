"""
搜索实体工具

按名称/内容搜索实体
"""
from typing import Dict, Any, List, Optional

from .base import BaseTool, ToolDefinition, ToolParameter


class SearchEntityTool(BaseTool):
    """搜索实体工具"""
    
    @classmethod
    def get_definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name="search_entity",
            description="""按名称或内容搜索记忆库中的实体。返回匹配的实体列表。

返回字段说明：
- entity_id: 实体唯一ID，用于后续调用其他工具
- name: 实体名称
- content: 实体描述
- physical_time: 该记录的时间点（用于时序推理）
- memory_cache_id: 关联的记忆来源ID""",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="搜索查询文本（实体名称或相关描述）",
                    required=True
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="可选的内容描述，用于更精确的搜索",
                    required=False
                ),
                ToolParameter(
                    name="search_mode",
                    type="string",
                    description="搜索模式",
                    required=False,
                    enum=["name_only", "content_only", "name_and_content"],
                    default="name_and_content"
                ),
                ToolParameter(
                    name="similarity_method",
                    type="string",
                    description="相似度计算方法",
                    required=False,
                    enum=["embedding", "text", "jaccard"],
                    default="embedding"
                ),
                ToolParameter(
                    name="threshold",
                    type="number",
                    description="相似度阈值（0-1），默认 0.5",
                    required=False,
                    default=0.5
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="返回的最大结果数量，默认 10",
                    required=False,
                    default=10
                )
            ]
        )
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行实体搜索
        
        Args:
            query: 搜索查询文本
            content: 可选的内容描述
            search_mode: 搜索模式
            similarity_method: 相似度计算方法
            threshold: 相似度阈值
            max_results: 最大结果数量
            
        Returns:
            {
                "success": bool,
                "entities": List[Dict],
                "count": int,
                "message": str
            }
        """
        query = kwargs.get("query", "")
        content = kwargs.get("content")
        search_mode = kwargs.get("search_mode", "name_and_content")
        similarity_method = kwargs.get("similarity_method", "embedding")
        threshold = kwargs.get("threshold", 0.5)
        max_results = kwargs.get("max_results", 10)
        
        if not query:
            return {
                "success": False,
                "entities": [],
                "count": 0,
                "message": "搜索查询不能为空"
            }
        
        try:
            # 调用存储管理器的搜索方法
            entities = self.storage_manager.search_entities_by_similarity(
                query_name=query,
                query_content=content,
                threshold=threshold,
                max_results=max_results,
                text_mode=search_mode,
                similarity_method=similarity_method
            )
            
            # 格式化结果
            formatted_entities = [self._format_entity(e) for e in entities]
            
            return {
                "success": True,
                "entities": formatted_entities,
                "count": len(formatted_entities),
                "message": f"找到 {len(formatted_entities)} 个匹配的实体"
            }
            
        except Exception as e:
            return {
                "success": False,
                "entities": [],
                "count": 0,
                "message": f"搜索失败: {str(e)}"
            }
