"""
搜索关系工具

按内容搜索关系边
"""
from typing import Dict, Any, List, Optional

from .base import BaseTool, ToolDefinition, ToolParameter


class SearchRelationsTool(BaseTool):
    """搜索关系工具"""
    
    @classmethod
    def get_definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name="search_relations",
            description="""按内容搜索记忆库中的关系边。可以找到描述特定事件、行为或关联的关系。

返回字段说明：
- relation_id: 关系唯一ID，用于查询版本历史
- content: 关系描述
- physical_time: 该关系记录的时间点（用于时序推理）
- entity1_name, entity2_name: 关联的两个实体名称""",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="搜索查询文本（关系描述或关键词）",
                    required=True
                ),
                ToolParameter(
                    name="threshold",
                    type="number",
                    description="相似度阈值（0-1），默认 0.3",
                    required=False,
                    default=0.3
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
        执行关系搜索
        
        Args:
            query: 搜索查询文本
            threshold: 相似度阈值
            max_results: 最大结果数量
            
        Returns:
            {
                "success": bool,
                "relations": List[Dict],
                "count": int,
                "message": str
            }
        """
        query = kwargs.get("query", "")
        threshold = kwargs.get("threshold", 0.3)
        max_results = kwargs.get("max_results", 10)
        
        if not query:
            return {
                "success": False,
                "relations": [],
                "count": 0,
                "message": "搜索查询不能为空"
            }
        
        try:
            # 调用存储管理器的关系搜索方法
            relations = self.storage_manager.search_relations_by_similarity(
                query_text=query,
                threshold=threshold,
                max_results=max_results
            )
            
            # 格式化结果，并添加关联实体的名称
            formatted_relations = []
            for rel in relations:
                rel_dict = self._format_relation(rel)
                
                # 尝试获取关联实体的名称
                entity1 = self.storage_manager.get_entity_by_absolute_id(rel.entity1_absolute_id)
                entity2 = self.storage_manager.get_entity_by_absolute_id(rel.entity2_absolute_id)
                
                rel_dict["entity1_name"] = entity1.name if entity1 else "Unknown"
                rel_dict["entity2_name"] = entity2.name if entity2 else "Unknown"
                
                formatted_relations.append(rel_dict)
            
            return {
                "success": True,
                "relations": formatted_relations,
                "count": len(formatted_relations),
                "message": f"找到 {len(formatted_relations)} 个匹配的关系"
            }
            
        except Exception as e:
            return {
                "success": False,
                "relations": [],
                "count": 0,
                "message": f"搜索关系失败: {str(e)}"
            }
