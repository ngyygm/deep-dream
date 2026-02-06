"""
获取关系工具

获取实体间的关系边
"""
from typing import Dict, Any, List, Optional

from .base import BaseTool, ToolDefinition, ToolParameter


class GetRelationsTool(BaseTool):
    """获取实体关系工具"""
    
    @classmethod
    def get_definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name="get_entity_relations",
            description="""获取指定实体的所有关系边，或获取两个实体之间的关系。

返回字段说明：
- relation_id: 关系唯一ID，用于查询版本历史
- content: 关系描述
- physical_time: 该关系记录的时间点（用于时序推理，判断"第几次"等问题）
- entity1_name, entity2_name: 关联的两个实体名称
- memory_cache_id: 关联的记忆来源ID""",
            parameters=[
                ToolParameter(
                    name="entity_id",
                    type="string",
                    description="实体的 entity_id（不是 absolute_id）",
                    required=True
                ),
                ToolParameter(
                    name="entity2_id",
                    type="string",
                    description="第二个实体的 entity_id（可选，如果提供则只返回这两个实体之间的关系）",
                    required=False
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="返回的最大关系数量，默认 20",
                    required=False,
                    default=20
                )
            ]
        )
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行获取关系
        
        Args:
            entity_id: 实体 ID
            entity2_id: 第二个实体 ID（可选）
            limit: 最大结果数量
            
        Returns:
            {
                "success": bool,
                "relations": List[Dict],
                "count": int,
                "message": str
            }
        """
        entity_id = kwargs.get("entity_id", "")
        entity2_id = kwargs.get("entity2_id")
        limit = kwargs.get("limit", 20)
        
        if not entity_id:
            return {
                "success": False,
                "relations": [],
                "count": 0,
                "message": "entity_id 不能为空"
            }
        
        try:
            if entity2_id:
                # 获取两个实体之间的关系
                relations = self.storage_manager.get_relations_by_entities(
                    from_entity_id=entity_id,
                    to_entity_id=entity2_id
                )
            else:
                # 获取单个实体的所有关系
                relations = self.storage_manager.get_entity_relations_by_entity_id(
                    entity_id=entity_id,
                    limit=limit
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
                "message": f"找到 {len(formatted_relations)} 个关系"
            }
            
        except Exception as e:
            return {
                "success": False,
                "relations": [],
                "count": 0,
                "message": f"获取关系失败: {str(e)}"
            }
