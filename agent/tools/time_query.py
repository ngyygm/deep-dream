"""
时间查询工具

按时间点查询实体/关系状态
"""
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseTool, ToolDefinition, ToolParameter


class TimeQueryTool(BaseTool):
    """时间查询工具"""
    
    @classmethod
    def get_definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name="query_by_time",
            description="查询指定时间点的实体状态或所有实体。可以获取某个时间点的知识图谱快照，或查询特定实体在某个时间点的版本。",
            parameters=[
                ToolParameter(
                    name="time_point",
                    type="string",
                    description="时间点（ISO 格式，如 '2025-01-30T10:00:00'）",
                    required=True
                ),
                ToolParameter(
                    name="entity_id",
                    type="string",
                    description="实体 ID（可选，如果提供则只查询该实体在指定时间点的状态）",
                    required=False
                ),
                ToolParameter(
                    name="include_relations",
                    type="boolean",
                    description="是否包含该时间点的关系（仅当指定了 entity_id 时有效）",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="返回的最大实体数量（仅当未指定 entity_id 时有效），默认 50",
                    required=False,
                    default=50
                )
            ]
        )
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行时间查询
        
        Args:
            time_point: 时间点（ISO 格式）
            entity_id: 实体 ID（可选）
            include_relations: 是否包含关系
            limit: 最大结果数量
            
        Returns:
            {
                "success": bool,
                "time_point": str,
                "entity": Dict (if entity_id provided),
                "entities": List[Dict] (if entity_id not provided),
                "relations": List[Dict] (if include_relations),
                "message": str
            }
        """
        time_point_str = kwargs.get("time_point", "")
        entity_id = kwargs.get("entity_id")
        include_relations = kwargs.get("include_relations", False)
        limit = kwargs.get("limit", 50)
        
        if not time_point_str:
            return {
                "success": False,
                "message": "时间点不能为空"
            }
        
        try:
            # 解析时间
            time_point = datetime.fromisoformat(time_point_str)
        except ValueError:
            return {
                "success": False,
                "message": f"时间格式无效: {time_point_str}，请使用 ISO 格式（如 '2025-01-30T10:00:00'）"
            }
        
        try:
            result = {
                "success": True,
                "time_point": time_point_str
            }
            
            if entity_id:
                # 查询特定实体在指定时间点的版本
                entity = self.storage_manager.get_entity_version_at_time(entity_id, time_point)
                
                if entity:
                    result["entity"] = self._format_entity(entity)
                    result["message"] = f"找到实体 {entity.name} 在 {time_point_str} 的版本"
                    
                    # 如果需要包含关系
                    if include_relations:
                        relations = self.storage_manager.get_entity_relations_by_entity_id(
                            entity_id=entity_id,
                            time_point=time_point
                        )
                        formatted_relations = []
                        for rel in relations:
                            rel_dict = self._format_relation(rel)
                            entity1 = self.storage_manager.get_entity_by_absolute_id(rel.entity1_absolute_id)
                            entity2 = self.storage_manager.get_entity_by_absolute_id(rel.entity2_absolute_id)
                            rel_dict["entity1_name"] = entity1.name if entity1 else "Unknown"
                            rel_dict["entity2_name"] = entity2.name if entity2 else "Unknown"
                            formatted_relations.append(rel_dict)
                        result["relations"] = formatted_relations
                else:
                    result["entity"] = None
                    result["message"] = f"在 {time_point_str} 之前未找到实体 {entity_id}"
            else:
                # 获取指定时间点的所有实体
                entities = self.storage_manager.get_all_entities_before_time(time_point, limit=limit)
                result["entities"] = [self._format_entity(e) for e in entities]
                result["count"] = len(entities)
                result["message"] = f"找到 {len(entities)} 个在 {time_point_str} 之前存在的实体"
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "message": f"时间查询失败: {str(e)}"
            }
