"""
获取版本历史工具

获取实体或关系的历史版本
"""
from typing import Dict, Any, List, Optional

from .base import BaseTool, ToolDefinition, ToolParameter


class GetVersionsTool(BaseTool):
    """获取版本历史工具"""
    
    @classmethod
    def get_definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name="get_entity_versions",
            description="""获取实体或关系的所有历史版本。可用于追踪实体/关系随时间的变化，找出信息最早出现的时间点。

**时序推理关键工具**：当需要判断"第几次"、"最早"、"最晚"等时序问题时，使用此工具获取所有版本的 physical_time。

返回字段说明：
- versions: 所有版本列表，每个版本包含 physical_time
- earliest_time: 最早版本的时间（第一次出现）
- latest_time: 最新版本的时间（最后更新）
- 每个版本的 physical_time 表示该版本创建的时间点""",
            parameters=[
                ToolParameter(
                    name="target_type",
                    type="string",
                    description="目标类型：entity（实体）或 relation（关系）",
                    required=True,
                    enum=["entity", "relation"]
                ),
                ToolParameter(
                    name="target_id",
                    type="string",
                    description="目标的 ID（entity_id 或 relation_id）",
                    required=True
                ),
                ToolParameter(
                    name="include_cache_text",
                    type="boolean",
                    description="是否包含每个版本对应的原始文本（memory_cache 中的 text）",
                    required=False,
                    default=False
                )
            ]
        )
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行获取版本历史
        
        Args:
            target_type: 目标类型（entity 或 relation）
            target_id: 目标 ID
            include_cache_text: 是否包含缓存文本
            
        Returns:
            {
                "success": bool,
                "versions": List[Dict],
                "count": int,
                "earliest_time": str,
                "latest_time": str,
                "message": str
            }
        """
        target_type = kwargs.get("target_type", "")
        target_id = kwargs.get("target_id", "")
        include_cache_text = kwargs.get("include_cache_text", False)
        
        if not target_type or not target_id:
            return {
                "success": False,
                "versions": [],
                "count": 0,
                "message": "target_type 和 target_id 不能为空"
            }
        
        try:
            if target_type == "entity":
                versions = self.storage_manager.get_entity_versions(target_id)
                formatted_versions = [self._format_entity(v) for v in versions]
            elif target_type == "relation":
                versions = self.storage_manager.get_relation_versions(target_id)
                formatted_versions = [self._format_relation(v) for v in versions]
            else:
                return {
                    "success": False,
                    "versions": [],
                    "count": 0,
                    "message": f"不支持的目标类型: {target_type}"
                }
            
            # 如果需要包含缓存文本
            if include_cache_text:
                for v in formatted_versions:
                    cache_id = v.get("memory_cache_id")
                    if cache_id:
                        cache_text = self.storage_manager.get_memory_cache_text(cache_id)
                        v["cache_text"] = cache_text[:500] if cache_text else None  # 限制长度
            
            # 计算时间范围
            earliest_time = None
            latest_time = None
            if formatted_versions:
                times = [v["physical_time"] for v in formatted_versions if v.get("physical_time")]
                if times:
                    earliest_time = min(times)
                    latest_time = max(times)
            
            return {
                "success": True,
                "versions": formatted_versions,
                "count": len(formatted_versions),
                "earliest_time": earliest_time,
                "latest_time": latest_time,
                "message": f"找到 {len(formatted_versions)} 个版本"
            }
            
        except Exception as e:
            return {
                "success": False,
                "versions": [],
                "count": 0,
                "message": f"获取版本历史失败: {str(e)}"
            }
