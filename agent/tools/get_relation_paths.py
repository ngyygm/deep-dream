"""
多跳关系路径检索工具

查找两个实体之间的多跳关系路径（通过中间实体连接）
"""
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import deque

from .base import BaseTool, ToolDefinition, ToolParameter


class GetRelationPathsTool(BaseTool):
    """多跳关系路径检索工具"""
    
    @classmethod
    def get_definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name="get_relation_paths",
            description="""查找两个实体之间的多跳关系路径。当两个实体之间没有直接关系时，可以通过此工具找到它们通过中间实体连接的间接路径。

返回字段说明：
- paths: 所有找到的路径列表
  - nodes: 路径经过的实体列表（包含 entity_id, name, content）
  - edges: 路径经过的关系列表（包含 relation_id, content, physical_time）
  - hop_count: 跳数
  - path_description: 路径的文字描述
- shortest_path_length: 最短路径的跳数
- 每条边的 physical_time 可用于时序推理""",
            parameters=[
                ToolParameter(
                    name="entity1_id",
                    type="string",
                    description="起始实体的 entity_id",
                    required=True
                ),
                ToolParameter(
                    name="entity2_id",
                    type="string",
                    description="目标实体的 entity_id",
                    required=True
                ),
                ToolParameter(
                    name="max_hops",
                    type="integer",
                    description="最大跳数限制（默认 3，最大 5）",
                    required=False,
                    default=3
                ),
                ToolParameter(
                    name="max_paths",
                    type="integer",
                    description="返回的最大路径数量（默认 5）",
                    required=False,
                    default=5
                ),
                ToolParameter(
                    name="include_relation_content",
                    type="boolean",
                    description="是否包含关系边的完整内容（默认 True）",
                    required=False,
                    default=True
                )
            ]
        )
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行多跳关系路径检索
        
        使用 BFS 算法查找两个实体之间的所有路径
        
        Args:
            entity1_id: 起始实体 ID
            entity2_id: 目标实体 ID
            max_hops: 最大跳数
            max_paths: 最大路径数量
            include_relation_content: 是否包含关系内容
            
        Returns:
            {
                "success": bool,
                "paths": List[Dict],  # 每条路径包含 nodes 和 edges
                "path_count": int,
                "shortest_path_length": int,
                "message": str
            }
        """
        entity1_id = kwargs.get("entity1_id", "")
        entity2_id = kwargs.get("entity2_id", "")
        max_hops = min(kwargs.get("max_hops", 3), 5)  # 限制最大5跳
        max_paths = kwargs.get("max_paths", 5)
        include_relation_content = kwargs.get("include_relation_content", True)
        
        if not entity1_id or not entity2_id:
            return {
                "success": False,
                "paths": [],
                "path_count": 0,
                "message": "entity1_id 和 entity2_id 不能为空"
            }
        
        if entity1_id == entity2_id:
            return {
                "success": False,
                "paths": [],
                "path_count": 0,
                "message": "起始实体和目标实体不能相同"
            }
        
        try:
            # 验证实体存在
            entity1 = self.storage_manager.get_entity_by_id(entity1_id)
            entity2 = self.storage_manager.get_entity_by_id(entity2_id)
            
            if not entity1:
                return {
                    "success": False,
                    "paths": [],
                    "path_count": 0,
                    "message": f"起始实体不存在: {entity1_id}"
                }
            
            if not entity2:
                return {
                    "success": False,
                    "paths": [],
                    "path_count": 0,
                    "message": f"目标实体不存在: {entity2_id}"
                }
            
            # BFS 查找路径
            paths = self._find_paths_bfs(
                entity1_id, entity2_id, 
                max_hops, max_paths,
                include_relation_content
            )
            
            if not paths:
                return {
                    "success": True,
                    "paths": [],
                    "path_count": 0,
                    "shortest_path_length": -1,
                    "message": f"在 {max_hops} 跳内未找到 {entity1.name} 和 {entity2.name} 之间的路径"
                }
            
            shortest_length = min(len(p["edges"]) for p in paths)
            
            return {
                "success": True,
                "paths": paths,
                "path_count": len(paths),
                "shortest_path_length": shortest_length,
                "entity1_name": entity1.name,
                "entity2_name": entity2.name,
                "message": f"找到 {len(paths)} 条路径，最短路径长度为 {shortest_length} 跳"
            }
            
        except Exception as e:
            return {
                "success": False,
                "paths": [],
                "path_count": 0,
                "message": f"路径检索失败: {str(e)}"
            }
    
    def _find_paths_bfs(
        self,
        start_id: str,
        end_id: str,
        max_hops: int,
        max_paths: int,
        include_content: bool
    ) -> List[Dict[str, Any]]:
        """
        使用 BFS 查找所有路径
        
        Returns:
            路径列表，每条路径包含:
            - nodes: 经过的实体列表
            - edges: 经过的关系边列表
            - path_description: 路径的文字描述
        """
        paths = []
        
        # BFS 队列：(当前实体ID, 路径上的实体ID列表, 路径上的关系列表)
        queue = deque([(start_id, [start_id], [])])
        
        # 记录已访问的路径状态，避免环路
        visited_states: Set[Tuple[str, ...]] = set()
        visited_states.add((start_id,))
        
        while queue and len(paths) < max_paths:
            current_id, path_nodes, path_edges = queue.popleft()
            
            # 检查跳数限制
            if len(path_edges) >= max_hops:
                continue
            
            # 获取当前实体的所有关系
            relations = self.storage_manager.get_entity_relations_by_entity_id(
                entity_id=current_id,
                limit=50  # 限制每个节点的邻居数量
            )
            
            for relation in relations:
                # 找到关系的另一端实体
                other_entity_id = self._get_other_entity_id(relation, current_id)
                
                if not other_entity_id:
                    continue
                
                # 检查是否形成环路（除了目标节点）
                new_path_state = tuple(path_nodes + [other_entity_id])
                if other_entity_id != end_id and new_path_state in visited_states:
                    continue
                
                # 如果到达目标
                if other_entity_id == end_id:
                    path = self._build_path_result(
                        path_nodes + [end_id],
                        path_edges + [relation],
                        include_content
                    )
                    paths.append(path)
                    
                    if len(paths) >= max_paths:
                        break
                else:
                    # 继续搜索
                    visited_states.add(new_path_state)
                    queue.append((
                        other_entity_id,
                        path_nodes + [other_entity_id],
                        path_edges + [relation]
                    ))
        
        return paths
    
    def _get_other_entity_id(self, relation, current_entity_id: str) -> Optional[str]:
        """获取关系边的另一端实体 ID"""
        # 通过 absolute_id 找到 entity_id
        entity1 = self.storage_manager.get_entity_by_absolute_id(relation.entity1_absolute_id)
        entity2 = self.storage_manager.get_entity_by_absolute_id(relation.entity2_absolute_id)
        
        if not entity1 or not entity2:
            return None
        
        if entity1.entity_id == current_entity_id:
            return entity2.entity_id
        elif entity2.entity_id == current_entity_id:
            return entity1.entity_id
        else:
            return None
    
    def _build_path_result(
        self,
        node_ids: List[str],
        relations: List,
        include_content: bool
    ) -> Dict[str, Any]:
        """构建路径结果"""
        # 获取所有节点信息
        nodes = []
        for node_id in node_ids:
            entity = self.storage_manager.get_entity_by_id(node_id)
            if entity:
                node_info = {
                    "entity_id": entity.entity_id,
                    "name": entity.name
                }
                if include_content:
                    node_info["content"] = entity.content[:200] + "..." if len(entity.content) > 200 else entity.content
                nodes.append(node_info)
        
        # 获取所有边信息
        edges = []
        for relation in relations:
            entity1 = self.storage_manager.get_entity_by_absolute_id(relation.entity1_absolute_id)
            entity2 = self.storage_manager.get_entity_by_absolute_id(relation.entity2_absolute_id)
            
            edge_info = {
                "relation_id": relation.relation_id,
                "entity1_name": entity1.name if entity1 else "Unknown",
                "entity2_name": entity2.name if entity2 else "Unknown",
                "physical_time": relation.physical_time.isoformat() if relation.physical_time else None,
            }
            if include_content:
                edge_info["content"] = relation.content[:300] + "..." if len(relation.content) > 300 else relation.content
            edges.append(edge_info)
        
        # 生成路径描述
        path_description = self._generate_path_description(nodes, edges)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "hop_count": len(edges),
            "path_description": path_description
        }
    
    def _generate_path_description(
        self,
        nodes: List[Dict],
        edges: List[Dict]
    ) -> str:
        """生成路径的文字描述"""
        if not nodes:
            return ""
        
        parts = [nodes[0]["name"]]
        for i, edge in enumerate(edges):
            # 简化边的内容作为连接描述
            edge_content = edge.get("content", "相关")
            if len(edge_content) > 50:
                edge_content = edge_content[:50] + "..."
            
            parts.append(f" --[{edge_content}]--> ")
            if i + 1 < len(nodes):
                parts.append(nodes[i + 1]["name"])
        
        return "".join(parts)
