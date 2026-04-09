"""图遍历搜索 - BFS 扩展 + 社区感知搜索。"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models import Entity, Relation

logger = logging.getLogger(__name__)


class GraphTraversalSearcher:
    """图遍历搜索引擎：BFS 扩展、社区感知搜索。"""

    def __init__(self, storage: Any):
        self.storage = storage

    def bfs_expand(
        self,
        seed_family_ids: List[str],
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> List[Entity]:
        """从种子实体 BFS 扩展，返回发现的实体。

        优先使用存储层的 batch_bfs_traverse（单次 Cypher 查询），
        回退到逐节点扩展（兼容 SQLite 后端）。

        Args:
            seed_family_ids: 种子实体的 family_id 列表
            max_depth: 最大扩展深度（跳数）
            max_nodes: 最多返回的节点数

        Returns:
            发现的实体列表（包含种子实体）
        """
        # 优先使用批量 BFS（Neo4j 后端）
        if hasattr(self.storage, 'batch_bfs_traverse'):
            try:
                entities, _, _ = self.storage.batch_bfs_traverse(
                    seed_family_ids, max_depth=max_depth, max_nodes=max_nodes)
                return entities[:max_nodes]
            except Exception as e:
                logger.debug("batch_bfs_traverse failed, fallback to iterative: %s", e)

        # 回退：逐节点扩展（SQLite 后端）
        return self._iterative_bfs(seed_family_ids, max_depth, max_nodes)

    def _iterative_bfs(
        self,
        seed_family_ids: List[str],
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> List[Entity]:
        """逐节点 BFS 扩展（兼容 SQLite 后端）。"""
        visited: Set[str] = set()
        queue: List[Tuple[str, int]] = []  # (family_id, depth)
        result_entities: List[Entity] = []

        for eid in seed_family_ids:
            resolved = self.storage.resolve_family_id(eid)
            if resolved and resolved not in visited:
                visited.add(resolved)
                queue.append((resolved, 0))

        while queue and len(result_entities) < max_nodes:
            current_id, depth = queue.pop(0)
            entity = self.storage.get_entity_by_family_id(current_id)
            if entity:
                result_entities.append(entity)

            if depth >= max_depth:
                continue

            # 获取当前实体的关系
            relations = self.storage.get_relations_by_family_ids([current_id])

            # 收集所有 neighbor absolute_ids，批量获取
            neighbor_abs_ids = set()
            for rel in relations:
                neighbor_abs_ids.add(rel.entity1_absolute_id)
                neighbor_abs_ids.add(rel.entity2_absolute_id)

            if not neighbor_abs_ids:
                continue

            # 批量获取 neighbor 实体
            batch_fn = getattr(self.storage, 'get_entities_by_absolute_ids', None)
            if batch_fn:
                neighbor_entities = batch_fn(list(neighbor_abs_ids))
                abs_to_family = {e.absolute_id: e.family_id for e in neighbor_entities if e}
            else:
                # 回退：逐个获取（兼容旧后端）
                abs_to_family = {}
                for aid in neighbor_abs_ids:
                    ne = self.storage.get_entity_by_absolute_id(aid)
                    if ne:
                        abs_to_family[ne.absolute_id] = ne.family_id

            for fid in abs_to_family.values():
                if fid not in visited:
                    visited.add(fid)
                    queue.append((fid, depth + 1))

        return result_entities[:max_nodes]

    def get_entity_degree(self, family_id: str) -> int:
        """获取实体的度（连接数）。"""
        relations = self.storage.get_relations_by_family_ids([family_id])
        return len(relations)
