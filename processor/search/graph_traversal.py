"""图遍历搜索 - BFS 扩展 + 社区感知搜索。"""

import logging
from collections import deque
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
        entities, relations, _ = self.bfs_expand_with_relations(
            seed_family_ids, max_depth=max_depth, max_nodes=max_nodes)
        return entities[:max_nodes]

    def bfs_expand_with_relations(
        self,
        seed_family_ids: List[str],
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> Tuple[List[Entity], List[Relation], Set[str]]:
        """从种子实体 BFS 扩展，返回实体 + 关系 + 访问集合。

        Returns:
            (entities, relations, visited_family_ids)
        """
        # 优先使用批量 BFS（Neo4j 后端）
        if hasattr(self.storage, 'batch_bfs_traverse'):
            try:
                entities, relations, visited = self.storage.batch_bfs_traverse(
                    seed_family_ids, max_depth=max_depth, max_nodes=max_nodes)
                return entities[:max_nodes], relations, visited
            except Exception as e:
                logger.debug("batch_bfs_traverse failed, fallback to iterative: %s", e)

        # 回退：逐节点扩展（SQLite 后端）
        return self._iterative_bfs_with_relations(seed_family_ids, max_depth, max_nodes)

    def _iterative_bfs(
        self,
        seed_family_ids: List[str],
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> List[Entity]:
        """逐节点 BFS 扩展（兼容 SQLite 后端）。"""
        entities, _, _ = self._iterative_bfs_with_relations(
            seed_family_ids, max_depth, max_nodes)
        return entities

    def _iterative_bfs_with_relations(
        self,
        seed_family_ids: List[str],
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> Tuple[List[Entity], List[Relation], Set[str]]:
        """逐节点 BFS 扩展，同时收集关系（兼容 SQLite 后端）。"""
        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque()  # (family_id, depth)
        result_entities: List[Entity] = []
        result_relations: List[Relation] = []
        seen_rel_fids: Set[str] = set()

        # 批量 resolve 种子 family_ids
        resolve_fn = getattr(self.storage, 'resolve_family_ids', None)
        if resolve_fn:
            try:
                resolved_map = resolve_fn(seed_family_ids) or {}
                for eid in seed_family_ids:
                    resolved = resolved_map.get(eid, eid)
                    if resolved and resolved not in visited:
                        visited.add(resolved)
                        queue.append((resolved, 0))
            except Exception as exc:
                logger.debug("resolve_family_ids failed, fallback: %s", exc)
                for eid in seed_family_ids:
                    resolved = self.storage.resolve_family_id(eid)
                    if resolved and resolved not in visited:
                        visited.add(resolved)
                        queue.append((resolved, 0))
        else:
            for eid in seed_family_ids:
                resolved = self.storage.resolve_family_id(eid)
                if resolved and resolved not in visited:
                    visited.add(resolved)
                    queue.append((resolved, 0))

        while queue and len(result_entities) < max_nodes:
            current_id, depth = queue.popleft()
            entity = self.storage.get_entity_by_family_id(current_id)
            if entity:
                result_entities.append(entity)

            if depth >= max_depth:
                continue

            # 获取当前实体的关系
            relations = self.storage.get_relations_by_family_ids([current_id])

            # 收集关系 + neighbor absolute_ids
            neighbor_abs_ids = set()
            for rel in relations:
                if rel.family_id not in seen_rel_fids:
                    seen_rel_fids.add(rel.family_id)
                    result_relations.append(rel)
                neighbor_abs_ids.add(rel.entity1_absolute_id)
                neighbor_abs_ids.add(rel.entity2_absolute_id)

            if not neighbor_abs_ids:
                continue

            # 批量获取 neighbor 实体
            neighbor_entities = self.storage.get_entities_by_absolute_ids(list(neighbor_abs_ids))
            abs_to_family = {e.absolute_id: e.family_id for e in neighbor_entities if e}

            for fid in abs_to_family.values():
                if fid not in visited:
                    visited.add(fid)
                    queue.append((fid, depth + 1))

        return result_entities[:max_nodes], result_relations, visited

