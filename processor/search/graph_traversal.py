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
        seed_entity_ids: List[str],
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> List[Entity]:
        """从种子实体 BFS 扩展，返回发现的实体。

        Args:
            seed_entity_ids: 种子实体的 entity_id 列表
            max_depth: 最大扩展深度（跳数）
            max_nodes: 最多返回的节点数

        Returns:
            发现的实体列表（包含种子实体）
        """
        visited: Set[str] = set()
        queue: List[Tuple[str, int]] = []  # (entity_id, depth)
        result_entities: List[Entity] = []

        for eid in seed_entity_ids:
            resolved = self.storage.resolve_entity_id(eid)
            if resolved and resolved not in visited:
                visited.add(resolved)
                queue.append((resolved, 0))

        while queue and len(result_entities) < max_nodes:
            current_id, depth = queue.pop(0)
            entity = self.storage.get_entity_by_entity_id(current_id)
            if entity:
                result_entities.append(entity)

            if depth >= max_depth:
                continue

            # 获取当前实体的关系
            relations = self.storage.get_relations_by_entity_ids([current_id])
            for rel in relations:
                e1_id = rel.entity1_absolute_id
                e2_id = rel.entity2_absolute_id

                # 获取关系对端的实体
                for abs_id in [e1_id, e2_id]:
                    neighbor = self.storage.get_entity_by_absolute_id(abs_id)
                    if neighbor and neighbor.entity_id not in visited:
                        visited.add(neighbor.entity_id)
                        queue.append((neighbor.entity_id, depth + 1))

        return result_entities[:max_nodes]

    def community_aware_search(
        self,
        query_entity_ids: List[str],
        community_id: Optional[str] = None,
        max_results: int = 30,
    ) -> List[Tuple[Entity, float]]:
        """社区感知搜索：优先返回同社区实体。

        Args:
            query_entity_ids: 查询实体 ID 列表
            community_id: 目标社区 ID（可选，不传则自动检测查询实体所在社区）
            max_results: 最大返回数量

        Returns:
            [(Entity, relevance_score), ...] 按相关性降序
        """
        # 确定查询实体所在社区
        query_community = community_id
        if not query_community and query_entity_ids:
            # 尝试从存储获取社区信息
            for eid in query_entity_ids[:1]:
                entity = self.storage.get_entity_by_entity_id(eid)
                if entity:
                    query_community = getattr(entity, 'community_id', None)
                    break

        # 获取同社区实体
        community_entities: List[Entity] = []
        other_entities: List[Entity] = []

        if query_community:
            try:
                # 尝试获取社区成员
                all_entities = self.storage.get_all_entities(limit=500, exclude_embedding=True)
                for e in all_entities:
                    cid = getattr(e, 'community_id', None)
                    if cid == query_community:
                        community_entities.append(e)
                    else:
                        other_entities.append(e)
            except Exception as ex:
                logger.debug("社区感知搜索回退到普通搜索: %s", ex)
                community_entities = self.storage.get_all_entities(limit=max_results, exclude_embedding=True)
        else:
            # 无社区信息，回退到 BFS 扩展
            community_entities = self.bfs_expand(query_entity_ids, max_depth=1, max_nodes=max_results)

        # 同社区实体排在前面
        results: List[Tuple[Entity, float]] = []
        for i, e in enumerate(community_entities[:max_results]):
            score = 1.0 - (i * 0.02)
            results.append((e, max(0.0, score)))

        # 补充其他社区实体
        remaining = max_results - len(results)
        if remaining > 0:
            for i, e in enumerate(other_entities[:remaining]):
                score = 0.5 - (i * 0.02)
                results.append((e, max(0.0, score)))

        return results[:max_results]

    def get_entity_degree(self, entity_id: str) -> int:
        """获取实体的度（连接数）。"""
        relations = self.storage.get_relations_by_entity_ids([entity_id])
        return len(relations)
