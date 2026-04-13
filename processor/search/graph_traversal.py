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
        time_point: Optional[str] = None,
    ) -> List[Entity]:
        """从种子实体 BFS 扩展，返回发现的实体。

        优先使用存储层的 batch_bfs_traverse（单次 Cypher 查询），
        回退到逐节点扩展（兼容 SQLite 后端）。

        Args:
            seed_family_ids: 种子实体的 family_id 列表
            max_depth: 最大扩展深度（跳数）
            max_nodes: 最多返回的节点数
            time_point: ISO 8601 时间点，仅返回该时间点有效的实体和关系

        Returns:
            发现的实体列表（包含种子实体）
        """
        entities, relations, _ = self.bfs_expand_with_relations(
            seed_family_ids, max_depth=max_depth, max_nodes=max_nodes,
            time_point=time_point)
        return entities[:max_nodes]

    def bfs_expand_with_relations(
        self,
        seed_family_ids: List[str],
        max_depth: int = 2,
        max_nodes: int = 50,
        time_point: Optional[str] = None,
    ) -> Tuple[List[Entity], List[Relation], Set[str]]:
        """从种子实体 BFS 扩展，返回实体 + 关系 + 访问集合。

        Args:
            seed_family_ids: 种子实体的 family_id 列表
            max_depth: 最大扩展深度
            max_nodes: 最多返回的节点数
            time_point: ISO 8601 时间点，仅返回该时间点有效的实体和关系

        Returns:
            (entities, relations, visited_family_ids)
        """
        # 优先使用批量 BFS（Neo4j 后端）
        if hasattr(self.storage, 'batch_bfs_traverse'):
            try:
                entities, relations, visited = self.storage.batch_bfs_traverse(
                    seed_family_ids, max_depth=max_depth, max_nodes=max_nodes,
                    time_point=time_point)
                return entities[:max_nodes], relations, visited
            except Exception as e:
                logger.warning("batch_bfs_traverse failed, fallback to iterative: %s", e)

        # 回退：逐节点扩展（SQLite 后端）
        return self._iterative_bfs_with_relations(
            seed_family_ids, max_depth, max_nodes, time_point=time_point)

    def _iterative_bfs(
        self,
        seed_family_ids: List[str],
        max_depth: int = 2,
        max_nodes: int = 50,
        time_point: Optional[str] = None,
    ) -> List[Entity]:
        """逐节点 BFS 扩展（兼容 SQLite 后端）。"""
        entities, _, _ = self._iterative_bfs_with_relations(
            seed_family_ids, max_depth, max_nodes, time_point=time_point)
        return entities

    def _iterative_bfs_with_relations(
        self,
        seed_family_ids: List[str],
        max_depth: int = 2,
        max_nodes: int = 50,
        time_point: Optional[str] = None,
    ) -> Tuple[List[Entity], List[Relation], Set[str]]:
        """逐节点 BFS 扩展，同时收集关系（兼容 SQLite 后端）。"""
        _tp_cache: Dict[str, Any] = {}
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
                # time_point 过滤：跳过在 time_point 之后创建的实体版本
                if time_point and hasattr(entity, 'valid_at') and entity.valid_at:
                    from datetime import datetime as _dt
                    try:
                        tp_dt = _tp_cache.get(time_point) or _dt.fromisoformat(
                            time_point.replace('Z', '+00:00'))
                        _tp_cache[time_point] = tp_dt
                    except (ValueError, TypeError):
                        tp_dt = None
                    if tp_dt is not None:
                        va = entity.valid_at
                        if isinstance(va, str):
                            try:
                                va = _dt.fromisoformat(va.replace('Z', '+00:00'))
                            except (ValueError, TypeError):
                                pass
                        if isinstance(va, _dt) and va > tp_dt:
                            continue
                result_entities.append(entity)

            if depth >= max_depth:
                continue

            # 获取当前实体的关系（带 time_point 过滤）
            get_rels_fn = getattr(self.storage, 'get_relations_by_family_ids', None)
            if get_rels_fn:
                # 检查是否支持 time_point 参数
                import inspect
                sig = inspect.signature(get_rels_fn)
                if 'time_point' in sig.parameters:
                    relations = get_rels_fn([current_id], time_point=time_point)
                else:
                    relations = get_rels_fn([current_id])
            else:
                relations = []

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
