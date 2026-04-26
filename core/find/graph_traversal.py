"""图遍历搜索 - BFS 扩展 + 社区感知搜索。"""

import logging
import inspect
from collections import deque
from datetime import datetime as _dt
from typing import Any, Dict, List, Optional, Set, Tuple

from core.models import Entity, Relation

logger = logging.getLogger(__name__)


class GraphTraversalSearcher:
    """图遍历搜索引擎：BFS 扩展、社区感知搜索。"""

    def __init__(self, storage: Any):
        self.storage = storage
        # Cache: does storage.get_relations_by_family_ids support time_point?
        self._rels_supports_time_point = False
        _fn = getattr(storage, 'get_relations_by_family_ids', None)
        if _fn:
            self._rels_supports_time_point = 'time_point' in inspect.signature(_fn).parameters

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
        """Level-at-a-time BFS: fetches relations for all nodes at a depth level
        in a single batch call, reducing N queries per level to 1."""
        _tp_cache: Dict[str, Any] = {}
        _va_cache: Dict[str, Any] = {}  # valid_at string -> parsed datetime
        visited: Set[str] = set()
        result_entities: List[Entity] = []
        result_relations: List[Relation] = []
        seen_rel_fids: Set[str] = set()
        entity_cache: Dict[str, Any] = {}  # family_id -> Entity

        # Pre-check: does get_relations_by_family_ids support time_point?
        _get_rels_fn = getattr(self.storage, 'get_relations_by_family_ids', None)
        _get_rels_supports_tp = self._rels_supports_time_point
        _resolve_fn = getattr(self.storage, 'resolve_family_ids', None)
        _batch_fn = getattr(self.storage, 'get_entities_by_family_ids', None)

        # Helper: resolve family_ids
        def _resolve_ids(ids):
            if _resolve_fn:
                try:
                    resolved_map = _resolve_fn(ids) or {}
                    return [resolved_map.get(eid, eid) for eid in ids]
                except Exception as exc:
                    logger.debug("resolve_family_ids failed, fallback: %s", exc)
            return [self.storage.resolve_family_id(eid) for eid in ids]

        # Helper: time_point filter
        def _passes_time_filter(entity):
            if not time_point or not entity.valid_at:
                return True
            try:
                tp_dt = _tp_cache.get(time_point) or _dt.fromisoformat(
                    time_point.replace('Z', '+00:00'))
                _tp_cache[time_point] = tp_dt
            except (ValueError, TypeError):
                return True
            va = entity.valid_at
            if isinstance(va, str):
                parsed = _va_cache.get(va)
                if parsed is None:
                    try:
                        parsed = _dt.fromisoformat(va.replace('Z', '+00:00'))
                        _va_cache[va] = parsed
                    except (ValueError, TypeError):
                        return True
                va = parsed
            return not (isinstance(va, _dt) and va > tp_dt)

        # Initialize: resolve + enqueue seeds
        current_level: List[str] = []
        for resolved in _resolve_ids(seed_family_ids):
            if resolved and resolved not in visited:
                visited.add(resolved)
                current_level.append(resolved)

        for depth in range(max_depth + 1):
            if not current_level or len(result_entities) >= max_nodes:
                break

            # Batch-fetch entities for current level (uncached only)
            uncached = [fid for fid in current_level if fid not in entity_cache]
            if uncached:
                # Use family_ids (not absolute_ids) since current_level holds family IDs
                if _batch_fn:
                    fetched = _batch_fn(uncached)
                    for fid, e in fetched.items():
                        if e:
                            entity_cache[fid] = e
                else:
                    # Fallback for storage backends without batch method
                    for e in self.storage.get_entities_by_absolute_ids(uncached):
                        if e:
                            entity_cache[e.family_id] = e

            # Collect valid entities from current level
            next_level_ids: List[str] = []
            for fid in current_level:
                if len(result_entities) >= max_nodes:
                    break
                entity = entity_cache.get(fid) or self.storage.get_entity_by_family_id(fid)
                if entity:
                    if _passes_time_filter(entity):
                        result_entities.append(entity)
                        entity_cache[fid] = entity

            # Don't expand beyond max_depth
            if depth >= max_depth:
                break

            # Batch-fetch relations for ALL nodes at current level
            if _get_rels_fn:
                kw = dict(time_point=time_point) if _get_rels_supports_tp else {}
                all_relations = _get_rels_fn(current_level, **kw)
            else:
                all_relations = []

            # Collect unique neighbor absolute_ids across all relations
            neighbor_abs_ids: Set[str] = set()
            for rel in all_relations:
                if rel.family_id not in seen_rel_fids:
                    seen_rel_fids.add(rel.family_id)
                    result_relations.append(rel)
                neighbor_abs_ids.add(rel.entity1_absolute_id)
                neighbor_abs_ids.add(rel.entity2_absolute_id)

            if not neighbor_abs_ids:
                break

            # Batch-fetch all neighbor entities
            neighbor_entities = self.storage.get_entities_by_absolute_ids(neighbor_abs_ids)
            neighbor_family_ids: Set[str] = set()
            for e in neighbor_entities:
                if e:
                    entity_cache[e.family_id] = e
                    neighbor_family_ids.add(e.family_id)

            # Resolve neighbor family_ids + enqueue unvisited for next level
            # Skip resolution when we have no resolve function (family_ids are already canonical)
            _resolved_fids = _resolve_ids(neighbor_family_ids) if _resolve_fn else neighbor_family_ids
            for resolved in _resolved_fids:
                if resolved and resolved not in visited:
                    visited.add(resolved)
                    next_level_ids.append(resolved)

            current_level = next_level_ids

        return result_entities[:max_nodes], result_relations, visited
