"""Neo4j EntityEmbeddingMixin — embedding computation, cache management, and invalidation."""
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from ...models import Entity, Relation
from ...perf import _perf_timer
from ._helpers import _encode_and_normalize, _ENTITY_RETURN_FIELDS_WITH_EMB, _neo4j_record_to_entity, _neo4j_record_to_relation, _q

logger = logging.getLogger(__name__)


class EntityEmbeddingMixin:
    """Embedding computation, embedding cache management, and cache invalidation for entities.

    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              -> Neo4j session factory
        self._run(session, cypher, **kw) -> execute Cypher with graph_id injection
        self._graph_id: str          -> active graph ID
        self._cache                  -> QueryCache
        self.embedding_client        -> EmbeddingClient (optional)
        self._entity_emb_cache       -> embedding cache list
        self._entity_emb_cache_ts    -> embedding cache timestamp
        self._emb_cache_ttl          -> cache TTL in seconds
    """

    # Maximum content length used for embedding computation (avoids oversized inputs)
    _EMB_CONTENT_MAX = 512

    def _compute_entity_embedding(self, entity: Entity) -> Optional[tuple]:
        """计算实体的 embedding 向量（L2 归一化后存储）。

        Returns:
            (emb_bytes, emb_array) tuple or None. Caller can use emb_array directly
            to avoid a redundant np.frombuffer round-trip.
        """
        content = entity.content or ""
        if len(content) > self._EMB_CONTENT_MAX:
            content = content[:self._EMB_CONTENT_MAX]
        text = f"# {entity.name}\n{content}"
        return _encode_and_normalize(self.embedding_client, text)

    def _get_all_absolute_ids_for_entity(self, family_id: str) -> List[str]:
        """获取实体的所有版本的 absolute_id。"""
        with self._session() as session:
            result = self._run(session,
                "MATCH (e:Entity {family_id: $fid}) RETURN e.uuid AS uuid",
                fid=family_id,
            )
            return [record["uuid"] for record in result]

    def get_latest_absolute_ids_by_family_ids(self, family_ids: List[str]) -> Dict[str, str]:
        """批量获取每个 family_id 的最新版本 absolute_id（轻量，不含 embedding）。

        比 get_entities_by_family_ids 轻量得多，适用于只需要 UUID 映射的场景。
        """
        if not family_ids:
            return {}
        with self._session() as session:
            result = self._run(session, """
                MATCH (e:Entity)
                WHERE e.family_id IN $fids AND e.invalid_at IS NULL
                WITH e.family_id AS fid, e ORDER BY e.processed_time DESC
                WITH fid, collect(e.uuid)[0] AS latest_uuid
                RETURN fid, latest_uuid
            """, fids=family_ids)
            return {r["fid"]: r["latest_uuid"] for r in result if r["latest_uuid"]}

    def _get_entities_with_embeddings(self) -> List[tuple]:
        """获取所有实体的最新版本及其 embedding（带短 TTL 缓存）。"""
        now = time.time()
        if self._entity_emb_cache is not None and (now - self._entity_emb_cache_ts) < self._emb_cache_ttl:
            return self._entity_emb_cache
        with _perf_timer("_get_entities_with_embeddings"):
            result = self._get_entities_with_embeddings_impl()
        self._entity_emb_cache = result
        self._entity_emb_cache_ts = time.time()
        return result

    def _get_entities_with_embeddings_impl(self) -> List[tuple]:
        """获取所有实体的最新版本及其 embedding（实际实现）。"""
        with self._session() as session:
            limit = getattr(self, '_emb_cache_max_size', 10000)
            result = self._run(session,
                f"""
                MATCH (e:Entity)
                WITH e.family_id AS fid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(COLLECT(e)) AS e
                RETURN {_ENTITY_RETURN_FIELDS_WITH_EMB}
                ORDER BY e.processed_time DESC
                LIMIT $limit
                """,
                limit=limit,
            )
            records = list(result)

        if not records:
            return []

        entities = []
        for record in records:
            entity = _neo4j_record_to_entity(record)
            emb_array = np.frombuffer(entity.embedding, dtype=np.float32) if entity.embedding else None
            entities.append((entity, emb_array))
        return entities

    def _get_entity_relations_by_family_id_impl(self, family_id: str, limit: Optional[int] = None,
                                                 time_point: Optional[datetime] = None,
                                                 max_version_absolute_id: Optional[str] = None) -> List[Relation]:
        """通过 family_id 获取实体的所有关系（实际实现）。

        Merged into a single Neo4j session to avoid 3 separate round-trips:
        1. Collect absolute_ids (inline subquery)
        2. Optional max_version filter (inline WITH clause)
        3. Relation lookup (main query)
        """
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []

        with self._session() as session:
            # Build the abs_ids subquery inline
            if max_version_absolute_id:
                abs_query = """
                    MATCH (e2:Entity {uuid: $max_abs})
                    WITH e2.processed_time AS max_pt
                    MATCH (e:Entity {family_id: $fid})
                    WHERE e.processed_time <= max_pt
                    WITH COLLECT(e.uuid) AS abs_ids
                """
                abs_params = {"max_abs": max_version_absolute_id, "fid": family_id}
            else:
                abs_query = """
                    MATCH (e:Entity {family_id: $fid})
                    WITH COLLECT(e.uuid) AS abs_ids
                """
                abs_params = {"fid": family_id}

            if time_point:
                rel_query = """
                    UNWIND abs_ids AS aid
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id = aid OR r.entity2_absolute_id = aid)
                    AND r.event_time <= datetime($tp)
                    AND r.invalid_at IS NULL
                    WITH r.family_id AS fid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH fid, r ORDER BY r.processed_time DESC
                    WITH fid, HEAD(COLLECT(r)) AS r
                """
                abs_params["tp"] = time_point.isoformat()
            else:
                rel_query = """
                    UNWIND abs_ids AS aid
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id = aid OR r.entity2_absolute_id = aid)
                    AND r.invalid_at IS NULL
                    WITH r.family_id AS fid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH fid, r ORDER BY r.processed_time DESC
                    WITH fid, HEAD(COLLECT(r)) AS r
                """

            query = abs_query + rel_query + _q("RETURN __REL_FIELDS__ ORDER BY r.processed_time DESC")
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = self._run(session, query, **abs_params)
            return [_neo4j_record_to_relation(r) for r in result]

    def _update_entity_emb_cache(self, entity: Entity, emb_array: Optional[np.ndarray]):
        """Append-only update to entity embedding cache.

        If cache is warm, update existing family_id entry or append new one.
        If cache is cold, skip — it will be rebuilt from scratch on next access.
        """
        if self._entity_emb_cache is None:
            return
        # Use dict-based lookup instead of linear scan
        if not hasattr(self, '_entity_emb_fid_idx') or self._entity_emb_fid_idx is None:
            self._entity_emb_fid_idx = {e.family_id: i for i, (e, _) in enumerate(self._entity_emb_cache)}
        idx = self._entity_emb_fid_idx.get(entity.family_id)
        if idx is not None:
            self._entity_emb_cache[idx] = (entity, emb_array)
        else:
            self._entity_emb_cache.append((entity, emb_array))
            self._entity_emb_fid_idx[entity.family_id] = len(self._entity_emb_cache) - 1

    def _update_entity_emb_cache_batch(self, items: List[tuple]):
        """Batch append-only update for entity embedding cache.

        Args:
            items: List of (entity, emb_array) tuples.
        """
        if self._entity_emb_cache is None or not items:
            return
        # Build lookup for O(1) family_id -> index
        if hasattr(self, '_entity_emb_fid_idx') and self._entity_emb_fid_idx is not None:
            fid_to_idx = self._entity_emb_fid_idx
        else:
            fid_to_idx = {e.family_id: i for i, (e, _) in enumerate(self._entity_emb_cache)}
            self._entity_emb_fid_idx = fid_to_idx
        for entity, emb_array in items:
            idx = fid_to_idx.get(entity.family_id)
            if idx is not None:
                self._entity_emb_cache[idx] = (entity, emb_array)
            else:
                self._entity_emb_cache.append((entity, emb_array))
                fid_to_idx[entity.family_id] = len(self._entity_emb_cache) - 1

    def _invalidate_entity_cache(self, family_id: str):
        """Scoped cache invalidation for a single entity family_id.

        Replaces broad pattern invalidation to preserve cache entries
        for unrelated entities during batch processing.
        """
        keys = [
            f"entity:by_fid:{family_id}",
            f"resolve:{family_id}",
        ]
        # Also invalidate absolute_id cache entries for all versions
        abs_ids = self._get_all_absolute_ids_for_entity(family_id)
        for aid in abs_ids:
            keys.append(f"entity:by_abs:{aid}")
        self._cache.invalidate_keys(keys)

    def _invalidate_entity_cache_bulk(self):
        """Broad entity cache invalidation — only for bulk operations."""
        self._cache.invalidate("entity:")
        self._cache.invalidate("resolve:")
        self._cache.invalidate("sim_search:")
        self.invalidate_entity_remap_cache()
