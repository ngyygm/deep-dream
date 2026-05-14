"""Neo4j EntityConfidenceMixin — confidence adjustment on contradiction/corroboration."""
import logging
from typing import List

logger = logging.getLogger(__name__)


class EntityConfidenceMixin:
    """Confidence adjustment methods for entities.

    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              -> Neo4j session factory
        self._run(session, cypher, **kw) -> execute Cypher with graph_id injection
        self._invalidate_entity_cache(family_id) -> scoped cache invalidation
        self._invalidate_entity_cache_bulk() -> broad cache invalidation
        self._invalidate_relation_cache_bulk() -> broad relation cache invalidation
    """

    def adjust_confidence_on_contradiction(self, family_id: str, source_type: str = "entity"):
        """矛盾证据时降低置信度。每次矛盾 -0.1，下限 0.0。"""
        label = "Entity" if source_type == "entity" else "Relation"
        with self._session() as session:
            self._run(session, f"""
                MATCH (n:{label} {{family_id: $fid}})
                WHERE n.invalid_at IS NULL AND n.confidence IS NOT NULL
                WITH n ORDER BY n.processed_time DESC LIMIT 1
                SET n.confidence = CASE
                    WHEN n.confidence - 0.1 < 0.0 THEN 0.0
                    ELSE n.confidence - 0.1
                END
            """, fid=family_id)
        if source_type == "entity":
            self._invalidate_entity_cache(family_id)
        else:
            self._invalidate_relation_cache_bulk()

    def adjust_confidence_on_contradiction_batch(self, family_ids: List[str], source_type: str = "entity"):
        """Batch version — lowers confidence for multiple family_ids in a single query."""
        if not family_ids:
            return
        label = "Entity" if source_type == "entity" else "Relation"
        with self._session() as session:
            self._run(session, f"""
                UNWIND $fids AS fid
                MATCH (n:{label} {{family_id: fid}})
                WHERE n.invalid_at IS NULL AND n.confidence IS NOT NULL
                WITH n ORDER BY n.processed_time DESC
                WITH n.family_id AS fid, collect(n)[0] AS latest
                SET latest.confidence = CASE
                    WHEN latest.confidence - 0.1 < 0.0 THEN 0.0
                    ELSE latest.confidence - 0.1
                END
            """, fids=family_ids)
        if source_type == "entity":
            self._invalidate_entity_cache_bulk()
        else:
            self._invalidate_relation_cache_bulk()

    def adjust_confidence_on_corroboration(self, family_id: str, source_type: str = "entity",
                                            is_dream: bool = False):
        """独立来源印证时提升置信度。

        Bayesian-inspired 增量调整：
        - 每次印证 +0.05，上限 1.0
        - Dream 来源印证权重减半 (+0.025)
        """
        label = "Entity" if source_type == "entity" else "Relation"
        delta = 0.025 if is_dream else 0.05
        with self._session() as session:
            self._run(session, f"""
                MATCH (n:{label} {{family_id: $fid}})
                WHERE n.invalid_at IS NULL AND n.confidence IS NOT NULL
                WITH n ORDER BY n.processed_time DESC LIMIT 1
                SET n.confidence = CASE
                    WHEN n.confidence + $delta > 1.0 THEN 1.0
                    ELSE n.confidence + $delta
                END
            """, fid=family_id, delta=delta)
        if source_type == "entity":
            self._invalidate_entity_cache(family_id)
        else:
            self._invalidate_relation_cache_bulk()

    def adjust_confidence_on_corroboration_batch(self, family_ids: List[str],
                                                  source_type: str = "entity",
                                                  is_dream: bool = False):
        """Batch version — adjusts confidence for multiple family_ids in a single query."""
        if not family_ids:
            return
        label = "Entity" if source_type == "entity" else "Relation"
        delta = 0.025 if is_dream else 0.05
        with self._session() as session:
            self._run(session, f"""
                UNWIND $fids AS fid
                MATCH (n:{label} {{family_id: fid}})
                WHERE n.invalid_at IS NULL AND n.confidence IS NOT NULL
                WITH n ORDER BY n.processed_time DESC
                WITH n.family_id AS fid, collect(n)[0] AS latest
                SET latest.confidence = CASE
                    WHEN latest.confidence + $delta > 1.0 THEN 1.0
                    ELSE latest.confidence + $delta
                END
            """, fids=family_ids, delta=delta)
        if source_type == "entity":
            self._invalidate_entity_cache_bulk()
        else:
            self._invalidate_relation_cache_bulk()
