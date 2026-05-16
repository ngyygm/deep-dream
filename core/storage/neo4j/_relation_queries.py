"""Neo4j RelationQueryMixin — read-only relation query methods."""
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...models import Relation
from ._helpers import (
    _RELATION_RETURN_FIELDS,
    _RELATION_RETURN_FIELDS_WITH_EMB,
    _expand_cypher,
    _neo4j_record_to_relation,
    _q,
)

logger = logging.getLogger(__name__)


class RelationQueryMixin:
    """Read-only relation query methods.

    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              -> Neo4j session factory
        self._run(session, cypher, **kw) -> execute Cypher with graph_id injection
        self._graph_id: str          -> active graph ID
        self._entity_remap_cache     -> entity abs_id remap dict or None
        self._entity_remap_cache_ts  -> remap cache timestamp
        self._entity_remap_cache_ttl -> remap cache TTL in seconds
    """

    def count_relations_since(self, since: str) -> int:
        """Count relations whose latest version has processed_time > since."""
        with self._session() as session:
            result = self._run(session, """
                MATCH (r:Relation)
                WITH r.family_id AS fid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH fid, r ORDER BY r.processed_time DESC
                WITH fid, HEAD(COLLECT(r)) AS r
                WHERE r.processed_time > datetime($since)
                RETURN COUNT(r) AS cnt
            """, since=since)
            rec = result.single()
            return rec["cnt"] if rec else 0

    def count_unique_relations(self) -> int:
        """Count unique family_ids among valid relations."""
        with self._session() as session:
            result = self._run(session,
                "MATCH (r:Relation) WHERE r.invalid_at IS NULL RETURN COUNT(DISTINCT r.family_id) AS cnt"
            )
            record = result.single()
            return record["cnt"] if record else 0

    def get_all_relations(self, limit: Optional[int] = None, offset: Optional[int] = None,
                           exclude_embedding: bool = False,
                           include_candidates: bool = False) -> List[Relation]:
        """Get all latest-version relations."""
        with self._session() as session:
            fields = _RELATION_RETURN_FIELDS if exclude_embedding else _RELATION_RETURN_FIELDS_WITH_EMB
            query = f"""
                MATCH (r:Relation)
                WHERE r.invalid_at IS NULL
                WITH r.family_id AS fid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH fid, r ORDER BY r.processed_time DESC
                WITH fid, HEAD(COLLECT(r)) AS r
                RETURN {fields}
                ORDER BY r.processed_time DESC
            """
            if offset is not None and offset > 0:
                query += f" SKIP {int(offset)}"
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = self._run(session, query)
            records = list(result)

        relations = [_neo4j_record_to_relation(r) for r in records]
        return self._filter_dream_candidates(relations, include_candidates)

    def get_invalidated_relations(self, limit: int = 100) -> List[Relation]:
        """List invalidated relations."""
        with self._session() as session:
            result = self._run(session, _q("""
                MATCH (r:Relation)
                WHERE r.invalid_at IS NOT NULL
                RETURN __REL_FIELDS__
                ORDER BY r.invalid_at DESC
                LIMIT $limit
            """), limit=limit)
            return [_neo4j_record_to_relation(r) for r in result]

    def get_relation_by_absolute_id(self, relation_absolute_id: str) -> Optional[Relation]:
        """Get relation by absolute_id."""
        with self._session() as session:
            result = self._run(session,
                f"""
                MATCH (r:Relation {{uuid: $uuid}})
                RETURN {_RELATION_RETURN_FIELDS_WITH_EMB}
                """,
                uuid=relation_absolute_id,
            )
            record = result.single()
            if not record:
                return None
            return _neo4j_record_to_relation(record)

    def get_relation_by_family_id(self, family_id: str) -> Optional[Relation]:
        with self._session() as session:
            result = self._run(session,
                f"""
                MATCH (r:Relation {{family_id: $fid}})
                RETURN {_RELATION_RETURN_FIELDS_WITH_EMB}
                ORDER BY r.processed_time DESC LIMIT 1
                """,
                fid=family_id,
            )
            record = result.single()
            if not record:
                return None
            return _neo4j_record_to_relation(record)

    def get_relation_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        """Get relation embedding preview."""
        with self._session() as session:
            result = self._run(session,
                "MATCH (r:Relation {uuid: $uuid}) RETURN r.embedding AS embedding",
                uuid=absolute_id,
            )
            record = result.single()
            if record and record["embedding"]:
                return record["embedding"][:num_values]
        return None

    def get_relation_embeddings(self, family_ids: List[str]) -> Dict[str, Any]:
        """Batch get relation embedding vectors.

        Returns:
            {family_id: np.ndarray(shape=(dim,), dtype=float32)} -- L2 normalized vectors.
        """
        if not family_ids:
            return {}
        with self._session() as session:
            result = self._run(session,
                "MATCH (r:Relation) WHERE r.family_id IN $fids AND r.embedding IS NOT NULL AND r.invalid_at IS NULL "
                "RETURN r.family_id AS fid, r.embedding AS emb",
                fids=family_ids,
            )
            return {rec["fid"]: (np.frombuffer(rec["emb"], dtype=np.float32) if isinstance(rec["emb"], (bytes, bytearray, memoryview)) else np.array(rec["emb"], dtype=np.float32)).copy() for rec in result}

    def get_relation_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
        """Batch get version counts for multiple relation family_ids."""
        if not family_ids:
            return {}
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = list({r for r in resolved_map.values() if r})
        if not canonical_ids:
            return {}
        with self._session() as session:
            result = self._run(session,
                """
                MATCH (r:Relation)
                WHERE r.family_id IN $fids
                RETURN r.family_id AS family_id, COUNT(r) AS cnt
                """,
                fids=canonical_ids,
            )
            return {record["family_id"]: record["cnt"] for record in result}

    def get_relation_versions(self, family_id: str) -> List[Relation]:
        """Get all versions of a relation."""
        with self._session() as session:
            result = self._run(session, _q("""
                MATCH (r:Relation {family_id: $fid})
                RETURN __REL_FIELDS__
                ORDER BY r.processed_time ASC
                """),
                fid=family_id,
            )
            return [_neo4j_record_to_relation(r) for r in result]

    def get_relation_versions_batch(self, family_ids: List[str]) -> Dict[str, List[Relation]]:
        """Batch get all relation versions for multiple family_ids (single Cypher query)."""
        if not family_ids:
            return {}
        with self._session() as session:
            result = self._run(session,
                _q("""
                UNWIND $fids AS fid
                MATCH (r:Relation {family_id: fid})
                RETURN r.family_id AS fid, __REL_FIELDS__
                ORDER BY r.processed_time ASC
                """),
                fids=family_ids,
            )
            versions_map: Dict[str, List[Relation]] = {fid: [] for fid in family_ids}
            for record in result:
                fid = record["fid"]
                if fid in versions_map:
                    versions_map[fid].append(_neo4j_record_to_relation(record))
        return versions_map

    def get_relations_by_absolute_ids(self, absolute_ids: List[str], valid_only: bool = False) -> List[Relation]:
        """Batch get relations by absolute_ids."""
        if not absolute_ids:
            return []
        extra_filter = " AND r.invalid_at IS NULL" if valid_only else ""
        with self._session() as session:
            result = self._run(session, _q(f"""
                MATCH (r:Relation)
                WHERE r.uuid IN $uuids{extra_filter}
                RETURN __REL_FIELDS__
                """),
                uuids=absolute_ids,
            )
            return [_neo4j_record_to_relation(r) for r in result]

    def get_relations_by_entities(self, from_family_id: str, to_family_id: str,
                                   include_candidates: bool = False) -> List[Relation]:
        """Get all relations between two family_ids (merged into 2 session queries)."""
        from ...perf import _perf_timer
        with _perf_timer("get_relations_by_entities"):
            result = self._get_relations_by_entities_impl(from_family_id, to_family_id)
            return self._filter_dream_candidates(result, include_candidates)

    def get_relations_by_entity_absolute_ids(self, entity_absolute_ids: List[str],
                                              limit: Optional[int] = None,
                                              include_candidates: bool = False) -> List[Relation]:
        """Get relations by absolute_id list."""
        if not entity_absolute_ids:
            return []
        with self._session() as session:
            query = _q("""
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id IN $abs_ids OR r.entity2_absolute_id IN $abs_ids)
                  AND r.invalid_at IS NULL
                WITH r.family_id AS fid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH fid, r ORDER BY r.processed_time DESC
                WITH fid, HEAD(COLLECT(r)) AS r
                RETURN __REL_FIELDS__
                ORDER BY r.processed_time DESC
            """)
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = self._run(session, query, abs_ids=entity_absolute_ids)
            relations = [_neo4j_record_to_relation(r) for r in result]
            return self._filter_dream_candidates(relations, include_candidates)

    def get_relations_by_entity_pairs(self, entity_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], List[Relation]]:
        """Batch get relations for multiple entity pairs via entity family_id matching."""
        if not entity_pairs:
            return {}

        # Deduplicate pair keys (sorted)
        seen_pair_keys: set = set()
        unique_pairs = []
        for e1, e2 in entity_pairs:
            pk = (e1, e2) if e1 <= e2 else (e2, e1)
            if pk not in seen_pair_keys:
                seen_pair_keys.add(pk)
                unique_pairs.append({"f1": pk[0], "f2": pk[1]})

        # Use entity1_family_id / entity2_family_id on Relation nodes (backfilled)
        with self._session() as session:
            result = self._run(session, _q("""
                UNWIND $pairs AS p
                MATCH (r:Relation)
                WHERE r.invalid_at IS NULL
                  AND (
                    (r.entity1_family_id = p.f1 AND r.entity2_family_id = p.f2)
                    OR (r.entity1_family_id = p.f2 AND r.entity2_family_id = p.f1)
                  )
                RETURN __REL_FIELDS__,
                    r.entity1_family_id AS e1fid,
                    r.entity2_family_id AS e2fid
                """),
                pairs=unique_pairs,
            )
            records = list(result)

        # Group by family_id pair, deduplicate by relation family_id
        _rel_by_pair: Dict[Tuple[str, str], List[Relation]] = defaultdict(list)
        seen_rel_fids: set = set()
        for rec in records:
            rel = _neo4j_record_to_relation(rec)
            if rel.family_id in seen_rel_fids:
                continue
            seen_rel_fids.add(rel.family_id)
            f1 = rec.get("e1fid")
            f2 = rec.get("e2fid")
            if f1 and f2:
                pk = (f1, f2) if f1 <= f2 else (f2, f1)
                _rel_by_pair[pk].append(rel)

        results: Dict[Tuple[str, str], List[Relation]] = {}
        for e1, e2 in entity_pairs:
            pk = (e1, e2) if e1 <= e2 else (e2, e1)
            if pk not in results:
                results[pk] = _rel_by_pair.get(pk, [])

        return results

    def get_relations_by_family_ids(self, family_ids: List[str], limit: int = 100,
                                    time_point: Optional[str] = None) -> List[Relation]:
        """Get all relations for specified entity family_ids.

        Uses a single Cypher query for family_id->absolute_id resolution + relation
        retrieval to avoid N+1 per-family_id calls.

        Args:
            family_ids: Entity family_id list
            limit: Max results
            time_point: ISO 8601 timestamp, only return relations with valid_at <= time_point
        """
        if not family_ids:
            return []
        _tp_filter = ""
        _tp_param = {}
        if time_point:
            _tp_filter = " AND (r.valid_at IS NULL OR r.valid_at <= datetime($tp))"
            _tp_param["tp"] = time_point
        with self._session() as session:
            result = self._run(session, _expand_cypher("""
                MATCH (e:Entity)
                WHERE e.family_id IN $family_ids AND e.invalid_at IS NULL
                WITH collect(DISTINCT e.uuid) AS abs_ids
                UNWIND abs_ids AS aid
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id = aid OR r.entity2_absolute_id = aid)
                  AND r.invalid_at IS NULL%s
                RETURN DISTINCT __REL_FIELDS__
                LIMIT $limit
            """ % _tp_filter), family_ids=family_ids, limit=limit, **_tp_param)
            return [_neo4j_record_to_relation(r) for r in result]

    def get_relations_referencing_absolute_id(self, absolute_id: str) -> List[Relation]:
        """Get all relations referencing the given absolute_id."""
        with self._session() as session:
            result = self._run(session, _q("""
                MATCH (r:Relation)
                WHERE r.entity1_absolute_id = $aid OR r.entity2_absolute_id = $aid
                RETURN __REL_FIELDS__
                """),
                aid=absolute_id,
            )
            rels = [_neo4j_record_to_relation(r) for r in result]
            self._remap_relation_endpoints(rels, session)
            return rels

    def stream_all_relations(self, exclude_embedding: bool = True,
                             include_candidates: bool = False,
                             since: Optional[str] = None):
        """Yield latest-version relations one by one from the Neo4j cursor.

        Unlike get_all_relations(), this does not materialize the full result
        set -- suitable for SSE streaming.

        Relation endpoints are remapped to the latest entity version absolute_id
        so they match the entities returned by stream_all_entities.

        If *since* (ISO timestamp) is given, only yield relations whose latest
        version has processed_time > since.
        """
        with self._streaming_session() as session:
            fields = _RELATION_RETURN_FIELDS if exclude_embedding else _RELATION_RETURN_FIELDS_WITH_EMB
            params = {}
            if since:
                query = f"""
                    MATCH (r:Relation)
                    WITH r.family_id AS fid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH fid, r ORDER BY r.processed_time DESC
                    WITH fid, HEAD(COLLECT(r)) AS r
                    WHERE r.processed_time > datetime($since)
                    RETURN {fields}
                    ORDER BY r.processed_time ASC
                """
                params["since"] = since
            else:
                query = f"""
                    MATCH (r:Relation)
                    WITH r.family_id AS fid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH fid, r ORDER BY r.processed_time DESC
                    WITH fid, HEAD(COLLECT(r)) AS r
                    RETURN {fields}
                    ORDER BY r.processed_time ASC
                """
            result = self._run(session, query, **params)
            remap = self._build_entity_abs_id_remap(session)
            for record in result:
                rel = _neo4j_record_to_relation(record)
                if remap:
                    rel.entity1_absolute_id = remap.get(rel.entity1_absolute_id, rel.entity1_absolute_id)
                    rel.entity2_absolute_id = remap.get(rel.entity2_absolute_id, rel.entity2_absolute_id)
                if not self._is_dream_candidate(rel) or include_candidates:
                    yield rel

    def _search_relations_with_embedding(self, query_text: str,
                                          relations_with_embeddings: List[tuple],
                                          threshold: float,
                                          max_results: int,
                                          query_embedding=None) -> List[Relation]:
        """Vector similarity search for relations using Neo4j HNSW index."""
        # 1. Encode + normalize query (skip if caller provided embedding)
        if query_embedding is None:
            query_embedding = self.embedding_client.encode(query_text)
        if query_embedding is None:
            return []

        query_emb = np.asarray(query_embedding, dtype=np.float32)
        if query_emb.ndim > 1:
            query_emb = query_emb[0]
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        # 2. Neo4j vector index KNN
        knn_limit = max_results * 5
        query_vector = query_emb.tolist()
        with self._session() as session:
            try:
                result = session.run(
                    """
                    CALL db.index.vector.queryNodes('relation_embedding', $k, $queryVector)
                    YIELD node, score
                    WHERE node.graph_id = $graph_id AND node.invalid_at IS NULL
                    RETURN node, score
                    ORDER BY score DESC
                    """,
                    k=knn_limit,
                    queryVector=query_vector,
                    graph_id=self._graph_id,
                )
                records = list(result)
            except Exception as e:
                logger.warning("Neo4j relation vector search failed: %s", e)
                return []

        if not records:
            return []

        # 3. Deduplicate (highest score per family_id) + threshold filter
        seen = set()
        results = []
        for record in records:
            node = record["node"]
            score = record["score"]
            if score < threshold:
                break
            family_id = node.get("family_id")
            if family_id in seen:
                continue
            seen.add(family_id)
            rel_dict = {
                "uuid": node.get("uuid"),
                "family_id": family_id,
                "entity1_absolute_id": node.get("entity1_absolute_id", ""),
                "entity2_absolute_id": node.get("entity2_absolute_id", ""),
                "content": node.get("content", ""),
                "event_time": node.get("event_time"),
                "processed_time": node.get("processed_time"),
                "episode_id": node.get("episode_id", ""),
                "source_document": node.get("source_document", ""),
                "valid_at": node.get("valid_at"),
                "invalid_at": node.get("invalid_at"),
                "summary": node.get("summary"),
                "attributes": node.get("attributes"),
                "confidence": node.get("confidence"),
                "provenance": node.get("provenance"),
                "embedding": node.get("embedding"),
            }
            relation = _neo4j_record_to_relation(rel_dict)
            results.append(relation)
            if len(results) >= max_results:
                break
        return results

    def _build_entity_abs_id_remap(self, session) -> dict:
        """Build mapping from any entity absolute_id -> latest absolute_id per family_id.

        Results are cached with a TTL to avoid repeated full-entity scans.
        Invalidate via self.invalidate_entity_remap_cache() after entity writes.
        """
        now = time.time()
        if (self._entity_remap_cache is not None
                and (now - self._entity_remap_cache_ts) < self._entity_remap_cache_ttl):
            return self._entity_remap_cache

        result = self._run(session, """
            MATCH (e:Entity)
            WITH e.family_id AS fid, COLLECT(e) AS ents
            UNWIND ents AS e
            WITH fid, e ORDER BY e.processed_time DESC
            WITH fid, COLLECT(e.uuid) AS uuids
            RETURN fid, uuids[0] AS latest, uuids AS all_uuids
        """)
        remap = {}
        for rec in result:
            latest = rec["latest"]
            for uuid in rec["all_uuids"]:
                if uuid != latest:
                    remap[uuid] = latest
        self._entity_remap_cache = remap
        self._entity_remap_cache_ts = now
        return remap

    def _remap_relation_endpoints(self, relations: list, session=None) -> list:
        if not relations:
            return relations
        own_session = session is None
        if own_session:
            with self._session() as session:
                return self._remap_relation_endpoints(relations, session=session)
        remap = self._build_entity_abs_id_remap(session)
        if remap:
            for rel in relations:
                rel.entity1_absolute_id = remap.get(rel.entity1_absolute_id, rel.entity1_absolute_id)
                rel.entity2_absolute_id = remap.get(rel.entity2_absolute_id, rel.entity2_absolute_id)
        return relations
