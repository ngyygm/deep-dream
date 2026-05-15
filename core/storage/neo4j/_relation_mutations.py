"""Neo4j RelationMutationMixin — write/mutation methods for relations."""
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from ...models import Relation
from ...perf import _perf_timer
from ._dream import _dream_source
from ._helpers import (
    _RELATION_RETURN_FIELDS,
    _fmt_dt,
    _neo4j_record_to_relation,
)

logger = logging.getLogger(__name__)


class RelationMutationMixin:
    """Relation write/mutation methods.

    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              -> Neo4j session factory
        self._run(session, cypher, **kw) -> execute Cypher with graph_id injection
        self._graph_id: str          -> active graph ID
        self._relation_write_lock    -> threading.Lock for relation writes
        self._cache                  -> QueryCache
        self.embedding_client        -> EmbeddingClient (optional)
        self._relation_emb_cache     -> embedding cache list
        self._relation_emb_fid_idx   -> family_id index for cache
    """

    def _save_relation_impl(self, relation: Relation,
                            names: Optional[Dict[str, str]] = None):
        """Save relation implementation.

        Computes embedding outside the write lock (CPU-bound, independent of DB),
        then acquires lock only for the Neo4j write operations.

        Args:
            relation: The relation to save.
            names: Optional dict[absolute_id -> entity_name] to avoid per-relation
                   Neo4j lookups for embedding text. When the caller already has
                   entity names (e.g. remember pipeline), pass them here.
        """
        valid_at = _fmt_dt(relation.valid_at or relation.event_time)

        # Phase 1: Compute embedding OUTSIDE the write lock (CPU-bound work)
        embedding_blob = self._compute_relation_embedding(relation, names=names)
        if embedding_blob is not None:
            relation.embedding = embedding_blob

        # Convert embedding bytes -> LIST<FLOAT> for Neo4j node property
        embedding_list = None
        if embedding_blob:
            emb_array_for_list = np.frombuffer(embedding_blob, dtype=np.float32)
            embedding_list = emb_array_for_list.tolist()

        # Phase 2: Acquire lock only for DB writes
        with self._relation_write_lock:
            with self._session() as session:
                params = {
                    "uuid": relation.absolute_id,
                    "family_id": relation.family_id,
                    "e1_abs": relation.entity1_absolute_id,
                    "e2_abs": relation.entity2_absolute_id,
                    "content": relation.content,
                    "event_time": _fmt_dt(relation.event_time),
                    "processed_time": _fmt_dt(relation.processed_time),
                    "cache_id": relation.episode_id,
                    "source": relation.source_document,
                    "summary": relation.summary,
                    "attributes": json.dumps(relation.attributes, ensure_ascii=False) if isinstance(relation.attributes, (dict, list)) else relation.attributes,
                    "confidence": relation.confidence,
                    "provenance": json.dumps(relation.provenance, ensure_ascii=False) if isinstance(relation.provenance, (dict, list)) else relation.provenance,
                    "content_format": getattr(relation, "content_format", "plain"),
                    "valid_at": valid_at,
                    "graph_id": self._graph_id,
                    "embedding": embedding_list,
                }

                # Single combined query: MERGE node + invalidate old + RELATES_TO edges
                self._run_with_retry(session,
                    """
                    MERGE (r:Relation {uuid: $uuid})
                    SET r:Concept, r.role = 'relation',
                        r.family_id = $family_id,
                        r.entity1_absolute_id = $e1_abs,
                        r.entity2_absolute_id = $e2_abs,
                        r.content = $content,
                        r.event_time = datetime($event_time),
                        r.processed_time = datetime($processed_time),
                        r.episode_id = $cache_id,
                        r.source_document = $source,
                        r.summary = $summary,
                        r.attributes = $attributes,
                        r.confidence = $confidence,
                        r.provenance = $provenance,
                        r.content_format = $content_format,
                        r.valid_at = datetime($valid_at),
                        r.graph_id = $graph_id,
                        r.embedding = $embedding
                    WITH r, r.entity1_absolute_id AS e1a, r.entity2_absolute_id AS e2a
                    MATCH (ref1:Entity {uuid: e1a})
                    MATCH (ref2:Entity {uuid: e2a})
                    SET r.entity1_family_id = ref1.family_id,
                        r.entity2_family_id = ref2.family_id
                    WITH r, ref1, ref2
                    OPTIONAL MATCH (old:Relation {family_id: $family_id})
                    WHERE old.uuid <> $uuid AND old.invalid_at IS NULL
                    SET old.invalid_at = datetime($event_time)
                    WITH r, ref1, ref2 WHERE r IS NOT NULL
                    MATCH (n1:Entity {family_id: ref1.family_id}) WHERE n1.invalid_at IS NULL
                    MATCH (n2:Entity {family_id: ref2.family_id}) WHERE n2.invalid_at IS NULL
                    MERGE (n1)-[rel:RELATES_TO {relation_uuid: $uuid}]->(n2)
                    SET rel.fact = $content
                    """,
                    operation_name="save_relation",
                    **params,
                )

        # Phase 3: Cache update
        emb_array = None
        if embedding_blob:
            emb_array = np.frombuffer(embedding_blob, dtype=np.float32)

        self._invalidate_relation_cache_bulk()
        return emb_array

    def batch_delete_relation_versions_by_absolute_ids(self, absolute_ids: List[str]) -> int:
        """Batch delete specific relation versions, return deleted count."""
        if not absolute_ids:
            return 0
        with self._relation_write_lock:
            with self._session() as session:
                result = self._run(session,
                    """
                    MATCH (r:Relation) WHERE r.uuid IN $aids
                    DETACH DELETE r
                    RETURN count(r) AS deleted
                    """,
                    aids=absolute_ids,
                )
                record = result.single()
                deleted = record["deleted"] if record else 0
            self._invalidate_relation_cache_bulk()
        return deleted

    def batch_delete_relations(self, family_ids: List[str]) -> int:
        """Batch delete relations -- single transaction replacing N individual deletes."""
        if not family_ids:
            return 0
        all_uuids = []
        count = 0
        with self._relation_write_lock:
            # Single session: collect UUIDs + delete in one transaction
            with self._session() as session:
                # Collect UUIDs before deleting
                result = self._run_with_retry(session,
                    "UNWIND $fids AS fid MATCH (r:Relation {family_id: fid}) RETURN r.uuid AS uuid",
                    fids=family_ids,
                )
                all_uuids = [r["uuid"] for r in result]
                # Delete in the same session
                result = self._run_with_retry(session,
                    "UNWIND $fids AS fid MATCH (r:Relation {family_id: fid}) DETACH DELETE r RETURN count(r) AS cnt",
                    fids=family_ids,
                )
                record = result.single()
                count = record["cnt"] if record else 0
            self._invalidate_relation_cache_bulk()
        return count

    def batch_get_relations_referencing_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, List[Relation]]:
        """Batch get relations referencing given entity absolute_ids (eliminates N+1 queries)."""
        if not absolute_ids:
            return {}
        with self._session() as session:
            from ._helpers import _q
            result = self._run(session, _q("""
                MATCH (r:Relation)
                WHERE r.entity1_absolute_id IN $aids OR r.entity2_absolute_id IN $aids
                RETURN __REL_FIELDS__
                """),
                aids=absolute_ids,
            )
            result_map: Dict[str, List[Relation]] = {aid: [] for aid in absolute_ids}
            for record in result:
                rel = _neo4j_record_to_relation(record)
                if rel.entity1_absolute_id in result_map:
                    result_map[rel.entity1_absolute_id].append(rel)
                if rel.entity2_absolute_id in result_map:
                    result_map[rel.entity2_absolute_id].append(rel)
            return result_map

    def bulk_save_relations(self, relations: List[Relation]):
        """Batch save relations (UNWIND bulk write).

        Writes metadata first (no embedding), embedding computed in background thread.
        """
        if not relations:
            return

        # --- Phase 1: Fast Neo4j write (no embedding) ---
        rows = []
        for relation in relations:
            rows.append({
                "uuid": relation.absolute_id,
                "family_id": relation.family_id,
                "e1_abs": relation.entity1_absolute_id,
                "e2_abs": relation.entity2_absolute_id,
                "content": relation.content,
                "event_time": _fmt_dt(relation.event_time),
                "processed_time": _fmt_dt(relation.processed_time),
                "cache_id": relation.episode_id,
                "source": relation.source_document,
                "summary": getattr(relation, 'summary', None),
                "attributes": json.dumps(_attrs, ensure_ascii=False) if isinstance(_attrs := getattr(relation, 'attributes', None), (dict, list)) else _attrs,
                "confidence": getattr(relation, 'confidence', None),
                "provenance": json.dumps(_prov, ensure_ascii=False) if isinstance(_prov := getattr(relation, 'provenance', None), (dict, list)) else _prov,
                "content_format": getattr(relation, 'content_format', None),
                "valid_at": _fmt_dt(relation.valid_at or relation.event_time) if relation.valid_at or relation.event_time else None,
                "graph_id": self._graph_id,
            })

        with self._relation_write_lock:
            with self._session() as session:
                self._run_with_retry(session,
                    """
                    UNWIND $rows AS row
                    MERGE (r:Relation {uuid: row.uuid})
                    SET r:Concept, r.role = 'relation',
                        r.family_id = row.family_id,
                        r.entity1_absolute_id = row.e1_abs,
                        r.entity2_absolute_id = row.e2_abs,
                        r.content = row.content,
                        r.event_time = datetime(row.event_time),
                        r.processed_time = datetime(row.processed_time),
                        r.episode_id = row.cache_id,
                        r.source_document = row.source,
                        r.summary = row.summary,
                        r.attributes = row.attributes,
                        r.confidence = row.confidence,
                        r.provenance = row.provenance,
                        r.content_format = row.content_format,
                        r.valid_at = CASE WHEN row.valid_at IS NOT NULL THEN datetime(row.valid_at) ELSE NULL END,
                        r.graph_id = row.graph_id
                    WITH r, row
                    MATCH (ref1:Entity {uuid: row.e1_abs})
                    MATCH (ref2:Entity {uuid: row.e2_abs})
                    SET r.entity1_family_id = ref1.family_id,
                        r.entity2_family_id = ref2.family_id
                    WITH row, ref1, ref2
                    OPTIONAL MATCH (r:Relation {family_id: row.family_id})
                    WHERE r.uuid <> row.uuid AND r.invalid_at IS NULL
                    SET r.invalid_at = datetime(row.event_time)
                    WITH row, ref1, ref2 WHERE row IS NOT NULL
                    MATCH (n1:Entity {family_id: ref1.family_id}) WHERE n1.invalid_at IS NULL
                    MATCH (n2:Entity {family_id: ref2.family_id}) WHERE n2.invalid_at IS NULL
                    MERGE (n1)-[rel:RELATES_TO {relation_uuid: row.uuid}]->(n2)
                    SET rel.fact = row.content
                    """,
                    operation_name="bulk_save_relations",
                    rows=rows,
                )

        # --- Phase 2: Background thread for embedding computation + Neo4j update ---
        if self.embedding_client and self.embedding_client.is_available():
            threading.Thread(
                target=self._bulk_save_relation_embedding_bg,
                args=(list(relations),),
                daemon=True,
            ).start()

    def _bulk_save_relation_embedding_bg(self, relations: List[Relation]):
        """Background computation of relation embeddings and Neo4j update."""
        try:
            # Batch resolve entity names
            entity_names = {}
            all_abs_ids = set()
            for r in relations:
                if r.entity1_absolute_id:
                    all_abs_ids.add(r.entity1_absolute_id)
                if r.entity2_absolute_id:
                    all_abs_ids.add(r.entity2_absolute_id)
            if all_abs_ids:
                try:
                    with self._session() as session:
                        result = self._run(session,
                            "MATCH (e:Entity) WHERE e.uuid IN $aids RETURN e.uuid AS aid, e.name AS name",
                            aids=list(all_abs_ids),
                        )
                        for rec in result:
                            entity_names[rec["aid"]] = rec["name"] or ""
                except Exception:
                    pass

            texts = [
                self._build_relation_embedding_text(
                    r,
                    entity_names.get(r.entity1_absolute_id, ""),
                    entity_names.get(r.entity2_absolute_id, ""),
                )
                for r in relations
            ]
            embeddings = self.embedding_client.encode(texts)

            cache_items = []
            emb_rows = []
            for idx, relation in enumerate(relations):
                try:
                    emb_array = np.array(embeddings[idx], dtype=np.float32)
                    norm = np.linalg.norm(emb_array)
                    if norm > 0:
                        emb_array = emb_array / norm
                    relation.embedding = emb_array.tobytes()
                    embedding_list = emb_array.tolist()
                except Exception:
                    continue
                emb_rows.append({"uuid": relation.absolute_id, "embedding": embedding_list})
                cache_items.append((relation, emb_array))

            if emb_rows:
                with self._session() as session:
                    self._run_with_retry(session,
                        """
                        UNWIND $rows AS row
                        MATCH (r:Relation {uuid: row.uuid})
                        SET r.embedding = row.embedding
                        """,
                        operation_name="bulk_save_rel_emb_update",
                        rows=emb_rows,
                    )

            if self._relation_emb_cache is not None and cache_items:
                if self._relation_emb_fid_idx is not None:
                    fid_to_idx = self._relation_emb_fid_idx
                else:
                    fid_to_idx = {r.family_id: i for i, (r, _) in enumerate(self._relation_emb_cache)}
                    self._relation_emb_fid_idx = fid_to_idx
                for relation, emb_array in cache_items:
                    idx = fid_to_idx.get(relation.family_id)
                    if idx is not None:
                        self._relation_emb_cache[idx] = (relation, emb_array)
                    else:
                        self._relation_emb_cache.append((relation, emb_array))
                        fid_to_idx[relation.family_id] = len(self._relation_emb_cache) - 1
        except Exception:
            logger.debug("Background relation embedding update failed", exc_info=True)

            self._invalidate_relation_cache_bulk()

    def bulk_save_relations_with_embedding(self, relations: List[Relation]):
        """Batch save relations with pre-computed embeddings (UNWIND).

        Unlike bulk_save_relations: embedding is written immediately, not deferred
        to a background thread. Used when embeddings are needed immediately for
        subsequent queries.
        """
        if not relations:
            return

        rows = []
        cache_items = []
        for relation in relations:
            emb_blob = getattr(relation, 'embedding', None)
            embedding_list = None
            emb_array = None
            if emb_blob is not None:
                if isinstance(emb_blob, np.ndarray):
                    emb_array = emb_blob
                else:
                    emb_array = np.frombuffer(emb_blob, dtype=np.float32)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_array = emb_array / norm
                embedding_list = emb_array.tolist()
                relation.embedding = emb_array.tobytes()
                cache_items.append((relation, emb_array))
            else:
                cache_items.append((relation, None))

            _attrs = getattr(relation, 'attributes', None)
            _prov = getattr(relation, 'provenance', None)
            rows.append({
                "uuid": relation.absolute_id,
                "family_id": relation.family_id,
                "e1_abs": relation.entity1_absolute_id,
                "e2_abs": relation.entity2_absolute_id,
                "content": relation.content,
                "event_time": _fmt_dt(relation.event_time),
                "processed_time": _fmt_dt(relation.processed_time),
                "cache_id": relation.episode_id,
                "source": relation.source_document,
                "summary": getattr(relation, 'summary', None),
                "attributes": json.dumps(_attrs, ensure_ascii=False) if isinstance(_attrs, (dict, list)) else _attrs,
                "confidence": getattr(relation, 'confidence', None),
                "provenance": json.dumps(_prov, ensure_ascii=False) if isinstance(_prov, (dict, list)) else _prov,
                "content_format": getattr(relation, 'content_format', None),
                "valid_at": _fmt_dt(relation.valid_at or relation.event_time) if relation.valid_at or relation.event_time else None,
                "graph_id": self._graph_id,
                "embedding": embedding_list,
            })

        with self._relation_write_lock:
            with self._session() as session:
                self._run_with_retry(session,
                    """
                    UNWIND $rows AS row
                    MERGE (r:Relation {uuid: row.uuid})
                    SET r:Concept, r.role = 'relation',
                        r.family_id = row.family_id,
                        r.entity1_absolute_id = row.e1_abs,
                        r.entity2_absolute_id = row.e2_abs,
                        r.content = row.content,
                        r.event_time = datetime(row.event_time),
                        r.processed_time = datetime(row.processed_time),
                        r.episode_id = row.cache_id,
                        r.source_document = row.source,
                        r.summary = row.summary,
                        r.attributes = row.attributes,
                        r.confidence = row.confidence,
                        r.provenance = row.provenance,
                        r.content_format = row.content_format,
                        r.valid_at = CASE WHEN row.valid_at IS NOT NULL THEN datetime(row.valid_at) ELSE NULL END,
                        r.graph_id = row.graph_id,
                        r.embedding = row.embedding
                    WITH r, row
                    MATCH (ref1:Entity {uuid: row.e1_abs})
                    MATCH (ref2:Entity {uuid: row.e2_abs})
                    SET r.entity1_family_id = ref1.family_id,
                        r.entity2_family_id = ref2.family_id
                    WITH row, ref1, ref2
                    OPTIONAL MATCH (r:Relation {family_id: row.family_id})
                    WHERE r.uuid <> row.uuid AND r.invalid_at IS NULL
                    SET r.invalid_at = datetime(row.event_time)
                    WITH row, ref1, ref2 WHERE row IS NOT NULL
                    MATCH (n1:Entity {family_id: ref1.family_id}) WHERE n1.invalid_at IS NULL
                    MATCH (n2:Entity {family_id: ref2.family_id}) WHERE n2.invalid_at IS NULL
                    MERGE (n1)-[rel:RELATES_TO {relation_uuid: row.uuid}]->(n2)
                    SET rel.fact = row.content
                    """,
                    operation_name="bulk_save_rels_with_emb",
                    rows=rows,
                )

        if self._relation_emb_cache is not None and cache_items:
            if self._relation_emb_fid_idx is not None:
                fid_to_idx = self._relation_emb_fid_idx
            else:
                fid_to_idx = {r.family_id: i for i, (r, _) in enumerate(self._relation_emb_cache)} if self._relation_emb_cache else {}
                self._relation_emb_fid_idx = fid_to_idx
            for relation, emb_array in cache_items:
                idx = fid_to_idx.get(relation.family_id)
                if idx is not None:
                    self._relation_emb_cache[idx] = (relation, emb_array)
                else:
                    self._relation_emb_cache.append((relation, emb_array))
                    fid_to_idx[relation.family_id] = len(self._relation_emb_cache) - 1
        self._invalidate_relation_cache_bulk()

    def delete_relation_all_versions(self, family_id: str) -> int:
        """Delete all versions of a relation. Returns deleted row count."""
        return self.delete_relation_by_id(family_id)

    def delete_relation_by_absolute_id(self, absolute_id: str) -> bool:
        """Delete relation by absolute_id, return whether deletion succeeded."""
        with self._relation_write_lock:
            with self._session() as session:
                result = self._run(session,
                    "MATCH (r:Relation {uuid: $aid}) DETACH DELETE r RETURN count(r) AS cnt",
                    aid=absolute_id,
                )
                record = result.single()
                deleted = record is not None and record["cnt"] > 0
            self._invalidate_relation_cache_bulk()
        return deleted

    def delete_relation_by_id(self, family_id: str) -> int:
        """Delete all versions of a relation. Returns deleted row count."""
        abs_ids = []
        count = 0
        with self._relation_write_lock:
            with self._session() as session:
                # Collect absolute_ids first (unavailable after DETACH DELETE) -- lightweight UUID-only query
                result = self._run_with_retry(session,
                    "MATCH (r:Relation {family_id: $fid}) RETURN r.uuid AS uuid",
                    fid=family_id,
                )
                abs_ids = [r["uuid"] for r in result]
                # Delete relation nodes
                result = self._run_with_retry(session,
                    "MATCH (r:Relation {family_id: $fid}) DETACH DELETE r RETURN count(r) AS cnt",
                    fid=family_id,
                )
                record = result.single()
                count = record["cnt"] if record else 0
            self._invalidate_relation_cache_bulk()
        return count

    def invalidate_relation(self, family_id: str, reason: str = "") -> int:
        """Mark relation as invalidated."""
        now = datetime.now(timezone.utc).isoformat()
        with self._session() as session:
            result = self._run(session, """
                MATCH (r:Relation {family_id: $family_id})
                WHERE r.invalid_at IS NULL
                SET r.invalid_at = $now
                RETURN count(r) AS cnt
            """, family_id=family_id, now=now)
            record = result.single()
            return record["cnt"] if record else 0

    def redirect_relation(self, family_id: str, side: str, new_family_id: str) -> int:
        """Redirect all relations for a family_id on one side to a new entity family_id.

        Args:
            family_id: Family ID of the relations to redirect.
            side: "entity1" or "entity2".
            new_family_id: New target entity family_id.

        Returns:
            Number of updated relations.
        """
        if side not in ("entity1", "entity2"):
            raise ValueError(f"side must be 'entity1' or 'entity2', got '{side}'")

        side_field = f"{side}_absolute_id"

        with self._relation_write_lock:
            with self._session() as session:
                # 1. Get latest absolute_id for new_family_id
                target_result = self._run(session,
                    """
                    MATCH (e:Entity {family_id: $fid})
                    RETURN e.uuid AS uuid
                    ORDER BY e.processed_time DESC LIMIT 1
                    """,
                    fid=new_family_id,
                )
                target_record = target_result.single()
                if not target_record:
                    return 0
                new_abs_id = target_record["uuid"]

                # 2. Update all matching relations
                update_result = self._run(session,
                    f"MATCH (r:Relation {{family_id: $fid}}) "
                    f"SET r.{side_field} = $new_abs_id "
                    f"RETURN count(r) AS cnt",
                    fid=family_id,
                    new_abs_id=new_abs_id,
                )
                update_record = update_result.single()
                count = update_record["cnt"] if update_record else 0
            self._invalidate_relation_cache_bulk()
            return count

    def refresh_relates_to_edges(self, family_ids: List[str] = None):
        """Rebuild RELATES_TO edges that point to invalidated entity versions.

        Args:
            family_ids: If provided, only refresh edges for relations involving
                these entity family_ids (incremental). If None, full refresh.
        """
        with self._session() as session:
            if family_ids:
                # Incremental: only refresh edges involving specified entity families
                result = self._run(session, """
                    MATCH (rel:Relation) WHERE rel.invalid_at IS NULL
                    MATCH (ref1:Entity {uuid: rel.entity1_absolute_id})
                    WHERE ref1.family_id IN $fids
                    MATCH (ref2:Entity {uuid: rel.entity2_absolute_id})
                    // Delete stale edges for these relations
                    WITH rel, ref1, ref2
                    OPTIONAL MATCH (a:Entity)-[r:RELATES_TO {relation_uuid: rel.uuid}]->(b:Entity)
                    WHERE a.invalid_at IS NOT NULL OR b.invalid_at IS NOT NULL
                    DELETE r
                    WITH DISTINCT rel, ref1, ref2
                    // Recreate edges pointing to current versions
                    MATCH (cur1:Entity {family_id: ref1.family_id}) WHERE cur1.invalid_at IS NULL
                    MATCH (cur2:Entity {family_id: ref2.family_id}) WHERE cur2.invalid_at IS NULL
                    MERGE (cur1)-[r:RELATES_TO {relation_uuid: rel.uuid}]->(cur2)
                    SET r.fact = rel.content
                    RETURN count(r) AS refreshed
                """, fids=family_ids)
                refreshed = result.single()["refreshed"]
                if refreshed > 0:
                    logger.info("refresh_relates_to_edges: incremental refresh for %d families, %d edges", len(family_ids), refreshed)
                return {"refreshed": refreshed}
            else:
                # Full refresh -- combined into single Cypher call
                result = self._run(session, """
                    // Step 1: Delete stale RELATES_TO edges
                    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                    WHERE a.invalid_at IS NOT NULL OR b.invalid_at IS NOT NULL
                    DELETE r
                    WITH count(r) AS deleted
                    // Step 2: Recreate edges pointing to current versions
                    MATCH (rel:Relation) WHERE rel.invalid_at IS NULL
                    MATCH (ref1:Entity {uuid: rel.entity1_absolute_id})
                    MATCH (cur1:Entity {family_id: ref1.family_id})
                    WHERE cur1.invalid_at IS NULL
                    MATCH (ref2:Entity {uuid: rel.entity2_absolute_id})
                    MATCH (cur2:Entity {family_id: ref2.family_id})
                    WHERE cur2.invalid_at IS NULL
                    MERGE (cur1)-[r:RELATES_TO {relation_uuid: rel.uuid}]->(cur2)
                    SET r.fact = rel.content
                    RETURN deleted, count(r) AS created
                """)
                row = result.single()
                deleted = row["deleted"]
                created = row["created"]
                if deleted > 0 or created > 0:
                    logger.info("refresh_relates_to_edges: deleted=%d stale, created=%d new", deleted, created)
                return {"deleted": deleted, "created": created}

    def save_dream_relation(self, entity1_id: str, entity2_id: str,
                            content: str, confidence: float, reasoning: str,
                            dream_cycle_id: Optional[str] = None,
                            episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Create or merge a dream-discovered relation.

        Blueprint line 147: Dream relations start as candidates (tier=candidate,
        status=hypothesized, confidence capped at 0.5).

        Returns: {"family_id": "...", "entity1_family_id": "...", "entity2_family_id": "...", "action": "created"|"merged"}
        Raises: ValueError if entities do not exist
        """
        # Resolve entities (batch)
        resolved_map = self.resolve_family_ids([entity1_id, entity2_id])
        resolved1 = resolved_map.get(entity1_id, entity1_id)
        resolved2 = resolved_map.get(entity2_id, entity2_id)
        if not resolved1:
            raise ValueError(f"Entity not found: {entity1_id}")
        if not resolved2:
            raise ValueError(f"Entity not found: {entity2_id}")

        entities_map = self.get_entities_by_family_ids([resolved1, resolved2])
        entity1 = entities_map.get(resolved1)
        entity2 = entities_map.get(resolved2)
        if not entity1:
            raise ValueError(f"Entity not found: {entity1_id}")
        if not entity2:
            raise ValueError(f"Entity not found: {entity2_id}")

        # Check existing relation (include candidates so we can merge with them)
        existing = self.get_relations_by_entities(resolved1, resolved2, include_candidates=True)
        if existing:
            latest = existing[0]
            # Merge: take higher confidence, append reasoning
            new_confidence = max(latest.confidence or 0, confidence)
            # Build new provenance entry
            new_prov_entry = {
                "source": "dream",
                "dream_cycle_id": dream_cycle_id,
                "confidence": confidence,
                "reasoning": reasoning,
            }
            try:
                old_prov = json.loads(latest.provenance) if latest.provenance else []
            except Exception as _prov_err:
                logger.warning("provenance JSON parse failed, discarding old history: %s", _prov_err)
                old_prov = []
            old_prov.append(new_prov_entry)

            # Create new version (keep same family_id)
            now = datetime.now()
            record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            source_doc = _dream_source(dream_cycle_id)
            merged_content = f"{latest.content}\n[Dream update] {content}" if content != latest.content else latest.content

            # Preserve existing attributes (tier, status, corroboration state)
            try:
                merged_attrs = json.loads(latest.attributes) if latest.attributes else {}
            except (json.JSONDecodeError, TypeError):
                merged_attrs = {}
            # Track additional dream cycle
            if dream_cycle_id:
                merged_attrs.setdefault("additional_dream_cycles", [])
                merged_attrs["additional_dream_cycles"].append(dream_cycle_id)

            relation = Relation(
                absolute_id=record_id,
                family_id=latest.family_id,
                entity1_absolute_id=latest.entity1_absolute_id,
                entity2_absolute_id=latest.entity2_absolute_id,
                content=merged_content,
                event_time=now,
                processed_time=now,
                episode_id=episode_id or latest.episode_id or "",
                source_document=source_doc,
                confidence=new_confidence,
                provenance=json.dumps(old_prov, ensure_ascii=False),
                attributes=json.dumps(merged_attrs) if merged_attrs else latest.attributes,
            )
            self.save_relation(relation)
            return {
                "family_id": latest.family_id,
                "entity1_family_id": resolved1,
                "entity2_family_id": resolved2,
                "entity1_name": entity1.name,
                "entity2_name": entity2.name,
                "action": "merged",
            }

        # Sort so (A,B) and (B,A) are treated as same relation
        if entity1.name <= entity2.name:
            e1_abs, e2_abs = entity1.absolute_id, entity2.absolute_id
        else:
            e1_abs, e2_abs = entity2.absolute_id, entity1.absolute_id

        now = datetime.now()
        family_id = f"rel_{uuid.uuid4().hex[:12]}"
        record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        source_doc = _dream_source(dream_cycle_id)
        provenance_data = {
            "source": "dream",
            "dream_cycle_id": dream_cycle_id,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        relation = Relation(
            absolute_id=record_id,
            family_id=family_id,
            entity1_absolute_id=e1_abs,
            entity2_absolute_id=e2_abs,
            content=content,
            event_time=now,
            processed_time=now,
            episode_id=episode_id or "",
            source_document=source_doc,
            confidence=min(confidence, 0.5),  # Blueprint: cap at 0.5 for new candidates
            provenance=json.dumps([provenance_data], ensure_ascii=False),
            attributes=json.dumps({
                "tier": "candidate",
                "status": "hypothesized",
                "corroboration_count": 0,
                "created_by_dream": dream_cycle_id or "unknown",
                "created_at": now.isoformat(),
            }),
        )

        self.save_relation(relation)

        return {
            "family_id": family_id,
            "entity1_family_id": resolved1,
            "entity2_family_id": resolved2,
            "entity1_name": entity1.name,
            "entity2_name": entity2.name,
            "action": "created",
        }

    def save_relation(self, relation: Relation):
        """Save relation to Neo4j (merged into single Cypher)."""
        with _perf_timer("save_relation"):
            emb_array = self._save_relation_impl(relation)
            # Incremental relation emb cache update (reuse array from _save_relation_impl)
            if emb_array is not None:
                self._update_relation_emb_cache(relation, emb_array)

    def update_entity_names_in_relations(self, family_id: str, new_name: str) -> int:
        """Update entity1_name/entity2_name in all relations referencing this family_id."""
        with self._session() as session:
            r1 = self._run(session,
                "MATCH (r:Relation) WHERE r.entity1_family_id = $fid AND r.invalid_at IS NULL "
                "SET r.entity1_name = $name RETURN count(r) AS c",
                fid=family_id, name=new_name,
            )
            c1 = (r1.single() or {}).get("c", 0)
            r2 = self._run(session,
                "MATCH (r:Relation) WHERE r.entity2_family_id = $fid AND r.invalid_at IS NULL "
                "SET r.entity2_name = $name RETURN count(r) AS c",
                fid=family_id, name=new_name,
            )
            c2 = (r2.single() or {}).get("c", 0)
            return c1 + c2

    def fix_dangling_relation_refs(self, dry_run: bool = False) -> dict:
        """Fix relations whose entity1/2_absolute_id points to a non-existent entity.
        Rewrites stale UUIDs to the latest valid UUID for the same family_id.
        """
        with self._session() as session:
            # Find all distinct absolute_ids referenced by valid relations
            r = self._run(session, """
                MATCH (rel:Relation) WHERE rel.invalid_at IS NULL
                WITH collect(DISTINCT rel.entity1_absolute_id) + collect(DISTINCT rel.entity2_absolute_id) AS all_aids
                UNWIND all_aids AS aid
                RETURN collect(DISTINCT aid) AS rel_aids
            """)
            row = r.single()
            rel_aids = set(row["rel_aids"] or ())

            # Find all valid entity UUIDs
            r2 = self._run(session, """
                MATCH (e:Entity) WHERE e.invalid_at IS NULL
                RETURN collect(DISTINCT e.uuid) AS ent_uuids
            """)
            row2 = r2.single()
            valid_uuids = set(row2["ent_uuids"] or ())

            dangling = rel_aids - valid_uuids
            if not dangling:
                return {"fixed": 0, "dangling_found": 0, "dry_run": dry_run}

            if dry_run:
                return {"fixed": 0, "dangling_found": len(dangling), "dry_run": True}

            # For each dangling UUID, find the family_id and latest valid UUID
            dangling_list = list(dangling)
            r3 = self._run(session, """
                UNWIND $dangling AS aid
                MATCH (e:Entity {uuid: aid})
                RETURN e.uuid AS old_uuid, e.family_id AS fid
            """, dangling=dangling_list)
            old_to_fid = {rec["old_uuid"]: rec["fid"] for rec in r3 if rec.get("fid")}

            # Also look up family_ids from Relation nodes for fully deleted entities
            if len(old_to_fid) < len(dangling):
                missing_uuids = [u for u in dangling_list if u not in old_to_fid]
                r3b = self._run(session, """
                    UNWIND $missing AS aid
                    MATCH (r:Relation) WHERE r.entity1_absolute_id = aid AND r.invalid_at IS NULL
                    RETURN aid AS old_uuid, r.entity1_family_id AS fid
                """, missing=missing_uuids[:500])
                for rec in r3b:
                    if rec.get("fid") and rec["old_uuid"] not in old_to_fid:
                        old_to_fid[rec["old_uuid"]] = rec["fid"]
                r3c = self._run(session, """
                    UNWIND $missing AS aid
                    MATCH (r:Relation) WHERE r.entity2_absolute_id = aid AND r.invalid_at IS NULL
                    RETURN aid AS old_uuid, r.entity2_family_id AS fid
                """, missing=missing_uuids[:500])
                for rec in r3c:
                    if rec.get("fid") and rec["old_uuid"] not in old_to_fid:
                        old_to_fid[rec["old_uuid"]] = rec["fid"]

            if not old_to_fid:
                return {"fixed": 0, "dangling_found": len(dangling), "dry_run": False}

            # Get latest valid UUID for each family_id
            fids = list(set(old_to_fid.values()))
            r4 = self._run(session, """
                UNWIND $fids AS fid
                MATCH (e:Entity {family_id: fid})
                WHERE e.invalid_at IS NULL
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(collect(e.uuid)) AS latest_uuid
                RETURN fid, latest_uuid
            """, fids=fids)
            fid_to_latest = {rec["fid"]: rec["latest_uuid"] for rec in r4 if rec.get("latest_uuid")}

            # Build old_uuid -> latest_uuid mapping
            remap = {}
            for old_uuid, fid in old_to_fid.items():
                latest = fid_to_latest.get(fid)
                if latest and latest != old_uuid:
                    remap[old_uuid] = latest

            if not remap:
                return {"fixed": 0, "dangling_found": len(dangling), "dry_run": False, "remappable": 0}

            # Apply fixes
            fixed1 = 0
            fixed2 = 0
            for old_uuid, new_uuid in remap.items():
                r5 = self._run(session,
                    "MATCH (r:Relation) WHERE r.entity1_absolute_id = $old AND r.invalid_at IS NULL "
                    "SET r.entity1_absolute_id = $new RETURN count(r) AS c",
                    old=old_uuid, new=new_uuid,
                )
                fixed1 += (r5.single() or {}).get("c", 0)
                r6 = self._run(session,
                    "MATCH (r:Relation) WHERE r.entity2_absolute_id = $old AND r.invalid_at IS NULL "
                    "SET r.entity2_absolute_id = $new RETURN count(r) AS c",
                    old=old_uuid, new=new_uuid,
                )
                fixed2 += (r6.single() or {}).get("c", 0)

            return {
                "fixed": fixed1 + fixed2,
                "dangling_found": len(dangling),
                "remapped_uuids": len(remap),
                "dry_run": False,
            }

    def update_relation_by_absolute_id(self, absolute_id: str, **fields) -> Optional[Relation]:
        """Update specified fields by absolute_id, return updated Relation or None.

        When content changes, automatically recomputes embedding.
        Embedding computed BEFORE write lock; vector store I/O AFTER lock.
        """
        valid_keys = {"content", "summary", "attributes", "confidence"}
        filtered = {k: v for k, v in fields.items() if k in valid_keys and v is not None}
        if not filtered:
            return None

        needs_emb_update = "content" in filtered

        # Phase 1: Pre-compute embedding BEFORE write lock (ML inference is slow)
        _precomputed_emb = None
        if needs_emb_update and self.embedding_client and self.embedding_client.is_available():
            current = self.get_relation_by_absolute_id(absolute_id)
            if current:
                merged = Relation(
                    name="",
                    content=filtered.get("content", current.content),
                    entity1_absolute_id=current.entity1_absolute_id,
                    entity2_absolute_id=current.entity2_absolute_id,
                )
                _emb_result = self._compute_relation_embedding(merged)
                if _emb_result is not None:
                    _precomputed_emb = _emb_result

        # Convert embedding bytes -> LIST<FLOAT> for Neo4j
        embedding_list = None
        if _precomputed_emb is not None:
            emb_array_for_list = np.frombuffer(_precomputed_emb, dtype=np.float32)
            embedding_list = emb_array_for_list.tolist()

        # Phase 2: Acquire lock only for Neo4j write
        with self._relation_write_lock:
            with self._session() as session:
                set_parts = [f"r.{k} = ${k}" for k in filtered]
                params = {**filtered, "aid": absolute_id}
                if _precomputed_emb is not None:
                    set_parts.append("r.embedding = $embedding")
                    params["embedding"] = embedding_list
                set_clauses = ", ".join(set_parts)
                cypher = (
                    f"MATCH (r:Relation {{uuid: $aid}}) "
                    f"SET {set_clauses} "
                    f"RETURN {_RELATION_RETURN_FIELDS}"
                )
                result = self._run(session, cypher, **params)
                record = result.single()
                if not record:
                    return None
                relation = _neo4j_record_to_relation(record)

            if _precomputed_emb is not None:
                relation.embedding = _precomputed_emb
            self._invalidate_relation_cache_bulk()

        # Phase 3: Cache update
        if _precomputed_emb is not None:
            emb_array = np.frombuffer(_precomputed_emb, dtype=np.float32)
            self._update_relation_emb_cache(relation, emb_array)
        elif needs_emb_update:
            self._update_relation_emb_cache(relation, None)

        return relation

    def update_relation_confidence(self, family_id: str, confidence: float):
        """Update confidence for the latest relation version. Range [0.0, 1.0]."""
        confidence = max(0.0, min(1.0, confidence))
        with self._session() as session:
            self._run(session, """
                MATCH (r:Relation {family_id: $fid})
                WHERE r.invalid_at IS NULL
                WITH r ORDER BY r.processed_time DESC LIMIT 1
                SET r.confidence = $confidence
            """, fid=family_id, confidence=confidence)
        self._invalidate_relation_cache_bulk()
