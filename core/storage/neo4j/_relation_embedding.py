"""Neo4j RelationEmbeddingMixin — embedding computation and cache for relations."""
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...models import Relation
from ...perf import _perf_timer
from ._helpers import (
    _RELATION_RETURN_FIELDS_WITH_EMB,
    _encode_and_normalize,
    _neo4j_record_to_relation,
)

logger = logging.getLogger(__name__)


class RelationEmbeddingMixin:
    """Relation embedding helpers: cache invalidation, embedding text construction,
    embedding computation, embedding cache management.

    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              -> Neo4j session factory
        self._run(session, cypher, **kw) -> execute Cypher with graph_id injection
        self._graph_id: str          -> active graph ID
        self._cache                  -> QueryCache
        self.embedding_client        -> EmbeddingClient (optional)
        self._entity_name_cache      -> dict[absolute_id -> name]
        self._entity_name_cache_lock -> threading.Lock
        self._relation_emb_cache     -> embedding cache list
        self._relation_emb_cache_ts  -> embedding cache timestamp
        self._emb_cache_ttl          -> cache TTL in seconds
        self.relation_content_snippet_length -> content snippet length
    """

    def _invalidate_relation_cache(self, family_id: str = None):
        """Scoped relation cache invalidation -- replaces broad pattern invalidation."""
        keys = ["graph_stats"]
        if family_id:
            keys.append(f"relation:by_fid:{family_id}")
        self._cache.invalidate_keys(keys)

    def _invalidate_relation_cache_bulk(self):
        """Bulk invalidation for operations affecting many/all relations."""
        self._cache.invalidate_keys(["graph_stats"])

    def _resolve_entity_names_for_embedding(self, relation: Relation,
                                             names: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
        """Resolve entity names for relation embedding text.

        Resolution order (fastest to slowest):
        1. Caller-supplied *names* dict (e.g. from batch lookup in remember pipeline)
        2. In-memory _entity_name_cache (populated by entity saves + prior lookups)
        3. Single Neo4j query for remaining cache misses
        """
        aid1, aid2 = relation.entity1_absolute_id, relation.entity2_absolute_id
        _enc = self._entity_name_cache

        # Seed cache from caller-supplied names (free -- no I/O)
        if names:
            _cache = self._cache_entity_name
            for k, v in names.items():
                if k not in _enc:
                    _cache(k, v)

        name1 = _enc.get(aid1, ...)
        name2 = _enc.get(aid2, ...)
        if name1 is not ... and name2 is not ...:
            return name1, name2

        # Cache miss -- single query for missing names
        try:
            with self._session() as session:
                result = self._run(session,
                    "MATCH (e:Entity) WHERE e.uuid IN [$aid1, $aid2] RETURN e.uuid AS aid, e.name AS name",
                    aid1=aid1, aid2=aid2,
                )
                for record in result:
                    aid, name = record["aid"], record["name"] or ""
                    self._cache_entity_name(aid, name)
                    if aid == aid1:
                        name1 = name
                    else:
                        name2 = name
        except Exception:
            if name1 is ...:
                name1 = ""
            if name2 is ...:
                name2 = ""
        return name1, name2

    def _build_relation_embedding_text(self, relation: Relation, entity1_name: str = "", entity2_name: str = "") -> str:
        """Build relation embedding text: Markdown format "# name1 -> name2\\ncontent"."""
        content = relation.content or ""
        if entity1_name and entity2_name:
            return f"# {entity1_name} → {entity2_name}\n{content}"
        elif entity1_name or entity2_name:
            return f"# {entity1_name or entity2_name}\n{content}"
        return content

    def _compute_relation_embedding(self, relation: Relation,
                                     names: Optional[Dict[str, str]] = None) -> Optional[bytes]:
        name1, name2 = self._resolve_entity_names_for_embedding(relation, names=names)
        text = self._build_relation_embedding_text(relation, name1, name2)
        result = _encode_and_normalize(self.embedding_client, text)
        return result[0] if result else None

    def _get_relations_by_entities_impl(self, from_family_id: str, to_family_id: str) -> List[Relation]:
        """Get all relations between two family_ids (actual implementation)."""
        from_family_id = self.resolve_family_id(from_family_id)
        to_family_id = self.resolve_family_id(to_family_id)
        if not from_family_id or not to_family_id:
            return []

        with self._session() as session:
            # Step 1: Batch fetch all absolute_ids for both family_ids
            result = self._run(session,
                """
                MATCH (e:Entity)
                WHERE e.family_id IN [$fid1, $fid2]
                WITH e.family_id AS fid, collect(e.uuid) AS abs_ids
                RETURN fid, abs_ids
                """,
                fid1=from_family_id,
                fid2=to_family_id,
            )
            fid_to_abs: Dict[str, List[str]] = {}
            for record in result:
                fid_to_abs[record["fid"]] = record["abs_ids"]

            from_ids = fid_to_abs.get(from_family_id, [])
            to_ids = fid_to_abs.get(to_family_id, [])
            if not from_ids or not to_ids:
                return []

            # Step 2: Query relations
            from ._helpers import _q
            result = self._run(session,
                _q("""
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id IN $from_ids AND r.entity2_absolute_id IN $to_ids)
                   OR (r.entity1_absolute_id IN $to_ids AND r.entity2_absolute_id IN $from_ids)
                WITH r.family_id AS fid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH fid, r ORDER BY r.processed_time DESC
                WITH fid, HEAD(COLLECT(r)) AS r
                RETURN __REL_FIELDS__
                ORDER BY r.processed_time DESC
                """),
                from_ids=from_ids,
                to_ids=to_ids,
            )
            return [_neo4j_record_to_relation(r) for r in result]

    def _update_relation_emb_cache(self, relation: Relation, emb_array: Optional[np.ndarray]):
        """Append-only update to relation embedding cache (O(1) via dict index)."""
        if self._relation_emb_cache is None:
            return
        if not hasattr(self, '_relation_emb_fid_idx') or self._relation_emb_fid_idx is None:
            self._relation_emb_fid_idx = {r.family_id: i for i, (r, _) in enumerate(self._relation_emb_cache)}
        idx = self._relation_emb_fid_idx.get(relation.family_id)
        if idx is not None:
            self._relation_emb_cache[idx] = (relation, emb_array)
        else:
            self._relation_emb_cache.append((relation, emb_array))
            self._relation_emb_fid_idx[relation.family_id] = len(self._relation_emb_cache) - 1

    def _get_relations_with_embeddings(self) -> List[tuple]:
        """Get all relations with embeddings (short TTL cache)."""
        now = time.time()
        if self._relation_emb_cache is not None and (now - self._relation_emb_cache_ts) < self._emb_cache_ttl:
            return self._relation_emb_cache
        with _perf_timer("_get_relations_with_embeddings"):
            result = self._get_relations_with_embeddings_impl()
        self._relation_emb_cache = result
        self._relation_emb_fid_idx = None  # Reset index; rebuilt lazily on next update
        self._relation_emb_cache_ts = time.time()
        return result

    def _get_relations_with_embeddings_impl(self) -> List[tuple]:
        """Get all relations with embeddings (actual implementation)."""
        with self._session() as session:
            limit = getattr(self, '_emb_cache_max_size', 10000)
            result = self._run(session,
                f"""
                MATCH (r:Relation)
                WITH r.family_id AS fid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH fid, r ORDER BY r.processed_time DESC
                WITH fid, HEAD(COLLECT(r)) AS r
                RETURN {_RELATION_RETURN_FIELDS_WITH_EMB}
                ORDER BY r.processed_time DESC
                LIMIT $limit
                """, limit=limit)
            records = list(result)

        if not records:
            return []

        relations = []
        for record in records:
            relation = _neo4j_record_to_relation(record)
            emb_array = np.frombuffer(relation.embedding, dtype=np.float32) if relation.embedding else None
            relations.append((relation, emb_array))
        return relations
