"""Neo4j RelationStoreMixin — extracted from neo4j_store."""
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...models import Entity, Relation
from ...perf import _perf_timer
from ._helpers import _RELATION_RETURN_FIELDS, _RELATION_RETURN_FIELDS_WITH_EMB, _expand_cypher, _fmt_dt, _neo4j_record_to_relation, _q
from ._dream import _dream_source

logger = logging.getLogger(__name__)


class RelationStoreMixin:
    """RelationStore operations for Neo4j backend.
    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              → Neo4j session factory
        self._run(session, cypher, **kw) → execute Cypher with graph_id injection
        self._graph_id: str          → active graph ID
        self._relation_write_lock    → threading.Lock for relation writes
        self._cache                  → QueryCache
        self.embedding_client        → EmbeddingClient (optional)
        self._entity_name_cache      → dict[absolute_id → name] for embedding text
        self._relation_emb_cache     → embedding cache list
        self._relation_emb_cache_ts  → embedding cache timestamp
        self._emb_cache_ttl          → cache TTL in seconds
        self.relation_content_snippet_length → content snippet length
    """

    def _invalidate_relation_cache(self, family_id: str = None):
        """Scoped relation cache invalidation — replaces broad pattern invalidation."""
        keys = ["graph_stats"]
        if family_id:
            keys.append(f"relation:by_fid:{family_id}")
        self._cache.invalidate_keys(keys)

    def _invalidate_relation_cache_bulk(self):
        """Bulk invalidation for operations affecting many/all relations."""
        self._cache.invalidate_keys(["graph_stats"])


    def _resolve_entity_names_for_embedding(self, relation: Relation,
                                             names: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
        """解析关系两端实体名称，用于 embedding 编码。

        Resolution order (fastest to slowest):
        1. Caller-supplied *names* dict (e.g. from batch lookup in remember pipeline)
        2. In-memory _entity_name_cache (populated by entity saves + prior lookups)
        3. Single Neo4j query for remaining cache misses
        """
        aid1, aid2 = relation.entity1_absolute_id, relation.entity2_absolute_id
        _enc = self._entity_name_cache

        # Seed cache from caller-supplied names (free — no I/O)
        if names:
            for k, v in names.items():
                if k not in _enc:
                    _enc[k] = v

        name1 = _enc.get(aid1, ...)
        name2 = _enc.get(aid2, ...)
        if name1 is not ... and name2 is not ...:
            return name1, name2

        # Cache miss — single query for missing names
        try:
            with self._session() as session:
                result = self._run(session,
                    "MATCH (e:Entity) WHERE e.uuid IN [$aid1, $aid2] RETURN e.uuid AS aid, e.name AS name",
                    aid1=aid1, aid2=aid2,
                )
                for record in result:
                    aid, name = record["aid"], record["name"] or ""
                    _enc[aid] = name
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
        """构建关系 embedding 文本：Markdown 格式 "# name1 → name2\\ncontent"。"""
        content = relation.content or ""
        if entity1_name and entity2_name:
            return f"# {entity1_name} → {entity2_name}\n{content}"
        elif entity1_name or entity2_name:
            return f"# {entity1_name or entity2_name}\n{content}"
        return content

    def _compute_relation_embedding(self, relation: Relation,
                                     names: Optional[Dict[str, str]] = None) -> Optional[bytes]:
        """计算关系的 embedding 向量（L2 归一化后存储）。

        编码 "{entity1_name} {content} {entity2_name}"，让不同实体间的同语义关系产生不同向量。

        Args:
            relation: The relation to compute embedding for.
            names: Optional dict[absolute_id -> entity_name]. When the caller
                   already has entity names (e.g. from a batch lookup), passing
                   them here avoids a separate Neo4j session per relation.
        """
        if not self.embedding_client or not self.embedding_client.is_available():
            return None
        name1, name2 = self._resolve_entity_names_for_embedding(relation, names=names)
        text = self._build_relation_embedding_text(relation, name1, name2)
        embedding = self.embedding_client.encode(text)
        if embedding is None or (isinstance(embedding, (list, tuple)) and len(embedding) == 0):
            return None
        if isinstance(embedding, np.ndarray) and embedding.size == 0:
            return None
        emb_array = np.array(embedding[0] if isinstance(embedding, list) else embedding, dtype=np.float32)
        norm = np.linalg.norm(emb_array)
        if norm > 0:
            emb_array = emb_array / norm
        return emb_array.tobytes()

    # ------------------------------------------------------------------
    # Entity 操作
    # ------------------------------------------------------------------



    def _get_relations_by_entities_impl(self, from_family_id: str, to_family_id: str) -> List[Relation]:
        """根据两个 family_id 获取所有关系（实际实现）。"""
        from_family_id = self.resolve_family_id(from_family_id)
        to_family_id = self.resolve_family_id(to_family_id)
        if not from_family_id or not to_family_id:
            return []

        with self._session() as session:
            # Step 1: 批量获取两个 family_id 的所有 absolute_id（合并 2 次 resolve + 2 次 _get_all_absolute_ids）
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

            # Step 2: 查询关系
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
        """获取所有关系的最新版本及其 embedding（带短 TTL 缓存）。"""
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
        """获取所有关系的最新版本及其 embedding（实际实现）。"""
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



    def _save_relation_impl(self, relation: Relation,
                            names: Optional[Dict[str, str]] = None):
        """保存关系的实际实现。

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

        # Convert embedding bytes → LIST<FLOAT> for Neo4j node property
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
                    "attributes": relation.attributes,
                    "confidence": relation.confidence,
                    "provenance": relation.provenance,
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
                    WITH 1 AS _dummy
                    MATCH (old:Relation {family_id: $family_id})
                    WHERE old.uuid <> $uuid AND old.invalid_at IS NULL
                    SET old.invalid_at = datetime($event_time)
                    WITH 1 AS _dummy2
                    MATCH (ref1:Entity {uuid: $e1_abs})
                    MATCH (n1:Entity {family_id: ref1.family_id}) WHERE n1.invalid_at IS NULL
                    MATCH (ref2:Entity {uuid: $e2_abs})
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



    def _search_relations_with_embedding(self, query_text: str,
                                          relations_with_embeddings: List[tuple],
                                          threshold: float,
                                          max_results: int,
                                          query_embedding=None) -> List[Relation]:
        """使用 Neo4j 向量索引进行关系相似度搜索。"""
        # 1. Encode + 归一化 query (skip if caller provided embedding)
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

        # 3. 去重（同 family_id 取最高分）+ 过滤 threshold
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

    # ------------------------------------------------------------------
    # 文档操作
    # ------------------------------------------------------------------



    def batch_delete_relation_versions_by_absolute_ids(self, absolute_ids: List[str]) -> int:
        """批量删除指定关系版本，返回成功删除的数量。"""
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
        """批量删除关系 — 单次事务，替代 N 次删除。含向量清理。"""
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
        """批量获取引用指定实体绝对ID的关系（消除 N+1 查询）。"""
        if not absolute_ids:
            return {}
        with self._session() as session:
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
        """批量保存关系（UNWIND 批量写入）。

        先写入元数据（不含 embedding），embedding 在后台线程异步计算并更新。
        """
        if not relations:
            return

        # --- Phase 1: 快速写入 Neo4j（不含 embedding）---
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
                "attributes": json.dumps(_attrs) if isinstance(_attrs := getattr(relation, 'attributes', None), dict) else _attrs,
                "confidence": getattr(relation, 'confidence', None),
                "provenance": getattr(relation, 'provenance', None),
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
                    WITH row
                    MATCH (r:Relation {family_id: row.family_id})
                    WHERE r.uuid <> row.uuid AND r.invalid_at IS NULL
                    SET r.invalid_at = datetime(row.event_time)
                    WITH row
                    MATCH (ref1:Entity {uuid: row.e1_abs})
                    MATCH (n1:Entity {family_id: ref1.family_id}) WHERE n1.invalid_at IS NULL
                    MATCH (ref2:Entity {uuid: row.e2_abs})
                    MATCH (n2:Entity {family_id: ref2.family_id}) WHERE n2.invalid_at IS NULL
                    MERGE (n1)-[rel:RELATES_TO {relation_uuid: row.uuid}]->(n2)
                    SET rel.fact = row.content
                    """,
                    operation_name="bulk_save_relations",
                    rows=rows,
                )

        # --- Phase 2: 后台线程计算 embedding + 更新 Neo4j ---
        if self.embedding_client and self.embedding_client.is_available():
            threading.Thread(
                target=self._bulk_save_relation_embedding_bg,
                args=(list(relations),),
                daemon=True,
            ).start()

    def _bulk_save_relation_embedding_bg(self, relations: List[Relation]):
        """后台计算关系 embedding 并更新到 Neo4j。"""
        try:
            # 批量解析实体名称
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



    def count_unique_relations(self) -> int:
        """统计有效关系中不重复的 family_id 数量。"""
        with self._session() as session:
            result = self._run(session, 
                "MATCH (r:Relation) WHERE r.invalid_at IS NULL RETURN COUNT(DISTINCT r.family_id) AS cnt"
            )
            record = result.single()
            return record["cnt"] if record else 0



    def delete_relation_all_versions(self, family_id: str) -> int:
        """删除关系的所有版本。返回删除的行数。"""
        return self.delete_relation_by_id(family_id)



    def delete_relation_by_absolute_id(self, absolute_id: str) -> bool:
        """根据 absolute_id 删除关系，返回是否成功删除。"""
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
        """删除关系的所有版本。返回删除的行数。"""
        abs_ids = []
        count = 0
        with self._relation_write_lock:
            with self._session() as session:
                # 先收集 absolute_ids（DETACH DELETE 后就查不到了）— 轻量 UUID-only 查询
                result = self._run_with_retry(session,
                    "MATCH (r:Relation {family_id: $fid}) RETURN r.uuid AS uuid",
                    fid=family_id,
                )
                abs_ids = [r["uuid"] for r in result]
                # 删除关系节点
                result = self._run_with_retry(session,
                    "MATCH (r:Relation {family_id: $fid}) DETACH DELETE r RETURN count(r) AS cnt",
                    fid=family_id,
                )
                record = result.single()
                count = record["cnt"] if record else 0
            self._invalidate_relation_cache_bulk()
        return count



    def get_all_relations(self, limit: Optional[int] = None, offset: Optional[int] = None,
                           exclude_embedding: bool = False,
                           include_candidates: bool = False) -> List[Relation]:
        """获取所有关系的最新版本。"""
        with self._session() as session:
            fields = _RELATION_RETURN_FIELDS if exclude_embedding else _RELATION_RETURN_FIELDS_WITH_EMB
            query = f"""
                MATCH (r:Relation)
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
        """列出已失效的关系"""
        with self._session() as session:
            result = self._run(session, _q("""
                MATCH (r:Relation)
                WHERE r.invalid_at IS NOT NULL
                RETURN __REL_FIELDS__
                ORDER BY r.invalid_at DESC
                LIMIT $limit
            """), limit=limit)
            return [_neo4j_record_to_relation(r) for r in result]

    # ------------------------------------------------------------------
    # Phase A/C/D/E: 新增方法
    # ------------------------------------------------------------------



    def get_relation_by_absolute_id(self, relation_absolute_id: str) -> Optional[Relation]:
        """根据 absolute_id 获取关系。"""
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
        """获取关系 embedding 预览。"""
        with self._session() as session:
            result = self._run(session,
                "MATCH (r:Relation {uuid: $uuid}) RETURN r.embedding AS embedding",
                uuid=absolute_id,
            )
            record = result.single()
            if record and record["embedding"]:
                return record["embedding"][:num_values]
        return None



    def get_relation_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
        """批量获取多个 relation family_id 的版本数量。"""
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
        """获取关系的所有版本。"""
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
        """批量获取多个 family_id 的所有关系版本（单次 Cypher 查询）。"""
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
        """批量根据 absolute_id 获取关系。"""
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
        """根据两个 family_id 获取所有关系（合并为 2 次 session 查询）。"""
        with _perf_timer("get_relations_by_entities"):
            result = self._get_relations_by_entities_impl(from_family_id, to_family_id)
            return self._filter_dream_candidates(result, include_candidates)



    def get_relations_by_entity_absolute_ids(self, entity_absolute_ids: List[str],
                                              limit: Optional[int] = None,
                                              include_candidates: bool = False) -> List[Relation]:
        """根据 absolute_id 列表获取关系。"""
        if not entity_absolute_ids:
            return []
        with self._session() as session:
            query = _q("""
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id IN $abs_ids OR r.entity2_absolute_id IN $abs_ids)
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
        """批量获取多个实体对的关系 — Cypher-level pair filtering."""
        if not entity_pairs:
            return {}

        # 收集所有唯一的 family_id
        all_family_ids = set()
        for e1, e2 in entity_pairs:
            all_family_ids.add(e1)
            all_family_ids.add(e2)

        # 单次查询获取所有相关的绝对 ID
        with self._session() as session:
            result = self._run(session,
                "MATCH (e:Entity) WHERE e.family_id IN $fids AND e.invalid_at IS NULL RETURN e.family_id AS fid, e.uuid AS uuid",
                fids=list(all_family_ids),
            )
            fid_to_aids: Dict[str, List[str]] = defaultdict(list)
            for record in result:
                fid_to_aids[record["fid"]].append(record["uuid"])

        # Build pair-level absolute_id pairs for Cypher filtering
        aid_pairs = []
        seen_pair_keys: set = set()
        for e1_fid, e2_fid in entity_pairs:
            pair_key = (e1_fid, e2_fid) if e1_fid <= e2_fid else (e2_fid, e1_fid)
            if pair_key in seen_pair_keys:
                continue
            seen_pair_keys.add(pair_key)
            e1_aids = fid_to_aids.get(e1_fid, [])
            e2_aids = fid_to_aids.get(e2_fid, [])
            for a1 in e1_aids:
                for a2 in e2_aids:
                    aid_pairs.append({"a1": a1, "a2": a2})

        if not aid_pairs:
            return {(e1, e2) if e1 <= e2 else (e2, e1): [] for e1, e2 in entity_pairs}

        # Single query with pair-level filtering via UNWIND
        with self._session() as session:
            result = self._run(session, _q("""
                UNWIND $pairs AS p
                MATCH (r:Relation)
                WHERE r.invalid_at IS NULL
                  AND (
                    (r.entity1_absolute_id = p.a1 AND r.entity2_absolute_id = p.a2)
                    OR (r.entity1_absolute_id = p.a2 AND r.entity2_absolute_id = p.a1)
                  )
                RETURN __REL_FIELDS__
                """),
                pairs=aid_pairs,
            )
            all_relations = [_neo4j_record_to_relation(rec) for rec in result]

        # Group by family_id pair using O(1) index lookup
        # Build index: (absolute_id_1, absolute_id_2) -> relation list
        _rel_index: Dict[Tuple[str, str], List[Relation]] = {}
        for rel in all_relations:
            _key = (rel.entity1_absolute_id, rel.entity2_absolute_id) if rel.entity1_absolute_id <= rel.entity2_absolute_id else (rel.entity2_absolute_id, rel.entity1_absolute_id)
            if _key not in _rel_index:
                _rel_index[_key] = []
            _rel_index[_key].append(rel)

        results: Dict[Tuple[str, str], List[Relation]] = {}
        for e1_fid, e2_fid in entity_pairs:
            pair_key = (e1_fid, e2_fid) if e1_fid <= e2_fid else (e2_fid, e1_fid)
            if pair_key in results:
                continue
            e1_aids = fid_to_aids.get(e1_fid) or ()
            e2_aids = fid_to_aids.get(e2_fid) or ()
            pair_rels = []
            for a1 in e1_aids:
                for a2 in e2_aids:
                    _rk = (a1, a2) if a1 <= a2 else (a2, a1)
                    if _rk in _rel_index:
                        pair_rels.extend(_rel_index[_rk])
            results[pair_key] = pair_rels

        return results



    def get_relations_by_family_ids(self, family_ids: List[str], limit: int = 100,
                                    time_point: Optional[str] = None) -> List[Relation]:
        """获取指定实体 ID 列表相关的所有关系。

        使用单次 Cypher 查询完成 family_id→absolute_id 解析 + 关系检索，
        避免逐个 family_id 调用 resolve_family_id + get_entity_by_family_id 的 N+1 问题。

        Args:
            family_ids: 实体 family_id 列表
            limit: 最大返回数量
            time_point: ISO 8601 时间点，仅返回 valid_at <= time_point 且未失效的关系
        """
        if not family_ids:
            return []
        _tp_filter = ""
        _tp_param = {}
        if time_point:
            _tp_filter = " AND (r.valid_at IS NULL OR r.valid_at <= datetime($tp))"
            _tp_param["tp"] = time_point
        with self._session() as session:
            # 单次查询：解析 family_id → 最新 absolute_id，再查找关联关系
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
        """获取所有引用了指定 absolute_id 的关系。"""
        with self._session() as session:
            result = self._run(session, _q("""
                MATCH (r:Relation)
                WHERE r.entity1_absolute_id = $aid OR r.entity2_absolute_id = $aid
                RETURN __REL_FIELDS__
                """),
                aid=absolute_id,
            )
            return [_neo4j_record_to_relation(r) for r in result]



    def invalidate_relation(self, family_id: str, reason: str = "") -> int:
        """标记关系为失效"""
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
        """将指定 family_id 的所有关系在 side 侧重定向到 new_family_id。

        Args:
            family_id: 要重定向的关系的 family_id。
            side: "entity1" 或 "entity2"。
            new_family_id: 新目标实体的 family_id。

        Returns:
            更新的关系数量。
        """
        if side not in ("entity1", "entity2"):
            raise ValueError(f"side must be 'entity1' or 'entity2', got '{side}'")

        side_field = f"{side}_absolute_id"

        with self._relation_write_lock:
            with self._session() as session:
                # 1. 获取 new_family_id 对应的最新实体 absolute_id
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

                # 2. 更新所有匹配的关系
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

    # ------------------------------------------------------------------
    # Concept 统一查询方法（Phase 2: 所有节点共享 :Concept 标签 + role 属性）
    # ------------------------------------------------------------------

    @staticmethod



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
                # Full refresh — combined into single Cypher call
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
        """创建或合并梦境发现的关系。

        Blueprint line 147: Dream relations start as candidates (tier=candidate,
        status=hypothesized, confidence capped at 0.5).

        Returns: {"family_id": "...", "entity1_family_id": "...", "entity2_family_id": "...", "action": "created"|"merged"}
        Raises: ValueError 如果实体不存在
        """
        # 解析实体（batch）
        resolved_map = self.resolve_family_ids([entity1_id, entity2_id])
        resolved1 = resolved_map.get(entity1_id, entity1_id)
        resolved2 = resolved_map.get(entity2_id, entity2_id)
        if not resolved1:
            raise ValueError(f"实体不存在: {entity1_id}")
        if not resolved2:
            raise ValueError(f"实体不存在: {entity2_id}")

        entities_map = self.get_entities_by_family_ids([resolved1, resolved2])
        entity1 = entities_map.get(resolved1)
        entity2 = entities_map.get(resolved2)
        if not entity1:
            raise ValueError(f"实体不存在: {entity1_id}")
        if not entity2:
            raise ValueError(f"实体不存在: {entity2_id}")

        # Check existing relation (include candidates so we can merge with them)
        existing = self.get_relations_by_entities(resolved1, resolved2, include_candidates=True)
        if existing:
            latest = existing[0]
            # 合并：取较高 confidence，追加 reasoning
            new_confidence = max(latest.confidence or 0, confidence)
            # 构建新的 provenance entry
            new_prov_entry = {
                "source": "dream",
                "dream_cycle_id": dream_cycle_id,
                "confidence": confidence,
                "reasoning": reasoning,
            }
            try:
                old_prov = json.loads(latest.provenance) if latest.provenance else []
            except Exception as _prov_err:
                logger.warning("provenance JSON 解析失败，丢弃旧历史: %s", _prov_err)
                old_prov = []
            old_prov.append(new_prov_entry)

            # 创建新版本（保留同一 family_id）
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

        # 排序确保 (A,B) 和 (B,A) 视为同一关系
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

    # ------------------------------------------------------------------
    # Dream candidate lifecycle methods
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------

    def save_relation(self, relation: Relation):
        """保存关系到 Neo4j（合并为单条 Cypher）。"""
        with _perf_timer("save_relation"):
            emb_array = self._save_relation_impl(relation)
            # Incremental relation emb cache update (reuse array from _save_relation_impl)
            if emb_array is not None:
                self._update_relation_emb_cache(relation, emb_array)



    def update_relation_by_absolute_id(self, absolute_id: str, **fields) -> Optional[Relation]:
        """根据 absolute_id 更新指定字段，返回更新后的 Relation 或 None。

        当 content 变更时自动重算 embedding 并更新。
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

        # Convert embedding bytes → LIST<FLOAT> for Neo4j
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
        """更新关系最新版本的置信度。值域 [0.0, 1.0]。"""
        confidence = max(0.0, min(1.0, confidence))
        with self._session() as session:
            self._run(session, """
                MATCH (r:Relation {family_id: $fid})
                WHERE r.invalid_at IS NULL
                WITH r ORDER BY r.processed_time DESC LIMIT 1
                SET r.confidence = $confidence
            """, fid=family_id, confidence=confidence)
        self._invalidate_relation_cache_bulk()

