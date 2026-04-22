"""Neo4j EntityStoreMixin — extracted from neo4j_store."""
import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ...models import ContentPatch, Entity, Relation
from ...perf import _perf_timer
from ..cache import QueryCache
from ..vector_store import VectorStore
from ._helpers import _ENTITY_RETURN_FIELDS, _neo4j_record_to_entity, _neo4j_record_to_relation, _parse_dt, _q

logger = logging.getLogger(__name__)


class EntityStoreMixin:
    """EntityStore operations for Neo4j backend.
    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              → Neo4j session factory
        self._run(session, cypher, **kw) → execute Cypher with graph_id injection
        self._graph_id: str          → active graph ID
        self._entity_write_lock      → threading.Lock for entity writes
        self._cache                  → QueryCache
        self.embedding_client        → EmbeddingClient (optional)
        self._entity_emb_cache       → embedding cache list
        self._entity_emb_cache_ts    → embedding cache timestamp
        self._emb_cache_ttl          → cache TTL in seconds
        self.entity_content_snippet_length → content snippet length
    """

    # ------------------------------------------------------------------

    def _compute_entity_embedding(self, entity: Entity) -> Optional[bytes]:
        """计算实体的 embedding 向量（L2 归一化后存储）。"""
        if not self.embedding_client or not self.embedding_client.is_available():
            return None
        n = self.entity_content_snippet_length
        text = f"{entity.name} {entity.content[:n]}"
        embedding = self.embedding_client.encode(text)
        if embedding is None or len(embedding) == 0:
            return None
        emb_array = np.array(embedding[0] if isinstance(embedding, list) else embedding, dtype=np.float32)
        norm = np.linalg.norm(emb_array)
        if norm > 0:
            emb_array = emb_array / norm
        return emb_array.tobytes()



    def _get_all_absolute_ids_for_entity(self, family_id: str) -> List[str]:
        """获取实体的所有版本的 absolute_id。"""
        with self._session() as session:
            result = self._run(session, 
                "MATCH (e:Entity {family_id: $fid}) RETURN e.uuid AS uuid",
                fid=family_id,
            )
            return [record["uuid"] for record in result]




    def _get_entities_with_embeddings(self) -> List[tuple]:
        """获取所有实体的最新版本及其 embedding（带短 TTL 缓存）。"""
        import time as _time
        now = _time.time()
        if self._entity_emb_cache is not None and (now - self._entity_emb_cache_ts) < self._emb_cache_ttl:
            return self._entity_emb_cache
        with _perf_timer("_get_entities_with_embeddings"):
            result = self._get_entities_with_embeddings_impl()
        self._entity_emb_cache = result
        self._entity_emb_cache_ts = _time.time()
        return result



    def _get_entities_with_embeddings_impl(self) -> List[tuple]:
        """获取所有实体的最新版本及其 embedding（实际实现）。"""
        with self._session() as session:
            result = self._run(session, 
                f"""
                MATCH (e:Entity)
                WITH e.family_id AS fid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(COLLECT(e)) AS e
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time DESC
                """
            )
            records = list(result)

        if not records:
            return []

        # 批量获取所有 embedding（1 次 sqlite 查询 vs N 次）
        uuids = [record["uuid"] for record in records]
        emb_map = self._vector_store.get_batch("entity_vectors", uuids)

        entities = []
        for record in records:
            entity = _neo4j_record_to_entity(record)
            emb_list = emb_map.get(entity.absolute_id)
            emb_array = np.array(emb_list, dtype=np.float32) if emb_list else None
            entities.append((entity, emb_array))
        return entities



    def _get_entity_relations_by_family_id_impl(self, family_id: str, limit: Optional[int] = None,
                                                 time_point: Optional[datetime] = None,
                                                 max_version_absolute_id: Optional[str] = None) -> List[Relation]:
        """通过 family_id 获取实体的所有关系（实际实现）。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        abs_ids = self._get_all_absolute_ids_for_entity(family_id)
        if not abs_ids:
            return []
        if max_version_absolute_id:
            # 获取从最早到 max_version 的所有 absolute_id（拆分为两次查询，避免嵌套 MATCH 语法错误）
            with self._session() as session:
                result = self._run(session, 
                    "MATCH (e2:Entity {uuid: $max_abs}) RETURN e2.processed_time AS max_pt",
                    max_abs=max_version_absolute_id,
                )
                record = result.single()
                if record and record["max_pt"]:
                    max_pt = record["max_pt"]
                    result = self._run(session, 
                        """
                        MATCH (e:Entity {family_id: $fid})
                        WHERE e.processed_time <= $max_pt
                        RETURN e.uuid AS uuid
                        ORDER BY e.processed_time ASC
                        """,
                        fid=family_id,
                        max_pt=max_pt,
                    )
                    abs_ids = [r["uuid"] for r in result]

        if not abs_ids:
            return []

        with self._session() as session:
            if time_point:
                query = _q("""
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id IN $abs_ids OR r.entity2_absolute_id IN $abs_ids)
                    AND r.event_time <= datetime($tp)
                    WITH r.family_id AS fid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH fid, r ORDER BY r.processed_time DESC
                    WITH fid, HEAD(COLLECT(r)) AS r
                    RETURN __REL_FIELDS__
                    ORDER BY r.processed_time DESC
                """)
                params = {"abs_ids": abs_ids, "tp": time_point.isoformat()}
            else:
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
                params = {"abs_ids": abs_ids}

            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = self._run(session, query, **params)
            return [_neo4j_record_to_relation(r) for r in result]



    def _invalidate_emb_cache(self):
        """清除 embedding 缓存（在实体/关系写入时调用）。"""
        self._entity_emb_cache = None
        self._entity_emb_cache_ts = 0.0
        self._relation_emb_cache = None
        self._relation_emb_cache_ts = 0.0



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
            self._cache.invalidate("entity:")
        else:
            self._cache.invalidate("relation:")



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
            self._cache.invalidate("entity:")
        else:
            self._cache.invalidate("relation:")



    def batch_delete_entities(self, family_ids: List[str]) -> int:
        """批量删除实体 — 单次事务，替代 N 次 DETACH DELETE。含向量清理。"""
        resolved_map = self.resolve_family_ids(family_ids)
        resolved = list(set(r for r in resolved_map.values() if r))
        if not resolved:
            return 0
        with self._write_lock:
            # 先收集所有 absolute_ids（DETACH DELETE 后就查不到了）
            all_uuids = []
            with self._session() as session:
                result = self._run(session, 
                    "UNWIND $fids AS fid MATCH (e:Entity {family_id: fid}) RETURN e.uuid AS uuid",
                    fids=resolved,
                )
                all_uuids = [r["uuid"] for r in result]
            with self._session() as session:
                result = self._run(session, 
                    "UNWIND $fids AS fid MATCH (e:Entity {family_id: fid}) DETACH DELETE e RETURN count(e) AS cnt",
                    fids=resolved,
                )
                record = result.single()
                count = record["cnt"] if record else 0
            # 清理向量存储
            if all_uuids:
                try:
                    self._vector_store.delete_batch("entity_vectors", all_uuids)
                except Exception as e:
                    logger.warning("batch_delete_entities vector cleanup failed: %s", e)
            self._cache.invalidate("entity:")
            self._cache.invalidate("resolve:")
            self._cache.invalidate("sim_search:")
            self._cache.invalidate("graph_stats")
            return count



    def batch_delete_entity_versions_by_absolute_ids(self, absolute_ids: List[str]) -> int:
        """批量删除指定实体版本，返回成功删除的数量。含向量清理和缓存失效。"""
        if not absolute_ids:
            return 0
        with self._write_lock:
            with self._session() as session:
                result = self._run(session, 
                    """
                    MATCH (e:Entity) WHERE e.uuid IN $aids
                    DETACH DELETE e
                    RETURN count(e) AS deleted
                    """,
                    aids=absolute_ids,
                )
                record = result.single()
                deleted = record["deleted"] if record else 0
            if deleted > 0:
                try:
                    self._vector_store.delete_batch("entity_vectors", absolute_ids)
                except Exception as e:
                    logger.warning("batch_delete_entity_versions vector cleanup failed: %s", e)
            self._cache.invalidate("entity:")
            self._cache.invalidate("resolve:")
            self._cache.invalidate("sim_search:")
            self._cache.invalidate("graph_stats")
            return deleted



    def batch_get_entity_profiles(self, family_ids: List[str]) -> List[Dict[str, Any]]:
        """批量获取实体档案（entity + relations + version_count），一次查询。

        替代对每个 family_id 分别调用 get_entity_by_family_id +
        get_entity_relations_by_family_id + get_entity_version_count 的 N+1 模式。

        Returns:
            [{"family_id", "entity", "relations", "version_count"}, ...]
        """
        if not family_ids:
            return []

        # 去重 + 解析 canonical IDs（批量解析）
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_map: Dict[str, str] = {}  # original -> canonical
        canonical_set: List[str] = []
        for fid in family_ids:
            resolved = resolved_map.get(fid, fid)
            if resolved and resolved not in canonical_map.values():
                canonical_map[fid] = resolved
                canonical_set.append(resolved)

        if not canonical_set:
            return [{"family_id": fid, "entity": None, "relations": [], "version_count": 0} for fid in family_ids]

        # Session 1: 批量获取实体 + 版本数
        with self._session() as session:
            result = self._run(session, 
                f"""
                MATCH (e:Entity)
                WHERE e.family_id IN $fids AND e.invalid_at IS NULL
                WITH e.family_id AS fid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(COLLECT(e)) AS latest, COUNT(e) AS vcnt
                WITH fid, latest AS e, vcnt
                RETURN {_ENTITY_RETURN_FIELDS}, vcnt
                ORDER BY e.processed_time DESC
                """,
                fids=canonical_set,
            )
            records = list(result)

        entity_map: Dict[str, tuple] = {}  # family_id -> (entity, version_count)
        for record in records:
            entity = _neo4j_record_to_entity(record)
            vc = record.get("vcnt", 1)
            entity_map[entity.family_id] = (entity, vc)

        # Session 2: 批量获取所有相关关系
        all_aids = set()
        for entity, _ in entity_map.values():
            # 获取每个实体的所有 absolute_id
            all_aids.add(entity.absolute_id)

        # 还需要每个实体的所有版本 ID 才能找到关系
        with self._session() as session:
            result = self._run(session, 
                """
                MATCH (e:Entity)
                WHERE e.family_id IN $fids AND e.invalid_at IS NULL
                RETURN e.family_id AS fid, e.uuid AS uuid
                """,
                fids=canonical_set,
            )
            fid_to_aids: Dict[str, List[str]] = {}
            for record in result:
                fid_to_aids.setdefault(record["fid"], []).append(record["uuid"])

        all_aids = set()
        for aids in fid_to_aids.values():
            all_aids.update(aids)

        relations_map: Dict[str, List] = {fid: [] for fid in canonical_set}
        if all_aids:
            with self._session() as session:
                result = self._run(session, _q("""
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id IN $aids OR r.entity2_absolute_id IN $aids)
                      AND r.invalid_at IS NULL
                    RETURN __REL_FIELDS__
                    """),
                    aids=list(all_aids),
                )
                all_rels = [_neo4j_record_to_relation(rec) for rec in result]

            # 分配关系到对应的 family_id
            for rel in all_rels:
                for fid, aids in fid_to_aids.items():
                    if rel.entity1_absolute_id in aids or rel.entity2_absolute_id in aids:
                        relations_map[fid].append(rel)

        # 组装结果
        results = []
        seen_fids = set()
        for fid in family_ids:
            canonical = canonical_map.get(fid, fid)
            if canonical in seen_fids:
                results.append({"family_id": fid, "entity": None, "relations": [], "version_count": 0})
                continue
            seen_fids.add(canonical)
            if canonical in entity_map:
                entity, vc = entity_map[canonical]
                results.append({
                    "family_id": canonical,
                    "entity": entity,
                    "relations": relations_map.get(canonical, []),
                    "version_count": vc,
                })
            else:
                results.append({"family_id": fid, "entity": None, "relations": [], "version_count": 0})

        return results



    def bulk_save_entities(self, entities: List[Entity]):
        """批量保存实体（UNWIND 批量写入）。"""
        if not entities:
            return
        self._invalidate_emb_cache()

        with _perf_timer(f"bulk_save_entities | count={len(entities)}"):
            # 批量计算 embedding
            embeddings = None
            if self.embedding_client and self.embedding_client.is_available():
                texts = [
                    f"{e.name} {e.content[:self.entity_content_snippet_length]}"
                    for e in entities
                ]
                embeddings = self.embedding_client.encode(texts)

            with self._write_lock:
                vec_items = []
                rows = []
                for idx, entity in enumerate(entities):
                    embedding_blob = None
                    if embeddings is not None:
                        try:
                            emb_arr = np.array(embeddings[idx], dtype=np.float32)
                            norm = np.linalg.norm(emb_arr)
                            if norm > 0:
                                emb_arr = emb_arr / norm
                            embedding_blob = emb_arr.tobytes()
                        except Exception as e:
                            logger.debug("Embedding decode failed for entity index %d: %s", idx, e)
                            embedding_blob = None
                    entity.embedding = embedding_blob
                    entity.processed_time = datetime.now()

                    rows.append({
                        "uuid": entity.absolute_id,
                        "family_id": entity.family_id,
                        "name": entity.name,
                        "content": entity.content,
                        "event_time": entity.event_time.isoformat(),
                        "processed_time": entity.processed_time.isoformat(),
                        "cache_id": entity.episode_id,
                        "source": entity.source_document,
                        "summary": entity.summary,
                        "attributes": entity.attributes,
                        "confidence": entity.confidence,
                        "valid_at": (entity.valid_at or entity.event_time).isoformat(),
                        "graph_id": self._graph_id,
                    })

                    if embedding_blob:
                        emb_list = list(np.frombuffer(embedding_blob, dtype=np.float32))
                        vec_items.append((entity.absolute_id, emb_list))

                # 一次 UNWIND 替代 N 次 session.run
                with self._session() as session:
                    self._run(session,
                        """
                        UNWIND $rows AS row
                        MERGE (e:Entity {uuid: row.uuid})
                        SET e:Concept, e.role = 'entity',
                            e.family_id = row.family_id,
                            e.name = row.name,
                            e.content = row.content,
                            e.event_time = datetime(row.event_time),
                            e.processed_time = datetime(row.processed_time),
                            e.episode_id = row.cache_id,
                            e.source_document = row.source,
                            e.summary = row.summary,
                            e.attributes = row.attributes,
                            e.confidence = row.confidence,
                            e.valid_at = datetime(row.valid_at),
                            e.graph_id = row.graph_id
                        WITH row
                        MATCH (e:Entity {family_id: row.family_id})
                        WHERE e.uuid <> row.uuid AND e.invalid_at IS NULL
                        SET e.invalid_at = datetime(row.event_time)
                        """,
                        rows=rows,
                    )

                if vec_items:
                    self._vector_store.upsert_batch("entity_vectors", vec_items)

            self._cache.invalidate("entity:")
            self._cache.invalidate("resolve:")
            self._cache.invalidate("sim_search:")



    def cleanup_invalidated_versions(self, before_date: str = None, dry_run: bool = False) -> Dict[str, Any]:
        """清理已失效的旧版本节点。"""
        with self._session() as session:
            date_filter = ""
            params = {}
            if before_date:
                date_filter = " AND e.invalid_at < datetime($before_date)"
                params["before_date"] = before_date

            # Count entities to remove
            r = self._run(session, f"""
                MATCH (e:Entity) WHERE e.invalid_at IS NOT NULL {date_filter}
                RETURN count(e) AS cnt
            """, **params)
            entity_count = r.single()["cnt"]

            r = self._run(session, f"""
                MATCH (r:Relation) WHERE r.invalid_at IS NOT NULL {date_filter}
                RETURN count(r) AS cnt
            """, **params)
            relation_count = r.single()["cnt"]

            if dry_run:
                return {
                    "dry_run": True,
                    "entities_to_remove": entity_count,
                    "relations_to_remove": relation_count,
                    "message": f"预览：将删除 {entity_count} 个已失效实体版本和 {relation_count} 个已失效关系版本",
                }

            # Actually delete
            r = self._run(session, f"""
                MATCH (e:Entity) WHERE e.invalid_at IS NOT NULL {date_filter}
                DELETE e
                RETURN count(*) AS cnt
            """, **params)
            deleted_entities = r.single()["cnt"]

            r = self._run(session, f"""
                MATCH (r:Relation) WHERE r.invalid_at IS NOT NULL {date_filter}
                DELETE r
                RETURN count(*) AS cnt
            """, **params)
            deleted_relations = r.single()["cnt"]

            return {
                "dry_run": False,
                "deleted_entity_versions": deleted_entities,
                "deleted_relation_versions": deleted_relations,
                "message": f"已删除 {deleted_entities} 个已失效实体版本和 {deleted_relations} 个已失效关系版本",
            }




    def count_isolated_entities(self) -> int:
        """统计孤立实体数量。"""
        with self._session() as session:
            r = self._run(session, """
                MATCH (rel:Relation) WHERE rel.invalid_at IS NULL
                WITH collect(DISTINCT rel.entity1_absolute_id)
                   + collect(DISTINCT rel.entity2_absolute_id) AS aids
                UNWIND aids AS aid
                WITH collect(DISTINCT aid) AS connected
                MATCH (e:Entity)
                WHERE e.invalid_at IS NULL AND e.family_id IS NOT NULL
                  AND NOT e.uuid IN connected
                RETURN count(DISTINCT e.family_id) AS cnt
            """)
            row = r.single()
            return row["cnt"] if row else 0



    def count_unique_entities(self) -> int:
        """统计有效实体中不重复的 family_id 数量。"""
        with self._session() as session:
            result = self._run(session, 
                "MATCH (e:Entity) WHERE e.invalid_at IS NULL RETURN COUNT(DISTINCT e.family_id) AS cnt"
            )
            record = result.single()
            return record["cnt"] if record else 0



    def delete_entity_all_versions(self, family_id: str) -> int:
        """删除实体的所有版本（含关系边）。返回删除的行数。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        with self._write_lock:
            # 先收集 absolute_ids（DETACH DELETE 后就查不到了）
            abs_ids = [e.absolute_id for e in self.get_entity_versions(family_id)]
            with self._session() as session:
                # 删除相关关系
                self._run(session, 
                    """MATCH (e:Entity {family_id: $fid})-[r:RELATES_TO]-()
                       DETACH DELETE r""",
                    fid=family_id,
                )
                # 删除实体节点
                result = self._run(session, 
                    "MATCH (e:Entity {family_id: $fid}) DETACH DELETE e RETURN count(e) AS cnt",
                    fid=family_id,
                )
                record = result.single()
                count = record["cnt"] if record else 0
                # 清理向量存储
                try:
                    self._vector_store.delete_batch("entity_vectors", abs_ids)
                except Exception as e:
                    logger.warning("Failed to clean up entity vectors for %s: %s", family_id, e)
                self._cache.invalidate("entity:")
                self._cache.invalidate("resolve:")
                self._cache.invalidate("sim_search:")
                self._cache.invalidate("graph_stats")
                return count



    def delete_entity_by_absolute_id(self, absolute_id: str) -> bool:
        """根据 absolute_id 删除实体及其所有关系，返回是否成功删除。"""
        with self._write_lock:
            with self._session() as session:
                result = self._run(session, 
                    "MATCH (e:Entity {uuid: $aid}) DETACH DELETE e RETURN count(e) AS cnt",
                    aid=absolute_id,
                )
                record = result.single()
                deleted = record is not None and record["cnt"] > 0
            if deleted:
                self._vector_store.delete_batch("entity_vectors", [absolute_id])
            self._cache.invalidate("entity:")
            self._cache.invalidate("resolve:")
            self._cache.invalidate("sim_search:")
            self._cache.invalidate("graph_stats")
            return deleted



    def find_entity_by_name_prefix(self, prefix: str, limit: int = 5) -> list:
        """查找名称以 prefix 开头的实体（处理消歧括号场景）。
        例如 prefix="Go语言" 可匹配 "Go语言（Golang）"。
        返回 Entity 对象列表，按 processed_time 倒序。
        """
        if not prefix:
            return []
        try:
            with self._session() as session:
                result = self._run(session, 
                    """
                    MATCH (e:Entity)
                    WHERE (e.name STARTS WITH $prefix OR e.name = $prefix)
                      AND e.invalid_at IS NULL
                    RETURN e.uuid AS uuid, e.family_id AS family_id,
                           e.name AS name, e.content AS content,
                           e.summary AS summary,
                           e.attributes AS attributes, e.confidence AS confidence,
                           e.source_document AS source_document, e.episode_id AS episode_id,
                           e.doc_name AS doc_name,
                           e.processed_time AS processed_time, e.event_time AS event_time,
                           e.content_format AS content_format
                    ORDER BY e.processed_time DESC
                    LIMIT $limit
                    """,
                    prefix=prefix,
                    limit=limit,
                )
                entities = []
                seen_fids = set()
                for record in result:
                    fid = record.get("family_id")
                    if fid and fid not in seen_fids:
                        seen_fids.add(fid)
                        entities.append(_neo4j_record_to_entity(record))
                return entities
        except Exception as e:
            logger.debug("find_entity_by_name_prefix failed for '%s': %s", prefix, e)
            return []

    # ------------------------------------------------------------------
    # BM25 Full-Text Search
    # ------------------------------------------------------------------



    def get_all_entities(self, limit: Optional[int] = None, offset: Optional[int] = None, exclude_embedding: bool = False) -> List[Entity]:
        """获取所有实体的最新版本。"""
        with self._session() as session:
            query = f"""
                MATCH (e:Entity)
                WITH e.family_id AS fid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(COLLECT(e)) AS e
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time DESC
            """
            if offset is not None and offset > 0:
                query += f" SKIP {int(offset)}"
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = self._run(session, query)
            records = list(result)

        entities = [_neo4j_record_to_entity(r) for r in records]

        if not exclude_embedding and entities:
            uuids = [e.absolute_id for e in entities]
            emb_map = self._vector_store.get_batch("entity_vectors", uuids)
            for entity in entities:
                emb_list = emb_map.get(entity.absolute_id)
                if emb_list:
                    entity.embedding = np.array(emb_list, dtype=np.float32).tobytes()

        return entities



    def get_all_entities_before_time(self, time_point: datetime, limit: Optional[int] = None,
                                      exclude_embedding: bool = False) -> List[Entity]:
        """获取指定时间点之前的所有实体最新版本。"""
        with self._session() as session:
            query = f"""
                MATCH (e:Entity)
                WHERE e.event_time <= datetime($tp)
                WITH e.family_id AS fid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(COLLECT(e)) AS e
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time DESC
            """
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = self._run(session, query, tp=time_point.isoformat())
            records = list(result)

        entities = [_neo4j_record_to_entity(r) for r in records]

        if not exclude_embedding and entities:
            uuids = [e.absolute_id for e in entities]
            emb_map = self._vector_store.get_batch("entity_vectors", uuids)
            for entity in entities:
                emb_list = emb_map.get(entity.absolute_id)
                if emb_list:
                    entity.embedding = np.array(emb_list, dtype=np.float32).tobytes()

        return entities



    def get_content_patches(self, family_id: str, section_key: str = None) -> list:
        """查询指定 family_id 的 ContentPatch 记录。"""

        with self._session() as session:
            if section_key:
                result = session.run(
                    """
                    MATCH (cp:ContentPatch {target_family_id: $fid, section_key: $sk})
                    RETURN cp ORDER BY cp.event_time DESC
                    """,
                    fid=family_id, sk=section_key,
                )
            else:
                result = session.run(
                    """
                    MATCH (cp:ContentPatch {target_family_id: $fid})
                    RETURN cp ORDER BY cp.event_time DESC
                    """,
                    fid=family_id,
                )
            patches = []
            for record in result:
                cp = record["cp"]
                patches.append(ContentPatch(
                    uuid=cp["uuid"],
                    target_type=cp["target_type"],
                    target_absolute_id=cp["target_absolute_id"],
                    target_family_id=cp["target_family_id"],
                    section_key=cp["section_key"],
                    change_type=cp["change_type"],
                    old_hash=cp.get("old_hash", ""),
                    new_hash=cp.get("new_hash", ""),
                    diff_summary=cp.get("diff_summary", ""),
                    source_document=cp.get("source_document", ""),
                    event_time=_parse_dt(cp.get("event_time")),
                ))
            return patches



    def get_data_quality_report(self) -> Dict[str, Any]:
        """返回数据质量报告。"""
        # Session 1: 所有计数查询（实体 + 关系）
        with self._session() as session:
            # 有效实体
            r = self._run(session, """
                MATCH (e:Entity) WHERE e.invalid_at IS NULL AND e.family_id IS NOT NULL
                RETURN count(DISTINCT e.family_id) AS valid_families, count(e) AS valid_nodes
            """)
            row = r.single()
            valid_families = row["valid_families"]
            valid_nodes = row["valid_nodes"]

            # 失效版本
            r = self._run(session, "MATCH (e:Entity) WHERE e.invalid_at IS NOT NULL RETURN count(e) AS cnt")
            invalidated_entity_versions = r.single()["cnt"]

            # 无 family_id 的实体
            r = self._run(session, "MATCH (e:Entity) WHERE e.family_id IS NULL RETURN count(e) AS cnt")
            no_family_id = r.single()["cnt"]

            # 有效关系
            r = self._run(session, """
                MATCH (rel:Relation) WHERE rel.invalid_at IS NULL
                RETURN count(DISTINCT rel.family_id) AS valid_families, count(rel) AS valid_nodes
            """)
            row = r.single()
            valid_relation_families = row["valid_families"]
            valid_relation_nodes = row["valid_nodes"]

            # 失效关系版本
            r = self._run(session, "MATCH (rel:Relation) WHERE rel.invalid_at IS NOT NULL RETURN count(rel) AS cnt")
            invalidated_relation_versions = r.single()["cnt"]

        # 孤立实体
        isolated_count = self.count_isolated_entities()

        # Session 2: 悬空引用检测
        with self._session() as session:
            r = self._run(session, """
                MATCH (rel:Relation) WHERE rel.invalid_at IS NULL
                RETURN collect(DISTINCT rel.entity1_absolute_id) AS e1_ids,
                       collect(DISTINCT rel.entity2_absolute_id) AS e2_ids
            """)
            row = r.single()
            rel_aids = set()
            if row:
                if row.get("e1_ids"):
                    rel_aids.update(row["e1_ids"])
                if row.get("e2_ids"):
                    rel_aids.update(row["e2_ids"])

            r = self._run(session, "MATCH (e:Entity) WHERE e.invalid_at IS NULL RETURN collect(DISTINCT e.uuid) AS uuids")
            row = r.single()
            valid_uuids = set(row["uuids"]) if row and row["uuids"] else set()

        dangling_refs = len(rel_aids - valid_uuids)

        return {
            "entities": {
                "valid_unique": valid_families,
                "valid_versions": valid_nodes,
                "invalidated_versions": invalidated_entity_versions,
                "no_family_id": no_family_id,
                "isolated": isolated_count,
            },
            "relations": {
                "valid_unique": valid_relation_families,
                "valid_versions": valid_relation_nodes,
                "invalidated_versions": invalidated_relation_versions,
                "dangling_entity_refs": dangling_refs,
            },
            "total_nodes": valid_nodes + invalidated_entity_versions + valid_relation_nodes + invalidated_relation_versions + no_family_id,
        }



    def get_entities_by_absolute_ids(self, absolute_ids: List[str]) -> List[Entity]:
        """批量根据 absolute_id 获取实体。"""
        if not absolute_ids:
            return []
        with self._session() as session:
            result = self._run(session, 
                f"""
                MATCH (e:Entity)
                WHERE e.uuid IN $uuids
                RETURN {_ENTITY_RETURN_FIELDS}
                """,
                uuids=absolute_ids,
            )
            return [_neo4j_record_to_entity(r) for r in result]



    def get_entities_by_family_ids(self, family_ids: List[str]) -> Dict[str, "Entity"]:
        """批量根据 family_id 获取最新版本实体，返回 {family_id: Entity}。"""
        if not family_ids:
            return {}
        # 先 resolve，利用缓存
        resolved_map = self.resolve_family_ids(list(family_ids))
        valid_fids = set(resolved_map.keys()) | set(resolved_map.values())
        if not valid_fids:
            return {}
        # 检查缓存
        result: Dict[str, "Entity"] = {}
        uncached = set()
        for fid in valid_fids:
            cached = self._cache.get(f"entity:by_fid:{fid}")
            if cached is not None:
                result[fid] = cached
            else:
                uncached.add(fid)
        # 批量查询未命中缓存的
        if uncached:
            with self._session() as session:
                cypher = (
                    f"MATCH (e:Entity) WHERE e.family_id IN $fids "
                    f"WITH e ORDER BY e.processed_time DESC "
                    f"WITH e.family_id AS fid, collect(e)[0] AS latest "
                    f"RETURN latest"
                )
                records = self._run(session, cypher, fids=list(uncached)).data()
                entities = [_neo4j_record_to_entity(rec["latest"]) for rec in records]
                # 批量获取 embedding（替代逐个 _vector_store.get）
                if entities:
                    uuids = [e.absolute_id for e in entities]
                    emb_map = self._vector_store.get_batch("entity_vectors", uuids)
                    for entity in entities:
                        emb = emb_map.get(entity.absolute_id)
                        if emb:
                            entity.embedding = np.array(emb, dtype=np.float32).tobytes()
                        result[entity.family_id] = entity
                        self._cache.set(f"entity:by_fid:{entity.family_id}", entity, ttl=60)
        # 映射原始 ID → 实体
        for orig_fid, resolved_fid in resolved_map.items():
            if resolved_fid in result and orig_fid not in result:
                result[orig_fid] = result[resolved_fid]
        return result



    def get_entity_absolute_ids_up_to_version(self, family_id: str, max_absolute_id: str) -> List[str]:
        """获取从最早版本到指定版本的所有 absolute_id。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        with self._session() as session:
            result = self._run(session, 
                """
                MATCH (e:Entity {family_id: $fid})
                WHERE e.processed_time <= (
                    MATCH (e2:Entity {uuid: $max_abs}) RETURN e2.processed_time
                )
                RETURN e.uuid AS uuid
                ORDER BY e.processed_time ASC
                """,
                fid=family_id,
                max_abs=max_absolute_id,
            )
            return [r["uuid"] for r in result]



    def get_entity_by_absolute_id(self, absolute_id: str) -> Optional[Entity]:
        """根据 absolute_id 获取实体。"""
        cache_key = f"entity:by_abs:{absolute_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with self._session() as session:
            result = self._run(session, 
                f"""
                MATCH (e:Entity {{uuid: $uuid}})
                RETURN {_ENTITY_RETURN_FIELDS}
                """,
                uuid=absolute_id,
            )
            record = result.single()
            if not record:
                return None
            entity = _neo4j_record_to_entity(record)
            emb = self._vector_store.get("entity_vectors", entity.absolute_id)
            if emb:
                entity.embedding = np.array(emb, dtype=np.float32).tobytes()
            self._cache.set(cache_key, entity, ttl=60)
            return entity



    def get_entity_by_family_id(self, family_id: str) -> Optional[Entity]:
        """根据 family_id 获取最新版本的实体。"""
        cache_key = f"entity:by_fid:{family_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with _perf_timer("get_entity_by_family_id"):
            family_id = self.resolve_family_id(family_id)
            if not family_id:
                return None
            with self._session() as session:
                result = self._run(session, 
                    f"""
                    MATCH (e:Entity {{family_id: $fid}})
                    RETURN {_ENTITY_RETURN_FIELDS}
                    ORDER BY e.processed_time DESC LIMIT 1
                    """,
                    fid=family_id,
                )
                record = result.single()
                if not record:
                    return None
                entity = _neo4j_record_to_entity(record)
                # 从 sqlite-vec 获取 embedding
                emb = self._vector_store.get("entity_vectors", entity.absolute_id)
                if emb:
                    entity.embedding = np.array(emb, dtype=np.float32).tobytes()
                self._cache.set(cache_key, entity, ttl=60)
                return entity



    def get_entity_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        """获取实体 embedding 预览。"""
        emb = self._vector_store.get("entity_vectors", absolute_id)
        if emb:
            return emb[:num_values]
        return None



    def get_entity_names_by_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, str]:
        """批量根据 absolute_id 查询实体名称。"""
        if not absolute_ids:
            return {}
        with self._session() as session:
            result = self._run(session, 
                """
                MATCH (e:Entity)
                WHERE e.uuid IN $uuids
                RETURN e.uuid AS uuid, e.name AS name
                """,
                uuids=absolute_ids,
            )
            return {record["uuid"]: record["name"] for record in result}



    def get_entity_provenance(self, family_id: str) -> List[dict]:
        """获取提及该实体的所有 Episode。

        查询所有版本的 MENTIONS 边（与 SQLite 实现一致）。
        先查找 Episode->Entity 的直接 MENTIONS；如果无结果，
        再查找通过该实体参与的关系（Episode->Relation 的 MENTIONS）间接关联的 Episode。

        注意：Episode 节点可能缺少 graph_id（旧数据），因此只对 Entity/Relation 侧
        进行 graph_id 过滤，不依赖 Episode 的 graph_id。
        """
        abs_ids = self._get_all_absolute_ids_for_entity(family_id)
        if not abs_ids:
            return []
        with self._session() as session:
            # 1. 直接 MENTIONS: Episode -> Entity (any version)
            # 不对 Episode 注入 graph_id 过滤（旧 Episode 可能缺少 graph_id）
            result = self._run(session, """
                MATCH (ep:Episode)-[m:MENTIONS]->(e:Entity)
                WHERE e.uuid IN $abs_ids AND e.graph_id = $graph_id
                RETURN DISTINCT ep.uuid AS episode_id, m.context AS context
            """, abs_ids=abs_ids, graph_id=self._graph_id, graph_id_safe=False)
            provenance = [{"episode_id": r["episode_id"], "context": r.get("context", "")} for r in result]

            if provenance:
                return provenance

            # 2. 间接 MENTIONS: Episode -> Relation（该实体参与的关系）
            result = self._run(session, """
                MATCH (ep:Episode)-[m:MENTIONS]->(r:Relation)
                WHERE (r.entity1_absolute_id IN $abs_ids OR r.entity2_absolute_id IN $abs_ids)
                      AND r.graph_id = $graph_id
                RETURN DISTINCT ep.uuid AS episode_id, m.context AS context
            """, abs_ids=abs_ids, graph_id=self._graph_id, graph_id_safe=False)
            provenance = [{"episode_id": r["episode_id"], "context": r.get("context", "")} for r in result]

            return provenance




    def get_entity_relations(self, entity_absolute_id: str, limit: Optional[int] = None,
                              time_point: Optional[datetime] = None,
                              include_candidates: bool = False) -> List[Relation]:
        """获取与指定实体相关的所有关系。"""
        with self._session() as session:
            if time_point:
                query = _q("""
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id = $abs_id OR r.entity2_absolute_id = $abs_id)
                    AND r.event_time <= datetime($tp)
                    WITH r.family_id AS fid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH fid, r ORDER BY r.processed_time DESC
                    WITH fid, HEAD(COLLECT(r)) AS r
                    RETURN __REL_FIELDS__
                    ORDER BY r.processed_time DESC
                """)
                params = {"abs_id": entity_absolute_id, "tp": time_point.isoformat()}
            else:
                query = _q("""
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id = $abs_id OR r.entity2_absolute_id = $abs_id)
                    WITH r.family_id AS fid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH fid, r ORDER BY r.processed_time DESC
                    WITH fid, HEAD(COLLECT(r)) AS r
                    RETURN __REL_FIELDS__
                    ORDER BY r.processed_time DESC
                """)
                params = {"abs_id": entity_absolute_id}

            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = self._run(session, query, **params)
            relations = [_neo4j_record_to_relation(r) for r in result]
            return self._filter_dream_candidates(relations, include_candidates)



    def get_entity_relations_by_family_id(self, family_id: str, limit: Optional[int] = None,
                                           time_point: Optional[datetime] = None,
                                           max_version_absolute_id: Optional[str] = None,
                                           include_candidates: bool = False) -> List[Relation]:
        """通过 family_id 获取实体的所有关系（包含所有版本）。"""
        with _perf_timer("get_entity_relations_by_family_id"):
            result = self._get_entity_relations_by_family_id_impl(family_id, limit, time_point, max_version_absolute_id)
            return self._filter_dream_candidates(result, include_candidates)



    def get_entity_relations_timeline(self, family_id: str, version_abs_ids: List[str]) -> List[Dict]:
        """批量获取实体在各版本时间点的关系（消除 N+1 查询）。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id or not version_abs_ids:
            return []
        abs_ids = self._get_all_absolute_ids_for_entity(family_id)
        if not abs_ids:
            return []

        with self._session() as session:
            # 获取各版本的 processed_time
            result = self._run(session, 
                """
                MATCH (e:Entity)
                WHERE e.uuid IN $version_abs_ids
                RETURN e.uuid AS uuid, e.processed_time AS pt
                ORDER BY e.pt ASC
                """,
                version_abs_ids=version_abs_ids,
            )
            version_times = [(record["uuid"], record["pt"]) for record in result]
            if not version_times:
                return []

            # 获取所有相关关系（一次查询）
            result = self._run(session, _q("""
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id IN $abs_ids OR r.entity2_absolute_id IN $abs_ids)
                WITH r.family_id AS fid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH fid, r ORDER BY r.processed_time DESC
                WITH fid, HEAD(COLLECT(r)) AS r
                RETURN __REL_FIELDS__
                ORDER BY r.processed_time ASC
                """),
                abs_ids=abs_ids,
            )

            # 按 processed_time 过滤：只返回在每个版本时间点之前出现的关系
            timeline = []
            seen = set()
            for record in result:
                if record["uuid"] in seen:
                    continue
                rel_pt = record["processed_time"]
                for v_uuid, v_pt in version_times:
                    if rel_pt and v_pt and rel_pt <= v_pt:
                        seen.add(record["uuid"])
                        timeline.append({
                            "family_id": record["family_id"],
                            "content": record["content"],
                            "event_time": record["event_time"].isoformat() if record["event_time"] else None,
                            "absolute_id": record["uuid"],
                        })
                        break
            return timeline



    def get_entity_version_at_time(self, family_id: str, time_point: datetime) -> Optional[Entity]:
        """获取实体在指定时间点的版本。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return None
        with self._session() as session:
            result = self._run(session, 
                f"""
                MATCH (e:Entity {{family_id: $fid}})
                WHERE e.event_time <= datetime($tp)
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time DESC LIMIT 1
                """,
                fid=family_id,
                tp=time_point.isoformat(),
            )
            record = result.single()
            if not record:
                return None
            entity = _neo4j_record_to_entity(record)
            emb = self._vector_store.get("entity_vectors", entity.absolute_id)
            if emb:
                entity.embedding = np.array(emb, dtype=np.float32).tobytes()
            return entity



    def get_entity_version_count(self, family_id: str) -> int:
        """获取指定 family_id 的版本数量。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        with self._session() as session:
            result = self._run(session, 
                "MATCH (e:Entity {family_id: $fid}) RETURN COUNT(e) AS cnt",
                fid=family_id,
            )
            record = result.single()
            return record["cnt"] if record else 0



    def get_entity_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
        """批量获取多个 family_id 的版本数量。"""
        if not family_ids:
            return {}
        # 批量解析重定向
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = list(set(r for r in resolved_map.values() if r))
        if not canonical_ids:
            return {}
        with self._session() as session:
            result = self._run(session, 
                """
                MATCH (e:Entity)
                WHERE e.family_id IN $fids
                RETURN e.family_id AS family_id, COUNT(e) AS cnt
                """,
                fids=canonical_ids,
            )
            return {record["family_id"]: record["cnt"] for record in result}



    def get_entity_versions(self, family_id: str) -> List[Entity]:
        """获取实体的所有版本。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        with self._session() as session:
            result = self._run(session, 
                f"""
                MATCH (e:Entity {{family_id: $fid}})
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time ASC
                """,
                fid=family_id,
            )
            entities = []
            for record in result:
                entities.append(_neo4j_record_to_entity(record))
            return entities



    def get_family_ids_by_names(self, names: list) -> dict:
        """按名称批量查询 family_id。"""
        if not names:
            return {}
        with self._session() as session:
            result = self._run(session, 
                """
                MATCH (e:Entity)
                WHERE e.name IN $names
                RETURN e.name AS name, e.family_id AS family_id
                ORDER BY e.processed_time DESC
                """,
                names=names,
            )
            # 收集所有唯一的 family_ids
            raw_output = {}
            unique_fids = set()
            for record in result:
                name = record["name"]
                fid = record["family_id"]
                if name not in raw_output:
                    raw_output[name] = fid
                    unique_fids.add(fid)
            # 批量解析
            resolved_map = self.resolve_family_ids(list(unique_fids)) if unique_fids else {}
            return {name: resolved_map.get(fid, fid) for name, fid in raw_output.items()}



    def get_graph_statistics(self) -> Dict[str, Any]:
        """返回图谱结构统计数据（仅统计有效版本，排除已失效的旧版本节点）

        优化：合并 9 次串行 Cypher 为 3 次：
        1. 基础计数 + 度数统计（一条 UNWIND 聚合）
        2. 实体时间趋势
        3. 关系时间趋势
        """
        cached = self._cache.get("graph_stats")
        if cached is not None:
            return cached
        with self._session() as session:
            # Query 1: 基础计数 + 度数统计（合并原 6 次查询为 1 次）
            r = self._run(session, """
                // 基础计数
                MATCH (all_e:Entity)
                WITH count(all_e) AS total_entity_versions
                MATCH (all_r:Relation)
                WITH total_entity_versions, count(all_r) AS total_relation_versions
                MATCH (valid_e:Entity) WHERE valid_e.invalid_at IS NULL
                WITH total_entity_versions, total_relation_versions,
                     count(DISTINCT valid_e.family_id) AS entity_count
                MATCH (valid_r:Relation) WHERE valid_r.invalid_at IS NULL
                WITH total_entity_versions, total_relation_versions, entity_count,
                     count(DISTINCT valid_r.family_id) AS relation_count,
                     collect(DISTINCT valid_r.entity1_absolute_id)
                     + collect(DISTINCT valid_r.entity2_absolute_id) AS rel_uuids
                // 度数：按 RELATES_TO 边判断实体是否连通
                UNWIND CASE WHEN entity_count > 0 THEN [1] ELSE [] END AS _trigger
                MATCH (e:Entity) WHERE e.invalid_at IS NULL AND e.family_id IS NOT NULL
                WITH total_entity_versions, total_relation_versions, entity_count,
                     relation_count, rel_uuids,
                     e.family_id AS fid, e.uuid AS uid, e AS ent
                OPTIONAL MATCH (ent)-[rt:RELATES_TO]-()
                WITH total_entity_versions, total_relation_versions, entity_count,
                     relation_count, rel_uuids, fid,
                     CASE WHEN rt IS NOT NULL THEN 1 ELSE 0 END AS is_connected
                RETURN total_entity_versions, total_relation_versions,
                       entity_count, relation_count,
                       avg(CASE WHEN is_connected = 1 THEN 1.0 ELSE 0.0 END) AS avg_degree,
                       max(CASE WHEN is_connected = 1 THEN 1 ELSE 0 END) AS max_degree_raw,
                       sum(CASE WHEN is_connected = 0 THEN 1 ELSE 0 END) AS isolated_count
            """)
            row = r.single()

            # When entity_count=0, UNWIND produces no rows → r.single() returns None.
            # Fall back to a lightweight count-only query.
            if row is None:
                r2 = self._run(session, """
                    MATCH (all_e:Entity)  WITH count(all_e) AS total_entity_versions
                    MATCH (all_r:Relation) WITH total_entity_versions, count(all_r) AS total_relation_versions
                    MATCH (valid_e:Entity) WHERE valid_e.invalid_at IS NULL
                    WITH total_entity_versions, total_relation_versions, count(DISTINCT valid_e.family_id) AS entity_count
                    MATCH (valid_r:Relation) WHERE valid_r.invalid_at IS NULL
                    RETURN total_entity_versions, total_relation_versions,
                           entity_count, count(DISTINCT valid_r.family_id) AS relation_count
                """)
                row = r2.single()

            if row is None:
                return {}

            total_entity_versions = row["total_entity_versions"]
            total_relation_versions = row["total_relation_versions"]
            entity_count = row["entity_count"]
            relation_count = row["relation_count"]

            stats = {
                "entity_count": entity_count,
                "relation_count": relation_count,
                "total_entity_versions": total_entity_versions,
                "total_relation_versions": total_relation_versions,
            }

            if entity_count > 0 and row.get("isolated_count") is not None:
                # 注意：avg_degree 是 per-version 而非 per-family；这里用 family 数重新计算
                # 但 row 中的 isolated_count 已经正确
                isolated = row["isolated_count"]
                connected = entity_count - isolated
                stats["avg_relations_per_entity"] = round(connected / entity_count, 2) if entity_count else 0
                stats["max_relations_per_entity"] = row["max_degree_raw"]
                stats["isolated_entities"] = isolated

                if entity_count > 1:
                    max_possible = entity_count * (entity_count - 1) / 2
                    stats["graph_density"] = round(relation_count / max_possible, 4)
                else:
                    stats["graph_density"] = 0.0
            else:
                stats.update({
                    "avg_relations_per_entity": 0,
                    "max_relations_per_entity": 0,
                    "isolated_entities": 0,
                    "graph_density": 0.0,
                })

            # Query 2: 实体时间趋势
            r = self._run(session, """
                MATCH (e:Entity)
                WHERE e.invalid_at IS NULL AND e.event_time IS NOT NULL
                WITH date(e.event_time) AS d, e.family_id AS fid
                RETURN d AS date, count(DISTINCT fid) AS cnt
                ORDER BY d
                LIMIT 30
            """)
            stats["entity_count_over_time"] = [{"date": str(rec["date"]), "count": rec["cnt"]} for rec in r]

            # Query 3: 关系时间趋势
            r = self._run(session, """
                MATCH (r:Relation)
                WHERE r.invalid_at IS NULL AND r.event_time IS NOT NULL
                WITH date(r.event_time) AS d, r.family_id AS fid
                RETURN d AS date, count(DISTINCT fid) AS cnt
                ORDER BY d
                LIMIT 30
            """)
            stats["relation_count_over_time"] = [{"date": str(rec["date"]), "count": rec["cnt"]} for rec in r]

        self._cache.set("graph_stats", stats, ttl=60)
        return stats

    # ------------------------------------------------------------------
    # 管理类方法：孤立实体、数据质量报告、版本清理
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------

    def get_isolated_entities(self, limit: int = 100, offset: int = 0) -> List[Entity]:
        """获取所有孤立实体（有效实体中没有 RELATES_TO 边的）。"""
        with self._session() as session:
            r = self._run(session, f"""
                MATCH (e:Entity)
                WHERE e.invalid_at IS NULL AND e.family_id IS NOT NULL
                AND NOT EXISTS {{ MATCH (e)-[:RELATES_TO]-() }}
                WITH e.family_id AS fid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(COLLECT(e)) AS e
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time DESC
                SKIP $offset LIMIT $limit
            """, offset=offset, limit=limit)
            return [_neo4j_record_to_entity(rec) for rec in r]



    def get_latest_entities_projection(self, content_snippet_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取最新实体投影。"""
        snippet_length = content_snippet_length or self.entity_content_snippet_length
        entities_with_emb = self._get_entities_with_embeddings()
        version_counts = self.get_entity_version_counts([
            e.family_id for e, _ in entities_with_emb
        ])
        results: List[Dict[str, Any]] = []
        for entity, embedding_array in entities_with_emb:
            results.append({
                "entity": entity,
                "family_id": entity.family_id,
                "name": entity.name,
                "content": entity.content,
                "content_snippet": (entity.content or "")[:snippet_length],
                "version_count": version_counts.get(entity.family_id, 1),
                "embedding_array": embedding_array,
            })
        return results



    def get_section_history(self, family_id: str, section_key: str) -> list:
        """获取单个 section 的全版本变更历史。"""
        return self.get_content_patches(family_id, section_key=section_key)



    def get_stats(self) -> Dict[str, Any]:
        """返回当前图谱的基础统计：有效实体数和关系数。

        用于 GraphRegistry.get_graph_info() 显示图谱列表信息。
        """
        try:
            with self._session() as session:
                r1 = self._run(session,
                    "MATCH (e:Entity) WHERE e.invalid_at IS NULL "
                    "RETURN count(DISTINCT e.family_id) AS cnt"
                )
                entity_count = r1.single()["cnt"]

                r2 = self._run(session,
                    "MATCH (r:Relation) WHERE r.invalid_at IS NULL "
                    "RETURN count(DISTINCT r.family_id) AS cnt"
                )
                relation_count = r2.single()["cnt"]

                return {"entities": entity_count, "relations": relation_count}
        except Exception as e:
            logger.warning("get_stats failed: %s", e)
            return {"entities": 0, "relations": 0}



    def get_version_diff(self, family_id: str, v1: str, v2: str) -> dict:
        """获取两个版本之间的 section 级 diff。

        v1, v2 是两个 absolute_id（版本 uuid）。
        返回 {section_key: {"v1": content_or_None, "v2": content_or_None, "changed": bool}}
        """
        from ..content_schema import parse_markdown_sections, compute_section_diff
        with self._session() as session:
            v1_content = ""
            v2_content = ""
            result = self._run(session, 
                """
                MATCH (e:Entity) WHERE e.uuid = $v1 OR e.uuid = $v2
                RETURN e.uuid AS uid, e.content AS content
                """,
                v1=v1, v2=v2,
            )
            for record in result:
                if record["uid"] == v1:
                    v1_content = record["content"] or ""
                elif record["uid"] == v2:
                    v2_content = record["content"] or ""
            s1 = parse_markdown_sections(v1_content)
            s2 = parse_markdown_sections(v2_content)
            return compute_section_diff(s1, s2)


    # ------------------------------------------------------------------

    def save_content_patches(self, patches: list):
        """批量保存 ContentPatch 节点到 Neo4j。"""

        if not patches:
            return
        with self._entity_write_lock:
            with self._session() as session:
                rows = [
                    {
                        "uuid": p.uuid,
                        "target_type": p.target_type,
                        "target_abs_id": p.target_absolute_id,
                        "target_family_id": p.target_family_id,
                        "section_key": p.section_key,
                        "change_type": p.change_type,
                        "old_hash": p.old_hash,
                        "new_hash": p.new_hash,
                        "diff_summary": p.diff_summary,
                        "source": p.source_document,
                        "event_time": p.event_time.isoformat() if p.event_time else datetime.now().isoformat(),
                    }
                    for p in patches
                ]
                session.run(
                    """
                    UNWIND $rows AS row
                    CREATE (cp:ContentPatch {
                        uuid: row.uuid,
                        target_type: row.target_type,
                        target_absolute_id: row.target_abs_id,
                        target_family_id: row.target_family_id,
                        section_key: row.section_key,
                        change_type: row.change_type,
                        old_hash: row.old_hash,
                        new_hash: row.new_hash,
                        diff_summary: row.diff_summary,
                        source_document: row.source,
                        event_time: datetime(row.event_time)
                    })
                    WITH cp, row.target_abs_id AS abs_id
                    MATCH (t) WHERE t.uuid = abs_id
                    MERGE (cp)-[:PATCHES]->(t)
                    """,
                    rows=rows,
                )


    # ------------------------------------------------------------------

    def save_entity(self, entity: Entity):
        """保存实体到 Neo4j + sqlite-vec（合并为单条 Cypher）。"""
        self._invalidate_emb_cache()
        with _perf_timer("save_entity"):
            embedding_blob = self._compute_entity_embedding(entity)
            entity.embedding = embedding_blob
            # processed_time = 实际写入时刻（而非构造时刻）
            entity.processed_time = datetime.now()

            valid_at = (entity.valid_at or entity.event_time).isoformat()

            with self._write_lock:
                with self._session() as session:
                    self._run(session,
                        """
                        MERGE (e:Entity {uuid: $uuid})
                        SET e:Concept, e.role = 'entity',
                            e.family_id = $family_id,
                            e.name = $name,
                            e.content = $content,
                            e.event_time = datetime($event_time),
                            e.processed_time = datetime($processed_time),
                            e.episode_id = $cache_id,
                            e.source_document = $source,
                            e.summary = $summary,
                            e.attributes = $attributes,
                            e.confidence = $confidence,
                            e.content_format = $content_format,
                            e.valid_at = datetime($valid_at),
                            e.graph_id = $graph_id
                        WITH $uuid AS abs_id, $family_id AS fid, $event_time AS et
                        MATCH (e:Entity {family_id: fid})
                        WHERE e.uuid <> abs_id AND e.invalid_at IS NULL
                        SET e.invalid_at = datetime(et)
                        """,
                        uuid=entity.absolute_id,
                        family_id=entity.family_id,
                        name=entity.name,
                        content=entity.content,
                        event_time=entity.event_time.isoformat(),
                        processed_time=entity.processed_time.isoformat(),
                    cache_id=entity.episode_id,
                    source=entity.source_document,
                    summary=entity.summary,
                    attributes=entity.attributes,
                    confidence=entity.confidence,
                    content_format=getattr(entity, "content_format", "plain"),
                    valid_at=valid_at,
                    graph_id=self._graph_id,
                )

            # 存储向量
            if embedding_blob:
                emb_list = list(np.frombuffer(embedding_blob, dtype=np.float32))
                self._vector_store.upsert("entity_vectors", entity.absolute_id, emb_list)

        self._cache.invalidate("entity:")
        self._cache.invalidate("resolve:")
        self._cache.invalidate("sim_search:")



    def split_entity_version(self, absolute_id: str, new_family_id: str = "") -> Optional[Entity]:
        """将实体拆分到新的 family_id，返回更新后的 Entity。"""
        import uuid as _uuid

        if not new_family_id:
            new_family_id = f"ent_{_uuid.uuid4().hex[:12]}"

        with self._write_lock:
            with self._session() as session:
                result = self._run(session, 
                    f"""
                    MATCH (e:Entity {{uuid: $aid}})
                    SET e.family_id = $new_fid
                    RETURN {_ENTITY_RETURN_FIELDS}
                    """,
                    aid=absolute_id,
                    new_fid=new_family_id,
                )
                record = result.single()
                if not record:
                    return None
                entity = _neo4j_record_to_entity(record)
            self._cache.invalidate("entity:")
            self._cache.invalidate("resolve:")
            return entity



    def update_entity_attributes(self, family_id: str, attributes: str):
        """更新实体结构化属性。"""
        resolved = self.resolve_family_id(family_id)
        if not resolved:
            return
        with self._session() as session:
            self._run(session, """
                MATCH (e:Entity {family_id: $fid})
                WHERE e.invalid_at IS NULL
                SET e.attributes = $attributes
            """, fid=resolved, attributes=attributes)
        self._cache.invalidate("entity:")



    def update_entity_by_absolute_id(self, absolute_id: str, **fields) -> Optional[Entity]:
        """根据 absolute_id 更新指定字段，返回更新后的 Entity 或 None。"""
        valid_keys = {"name", "content", "summary", "attributes", "confidence"}
        filtered = {k: v for k, v in fields.items() if k in valid_keys and v is not None}
        if not filtered:
            return self.get_entity_by_absolute_id(absolute_id)

        with self._write_lock:
            with self._session() as session:
                set_clauses = ", ".join(f"e.{k} = ${k}" for k in filtered)
                params = dict(filtered, aid=absolute_id)
                cypher = (
                    f"MATCH (e:Entity {{uuid: $aid}}) "
                    f"SET {set_clauses} "
                    f"RETURN {_ENTITY_RETURN_FIELDS}"
                )
                result = self._run(session, cypher, **params)
                record = result.single()
                if not record:
                    return None
                entity = _neo4j_record_to_entity(record)
            self._cache.invalidate("entity:")
            return entity



    def update_entity_confidence(self, family_id: str, confidence: float):
        """更新实体最新版本的置信度。值域 [0.0, 1.0]。"""
        confidence = max(0.0, min(1.0, confidence))
        with self._session() as session:
            self._run(session, """
                MATCH (e:Entity {family_id: $fid})
                WHERE e.invalid_at IS NULL
                WITH e ORDER BY e.processed_time DESC LIMIT 1
                SET e.confidence = $confidence
            """, fid=family_id, confidence=confidence)
        self._cache.invalidate("entity:")


    # ------------------------------------------------------------------

    def update_entity_summary(self, family_id: str, summary: str):
        """更新实体摘要。"""
        resolved = self.resolve_family_id(family_id)
        if not resolved:
            return
        with self._session() as session:
            self._run(session, """
                MATCH (e:Entity {family_id: $fid})
                WHERE e.invalid_at IS NULL
                SET e.summary = $summary
            """, fid=resolved, summary=summary)
        self._cache.invalidate("entity:")

    def delete_entity_by_id(self, family_id: str) -> int:
        """删除实体的所有版本。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        with self._write_lock:
            with self._session() as session:
                # 获取所有 absolute_id
                result = self._run(session,
                    "MATCH (e:Entity {family_id: $fid}) RETURN e.uuid AS uuid",
                    fid=family_id,
                )
                uuids = [record["uuid"] for record in result]
                count = len(uuids)
                if uuids:
                    self._run(session,
                        """
                        MATCH (e:Entity {family_id: $fid})
                        DETACH DELETE e
                        """,
                        fid=family_id,
                    )
                    self._vector_store.delete_batch("entity_vectors", uuids)
                self._cache.invalidate("entity:")
                self._cache.invalidate("resolve:")
                self._cache.invalidate("sim_search:")
                self._cache.invalidate("graph_stats")
                return count

