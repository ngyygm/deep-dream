"""Neo4j EntityStoreMixin — extracted from neo4j_store."""
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
from ...content_schema import parse_markdown_sections, compute_section_diff
from ..cache import QueryCache
from ._helpers import _ENTITY_RETURN_FIELDS, _ENTITY_RETURN_FIELDS_WITH_EMB, _fmt_dt, _neo4j_record_to_entity, _neo4j_record_to_relation, _parse_dt, _q

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

    # Maximum content length used for embedding computation (avoids oversized inputs)
    _EMB_CONTENT_MAX = 512

    def _compute_entity_embedding(self, entity: Entity) -> Optional[tuple]:
        """计算实体的 embedding 向量（L2 归一化后存储）。

        Returns:
            (emb_bytes, emb_array) tuple or None. Caller can use emb_array directly
            to avoid a redundant np.frombuffer round-trip.
        """
        if not self.embedding_client or not self.embedding_client.is_available():
            return None
        content = entity.content or ""
        if len(content) > self._EMB_CONTENT_MAX:
            content = content[:self._EMB_CONTENT_MAX]
        text = f"# {entity.name}\n{content}"
        embedding = self.embedding_client.encode(text)
        if embedding is None or (isinstance(embedding, (list, tuple)) and len(embedding) == 0):
            return None
        if isinstance(embedding, np.ndarray) and embedding.size == 0:
            return None
        emb_array = np.array(embedding[0] if isinstance(embedding, list) else embedding, dtype=np.float32)
        norm = np.linalg.norm(emb_array)
        if norm > 0:
            emb_array = emb_array / norm
        return emb_array.tobytes(), emb_array



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



    def _invalidate_emb_cache(self):
        """清除 embedding 缓存（仅在不确定增量更新是否安全时调用）。"""
        self._entity_emb_cache = None
        self._entity_emb_fid_idx = None
        self._entity_emb_cache_ts = 0.0
        self._relation_emb_cache = None
        self._relation_emb_cache_ts = 0.0

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
        # Build lookup for O(1) family_id → index
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
        self._cache.invalidate_keys([
            f"entity:by_fid:{family_id}",
            f"resolve:{family_id}",
        ])

    def _invalidate_entity_cache_bulk(self):
        """Broad entity cache invalidation — only for bulk operations."""
        self._cache.invalidate("entity:")
        self._cache.invalidate("resolve:")
        self._cache.invalidate("sim_search:")



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



    def batch_delete_entities(self, family_ids: List[str]) -> int:
        """批量删除实体 — 单次事务，替代 N 次 DETACH DELETE。含向量清理。"""
        resolved_map = self.resolve_family_ids(family_ids)
        resolved = list({r for r in resolved_map.values() if r})
        if not resolved:
            return 0
        all_uuids = []
        count = 0
        with self._write_lock:
            # 单次 session：收集 uuid + DETACH DELETE
            with self._session() as session:
                result = self._run_with_retry(session,
                    "UNWIND $fids AS fid MATCH (e:Entity {family_id: fid}) RETURN e.uuid AS uuid",
                    operation_name="batch_delete_entities_collect",
                    fids=resolved,
                )
                all_uuids = [r["uuid"] for r in result]
                result = self._run_with_retry(session,
                    "UNWIND $fids AS fid MATCH (e:Entity {family_id: fid}) DETACH DELETE e RETURN count(e) AS cnt",
                    operation_name="batch_delete_entities",
                    fids=resolved,
                )
                record = result.single()
                count = record["cnt"] if record else 0
            self._invalidate_entity_cache_bulk()
            self._cache.invalidate_keys(["graph_stats"])
        return count



    def batch_delete_entity_versions_by_absolute_ids(self, absolute_ids: List[str]) -> int:
        """批量删除指定实体版本，返回成功删除的数量。含向量清理和缓存失效。"""
        if not absolute_ids:
            return 0
        with self._write_lock:
            with self._session() as session:
                result = self._run_with_retry(session,
                    """
                    MATCH (e:Entity) WHERE e.uuid IN $aids
                    DETACH DELETE e
                    RETURN count(e) AS deleted
                    """,
                    aids=absolute_ids,
                )
                record = result.single()
                deleted = record["deleted"] if record else 0
            self._invalidate_entity_cache_bulk()
            self._cache.invalidate_keys(["graph_stats"])
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
        _seen_resolved: set = set()  # O(1) membership check
        for fid in family_ids:
            resolved = resolved_map.get(fid, fid)
            if resolved and resolved not in _seen_resolved:
                canonical_map[fid] = resolved
                canonical_set.append(resolved)
                _seen_resolved.add(resolved)

        if not canonical_set:
            return [{"family_id": fid, "entity": None, "relations": [], "version_count": 0} for fid in family_ids]

        # Session 1: 批量获取实体 + 版本数 + 所有 absolute_ids（单次查询）
        # 2026-04-26: Optimized - use efficient aggregation pattern without HEAD(COLLECT())
        with self._session() as session:
            result = self._run(session,
                f"""
                MATCH (e:Entity)
                WHERE e.family_id IN $fids AND e.invalid_at IS NULL
                WITH e.family_id AS fid, e
                ORDER BY e.processed_time DESC
                WITH fid, COLLECT(e) AS entities, COUNT(e) AS vcnt
                WITH entities[0] AS latest, vcnt, [entity IN entities | entity.uuid] AS all_uuids
                RETURN latest AS e, vcnt, all_uuids
                ORDER BY e.processed_time DESC
                """,
                fids=canonical_set,
            )
            records = list(result)

        entity_map: Dict[str, tuple] = {}  # family_id -> (entity, version_count)
        fid_to_aids: Dict[str, List[str]] = {}
        all_aids = set()
        for record in records:
            entity = _neo4j_record_to_entity(record["e"])
            vc = record.get("vcnt", 1)
            entity_map[entity.family_id] = (entity, vc)
            aids = record.get("all_uuids", [])
            fid_to_aids[entity.family_id] = aids
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

            # 分配关系到对应的 family_id (O(R) via reverse lookup)
            aid_to_fid = {}
            for fid, aids in fid_to_aids.items():
                for aid in aids:
                    aid_to_fid[aid] = fid
            _seen_rel_fids = set()
            for rel in all_rels:
                fid1 = aid_to_fid.get(rel.entity1_absolute_id)
                fid2 = aid_to_fid.get(rel.entity2_absolute_id)
                if fid1:
                    relations_map[fid1].append(rel)
                if fid2 and fid2 != fid1:
                    relations_map[fid2].append(rel)

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
        """批量保存实体（UNWIND 批量写入）。

        先写入元数据（不含 embedding），embedding 在后台线程异步计算并更新。
        """
        if not entities:
            return

        # --- Phase 1: 快速写入 Neo4j（不含 embedding）---
        _now = datetime.now()
        rows = []
        for entity in entities:
            entity.processed_time = _now
            rows.append({
                "uuid": entity.absolute_id,
                "family_id": entity.family_id,
                "name": entity.name,
                "content": entity.content,
                "event_time": _fmt_dt(entity.event_time),
                "processed_time": _fmt_dt(entity.processed_time),
                "cache_id": entity.episode_id,
                "source": entity.source_document,
                "summary": entity.summary,
                "attributes": entity.attributes,
                "confidence": entity.confidence,
                "valid_at": _fmt_dt(entity.valid_at or entity.event_time),
                "graph_id": self._graph_id,
            })

        with self._write_lock:
            with self._session() as session:
                self._run_with_retry(session,
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
                    operation_name="bulk_save_entities",
                    rows=rows,
                )

        for entity in entities:
            self._invalidate_entity_cache(entity.family_id)

        # --- Phase 2: 后台线程计算 embedding + 更新 Neo4j ---
        if self.embedding_client and self.embedding_client.is_available():
            threading.Thread(
                target=self._bulk_save_embedding_bg,
                args=(list(entities),),
                daemon=True,
            ).start()

    def _bulk_save_embedding_bg(self, entities: List[Entity]):
        """后台计算 embedding 并更新到 Neo4j。"""
        try:
            texts = [f"# {e.name}\n{(e.content or '')[:self._EMB_CONTENT_MAX]}" for e in entities]
            embeddings = self.embedding_client.encode(texts)

            cache_items = []
            emb_rows = []
            for idx, entity in enumerate(entities):
                try:
                    emb_array = np.array(embeddings[idx], dtype=np.float32)
                    norm = np.linalg.norm(emb_array)
                    if norm > 0:
                        emb_array = emb_array / norm
                    entity.embedding = emb_array.tobytes()
                    embedding_list = emb_array.tolist()
                except Exception:
                    continue
                emb_rows.append({"uuid": entity.absolute_id, "embedding": embedding_list})
                cache_items.append((entity, emb_array))

            if emb_rows:
                with self._session() as session:
                    self._run_with_retry(session,
                        """
                        UNWIND $rows AS row
                        MATCH (e:Entity {uuid: row.uuid})
                        SET e.embedding = row.embedding
                        """,
                        operation_name="bulk_save_emb_update",
                        rows=emb_rows,
                    )

            self._update_entity_emb_cache_batch(cache_items)
        except Exception:
            logger.debug("Background embedding update failed", exc_info=True)



    def cleanup_invalidated_versions(self, before_date: str = None, dry_run: bool = False) -> Dict[str, Any]:
        """清理已失效的旧版本节点。

        优化：合并计数/删除查询，从 4 次串行 Neo4j 往返减少为 2 次
        （dry_run 模式 1 次，实际删除 1 次）。
        """
        with self._session() as session:
            date_filter = ""
            params = {}
            if before_date:
                date_filter = " AND n.invalid_at < datetime($before_date)"
                params["before_date"] = before_date

            # Combined count query for both entities and relations
            r = self._run(session, f"""
                CALL {{
                    MATCH (e:Entity) WHERE e.invalid_at IS NOT NULL {date_filter}
                    RETURN count(e) AS entity_count
                }}
                CALL {{
                    MATCH (r:Relation) WHERE r.invalid_at IS NOT NULL {date_filter}
                    RETURN count(r) AS relation_count
                }}
                RETURN entity_count, relation_count
            """, **params)
            row = r.single()
            entity_count = row["entity_count"] if row else 0
            relation_count = row["relation_count"] if row else 0

            if dry_run:
                return {
                    "dry_run": True,
                    "entities_to_remove": entity_count,
                    "relations_to_remove": relation_count,
                    "message": f"预览：将删除 {entity_count} 个已失效实体版本和 {relation_count} 个已失效关系版本",
                }

            # Combined delete query for both entities and relations
            r = self._run(session, f"""
                CALL {{
                    MATCH (e:Entity) WHERE e.invalid_at IS NOT NULL {date_filter}
                    DELETE e
                    RETURN count(*) AS deleted_entities
                }}
                CALL {{
                    MATCH (r:Relation) WHERE r.invalid_at IS NOT NULL {date_filter}
                    DELETE r
                    RETURN count(*) AS deleted_relations
                }}
                RETURN deleted_entities, deleted_relations
            """, graph_id_safe=False, **params)
            row = r.single()
            deleted_entities = row["deleted_entities"] if row else 0
            deleted_relations = row["deleted_relations"] if row else 0

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
        # Collect UUIDs before lock (lightweight read-only query)
        abs_ids = self._get_all_absolute_ids_for_entity(family_id)
        with self._write_lock:
            with self._session() as session:
                # 删除相关关系
                self._run_with_retry(session,
                    """MATCH (e:Entity {family_id: $fid})-[r:RELATES_TO]-()
                       DETACH DELETE r""",
                    fid=family_id,
                )
                # 删除实体节点
                result = self._run_with_retry(session,
                    "MATCH (e:Entity {family_id: $fid}) DETACH DELETE e RETURN count(e) AS cnt",
                    fid=family_id,
                )
                record = result.single()
                count = record["cnt"] if record else 0
                self._invalidate_entity_cache(family_id)
                self._cache.invalidate("sim_search:")
                self._cache.invalidate_keys(["graph_stats"])
        return count



    def delete_entity_by_absolute_id(self, absolute_id: str) -> bool:
        """根据 absolute_id 删除实体及其所有关系，返回是否成功删除。"""
        deleted = False
        fid = None
        with self._write_lock:
            with self._session() as session:
                result = self._run_with_retry(session,
                    "MATCH (e:Entity {uuid: $aid}) DETACH DELETE e RETURN count(e) AS cnt, e.family_id AS fid",
                    aid=absolute_id,
                )
                record = result.single()
                deleted = record is not None and record["cnt"] > 0
                fid = record["fid"] if record else None
            if fid:
                self._invalidate_entity_cache(fid)
            else:
                self._invalidate_entity_cache_bulk()
            self._cache.invalidate_keys(["graph_stats"])
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
                RETURN {(_ENTITY_RETURN_FIELDS if exclude_embedding else _ENTITY_RETURN_FIELDS_WITH_EMB)}
                ORDER BY e.processed_time DESC
            """
            if offset is not None and offset > 0:
                query += f" SKIP {int(offset)}"
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = self._run(session, query)
            records = list(result)

        return [_neo4j_record_to_entity(r) for r in records]



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
                RETURN {(_ENTITY_RETURN_FIELDS if exclude_embedding else _ENTITY_RETURN_FIELDS_WITH_EMB)}
                ORDER BY e.processed_time DESC
            """
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = self._run(session, query, tp=time_point.isoformat())
            records = list(result)

        return [_neo4j_record_to_entity(r) for r in records]



    def get_content_patches(self, family_id: str, section_key: str = None) -> list:
        """查询指定 family_id 的 ContentPatch 记录。"""
        _PATCH_FIELDS = (
            "cp.uuid AS uuid, cp.target_type AS target_type, "
            "cp.target_absolute_id AS target_absolute_id, cp.target_family_id AS target_family_id, "
            "cp.section_key AS section_key, cp.change_type AS change_type, "
            "cp.old_hash AS old_hash, cp.new_hash AS new_hash, "
            "cp.diff_summary AS diff_summary, cp.source_document AS source_document, "
            "cp.event_time AS event_time"
        )
        with self._session() as session:
            if section_key:
                result = self._run(session,
                    f"MATCH (cp:ContentPatch {{target_family_id: $fid, section_key: $sk}}) "
                    f"RETURN {_PATCH_FIELDS} ORDER BY cp.event_time DESC",
                    graph_id_safe=False,
                    fid=family_id, sk=section_key,
                )
            else:
                result = self._run(session,
                    f"MATCH (cp:ContentPatch {{target_family_id: $fid}}) "
                    f"RETURN {_PATCH_FIELDS} ORDER BY cp.event_time DESC",
                    graph_id_safe=False,
                    fid=family_id,
                )
            patches = []
            for record in result:
                patches.append(ContentPatch(
                    uuid=record["uuid"],
                    target_type=record["target_type"],
                    target_absolute_id=record["target_absolute_id"],
                    target_family_id=record["target_family_id"],
                    section_key=record["section_key"],
                    change_type=record["change_type"],
                    old_hash=record.get("old_hash", ""),
                    new_hash=record.get("new_hash", ""),
                    diff_summary=record.get("diff_summary", ""),
                    source_document=record.get("source_document", ""),
                    event_time=_parse_dt(record.get("event_time")),
                ))
            return patches



    def get_data_quality_report(self) -> Dict[str, Any]:
        """返回数据质量报告（合并查询，减少 Neo4j 往返次数）。"""
        with self._session() as session:
            # Query 1: All entity + relation counts in one aggregated query
            r = self._run(session, """
                CALL () {
                    MATCH (e:Entity) WHERE e.invalid_at IS NULL AND e.family_id IS NOT NULL
                    RETURN count(DISTINCT e.family_id) AS ent_valid_families, count(e) AS ent_valid_nodes
                }
                CALL () {
                    MATCH (e:Entity) WHERE e.invalid_at IS NOT NULL RETURN count(e) AS ent_invalidated
                }
                CALL () {
                    MATCH (e:Entity) WHERE e.family_id IS NULL RETURN count(e) AS ent_no_fid
                }
                CALL () {
                    MATCH (rel:Relation) WHERE rel.invalid_at IS NULL
                    RETURN count(DISTINCT rel.family_id) AS rel_valid_families, count(rel) AS rel_valid_nodes
                }
                CALL () {
                    MATCH (rel:Relation) WHERE rel.invalid_at IS NOT NULL RETURN count(rel) AS rel_invalidated
                }
                RETURN ent_valid_families, ent_valid_nodes, ent_invalidated, ent_no_fid,
                       rel_valid_families, rel_valid_nodes, rel_invalidated
            """)
            row = r.single()
            valid_families = row["ent_valid_families"]
            valid_nodes = row["ent_valid_nodes"]
            invalidated_entity_versions = row["ent_invalidated"]
            no_family_id = row["ent_no_fid"]
            valid_relation_families = row["rel_valid_families"]
            valid_relation_nodes = row["rel_valid_nodes"]
            invalidated_relation_versions = row["rel_invalidated"]

            # Query 2: Dangling ref detection (relation endpoints vs valid entity UUIDs)
            r = self._run(session, """
                CALL () {
                    MATCH (rel:Relation) WHERE rel.invalid_at IS NULL
                    WITH collect(DISTINCT rel.entity1_absolute_id) + collect(DISTINCT rel.entity2_absolute_id) AS rel_aids
                    UNWIND rel_aids AS aid
                    RETURN collect(DISTINCT aid) AS all_rel_aids
                }
                CALL () {
                    MATCH (e:Entity) WHERE e.invalid_at IS NULL
                    RETURN collect(DISTINCT e.uuid) AS ent_uuids
                }
                RETURN all_rel_aids, ent_uuids
            """)
            row = r.single()
            rel_aids = set(row["all_rel_aids"] or ())
            valid_uuids = set(row["ent_uuids"] or ())
            dangling_refs = len(rel_aids - valid_uuids)

        # 孤立实体
        isolated_count = self.count_isolated_entities()

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



    def get_entities_by_absolute_ids(self, absolute_ids: List[str], valid_only: bool = False) -> List[Entity]:
        """批量根据 absolute_id 获取实体。"""
        if not absolute_ids:
            return []
        with self._session() as session:
            extra_filter = " AND e.invalid_at IS NULL" if valid_only else ""
            result = self._run(session,
                f"""
                MATCH (e:Entity)
                WHERE e.uuid IN $uuids{extra_filter}
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
                    f"RETURN fid, latest.name AS name, latest.content AS content, latest.uuid AS uuid, latest.family_id AS family_id, latest.summary AS summary, latest.attributes AS attributes, latest.confidence AS confidence, latest.content_format AS content_format, latest.community_id AS community_id, latest.valid_at AS valid_at, latest.invalid_at AS invalid_at, latest.event_time AS event_time, latest.processed_time AS processed_time, latest.episode_id AS episode_id, latest.source_document AS source_document, latest.embedding AS embedding"
                )
                records = self._run(session, cypher, fids=list(uncached)).data()
                entities = [_neo4j_record_to_entity(rec) for rec in records]
                for entity in entities:
                    fid = entity.family_id
                    result[fid] = entity
                    self._cache.set(f"entity:by_fid:{fid}", entity, ttl=60)
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
                RETURN {_ENTITY_RETURN_FIELDS_WITH_EMB}
                """,
                uuid=absolute_id,
            )
            record = result.single()
            if not record:
                return None
            entity = _neo4j_record_to_entity(record)
            self._cache.set(cache_key, entity, ttl=60)
            return entity



    def get_entity_by_family_id(self, family_id: str) -> Optional[Entity]:
        """根据 family_id 获取最新版本的实体。"""
        # Fast path: check cache with raw family_id before resolve
        cache_key = f"entity:by_fid:{family_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        # Also check with canonical family_id in case it was previously resolved
        resolved_fid = self.resolve_family_id(family_id)
        if resolved_fid != family_id:
            cache_key2 = f"entity:by_fid:{resolved_fid}"
            cached = self._cache.get(cache_key2)
            if cached is not None:
                self._cache.set(cache_key, cached, ttl=60)  # warm raw key
                return cached
        family_id = resolved_fid
        if not family_id:
            return None
        with _perf_timer("get_entity_by_family_id"):
            with self._session() as session:
                result = self._run(session,
                    f"""
                    MATCH (e:Entity {{family_id: $fid}})
                    RETURN {_ENTITY_RETURN_FIELDS_WITH_EMB}
                    ORDER BY e.processed_time DESC LIMIT 1
                    """,
                    fid=family_id,
                )
                record = result.single()
                if not record:
                    return None
                entity = _neo4j_record_to_entity(record)
                self._cache.set(cache_key, entity, ttl=60)
                if resolved_fid != family_id:
                    self._cache.set(f"entity:by_fid:{family_id}", entity, ttl=60)
                return entity



    def get_entity_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        """获取实体 embedding 预览。"""
        with self._session() as session:
            result = self._run(session,
                "MATCH (e:Entity {uuid: $uuid}) RETURN e.embedding AS embedding",
                uuid=absolute_id,
            )
            record = result.single()
            if record and record["embedding"]:
                return record["embedding"][:num_values]
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

        查询所有版本的 MENTIONS 边。
        先查找 Episode->Entity 的直接 MENTIONS；如果无结果，
        再查找通过该实体参与的关系（Episode->Relation 的 MENTIONS）间接关联的 Episode。

        优化：合并为单次 Cypher 查询（collect absolute_ids + direct MENTIONS +
        indirect MENTIONS），避免 3 次串行 Neo4j 往返。

        注意：Episode 节点可能缺少 graph_id（旧数据），因此只对 Entity/Relation 侧
        进行 graph_id 过滤，不依赖 Episode 的 graph_id。
        """
        with self._session() as session:
            # Single query: collect absolute_ids, try direct MENTIONS first,
            # fall back to indirect MENTIONS via relations
            result = self._run(session, """
                MATCH (e:Entity {family_id: $fid})
                WITH collect(e.uuid) AS abs_ids
                CALL () {
                    WITH abs_ids
                    MATCH (ep:Episode)-[m:MENTIONS]->(e:Entity)
                    WHERE e.uuid IN abs_ids AND e.graph_id = $graph_id
                    RETURN DISTINCT ep.uuid AS episode_id, m.context AS context, 0 AS priority
                }
                RETURN episode_id, context, priority
                ORDER BY priority
            """, fid=family_id, graph_id=self._graph_id, graph_id_safe=False)
            provenance = [{"episode_id": r["episode_id"], "context": r.get("context", "")} for r in result]

            if provenance:
                return provenance

            # Fallback: indirect MENTIONS via relations (only if direct found nothing)
            # Re-collect abs_ids in subquery to avoid carrying state
            result = self._run(session, """
                MATCH (e:Entity {family_id: $fid})
                WITH collect(e.uuid) AS abs_ids
                UNWIND abs_ids AS aid
                MATCH (ep:Episode)-[m:MENTIONS]->(r:Relation)
                WHERE (r.entity1_absolute_id = aid OR r.entity2_absolute_id = aid)
                      AND r.graph_id = $graph_id
                RETURN DISTINCT ep.uuid AS episode_id, m.context AS context
            """, fid=family_id, graph_id=self._graph_id, graph_id_safe=False)
            return [{"episode_id": r["episode_id"], "context": r.get("context", "")} for r in result]




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
        """批量获取实体在各版本时间点的关系（消除 N+1 查询）。

        优化：合并为单次 Cypher 查询（collect abs_ids + version times + relations），
        从 3 次串行 Neo4j 往返减少为 1 次。
        """
        family_id = self.resolve_family_id(family_id)
        if not family_id or not version_abs_ids:
            return []

        with self._session() as session:
            # Single combined query: collect abs_ids + get version times + get relations
            # Uses CALL subqueries to keep the pipeline in one round-trip
            result = self._run(session, """
                // Step 1: Collect all absolute_ids for this family_id
                MATCH (e:Entity {family_id: $fid})
                WITH collect(e.uuid) AS abs_ids

                // Step 2: Get version processed_times
                CALL () {
                    WITH abs_ids
                    MATCH (e:Entity)
                    WHERE e.uuid IN $version_abs_ids
                    RETURN e.uuid AS uuid, e.processed_time AS pt
                    ORDER BY e.processed_time ASC
                }
                WITH abs_ids, collect({uuid: uuid, pt: pt}) AS version_times

                // Step 3: Get latest-version relations referencing any abs_id
                CALL () {
                    WITH abs_ids
                    UNWIND abs_ids AS aid
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id = aid OR r.entity2_absolute_id = aid)
                    WITH r.family_id AS fid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH fid, r ORDER BY r.processed_time DESC
                    WITH fid, HEAD(COLLECT(r)) AS r
                    RETURN r.uuid AS uuid, r.family_id AS family_id,
                           r.content AS content, r.event_time AS event_time,
                           r.processed_time AS processed_time
                }

                // Step 4: Return both version_times and relations
                RETURN version_times,
                       collect({
                           uuid: uuid, family_id: family_id,
                           content: content, event_time: event_time,
                           processed_time: processed_time
                       }) AS relations
            """, fid=family_id, version_abs_ids=version_abs_ids, graph_id_safe=False)

            record = result.single()
            if not record:
                return []

            version_times = record["version_times"]
            if not version_times:
                return []

            relations = record["relations"]

            # Filter: only include relations that appeared before at least one version time point
            timeline = []
            seen = set()
            for rel in relations:
                rel_uuid = rel["uuid"]
                if rel_uuid in seen:
                    continue
                rel_pt = rel["processed_time"]
                for v in version_times:
                    v_pt = v["pt"]
                    if rel_pt and v_pt and rel_pt <= v_pt:
                        seen.add(rel_uuid)
                        timeline.append({
                            "family_id": rel["family_id"],
                            "content": rel["content"],
                            "event_time": _fmt_dt(rel["event_time"]) if rel["event_time"] else None,
                            "absolute_id": rel_uuid,
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
                RETURN {_ENTITY_RETURN_FIELDS_WITH_EMB}
                ORDER BY e.processed_time DESC LIMIT 1
                """,
                fid=family_id,
                tp=time_point.isoformat(),
            )
            record = result.single()
            if not record:
                return None
            entity = _neo4j_record_to_entity(record)
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
        canonical_ids = list({r for r in resolved_map.values() if r})
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

    def count_entity_relations_by_family_ids(self, family_ids: List[str]) -> Dict[str, int]:
        """批量获取多个 family_id 的关系数量（单个 Cypher 聚合查询）。"""
        if not family_ids:
            return {}
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = list({r for r in resolved_map.values() if r})
        if not canonical_ids:
            return {}
        with self._session() as session:
            result = self._run(session,
                """
                UNWIND $fids AS fid
                MATCH (e:Entity {family_id: fid})
                WITH fid, e.uuid AS aid
                OPTIONAL MATCH (r:Relation)
                WHERE r.entity1_absolute_id = aid OR r.entity2_absolute_id = aid
                RETURN fid AS family_id, count(DISTINCT r) AS cnt
                """,
                fids=canonical_ids,
            )
            counts = {record["family_id"]: record["cnt"] for record in result}
            # Map back to original (pre-resolve) family_ids
            return {orig: counts.get(resolved, 0) for orig, resolved in resolved_map.items() if resolved}



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

    def get_entity_versions_batch(self, family_ids: List[str]) -> Dict[str, List[Entity]]:
        """批量获取多个 family_id 的所有版本（单次 Cypher 查询）。"""
        if not family_ids:
            return {}
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = list({r for r in resolved_map.values() if r})
        if not canonical_ids:
            return {}
        with self._session() as session:
            result = self._run(session,
                f"""
                UNWIND $fids AS fid
                MATCH (e:Entity {{family_id: fid}})
                RETURN e.family_id AS fid, {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time ASC
                """,
                fids=canonical_ids,
            )
            versions_map: Dict[str, List[Entity]] = {fid: [] for fid in canonical_ids}
            for record in result:
                fid = record["fid"]
                if fid in versions_map:
                    versions_map[fid].append(_neo4j_record_to_entity(record))
        return {orig: versions_map.get(resolved, []) for orig, resolved in resolved_map.items() if resolved}



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
                     count(DISTINCT valid_r.family_id) AS relation_count
                // 度数：统计每个 family 有多少条不同的 Relation
                UNWIND CASE WHEN entity_count > 0 THEN [1] ELSE [] END AS _trigger
                MATCH (e:Entity) WHERE e.invalid_at IS NULL AND e.family_id IS NOT NULL
                WITH total_entity_versions, total_relation_versions, entity_count,
                     relation_count, e.family_id AS fid, collect(DISTINCT e.uuid) AS uuids
                UNWIND uuids AS uid
                OPTIONAL MATCH (r:Relation) WHERE r.invalid_at IS NULL
                    AND (r.entity1_absolute_id = uid OR r.entity2_absolute_id = uid)
                WITH total_entity_versions, total_relation_versions, entity_count,
                     relation_count, fid, count(DISTINCT r) AS degree
                RETURN total_entity_versions, total_relation_versions,
                       entity_count, relation_count,
                       avg(toFloat(degree)) AS avg_degree,
                       max(degree) AS max_degree_raw,
                       sum(CASE WHEN degree = 0 THEN 1 ELSE 0 END) AS isolated_count
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
                isolated = row["isolated_count"]
                stats["avg_relations_per_entity"] = round(row["avg_degree"], 2)
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
                r = self._run(session,
                    "MATCH (e:Entity) WHERE e.invalid_at IS NULL "
                    "WITH count(DISTINCT e.family_id) AS ec "
                    "MATCH (r:Relation) WHERE r.invalid_at IS NULL "
                    "RETURN ec AS entity_count, count(DISTINCT r.family_id) AS relation_count"
                )
                row = r.single()
                return {"entities": row["entity_count"], "relations": row["relation_count"]}
        except Exception as e:
            logger.warning("get_stats failed: %s", e)
            return {"entities": 0, "relations": 0}



    def get_version_diff(self, family_id: str, v1: str, v2: str) -> dict:
        """获取两个版本之间的 section 级 diff。

        v1, v2 是两个 absolute_id（版本 uuid）。
        返回 {section_key: {"v1": content_or_None, "v2": content_or_None, "changed": bool}}
        """
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
                        "event_time": _fmt_dt(p.event_time) if p.event_time else datetime.now().isoformat(),
                    }
                    for p in patches
                ]
                self._run(session,
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
                    graph_id_safe=False,
                    rows=rows,
                )


    # ------------------------------------------------------------------

    def save_entity(self, entity: Entity, _precomputed_embedding=None):
        """保存实体到 Neo4j（合并为单条 Cypher）。

        Args:
            _precomputed_embedding: Optional pre-computed embedding bytes to skip re-encoding.
        """
        with _perf_timer("save_entity"):
            _emb_array = None  # keep numpy array reference to avoid np.frombuffer round-trip
            if _precomputed_embedding is not None:
                embedding_blob = _precomputed_embedding
            else:
                _emb_result = self._compute_entity_embedding(entity)
                if _emb_result is not None:
                    embedding_blob, _emb_array = _emb_result
                else:
                    embedding_blob = None
            entity.embedding = embedding_blob
            # processed_time = 实际写入时刻（而非构造时刻）
            entity.processed_time = datetime.now()

            valid_at = _fmt_dt(entity.valid_at or entity.event_time)

            # Convert embedding bytes → LIST<FLOAT> for Neo4j node property
            embedding_list = None
            if embedding_blob:
                if _emb_array is not None:
                    emb_array = _emb_array
                else:
                    emb_array = np.frombuffer(embedding_blob, dtype=np.float32)
                embedding_list = emb_array.tolist()

            with self._write_lock:
                with self._session() as session:
                    self._run_with_retry(session,
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
                            e.graph_id = $graph_id,
                            e.embedding = $embedding
                        WITH $uuid AS abs_id, $family_id AS fid, $event_time AS et
                        MATCH (e:Entity {family_id: fid})
                        WHERE e.uuid <> abs_id AND e.invalid_at IS NULL
                        SET e.invalid_at = datetime(et)
                        """,
                        operation_name="save_entity",
                        uuid=entity.absolute_id,
                        family_id=entity.family_id,
                        name=entity.name,
                        content=entity.content,
                        event_time=_fmt_dt(entity.event_time),
                        processed_time=_fmt_dt(entity.processed_time),
                    cache_id=entity.episode_id,
                    source=entity.source_document,
                    summary=entity.summary,
                    attributes=entity.attributes,
                    confidence=entity.confidence,
                    content_format=getattr(entity, "content_format", "plain"),
                    valid_at=valid_at,
                    graph_id=self._graph_id,
                    embedding=embedding_list,
                )

            # Incremental cache update
            if embedding_blob:
                emb_array = _emb_array if _emb_array is not None else np.frombuffer(embedding_blob, dtype=np.float32)
                self._update_entity_emb_cache(entity, emb_array)
            else:
                self._update_entity_emb_cache(entity, None)

        self._invalidate_entity_cache(entity.family_id)
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
            self._invalidate_entity_cache(new_family_id)
            self._invalidate_entity_cache(absolute_id)  # old absolute_id may have been cached
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
        self._invalidate_entity_cache(resolved)



    def update_entity_by_absolute_id(self, absolute_id: str, **fields) -> Optional[Entity]:
        """根据 absolute_id 更新指定字段，返回更新后的 Entity 或 None。

        当 name 或 content 变更时自动重算 embedding 并更新。
        """
        valid_keys = {"name", "content", "summary", "attributes", "confidence"}
        filtered = {k: v for k, v in fields.items() if k in valid_keys and v is not None}
        if not filtered:
            return self.get_entity_by_absolute_id(absolute_id)

        needs_emb_update = "name" in filtered or "content" in filtered

        # Compute embedding BEFORE acquiring write lock (ML inference is slow)
        _precomputed_emb = None
        if needs_emb_update and self.embedding_client and self.embedding_client.is_available():
            # Fetch current entity to compute embedding from (name + content)
            current = self.get_entity_by_absolute_id(absolute_id)
            if current:
                _merged = Entity(
                    name=filtered.get("name", current.name),
                    content=filtered.get("content", current.content),
                )
                _emb_result = self._compute_entity_embedding(_merged)
                if _emb_result is not None:
                    _precomputed_emb = _emb_result

        embedding_list = None
        if _precomputed_emb is not None:
            embedding_blob, emb_array = _precomputed_emb
            embedding_list = emb_array.tolist()

        with self._write_lock:
            with self._session() as session:
                set_parts = [f"e.{k} = ${k}" for k in filtered]
                params = {**filtered, "aid": absolute_id}
                if _precomputed_emb is not None:
                    set_parts.append("e.embedding = $embedding")
                    params["embedding"] = embedding_list
                set_clauses = ", ".join(set_parts)
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

            self._invalidate_entity_cache(entity.family_id)

        # Cache update
        if _precomputed_emb is not None:
            entity.embedding = embedding_blob
            self._update_entity_emb_cache(entity, emb_array)
        elif needs_emb_update:
            self._update_entity_emb_cache(entity, None)

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
        self._invalidate_entity_cache(family_id)


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
        self._invalidate_entity_cache(resolved)

    def batch_update_entity_summaries(self, updates: Dict[str, str]):
        """Batch update summaries for multiple entities in a single session."""
        if not updates:
            return
        resolved_map = self.resolve_family_ids(list(updates))
        rows = []
        for orig_fid, summary in updates.items():
            resolved = resolved_map.get(orig_fid)
            if resolved:
                rows.append({"fid": resolved, "summary": summary})
        if not rows:
            return
        with self._session() as session:
            self._run(session, """
                UNWIND $rows AS row
                MATCH (e:Entity {family_id: row.fid})
                WHERE e.invalid_at IS NULL
                SET e.summary = row.summary
            """, rows=rows)
        self._invalidate_entity_cache_bulk()

    def get_family_ids_by_names(self, names: list) -> dict:
        """按名称批量查询实体的 family_id（每个 name 取最新版本）。

        Returns:
            {name: family_id} 仅包含能找到的名称。
        """
        if not names:
            return {}
        with self._session() as session:
            result = self._run(session,
                """
                MATCH (e:Entity)
                WHERE e.name IN $names
                WITH e.name AS name, e.family_id AS fid, e.processed_time AS pt
                ORDER BY name, pt DESC
                WITH name, COLLECT(fid)[0] AS latest_fid
                RETURN name, latest_fid
                """,
                names=names,
            )
            return {record["name"]: record["latest_fid"] for record in result}

    def delete_entity_by_id(self, family_id: str) -> int:
        """删除实体的所有版本。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        uuids = []
        count = 0
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
            self._invalidate_entity_cache(family_id)
            self._cache.invalidate("sim_search:")
            self._cache.invalidate_keys(["graph_stats"])
        return count

