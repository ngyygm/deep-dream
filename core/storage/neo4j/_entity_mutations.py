"""Neo4j EntityMutationMixin — all entity write/mutation methods."""
import json
import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ...models import Entity
from ._helpers import _ENTITY_RETURN_FIELDS, _fmt_dt, _neo4j_record_to_entity

logger = logging.getLogger(__name__)


class EntityMutationMixin:
    """Entity write/mutation methods.

    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              -> Neo4j session factory
        self._run(session, cypher, **kw) -> execute Cypher with graph_id injection
        self._graph_id: str          -> active graph ID
        self._write_lock             -> threading.Lock for entity writes
        self._cache                  -> QueryCache
        self.embedding_client        -> EmbeddingClient (optional)
        self.resolve_family_id(fid)  -> resolve redirect
        self.resolve_family_ids(fids)-> batch resolve redirects
    """

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
                "attributes": json.dumps(entity.attributes, ensure_ascii=False) if isinstance(entity.attributes, (dict, list)) else entity.attributes,
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

    def bulk_save_entities_with_embedding(self, entities: List[Entity]):
        """批量保存实体（UNWIND），embedding 已预计算直接写入。

        与 bulk_save_entities 不同：embedding 不延迟到后台线程，而是立即写入。
        适用于 step 9 后续 step 10 需要立即读取 embedding 的场景。
        """
        if not entities:
            return

        _now = datetime.now()
        rows = []
        cache_items = []
        for entity in entities:
            entity.processed_time = _now
            emb_blob = getattr(entity, 'embedding', None)
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
                entity.embedding = emb_array.tobytes()
                cache_items.append((entity, emb_array))
            else:
                cache_items.append((entity, None))

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
                "attributes": json.dumps(entity.attributes, ensure_ascii=False) if isinstance(entity.attributes, (dict, list)) else entity.attributes,
                "confidence": entity.confidence,
                "content_format": getattr(entity, "content_format", "plain"),
                "valid_at": _fmt_dt(entity.valid_at or entity.event_time),
                "graph_id": self._graph_id,
                "embedding": embedding_list,
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
                        e.content_format = row.content_format,
                        e.valid_at = datetime(row.valid_at),
                        e.graph_id = row.graph_id,
                        e.embedding = row.embedding
                    WITH row
                    MATCH (e:Entity {family_id: row.family_id})
                    WHERE e.uuid <> row.uuid AND e.invalid_at IS NULL
                    SET e.invalid_at = datetime(row.event_time)
                    """,
                    operation_name="bulk_save_entities_with_emb",
                    rows=rows,
                )

        for entity in entities:
            self._invalidate_entity_cache(entity.family_id)
        self._update_entity_emb_cache_batch(cache_items)

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

            # Batched delete: relations first, then edges, then entities
            deleted_entities = 0
            deleted_relations = 0
            batch_size = 200
            # Step 1: Delete invalidated relations (they reference entity UUIDs)
            while True:
                r = self._run(session, f"""
                    MATCH (r:Relation) WHERE r.invalid_at IS NOT NULL {date_filter}
                    WITH r LIMIT $batch_size
                    DETACH DELETE r
                    RETURN count(*) AS cnt
                """, graph_id_safe=False, batch_size=batch_size, **params)
                row = r.single()
                batch_del = row["cnt"] if row else 0
                deleted_relations += batch_del
                if batch_del < batch_size:
                    break
            # Step 2: Remove RELATES_TO edges pointing to invalidated entities
            # (valid entities may have edges to their old invalidated versions)
            while True:
                r = self._run(session, f"""
                    MATCH (e:Entity) WHERE e.invalid_at IS NOT NULL {date_filter}
                    MATCH (e)-[r:RELATES_TO]-()
                    WITH r LIMIT $batch_size
                    DELETE r
                    RETURN count(*) AS cnt
                """, graph_id_safe=False, batch_size=batch_size, **params)
                row = r.single()
                batch_del = row["cnt"] if row else 0
                if batch_del < batch_size:
                    break
            # Step 3: Delete invalidated entities (DETACH DELETE for remaining edges)
            while True:
                r = self._run(session, f"""
                    MATCH (e:Entity) WHERE e.invalid_at IS NOT NULL {date_filter}
                    WITH e LIMIT $batch_size
                    DETACH DELETE e
                    RETURN count(*) AS cnt
                """, graph_id_safe=False, batch_size=batch_size, **params)
                row = r.single()
                batch_del = row["cnt"] if row else 0
                deleted_entities += batch_del
                if batch_del < batch_size:
                    break

            return {
                "dry_run": False,
                "deleted_entity_versions": deleted_entities,
                "deleted_relation_versions": deleted_relations,
                "message": f"已删除 {deleted_entities} 个已失效实体版本和 {deleted_relations} 个已失效关系版本",
            }

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

    def save_content_patches(self, patches: list):
        """批量保存 ContentPatch 节点到 Neo4j。"""

        if not patches:
            return
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

    def save_entity(self, entity: Entity, _precomputed_embedding=None):
        """保存实体到 Neo4j（合并为单条 Cypher）。

        Args:
            _precomputed_embedding: Optional pre-computed embedding bytes to skip re-encoding.
        """
        from ...perf import _perf_timer
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

            # Convert embedding bytes -> LIST<FLOAT> for Neo4j node property
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
                        attributes=json.dumps(entity.attributes, ensure_ascii=False) if isinstance(entity.attributes, (dict, list)) else entity.attributes,
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
