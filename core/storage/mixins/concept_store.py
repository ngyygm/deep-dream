"""
ConceptStoreMixin — concept-related and dream-log storage methods.

Extracted from StorageManager.  Relies on host-class state:
    self._get_conn()
    self._write_lock
    self.embedding_client
    self._concept_emb_cache / self._concept_emb_cache_ts / self._emb_cache_ttl
    self._invalidate_emb_cache()
    self._safe_parse_datetime()
    self.get_concept_by_family_id()
    self.get_concepts_by_family_ids()
    self.get_episode_entities()
"""
import json
import logging
import time
from datetime import datetime
from typing import Any, Optional, List, Dict

import numpy as np

from ...models import Entity, Relation, Episode

logger = logging.getLogger(__name__)


class ConceptStoreMixin:
    """Mixin providing concept table CRUD and dream-log storage."""

    # ------------------------------------------------------------------
    # Dream candidate filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _is_dream_candidate_concept(concept: dict) -> bool:
        """Check if a concept dict represents an unverified dream candidate relation."""
        if concept.get("role") != "relation":
            return False
        attrs = concept.get("attributes") or ""
        if isinstance(attrs, str):
            try:
                attrs = json.loads(attrs) if attrs else {}
            except (json.JSONDecodeError, TypeError):
                return False
        if not isinstance(attrs, dict):
            return False
        return attrs.get("tier") == "candidate" and attrs.get("status") == "hypothesized"

    # ========== Phase E: Dream Logs ==========

    def save_dream_log(self, report):
        """保存梦境报告。"""
        # Pre-serialize JSON OUTSIDE lock (CPU-bound work)
        _insights = json.dumps(report.insights, ensure_ascii=False)
        _connections = json.dumps(report.new_connections, ensure_ascii=False)
        _consolidations = json.dumps(report.consolidations, ensure_ascii=False)
        _config = json.dumps({}, ensure_ascii=False)
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO dream_logs
                (cycle_id, graph_id, start_time, end_time, status, narrative,
                 insights_json, connections_json, consolidations_json, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.cycle_id, report.graph_id,
                report.start_time.isoformat(),
                report.end_time.isoformat() if report.end_time else None,
                report.status, report.narrative,
                _insights, _connections, _consolidations, _config,
            ))
            conn.commit()

    def list_dream_logs(self, graph_id: str = "default", limit: int = 20) -> List[dict]:
        """列出梦境日志。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT cycle_id, graph_id, start_time, end_time, status, narrative
            FROM dream_logs WHERE graph_id = ?
            ORDER BY start_time DESC LIMIT ?
        """, (graph_id, limit))
        return [
            {"cycle_id": row[0], "graph_id": row[1], "start_time": row[2],
             "end_time": row[3], "status": row[4], "narrative": (row[5] or "")[:200]}
            for row in cursor.fetchall()
        ]

    def get_dream_log(self, cycle_id: str) -> Optional[dict]:
        """获取单次梦境日志详情。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT cycle_id, graph_id, start_time, end_time, status, narrative,
                   insights_json, connections_json, consolidations_json
            FROM dream_logs WHERE cycle_id = ?
        """, (cycle_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "cycle_id": row[0], "graph_id": row[1], "start_time": row[2],
            "end_time": row[3], "status": row[4], "narrative": row[5] or "",
            "insights": json.loads(row[6]) if row[6] else [],
            "connections": json.loads(row[7]) if row[7] else [],
            "consolidations": json.loads(row[8]) if row[8] else [],
        }

    # ========== Phase 2: concepts 统一表双写 ==========

    # Concept embedding text max length (sentence-transformers context limit)
    _CONCEPT_EMB_TEXT_MAX = 512

    def _compute_concept_embedding(self, role: str, name: str = "",
                                    content: str = "", extra: str = "") -> Optional[bytes]:
        """计算概念级别的独立 embedding。

        与 legacy Entity/Relation embedding 不同，concept embedding 编码完整概念信息：
        - entity concept: "{name}: {content}"
        - relation concept: "{extra} {content}"  (extra 包含两端实体名称)
        - observation concept: "{content}"

        这样 concept 层的向量搜索能区分不同角色的概念。
        """
        if not self.embedding_client or not self.embedding_client.is_available():
            return None

        parts = []
        if role == 'entity' and name:
            parts.append(f"{name}:")
        elif role == 'relation' and extra:
            parts.append(extra)

        if content:
            parts.append(content[:self._CONCEPT_EMB_TEXT_MAX])

        text = " ".join(parts) if parts else ""
        if not text:
            return None

        embedding = self.embedding_client.encode(text)
        if embedding is None or not embedding:
            return None

        embedding_array = np.array(
            embedding[0] if isinstance(embedding, list) else embedding,
            dtype=np.float32
        )
        return embedding_array.tobytes()

    def _write_concept_from_entity(self, entity: Entity, cursor, precomputed_embedding: Optional[bytes] = None):
        """Dual-write: write Entity to concepts table with concept-level embedding."""
        try:
            # Use pre-computed embedding if available, otherwise compute on the fly
            concept_emb = precomputed_embedding
            if concept_emb is None:
                concept_emb = self._compute_concept_embedding(
                    role='entity', name=entity.name, content=entity.content
                )
            cursor.execute("""
                INSERT OR REPLACE INTO concepts
                (id, family_id, role, name, content, event_time, processed_time,
                 source_document, episode_id, embedding, valid_at, invalid_at,
                 summary, attributes, confidence, content_format, provenance)
                VALUES (?, ?, 'entity', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '')
            """, (
                entity.absolute_id,
                entity.family_id,
                entity.name,
                entity.content,
                entity.event_time.isoformat(),
                entity.processed_time.isoformat(),
                entity.source_document or '',
                entity.episode_id or '',
                concept_emb,
                (entity.valid_at or entity.event_time).isoformat(),
                getattr(entity, 'invalid_at', None),
                getattr(entity, 'summary', None),
                json.dumps(_attrs) if isinstance(_attrs := getattr(entity, 'attributes', None), dict) else _attrs,
                getattr(entity, 'confidence', None),
                getattr(entity, 'content_format', 'plain'),
            ))
            # FTS — use lastrowid from INSERT OR REPLACE (always correct since REPLACE = DELETE+INSERT)
            try:
                _rowid = cursor.lastrowid
                if _rowid:
                    cursor.execute("""
                        INSERT INTO concept_fts(rowid, name, content)
                        VALUES (?, ?, ?)
                    """, (_rowid, entity.name, entity.content))
            except Exception as exc:
                logger.debug("concept_fts entity write failed: %s", exc)
        except Exception as exc:
            logger.debug("concept entity dual-write failed: %s", exc)

    def _write_concept_from_relation(self, relation: Relation, cursor,
                                      e1_name: str = None, e2_name: str = None,
                                      precomputed_embedding: Optional[bytes] = None):
        """Dual-write: write Relation to concepts table with concept-level embedding.

        Concept embedding 包含两端实体名称 + 关系内容，使 concept 层语义搜索
        能区分不同实体对之间的同类关系。

        Args:
            e1_name, e2_name: Pre-resolved entity names (avoids N+1 when called in bulk).
            precomputed_embedding: Pre-computed concept embedding bytes (skips per-item encode).
        """
        try:
            connects = json.dumps([relation.entity1_absolute_id, relation.entity2_absolute_id])
            # Use pre-resolved names if provided, otherwise resolve individually
            if e1_name is None:
                try:
                    e1 = self.get_entity_by_absolute_id(relation.entity1_absolute_id)
                    e1_name = e1.name if e1 else ""
                except Exception:
                    e1_name = ""
            if e2_name is None:
                try:
                    e2 = self.get_entity_by_absolute_id(relation.entity2_absolute_id)
                    e2_name = e2.name if e2 else ""
                except Exception:
                    e2_name = ""
            # Use pre-computed embedding if available, otherwise compute on the fly
            concept_emb = precomputed_embedding
            if concept_emb is None:
                concept_emb = self._compute_concept_embedding(
                    role='relation', content=relation.content,
                    extra=f"{e1_name} {e2_name}".strip()
                )
            cursor.execute("""
                INSERT OR REPLACE INTO concepts
                (id, family_id, role, name, content, event_time, processed_time,
                 source_document, episode_id, embedding, valid_at, invalid_at,
                 summary, attributes, confidence, content_format, provenance, connects)
                VALUES (?, ?, 'relation', '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                relation.absolute_id,
                relation.family_id,
                relation.content,
                relation.event_time.isoformat(),
                relation.processed_time.isoformat(),
                relation.source_document or '',
                relation.episode_id or '',
                concept_emb,
                (relation.valid_at or relation.event_time).isoformat(),
                getattr(relation, 'invalid_at', None),
                getattr(relation, 'summary', None),
                json.dumps(_attrs) if isinstance(_attrs := getattr(relation, 'attributes', None), dict) else _attrs,
                getattr(relation, 'confidence', None),
                getattr(relation, 'content_format', 'plain'),
                getattr(relation, 'provenance', '') or '',
                connects,
            ))
            # FTS — use lastrowid from INSERT OR REPLACE
            try:
                _rowid = cursor.lastrowid
                if _rowid:
                    cursor.execute("""
                        INSERT INTO concept_fts(rowid, name, content)
                        VALUES (?, '', ?)
                    """, (_rowid, relation.content))
            except Exception as exc:
                logger.debug("concept_fts relation write failed: %s", exc)
        except Exception as exc:
            logger.debug("concept relation dual-write failed: %s", exc)

    def _write_concept_from_episode(self, cache: Episode, doc_hash: str, cursor,
                                     family_id: str = None,
                                     precomputed_embedding: Optional[bytes] = None,
                                     _now_iso: Optional[str] = None):
        """Dual-write: write Episode to concepts table as 'observation' with embedding.

        Args:
            family_id: Optional resolved family_id. If provided, uses it for versioning.
                       Falls back to cache.absolute_id for standalone observations.
            precomputed_embedding: Pre-computed concept embedding bytes (skips per-item compute).
            _now_iso: Pre-computed ISO timestamp string (skips datetime.now() call inside lock).
        """
        try:
            fid = family_id or cache.absolute_id
            # Use pre-computed embedding if available, otherwise compute on the fly
            concept_emb = precomputed_embedding
            if concept_emb is None:
                concept_emb = self._compute_concept_embedding(
                    role='observation', content=cache.content[:self._CONCEPT_EMB_TEXT_MAX]
                )
            cursor.execute("""
                INSERT OR REPLACE INTO concepts
                (id, family_id, role, name, content, event_time, processed_time,
                 source_document, activity_type, episode_type, provenance, embedding)
                VALUES (?, ?, 'observation', '', ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache.absolute_id,
                fid,
                cache.content,
                cache.event_time.isoformat(),
                _now_iso or datetime.now().isoformat(),
                cache.source_document or '',
                cache.activity_type or '',
                getattr(cache, 'episode_type', '') or '',
                json.dumps({"doc_hash": doc_hash}) if doc_hash else '',
                concept_emb,
            ))
            # FTS — use lastrowid from INSERT OR REPLACE
            try:
                _rowid = cursor.lastrowid
                if _rowid:
                    cursor.execute("""
                        INSERT INTO concept_fts(rowid, name, content)
                        VALUES (?, '', ?)
                    """, (_rowid, cache.content))
            except Exception as exc:
                logger.debug("concept_fts episode write failed: %s", exc)
        except Exception as exc:
            logger.debug("concept episode dual-write failed: %s", exc)

    def _delete_concept_by_id(self, absolute_id: str, cursor):
        """Delete a concept version by absolute_id (called within existing write transaction)."""
        try:
            # Get integer rowid before delete for FTS cleanup
            cursor.execute("SELECT rowid FROM concepts WHERE id = ?", (absolute_id,))
            _row = cursor.fetchone()
            if _row:
                try:
                    cursor.execute("DELETE FROM concept_fts WHERE rowid = ?", (_row[0],))
                except Exception as exc:
                    logger.debug("concept_fts delete by id failed: %s", exc)
            cursor.execute("DELETE FROM concepts WHERE id = ?", (absolute_id,))
        except Exception as exc:
            logger.debug("concept delete by id failed: %s", exc)

    def _delete_concepts_by_family(self, family_id: str, cursor):
        """Delete all concept versions by family_id (called within existing write transaction)."""
        try:
            # Collect integer rowids before delete for FTS cleanup
            cursor.execute("SELECT rowid FROM concepts WHERE family_id = ?", (family_id,))
            _rowids = [r[0] for r in cursor.fetchall()]
            if _rowids:
                try:
                    placeholders = ",".join("?" * len(_rowids))
                    cursor.execute(
                        f"DELETE FROM concept_fts WHERE rowid IN ({placeholders})",
                        _rowids,
                    )
                except Exception as exc:
                    logger.debug("concept_fts delete by family failed: %s", exc)
            cursor.execute("DELETE FROM concepts WHERE family_id = ?", (family_id,))
        except Exception as exc:
            logger.debug("concept delete by family failed: %s", exc)

    def _batch_delete_concepts_by_ids(self, absolute_ids: List[str], cursor):
        """Batch delete concept versions by absolute_ids (called within existing write transaction).

        Replaces N calls to _delete_concept_by_id with 3 SQL statements total.
        """
        if not absolute_ids:
            return
        try:
            placeholders = ",".join("?" * len(absolute_ids))
            # Collect integer rowids for FTS cleanup
            cursor.execute(
                f"SELECT rowid FROM concepts WHERE id IN ({placeholders})",
                absolute_ids,
            )
            rowids = [r[0] for r in cursor.fetchall()]
            if rowids:
                try:
                    fts_ph = ",".join("?" * len(rowids))
                    cursor.execute(
                        f"DELETE FROM concept_fts WHERE rowid IN ({fts_ph})",
                        rowids,
                    )
                except Exception as exc:
                    logger.debug("batch concept_fts delete failed: %s", exc)
            cursor.execute(
                f"DELETE FROM concepts WHERE id IN ({placeholders})",
                absolute_ids,
            )
        except Exception as exc:
            logger.debug("batch concept delete by ids failed: %s", exc)

    def _batch_delete_concepts_by_families(self, family_ids: List[str], cursor):
        """Batch delete concept versions by family_ids (called within existing write transaction).

        Replaces N calls to _delete_concepts_by_family with 3 SQL statements total.
        """
        if not family_ids:
            return
        try:
            placeholders = ",".join("?" * len(family_ids))
            # Collect integer rowids for FTS cleanup
            cursor.execute(
                f"SELECT rowid FROM concepts WHERE family_id IN ({placeholders})",
                family_ids,
            )
            rowids = [r[0] for r in cursor.fetchall()]
            if rowids:
                try:
                    fts_ph = ",".join("?" * len(rowids))
                    cursor.execute(
                        f"DELETE FROM concept_fts WHERE rowid IN ({fts_ph})",
                        rowids,
                    )
                except Exception as exc:
                    logger.debug("batch concept_fts delete by families failed: %s", exc)
            cursor.execute(
                f"DELETE FROM concepts WHERE family_id IN ({placeholders})",
                family_ids,
            )
        except Exception as exc:
            logger.debug("batch concept delete by families failed: %s", exc)

    def _sync_concept_entity_fields(self, absolute_id: str, updates: Dict[str, Any], cursor):
        """Sync updated entity fields to the concepts table (called within existing write transaction).

        Args:
            absolute_id: Entity absolute_id (maps to concepts.id)
            updates: Dict of field_name -> new_value for fields that changed
        """
        if not updates:
            return
        # Map entity field names to concept column names
        concept_fields = {k: v for k, v in updates.items()
                         if k in {'name', 'content', 'summary', 'attributes', 'confidence'}}
        if not concept_fields:
            return
        try:
            set_clause = ", ".join(f"{k} = ?" for k in concept_fields)
            values = list(concept_fields.values()) + [absolute_id]
            cursor.execute(
                f"UPDATE concepts SET {set_clause} WHERE id = ?",
                values,
            )
            # Also update concept_fts if name or content changed.
            # Must read BOTH name and content from the row to avoid partial
            # FTS overwrite (e.g. only name changed → content would be lost).
            if 'name' in concept_fields or 'content' in concept_fields:
                try:
                    cursor.execute(
                        "SELECT rowid, name, content FROM concepts WHERE id = ?",
                        (absolute_id,),
                    )
                    _row = cursor.fetchone()
                    if _row:
                        fts_name = concept_fields.get('name', _row[1] or '')
                        fts_content = concept_fields.get('content', _row[2] or '')
                        cursor.execute(
                            "INSERT OR REPLACE INTO concept_fts(rowid, name, content) VALUES (?, ?, ?)",
                            (_row[0], fts_name, fts_content),
                        )
                except Exception as exc:
                    logger.debug("concept_fts sync failed: %s", exc)
        except Exception as exc:
            logger.debug("concept entity sync failed: %s", exc)

    def _sync_concept_relation_fields(self, absolute_id: str, updates: Dict[str, Any], cursor):
        """Sync updated relation fields to the concepts table (called within existing write transaction)."""
        if not updates:
            return
        concept_fields = {k: v for k, v in updates.items()
                         if k in {'content', 'summary', 'attributes', 'confidence'}}
        if not concept_fields:
            return
        try:
            set_clause = ", ".join(f"{k} = ?" for k in concept_fields)
            values = list(concept_fields.values()) + [absolute_id]
            cursor.execute(
                f"UPDATE concepts SET {set_clause} WHERE id = ?",
                values,
            )
            # Also update concept_fts if content changed
            if 'content' in concept_fields:
                try:
                    cursor.execute("SELECT rowid FROM concepts WHERE id = ?", (absolute_id,))
                    _row = cursor.fetchone()
                    if _row:
                        cursor.execute(
                            "INSERT OR REPLACE INTO concept_fts(rowid, name, content) VALUES (?, '', ?)",
                            (_row[0], concept_fields['content']),
                        )
                except Exception as exc:
                    logger.debug("concept_fts relation sync failed: %s", exc)
        except Exception as exc:
            logger.debug("concept relation sync failed: %s", exc)

    def _get_latest_concepts_with_embeddings(self, role: Optional[str] = None,
                                               time_point: Optional[str] = None) -> List[tuple]:
        """获取概念的最新版本及其 embedding（带短 TTL 缓存）。

        注意: time_point 过滤不使用缓存，因为缓存不分时间点。
        当 time_point 有值时跳过缓存直接查库。

        Returns:
            List of (concept_dict, embedding_array) tuples. embedding_array 为 None 表示没有 embedding。
        """
        now = time.time()
        # time_point 过滤不走缓存，因为缓存不分时间点
        use_cache = not time_point
        if use_cache:
            with self._emb_cache_lock:
                if self._concept_emb_cache is not None and (now - self._concept_emb_cache_ts) < self._emb_cache_ttl:
                    if role is None:
                        return self._concept_emb_cache
                    # 缓存不区分 role，在内存中过滤
                    return [(c, e) for c, e in self._concept_emb_cache if role is None or c.get('role') == role]

        conn = self._get_conn()
        cursor = conn.cursor()
        tp_sql, tp_params = self._time_point_sql(time_point)
        # Push role filter into SQL to avoid loading all concepts when role is specified
        role_sql = ""
        role_params: list = []
        if role is not None and not use_cache:
            role_sql = " AND role = ?"
            role_params = [role]
        # ROW_NUMBER 窗口函数获取每个 family_id 的最新版本
        cursor.execute(f"""
            SELECT id, family_id, role, name, content, event_time, processed_time,
                   source_document, episode_id, embedding, connects,
                   summary, attributes, confidence, content_format
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY family_id ORDER BY processed_time DESC) AS rn
                FROM concepts WHERE 1=1{tp_sql}{role_sql}
            )
            WHERE rn = 1
        """, tp_params + role_params)

        results = []
        for row in cursor.fetchall():
            embedding_array = None
            if row[9] is not None:  # embedding column
                try:
                    embedding_array = np.frombuffer(row[9], dtype=np.float32)
                except (ValueError, TypeError):
                    pass
            concept = {
                'id': row[0],
                'family_id': row[1],
                'role': row[2],
                'name': row[3] or '',
                'content': row[4] or '',
                'event_time': row[5],
                'processed_time': row[6],
                'source_document': row[7] or '',
                'episode_id': row[8] or '',
                'connects': row[10] or '',
                'summary': row[11],
                'attributes': row[12],
                'confidence': row[13],
                'content_format': row[14] or 'plain',
            }
            results.append((concept, embedding_array))

        # Filter out dream candidate relations
        results = [(c, e) for c, e in results if not self._is_dream_candidate_concept(c)]

        if use_cache:
            with self._emb_cache_lock:
                self._concept_emb_cache = results
                self._concept_emb_cache_ts = time.time()

        if role is not None:
            # If SQL already filtered by role, skip Python filtering
            if not use_cache:
                return results
            return [(c, e) for c, e in results if c.get('role') == role]
        return results

    def migrate_to_concepts(self):
        """Migrate existing entities + relations + episodes to concepts table (idempotent)."""
        conn = self._get_conn()
        with self._write_lock:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM concepts")
                if cursor.fetchone()[0] > 0:
                    return  # already migrated
                # entities -> concepts
                cursor.execute("""
                    INSERT OR REPLACE INTO concepts
                    (id, family_id, role, name, content, event_time, processed_time,
                     source_document, episode_id, embedding, valid_at, invalid_at,
                     summary, attributes, confidence, content_format)
                    SELECT id, family_id, 'entity', name, content, event_time, processed_time,
                           source_document, episode_id, embedding, valid_at, invalid_at,
                           summary, attributes, confidence, content_format
                    FROM entities
                """)
                # relations -> concepts
                cursor.execute("""
                    INSERT OR REPLACE INTO concepts
                    (id, family_id, role, content, event_time, processed_time,
                     source_document, episode_id, embedding, valid_at, invalid_at,
                     summary, attributes, confidence, content_format, connects)
                    SELECT id, family_id, 'relation', content, event_time, processed_time,
                           source_document, episode_id, embedding, valid_at, invalid_at,
                           summary, attributes, confidence, content_format,
                           json_array(entity1_absolute_id, entity2_absolute_id)
                    FROM relations
                """)
                # episodes -> concepts
                cursor.execute("""
                    INSERT OR REPLACE INTO concepts
                    (id, family_id, role, content, event_time, processed_time,
                     source_document, activity_type, episode_type)
                    SELECT id, family_id, 'observation', content, event_time, processed_time,
                           source_document, activity_type, episode_type
                    FROM episodes
                """)
                # Rebuild concept_fts（content-sync FTS5 使用 rebuild 命令）
                cursor.execute("INSERT INTO concept_fts(concept_fts) VALUES('rebuild')")
                conn.commit()
                logger.info("concepts 表迁移完成")
            except Exception as exc:
                logger.warning("concepts 迁移失败: %s", exc)
                conn.rollback()

    # ========== Phase 3: 统一概念查询接口 ==========

    @staticmethod
    def _time_point_sql(time_point: Optional[str], param_idx_offset: int = 0):
        """Build time_point filter SQL fragment and params.

        Returns (sql_fragment, params_list) where sql_fragment starts with 'AND'.
        The param indices are NOT used (we use ? placeholders), but param_idx_offset
        is kept for API compatibility.
        """
        if not time_point:
            return "", []
        return (
            " AND (valid_at IS NULL OR valid_at <= ?) AND (invalid_at IS NULL OR invalid_at > ?)",
            [time_point, time_point],
        )

    def get_concept_by_family_id(self, family_id: str,
                                  time_point: Optional[str] = None) -> Optional[dict]:
        """获取任意 role 的概念最新版本。支持 time_point 时间过滤。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        tp_sql, tp_params = self._time_point_sql(time_point)
        cursor.execute(
            f"SELECT * FROM concepts WHERE family_id = ?{tp_sql} ORDER BY processed_time DESC LIMIT 1",
            [family_id] + tp_params,
        )
        row = cursor.fetchone()
        if not row:
            return None
        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))

    def get_concepts_by_family_ids(self, family_ids: List[str],
                                    time_point: Optional[str] = None) -> Dict[str, dict]:
        """批量获取概念最新版本。支持 time_point 时间过滤。"""
        if not family_ids:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ','.join('?' * len(family_ids))
        tp_sql, tp_params = self._time_point_sql(time_point)
        # Get latest version of each family_id using a subquery
        cursor.execute(f"""
            SELECT c.* FROM concepts c
            INNER JOIN (
                SELECT family_id, MAX(processed_time) as max_pt
                FROM concepts WHERE family_id IN ({placeholders}){tp_sql}
                GROUP BY family_id
            ) latest ON c.family_id = latest.family_id AND c.processed_time = latest.max_pt
        """, family_ids + tp_params)
        cols = [desc[0] for desc in cursor.description]
        result = {}
        for row in cursor.fetchall():
            d = dict(zip(cols, row))
            result[d['family_id']] = d
        return result

    def search_concepts_by_bm25(self, query: str, role: str = None,
                                 limit: int = 20, time_point: Optional[str] = None) -> List[dict]:
        """BM25 搜索概念，可选按 role 过滤。支持 time_point 时间过滤。"""
        if not query:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        tp_sql, tp_params = self._time_point_sql(time_point)
        try:
            if role:
                cursor.execute(f"""
                    SELECT c.* FROM concepts c
                    JOIN concept_fts f ON c.rowid = f.rowid
                    WHERE concept_fts MATCH ? AND c.role = ?{tp_sql}
                    ORDER BY f.rank
                    LIMIT ?
                """, [query, role] + tp_params + [limit])
            else:
                cursor.execute(f"""
                    SELECT c.* FROM concepts c
                    JOIN concept_fts f ON c.rowid = f.rowid
                    WHERE concept_fts MATCH ?{tp_sql}
                    ORDER BY f.rank
                    LIMIT ?
                """, [query] + tp_params + [limit])
            cols = [desc[0] for desc in cursor.description]
            results = [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception:
            # FTS match syntax error -- fallback to LIKE
            query_lower = query.lower()
            q = f"%{query_lower}%"
            if role:
                cursor.execute(
                    f"SELECT * FROM concepts WHERE LOWER(content) LIKE ? AND role = ?{tp_sql} "
                    "ORDER BY processed_time DESC LIMIT ?",
                    [q, role] + tp_params + [limit]
                )
            else:
                cursor.execute(
                    f"SELECT * FROM concepts WHERE LOWER(content) LIKE ?{tp_sql} "
                    "ORDER BY processed_time DESC LIMIT ?",
                    [q] + tp_params + [limit]
                )
            cols = [desc[0] for desc in cursor.description]
            results = [dict(zip(cols, row)) for row in cursor.fetchall()]

        # Filter out dream candidate relations
        return [r for r in results if not self._is_dream_candidate_concept(r)]

    def search_concepts_by_similarity(self, query_text: str, role: str = None,
                                       threshold: float = 0.5, max_results: int = 10,
                                       time_point: Optional[str] = None) -> List[dict]:
        """语义相似度搜索：使用 embedding 余弦相似度，回退到 BM25。

        当 embedding 客户端可用时，编码查询文本并与 concepts 表中存储的
        embedding BLOB 进行余弦相似度比较。无 embedding 或编码失败时回退 BM25。
        """
        if not query_text:
            return []

        # 尝试 embedding 搜索
        if self.embedding_client and self.embedding_client.is_available():
            concepts_with_emb = self._get_latest_concepts_with_embeddings(role=role, time_point=time_point)
            if not concepts_with_emb:
                return []

            # 检查是否有任何概念有 embedding
            has_any_embedding = any(emb is not None for _, emb in concepts_with_emb)
            if not has_any_embedding:
                return self.search_concepts_by_bm25(query_text, role=role, limit=max_results, time_point=time_point)

            query_embedding = self.embedding_client.encode(query_text)
            if query_embedding is None:
                return self.search_concepts_by_bm25(query_text, role=role, limit=max_results, time_point=time_point)

            query_vec = np.array(query_embedding[0] if isinstance(query_embedding, (list, np.ndarray)) else query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                return self.search_concepts_by_bm25(query_text, role=role, limit=max_results, time_point=time_point)
            query_vec = query_vec / query_norm

            # 向量化计算：构建存储的 embedding 矩阵
            stored_rows = []
            stored_concepts = []
            for concept, emb in concepts_with_emb:
                if emb is not None and len(emb) > 0:
                    stored_rows.append(emb)
                    stored_concepts.append(concept)

            if not stored_rows:
                return self.search_concepts_by_bm25(query_text, role=role, limit=max_results, time_point=time_point)

            stored_matrix = np.stack(stored_rows)  # (M, D)
            norms = np.linalg.norm(stored_matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            stored_matrix = stored_matrix / norms

            # 余弦相似度 = 归一化后的点积
            similarities = stored_matrix @ query_vec  # (M,)

            # 收集超过阈值的结果
            scored = []
            for i, sim in enumerate(similarities):
                if sim >= threshold:
                    scored.append((float(sim), stored_concepts[i]))
            scored.sort(key=lambda x: x[0], reverse=True)

            results = []
            for sim, concept in scored[:max_results]:
                concept['_similarity_score'] = sim
                results.append(concept)

            if results:
                return results
            # embedding 搜索无结果，回退 BM25
            return self.search_concepts_by_bm25(query_text, role=role, limit=max_results, time_point=time_point)

        # 无 embedding 客户端，回退 BM25
        return self.search_concepts_by_bm25(query_text, role=role, limit=max_results, time_point=time_point)

    def _batch_get_concept_neighbors(self, family_ids: List[str],
                                      time_point: Optional[str] = None) -> Dict[str, List[dict]]:
        """Batch-fetch neighbors for multiple concepts in minimal SQL round-trips.

        Returns dict mapping family_id -> list of neighbor concept dicts.
        """
        if not family_ids:
            return {}

        # 1. Batch-fetch all concepts
        concepts_map = self.get_concepts_by_family_ids(family_ids, time_point=time_point)
        if not concepts_map:
            return {}

        # 2. Group by role
        entity_fids = [fid for fid, c in concepts_map.items() if c.get('role') == 'entity']
        relation_fids = [fid for fid, c in concepts_map.items() if c.get('role') == 'relation']
        obs_fids = [fid for fid, c in concepts_map.items() if c.get('role') == 'observation']

        result: Dict[str, List[dict]] = {fid: [] for fid in family_ids}
        conn = self._get_conn()
        cursor = conn.cursor()
        tp_sql, tp_params = self._time_point_sql(time_point)

        # 3a. Entity neighbors: batch-resolve all absolute_ids, then find connected relations
        if entity_fids:
            placeholders = ','.join('?' * len(entity_fids))
            cursor.execute(
                f"SELECT family_id, id FROM entities WHERE family_id IN ({placeholders})",
                entity_fids
            )
            entity_abs_map: Dict[str, List[str]] = {}
            all_abs_ids = []
            for row in cursor.fetchall():
                fid, aid = row[0], row[1]
                entity_abs_map.setdefault(fid, []).append(aid)
                all_abs_ids.append(aid)

            if all_abs_ids:
                abs_ph = ','.join('?' * len(all_abs_ids))
                cursor.execute(f"""
                    SELECT DISTINCT c.family_id, je.value FROM concepts c, json_each(c.connects) je
                    WHERE c.role = 'relation' AND je.value IN ({abs_ph}){tp_sql}
                """, all_abs_ids + tp_params)
                # Build reverse map: abs_id -> set of rel_family_ids
                abs_to_rel_fids: Dict[str, set] = {}
                all_rel_fids = set()
                for row in cursor.fetchall():
                    rel_fid, abs_id = row[0], row[1]
                    abs_to_rel_fids.setdefault(abs_id, set()).add(rel_fid)
                    all_rel_fids.add(rel_fid)

                # Batch-fetch relation concepts
                if all_rel_fids:
                    rel_concepts = self.get_concepts_by_family_ids(
                        list(all_rel_fids), time_point=time_point
                    )
                    rel_concepts = {k: v for k, v in rel_concepts.items()
                                    if not self._is_dream_candidate_concept(v)}
                    # Map back to entity fids
                    for efid, aids in entity_abs_map.items():
                        seen = set()
                        for aid in aids:
                            for rel_fid in abs_to_rel_fids.get(aid, ()):
                                if rel_fid not in seen and rel_fid in rel_concepts:
                                    result[efid].append(rel_concepts[rel_fid])
                                    seen.add(rel_fid)

        # 3b. Relation neighbors: batch-resolve connects to entity family_ids
        if relation_fids:
            all_connects_abs: List[str] = []
            connects_map: Dict[str, List[str]] = {}
            for fid in relation_fids:
                connects = concepts_map[fid].get('connects', '')
                if connects:
                    try:
                        aids = json.loads(connects) if isinstance(connects, str) else connects
                        if aids:
                            connects_map[fid] = aids
                            all_connects_abs.extend(aids)
                    except (json.JSONDecodeError, Exception):
                        pass

            if all_connects_abs:
                abs_ph = ','.join('?' * len(all_connects_abs))
                cursor.execute(
                    f"SELECT DISTINCT family_id, id FROM concepts WHERE id IN ({abs_ph}){tp_sql}",
                    all_connects_abs + tp_params
                )
                # Build abs_id -> family_id mapping
                abs_to_ent_fid: Dict[str, str] = {}
                ent_fids_set = set()
                for row in cursor.fetchall():
                    ent_fid, abs_id = row[0], row[1]
                    abs_to_ent_fid[abs_id] = ent_fid
                    ent_fids_set.add(ent_fid)

                if ent_fids_set:
                    ent_concepts = self.get_concepts_by_family_ids(
                        list(ent_fids_set), time_point=time_point
                    )
                    ent_concepts = {k: v for k, v in ent_concepts.items()
                                    if not self._is_dream_candidate_concept(v)}
                    for fid, aids in connects_map.items():
                        seen = set()
                        for aid in aids:
                            ent_fid = abs_to_ent_fid.get(aid)
                            if ent_fid and ent_fid not in seen and ent_fid in ent_concepts:
                                result[fid].append(ent_concepts[ent_fid])
                                seen.add(ent_fid)

        # 3c. Observation neighbors: batch via episode_mentions
        if obs_fids:
            obs_abs_ids = [concepts_map[fid]['id'] for fid in obs_fids if 'id' in concepts_map[fid]]
            if obs_abs_ids:
                abs_ph = ','.join('?' * len(obs_abs_ids))
                cursor.execute(f"""
                    SELECT episode_id, target_absolute_id FROM episode_mentions
                    WHERE episode_id IN ({abs_ph})
                """, obs_abs_ids)
                # Build episode_abs -> target_abs_ids
                ep_to_targets: Dict[str, List[str]] = {}
                all_target_abs = set()
                for row in cursor.fetchall():
                    ep_abs, target_abs = row[0], row[1]
                    ep_to_targets.setdefault(ep_abs, []).append(target_abs)
                    all_target_abs.add(target_abs)

                # Also need abs_id -> fid for episode lookup
                obs_abs_to_fid = {concepts_map[fid]['id']: fid for fid in obs_fids if 'id' in concepts_map[fid]}

                if all_target_abs:
                    target_list = list(all_target_abs)
                    target_ph = ','.join('?' * len(target_list))
                    cursor.execute(f"""
                        SELECT family_id, id FROM concepts WHERE id IN ({target_ph}){tp_sql}
                    """, target_list + tp_params)
                    target_abs_to_fid: Dict[str, str] = {}
                    target_fid_set = set()
                    for row in cursor.fetchall():
                        t_fid, t_abs = row[0], row[1]
                        target_abs_to_fid[t_abs] = t_fid
                        target_fid_set.add(t_fid)

                    if target_fid_set:
                        target_concepts = self.get_concepts_by_family_ids(
                            list(target_fid_set), time_point=time_point
                        )
                        target_concepts = {k: v for k, v in target_concepts.items()
                                           if not self._is_dream_candidate_concept(v)}
                        for ep_abs, targets in ep_to_targets.items():
                            ep_fid = obs_abs_to_fid.get(ep_abs)
                            if not ep_fid:
                                continue
                            seen = set()
                            for t_abs in targets:
                                t_fid = target_abs_to_fid.get(t_abs)
                                if t_fid and t_fid not in seen and t_fid in target_concepts:
                                    result[ep_fid].append(target_concepts[t_fid])
                                    seen.add(t_fid)

        return result

    def get_concept_neighbors(self, family_id: str, max_depth: int = 1,
                               time_point: Optional[str] = None) -> List[dict]:
        """获取概念的邻居（无论 role）。支持 time_point 时间过滤。

        - entity: 返回关联的 relation（通过 connects 字段包含该 entity family_id 的任意版本的 absolute_id）
        - relation: 返回它连接的 entity
        - observation: 返回它 MENTIONS 的所有 concept
        """
        concept = self.get_concept_by_family_id(family_id, time_point=time_point)
        if not concept:
            return []
        role = concept.get('role', 'entity')
        results = []

        if role == 'entity':
            # Find all absolute_ids for this entity family
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM entities WHERE family_id = ?", (family_id,)
            )
            abs_ids = [row[0] for row in cursor.fetchall()]
            if abs_ids:
                placeholders = ','.join('?' * len(abs_ids))
                tp_sql, tp_params = self._time_point_sql(time_point)
                # Find relations that connect to this entity
                cursor.execute(f"""
                    SELECT DISTINCT c.family_id FROM concepts c, json_each(c.connects) je
                    WHERE c.role = 'relation' AND je.value IN ({placeholders}){tp_sql}
                """, abs_ids + tp_params)
                rel_fids = list({r[0] for r in cursor.fetchall()})
                if rel_fids:
                    rel_concepts = self.get_concepts_by_family_ids(rel_fids, time_point=time_point)
                    results.extend(rel_concepts.values())

        elif role == 'relation':
            connects = concept.get('connects', '')
            if connects:
                try:
                    abs_ids = json.loads(connects) if isinstance(connects, str) else connects
                    # Batch resolve absolute_ids to family_ids (replacing N+1 loop)
                    entity_fids = set()
                    if abs_ids:
                        conn = self._get_conn()
                        cursor = conn.cursor()
                        placeholders = ','.join('?' * len(abs_ids))
                        tp_sql, tp_params = self._time_point_sql(time_point)
                        cursor.execute(
                            f"SELECT DISTINCT family_id FROM concepts WHERE id IN ({placeholders}){tp_sql}",
                            abs_ids + tp_params
                        )
                        entity_fids = set(r[0] for r in cursor.fetchall())
                    if entity_fids:
                        ent_concepts = self.get_concepts_by_family_ids(list(entity_fids), time_point=time_point)
                        results.extend(ent_concepts.values())
                except (json.JSONDecodeError, Exception):
                    pass

        elif role == 'observation':
            # Get concepts mentioned by this episode
            abs_id = concept['id']
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT target_absolute_id FROM episode_mentions
                WHERE episode_id = ?
            """, (abs_id,))
            target_abs_ids = list({r[0] for r in cursor.fetchall()})
            # Resolve to family_ids
            if target_abs_ids:
                placeholders = ','.join('?' * len(target_abs_ids))
                tp_sql, tp_params = self._time_point_sql(time_point)
                cursor.execute(f"""
                    SELECT family_id, id FROM concepts WHERE id IN ({placeholders}){tp_sql}
                """, target_abs_ids + tp_params)
                fid_set = set()
                for row in cursor.fetchall():
                    fid_set.add(row[0])
                if fid_set:
                    mentioned = self.get_concepts_by_family_ids(list(fid_set), time_point=time_point)
                    results.extend(mentioned.values())

        # Filter dream candidate relations from neighbor results
        results = [r for r in results if not self._is_dream_candidate_concept(r)]

        return results

    def get_concept_provenance(self, family_id: str,
                                time_point: Optional[str] = None) -> List[dict]:
        """溯源：返回所有提及此概念的 observation。

        支持所有 role：entity、relation、observation。
        先确定概念的 role，再从对应表查询 absolute_ids。
        """
        concept = self.get_concept_by_family_id(family_id, time_point=time_point)
        if not concept:
            return []
        role = concept.get('role', 'entity')
        conn = self._get_conn()
        cursor = conn.cursor()

        abs_ids = []
        if role == 'entity':
            cursor.execute(
                "SELECT id FROM entities WHERE family_id = ?", (family_id,)
            )
            abs_ids = [row[0] for row in cursor.fetchall()]
        elif role == 'relation':
            cursor.execute(
                "SELECT id FROM relations WHERE family_id = ?", (family_id,)
            )
            abs_ids = [row[0] for row in cursor.fetchall()]
        elif role == 'observation':
            # observation: query episodes table by family_id for all versions
            cursor.execute(
                "SELECT id FROM episodes WHERE family_id = ?", (family_id,)
            )
            abs_ids = [row[0] for row in cursor.fetchall()]
            if not abs_ids:
                # Fallback: standalone observation (no versioning yet)
                abs_ids = [concept['id']]

        if not abs_ids:
            return []
        placeholders = ','.join('?' * len(abs_ids))
        cursor.execute(f"""
            SELECT DISTINCT ep.id, ep.content, ep.event_time, ep.source_document
            FROM episodes ep
            JOIN episode_mentions em ON ep.id = em.episode_id
            WHERE em.target_absolute_id IN ({placeholders})
            ORDER BY ep.event_time DESC
        """, abs_ids)
        return [
            {"episode_id": row[0], "content": row[1] or "",
             "event_time": row[2] or "", "source_document": row[3] or ""}
            for row in cursor.fetchall()
        ]

    def get_concept_mentions(self, family_id: str,
                              time_point: Optional[str] = None) -> List[dict]:
        """获取提及此概念的所有 Episode。"""
        return self.get_concept_provenance(family_id, time_point=time_point)

    def get_episode_concepts(self, episode_id: str) -> List[dict]:
        """获取 Episode 提及的所有概念（entity + relation）。"""
        return self.get_episode_entities(episode_id)

    def traverse_concepts(self, start_family_ids: List[str], max_depth: int = 2,
                           time_point: Optional[str] = None) -> dict:
        """BFS 遍历概念图。支持 time_point 时间过滤。

        Uses batch neighbor queries per BFS level to minimize SQL round-trips.
        """
        visited = set()
        queue = list(start_family_ids)
        all_concepts = {}
        all_relations_info = []
        seen_rel_fids = set()

        depth = 0
        while queue and depth <= max_depth:
            # Filter out already-visited nodes
            frontier = [fid for fid in queue if fid not in visited]
            if not frontier:
                break
            visited.update(frontier)

            # Batch-fetch concepts for the entire frontier
            frontier_concepts = self.get_concepts_by_family_ids(frontier, time_point=time_point)
            for fid in frontier:
                if fid in frontier_concepts:
                    all_concepts[fid] = frontier_concepts[fid]
                else:
                    visited.discard(fid)

            # Batch-fetch neighbors for the entire frontier
            batch_neighbors = self._batch_get_concept_neighbors(frontier, time_point=time_point)

            next_queue = []
            for fid in frontier:
                if fid not in batch_neighbors:
                    continue
                for n in batch_neighbors[fid]:
                    nfid = n.get('family_id', '')
                    if nfid and nfid not in visited:
                        next_queue.append(nfid)
                    if n.get('role') == 'relation' and nfid not in seen_rel_fids:
                        seen_rel_fids.add(nfid)
                        all_relations_info.append(n)
            queue = next_queue
            depth += 1

        return {
            "concepts": all_concepts,
            "relations": all_relations_info,
            "visited_count": len(visited),
        }

    def list_concepts(self, role: str = None, limit: int = 50, offset: int = 0,
                       time_point: Optional[str] = None) -> List[dict]:
        """列出概念（分页 + 可选 role 过滤）。支持 time_point 时间过滤。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        tp_sql, tp_params = self._time_point_sql(time_point)
        if role:
            cursor.execute(
                f"SELECT * FROM concepts WHERE role = ?{tp_sql} ORDER BY processed_time DESC LIMIT ? OFFSET ?",
                [role] + tp_params + [limit, offset]
            )
        else:
            cursor.execute(
                f"SELECT * FROM concepts{(' WHERE 1=1' if tp_sql else '')}{tp_sql} ORDER BY processed_time DESC LIMIT ? OFFSET ?",
                tp_params + [limit, offset]
            )
        cols = [desc[0] for desc in cursor.description]
        results = [dict(zip(cols, row)) for row in cursor.fetchall()]
        # Filter out dream candidate relations
        return [r for r in results if not self._is_dream_candidate_concept(r)]

    def count_concepts(self, role: str = None, time_point: Optional[str] = None) -> int:
        """统计概念数量。支持 time_point 时间过滤。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        tp_sql, tp_params = self._time_point_sql(time_point)
        if role:
            cursor.execute(f"SELECT COUNT(*) FROM concepts WHERE role = ?{tp_sql}", [role] + tp_params)
        else:
            cursor.execute(f"SELECT COUNT(*) FROM concepts{(' WHERE 1=1' if tp_sql else '')}{tp_sql}", tp_params)
        return cursor.fetchone()[0]
