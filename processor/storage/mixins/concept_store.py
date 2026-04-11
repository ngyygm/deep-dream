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

    # ========== Phase E: Dream Logs ==========

    def save_dream_log(self, report):
        """保存梦境报告。"""
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
                json.dumps(report.insights, ensure_ascii=False),
                json.dumps(report.new_connections, ensure_ascii=False),
                json.dumps(report.consolidations, ensure_ascii=False),
                json.dumps({}, ensure_ascii=False),
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

    def _write_concept_from_entity(self, entity: Entity, cursor):
        """Dual-write: write Entity to concepts table (called within existing write transaction)."""
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO concepts
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
                entity.embedding,
                (entity.valid_at or entity.event_time).isoformat(),
                getattr(entity, 'invalid_at', None),
                getattr(entity, 'summary', None),
                json.dumps(getattr(entity, 'attributes', None)) if isinstance(getattr(entity, 'attributes', None), dict) else getattr(entity, 'attributes', None),
                getattr(entity, 'confidence', None),
                getattr(entity, 'content_format', 'plain'),
            ))
            # FTS (content-sync: use integer rowid from concepts table)
            try:
                cursor.execute(
                    "SELECT rowid FROM concepts WHERE id = ?", (entity.absolute_id,)
                )
                _row = cursor.fetchone()
                if _row:
                    cursor.execute("""
                        INSERT INTO concept_fts(rowid, name, content)
                        VALUES (?, ?, ?)
                    """, (_row[0], entity.name, entity.content))
            except Exception as exc:
                logger.debug("concept_fts entity write failed: %s", exc)
        except Exception as exc:
            logger.debug("concept entity dual-write failed: %s", exc)

    def _write_concept_from_relation(self, relation: Relation, cursor):
        """Dual-write: write Relation to concepts table (called within existing write transaction)."""
        try:
            connects = json.dumps([relation.entity1_absolute_id, relation.entity2_absolute_id])
            cursor.execute("""
                INSERT OR IGNORE INTO concepts
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
                relation.embedding,
                (relation.valid_at or relation.event_time).isoformat(),
                getattr(relation, 'invalid_at', None),
                getattr(relation, 'summary', None),
                json.dumps(getattr(relation, 'attributes', None)) if isinstance(getattr(relation, 'attributes', None), dict) else getattr(relation, 'attributes', None),
                getattr(relation, 'confidence', None),
                getattr(relation, 'content_format', 'plain'),
                getattr(relation, 'provenance', '') or '',
                connects,
            ))
            # FTS (content-sync: use integer rowid from concepts table)
            try:
                cursor.execute(
                    "SELECT rowid FROM concepts WHERE id = ?", (relation.absolute_id,)
                )
                _row = cursor.fetchone()
                if _row:
                    cursor.execute("""
                        INSERT INTO concept_fts(rowid, name, content)
                        VALUES (?, '', ?)
                    """, (_row[0], relation.content))
            except Exception as exc:
                logger.debug("concept_fts relation write failed: %s", exc)
        except Exception as exc:
            logger.debug("concept relation dual-write failed: %s", exc)

    def _write_concept_from_episode(self, cache: Episode, doc_hash: str, cursor,
                                     family_id: str = None):
        """Dual-write: write Episode to concepts table as 'observation' (called within existing write transaction).

        Args:
            family_id: Optional resolved family_id. If provided, uses it for versioning.
                       Falls back to cache.absolute_id for standalone observations.
        """
        try:
            fid = family_id or cache.absolute_id
            cursor.execute("""
                INSERT OR IGNORE INTO concepts
                (id, family_id, role, name, content, event_time, processed_time,
                 source_document, activity_type, episode_type, provenance)
                VALUES (?, ?, 'observation', '', ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache.absolute_id,
                fid,
                cache.content,
                cache.event_time.isoformat(),
                datetime.now().isoformat(),
                cache.source_document or '',
                cache.activity_type or '',
                getattr(cache, 'episode_type', '') or '',
                json.dumps({"doc_hash": doc_hash}) if doc_hash else '',
            ))
            # FTS (content-sync: use integer rowid from concepts table)
            try:
                cursor.execute(
                    "SELECT rowid FROM concepts WHERE id = ?", (cache.absolute_id,)
                )
                _row = cursor.fetchone()
                if _row:
                    cursor.execute("""
                        INSERT INTO concept_fts(rowid, name, content)
                        VALUES (?, '', ?)
                    """, (_row[0], cache.content))
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
            # Also update concept_fts if name or content changed
            if 'name' in concept_fields or 'content' in concept_fields:
                try:
                    cursor.execute("SELECT rowid FROM concepts WHERE id = ?", (absolute_id,))
                    _row = cursor.fetchone()
                    if _row:
                        name_val = concept_fields.get('name', '')
                        content_val = concept_fields.get('content', '')
                        if name_val or content_val:
                            cursor.execute(
                                "INSERT OR REPLACE INTO concept_fts(rowid, name, content) VALUES (?, ?, ?)",
                                (_row[0], name_val, content_val),
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

    def _get_latest_concepts_with_embeddings(self, role: Optional[str] = None) -> List[tuple]:
        """获取概念的最新版本及其 embedding（带短 TTL 缓存）。

        Returns:
            List of (concept_dict, embedding_array) tuples. embedding_array 为 None 表示没有 embedding。
        """
        now = time.time()
        with self._emb_cache_lock:
            if self._concept_emb_cache is not None and (now - self._concept_emb_cache_ts) < self._emb_cache_ttl:
                if role is None:
                    return self._concept_emb_cache
                # 缓存不区分 role，在内存中过滤
                return [(c, e) for c, e in self._concept_emb_cache if role is None or c.get('role') == role]

        conn = self._get_conn()
        cursor = conn.cursor()
        # ROW_NUMBER 窗口函数获取每个 family_id 的最新版本
        cursor.execute("""
            SELECT id, family_id, role, name, content, event_time, processed_time,
                   source_document, episode_id, embedding, connects,
                   summary, attributes, confidence, content_format
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY family_id ORDER BY processed_time DESC) AS rn
                FROM concepts
            )
            WHERE rn = 1
        """)

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

        with self._emb_cache_lock:
            self._concept_emb_cache = results
            self._concept_emb_cache_ts = time.time()

        if role is not None:
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
                    INSERT OR IGNORE INTO concepts
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
                    INSERT OR IGNORE INTO concepts
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
                    INSERT OR IGNORE INTO concepts
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

    def get_concept_by_family_id(self, family_id: str) -> Optional[dict]:
        """获取任意 role 的概念最新版本。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM concepts WHERE family_id = ? ORDER BY processed_time DESC LIMIT 1",
            (family_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))

    def get_concepts_by_family_ids(self, family_ids: List[str]) -> Dict[str, dict]:
        """批量获取概念最新版本。"""
        if not family_ids:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ','.join('?' * len(family_ids))
        # Get latest version of each family_id using a subquery
        cursor.execute(f"""
            SELECT c.* FROM concepts c
            INNER JOIN (
                SELECT family_id, MAX(processed_time) as max_pt
                FROM concepts WHERE family_id IN ({placeholders})
                GROUP BY family_id
            ) latest ON c.family_id = latest.family_id AND c.processed_time = latest.max_pt
        """, family_ids)
        cols = [desc[0] for desc in cursor.description]
        result = {}
        for row in cursor.fetchall():
            d = dict(zip(cols, row))
            result[d['family_id']] = d
        return result

    def search_concepts_by_bm25(self, query: str, role: str = None, limit: int = 20) -> List[dict]:
        """BM25 搜索概念，可选按 role 过滤。"""
        if not query:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            if role:
                cursor.execute("""
                    SELECT c.* FROM concepts c
                    JOIN concept_fts f ON c.rowid = f.rowid
                    WHERE concept_fts MATCH ? AND c.role = ?
                    ORDER BY f.rank
                    LIMIT ?
                """, (query, role, limit))
            else:
                cursor.execute("""
                    SELECT c.* FROM concepts c
                    JOIN concept_fts f ON c.rowid = f.rowid
                    WHERE concept_fts MATCH ?
                    ORDER BY f.rank
                    LIMIT ?
                """, (query, limit))
            cols = [desc[0] for desc in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception:
            # FTS match syntax error -- fallback to LIKE
            query_lower = query.lower()
            q = f"%{query_lower}%"
            if role:
                cursor.execute(
                    "SELECT * FROM concepts WHERE LOWER(content) LIKE ? AND role = ? "
                    "ORDER BY processed_time DESC LIMIT ?",
                    (q, role, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM concepts WHERE LOWER(content) LIKE ? "
                    "ORDER BY processed_time DESC LIMIT ?",
                    (q, limit)
                )
            cols = [desc[0] for desc in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def search_concepts_by_similarity(self, query_text: str, role: str = None,
                                       threshold: float = 0.5, max_results: int = 10) -> List[dict]:
        """语义相似度搜索：使用 embedding 余弦相似度，回退到 BM25。

        当 embedding 客户端可用时，编码查询文本并与 concepts 表中存储的
        embedding BLOB 进行余弦相似度比较。无 embedding 或编码失败时回退 BM25。
        """
        if not query_text:
            return []

        # 尝试 embedding 搜索
        if self.embedding_client and self.embedding_client.is_available():
            concepts_with_emb = self._get_latest_concepts_with_embeddings(role=role)
            if not concepts_with_emb:
                return []

            # 检查是否有任何概念有 embedding
            has_any_embedding = any(emb is not None for _, emb in concepts_with_emb)
            if not has_any_embedding:
                return self.search_concepts_by_bm25(query_text, role=role, limit=max_results)

            query_embedding = self.embedding_client.encode(query_text)
            if query_embedding is None:
                return self.search_concepts_by_bm25(query_text, role=role, limit=max_results)

            query_vec = np.array(query_embedding[0] if isinstance(query_embedding, (list, np.ndarray)) else query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                return self.search_concepts_by_bm25(query_text, role=role, limit=max_results)
            query_vec = query_vec / query_norm

            # 向量化计算：构建存储的 embedding 矩阵
            stored_rows = []
            stored_concepts = []
            for concept, emb in concepts_with_emb:
                if emb is not None and len(emb) > 0:
                    stored_rows.append(emb)
                    stored_concepts.append(concept)

            if not stored_rows:
                return self.search_concepts_by_bm25(query_text, role=role, limit=max_results)

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
            return self.search_concepts_by_bm25(query_text, role=role, limit=max_results)

        # 无 embedding 客户端，回退 BM25
        return self.search_concepts_by_bm25(query_text, role=role, limit=max_results)

    def get_concept_neighbors(self, family_id: str, max_depth: int = 1) -> List[dict]:
        """获取概念的邻居（无论 role）。

        - entity: 返回关联的 relation（通过 connects 字段包含该 entity family_id 的任意版本的 absolute_id）
        - relation: 返回它连接的 entity
        - observation: 返回它 MENTIONS 的所有 concept
        """
        concept = self.get_concept_by_family_id(family_id)
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
                # Find relations that connect to this entity
                cursor.execute(f"""
                    SELECT DISTINCT c.family_id FROM concepts c, json_each(c.connects) je
                    WHERE c.role = 'relation' AND je.value IN ({placeholders})
                """, abs_ids)
                rel_fids = list(set(r[0] for r in cursor.fetchall()))
                if rel_fids:
                    rel_concepts = self.get_concepts_by_family_ids(rel_fids)
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
                        cursor.execute(
                            f"SELECT DISTINCT family_id FROM concepts WHERE id IN ({placeholders})",
                            abs_ids
                        )
                        entity_fids = set(r[0] for r in cursor.fetchall())
                    if entity_fids:
                        ent_concepts = self.get_concepts_by_family_ids(list(entity_fids))
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
            target_abs_ids = list(set(r[0] for r in cursor.fetchall()))
            # Resolve to family_ids
            if target_abs_ids:
                placeholders = ','.join('?' * len(target_abs_ids))
                cursor.execute(f"""
                    SELECT family_id, id FROM concepts WHERE id IN ({placeholders})
                """, target_abs_ids)
                fid_set = set()
                for row in cursor.fetchall():
                    fid_set.add(row[0])
                if fid_set:
                    mentioned = self.get_concepts_by_family_ids(list(fid_set))
                    results.extend(mentioned.values())

        return results

    def get_concept_provenance(self, family_id: str) -> List[dict]:
        """溯源：返回所有提及此概念的 observation。

        支持所有 role：entity、relation、observation。
        先确定概念的 role，再从对应表查询 absolute_ids。
        """
        concept = self.get_concept_by_family_id(family_id)
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

    def get_concept_mentions(self, family_id: str) -> List[dict]:
        """获取提及此概念的所有 Episode。"""
        return self.get_concept_provenance(family_id)

    def get_episode_concepts(self, episode_id: str) -> List[dict]:
        """获取 Episode 提及的所有概念（entity + relation）。"""
        return self.get_episode_entities(episode_id)

    def traverse_concepts(self, start_family_ids: List[str], max_depth: int = 2) -> dict:
        """BFS 遍历概念图。"""
        visited = set()
        queue = list(start_family_ids)
        all_concepts = {}
        all_relations_info = []
        seen_rel_fids = set()

        depth = 0
        while queue and depth <= max_depth:
            next_queue = []
            for fid in queue:
                if fid in visited:
                    continue
                visited.add(fid)
                concept = self.get_concept_by_family_id(fid)
                if not concept:
                    # Not a real concept — remove from visited
                    visited.discard(fid)
                    continue
                all_concepts[fid] = concept
                neighbors = self.get_concept_neighbors(fid)
                for n in neighbors:
                    nfid = n.get('family_id', '')
                    if nfid and nfid not in visited:
                        next_queue.append(nfid)
                    # Track relation concepts with full metadata
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

    def list_concepts(self, role: str = None, limit: int = 50, offset: int = 0) -> List[dict]:
        """列出概念（分页 + 可选 role 过滤）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        if role:
            cursor.execute(
                "SELECT * FROM concepts WHERE role = ? ORDER BY processed_time DESC LIMIT ? OFFSET ?",
                (role, limit, offset)
            )
        else:
            cursor.execute(
                "SELECT * FROM concepts ORDER BY processed_time DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def count_concepts(self, role: str = None) -> int:
        """统计概念数量。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        if role:
            cursor.execute("SELECT COUNT(*) FROM concepts WHERE role = ?", (role,))
        else:
            cursor.execute("SELECT COUNT(*) FROM concepts")
        return cursor.fetchone()[0]
