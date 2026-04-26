"""
EntityStoreMixin — entity-specific methods extracted from StorageManager.

This mixin is designed to be inherited by StorageManager. It relies on the
following shared state being present on the host class:

    - self._get_conn()
    - self._write_lock
    - self._local
    - self._ENTITY_SELECT
    - self._safe_parse_datetime()
    - self.embedding_client
    - self.entity_content_snippet_length
    - self._entity_emb_cache, self._entity_emb_cache_ts, self._emb_cache_ttl
    - self._invalidate_emb_cache()
    - self.resolve_family_id() / resolve_family_ids()
    - self._resolve_family_id_with_cursor()
    - self._write_concept_from_entity()
    - self._delete_concept_by_id()
    - self._delete_concepts_by_family()
"""
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal

import numpy as np

from ...models import Entity

logger = logging.getLogger(__name__)


class EntityStoreMixin:
    """Mixin providing all entity CRUD / query / search methods.

    Do NOT add __init__ here — the host class (StorageManager) owns all shared state.
    """

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    _VALID_TABLES = frozenset({"entities", "relations"})

    @classmethod
    def _validate_table_name(cls, source_type: str) -> str:
        """Allowlist-validated table name to prevent SQL injection."""
        table = "entities" if source_type == "entity" else "relations"
        if table not in cls._VALID_TABLES:
            raise ValueError(f"Invalid source_type: {source_type}")
        return table

    def _row_to_entity(self, row) -> Entity:
        """Convert a SELECT row tuple to an Entity object using _ENTITY_SELECT column order."""
        return Entity(
            absolute_id=row[0],
            family_id=row[1],
            name=row[2],
            content=row[3],
            event_time=self._safe_parse_datetime(row[4]),
            processed_time=self._safe_parse_datetime(row[5]),
            episode_id=row[6],
            source_document=row[7] if len(row) > 7 else '',
            embedding=row[8] if len(row) > 8 else None,
            summary=row[9] if len(row) > 9 else None,
            attributes=row[10] if len(row) > 10 else None,
            confidence=row[11] if len(row) > 11 else None,
            valid_at=self._safe_parse_datetime(row[12]) if len(row) > 12 and row[12] else None,
            invalid_at=self._safe_parse_datetime(row[13]) if len(row) > 13 and row[13] else None,
            content_format=row[14] if len(row) > 14 else 'plain',
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _compute_entity_embedding(self, entity: Entity) -> Optional[bytes]:
        """计算实体的embedding向量并转换为BLOB"""
        text = f"{entity.name} {entity.content}"
        return self._compute_embedding_text(text)

    def _compute_embedding_text(self, text: str) -> Optional[bytes]:
        """Compute embedding from raw text string, returning BLOB."""
        if not self.embedding_client or not self.embedding_client.is_available():
            return None
        embedding = self.embedding_client.encode(text)
        if embedding is None or not embedding:
            return None
        embedding_array = np.array(embedding[0] if isinstance(embedding, list) else embedding, dtype=np.float32)
        return embedding_array.tobytes()

    # ------------------------------------------------------------------
    # Save / Write
    # ------------------------------------------------------------------

    def save_entity(self, entity: Entity):
        """保存实体（包含预计算的embedding向量）"""
        self._invalidate_emb_cache()
        # 计算embedding（无需锁，纯计算）
        embedding_blob = self._compute_entity_embedding(entity)
        entity.embedding = embedding_blob
        # Pre-compute concept embedding OUTSIDE lock
        _concept_emb = self._compute_concept_embedding(
            role='entity', name=entity.name, content=entity.content
        )
        # processed_time = 实际写入时刻（而非构造时刻）
        entity.processed_time = datetime.now()

        with self._write_lock:
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO entities (id, family_id, name, content, event_time, processed_time, episode_id, source_document, embedding, valid_at, invalid_at, summary, attributes, confidence, content_format)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.absolute_id,
                    entity.family_id,
                    entity.name,
                    entity.content,
                    entity.event_time.isoformat(),
                    entity.processed_time.isoformat(),
                    entity.episode_id,
                    entity.source_document,
                    embedding_blob,
                    (entity.valid_at or entity.event_time).isoformat(),
                    getattr(entity, 'invalid_at', None),
                    getattr(entity, 'summary', None),
                    getattr(entity, 'attributes', None),
                    getattr(entity, 'confidence', None),
                    getattr(entity, 'content_format', 'plain'),
                ))
                # 同步写入 FTS 表（使用整数 rowid，非文本 id）
                try:
                    cursor.execute("""
                        INSERT INTO entity_fts(rowid, name, content, family_id)
                        VALUES (?, ?, ?, ?)
                    """, (cursor.lastrowid, entity.name, entity.content, entity.family_id))
                except Exception as exc:
                    logger.warning("FTS entity write failed: %s", exc)
                # 设置旧版本 invalid_at
                try:
                    cursor.execute("""
                        UPDATE entities SET invalid_at = ?
                        WHERE family_id = ? AND id != ? AND invalid_at IS NULL
                    """, (entity.event_time.isoformat(), entity.family_id, entity.absolute_id))
                    # 同步更新 concepts 表中旧版本的 invalid_at
                    cursor.execute("""
                        UPDATE concepts SET invalid_at = ?
                        WHERE family_id = ? AND id != ? AND role = 'entity' AND invalid_at IS NULL
                    """, (entity.event_time.isoformat(), entity.family_id, entity.absolute_id))
                except Exception as exc:
                    logger.warning("FTS invalid_at update failed: %s", exc)
                # Phase 2: dual-write to concepts (embedding pre-computed above)
                self._write_concept_from_entity(entity, cursor, precomputed_embedding=_concept_emb)
                # 单次 commit 包含所有写操作
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        # Vector store I/O OUTSIDE lock (after successful commit)
        self._vector_store_upsert_entity(entity.absolute_id, embedding_blob)

    def bulk_save_entities(self, entities: List[Entity]):
        """批量保存实体，使用批量 embedding 与单事务写入。"""
        if not entities:
            return
        self._invalidate_emb_cache()

        # 批量计算 entity embedding（无需锁）
        embeddings = None
        concept_embeddings: Dict[str, Optional[bytes]] = {}
        if self.embedding_client and self.embedding_client.is_available():
            texts = [
                f"{entity.name} {entity.content}"
                for entity in entities
            ]
            embeddings = self.embedding_client.encode(texts)
            # Batch compute concept embeddings (different text template: "name: content")
            _concept_emb_client = getattr(self, 'embedding_client', None)
            if _concept_emb_client and _concept_emb_client.is_available():
                concept_texts = [
                    f"{entity.name}: {entity.content[:512]}" if entity.content else entity.name
                    for entity in entities
                ]
                concept_embs = _concept_emb_client.encode(concept_texts)
                if concept_embs is not None:
                    for i, entity in enumerate(entities):
                        try:
                            concept_embeddings[entity.absolute_id] = np.array(
                                concept_embs[i] if isinstance(concept_embs[i], (list, np.ndarray)) else concept_embs[i],
                                dtype=np.float32
                            ).tobytes()
                        except Exception:
                            concept_embeddings[entity.absolute_id] = None

        rows = []
        for idx, entity in enumerate(entities):
            embedding_blob = None
            if embeddings is not None:
                try:
                    embedding_blob = np.array(embeddings[idx], dtype=np.float32).tobytes()
                except Exception as exc:
                    logger.debug("embedding encode failed: %s", exc)
                    embedding_blob = None
            entity.embedding = embedding_blob
            rows.append((
                entity.absolute_id,
                entity.family_id,
                entity.name,
                entity.content,
                entity.event_time.isoformat(),
                entity.processed_time.isoformat(),
                entity.episode_id,
                entity.source_document,
                embedding_blob,
                (entity.valid_at or entity.event_time).isoformat(),
                getattr(entity, 'invalid_at', None),
                getattr(entity, 'summary', None),
                getattr(entity, 'attributes', None),
                getattr(entity, 'confidence', None),
                getattr(entity, 'content_format', 'plain'),
            ))

        with self._write_lock:
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR IGNORE INTO entities (id, family_id, name, content, event_time, processed_time, episode_id, source_document, embedding, valid_at, invalid_at, summary, attributes, confidence, content_format)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
                # 同步写入 FTS 表（使用整数 rowid）
                try:
                    # 查询刚插入的实体的整数 rowid
                    aids = [e.absolute_id for e in entities]
                    placeholders = ",".join("?" * len(aids))
                    cursor.execute(f"SELECT rowid, id FROM entities WHERE id IN ({placeholders})", aids)
                    id_to_rowid = {r[1]: r[0] for r in cursor.fetchall()}
                    fts_rows = [(id_to_rowid.get(e.absolute_id), e.name, e.content, e.family_id) for e in entities if id_to_rowid.get(e.absolute_id)]
                    if fts_rows:
                        cursor.executemany("""
                            INSERT OR REPLACE INTO entity_fts(rowid, name, content, family_id)
                            VALUES (?, ?, ?, ?)
                        """, fts_rows)
                except Exception as exc:
                    logger.debug("FTS bulk entity write failed: %s", exc)
                # 设置旧版本 invalid_at (batch) + Phase 2 dual-write
                _inv_rows = [
                    (e.event_time.isoformat(), e.family_id, e.absolute_id)
                    for e in entities
                ]
                try:
                    cursor.executemany("""
                        UPDATE entities SET invalid_at = ?
                        WHERE family_id = ? AND id != ? AND invalid_at IS NULL
                    """, _inv_rows)
                except Exception as exc:
                    logger.debug("bulk invalid_at update failed: %s", exc)
                for entity in entities:
                    _precomp_emb = concept_embeddings.get(entity.absolute_id)
                    self._write_concept_from_entity(entity, cursor, precomputed_embedding=_precomp_emb)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        # Batch sync to VectorStore OUTSIDE lock (after successful commit)
        vec_items = []
        for e in entities:
            if e.embedding:
                vec_items.append((e.absolute_id, e.embedding if isinstance(e.embedding, bytes) else e.embedding))
        if vec_items:
            try:
                vs = self._get_vector_store()
                if vs:
                    vs.upsert_batch("entity_vectors", vec_items)
            except Exception as exc:
                logger.debug("bulk entity vector upsert failed: %s", exc)

    # ------------------------------------------------------------------
    # Read / Query
    # ------------------------------------------------------------------

    def get_entity_by_family_id(self, family_id: str) -> Optional[Entity]:
        """根据family_id获取最新版本的实体"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return None
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM entities
            WHERE family_id = ?
            ORDER BY processed_time DESC
            LIMIT 1
        """, (family_id,))

        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_entity(row)

    def get_entities_by_family_ids(self, family_ids: List[str]) -> Dict[str, Entity]:
        """批量根据 family_id 获取最新版本实体，返回 {family_id: Entity}。"""
        if not family_ids:
            return {}
        # 先 resolve 所有 family_id
        resolved_map = self.resolve_family_ids(list(family_ids))
        valid_fids = set(resolved_map.keys()) | set(resolved_map.values())
        if not valid_fids:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(valid_fids))
        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM entities
            WHERE family_id IN ({placeholders})
            ORDER BY processed_time DESC
        """, list(valid_fids))
        seen = set()
        result: Dict[str, Entity] = {}
        # Build reverse map: resolved_fid -> [orig_fid, ...] for O(1) lookup
        reverse_map: Dict[str, List[str]] = {}
        for orig_fid, resolved_fid in resolved_map.items():
            if orig_fid != resolved_fid:
                reverse_map.setdefault(resolved_fid, []).append(orig_fid)
        for row in cursor.fetchall():
            fid = row[1]
            if fid in seen:
                continue
            seen.add(fid)
            entity = self._row_to_entity(row)
            result[fid] = entity
            # Map original family_ids that resolve to this fid
            for orig_fid in reverse_map.get(fid, []):
                result[orig_fid] = entity
        return result

    def get_entity_by_absolute_id(self, absolute_id: str) -> Optional[Entity]:
        """根据绝对ID获取实体"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM entities
            WHERE id = ?
        """, (absolute_id,))

        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_entity(row)

    def get_entities_by_absolute_ids(self, absolute_ids: List[str]) -> List[Entity]:
        """根据绝对ID列表批量获取实体。"""
        if not absolute_ids:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(absolute_ids))
        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM entities
            WHERE id IN ({placeholders})
        """, tuple(absolute_ids))
        rows = cursor.fetchall()
        return [self._row_to_entity(row) for row in rows]

    def get_entity_names_by_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, str]:
        """批量根据 absolute_id 查询实体名称"""
        if not absolute_ids:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        unique_ids = list(set(absolute_ids))
        placeholders = ','.join('?' * len(unique_ids))
        cursor.execute(f"SELECT id, name FROM entities WHERE id IN ({placeholders})", unique_ids)
        return {row[0]: row[1] or '' for row in cursor.fetchall()}

    def get_entity_version_at_time(self, family_id: str, time_point: datetime) -> Optional[Entity]:
        """获取实体在指定时间点的版本（该时间点之前或等于该时间点的最新版本）"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return None
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM entities
            WHERE family_id = ? AND event_time <= ?
            ORDER BY processed_time DESC
            LIMIT 1
        """, (family_id, time_point.isoformat()))

        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_entity(row)

    def get_entity_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        """获取实体embedding向量的前N个值"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT embedding
            FROM entities
            WHERE id = ?
        """, (absolute_id,))

        row = cursor.fetchone()

        if row is None or row[0] is None:
            return None

        try:
            embedding_array = np.frombuffer(row[0], dtype=np.float32)
            return embedding_array[:num_values].tolist()
        except Exception as exc:
            logger.debug("entity embedding preview failed: %s", exc)
            return None

    def get_entity_versions(self, family_id: str) -> List[Entity]:
        """获取实体的所有版本"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT {self._ENTITY_SELECT} FROM entities WHERE family_id = ? ORDER BY processed_time DESC",
            (family_id,),
        )

        rows = cursor.fetchall()

        return [self._row_to_entity(row) for row in rows]

    def _get_entities_with_embeddings(self) -> List[tuple]:
        """
        获取所有实体的最新版本及其embedding（带短 TTL 缓存）。
        线程安全：使用 _emb_cache_lock 保护缓存的读写。

        Returns:
            List of (Entity, embedding_array) tuples, embedding_array为None表示没有embedding
        """
        now = time.time()
        with self._emb_cache_lock:
            if self._entity_emb_cache is not None and (now - self._entity_emb_cache_ts) < self._emb_cache_ttl:
                return self._entity_emb_cache

        conn = self._get_conn()
        cursor = conn.cursor()

        # 使用窗口函数获取每个 family_id 的最新版本（O(N) 替代 O(N^2) 子查询）
        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY family_id ORDER BY processed_time DESC) AS rn
                FROM entities
                WHERE invalid_at IS NULL
            )
            WHERE rn = 1
        """)

        results = []
        for row in cursor.fetchall():
            embedding_array = None
            if len(row) > 8 and row[8] is not None:
                try:
                    embedding_array = np.frombuffer(row[8], dtype=np.float32)
                except (ValueError, TypeError):
                    embedding_array = None
            entity = self._row_to_entity(row)
            results.append((entity, embedding_array))

        with self._emb_cache_lock:
            self._entity_emb_cache = results
            self._entity_emb_cache_ts = time.time()
        return results

    def get_latest_entities_projection(self, content_snippet_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取最新实体投影，供窗口级批量候选生成使用。"""
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
                "content_snippet": entity.content[:snippet_length],
                "version_count": version_counts.get(entity.family_id, 1),
                "embedding_array": embedding_array,
            })
        return results

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_entities_by_bm25(self, query: str, limit: int = 20) -> List[Entity]:
        """BM25 全文搜索实体。"""
        if not query:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        # FTS5 BM25 搜索，按 bm25 排序
        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT_E},
                   fts.rank AS bm25_score
            FROM entity_fts AS fts
            JOIN entities AS e ON e.rowid = fts.rowid
            WHERE entity_fts MATCH ?
              AND e.invalid_at IS NULL
            ORDER BY fts.rank
            LIMIT ?
        """, (query, limit))

        rows = cursor.fetchall()
        raw_fids = [row[1] for row in rows]
        resolved_map = self.resolve_family_ids(raw_fids) if raw_fids else {}

        entities = []
        for row in rows:
            ent = self._row_to_entity(row)
            resolved_fid = resolved_map.get(row[1], row[1])
            if resolved_fid != row[1]:
                ent.family_id = resolved_fid
            entities.append(ent)
        return entities

    def search_entities_by_similarity(self, query_name: str, query_content: Optional[str] = None,
                                     threshold: float = 0.7, max_results: int = 10,
                                     content_snippet_length: int = 50,
                                     text_mode: Literal["name_only", "content_only", "name_and_content"] = "name_and_content",
                                     similarity_method: Literal["embedding", "text", "jaccard", "bleu"] = "embedding",
                                     _preloaded_entities: Optional[List[tuple]] = None) -> List[Entity]:
        """
        根据名称相似度搜索实体

        支持多种检索模式：
        - text_mode: 使用哪些字段进行检索（name_only/content_only/name_and_content）
        - similarity_method: 使用哪种相似度计算方法（embedding/text/jaccard/bleu）

        Args:
            query_name: 查询的实体名称
            query_content: 查询的实体内容（可选）
            threshold: 相似度阈值
            max_results: 返回的最大相似实体数量（默认10）
            content_snippet_length: 用于相似度搜索的content截取长度（默认50字符）
            text_mode: 文本模式
                - "name_only": 只使用name进行检索
                - "content_only": 只使用content进行检索
                - "name_and_content": 使用name + content进行检索
            similarity_method: 相似度计算方法
                - "embedding": 使用embedding向量相似度（优先使用已存储的embedding）
                - "text": 使用文本序列相似度（SequenceMatcher）
                - "jaccard": 使用Jaccard相似度
                - "bleu": 使用BLEU相似度
        """
        # 获取所有实体及其embedding（优先使用预加载数据，避免N+1）
        entities_with_embeddings = _preloaded_entities if _preloaded_entities is not None else self._get_entities_with_embeddings()

        if not entities_with_embeddings:
            return []

        all_entities = [e for e, _ in entities_with_embeddings]

        # 根据text_mode构建查询文本
        if text_mode == "name_only":
            query_text = query_name
            use_content = False
        elif text_mode == "content_only":
            if not query_content:
                return []  # 如果没有content，无法检索
            query_text = query_content
            use_content = True
        else:  # name_and_content
            if query_content:
                query_text = f"{query_name} {query_content}"
            else:
                query_text = query_name
            use_content = query_content is not None

        # 根据similarity_method选择检索方式
        if similarity_method == "embedding" and self.embedding_client and self.embedding_client.is_available():
            return self._search_with_embedding(
                query_text, entities_with_embeddings, threshold,
                use_content, max_results, content_snippet_length, text_mode
            )
        else:
            # 使用文本相似度（text/jaccard/bleu）
            return self._search_with_text_similarity(
                query_text, all_entities, threshold,
                use_content, max_results, content_snippet_length,
                text_mode, similarity_method
            )

    # ------------------------------------------------------------------
    # Bulk read
    # ------------------------------------------------------------------

    def get_all_entities(self, limit: Optional[int] = None, offset: Optional[int] = None, exclude_embedding: bool = False) -> List[Entity]:
        """获取所有实体的最新版本

        Args:
            limit: 限制返回的实体数量（按时间倒序），None表示不限制
            offset: 跳过前N条记录，None表示不跳过
            exclude_embedding: 是否排除 embedding 字段（前端展示等不需要 embedding 的场景应设为 True）
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        e1_cols = self._ENTITY_SELECT_E1
        if exclude_embedding:
            e1_cols = e1_cols.replace("e1.embedding", "NULL AS e1__embedding")
        query = f"""
            SELECT {e1_cols}
            FROM entities e1
            INNER JOIN (
                SELECT family_id, MAX(processed_time) as max_time
                FROM entities
                GROUP BY family_id
            ) e2 ON e1.family_id = e2.family_id AND e1.processed_time = e2.max_time
            ORDER BY e1.processed_time DESC
        """

        if limit is not None and offset is not None and offset > 0:
            query += f" LIMIT {int(limit)} OFFSET {int(offset)}"
        elif limit is not None:
            query += f" LIMIT {int(limit)}"

        cursor.execute(query)

        rows = cursor.fetchall()

        return [self._row_to_entity(row) for row in rows]

    def get_all_entities_before_time(self, time_point: datetime, limit: Optional[int] = None,
                                     exclude_embedding: bool = False) -> List[Entity]:
        """获取指定时间点之前或等于该时间点的所有实体的最新版本

        Args:
            time_point: 时间点
            limit: 限制返回的实体数量（按时间倒序），None表示不限制
            exclude_embedding: 是否排除 embedding 字段
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        e1_cols = self._ENTITY_SELECT_E1
        if exclude_embedding:
            e1_cols = e1_cols.replace("e1.embedding", "NULL AS e1__embedding")
        query = f"""
            SELECT {e1_cols}
            FROM entities e1
            INNER JOIN (
                SELECT family_id, MAX(processed_time) as max_time
                FROM entities
                WHERE event_time <= ?
                GROUP BY family_id
            ) e2 ON e1.family_id = e2.family_id AND e1.processed_time = e2.max_time
            ORDER BY e1.processed_time DESC
        """

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        cursor.execute(query, (time_point.isoformat(),))

        rows = cursor.fetchall()

        return [self._row_to_entity(row) for row in rows]

    # ------------------------------------------------------------------
    # Version count
    # ------------------------------------------------------------------

    def get_entity_version_count(self, family_id: str) -> int:
        """获取指定family_id的版本数量

        Args:
            family_id: 实体ID

        Returns:
            版本数量
        """
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM entities WHERE family_id = ?
        """, (family_id,))

        count = cursor.fetchone()[0]

        return count

    def get_entity_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
        """批量获取多个family_id的版本数量

        Args:
            family_ids: 实体ID列表

        Returns:
            Dict[family_id, version_count]
        """
        if not family_ids:
            return {}
        # 批量解析重定向
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = []
        seen_canonical = set()
        for family_id in family_ids:
            canonical_id = resolved_map.get(family_id, family_id)
            if canonical_id and canonical_id not in seen_canonical:
                seen_canonical.add(canonical_id)
                canonical_ids.append(canonical_id)
        if not canonical_ids:
            return {}

        conn = self._get_conn()
        cursor = conn.cursor()

        # 使用IN子句批量查询
        placeholders = ','.join(['?'] * len(canonical_ids))
        cursor.execute(f"""
            SELECT family_id, COUNT(*) as version_count
            FROM entities
            WHERE family_id IN ({placeholders})
            GROUP BY family_id
        """, canonical_ids)

        rows = cursor.fetchall()

        return {row[0]: row[1] for row in rows}

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_entity_by_id(self, family_id: str) -> int:
        """删除实体的所有版本。返回删除的行数。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM entities WHERE family_id = ?", (family_id,))
            # 清理 FTS 表
            try:
                cursor.execute("DELETE FROM entity_fts WHERE family_id = ?", (family_id,))
            except Exception as exc:
                logger.warning("FTS delete failed: %s", exc)
            # Phase 2: cleanup concepts
            self._delete_concepts_by_family(family_id, cursor)
            conn.commit()
            return cursor.rowcount

    def delete_entity_all_versions(self, family_id: str) -> int:
        """删除实体的所有版本（含重定向解析）。返回删除的行数。"""
        return self.delete_entity_by_id(family_id)

    def batch_delete_entities(self, family_ids: List[str]) -> int:
        """批量删除实体 — 单次事务，替代 N 次单独删除。含 concepts 清理。"""
        resolved_map = self.resolve_family_ids(family_ids)
        resolved = list({r for r in resolved_map.values() if r})
        if not resolved:
            return 0
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(resolved))
            # Batch cleanup concepts table
            self._batch_delete_concepts_by_families(resolved, cursor)
            cursor.execute(f"DELETE FROM entities WHERE family_id IN ({placeholders})", tuple(resolved))
            count = cursor.rowcount
            try:
                cursor.execute(f"DELETE FROM entity_fts WHERE family_id IN ({placeholders})", tuple(resolved))
            except Exception as exc:
                logger.warning("FTS delete failed: %s", exc)
            conn.commit()
            return count

    def delete_entity_by_absolute_id(self, absolute_id: str) -> bool:
        """根据绝对ID删除实体，同时清理 FTS 索引。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            # Check existence
            cursor.execute(
                "SELECT 1 FROM entities WHERE id = ?",
                (absolute_id,),
            )
            if cursor.fetchone() is None:
                return False
            # Delete FTS entry (using integer rowid)
            cursor.execute(
                "DELETE FROM entity_fts WHERE rowid = (SELECT rowid FROM entities WHERE id = ?)",
                (absolute_id,),
            )
            # Delete entity version
            cursor.execute(
                "DELETE FROM entities WHERE id = ?",
                (absolute_id,),
            )
            affected = cursor.rowcount
            # Phase 2: cleanup concepts
            self._delete_concept_by_id(absolute_id, cursor)
            conn.commit()
            return affected > 0

    def batch_delete_entity_versions_by_absolute_ids(self, absolute_ids: List[str]) -> int:
        """批量删除指定实体版本（带 FTS 和 concepts 清理），返回成功删除的数量。"""
        if not absolute_ids:
            return 0
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(absolute_ids))
            # Batch cleanup concepts + concept_fts (3N → 2 queries)
            cursor.execute(
                f"SELECT rowid FROM concepts WHERE id IN ({placeholders})",
                list(absolute_ids),
            )
            concept_rowids = [row[0] for row in cursor.fetchall()]
            if concept_rowids:
                try:
                    cph = ",".join("?" * len(concept_rowids))
                    cursor.execute(f"DELETE FROM concept_fts WHERE rowid IN ({cph})", concept_rowids)
                except Exception as exc:
                    logger.debug("batch concept_fts delete failed: %s", exc)
            cursor.execute(
                f"DELETE FROM concepts WHERE id IN ({placeholders})",
                list(absolute_ids),
            )
            # Delete FTS entries (using integer rowids)
            cursor.execute(
                f"DELETE FROM entity_fts WHERE rowid IN (SELECT rowid FROM entities WHERE id IN ({placeholders}))",
                list(absolute_ids),
            )
            # Delete entity versions
            cursor.execute(
                f"DELETE FROM entities WHERE id IN ({placeholders})",
                list(absolute_ids),
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_entity_families(self, target_family_id: str, source_family_ids: List[str]) -> Dict[str, Any]:
        """
        将多个source_family_id的记录合并到target_family_id

        Args:
            target_family_id: 目标实体ID（保留的ID）
            source_family_ids: 要合并的源实体ID列表

        Returns:
            合并结果统计，包含更新的实体数量和关系数量
        """
        target_family_id = self.resolve_family_id(target_family_id)
        if not target_family_id or not source_family_ids:
            return {"entities_updated": 0, "relations_updated": 0}

        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            entities_updated = 0
            relations_updated = 0

            try:
                # 1. Resolve all source family_ids → canonical IDs (batch)
                canonical_source_ids: List[str] = []
                for source_id in source_family_ids:
                    source_id = self._resolve_family_id_with_cursor(cursor, source_id)
                    if not source_id or source_id == target_family_id or source_id in canonical_source_ids:
                        continue
                    canonical_source_ids.append(source_id)

                if not canonical_source_ids:
                    return {"entities_updated": 0, "relations_updated": 0}

                # 2. Batch: get version counts for verification
                ph = ",".join("?" * len(canonical_source_ids))
                cursor.execute(
                    f"SELECT family_id, COUNT(*) FROM entities WHERE family_id IN ({ph}) GROUP BY family_id",
                    tuple(canonical_source_ids),
                )
                source_version_counts = {row[0]: row[1] for row in cursor.fetchall()}

                # 3. Batch UPDATE: move all source entities to target
                cursor.execute(
                    f"UPDATE entities SET family_id = ? WHERE family_id IN ({ph})",
                    (target_family_id, *canonical_source_ids),
                )
                entities_updated = cursor.rowcount

                # 4. Batch verify: check for residual records
                cursor.execute(
                    f"SELECT family_id, COUNT(*) FROM entities WHERE family_id IN ({ph}) GROUP BY family_id",
                    tuple(canonical_source_ids),
                )
                residuals = cursor.fetchall()
                if residuals:
                    conn.rollback()
                    residual_info = ", ".join(f"{r[0]}: {r[1]}" for r in residuals)
                    raise ValueError(f"合并失败：仍有残留记录 — {residual_info}")

                # 5. Batch INSERT redirects
                now_iso = datetime.now().isoformat()
                cursor.executemany(
                    "INSERT INTO entity_redirects (source_family_id, target_family_id, updated_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(source_family_id) DO UPDATE SET "
                    "  target_family_id = excluded.target_family_id, "
                    "  updated_at = excluded.updated_at",
                    [(sid, target_family_id, now_iso) for sid in canonical_source_ids],
                )

                # 6. Update concepts table: move source family_ids to target
                cursor.execute(
                    f"UPDATE concepts SET family_id = ? WHERE family_id IN ({ph}) AND role = 'entity'",
                    (target_family_id, *canonical_source_ids),
                )
                concepts_updated = cursor.rowcount

                # 7. Count relations now reachable through the merged target family
                cursor.execute(
                    "SELECT id FROM entities WHERE family_id = ?",
                    (target_family_id,),
                )
                all_target_abs = [row[0] for row in cursor.fetchall()]
                if all_target_abs:
                    ph = ",".join("?" * len(all_target_abs))
                    cursor.execute(
                        f"SELECT COUNT(DISTINCT id) FROM relations "
                        f"WHERE entity1_absolute_id IN ({ph}) "
                        f"   OR entity2_absolute_id IN ({ph})",
                        (*all_target_abs, *all_target_abs),
                    )
                    relations_updated = cursor.fetchone()[0]

                conn.commit()

            except Exception as e:
                conn.rollback()
                raise e

            return {
                "entities_updated": entities_updated,
                "relations_updated": relations_updated,
                "target_family_id": target_family_id,
                "merged_source_ids": canonical_source_ids
            }

    # ------------------------------------------------------------------
    # Name lookup
    # ------------------------------------------------------------------

    def get_family_ids_by_names(self, names: list) -> dict:
        """按名称批量查询实 family_id（每个 name 取最新版本）。

        Returns:
            {name: family_id} 仅包含能找到的名称。
        """
        if not names:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ','.join(['?'] * len(names))
        cursor.execute(f"""
            SELECT name, family_id FROM entities
            WHERE name IN ({placeholders})
            ORDER BY processed_time DESC
        """, names)
        result = {}
        pending_resolve = {}
        for name, eid in cursor.fetchall():
            if name not in result:
                pending_resolve[name] = eid
        # Batch resolve all family_ids at once
        if pending_resolve:
            resolved_map = self.resolve_family_ids(list(pending_resolve.values()))
            for name, eid in pending_resolve.items():
                result[name] = resolved_map.get(eid, eid)
        return result

    # ------------------------------------------------------------------
    # Update (Phase A: entity intelligence)
    # ------------------------------------------------------------------

    def _update_latest_entity_field(self, family_id: str, field: str, value, cursor):
        """Update a single field on the latest version of an entity and sync to concepts.

        Consolidates the repeated pattern: UPDATE with subquery → re-SELECT id → concept sync.
        """
        cursor.execute(
            f"UPDATE entities SET {field} = ? "
            "WHERE id = (SELECT id FROM entities WHERE family_id = ? ORDER BY processed_time DESC LIMIT 1)",
            (value, family_id),
        )
        cursor.execute(
            "SELECT id FROM entities WHERE family_id = ? ORDER BY processed_time DESC LIMIT 1",
            (family_id,),
        )
        row = cursor.fetchone()
        if row:
            self._sync_concept_entity_fields(row[0], {field: value}, cursor)

    def update_entity_summary(self, family_id: str, summary: str):
        """更新实体的摘要（最新版本）。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            self._update_latest_entity_field(family_id, "summary", summary, cursor)
            conn.commit()

    def batch_update_entity_summaries(self, updates: Dict[str, str]):
        """Batch update summaries for multiple entities in a single transaction.

        Resolves all latest absolute_ids in one query, then batch-updates
        entities and concepts instead of N×3 SQL calls.
        """
        if not updates:
            return
        family_ids = list(updates)
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            # Resolve all latest absolute_ids in one query
            placeholders = ",".join("?" * len(family_ids))
            cursor.execute(f"""
                SELECT family_id, id FROM entities
                WHERE id IN (
                    SELECT id FROM entities
                    WHERE family_id IN ({placeholders})
                    GROUP BY family_id
                    HAVING processed_time = MAX(processed_time)
                )
            """, family_ids)
            fid_to_absid = {row[0]: row[1] for row in cursor.fetchall()}
            # Batch UPDATE entities
            update_data = [(updates[fid], fid) for fid in family_ids if fid in fid_to_absid]
            if update_data:
                cursor.executemany(
                    "UPDATE entities SET summary = ? "
                    "WHERE id = (SELECT id FROM entities WHERE family_id = ? ORDER BY processed_time DESC LIMIT 1)",
                    update_data,
                )
                # Batch sync concepts — single executemany instead of N individual UPDATEs
                concept_sync_data = [
                    (updates[fid], fid_to_absid[fid])
                    for fid, _ in update_data
                    if fid in fid_to_absid
                ]
                if concept_sync_data:
                    cursor.executemany(
                        "UPDATE concepts SET summary = ? WHERE id = ? AND role = 'entity'",
                        concept_sync_data,
                    )
            conn.commit()

    def update_entity_attributes(self, family_id: str, attributes: str):
        """更新实体的属性字典（JSON 字符串）。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            self._update_latest_entity_field(family_id, "attributes", attributes, cursor)
            conn.commit()

    def update_entity_confidence(self, family_id: str, confidence: float):
        """更新实体最新版本的置信度。

        置信度调整规则（vision.md "置信度"）：
        - 多个独立来源印证同一事实 → 置信度上升
        - 新证据与之矛盾 → 置信度下降
        - 值域 [0.0, 1.0]，超出范围截断
        """
        confidence = max(0.0, min(1.0, confidence))
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            self._update_latest_entity_field(family_id, "confidence", confidence, cursor)
            conn.commit()

    def adjust_confidence_on_corroboration(self, family_id: str, source_type: str = "entity",
                                            is_dream: bool = False):
        """独立来源印证时提升置信度。

        使用 Bayesian-inspired 增量调整：
        - 每次印证 +0.05，上限 1.0
        - Dream 来源印证权重减半 (+0.025)

        Args:
            family_id: 实体或关系的 family_id
            source_type: "entity" 或 "relation"，决定查询哪个表
            is_dream: 是否来自 Dream 产物，Dream 权重减半
        """
        table = self._validate_table_name(source_type)
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT id, confidence FROM {table}
                WHERE family_id = ?
                ORDER BY processed_time DESC LIMIT 1
            """, (family_id,))
            row = cursor.fetchone()
            if not row or row[1] is None:
                return
            abs_id, current = row[0], row[1]
            delta = 0.025 if is_dream else 0.05
            new_conf = min(1.0, current + delta)
            cursor.execute(f"""
                UPDATE {table} SET confidence = ?
                WHERE id = ?
            """, (new_conf, abs_id))
            # Sync to concepts table
            sync_fn = self._sync_concept_entity_fields if source_type == "entity" else self._sync_concept_relation_fields
            sync_fn(abs_id, {"confidence": new_conf}, cursor)
            conn.commit()

    def adjust_confidence_on_contradiction(self, family_id: str, source_type: str = "entity"):
        """矛盾证据时降低置信度。

        - 每次矛盾 -0.1，下限 0.0
        """
        table = self._validate_table_name(source_type)
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT id, confidence FROM {table}
                WHERE family_id = ?
                ORDER BY processed_time DESC LIMIT 1
            """, (family_id,))
            row = cursor.fetchone()
            if not row or row[1] is None:
                return
            abs_id, current = row[0], row[1]
            new_conf = max(0.0, current - 0.1)
            cursor.execute(f"""
                UPDATE {table} SET confidence = ?
                WHERE id = ?
            """, (new_conf, abs_id))
            # Sync to concepts table
            sync_fn = self._sync_concept_entity_fields if source_type == "entity" else self._sync_concept_relation_fields
            sync_fn(abs_id, {"confidence": new_conf}, cursor)
            conn.commit()

    def adjust_confidence_on_contradiction_batch(self, family_ids: list, source_type: str = "entity"):
        """Batch version — lowers confidence for multiple family_ids in a single transaction.

        Resolves all latest absolute_ids in one query, then batch-updates
        instead of N×3 SQL calls.
        """
        if not family_ids:
            return
        table = self._validate_table_name(source_type)
        sync_fn = self._sync_concept_entity_fields if source_type == "entity" else self._sync_concept_relation_fields
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            # Resolve all latest (id, confidence) pairs in one query
            placeholders = ",".join("?" * len(family_ids))
            cursor.execute(f"""
                SELECT e.id, e.confidence FROM {table} e
                INNER JOIN (
                    SELECT family_id, MAX(processed_time) as max_pt
                    FROM {table} WHERE family_id IN ({placeholders})
                    GROUP BY family_id
                ) latest ON e.family_id = latest.family_id AND e.processed_time = latest.max_pt
            """, family_ids)
            rows = cursor.fetchall()

            # Batch UPDATE: compute new confidences
            update_data = []
            concept_syncs = []
            for abs_id, current in rows:
                if current is None:
                    continue
                new_conf = max(0.0, current - 0.1)
                update_data.append((new_conf, abs_id))
                concept_syncs.append((abs_id, new_conf))
            if update_data:
                cursor.executemany(
                    f"UPDATE {table} SET confidence = ? WHERE id = ?",
                    update_data,
                )
                # Batch sync concepts — single executemany instead of N individual UPDATEs
                role = "entity" if source_type == "entity" else "relation"
                cursor.executemany(
                    f"UPDATE concepts SET confidence = ? WHERE id = ? AND role = '{role}'",
                    concept_syncs,
                )
            conn.commit()

    # ------------------------------------------------------------------
    # Version-level CRUD (Phase 2)
    # ------------------------------------------------------------------

    def update_entity_by_absolute_id(self, absolute_id: str, **fields) -> Optional[Entity]:
        """根据绝对ID更新实体字段（name, content, summary, attributes, confidence）。

        当 name 或 content 变更时，自动重算 embedding 并同步到 VectorStore。
        Uses single multi-column UPDATE instead of per-field loop.
        """
        allowed = {"name", "content", "summary", "attributes", "confidence"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_entity_by_absolute_id(absolute_id)

        # Pre-compute embedding OUTSIDE lock (ML inference is slow)
        _precomputed_emb_blob = None
        needs_emb_update = "name" in updates or "content" in updates
        if needs_emb_update and self.embedding_client and self.embedding_client.is_available():
            name = updates.get("name")
            content = updates.get("content")
            if name is not None or content is not None:
                current = self.get_entity_by_absolute_id(absolute_id)
                if current:
                    emb_name = name if name is not None else current.name
                    emb_content = content if content is not None else current.content
                    emb_text = f"{emb_name} {emb_content}".strip()
                    _precomputed_emb_blob = self._compute_embedding_text(emb_text)

        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            # Single multi-column UPDATE
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            cursor.execute(
                f"UPDATE entities SET {set_clause} WHERE id = ?",
                list(updates.values()) + [absolute_id],
            )

            # Write pre-computed embedding
            if _precomputed_emb_blob is not None:
                cursor.execute(
                    "UPDATE entities SET embedding = ? WHERE id = ?",
                    (_precomputed_emb_blob, absolute_id),
                )

            # Sync to concepts table
            self._sync_concept_entity_fields(absolute_id, updates, cursor)
            conn.commit()

        # Vector store I/O OUTSIDE lock
        if _precomputed_emb_blob is not None:
            self._vector_store_upsert_entity(absolute_id, _precomputed_emb_blob)
            self._invalidate_emb_cache()

        return self.get_entity_by_absolute_id(absolute_id)

    def split_entity_version(self, absolute_id: str, new_family_id: str = "") -> Optional[Entity]:
        """将实体版本从当前 family 拆分到新 family。"""
        if not new_family_id:
            new_family_id = f"ent_{uuid.uuid4().hex[:12]}"

        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            # Check existence and get old family_id
            cursor.execute(
                "SELECT family_id FROM entities WHERE id = ?",
                (absolute_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            # Update family_id
            cursor.execute(
                "UPDATE entities SET family_id = ? WHERE id = ?",
                (new_family_id, absolute_id),
            )
            # Sync family_id change to concepts table
            try:
                cursor.execute(
                    "UPDATE concepts SET family_id = ? WHERE id = ? AND role = 'entity'",
                    (new_family_id, absolute_id),
                )
            except Exception as exc:
                logger.debug("concept family_id sync failed in split: %s", exc)
            conn.commit()

        return self.get_entity_by_absolute_id(absolute_id)
