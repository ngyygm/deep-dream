"""
RelationStoreMixin — relation CRUD, search & utility methods extracted from StorageManager.

Shared state (accessed via ``self``, defined in the host StorageManager):
    - self._get_conn()
    - self._write_lock
    - self._local
    - self._RELATION_SELECT, self._ENTITY_SELECT
    - self._safe_parse_datetime()
    - self._normalize_datetime_for_compare()
    - self._invalidate_emb_cache()
    - self.embedding_client
    - self.relation_content_snippet_length
    - self._relation_emb_cache, self._relation_emb_cache_ts, self._emb_cache_ttl
    - self.resolve_family_id() / resolve_family_ids()
    - self._resolve_family_id_with_cursor()
    - self.get_entity_by_family_id()
    - self.get_entity_versions()
    - self.get_entity_version_counts()
    - self._row_to_entity()
    - self._write_concept_from_relation()
    - self._delete_concept_by_id()
    - self._delete_concepts_by_family()
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

import difflib
import numpy as np
from functools import lru_cache

from ...models import Entity, Relation

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4096)
def _cached_bigram_set(text: str) -> frozenset:
    """Cached bigram set for Jaccard pre-filter before expensive SequenceMatcher."""
    if len(text) < 2:
        return frozenset()
    return frozenset(text[i:i+2] for i in range(len(text) - 1))

# Pre-computed dream filter SQL fragments
_DREAM_FILTER_R = (
    " AND (r.attributes IS NULL OR json_extract(r.attributes, '$.tier') != 'candidate'"
    " OR json_extract(r.attributes, '$.status') != 'hypothesized')"
)
_DREAM_FILTER_SUB = (
    " AND (attributes IS NULL OR json_extract(attributes, '$.tier') != 'candidate'"
    " OR json_extract(attributes, '$.status') != 'hypothesized')"
)

class RelationStoreMixin:
    """Mixin providing all relation-related storage operations."""

    # ------------------------------------------------------------------
    # Dream candidate filtering
    # ------------------------------------------------------------------

    def _is_dream_candidate(self, relation: Relation) -> bool:
        """Check if a relation is a dream candidate that should be hidden from normal search.

        Filters out both hypothesized and rejected candidates.
        Verified/promoted candidates are NOT filtered (they appear in normal search).
        """
        _a = relation.attributes
        if not _a:
            return False
        # Fast string probe: skip json.loads for the vast majority of relations
        if isinstance(_a, str) and ('"candidate"' not in _a or '"tier"' not in _a):
            return False
        try:
            attrs = json.loads(_a) if isinstance(_a, str) else _a
            tier = attrs.get("tier")
            status = attrs.get("status")
            return tier == "candidate" and status != "verified"
        except (json.JSONDecodeError, TypeError, AttributeError):
            return False

    def _filter_dream_candidates(self, relations: List[Relation],
                                  include_candidates: bool = False) -> List[Relation]:
        """Filter out dream candidate relations unless explicitly requested.

        Removes hypothesized and rejected candidates from normal search results.
        Verified/promoted candidates are always shown.
        """
        if include_candidates or not relations:
            return relations
        return [r for r in relations if not self._is_dream_candidate(r)]

    # ------------------------------------------------------------------
    # Row helper
    # ------------------------------------------------------------------

    def _row_to_relation(self, row) -> Relation:
        """Convert a SELECT row tuple to a Relation object using _RELATION_SELECT column order."""
        return Relation(
            absolute_id=row[0],
            family_id=row[1],
            entity1_absolute_id=row[2],
            entity2_absolute_id=row[3],
            content=row[4],
            event_time=self._safe_parse_datetime(row[5]),
            processed_time=self._safe_parse_datetime(row[6]),
            episode_id=row[7],
            source_document=row[8] if len(row) > 8 else '',
            embedding=row[9] if len(row) > 9 else None,
            summary=row[10] if len(row) > 10 else None,
            attributes=row[11] if len(row) > 11 else None,
            confidence=row[12] if len(row) > 12 else None,
            valid_at=self._safe_parse_datetime(row[13]) if len(row) > 13 and row[13] else None,
            invalid_at=self._safe_parse_datetime(row[14]) if len(row) > 14 and row[14] else None,
            content_format=row[15] if len(row) > 15 else 'plain',
            provenance=row[16] if len(row) > 16 else None,
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _resolve_entity_names_for_embedding(self, relation: Relation) -> Tuple[str, str]:
        """解析关系两端实体的名称，用于 embedding 编码。失败时返回空字符串。"""
        try:
            names = self.get_entity_names_by_absolute_ids(
                [relation.entity1_absolute_id, relation.entity2_absolute_id]
            )
            return (
                names.get(relation.entity1_absolute_id, ""),
                names.get(relation.entity2_absolute_id, ""),
            )
        except Exception:
            return "", ""

    def _build_relation_embedding_text(self, relation: Relation,
                                        entity1_name: str = "", entity2_name: str = "") -> str:
        """构建用于 embedding 编码的文本。包含两端实体名称 + 关系内容。"""
        parts = []
        if entity1_name:
            parts.append(entity1_name)
        if relation.content:
            parts.append(relation.content)
        if entity2_name:
            parts.append(entity2_name)
        return " ".join(parts) if parts else relation.content

    def _compute_relation_embedding(self, relation: Relation) -> Optional[bytes]:
        """计算关系的embedding向量并转换为BLOB。

        编码内容: "{entity1_name} {content} {entity2_name}"
        实体名称使关系 embedding 区分不同实体对之间的语义差异。
        """
        if not self.embedding_client or not self.embedding_client.is_available():
            return None

        # 解析实体名称以丰富 embedding
        name1, name2 = self._resolve_entity_names_for_embedding(relation)
        text = self._build_relation_embedding_text(relation, name1, name2)

        embedding = self.embedding_client.encode(text)

        if embedding is None or not embedding:
            return None

        # 转换为numpy数组并序列化为BLOB
        embedding_array = np.array(embedding[0] if isinstance(embedding, list) else embedding, dtype=np.float32)
        return embedding_array.tobytes()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_relation(self, relation: Relation):
        """保存关系（包含预计算的embedding向量）"""
        self._invalidate_emb_cache()
        # 计算embedding（无需锁）
        embedding_blob = self._compute_relation_embedding(relation)
        relation.embedding = embedding_blob
        # 预解析实体名称（锁外，避免 _write_concept_from_relation 内 N+1）
        _e1_name = _e2_name = None
        try:
            _e1 = self.get_entity_by_absolute_id(relation.entity1_absolute_id)
            _e1_name = _e1.name if _e1 else ""
        except Exception:
            pass
        try:
            _e2 = self.get_entity_by_absolute_id(relation.entity2_absolute_id)
            _e2_name = _e2.name if _e2 else ""
        except Exception:
            pass
        # Pre-compute concept embedding OUTSIDE lock
        _concept_emb = self._compute_concept_embedding(
            role='relation', content=relation.content,
            extra=f"{_e1_name or ''} {_e2_name or ''}".strip()
        )

        with self._write_lock:
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO relations (id, family_id, entity1_absolute_id, entity2_absolute_id, content, event_time, processed_time, episode_id, source_document, embedding, valid_at, invalid_at, summary, attributes, confidence, provenance, content_format)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    relation.absolute_id,
                    relation.family_id,
                    relation.entity1_absolute_id,
                    relation.entity2_absolute_id,
                    relation.content,
                    relation.event_time.isoformat(),
                    relation.processed_time.isoformat(),
                    relation.episode_id,
                    relation.source_document,
                    embedding_blob,
                    (relation.valid_at or relation.event_time).isoformat(),
                    getattr(relation, 'invalid_at', None),
                    getattr(relation, 'summary', None),
                    getattr(relation, 'attributes', None),
                    getattr(relation, 'confidence', None),
                    getattr(relation, 'provenance', None),
                    getattr(relation, 'content_format', 'plain'),
                ))
                # 同步写入 FTS 表（使用整数 rowid）
                try:
                    cursor.execute("""
                        INSERT INTO relation_fts(rowid, content, family_id)
                        VALUES (?, ?, ?)
                    """, (cursor.lastrowid, relation.content, relation.family_id))
                except Exception as exc:
                    logger.warning("FTS relation write failed: %s", exc)
                # 设置旧版本 invalid_at
                try:
                    cursor.execute("""
                        UPDATE relations SET invalid_at = ?
                        WHERE family_id = ? AND id != ? AND invalid_at IS NULL
                    """, (relation.event_time.isoformat(), relation.family_id, relation.absolute_id))
                    # 同步更新 concepts 表中旧版本的 invalid_at
                    cursor.execute("""
                        UPDATE concepts SET invalid_at = ?
                        WHERE family_id = ? AND id != ? AND role = 'relation' AND invalid_at IS NULL
                    """, (relation.event_time.isoformat(), relation.family_id, relation.absolute_id))
                except Exception as exc:
                    logger.warning("FTS relation invalid_at update failed: %s", exc)
                # Phase 2: dual-write to concepts (names + embedding pre-computed above)
                self._write_concept_from_relation(relation, cursor, e1_name=_e1_name, e2_name=_e2_name, precomputed_embedding=_concept_emb)
                # 单次 commit 包含所有写操作
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        # Vector store I/O OUTSIDE lock (after successful commit)
        self._vector_store_upsert_relation(relation.absolute_id, embedding_blob)

    def bulk_save_relations(self, relations: List[Relation]):
        """批量保存关系，使用批量 embedding 与单事务写入。"""
        if not relations:
            return

        # Batch resolve all entity names upfront (1 query instead of 2N)
        _abs_ids: set = set()
        for r in relations:
            if r.entity1_absolute_id:
                _abs_ids.add(r.entity1_absolute_id)
            if r.entity2_absolute_id:
                _abs_ids.add(r.entity2_absolute_id)
        all_abs_ids = list(_abs_ids)
        _name_map = {}
        if all_abs_ids:
            try:
                _name_map = self.get_entity_names_by_absolute_ids(all_abs_ids)
            except Exception:
                pass
        _resolved_names = {}
        for relation in relations:
            _resolved_names[relation.absolute_id] = (
                _name_map.get(relation.entity1_absolute_id, ""),
                _name_map.get(relation.entity2_absolute_id, ""),
            )

        # 批量计算 embedding（无需锁）— 使用预解析的名称
        embeddings = None
        concept_embeddings: Dict[str, Optional[bytes]] = {}
        if self.embedding_client and self.embedding_client.is_available():
            texts = []
            for relation in relations:
                name1, name2 = _resolved_names[relation.absolute_id]
                texts.append(self._build_relation_embedding_text(relation, name1, name2))
            embeddings = self.embedding_client.encode(texts)
            # Batch compute concept embeddings (includes entity names in text)
            concept_texts = []
            for relation in relations:
                name1, name2 = _resolved_names[relation.absolute_id]
                _extra = f"{name1} {name2}".strip()
                _c_text = f"{_extra} {relation.content[:512]}" if _extra else relation.content[:512]
                concept_texts.append(_c_text)
            concept_embs = self.embedding_client.encode(concept_texts)
            if concept_embs is not None:
                for i, relation in enumerate(relations):
                    try:
                        concept_embeddings[relation.absolute_id] = np.array(
                            concept_embs[i] if isinstance(concept_embs[i], (list, np.ndarray)) else concept_embs[i],
                            dtype=np.float32
                        ).tobytes()
                    except Exception:
                        concept_embeddings[relation.absolute_id] = None

        rows = []
        for idx, relation in enumerate(relations):
            embedding_blob = None
            if embeddings is not None:
                try:
                    embedding_blob = np.array(embeddings[idx], dtype=np.float32).tobytes()
                except Exception as exc:
                    logger.debug("embedding encode failed: %s", exc)
                    embedding_blob = None
            relation.embedding = embedding_blob
            rows.append((
                relation.absolute_id,
                relation.family_id,
                relation.entity1_absolute_id,
                relation.entity2_absolute_id,
                relation.content,
                relation.event_time.isoformat(),
                relation.processed_time.isoformat(),
                relation.episode_id,
                relation.source_document,
                embedding_blob,
                (relation.valid_at or relation.event_time).isoformat(),
                getattr(relation, 'invalid_at', None),
                getattr(relation, 'summary', None),
                getattr(relation, 'attributes', None),
                getattr(relation, 'confidence', None),
                getattr(relation, 'provenance', None),
                getattr(relation, 'content_format', 'plain'),
            ))

        with self._write_lock:
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO relations (id, family_id, entity1_absolute_id, entity2_absolute_id, content, event_time, processed_time, episode_id, source_document, embedding, valid_at, invalid_at, summary, attributes, confidence, provenance, content_format)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
                # 同步写入 FTS 表（使用整数 rowid）
                try:
                    aids = [r.absolute_id for r in relations]
                    placeholders = ",".join("?" * len(aids))
                    cursor.execute(f"SELECT rowid, id FROM relations WHERE id IN ({placeholders})", aids)
                    id_to_rowid = {r[1]: r[0] for r in cursor.fetchall()}
                    fts_rows = [(id_to_rowid.get(r.absolute_id), r.content, r.family_id) for r in relations if id_to_rowid.get(r.absolute_id)]
                    if fts_rows:
                        cursor.executemany("""
                            INSERT OR REPLACE INTO relation_fts(rowid, content, family_id)
                            VALUES (?, ?, ?)
                        """, fts_rows)
                except Exception as exc:
                    logger.debug("FTS bulk relation write failed: %s", exc)
                # 设置旧版本 invalid_at (batch executemany) + Phase 2 dual-write (with pre-resolved names)
                invalid_at_rows = [
                    (r.event_time.isoformat(), r.family_id, r.absolute_id)
                    for r in relations
                ]
                try:
                    cursor.executemany("""
                        UPDATE relations SET invalid_at = ?
                        WHERE family_id = ? AND id != ? AND invalid_at IS NULL
                    """, invalid_at_rows)
                except Exception as exc:
                    logger.debug("bulk relation invalid_at update failed: %s", exc)
                for relation in relations:
                    n1, n2 = _resolved_names.get(relation.absolute_id, ("", ""))
                    _precomp_emb = concept_embeddings.get(relation.absolute_id)
                    self._write_concept_from_relation(relation, cursor, e1_name=n1, e2_name=n2, precomputed_embedding=_precomp_emb)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        # Batch sync to VectorStore OUTSIDE lock (after successful commit)
        vec_items = []
        for r in relations:
            if r.embedding:
                vec_items.append((r.absolute_id, r.embedding if isinstance(r.embedding, bytes) else r.embedding))
        if vec_items:
            try:
                vs = self._get_vector_store()
                if vs:
                    vs.upsert_batch("relation_vectors", vec_items)
            except Exception as exc:
                logger.debug("bulk relation vector upsert failed: %s", exc)

    def get_relation_by_absolute_id(self, relation_absolute_id: str) -> Optional[Relation]:
        """根据关系行的主键 id（绝对ID）获取单条关系"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {self._RELATION_SELECT} FROM relations WHERE id = ?",
            (relation_absolute_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_relation(row)

    def get_relation_by_family_id(self, family_id: str) -> Optional[Relation]:
        """根据family_id获取最新版本的关系"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {self._RELATION_SELECT} FROM relations WHERE family_id = ? ORDER BY processed_time DESC LIMIT 1",
            (family_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_relation(row)

    def get_relation_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        """获取关系embedding向量的前N个值"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT embedding
            FROM relations
            WHERE id = ?
        """, (absolute_id,))

        row = cursor.fetchone()

        if row is None or row[0] is None:
            return None

        try:
            embedding_array = np.frombuffer(row[0], dtype=np.float32)
            return embedding_array[:num_values].tolist()
        except Exception as exc:
            logger.debug("relation embedding preview failed: %s", exc)
            return None

    def get_relation_versions(self, family_id: str,
                               include_candidates: bool = True) -> List[Relation]:
        """获取关系的所有版本。

        默认包含 dream 候选关系，因为按 family_id 查询是明确的定向查询。
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT {self._RELATION_SELECT} FROM relations WHERE family_id = ? ORDER BY processed_time DESC",
            (family_id,),
        )

        rows = cursor.fetchall()

        return self._filter_dream_candidates(
            [self._row_to_relation(row) for row in rows],
            include_candidates,
        )

    # ------------------------------------------------------------------
    # Search — BM25
    # ------------------------------------------------------------------

    def search_relations_by_bm25(self, query: str, limit: int = 20,
                                  include_candidates: bool = False) -> List[Relation]:
        """BM25 全文搜索关系。"""
        if not query:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT {self._RELATION_SELECT_R},
                   fts.rank AS bm25_score
            FROM relation_fts AS fts
            JOIN relations AS r ON r.rowid = fts.rowid
            WHERE relation_fts MATCH ?
              AND r.invalid_at IS NULL
            ORDER BY fts.rank
            LIMIT ?
        """, (query, limit))

        relations = []
        for row in cursor.fetchall():
            relations.append(self._row_to_relation(row))
        return self._filter_dream_candidates(relations, include_candidates)

    # ------------------------------------------------------------------
    # Search — embedding & text similarity
    # ------------------------------------------------------------------

    def search_relations_by_similarity(self, query_text: str,
                                       threshold: float = 0.3,
                                       max_results: int = 10,
                                       include_candidates: bool = False,
                                       query_embedding=None) -> List[Relation]:
        """
        根据embedding相似度搜索关系

        Args:
            query_text: 查询文本
            threshold: 相似度阈值
            max_results: 返回的最大关系数量
            include_candidates: 是否包含 dream 候选关系

        Returns:
            匹配的关系列表（按相似度排序）
        """
        # 获取所有关系及其embedding — choose path based on include_candidates upfront
        if include_candidates:
            relations_with_embeddings = self._get_all_relations_with_embeddings()
        else:
            relations_with_embeddings = self._get_relations_with_embeddings()

        # 使用embedding相似度（如果可用）
        if self.embedding_client and self.embedding_client.is_available():
            return self._search_relations_with_embedding(
                query_text, relations_with_embeddings, threshold, max_results,
                query_embedding=query_embedding,
            )
        else:
            # 使用文本相似度
            return self._search_relations_with_text_similarity(
                query_text, [r for r, _ in relations_with_embeddings], threshold, max_results
            )

    def _search_relations_with_embedding(self, query_text: str,
                                         relations_with_embeddings: List[tuple],
                                         threshold: float,
                                         max_results: int,
                                         query_embedding=None) -> List[Relation]:
        """使用embedding向量进行关系相似度搜索（向量化批量计算）"""
        # 编码查询文本（使用预计算的 embedding 如果可用）
        if query_embedding is None:
            query_embedding = self.embedding_client.encode(query_text)
        if query_embedding is None:
            return []

        query_embedding_array = np.array(query_embedding[0] if isinstance(query_embedding, (list, np.ndarray)) else query_embedding, dtype=np.float32)

        # Collect valid embeddings and corresponding relations
        valid_pairs = [(rel, emb) for rel, emb in relations_with_embeddings if emb is not None]
        if not valid_pairs:
            return []

        # Vectorized cosine similarity: single matmul instead of per-item loop
        _rels, _embs = zip(*valid_pairs)
        emb_matrix = np.array(_embs, dtype=np.float32)
        if emb_matrix.ndim == 1:
            emb_matrix = emb_matrix.reshape(1, -1)
        # Normalize
        q_norm = np.linalg.norm(query_embedding_array) + 1e-9
        emb_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9
        _query_normed = query_embedding_array / q_norm
        _emb_normed = emb_matrix / emb_norms
        similarities = (_emb_normed @ _query_normed).ravel()

        # Filter by threshold
        similarities = list(zip(_rels, similarities.tolist()))
        similarities = [(r, s) for r, s in similarities if s >= threshold]

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回关系列表（去重，每个family_id只保留一个，并限制最大数量）
        relations = []
        seen_ids = set()
        for relation, _ in similarities:
            if relation.family_id not in seen_ids:
                relations.append(relation)
                seen_ids.add(relation.family_id)
                if len(relations) >= max_results:
                    break

        return relations

    def _search_relations_with_text_similarity(self, query_text: str,
                                               all_relations: List[Relation],
                                               threshold: float,
                                               max_results: int) -> List[Relation]:
        """使用文本相似度进行关系搜索"""

        # 计算相似度并筛选
        query_lower = query_text.lower()
        _ql = len(query_lower)
        _qbigrams = _cached_bigram_set(query_lower)
        scored_relations = []
        for relation in all_relations:
            relation_text = relation.content.lower()
            _rl = len(relation_text)
            # Early exit: max possible ratio is min/max, skip obviously different lengths
            if _ql > 0 and _rl > 0 and abs(_ql - _rl) > max(_ql, _rl) * 0.6:
                continue
            # Jaccard bigram pre-filter: skip if bigram overlap too low
            if _qbigrams:
                _ebigrams = _cached_bigram_set(relation_text)
                if _ebigrams:
                    _jaccard = len(_qbigrams & _ebigrams) / len(_qbigrams | _ebigrams)
                    if _jaccard < threshold * 0.5:
                        continue
            similarity = difflib.SequenceMatcher(
                None,
                query_lower,
                relation_text
            ).ratio()

            if similarity >= threshold:
                scored_relations.append((relation, similarity))

        # 按相似度排序
        scored_relations.sort(key=lambda x: x[1], reverse=True)

        # 返回关系列表（去重，每个family_id只保留一个，并限制最大数量）
        relations = []
        seen_ids = set()
        for relation, _ in scored_relations:
            if relation.family_id not in seen_ids:
                relations.append(relation)
                seen_ids.add(relation.family_id)
                if len(relations) >= max_results:
                    break

        return relations

    # ------------------------------------------------------------------
    # Read operations — bulk / listing
    # ------------------------------------------------------------------

    def get_all_relations(self, limit: Optional[int] = None, offset: Optional[int] = None,
                          exclude_embedding: bool = False,
                          include_candidates: bool = False) -> List[Relation]:
        """获取所有关系的最新版本

        Args:
            limit: SQL 层限制返回条数（避免全量读取后在 Python 中截断）
            offset: SQL 层偏移量
            exclude_embedding: 是否排除 embedding 字段
            include_candidates: 是否包含 dream 候选关系
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        r1_cols = self._RELATION_SELECT_R1
        if exclude_embedding:
            r1_cols = r1_cols.replace("r1.embedding", "NULL AS r1__embedding")
        query = f"""
            SELECT {r1_cols}
            FROM relations r1
            INNER JOIN (
                SELECT family_id, MAX(processed_time) as max_time
                FROM relations
                GROUP BY family_id
            ) r2 ON r1.family_id = r2.family_id AND r1.processed_time = r2.max_time
            ORDER BY r1.processed_time DESC
        """

        if limit is not None and offset is not None and offset > 0:
            query += f" LIMIT {int(limit)} OFFSET {int(offset)}"
        elif limit is not None:
            query += f" LIMIT {int(limit)}"

        cursor.execute(query)

        rows = cursor.fetchall()

        return self._filter_dream_candidates(
            [self._row_to_relation(row) for row in rows],
            include_candidates,
        )

    def count_unique_entities(self) -> int:
        """轻量统计：返回不重复的 family_id 数量（不加载任何实体数据）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT family_id) FROM entities")
        return cursor.fetchone()[0]

    def count_unique_relations(self) -> int:
        """轻量统计：返回不重复的 family_id 数量（不加载任何关系数据）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT family_id) FROM relations")
        return cursor.fetchone()[0]

    def get_relation_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
        """批量获取多个family_id的版本数量。

        Args:
            family_ids: 关系ID列表

        Returns:
            Dict[family_id, version_count]
        """
        if not family_ids:
            return {}
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
        placeholders = ','.join(['?'] * len(canonical_ids))
        cursor.execute(f"""
            SELECT family_id, COUNT(*) as version_count
            FROM relations
            WHERE family_id IN ({placeholders})
            GROUP BY family_id
        """, canonical_ids)
        return {row[0]: row[1] for row in cursor.fetchall()}

    # ------------------------------------------------------------------
    # Read operations — by entities
    # ------------------------------------------------------------------

    def get_relations_by_entities(self, from_family_id: str, to_family_id: str,
                                  include_candidates: bool = False) -> List[Relation]:
        """根据两个实体ID获取所有关系（通过family_id查找，内部转换为绝对ID查询）

        每个 family_id 只返回最新版本（与 get_entity_relations 保持一致的去重逻辑）。
        """
        # Batch resolve + fetch absolute IDs in one pass
        resolved = self.resolve_family_ids([from_family_id, to_family_id])
        from_family_id = resolved.get(from_family_id, from_family_id)
        to_family_id = resolved.get(to_family_id, to_family_id)

        # Fetch absolute IDs for both family_ids in a single query
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, family_id FROM entities WHERE family_id IN (?, ?)",
            (from_family_id, to_family_id),
        )
        from_absolute_ids = []
        to_absolute_ids = []
        for abs_id, fid in cursor.fetchall():
            if fid == from_family_id:
                from_absolute_ids.append(abs_id)
            else:
                to_absolute_ids.append(abs_id)

        if not from_absolute_ids or not to_absolute_ids:
            return []

        # 查询关系（无向关系，考虑两个方向）
        # 每个 family_id 只返回最新版本（INNER JOIN MAX(processed_time) 去重）
        placeholders_from = ','.join(['?'] * len(from_absolute_ids))
        placeholders_to = ','.join(['?'] * len(to_absolute_ids))

        r1_cols = self._RELATION_SELECT_R1
        cursor.execute(f"""
            SELECT {r1_cols}
            FROM relations r1
            INNER JOIN (
                SELECT family_id, MAX(processed_time) as max_time
                FROM relations
                WHERE (entity1_absolute_id IN ({placeholders_from}) AND entity2_absolute_id IN ({placeholders_to}))
                   OR (entity1_absolute_id IN ({placeholders_to}) AND entity2_absolute_id IN ({placeholders_from}))
                GROUP BY family_id
            ) r2 ON r1.family_id = r2.family_id
                AND r1.processed_time = r2.max_time
                AND ((r1.entity1_absolute_id IN ({placeholders_from}) AND r1.entity2_absolute_id IN ({placeholders_to}))
                  OR (r1.entity1_absolute_id IN ({placeholders_to}) AND r1.entity2_absolute_id IN ({placeholders_from})))
            ORDER BY r1.processed_time DESC
        """, (from_absolute_ids + to_absolute_ids + to_absolute_ids + from_absolute_ids)
              + (from_absolute_ids + to_absolute_ids + to_absolute_ids + from_absolute_ids))

        rows = cursor.fetchall()

        return self._filter_dream_candidates(
            [self._row_to_relation(row) for row in rows],
            include_candidates,
        )

    def get_relations_by_entity_pairs(self, entity_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], List[Relation]]:
        """批量获取多个实体对的关系，按无向 pair 返回。

        优化：批量解析所有 family_id → absolute_id，单次关系查询，按 pair 分组。
        """
        if not entity_pairs:
            return {}

        # 去重 pair keys
        unique_pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for e1, e2 in entity_pairs:
            pair_key = (e1, e2) if e1 <= e2 else (e2, e1)
            if pair_key not in unique_pairs:
                unique_pairs[pair_key] = (e1, e2)

        # 批量解析所有 family_id → canonical
        all_raw_fids = set()
        for e1, e2 in unique_pairs.values():
            all_raw_fids.add(e1)
            all_raw_fids.add(e2)

        canonical_map: Dict[str, Optional[str]] = self.resolve_family_ids(all_raw_fids)

        # 过滤有效的 canonical pairs
        valid_canonical_fids = set()
        canonical_pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for pair_key, (e1, e2) in unique_pairs.items():
            c1 = canonical_map.get(e1)
            c2 = canonical_map.get(e2)
            if c1 and c2:
                canonical_pair = (c1, c2) if c1 <= c2 else (c2, c1)
                if canonical_pair not in canonical_pairs:
                    canonical_pairs[canonical_pair] = pair_key
                    valid_canonical_fids.add(c1)
                    valid_canonical_fids.add(c2)

        if not valid_canonical_fids:
            return {pk: [] for pk in unique_pairs}

        # 批量获取所有 canonical family_id 的 absolute_ids
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(valid_canonical_fids))
        cursor.execute(
            f"SELECT family_id, id FROM entities WHERE family_id IN ({placeholders})",
            tuple(valid_canonical_fids),
        )
        fid_to_aids: Dict[str, List[str]] = {}
        all_aids: List[str] = []
        for row in cursor.fetchall():
            fid_to_aids.setdefault(row[0], []).append(row[1])
            all_aids.append(row[1])

        if not all_aids:
            return {pk: [] for pk in unique_pairs}

        # 单次查询所有相关关系 — 使用完整的 _RELATION_SELECT 列表（含 embedding）
        _REL_COLS = "r.id, r.family_id, r.entity1_absolute_id, r.entity2_absolute_id, " \
                     "r.content, r.event_time, r.processed_time, r.episode_id, r.source_document, " \
                     "r.embedding, r.summary, r.attributes, r.confidence, r.valid_at, r.invalid_at"
        aid_placeholders = ",".join("?" * len(all_aids))
        cursor.execute(f"""
            SELECT {_REL_COLS}
            FROM relations r
            WHERE (r.entity1_absolute_id IN ({aid_placeholders})
                OR r.entity2_absolute_id IN ({aid_placeholders}))
              AND r.invalid_at IS NULL
        """, tuple(all_aids) * 2)

        all_rels = []
        for row in cursor.fetchall():
            all_rels.append(self._row_to_relation(row))

        # 按 canonical pair 分组 — O(R) 用 reverse lookup dict 替代 O(R*F) 嵌套循环
        aid_to_fid: Dict[str, str] = {}
        for fid, aids in fid_to_aids.items():
            for aid in aids:
                aid_to_fid[aid] = fid

        canonical_rels: Dict[Tuple[str, str], List[Relation]] = {cp: [] for cp in canonical_pairs}
        for rel in all_rels:
            e1_fid = aid_to_fid.get(rel.entity1_absolute_id)
            e2_fid = aid_to_fid.get(rel.entity2_absolute_id)
            if e1_fid and e2_fid:
                pair = (e1_fid, e2_fid) if e1_fid <= e2_fid else (e2_fid, e1_fid)
                if pair in canonical_rels:
                    canonical_rels[pair].append(rel)

        # 映射回原始 pair keys
        results: Dict[Tuple[str, str], List[Relation]] = {}
        for canonical_pair, original_key in canonical_pairs.items():
            results[original_key] = canonical_rels.get(canonical_pair, [])

        return results

    # ------------------------------------------------------------------
    # Read operations — entity-relation traversal
    # ------------------------------------------------------------------

    def get_entity_relations(self, entity_absolute_id: str, limit: Optional[int] = None,
                             time_point: Optional[datetime] = None,
                             include_candidates: bool = False) -> List[Relation]:
        """获取与指定实体相关的所有关系（作为起点或终点）

        Args:
            entity_absolute_id: 实体的绝对ID
            limit: 限制返回的关系数量（按时间倒序），None表示不限制
            time_point: 时间点（可选），如果提供，只返回该时间点之前或等于该时间点的关系，且每个family_id只返回最新版本
            include_candidates: 是否包含 dream 候选关系
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        r1_cols = self._RELATION_SELECT_R1
        if time_point:
            # 获取每个family_id在该时间点之前或等于该时间点的最新版本
            query = f"""
                SELECT {r1_cols}
                FROM relations r1
                INNER JOIN (
                    SELECT family_id, MAX(processed_time) as max_time
                    FROM relations
                    WHERE (entity1_absolute_id = ? OR entity2_absolute_id = ?)
                    AND event_time <= ?
                    GROUP BY family_id
                ) r2 ON r1.family_id = r2.family_id
                    AND r1.processed_time = r2.max_time
                    AND (r1.entity1_absolute_id = ? OR r1.entity2_absolute_id = ?)
                ORDER BY r1.processed_time DESC
            """
            params = (entity_absolute_id, entity_absolute_id, time_point.isoformat(), entity_absolute_id, entity_absolute_id)
        else:
            # 获取每个family_id的最新版本
            query = f"""
                SELECT {r1_cols}
                FROM relations r1
                INNER JOIN (
                    SELECT family_id, MAX(processed_time) as max_time
                    FROM relations
                    WHERE entity1_absolute_id = ? OR entity2_absolute_id = ?
                    GROUP BY family_id
                ) r2 ON r1.family_id = r2.family_id
                    AND r1.processed_time = r2.max_time
                    AND (r1.entity1_absolute_id = ? OR r1.entity2_absolute_id = ?)
                ORDER BY r1.processed_time DESC
            """
            params = (entity_absolute_id, entity_absolute_id, entity_absolute_id, entity_absolute_id)

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        cursor.execute(query, params)

        rows = cursor.fetchall()

        return self._filter_dream_candidates(
            [self._row_to_relation(row) for row in rows],
            include_candidates,
        )

    def get_entity_relations_by_family_id(self, family_id: str, limit: Optional[int] = None,
                                           time_point: Optional[datetime] = None,
                                           max_version_absolute_id: Optional[str] = None,
                                           include_candidates: bool = False) -> List[Relation]:
        """获取与指定实体相关的所有关系（通过family_id查找，包含该实体的所有版本）

        这个方法会查找该实体的所有版本（从最早版本开始）的所有关系，
        然后按family_id去重，保留每个family_id的最新版本。

        Args:
            family_id: 实体的family_id（不是absolute_id）
            limit: 限制返回的关系数量（按时间倒序），None表示不限制
            time_point: 时间点（可选），如果提供，只返回该时间点之前或等于该时间点的关系，且每个family_id只返回最新版本
            max_version_absolute_id: 最大版本absolute_id（可选），如果提供，只查询从最早版本到该版本的所有关系
        """
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        # 轻量查询：只取 id 和 processed_time，避免加载 content/embedding BLOB
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, processed_time FROM entities WHERE family_id = ? ORDER BY processed_time",
            (family_id,),
        )
        version_rows = cursor.fetchall()
        if not version_rows:
            return []

        # 如果指定了max_version_absolute_id，只取到该版本为止的所有版本
        if max_version_absolute_id:
            # 按时间排序，找到max_version_absolute_id对应的版本
            max_version = None
            for vid, pt in version_rows:
                if vid == max_version_absolute_id:
                    max_version = (vid, pt)
                    break

            if max_version:
                t_max = self._normalize_datetime_for_compare(
                    self._safe_parse_datetime(max_version[1])
                )
                # 只取到该版本（包含）为止的所有版本
                entity_absolute_ids = [
                    vid for vid, pt in version_rows
                    if self._normalize_datetime_for_compare(
                        self._safe_parse_datetime(pt)
                    ) <= t_max
                ]
                # 同时设置time_point为该版本的时间点
                if not time_point:
                    time_point = self._safe_parse_datetime(max_version[1])
                else:
                    # 如果已经设置了time_point，取较小值（避免 naive/aware 无法比较）
                    nt = self._normalize_datetime_for_compare(time_point)
                    if nt <= t_max:
                        pass  # 保持 time_point
                    else:
                        time_point = self._safe_parse_datetime(max_version[1])
            else:
                # 如果找不到指定的版本，使用所有版本
                entity_absolute_ids = [vid for vid, _ in version_rows]
        else:
            # 收集所有版本的absolute_id
            entity_absolute_ids = [vid for vid, _ in version_rows]

        conn = self._get_conn()
        cursor = conn.cursor()

        # 构建查询：查找所有版本的关系，按family_id去重
        placeholders = ','.join(['?'] * len(entity_absolute_ids))

        if time_point:
            # 获取每个family_id在该时间点之前或等于该时间点的最新版本
            r1_cols = self._RELATION_SELECT_R1
            query = f"""
                SELECT {r1_cols}
                FROM relations r1
                INNER JOIN (
                    SELECT family_id, MAX(processed_time) as max_time
                    FROM relations
                    WHERE (entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders}))
                    AND event_time <= ?
                    GROUP BY family_id
                ) r2 ON r1.family_id = r2.family_id
                    AND r1.processed_time = r2.max_time
                    AND (r1.entity1_absolute_id IN ({placeholders}) OR r1.entity2_absolute_id IN ({placeholders}))
                ORDER BY r1.processed_time DESC
            """
            params = tuple(entity_absolute_ids * 2 + [time_point.isoformat()] + entity_absolute_ids * 2)
        else:
            # 获取每个family_id的最新版本
            r1_cols = self._RELATION_SELECT_R1
            query = f"""
            SELECT {r1_cols}
            FROM relations r1
            INNER JOIN (
                SELECT family_id, MAX(processed_time) as max_time
                FROM relations
                WHERE entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})
                GROUP BY family_id
            ) r2 ON r1.family_id = r2.family_id
                AND r1.processed_time = r2.max_time
                AND (r1.entity1_absolute_id IN ({placeholders}) OR r1.entity2_absolute_id IN ({placeholders}))
            ORDER BY r1.processed_time DESC
            """
            params = tuple(entity_absolute_ids * 4)

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        cursor.execute(query, params)

        rows = cursor.fetchall()

        return self._filter_dream_candidates(
            [self._row_to_relation(row) for row in rows],
            include_candidates,
        )

    def get_entity_relations_timeline(self, family_id: str, version_abs_ids: List[str],
                                       include_candidates: bool = False) -> List[Dict]:
        """批量获取实体在各版本时间点的关系（消除 N+1 查询）。

        与 Neo4j 后端保持一致的接口：获取该实体所有版本关联的关系，
        按每个版本的 processed_time 过滤，只返回在该版本时间点之前出现的关系。
        """
        family_id = self.resolve_family_id(family_id)
        if not family_id or not version_abs_ids:
            return []

        conn = self._get_conn()
        cursor = conn.cursor()

        # 获取各版本的 processed_time
        placeholders = ",".join("?" * len(version_abs_ids))
        cursor.execute(
            f"SELECT id, processed_time FROM entities WHERE id IN ({placeholders})",
            tuple(version_abs_ids),
        )
        version_times = []
        for row in cursor.fetchall():
            pt = row[1]
            if pt:
                pt = datetime.fromisoformat(pt) if isinstance(pt, str) else pt
            version_times.append((row[0], pt))
        if not version_times:
            return []

        # 获取该实体的所有 absolute_id
        all_abs_ids = self.get_entity_absolute_ids_up_to_version(
            family_id, max(version_abs_ids)
        )
        if not all_abs_ids:
            all_abs_ids = version_abs_ids[:]

        # 一次查询获取所有相关关系（按 family_id 去重，保留最新版本）
        placeholders = ",".join("?" * len(all_abs_ids))
        dream_filter = "" if include_candidates else _DREAM_FILTER_R
        dream_filter_sub = "" if include_candidates else _DREAM_FILTER_SUB
        cursor.execute(
            f"""
            SELECT r.id, r.family_id, r.content, r.event_time, r.processed_time
            FROM relations r
            INNER JOIN (
                SELECT family_id, MAX(processed_time) AS max_pt
                FROM relations
                WHERE (entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders}))
                  AND invalid_at IS NULL{dream_filter_sub}
                GROUP BY family_id
            ) latest ON r.family_id = latest.family_id AND r.processed_time = latest.max_pt
            WHERE (r.entity1_absolute_id IN ({placeholders}) OR r.entity2_absolute_id IN ({placeholders}))
              AND r.invalid_at IS NULL{dream_filter}
            ORDER BY r.processed_time ASC
            """,
            tuple(all_abs_ids) * 4,
        )

        timeline = []
        seen = set()
        for row in cursor.fetchall():
            rel_uuid = row[0]
            if rel_uuid in seen:
                continue
            rel_pt = row[4]
            if rel_pt:
                rel_pt = datetime.fromisoformat(rel_pt) if isinstance(rel_pt, str) else rel_pt
            # 检查关系是否出现在某个版本之前
            for _, v_pt in version_times:
                if rel_pt and v_pt and rel_pt <= v_pt:
                    seen.add(rel_uuid)
                    timeline.append({
                        "family_id": row[1],
                        "content": row[2],
                        "event_time": row[3],
                        "absolute_id": rel_uuid,
                    })
                    break
        return timeline

    def get_relations_by_entity_absolute_ids(self, entity_absolute_ids: List[str],
                                              limit: Optional[int] = None,
                                              include_candidates: bool = False) -> List[Relation]:
        """获取与指定实体版本列表直接关联的所有关系（通过entity_absolute_id直接匹配）

        这个方法根据关系边中的 entity1_absolute_id 或 entity2_absolute_id 直接匹配，
        不使用时间过滤，只返回直接引用这些实体版本的关系边。
        按 family_id 去重，每个 family_id 只返回一条记录（保留最新的）。

        Args:
            entity_absolute_ids: 实体版本的absolute_id列表
            limit: 限制返回的关系数量，None表示不限制
            include_candidates: 是否包含 dream 候选关系

        Returns:
            直接与这些实体版本关联的关系列表
        """
        if not entity_absolute_ids:
            return []

        conn = self._get_conn()
        cursor = conn.cursor()

        # 构建查询：查找直接引用这些 entity_absolute_id 的关系边
        placeholders = ','.join(['?'] * len(entity_absolute_ids))
        query = f"""
            SELECT {self._RELATION_SELECT}
            FROM relations
            WHERE entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})
            ORDER BY processed_time DESC
        """

        params = tuple(entity_absolute_ids * 2)
        cursor.execute(query, params)

        rows = cursor.fetchall()

        # 按 family_id 去重，保留第一个（最新的）
        seen_family_ids = set()
        result = []
        for row in rows:
            family_id_val = row[1]
            if family_id_val not in seen_family_ids:
                seen_family_ids.add(family_id_val)
                result.append(self._row_to_relation(row))
                if limit is not None and len(result) >= limit:
                    break

        return self._filter_dream_candidates(result, include_candidates)

    def get_entity_absolute_ids_up_to_version(self, family_id: str, max_absolute_id: str) -> List[str]:
        """获取指定实体从最早版本到指定版本的所有 absolute_id 列表

        Args:
            family_id: 实体的 family_id
            max_absolute_id: 最大版本的 absolute_id（包含）

        Returns:
            从最早版本到指定版本的所有 absolute_id 列表（按时间顺序）
        """
        # Targeted SQL: only fetch id + processed_time, avoid loading full content blobs
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, processed_time FROM entities WHERE family_id = ? ORDER BY processed_time ASC",
            (family_id,),
        )
        rows = cursor.fetchall()

        if not rows:
            return []

        # Collect ids up to and including max_absolute_id
        result = []
        found = False
        for row in rows:
            result.append(row[0])
            if row[0] == max_absolute_id:
                found = True
                break

        if not found:
            return []

        return result

    # ------------------------------------------------------------------
    # Read operations — embedding cache
    # ------------------------------------------------------------------

    def _get_relations_with_embeddings(self) -> List[tuple]:
        """
        获取所有关系的最新版本及其embedding（带短 TTL 缓存）。
        Filters out dream candidates.

        Returns:
            List of (Relation, embedding_array) tuples, embedding_array为None表示没有embedding
        """
        now = time.time()
        with self._emb_cache_lock:
            if self._relation_emb_cache is not None and (now - self._relation_emb_cache_ts) < self._emb_cache_ttl:
                return self._relation_emb_cache

        # Delegate to _get_all_relations_with_embeddings, then filter
        all_results = self._get_all_relations_with_embeddings()
        results = [(r, e) for r, e in all_results if not self._is_dream_candidate(r)]

        with self._emb_cache_lock:
            self._relation_emb_cache = results
            self._relation_emb_cache_ts = time.time()
        return results

    def _get_all_relations_with_embeddings(self) -> List[tuple]:
        """Like _get_relations_with_embeddings but includes dream candidates (with TTL cache)."""
        now = time.time()
        with self._emb_cache_lock:
            if self._relation_emb_cache_all is not None and (now - self._relation_emb_cache_all_ts) < self._emb_cache_ttl:
                return self._relation_emb_cache_all

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT {self._RELATION_SELECT}
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY family_id ORDER BY processed_time DESC) AS rn
                FROM relations
            )
            WHERE rn = 1
        """)
        results = []
        for row in cursor.fetchall():
            embedding_array = None
            if len(row) > 9 and row[9] is not None:
                try:
                    embedding_array = np.frombuffer(row[9], dtype=np.float32)
                except (ValueError, TypeError):
                    embedding_array = None
            relation = self._row_to_relation(row)
            results.append((relation, embedding_array))

        with self._emb_cache_lock:
            self._relation_emb_cache_all = results
            self._relation_emb_cache_all_ts = time.time()
        return results

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    def delete_relation_by_id(self, family_id: str) -> int:
        """删除关系的所有版本。返回删除的行数。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM relations WHERE family_id = ?", (family_id,))
            count = cursor.rowcount
            # 清理 FTS 表
            try:
                cursor.execute("DELETE FROM relation_fts WHERE family_id = ?", (family_id,))
            except Exception as exc:
                logger.warning("FTS delete failed: %s", exc)
            # Phase 2: cleanup concepts
            self._delete_concepts_by_family(family_id, cursor)
            conn.commit()
            return count

    def delete_relation_all_versions(self, family_id: str) -> int:
        """删除关系的所有版本。返回删除的行数。"""
        return self.delete_relation_by_id(family_id)

    def batch_delete_relations(self, family_ids: List[str]) -> int:
        """批量删除关系 — 单次事务，替代 N 次单独删除。含 concepts 清理。"""
        if not family_ids:
            return 0
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(family_ids))
            # Batch cleanup concepts table
            self._batch_delete_concepts_by_families(family_ids, cursor)
            cursor.execute(f"DELETE FROM relations WHERE family_id IN ({placeholders})", tuple(family_ids))
            count = cursor.rowcount
            try:
                cursor.execute(f"DELETE FROM relation_fts WHERE family_id IN ({placeholders})", tuple(family_ids))
            except Exception as exc:
                logger.warning("FTS delete failed: %s", exc)
            conn.commit()
            return count

    # ------------------------------------------------------------------
    # Update operations
    # ------------------------------------------------------------------

    def update_relation_confidence(self, family_id: str, confidence: float):
        """更新关系最新版本的置信度。"""
        confidence = max(0.0, min(1.0, confidence))
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE relations SET confidence = ?
                WHERE id = (
                    SELECT id FROM relations
                    WHERE family_id = ?
                    ORDER BY processed_time DESC LIMIT 1
                )
            """, (confidence, family_id))
            # Sync to concepts table
            cursor.execute(
                "SELECT id FROM relations WHERE family_id = ? ORDER BY processed_time DESC LIMIT 1",
                (family_id,),
            )
            row = cursor.fetchone()
            if row:
                self._sync_concept_relation_fields(row[0], {"confidence": confidence}, cursor)
            conn.commit()

    # ------------------------------------------------------------------
    # Read operations — batch lookups
    # ------------------------------------------------------------------

    def get_relations_by_family_ids(self, family_ids: List[str], limit: Optional[int] = None,
                                     include_candidates: bool = False) -> List[Relation]:
        """获取与指定实体 ID 列表相关的所有关系。"""
        if not family_ids:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(family_ids))
        cursor.execute(f"""
            SELECT family_id, MAX(processed_time), id
            FROM entities
            WHERE family_id IN ({placeholders})
            GROUP BY family_id
        """, family_ids)
        abs_id_map = {row[0]: row[2] for row in cursor.fetchall()}
        abs_ids = list(abs_id_map.values())
        if not abs_ids:
            return []
        abs_placeholders = ",".join("?" * len(abs_ids))
        query = f"""
            SELECT {self._RELATION_SELECT}
            FROM relations
            WHERE entity1_absolute_id IN ({abs_placeholders})
               OR entity2_absolute_id IN ({abs_placeholders})
            ORDER BY processed_time DESC
        """
        params = abs_ids + abs_ids
        if limit:
            query += " LIMIT ?"
            params.append(int(limit))
        cursor.execute(query, params)
        return self._filter_dream_candidates(
            [self._row_to_relation(row) for row in cursor.fetchall()],
            include_candidates,
        )

    def batch_get_entity_degrees(self, family_ids: List[str]) -> Dict[str, int]:
        """批量获取实体度数 — 单次查询替代 N 次 get_entity_degree。"""
        if not family_ids:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(family_ids))
        cursor.execute(
            f"SELECT e.family_id, COUNT(r.id) "
            f"FROM entities e "
            f"LEFT JOIN relations r ON ("
            f"(r.entity1_absolute_id = e.id OR r.entity2_absolute_id = e.id) "
            f"AND r.invalid_at IS NULL) "
            f"WHERE e.family_id IN ({placeholders}) "
            f"GROUP BY e.family_id",
            tuple(family_ids),
        )
        result = {row[0]: row[1] for row in cursor.fetchall()}
        # 补零：未出现在结果中的 family_id 度数为 0
        for fid in family_ids:
            result.setdefault(fid, 0)
        return result

    def batch_get_entity_profiles(self, family_ids: List[str]) -> List[Dict[str, Any]]:
        """批量获取实体档案（entity + relations + version_count），消除 N+1。

        替代对每个 family_id 分别调用 get_entity_by_family_id +
        get_entity_relations_by_family_id + get_entity_version_count 的 N+1 模式。

        Returns:
            [{"family_id", "entity", "relations", "version_count"}, ...]
        """
        if not family_ids:
            return []

        # Step 1: 解析 canonical family_ids（批量 resolve）
        canonical_map: Dict[str, str] = {}  # original -> canonical
        canonical_set: List[str] = []
        _seen: set = set()
        batch_resolved = self.resolve_family_ids(family_ids) or {}
        for fid in family_ids:
            canonical = batch_resolved.get(fid) or fid
            canonical_map[fid] = canonical
            if canonical not in _seen:
                _seen.add(canonical)
                canonical_set.append(canonical)

        if not canonical_set:
            return [{"family_id": fid, "entity": None, "relations": [], "version_count": 0} for fid in family_ids]

        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(canonical_set))

        # Step 2: 批量获取最新实体（窗口函数取每组最新一条）
        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY family_id ORDER BY processed_time DESC) AS rn
                FROM entities WHERE family_id IN ({placeholders})
            ) WHERE rn = 1
        """, tuple(canonical_set))

        entity_map: Dict[str, Entity] = {}
        for row in cursor.fetchall():
            fid = row[1]
            entity_map[fid] = self._row_to_entity(row)

        # Step 3: 批量获取版本数
        version_counts = self.get_entity_version_counts(canonical_set)

        # Step 4: 批量获取所有相关 absolute_ids，再批量查关系
        cursor.execute(f"""
            SELECT family_id, id FROM entities
            WHERE family_id IN ({placeholders}) AND invalid_at IS NULL
        """, tuple(canonical_set))

        fid_to_aids: Dict[str, List[str]] = {}
        all_aids: List[str] = []
        for row in cursor.fetchall():
            fid_to_aids.setdefault(row[0], []).append(row[1])
            all_aids.append(row[1])

        relations_map: Dict[str, List[Relation]] = {fid: [] for fid in canonical_set}
        if all_aids:
            aid_placeholders = ",".join("?" * len(all_aids))
            cursor.execute(f"""
                SELECT {self._RELATION_SELECT}
                FROM relations
                WHERE (entity1_absolute_id IN ({aid_placeholders})
                    OR entity2_absolute_id IN ({aid_placeholders}))
                  AND invalid_at IS NULL
            """, tuple(all_aids) * 2)

            all_rels = [self._row_to_relation(row) for row in cursor.fetchall()]

            # Filter out dream candidates from entity profiles
            all_rels = [r for r in all_rels if not self._is_dream_candidate(r)]

            # 分配关系到对应的 family_id — O(R) 用 reverse lookup 替代 O(R*F)
            aid_to_fid: Dict[str, str] = {}
            for fid, aids in fid_to_aids.items():
                for aid in aids:
                    aid_to_fid[aid] = fid

            for rel in all_rels:
                fid1 = aid_to_fid.get(rel.entity1_absolute_id)
                fid2 = aid_to_fid.get(rel.entity2_absolute_id)
                if fid1:
                    relations_map[fid1].append(rel)
                if fid2 and fid2 != fid1:
                    relations_map[fid2].append(rel)

        # Step 5: 组装结果
        results = []
        seen_fids = set()
        for fid in family_ids:
            canonical = canonical_map.get(fid, fid)
            if canonical in seen_fids:
                results.append({"family_id": fid, "entity": None, "relations": [], "version_count": 0})
                continue
            seen_fids.add(canonical)
            entity = entity_map.get(canonical)
            if entity:
                results.append({
                    "family_id": canonical,
                    "entity": entity,
                    "relations": relations_map.get(canonical, []),
                    "version_count": version_counts.get(canonical, 1),
                })
            else:
                results.append({"family_id": canonical, "entity": None, "relations": [], "version_count": 0})

        return results

    # ------------------------------------------------------------------
    # Invalidate / soft-delete
    # ------------------------------------------------------------------

    def invalidate_relation(self, family_id: str, reason: str = "") -> int:
        """标记关系为失效（不删除数据，保留历史记录）"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            now_iso = datetime.now(timezone.utc).isoformat()
            cursor.execute("""
                UPDATE relations SET invalid_at = ?
                WHERE family_id = ? AND invalid_at IS NULL
            """, (now_iso, family_id))
            # 同步更新 concepts 表
            try:
                cursor.execute("""
                    UPDATE concepts SET invalid_at = ?
                    WHERE family_id = ? AND role = 'relation' AND invalid_at IS NULL
                """, (now_iso, family_id))
            except Exception as exc:
                logger.debug("concept invalidate sync failed: %s", exc)
            conn.commit()
            return cursor.rowcount

    def get_invalidated_relations(self, limit: int = 100) -> List[Relation]:
        """列出所有已失效的关系"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT {self._RELATION_SELECT}
            FROM relations
            WHERE invalid_at IS NOT NULL
            ORDER BY invalid_at DESC
            LIMIT ?
        """, (limit,))
        return [self._row_to_relation(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Update / delete by absolute_id
    # ------------------------------------------------------------------

    def update_relation_by_absolute_id(self, absolute_id: str, **fields) -> Optional[Relation]:
        """根据绝对ID更新关系字段（content, summary, attributes, confidence）。

        当 content 变更时，自动重算 embedding（含两端实体名称）并同步到 VectorStore。
        Uses single multi-column UPDATE instead of per-field loop.
        """
        allowed = {"content", "summary", "attributes", "confidence"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_relation_by_absolute_id(absolute_id)

        # Pre-compute embedding OUTSIDE lock (ML inference is slow)
        _precomputed_emb_blob = None
        needs_emb_update = "content" in updates
        if needs_emb_update and self.embedding_client and self.embedding_client.is_available():
            current = self.get_relation_by_absolute_id(absolute_id)
            if current:
                _precomputed_emb_blob = self._compute_relation_embedding(current)

        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            # Single multi-column UPDATE
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            cursor.execute(
                f"UPDATE relations SET {set_clause} WHERE id = ?",
                list(updates.values()) + [absolute_id],
            )

            # Write pre-computed embedding
            if _precomputed_emb_blob is not None:
                cursor.execute(
                    "UPDATE relations SET embedding = ? WHERE id = ?",
                    (_precomputed_emb_blob, absolute_id),
                )

            # Sync to concepts table
            self._sync_concept_relation_fields(absolute_id, updates, cursor)
            conn.commit()

        # Vector store I/O OUTSIDE lock
        if _precomputed_emb_blob is not None:
            self._vector_store_upsert_relation(absolute_id, _precomputed_emb_blob)
            self._invalidate_emb_cache()

        return self.get_relation_by_absolute_id(absolute_id)

    def delete_relation_by_absolute_id(self, absolute_id: str) -> bool:
        """根据绝对ID删除关系，同时清理 FTS 索引。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            # Check existence
            cursor.execute(
                "SELECT 1 FROM relations WHERE id = ?",
                (absolute_id,),
            )
            if cursor.fetchone() is None:
                return False
            # Delete FTS entry (using integer rowid)
            cursor.execute(
                "DELETE FROM relation_fts WHERE rowid = (SELECT rowid FROM relations WHERE id = ?)",
                (absolute_id,),
            )
            # Delete relation version
            cursor.execute(
                "DELETE FROM relations WHERE id = ?",
                (absolute_id,),
            )
            affected = cursor.rowcount
            # Phase 2: cleanup concepts
            self._delete_concept_by_id(absolute_id, cursor)
            conn.commit()
            return affected > 0

    def batch_delete_relation_versions_by_absolute_ids(self, absolute_ids: List[str]) -> int:
        """批量删除指定关系版本（带 FTS 和 concepts 清理），返回成功删除的数量。"""
        if not absolute_ids:
            return 0
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(absolute_ids))
            # Batch cleanup concepts table (replaces N×3 SQL calls with 3)
            self._batch_delete_concepts_by_ids(absolute_ids, cursor)
            # Delete FTS entries (using integer rowids)
            cursor.execute(
                f"DELETE FROM relation_fts WHERE rowid IN (SELECT rowid FROM relations WHERE id IN ({placeholders}))",
                list(absolute_ids),
            )
            # Delete relation versions
            cursor.execute(
                f"DELETE FROM relations WHERE id IN ({placeholders})",
                list(absolute_ids),
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted

    # ------------------------------------------------------------------
    # Reference lookups
    # ------------------------------------------------------------------

    def get_relations_referencing_absolute_id(self, absolute_id: str) -> List[Relation]:
        """获取所有引用指定实体绝对ID的关系。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {self._RELATION_SELECT} FROM relations WHERE entity1_absolute_id = ? OR entity2_absolute_id = ?",
            (absolute_id, absolute_id),
        )
        return [self._row_to_relation(row) for row in cursor.fetchall()]

    def batch_get_relations_referencing_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, List[Relation]]:
        """批量获取引用指定实体绝对ID的关系（消除 N+1 查询）。"""
        if not absolute_ids:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(absolute_ids))
        params = list(absolute_ids) + list(absolute_ids)
        cursor.execute(f"""
            SELECT {self._RELATION_SELECT}
            FROM relations
            WHERE entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})
        """, params)
        result_map: Dict[str, List[Relation]] = {aid: [] for aid in absolute_ids}
        for row in cursor.fetchall():
            rel = self._row_to_relation(row)
            if rel.entity1_absolute_id in result_map:
                result_map[rel.entity1_absolute_id].append(rel)
            if rel.entity2_absolute_id in result_map:
                result_map[rel.entity2_absolute_id].append(rel)
        return result_map

    # ------------------------------------------------------------------
    # Redirect
    # ------------------------------------------------------------------

    def redirect_relation(self, family_id: str, side: str, new_family_id: str) -> int:
        """将指定 family 下所有关系的某一端重定向到新 family 的最新实体。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            # Resolve new_family_id through redirect chain
            resolved_family = self._resolve_family_id_with_cursor(cursor, new_family_id)
            # Get latest entity absolute_id for the new family
            cursor.execute(
                "SELECT id FROM entities WHERE family_id = ? ORDER BY processed_time DESC LIMIT 1",
                (resolved_family,),
            )
            entity_row = cursor.fetchone()
            if entity_row is None:
                return 0
            new_absolute_id = entity_row[0]
            # Update the specified side
            if side == "entity1":
                cursor.execute(
                    "UPDATE relations SET entity1_absolute_id = ? WHERE family_id = ?",
                    (new_absolute_id, family_id),
                )
            elif side == "entity2":
                cursor.execute(
                    "UPDATE relations SET entity2_absolute_id = ? WHERE family_id = ?",
                    (new_absolute_id, family_id),
                )
            else:
                return 0
            affected = cursor.rowcount
            # Sync connects field in concepts table (batch)
            if affected > 0:
                try:
                    cursor.execute(
                        "SELECT id, entity1_absolute_id, entity2_absolute_id FROM relations WHERE family_id = ?",
                        (family_id,),
                    )
                    rows = cursor.fetchall()
                    if rows:
                        pairs = [
                            (json.dumps([row[1], row[2]]), row[0])
                            for row in rows
                        ]
                        cursor.executemany(
                            "UPDATE concepts SET connects = ? WHERE id = ? AND role = 'relation'",
                            pairs,
                        )
                except Exception as exc:
                    logger.debug("concept connects sync failed in redirect: %s", exc)
            conn.commit()
            return affected

    def count_entity_relations_by_family_ids(self, family_ids: List[str]) -> Dict[str, int]:
        """Batch count relations for multiple entity family_ids.

        Returns {family_id: relation_count} for each family_id.
        """
        if not family_ids:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()

        # Get all entity versions for these family_ids
        placeholders = ",".join("?" * len(family_ids))
        cursor.execute(
            f"SELECT id, family_id FROM entities WHERE family_id IN ({placeholders})",
            family_ids,
        )
        fid_to_absids: Dict[str, List[str]] = {}
        for row in cursor.fetchall():
            fid_to_absids.setdefault(row[1], []).append(row[0])

        if not fid_to_absids:
            return {fid: 0 for fid in family_ids}

        # Count distinct relation family_ids connected to these entity versions
        all_absids = [aid for aids in fid_to_absids.values() for aid in aids]
        aid_placeholders = ",".join("?" * len(all_absids))
        cursor.execute(
            f"SELECT entity1_absolute_id, entity2_absolute_id, family_id FROM relations "
            f"WHERE entity1_absolute_id IN ({aid_placeholders}) "
            f"OR entity2_absolute_id IN ({aid_placeholders})",
            all_absids + all_absids,
        )

        # Build abs_id -> owning family_id map
        abs_to_fid = {}
        for fid, aids in fid_to_absids.items():
            for aid in aids:
                abs_to_fid[aid] = fid

        # Count distinct relation family_ids per entity family_id
        result: Dict[str, int] = {fid: 0 for fid in family_ids}
        seen = set()  # (entity_fid, relation_fid) to deduplicate
        for row in cursor.fetchall():
            e1_abs, e2_abs, rel_fid = row[0], row[1], row[2]
            for abs_id in (e1_abs, e2_abs):
                owner_fid = abs_to_fid.get(abs_id)
                if owner_fid and (owner_fid, rel_fid) not in seen:
                    seen.add((owner_fid, rel_fid))
                    result[owner_fid] = result.get(owner_fid, 0) + 1

        return result

    def adjust_confidence_on_corroboration_batch(self, family_ids: List[str],
                                                  source_type: str = "relation",
                                                  is_dream: bool = False):
        """Batch version of adjust_confidence_on_corroboration for multiple family_ids.

        Resolves all latest (id, confidence) pairs in one query, then batch-updates.
        """
        if not family_ids:
            return
        table = self._validate_table_name(source_type) if hasattr(self, '_validate_table_name') else source_type
        sync_fn = self._sync_concept_relation_fields if source_type == "relation" else self._sync_concept_entity_fields
        delta = 0.025 if is_dream else 0.05
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            # Resolve all latest (id, confidence) in one query
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

            update_data = []
            concept_syncs = []
            for abs_id, current in rows:
                if current is None:
                    continue
                new_conf = min(1.0, current + delta)
                update_data.append((new_conf, abs_id))
                concept_syncs.append((new_conf, abs_id))
            if update_data:
                cursor.executemany(
                    f"UPDATE {table} SET confidence = ? WHERE id = ?",
                    update_data,
                )
                cursor.executemany(
                    "UPDATE concepts SET confidence = ? WHERE id = ? AND role = 'relation'",
                    concept_syncs,
                )
            conn.commit()
