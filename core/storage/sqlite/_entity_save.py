"""Entity save mixin for SQLiteGraphStorageManager — embedding computation,
cache management, and entity persistence (single + bulk)."""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from ...models import Entity
from ...perf import _perf_timer
from .helpers import ENTITY_COLUMNS, _encode_and_normalize, _fmt_dt, _row_to_entity

logger = logging.getLogger(__name__)

_EMB_CONTENT_MAX = 512


class _EntitySaveMixin:
    """Entity embedding computation, cache helpers, and save methods."""

    def _compute_entity_embedding(self, entity: Entity) -> Optional[tuple]:
        content = entity.content or ""
        if len(content) > _EMB_CONTENT_MAX:
            content = content[:_EMB_CONTENT_MAX]
        text = f"# {entity.name}\n{content}"
        return _encode_and_normalize(self.embedding_client, text)

    def _bulk_compute_entity_embeddings(self, entities: List[Entity]) -> Dict[str, bytes]:
        if not self.embedding_client or not self.embedding_client.is_available():
            return {}
        texts = []
        uuids = []
        for e in entities:
            content = e.content or ""
            if len(content) > _EMB_CONTENT_MAX:
                content = content[:_EMB_CONTENT_MAX]
            texts.append(f"# {e.name}\n{content}")
            uuids.append(e.absolute_id)
        try:
            embeddings = self.embedding_client.encode(texts)
        except Exception:
            return {}
        if embeddings is None:
            return {}
        result = {}
        for idx, uuid in enumerate(uuids):
            try:
                emb = np.array(embeddings[idx], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                result[uuid] = emb.tobytes()
            except Exception:
                pass
        return result

    # --- Cache helpers ---

    def _update_entity_emb_cache(self, entity: Entity, emb_array: Optional[np.ndarray]):
        if self._entity_emb_cache is None:
            return
        if self._entity_emb_fid_idx is None:
            self._entity_emb_fid_idx = {e.family_id: i for i, (e, _) in enumerate(self._entity_emb_cache)}
        idx = self._entity_emb_fid_idx.get(entity.family_id)
        if idx is not None:
            self._entity_emb_cache[idx] = (entity, emb_array)
        else:
            self._entity_emb_cache.append((entity, emb_array))
            self._entity_emb_fid_idx[entity.family_id] = len(self._entity_emb_cache) - 1

    def _update_entity_emb_cache_batch(self, items: List[tuple]):
        if self._entity_emb_cache is None or not items:
            return
        if self._entity_emb_fid_idx is not None:
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
        self._cache.invalidate_keys([
            f"entity:by_fid:{family_id}",
            f"resolve:{family_id}",
        ])
        # Invalidate all absolute-id caches for this family (version chain)
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?",
                (family_id, self._graph_id),
            ).fetchall()
        finally:
            conn.rollback()
        if rows:
            self._cache.invalidate_keys([f"entity:by_abs:{r[0]}" for r in rows])

    def _invalidate_entity_cache_bulk(self):
        self._cache.invalidate("entity:")
        self._cache.invalidate("resolve:")
        self._cache.invalidate("sim_search:")
        self.invalidate_entity_remap_cache()

    def _get_entities_with_embeddings(self) -> List[tuple]:
        now = time.time()
        if self._entity_emb_cache is not None and (now - self._entity_emb_cache_ts) < self._emb_cache_ttl:
            return self._entity_emb_cache
        with _perf_timer("_get_entities_with_embeddings"):
            result = self._get_entities_with_embeddings_impl()
        self._entity_emb_cache = result
        self._entity_emb_fid_idx = None
        self._entity_emb_cache_ts = time.time()
        # Build HNSW index alongside cache
        if result and self._vector_dim > 0:
            self._entity_hnsw, self._entity_hnsw_items = self._build_hnsw(result, self._vector_dim)
        else:
            self._entity_hnsw = None
            self._entity_hnsw_items = None
        return result

    def _get_entities_with_embeddings_impl(self) -> List[tuple]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT e.* FROM entity e "
                "INNER JOIN ("
                "  SELECT family_id, MAX(version_seq) AS max_vs FROM entity "
                "  WHERE graph_id = ? GROUP BY family_id"
                ") latest ON e.family_id = latest.family_id AND e.version_seq = latest.max_vs "
                "WHERE e.graph_id = ? ORDER BY e.processed_time DESC LIMIT ?",
                (self._graph_id, self._graph_id, self._emb_cache_max_size),
            ).fetchall()
        finally:
            conn.rollback()
        entities = []
        for row in rows:
            entity = _row_to_entity(dict(row))
            emb_array = np.frombuffer(entity.embedding, dtype=np.float32) if entity.embedding else None
            entities.append((entity, emb_array))
        return entities

    # --- Save ---

    def save_entity(self, entity: Entity, _precomputed_embedding=None):
        with _perf_timer("save_entity"):
            _emb_array = None
            if _precomputed_embedding is not None:
                embedding_blob = _precomputed_embedding
            else:
                _emb_result = self._compute_entity_embedding(entity)
                if _emb_result is not None:
                    embedding_blob, _emb_array = _emb_result
                else:
                    embedding_blob = None
            entity.embedding = embedding_blob
            entity.processed_time = datetime.now()
            valid_at = _fmt_dt(entity.valid_at or entity.event_time)
            attrs = json.dumps(entity.attributes, ensure_ascii=False) if isinstance(entity.attributes, (dict, list)) else entity.attributes
            with self._write_lock:
                conn = self._connect()
                try:
                    # Compute version_seq: max existing + 1
                    if entity.version_seq <= 1:
                        row = conn.execute(
                            "SELECT MAX(version_seq) FROM entity WHERE family_id = ? AND graph_id = ?",
                            (entity.family_id, self._graph_id),
                        ).fetchone()
                        entity.version_seq = (row[0] or 0) + 1
                    conn.execute(
                        f"INSERT OR REPLACE INTO entity ({', '.join(ENTITY_COLUMNS)}) VALUES ({', '.join('?' * len(ENTITY_COLUMNS))})",
                        (
                            entity.absolute_id, entity.family_id, self._graph_id,
                            entity.name, entity.content, entity.summary,
                            attrs, entity.confidence,
                            getattr(entity, "content_format", "plain"),
                            entity.community_id,
                            entity.version_seq,
                            valid_at,
                            _fmt_dt(entity.event_time), _fmt_dt(entity.processed_time),
                            entity.episode_id, entity.source_document,
                            embedding_blob,
                        ),
                    )
                    # Update FTS
                    conn.execute("DELETE FROM entity_fts WHERE rowid = (SELECT rowid FROM entity WHERE uuid = ?)",
                                 (entity.absolute_id,))
                    conn.execute(
                        "INSERT INTO entity_fts (rowid, name, content, graph_id) VALUES ((SELECT rowid FROM entity WHERE uuid = ?), ?, ?, ?)",
                        (entity.absolute_id, entity.name, entity.content, self._graph_id),
                    )
                    conn.commit()
                finally:
                    conn.rollback()
            # Cache update
            if embedding_blob:
                emb_array = _emb_array if _emb_array is not None else np.frombuffer(embedding_blob, dtype=np.float32)
                self._update_entity_emb_cache(entity, emb_array)
            else:
                self._update_entity_emb_cache(entity, None)
            self._cache_entity_name(entity.absolute_id, entity.name)
        self._invalidate_entity_cache(entity.family_id)
        self._cache.invalidate("sim_search:")

    def bulk_save_entities(self, entities: List[Entity]):
        if not entities:
            return
        _now = datetime.now()
        # Compute embeddings synchronously before writing
        emb_map = self._bulk_compute_entity_embeddings(entities)
        rows = []
        cache_items = []
        with self._write_lock:
            conn = self._connect()
            try:
                # Batch compute version_seq for all family_ids
                fid_map = {}
                for entity in entities:
                    fid_map.setdefault(entity.family_id, []).append(entity)
                for fid, fid_entities in fid_map.items():
                    row = conn.execute(
                        "SELECT MAX(version_seq) FROM entity WHERE family_id = ? AND graph_id = ?",
                        (fid, self._graph_id),
                    ).fetchone()
                    next_seq = (row[0] or 0) + 1
                    for entity in fid_entities:
                        if entity.version_seq <= 1:
                            entity.version_seq = next_seq
                            next_seq += 1
            finally:
                conn.rollback()
        for entity in entities:
            entity.processed_time = _now
            emb_blob = emb_map.get(entity.absolute_id)
            entity.embedding = emb_blob
            attrs = json.dumps(entity.attributes, ensure_ascii=False) if isinstance(entity.attributes, (dict, list)) else entity.attributes
            rows.append((
                entity.absolute_id, entity.family_id, self._graph_id,
                entity.name, entity.content, entity.summary, attrs,
                entity.confidence, getattr(entity, "content_format", "plain"),
                entity.community_id,
                entity.version_seq,
                _fmt_dt(entity.valid_at or entity.event_time),
                _fmt_dt(entity.event_time), _fmt_dt(entity.processed_time),
                entity.episode_id, entity.source_document, emb_blob,
            ))
            if emb_blob is not None:
                cache_items.append((entity, np.frombuffer(emb_blob, dtype=np.float32)))
            else:
                cache_items.append((entity, None))
        with self._write_lock:
            conn = self._connect()
            try:
                conn.executemany(
                    f"INSERT OR REPLACE INTO entity ({', '.join(ENTITY_COLUMNS)}) VALUES ({', '.join('?' * len(ENTITY_COLUMNS))})",
                    rows,
                )
                conn.commit()
            finally:
                conn.rollback()
        self._update_entity_emb_cache_batch(cache_items)
        for entity in entities:
            self._invalidate_entity_cache(entity.family_id)
            self._cache_entity_name(entity.absolute_id, entity.name)

    def bulk_save_entities_with_embedding(self, entities: List[Entity]):
        if not entities:
            return
        _now = datetime.now()
        rows = []
        cache_items = []
        # Pre-compute version_seq per family_id
        with self._write_lock:
            conn = self._connect()
            try:
                fid_map = {}
                for entity in entities:
                    fid_map.setdefault(entity.family_id, []).append(entity)
                for fid, fid_entities in fid_map.items():
                    row = conn.execute(
                        "SELECT MAX(version_seq) FROM entity WHERE family_id = ? AND graph_id = ?",
                        (fid, self._graph_id),
                    ).fetchone()
                    next_seq = (row[0] or 0) + 1
                    for entity in fid_entities:
                        if entity.version_seq <= 1:
                            entity.version_seq = next_seq
                            next_seq += 1
            finally:
                conn.rollback()
        for entity in entities:
            entity.processed_time = _now
            emb_blob = getattr(entity, 'embedding', None)
            emb_array = None
            if emb_blob is not None:
                if isinstance(emb_blob, np.ndarray):
                    emb_array = emb_blob
                else:
                    emb_array = np.frombuffer(emb_blob, dtype=np.float32)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_array = emb_array / norm
                entity.embedding = emb_array.tobytes()
                cache_items.append((entity, emb_array))
            else:
                cache_items.append((entity, None))
            attrs = json.dumps(entity.attributes, ensure_ascii=False) if isinstance(entity.attributes, (dict, list)) else entity.attributes
            rows.append((
                entity.absolute_id, entity.family_id, self._graph_id,
                entity.name, entity.content, entity.summary, attrs,
                entity.confidence, getattr(entity, "content_format", "plain"),
                entity.community_id,
                entity.version_seq,
                _fmt_dt(entity.valid_at or entity.event_time),
                _fmt_dt(entity.event_time), _fmt_dt(entity.processed_time),
                entity.episode_id, entity.source_document, entity.embedding,
            ))
        with self._write_lock:
            conn = self._connect()
            try:
                conn.executemany(
                    f"INSERT OR REPLACE INTO entity ({', '.join(ENTITY_COLUMNS)}) VALUES ({', '.join('?' * len(ENTITY_COLUMNS))})",
                    rows,
                )
                conn.commit()
            finally:
                conn.rollback()
        for entity in entities:
            self._invalidate_entity_cache(entity.family_id)
            self._cache_entity_name(entity.absolute_id, entity.name)
        self._update_entity_emb_cache_batch(cache_items)
