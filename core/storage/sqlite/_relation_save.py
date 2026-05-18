import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...models import Relation
from ...perf import _perf_timer
from .helpers import (
    RELATION_COLUMNS,
    _encode_and_normalize,
    _fmt_dt,
    _row_to_relation,
)

logger = logging.getLogger(__name__)


class _RelationSaveMixin:

    def _invalidate_relation_cache(self, family_id: str = None):
        keys = ["graph_stats"]
        if family_id:
            keys.append(f"relation:by_fid:{family_id}")
        self._cache.invalidate_keys(keys)

    def _invalidate_relation_cache_bulk(self):
        self._cache.invalidate_keys(["graph_stats"])

    def _build_relation_embedding_text(self, relation: Relation, entity1_name: str = "", entity2_name: str = "") -> str:
        content = relation.content or ""
        if entity1_name and entity2_name:
            return f"# {entity1_name} → {entity2_name}\n{content}"
        elif entity1_name or entity2_name:
            return f"# {entity1_name or entity2_name}\n{content}"
        return content

    def _resolve_entity_names_for_embedding(self, relation: Relation, names: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
        aid1, aid2 = relation.entity1_absolute_id, relation.entity2_absolute_id
        _enc = self._entity_name_cache
        if names:
            for k, v in names.items():
                if k not in _enc:
                    self._cache_entity_name(k, v)
        name1 = _enc.get(aid1, ...)
        name2 = _enc.get(aid2, ...)
        if name1 is not ... and name2 is not ...:
            return name1, name2
        try:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT uuid, name FROM entity WHERE uuid IN (?, ?)", (aid1, aid2)).fetchall()
            finally:
                conn.rollback()
            for r in rows:
                self._cache_entity_name(r["uuid"], r["name"] or "")
                if r["uuid"] == aid1:
                    name1 = r["name"] or ""
                else:
                    name2 = r["name"] or ""
        except Exception:
            if name1 is ...:
                name1 = ""
            if name2 is ...:
                name2 = ""
        return name1, name2

    def _compute_relation_embedding(self, relation: Relation, names: Optional[Dict[str, str]] = None) -> Optional[bytes]:
        name1, name2 = self._resolve_entity_names_for_embedding(relation, names=names)
        text = self._build_relation_embedding_text(relation, name1, name2)
        result = _encode_and_normalize(self.embedding_client, text)
        return result[0] if result else None

    def _bulk_compute_relation_embeddings(self, relations: List[Relation]) -> Dict[str, bytes]:
        if not self.embedding_client or not self.embedding_client.is_available():
            return {}
        # Batch resolve entity names
        all_abs_ids = set()
        for r in relations:
            if r.entity1_absolute_id:
                all_abs_ids.add(r.entity1_absolute_id)
            if r.entity2_absolute_id:
                all_abs_ids.add(r.entity2_absolute_id)
        entity_names = {}
        if all_abs_ids:
            try:
                conn = self._connect()
                try:
                    ph = ",".join("?" * len(all_abs_ids))
                    rows = conn.execute(f"SELECT uuid, name FROM entity WHERE uuid IN ({ph})", list(all_abs_ids)).fetchall()
                finally:
                    conn.rollback()
                for r in rows:
                    entity_names[r["uuid"]] = r["name"] or ""
            except Exception:
                pass
        # Build texts
        texts = []
        uuids = []
        for r in relations:
            name1 = entity_names.get(r.entity1_absolute_id, "")
            name2 = entity_names.get(r.entity2_absolute_id, "")
            texts.append(self._build_relation_embedding_text(r, name1, name2))
            uuids.append(r.absolute_id)
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

    def _update_relation_emb_cache(self, relation: Relation, emb_array: Optional[np.ndarray]):
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

    def _update_relation_emb_cache_batch(self, items: List[tuple]):
        if self._relation_emb_cache is None or not items:
            return
        if not hasattr(self, '_relation_emb_fid_idx') or self._relation_emb_fid_idx is None:
            self._relation_emb_fid_idx = {r.family_id: i for i, (r, _) in enumerate(self._relation_emb_cache)}
        for relation, emb_array in items:
            idx = self._relation_emb_fid_idx.get(relation.family_id)
            if idx is not None:
                self._relation_emb_cache[idx] = (relation, emb_array)
            else:
                self._relation_emb_cache.append((relation, emb_array))
                self._relation_emb_fid_idx[relation.family_id] = len(self._relation_emb_cache) - 1

    def _get_relations_with_embeddings(self) -> List[tuple]:
        now = time.time()
        if self._relation_emb_cache is not None and (now - self._relation_emb_cache_ts) < self._emb_cache_ttl:
            return self._relation_emb_cache
        with _perf_timer("_get_relations_with_embeddings"):
            result = self._get_relations_with_embeddings_impl()
        self._relation_emb_cache = result
        self._relation_emb_fid_idx = None
        self._relation_emb_cache_ts = time.time()
        # Build HNSW index alongside cache
        if result and self._vector_dim > 0:
            self._relation_hnsw, self._relation_hnsw_items = self._build_hnsw(result, self._vector_dim)
        else:
            self._relation_hnsw = None
            self._relation_hnsw_items = None
        return result

    def _get_relations_with_embeddings_impl(self) -> List[tuple]:
        conn = self._connect()
        try:
            limit = getattr(self, '_emb_cache_max_size', 10000)
            rows = conn.execute(
                f"SELECT r.* FROM relation r "
                f"INNER JOIN ("
                f"  SELECT family_id, MAX(version_seq) AS max_vs FROM relation "
                f"  WHERE graph_id = ? GROUP BY family_id"
                f") latest ON r.family_id = latest.family_id AND r.version_seq = latest.max_vs "
                f"WHERE r.graph_id = ? ORDER BY r.processed_time DESC LIMIT ?",
                (self._graph_id, self._graph_id, limit),
            ).fetchall()
        finally:
            conn.rollback()
        relations = []
        for row in rows:
            relation = _row_to_relation(dict(row))
            emb_array = np.frombuffer(relation.embedding, dtype=np.float32) if relation.embedding else None
            relations.append((relation, emb_array))
        return relations

    def save_relation(self, relation: Relation):
        with _perf_timer("save_relation"):
            emb_array = self._save_relation_impl(relation)
            if emb_array is not None:
                self._update_relation_emb_cache(relation, emb_array)

    def _save_relation_impl(self, relation: Relation, names: Optional[Dict[str, str]] = None):
        valid_at = _fmt_dt(relation.valid_at or relation.event_time)
        embedding_blob = self._compute_relation_embedding(relation, names=names)
        if embedding_blob is not None:
            relation.embedding = embedding_blob
        with self._relation_write_lock:
            conn = self._connect()
            try:
                # Auto-increment version_seq
                max_vs_row = conn.execute(
                    "SELECT MAX(version_seq) AS max_vs FROM relation WHERE family_id = ? AND graph_id = ?",
                    (relation.family_id, self._graph_id),
                ).fetchone()
                version_seq = (max_vs_row["max_vs"] or 0) + 1
                relation.version_seq = version_seq
                attrs = json.dumps(relation.attributes, ensure_ascii=False) if isinstance(relation.attributes, (dict, list)) else relation.attributes
                prov = json.dumps(relation.provenance, ensure_ascii=False) if isinstance(relation.provenance, (dict, list)) else relation.provenance
                conn.execute(
                    f"INSERT OR REPLACE INTO relation ({', '.join(RELATION_COLUMNS)}) "
                    f"VALUES ({', '.join('?' * len(RELATION_COLUMNS))})",
                    (
                        relation.absolute_id, relation.family_id, self._graph_id,
                        relation.entity1_absolute_id, relation.entity2_absolute_id,
                        None, None,  # entity1_family_id, entity2_family_id filled below
                        relation.content, relation.summary, attrs,
                        relation.confidence, prov,
                        getattr(relation, "content_format", "plain"),
                        version_seq, valid_at,
                        _fmt_dt(relation.event_time), _fmt_dt(relation.processed_time),
                        relation.episode_id, relation.source_document,
                        relation.embedding,
                    ),
                )
                # Look up entity family_ids
                e1_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (relation.entity1_absolute_id, self._graph_id)).fetchone()
                e2_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (relation.entity2_absolute_id, self._graph_id)).fetchone()
                e1_fid = e1_row["family_id"] if e1_row else None
                e2_fid = e2_row["family_id"] if e2_row else None
                conn.execute(
                    "UPDATE relation SET entity1_family_id = ?, entity2_family_id = ? WHERE uuid = ? AND graph_id = ?",
                    (e1_fid, e2_fid, relation.absolute_id, self._graph_id),
                )
                # Update RELATES_TO edges
                if e1_fid and e2_fid:
                    # Get latest entities for each family
                    e1_latest = conn.execute(
                        "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                        (e1_fid, self._graph_id),
                    ).fetchone()
                    e2_latest = conn.execute(
                        "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                        (e2_fid, self._graph_id),
                    ).fetchone()
                    if e1_latest and e2_latest:
                        conn.execute(
                            "INSERT OR REPLACE INTO relates_to (entity1_uuid, entity2_uuid, relation_uuid, fact, graph_id) VALUES (?, ?, ?, ?, ?)",
                            (e1_latest["uuid"], e2_latest["uuid"], relation.absolute_id, relation.content, self._graph_id),
                        )
                conn.commit()
            finally:
                conn.rollback()
        self._invalidate_relation_cache_bulk()
        emb_array = None
        if embedding_blob:
            emb_array = np.frombuffer(embedding_blob, dtype=np.float32)
        return emb_array

    def bulk_save_relations(self, relations: List[Relation]):
        if not relations:
            return
        # Compute embeddings synchronously before writing
        emb_map = self._bulk_compute_relation_embeddings(relations)
        rows = []
        cache_items = []
        for relation in relations:
            _attrs = getattr(relation, 'attributes', None)
            _prov = getattr(relation, 'provenance', None)
            emb_blob = emb_map.get(relation.absolute_id)
            relation.embedding = emb_blob
            rows.append((
                relation.absolute_id, relation.family_id, self._graph_id,
                relation.entity1_absolute_id, relation.entity2_absolute_id,
                None, None,
                relation.content, getattr(relation, 'summary', None),
                json.dumps(_attrs, ensure_ascii=False) if isinstance(_attrs, (dict, list)) else _attrs,
                getattr(relation, 'confidence', None),
                json.dumps(_prov, ensure_ascii=False) if isinstance(_prov, (dict, list)) else _prov,
                getattr(relation, 'content_format', None),
                _fmt_dt(relation.valid_at or relation.event_time) if (relation.valid_at or relation.event_time) else None,
                None,  # placeholder - will be replaced with version_seq below
                _fmt_dt(relation.event_time), _fmt_dt(relation.processed_time),
                relation.episode_id, relation.source_document,
                emb_blob,
            ))
            if emb_blob is not None:
                cache_items.append((relation, np.frombuffer(emb_blob, dtype=np.float32)))
            else:
                cache_items.append((relation, None))
        with self._relation_write_lock:
            conn = self._connect()
            try:
                # Auto-increment version_seq per family_id
                fid_to_max_vs = {}
                for relation in relations:
                    fid = relation.family_id
                    if fid not in fid_to_max_vs:
                        row = conn.execute(
                            "SELECT MAX(version_seq) AS max_vs FROM relation WHERE family_id = ? AND graph_id = ?",
                            (fid, self._graph_id),
                        ).fetchone()
                        fid_to_max_vs[fid] = row["max_vs"] or 0

                # Replace the version_seq placeholder (index 13) in each row
                final_rows = []
                for i, relation in enumerate(relations):
                    row = list(rows[i])
                    fid = relation.family_id
                    fid_to_max_vs[fid] += 1
                    row[13] = fid_to_max_vs[fid]
                    relation.version_seq = fid_to_max_vs[fid]
                    final_rows.append(tuple(row))

                conn.executemany(
                    f"INSERT OR REPLACE INTO relation ({', '.join(RELATION_COLUMNS)}) VALUES ({', '.join('?' * len(RELATION_COLUMNS))})",
                    final_rows,
                )
                # Batch-lookup entity family_ids: collect all unique abs_ids
                all_abs_ids = set()
                for r in relations:
                    all_abs_ids.add(r.entity1_absolute_id)
                    all_abs_ids.add(r.entity2_absolute_id)
                uuid_to_fid = {}
                if all_abs_ids:
                    id_list = list(all_abs_ids)
                    ph = ",".join("?" * len(id_list))
                    for row in conn.execute(
                        f"SELECT uuid, family_id FROM entity WHERE uuid IN ({ph}) AND graph_id = ?",
                        id_list + [self._graph_id],
                    ).fetchall():
                        uuid_to_fid[row["uuid"]] = row["family_id"]

                # Batch-lookup latest entity UUIDs per family_id
                all_fids = set(fid for fid in uuid_to_fid.values() if fid)
                fid_to_latest = {}
                if all_fids:
                    fid_list = list(all_fids)
                    ph = ",".join("?" * len(fid_list))
                    for row in conn.execute(
                        f"SELECT family_id, uuid FROM entity WHERE family_id IN ({ph}) AND graph_id = ? ORDER BY version_seq DESC",
                        fid_list + [self._graph_id],
                    ).fetchall():
                        fid_to_latest.setdefault(row["family_id"], row["uuid"])

                # Apply family_ids and RELATES_TO edges
                fid_update_rows = []
                relates_to_rows = []
                for relation in relations:
                    e1_fid = uuid_to_fid.get(relation.entity1_absolute_id)
                    e2_fid = uuid_to_fid.get(relation.entity2_absolute_id)
                    fid_update_rows.append((e1_fid, e2_fid, relation.absolute_id, self._graph_id))
                    if e1_fid and e2_fid:
                        e1_latest = fid_to_latest.get(e1_fid)
                        e2_latest = fid_to_latest.get(e2_fid)
                        if e1_latest and e2_latest:
                            relates_to_rows.append((e1_latest, e2_latest, relation.absolute_id, relation.content, self._graph_id))
                conn.executemany(
                    "UPDATE relation SET entity1_family_id = ?, entity2_family_id = ? WHERE uuid = ? AND graph_id = ?",
                    fid_update_rows,
                )
                if relates_to_rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO relates_to (entity1_uuid, entity2_uuid, relation_uuid, fact, graph_id) VALUES (?, ?, ?, ?, ?)",
                        relates_to_rows,
                    )
                conn.commit()
            finally:
                conn.rollback()
        self._update_relation_emb_cache_batch(cache_items)
        self._invalidate_relation_cache_bulk()

    def bulk_save_relations_with_embedding(self, relations: List[Relation]):
        if not relations:
            return
        rows = []
        cache_items = []
        for relation in relations:
            emb_blob = getattr(relation, 'embedding', None)
            emb_array = None
            if emb_blob is not None:
                if isinstance(emb_blob, np.ndarray):
                    emb_array = emb_blob
                else:
                    emb_array = np.frombuffer(emb_blob, dtype=np.float32)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_array = emb_array / norm
                relation.embedding = emb_array.tobytes()
                cache_items.append((relation, emb_array))
            else:
                cache_items.append((relation, None))
            _attrs = getattr(relation, 'attributes', None)
            _prov = getattr(relation, 'provenance', None)
            rows.append((
                relation.absolute_id, relation.family_id, self._graph_id,
                relation.entity1_absolute_id, relation.entity2_absolute_id,
                None, None,
                relation.content, getattr(relation, 'summary', None),
                json.dumps(_attrs, ensure_ascii=False) if isinstance(_attrs, (dict, list)) else _attrs,
                getattr(relation, 'confidence', None),
                json.dumps(_prov, ensure_ascii=False) if isinstance(_prov, (dict, list)) else _prov,
                getattr(relation, 'content_format', None),
                _fmt_dt(relation.valid_at or relation.event_time) if (relation.valid_at or relation.event_time) else None,
                None,  # placeholder - will be replaced with version_seq below
                _fmt_dt(relation.event_time), _fmt_dt(relation.processed_time),
                relation.episode_id, relation.source_document,
                relation.embedding,
            ))
        with self._relation_write_lock:
            conn = self._connect()
            try:
                # Auto-increment version_seq per family_id
                fid_to_max_vs = {}
                for relation in relations:
                    fid = relation.family_id
                    if fid not in fid_to_max_vs:
                        row = conn.execute(
                            "SELECT MAX(version_seq) AS max_vs FROM relation WHERE family_id = ? AND graph_id = ?",
                            (fid, self._graph_id),
                        ).fetchone()
                        fid_to_max_vs[fid] = row["max_vs"] or 0

                # Replace the version_seq placeholder (index 13) in each row
                final_rows = []
                for i, relation in enumerate(relations):
                    row = list(rows[i])
                    fid = relation.family_id
                    fid_to_max_vs[fid] += 1
                    row[13] = fid_to_max_vs[fid]
                    relation.version_seq = fid_to_max_vs[fid]
                    final_rows.append(tuple(row))

                conn.executemany(
                    f"INSERT OR REPLACE INTO relation ({', '.join(RELATION_COLUMNS)}) VALUES ({', '.join('?' * len(RELATION_COLUMNS))})",
                    final_rows,
                )
                for relation in relations:
                    e1_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (relation.entity1_absolute_id, self._graph_id)).fetchone()
                    e2_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (relation.entity2_absolute_id, self._graph_id)).fetchone()
                    e1_fid = e1_row["family_id"] if e1_row else None
                    e2_fid = e2_row["family_id"] if e2_row else None
                    conn.execute(
                        "UPDATE relation SET entity1_family_id = ?, entity2_family_id = ? WHERE uuid = ? AND graph_id = ?",
                        (e1_fid, e2_fid, relation.absolute_id, self._graph_id),
                    )
                    if e1_fid and e2_fid:
                        e1_latest = conn.execute(
                            "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                            (e1_fid, self._graph_id),
                        ).fetchone()
                        e2_latest = conn.execute(
                            "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                            (e2_fid, self._graph_id),
                        ).fetchone()
                        if e1_latest and e2_latest:
                            conn.execute(
                                "INSERT OR REPLACE INTO relates_to (entity1_uuid, entity2_uuid, relation_uuid, fact, graph_id) VALUES (?, ?, ?, ?, ?)",
                                (e1_latest["uuid"], e2_latest["uuid"], relation.absolute_id, relation.content, self._graph_id),
                            )
                conn.commit()
            finally:
                conn.rollback()
        if self._relation_emb_cache is not None and cache_items:
            if self._relation_emb_fid_idx is not None:
                fid_to_idx = self._relation_emb_fid_idx
            else:
                fid_to_idx = {r.family_id: i for i, (r, _) in enumerate(self._relation_emb_cache)} if self._relation_emb_cache else {}
                self._relation_emb_fid_idx = fid_to_idx
            for relation, emb_array in cache_items:
                idx = fid_to_idx.get(relation.family_id)
                if idx is not None:
                    self._relation_emb_cache[idx] = (relation, emb_array)
                else:
                    self._relation_emb_cache.append((relation, emb_array))
                    fid_to_idx[relation.family_id] = len(self._relation_emb_cache) - 1
        self._invalidate_relation_cache_bulk()
