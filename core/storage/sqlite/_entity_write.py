import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ...models import ContentPatch, Entity, Relation
from ...perf import _perf_timer
from .helpers import (
    ENTITY_COLUMNS,
    RELATION_COLUMNS,
    _encode_and_normalize,
    _fmt_dt,
    _parse_dt,
    _row_to_entity,
    _row_to_relation,
)

logger = logging.getLogger(__name__)

_EMB_CONTENT_MAX = 512


class _EntityWriteMixin:

    def get_entity_relations(self, entity_absolute_id: str, limit: Optional[int] = None,
                              time_point: Optional[datetime] = None,
                              include_candidates: bool = False) -> List[Relation]:
        conn = self._connect()
        try:
            tp_filter = " AND r.event_time <= ?" if time_point else ""
            params: list = [entity_absolute_id, entity_absolute_id, self._graph_id]
            if time_point:
                params.append(time_point.isoformat())
            query = (
                "SELECT r.* FROM relation r "
                "WHERE (r.entity1_absolute_id = ? OR r.entity2_absolute_id = ?) "
                "AND r.graph_id = ? "
                f"{tp_filter} "
                "ORDER BY r.version_seq DESC"
            )
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        relations = [_row_to_relation(dict(r)) for r in rows]
        return self._filter_dream_candidates(relations, include_candidates)

    def get_entity_relations_by_family_id(self, family_id: str, limit: Optional[int] = None,
                                           time_point: Optional[datetime] = None,
                                           max_version_absolute_id: Optional[str] = None,
                                           include_candidates: bool = False) -> List[Relation]:
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        conn = self._connect()
        try:
            # Get all abs_ids for this family (current versions only)
            abs_rows = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC",
                (family_id, self._graph_id),
            ).fetchall()
            abs_ids = [r["uuid"] for r in abs_rows]
        finally:
            conn.rollback()
        if not abs_ids:
            return []
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(abs_ids))
            query = (
                f"SELECT r.* FROM relation r "
                f"WHERE (r.entity1_absolute_id IN ({placeholders}) OR r.entity2_absolute_id IN ({placeholders})) "
                f"AND r.graph_id = ? "
            )
            params: list = abs_ids + abs_ids + [self._graph_id]
            if time_point:
                query += " AND r.event_time <= ?"
                params.append(time_point.isoformat())
            query += " ORDER BY r.processed_time DESC"
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        relations = [_row_to_relation(dict(r)) for r in rows]
        return self._filter_dream_candidates(relations, include_candidates)

    def get_entity_relations_timeline(self, family_id: str, version_abs_ids: List[str]) -> List[Dict]:
        family_id = self.resolve_family_id(family_id)
        if not family_id or not version_abs_ids:
            return []
        conn = self._connect()
        try:
            abs_rows = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?",
                (family_id, self._graph_id),
            ).fetchall()
            abs_ids = [r["uuid"] for r in abs_rows]
            if not abs_ids:
                return []
            placeholders_abs = ",".join("?" * len(abs_ids))
            placeholders_ver = ",".join("?" * len(version_abs_ids))
            version_rows = conn.execute(
                f"SELECT uuid, processed_time FROM entity WHERE uuid IN ({placeholders_ver}) AND graph_id = ? ORDER BY processed_time ASC",
                version_abs_ids + [self._graph_id],
            ).fetchall()
            version_times = [(r["uuid"], r["processed_time"]) for r in version_rows]
            if not version_times:
                return []
            rel_rows = conn.execute(
                f"SELECT uuid, family_id, content, event_time, processed_time FROM relation "
                f"WHERE (entity1_absolute_id IN ({placeholders_abs}) OR entity2_absolute_id IN ({placeholders_abs})) "
                f"AND graph_id = ?",
                abs_ids + abs_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        timeline = []
        seen = set()
        for rel in rel_rows:
            rel_uuid = rel["uuid"]
            if rel_uuid in seen:
                continue
            rel_pt = rel["processed_time"]
            for _, v_pt in version_times:
                if rel_pt and v_pt and rel_pt <= v_pt:
                    seen.add(rel_uuid)
                    timeline.append({
                        "family_id": rel["family_id"],
                        "content": rel["content"],
                        "event_time": rel["event_time"],
                        "absolute_id": rel_uuid,
                    })
                    break
        return timeline

    def count_entity_relations_by_family_ids(self, family_ids: List[str]) -> Dict[str, int]:
        if not family_ids:
            return {}
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = list({r for r in resolved_map.values() if r})
        if not canonical_ids:
            return {}
        conn = self._connect()
        try:
            counts = {}
            for fid in canonical_ids:
                abs_rows = conn.execute(
                    "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?",
                    (fid, self._graph_id),
                ).fetchall()
                abs_ids = [r["uuid"] for r in abs_rows]
                if abs_ids:
                    placeholders = ",".join("?" * len(abs_ids))
                    row = conn.execute(
                        f"SELECT COUNT(*) AS cnt FROM relation WHERE (entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})) AND graph_id = ?",
                        abs_ids + abs_ids + [self._graph_id],
                    ).fetchone()
                    counts[fid] = row["cnt"] if row else 0
                else:
                    counts[fid] = 0
        finally:
            conn.rollback()
        return {orig: counts.get(resolved, 0) for orig, resolved in resolved_map.items() if resolved}

    # --- Batch profiles ---

    def batch_get_entity_profiles(self, family_ids: List[str]) -> List[Dict[str, Any]]:
        if not family_ids:
            return []
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_map: Dict[str, str] = {}
        canonical_set: List[str] = []
        _seen_resolved: set = set()
        for fid in family_ids:
            resolved = resolved_map.get(fid, fid)
            if resolved and resolved not in _seen_resolved:
                canonical_map[fid] = resolved
                canonical_set.append(resolved)
                _seen_resolved.add(resolved)
        if not canonical_set:
            return [{"family_id": fid, "entity": None, "relations": [], "version_count": 0} for fid in family_ids]

        entity_map: Dict[str, tuple] = {}
        fid_to_aids: Dict[str, List[str]] = {}
        all_aids: Set[str] = set()

        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(canonical_set))
            rows = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE family_id IN ({placeholders}) AND graph_id = ? ORDER BY version_seq DESC",
                canonical_set + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()

        for row in rows:
            entity = _row_to_entity(dict(row))
            fid = entity.family_id
            if fid not in entity_map:
                entity_map[fid] = (entity, 1)
                fid_to_aids[fid] = [entity.absolute_id]
                all_aids.add(entity.absolute_id)
            else:
                ent, vc = entity_map[fid]
                entity_map[fid] = (ent, vc + 1)
                fid_to_aids[fid].append(entity.absolute_id)
                all_aids.add(entity.absolute_id)

        relations_map: Dict[str, List] = {fid: [] for fid in canonical_set}
        if all_aids:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(all_aids))
                rel_rows = conn.execute(
                    f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE (entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})) AND graph_id = ?",
                    list(all_aids) + list(all_aids) + [self._graph_id],
                ).fetchall()
            finally:
                conn.rollback()
            aid_to_fid = {}
            for fid, aids in fid_to_aids.items():
                for aid in aids:
                    aid_to_fid[aid] = fid
            for r in rel_rows:
                rel = _row_to_relation(dict(r))
                fid1 = aid_to_fid.get(rel.entity1_absolute_id)
                fid2 = aid_to_fid.get(rel.entity2_absolute_id)
                if fid1:
                    relations_map[fid1].append(rel)
                if fid2 and fid2 != fid1:
                    relations_map[fid2].append(rel)

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

    # --- Delete ---

    def delete_entity_all_versions(self, family_id: str) -> int:
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        abs_ids = self._get_all_absolute_ids_for_entity(family_id)
        with self._write_lock:
            conn = self._connect()
            try:
                for aid in abs_ids:
                    conn.execute("DELETE FROM relates_to WHERE entity1_uuid = ? OR entity2_uuid = ?", (aid, aid))
                count = conn.execute(
                    "DELETE FROM entity WHERE family_id = ? AND graph_id = ?",
                    (family_id, self._graph_id),
                ).rowcount
                conn.commit()
            finally:
                conn.rollback()
            self._invalidate_entity_cache(family_id)
            self._cache.invalidate("sim_search:")
            self._cache.invalidate_keys(["graph_stats"])
        return count

    def delete_entity_by_absolute_id(self, absolute_id: str) -> bool:
        deleted = False
        fid = None
        with self._write_lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (absolute_id, self._graph_id)).fetchone()
                if row:
                    fid = row["family_id"]
                    conn.execute("DELETE FROM relates_to WHERE entity1_uuid = ? OR entity2_uuid = ?", (absolute_id, absolute_id))
                    deleted = conn.execute("DELETE FROM entity WHERE uuid = ? AND graph_id = ?", (absolute_id, self._graph_id)).rowcount > 0
                conn.commit()
            finally:
                conn.rollback()
        if fid:
            self._invalidate_entity_cache(fid)
        else:
            self._invalidate_entity_cache_bulk()
        self._cache.invalidate_keys(["graph_stats"])
        return deleted

    def batch_delete_entities(self, family_ids: List[str]) -> int:
        resolved_map = self.resolve_family_ids(family_ids)
        resolved = list({r for r in resolved_map.values() if r})
        if not resolved:
            return 0
        count = 0
        with self._write_lock:
            conn = self._connect()
            try:
                for fid in resolved:
                    abs_rows = conn.execute("SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?", (fid, self._graph_id)).fetchall()
                    for r in abs_rows:
                        conn.execute("DELETE FROM relates_to WHERE entity1_uuid = ? OR entity2_uuid = ?", (r["uuid"], r["uuid"]))
                    count += conn.execute("DELETE FROM entity WHERE family_id = ? AND graph_id = ?", (fid, self._graph_id)).rowcount
                conn.commit()
            finally:
                conn.rollback()
            self._invalidate_entity_cache_bulk()
            self._cache.invalidate_keys(["graph_stats"])
        return count

    def batch_delete_entity_versions_by_absolute_ids(self, absolute_ids: List[str]) -> int:
        if not absolute_ids:
            return 0
        with self._write_lock:
            conn = self._connect()
            try:
                for aid in absolute_ids:
                    conn.execute("DELETE FROM relates_to WHERE entity1_uuid = ? OR entity2_uuid = ?", (aid, aid))
                placeholders = ",".join("?" * len(absolute_ids))
                deleted = conn.execute(
                    f"DELETE FROM entity WHERE uuid IN ({placeholders}) AND graph_id = ?",
                    absolute_ids + [self._graph_id],
                ).rowcount
                conn.commit()
            finally:
                conn.rollback()
            self._invalidate_entity_cache_bulk()
            self._cache.invalidate_keys(["graph_stats"])
        return deleted

    def delete_entity_by_id(self, family_id: str) -> int:
        return self.delete_entity_all_versions(family_id)

    # --- Update ---

    def update_entity_by_absolute_id(self, absolute_id: str, **fields) -> Optional[Entity]:
        valid_keys = {"name", "content", "summary", "attributes", "confidence"}
        filtered = {k: v for k, v in fields.items() if k in valid_keys and v is not None}
        if not filtered:
            return self.get_entity_by_absolute_id(absolute_id)
        needs_emb_update = "name" in filtered or "content" in filtered
        _precomputed_emb = None
        if needs_emb_update and self.embedding_client and self.embedding_client.is_available():
            current = self.get_entity_by_absolute_id(absolute_id)
            if current:
                merged_name = filtered.get("name", current.name)
                merged_content = filtered.get("content", current.content)
                if len(merged_content or "") > _EMB_CONTENT_MAX:
                    merged_content = merged_content[:_EMB_CONTENT_MAX]
                text = f"# {merged_name}\n{merged_content}"
                _emb_result = _encode_and_normalize(self.embedding_client, text)
                if _emb_result is not None:
                    _precomputed_emb = _emb_result
        embedding_blob = _precomputed_emb[0] if _precomputed_emb else None
        with self._write_lock:
            conn = self._connect()
            try:
                set_parts = [f"{k} = ?" for k in filtered]
                params = list(filtered.values())
                if _precomputed_emb:
                    set_parts.append("embedding = ?")
                    params.append(embedding_blob)
                params.append(absolute_id)
                params.append(self._graph_id)
                conn.execute(
                    f"UPDATE entity SET {', '.join(set_parts)} WHERE uuid = ? AND graph_id = ?",
                    params,
                )
                conn.commit()
            finally:
                conn.rollback()
        # Invalidate caches BEFORE re-reading
        self._cache.invalidate(f"entity:by_abs:{absolute_id}")
        entity = self.get_entity_by_absolute_id(absolute_id)
        if entity:
            self._invalidate_entity_cache(entity.family_id)
            if _precomputed_emb:
                entity.embedding = embedding_blob
                self._update_entity_emb_cache(entity, _precomputed_emb[1])
        return entity

    def update_entity_confidence(self, family_id: str, confidence: float):
        confidence = max(0.0, min(1.0, confidence))
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                (family_id, self._graph_id),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE entity SET confidence = ? WHERE uuid = ? AND graph_id = ?",
                    (confidence, row["uuid"], self._graph_id),
                )
                conn.commit()
        finally:
            conn.rollback()
        self._invalidate_entity_cache(family_id)

    def update_entity_summary(self, family_id: str, summary: str):
        resolved = self.resolve_family_id(family_id)
        if not resolved:
            return
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                (resolved, self._graph_id),
            ).fetchone()
            if row:
                conn.execute("UPDATE entity SET summary = ? WHERE uuid = ? AND graph_id = ?", (summary, row["uuid"], self._graph_id))
                conn.commit()
        finally:
            conn.rollback()
        self._invalidate_entity_cache(resolved)

    def batch_update_entity_summaries(self, updates: Dict[str, str]):
        if not updates:
            return
        resolved_map = self.resolve_family_ids(list(updates))
        conn = self._connect()
        try:
            for orig_fid, summary in updates.items():
                resolved = resolved_map.get(orig_fid)
                if resolved:
                    row = conn.execute(
                        "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                        (resolved, self._graph_id),
                    ).fetchone()
                    if row:
                        conn.execute("UPDATE entity SET summary = ? WHERE uuid = ? AND graph_id = ?", (summary, row["uuid"], self._graph_id))
            conn.commit()
        finally:
            conn.rollback()
        self._invalidate_entity_cache_bulk()

    def update_entity_attributes(self, family_id: str, attributes: str):
        resolved = self.resolve_family_id(family_id)
        if not resolved:
            return
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                (resolved, self._graph_id),
            ).fetchone()
            if row:
                conn.execute("UPDATE entity SET attributes = ? WHERE uuid = ? AND graph_id = ?", (attributes, row["uuid"], self._graph_id))
                conn.commit()
        finally:
            conn.rollback()
        self._invalidate_entity_cache(resolved)

