"""Entity utility mixin: confidence adjustments, content patches, misc queries, listing, cleanup."""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ...models import ContentPatch, Entity
from ...perf import _perf_timer
from .helpers import (
    ENTITY_COLUMNS,
    _encode_and_normalize,
    _fmt_dt,
    _parse_dt,
    _row_to_entity,
)

logger = logging.getLogger(__name__)

_EMB_CONTENT_MAX = 512


class _EntityUtilMixin:

    def split_entity_version(self, absolute_id: str, new_family_id: str = "") -> Optional[Entity]:
        if not new_family_id:
            new_family_id = f"ent_{uuid.uuid4().hex[:12]}"
        with self._write_lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE entity SET family_id = ? WHERE uuid = ? AND graph_id = ?",
                    (new_family_id, absolute_id, self._graph_id),
                )
                conn.commit()
                row = conn.execute(
                    f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE uuid = ? AND graph_id = ?",
                    (absolute_id, self._graph_id),
                ).fetchone()
            finally:
                conn.rollback()
        if not row:
            return None
        entity = _row_to_entity(dict(row))
        self._invalidate_entity_cache(new_family_id)
        return entity

    # --- Confidence ---

    def adjust_confidence_on_contradiction(self, family_id: str, source_type: str = "entity"):
        table = "entity" if source_type == "entity" else "relation"
        conn = self._connect()
        try:
            conn.execute(
                f"UPDATE {table} SET confidence = MAX(confidence - 0.1, 0.0) "
                f"WHERE family_id = ? AND confidence IS NOT NULL AND graph_id = ? "
                f"AND uuid = (SELECT uuid FROM {table} WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1)",
                (family_id, self._graph_id, family_id, self._graph_id),
            )
            conn.commit()
        finally:
            conn.rollback()
        if source_type == "entity":
            self._invalidate_entity_cache(family_id)
        else:
            self._invalidate_relation_cache_bulk()

    def adjust_confidence_on_contradiction_batch(self, family_ids: List[str], source_type: str = "entity"):
        if not family_ids:
            return
        table = "entity" if source_type == "entity" else "relation"
        conn = self._connect()
        try:
            for fid in family_ids:
                conn.execute(
                    f"UPDATE {table} SET confidence = MAX(confidence - 0.1, 0.0) "
                    f"WHERE family_id = ? AND confidence IS NOT NULL AND graph_id = ? "
                    f"AND uuid = (SELECT uuid FROM {table} WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1)",
                    (fid, self._graph_id, fid, self._graph_id),
                )
            conn.commit()
        finally:
            conn.rollback()
        if source_type == "entity":
            self._invalidate_entity_cache_bulk()
        else:
            self._invalidate_relation_cache_bulk()

    def adjust_confidence_on_corroboration(self, family_id: str, source_type: str = "entity", is_dream: bool = False):
        table = "entity" if source_type == "entity" else "relation"
        delta = 0.025 if is_dream else 0.05
        conn = self._connect()
        try:
            conn.execute(
                f"UPDATE {table} SET confidence = MIN(confidence + ?, 1.0) "
                f"WHERE family_id = ? AND confidence IS NOT NULL AND graph_id = ? "
                f"AND uuid = (SELECT uuid FROM {table} WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1)",
                (delta, family_id, self._graph_id, family_id, self._graph_id),
            )
            conn.commit()
        finally:
            conn.rollback()
        if source_type == "entity":
            self._invalidate_entity_cache(family_id)
        else:
            self._invalidate_relation_cache_bulk()

    def adjust_confidence_on_corroboration_batch(self, family_ids: List[str], source_type: str = "entity", is_dream: bool = False):
        if not family_ids:
            return
        table = "entity" if source_type == "entity" else "relation"
        delta = 0.025 if is_dream else 0.05
        conn = self._connect()
        try:
            for fid in family_ids:
                conn.execute(
                    f"UPDATE {table} SET confidence = MIN(confidence + ?, 1.0) "
                    f"WHERE family_id = ? AND confidence IS NOT NULL AND graph_id = ? "
                    f"AND uuid = (SELECT uuid FROM {table} WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1)",
                    (delta, fid, self._graph_id, fid, self._graph_id),
                )
            conn.commit()
        finally:
            conn.rollback()
        if source_type == "entity":
            self._invalidate_entity_cache_bulk()
        else:
            self._invalidate_relation_cache_bulk()

    # --- Content patches ---

    def save_content_patches(self, patches: list):
        if not patches:
            return
        rows = []
        now_iso = datetime.now().isoformat()
        for p in patches:
            if isinstance(p, dict):
                rows.append((
                    p["uuid"], p["target_type"], p["target_absolute_id"],
                    p["target_family_id"], p["section_key"], p["change_type"],
                    p["old_hash"], p["new_hash"], p["diff_summary"],
                    p["source_document"],
                    _fmt_dt(p["event_time"]) if p.get("event_time") else now_iso,
                ))
            else:
                rows.append((
                    p.uuid, p.target_type, p.target_absolute_id,
                    p.target_family_id, p.section_key, p.change_type,
                    p.old_hash, p.new_hash, p.diff_summary,
                    p.source_document,
                    _fmt_dt(p.event_time) if p.event_time else now_iso,
                ))
        conn = self._connect()
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO content_patch (uuid, target_type, target_absolute_id, target_family_id, section_key, change_type, old_hash, new_hash, diff_summary, source_document, event_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        finally:
            conn.rollback()

    def get_content_patches(self, family_id: str, section_key: str = None) -> list:
        conn = self._connect()
        try:
            if section_key:
                rows = conn.execute(
                    "SELECT * FROM content_patch WHERE target_family_id = ? AND section_key = ? ORDER BY event_time DESC",
                    (family_id, section_key),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM content_patch WHERE target_family_id = ? ORDER BY event_time DESC",
                    (family_id,),
                ).fetchall()
        finally:
            conn.rollback()
        patches = []
        for row in rows:
            rd = dict(row)
            patches.append(ContentPatch(
                uuid=rd["uuid"],
                target_type=rd["target_type"],
                target_absolute_id=rd["target_absolute_id"],
                target_family_id=rd["target_family_id"],
                section_key=rd["section_key"],
                change_type=rd["change_type"],
                old_hash=rd.get("old_hash", ""),
                new_hash=rd.get("new_hash", ""),
                diff_summary=rd.get("diff_summary", ""),
                source_document=rd.get("source_document", ""),
                event_time=_parse_dt(rd.get("event_time")),
            ))
        return patches

    def get_section_history(self, family_id: str, section_key: str) -> list:
        return self.get_content_patches(family_id, section_key=section_key)

    # --- Misc queries ---

    def find_entity_by_name_prefix(self, prefix: str, limit: int = 5) -> list:
        if not prefix:
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE (name LIKE ? OR name = ?) AND graph_id = ? ORDER BY version_seq DESC LIMIT ?",
                (prefix + "%", prefix, self._graph_id, limit),
            ).fetchall()
        finally:
            conn.rollback()
        entities = []
        seen_fids = set()
        for row in rows:
            entity = _row_to_entity(dict(row))
            if entity.family_id not in seen_fids:
                seen_fids.add(entity.family_id)
                entities.append(entity)
        return entities

    def get_entity_provenance(self, family_id: str) -> List[dict]:
        conn = self._connect()
        try:
            abs_rows = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?",
                (family_id, self._graph_id),
            ).fetchall()
            abs_ids = [r["uuid"] for r in abs_rows]
            if not abs_ids:
                return []
            placeholders = ",".join("?" * len(abs_ids))
            mention_rows = conn.execute(
                f"SELECT episode_uuid, context FROM mentions WHERE entity_absolute_id IN ({placeholders}) AND graph_id = ?",
                abs_ids + [self._graph_id],
            ).fetchall()
            if mention_rows:
                return [{"episode_id": r["episode_uuid"], "context": dict(r).get("context", "")} for r in mention_rows]
            # Fallback: indirect via relations
            rel_rows = conn.execute(
                f"SELECT DISTINCT m.episode_uuid, m.context FROM mentions m "
                f"INNER JOIN relation r ON m.target_uuid = r.uuid "
                f"WHERE (r.entity1_absolute_id IN ({placeholders}) OR r.entity2_absolute_id IN ({placeholders})) "
                f"AND m.graph_id = ?",
                abs_ids + abs_ids + [self._graph_id],
            ).fetchall()
            return [{"episode_id": r["episode_uuid"], "context": dict(r).get("context", "")} for r in rel_rows]
        finally:
            conn.rollback()

    def get_entity_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT embedding FROM entity WHERE uuid = ? AND graph_id = ?", (absolute_id, self._graph_id)).fetchone()
        finally:
            conn.rollback()
        if row and row["embedding"]:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            return emb[:num_values].tolist()
        return None

    def get_entity_names_by_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, str]:
        if not absolute_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(absolute_ids))
            rows = conn.execute(f"SELECT uuid, name FROM entity WHERE uuid IN ({placeholders}) AND graph_id = ?", absolute_ids + [self._graph_id]).fetchall()
        finally:
            conn.rollback()
        return {r["uuid"]: r["name"] for r in rows}

    def get_all_entity_names_map(self) -> Dict[str, str]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT uuid, name FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchall()
        finally:
            conn.rollback()
        return {r["uuid"]: r["name"] for r in rows}

    def get_family_ids_by_names(self, names: list) -> dict:
        if not names:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(names))
            rows = conn.execute(
                f"SELECT name, family_id FROM entity WHERE name IN ({placeholders}) AND graph_id = ? ORDER BY version_seq DESC",
                names + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        result = {}
        seen = set()
        for r in rows:
            if r["name"] not in seen:
                seen.add(r["name"])
                result[r["name"]] = r["family_id"]
        return result

    def get_all_entities(self, limit: Optional[int] = None, offset: Optional[int] = None, exclude_embedding: bool = False) -> List[Entity]:
        cols = [c for c in ENTITY_COLUMNS if not (exclude_embedding and c == "embedding")]
        conn = self._connect()
        try:
            query = (
                f"SELECT e.* FROM entity e "
                f"INNER JOIN ("
                f"  SELECT family_id, MAX(version_seq) AS max_vs FROM entity "
                f"  WHERE graph_id = ? GROUP BY family_id"
                f") latest ON e.family_id = latest.family_id AND e.version_seq = latest.max_vs "
                f"WHERE e.graph_id = ? ORDER BY e.processed_time DESC"
            )
            params: list = [self._graph_id, self._graph_id]
            if offset:
                query += f" OFFSET {int(offset)}"
            if limit:
                query += f" LIMIT {int(limit)}"
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [_row_to_entity(dict(r)) for r in rows]

    def stream_all_entities(self, exclude_embedding: bool = True, since: Optional[str] = None):
        entities = self.get_all_entities(exclude_embedding=exclude_embedding)
        for entity in entities:
            if since:
                pt = _fmt_dt(entity.processed_time)
                if pt and pt <= since:
                    continue
            vc = self.get_entity_version_count(entity.family_id)
            yield entity, vc

    def count_entities_since(self, since: str) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT family_id) AS cnt FROM entity "
                "WHERE processed_time > ? AND graph_id = ?",
                (since, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def get_all_entities_before_time(self, time_point: datetime, limit: Optional[int] = None,
                                      exclude_embedding: bool = False) -> List[Entity]:
        conn = self._connect()
        try:
            query = (
                "SELECT e.* FROM entity e "
                "INNER JOIN ("
                "  SELECT family_id, MAX(version_seq) AS max_vs FROM entity "
                "  WHERE event_time <= ? AND graph_id = ? GROUP BY family_id"
                ") latest ON e.family_id = latest.family_id AND e.version_seq = latest.max_vs "
                "WHERE e.graph_id = ? ORDER BY e.processed_time DESC"
            )
            params = [time_point.isoformat(), self._graph_id, self._graph_id]
            if limit:
                query += f" LIMIT {int(limit)}"
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [_row_to_entity(dict(r)) for r in rows]

    def get_latest_entities_projection(self, content_snippet_length: Optional[int] = None) -> List[Dict[str, Any]]:
        snippet_length = content_snippet_length or self.entity_content_snippet_length
        entities_with_emb = self._get_entities_with_embeddings()
        version_counts = self.get_entity_version_counts([e.family_id for e, _ in entities_with_emb])
        results = []
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

    def get_isolated_entities(self, limit: int = 100, offset: int = 0) -> List[Entity]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT e.* FROM entity e "
                "WHERE e.family_id IS NOT NULL AND e.graph_id = ? "
                "AND e.uuid NOT IN (SELECT entity1_uuid FROM relates_to WHERE graph_id = ? UNION SELECT entity2_uuid FROM relates_to WHERE graph_id = ?) "
                "ORDER BY e.processed_time DESC LIMIT ? OFFSET ?",
                (self._graph_id, self._graph_id, self._graph_id, limit, offset),
            ).fetchall()
        finally:
            conn.rollback()
        return [_row_to_entity(dict(r)) for r in rows]

    def count_isolated_entities(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT e.family_id) AS cnt FROM entity e "
                "WHERE e.family_id IS NOT NULL AND e.graph_id = ? "
                "AND e.uuid NOT IN (SELECT entity1_uuid FROM relates_to WHERE graph_id = ? UNION SELECT entity2_uuid FROM relates_to WHERE graph_id = ?)",
                (self._graph_id, self._graph_id, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def count_unique_entities(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def cleanup_old_versions(self, before_date: str = None, dry_run: bool = False) -> Dict[str, Any]:
        """Delete old entity/relation versions while keeping the latest per family_id."""
        conn = self._connect()
        try:
            date_filter = f" AND processed_time < '{before_date}'" if before_date else ""
            ent_count = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM entity WHERE graph_id = ?{date_filter} "
                f"AND uuid NOT IN (SELECT e2.uuid FROM entity e2 "
                f"WHERE e2.family_id = entity.family_id AND e2.graph_id = entity.graph_id "
                f"ORDER BY e2.version_seq DESC LIMIT 1)",
                (self._graph_id,),
            ).fetchone()["cnt"]
            rel_count = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM relation WHERE graph_id = ?{date_filter} "
                f"AND uuid NOT IN (SELECT r2.uuid FROM relation r2 "
                f"WHERE r2.family_id = relation.family_id AND r2.graph_id = relation.graph_id "
                f"ORDER BY r2.version_seq DESC LIMIT 1)",
                (self._graph_id,),
            ).fetchone()["cnt"]
            if dry_run:
                return {"dry_run": True, "entities_to_remove": ent_count, "relations_to_remove": rel_count,
                        "message": f"Preview: will delete {ent_count} old entity versions and {rel_count} old relation versions"}
            conn.execute(
                f"DELETE FROM entity WHERE graph_id = ?{date_filter} "
                f"AND uuid NOT IN (SELECT e2.uuid FROM entity e2 "
                f"WHERE e2.family_id = entity.family_id AND e2.graph_id = entity.graph_id "
                f"ORDER BY e2.version_seq DESC LIMIT 1)",
                (self._graph_id,),
            )
            conn.execute(
                f"DELETE FROM relation WHERE graph_id = ?{date_filter} "
                f"AND uuid NOT IN (SELECT r2.uuid FROM relation r2 "
                f"WHERE r2.family_id = relation.family_id AND r2.graph_id = relation.graph_id "
                f"ORDER BY r2.version_seq DESC LIMIT 1)",
                (self._graph_id,),
            )
            conn.commit()
            return {"dry_run": False, "deleted_entity_versions": ent_count, "deleted_relation_versions": rel_count,
                    "message": f"Deleted {ent_count} old entity versions and {rel_count} old relation versions"}
        finally:
            conn.rollback()

    def cleanup_invalidated_versions(self, before_date: str = None, dry_run: bool = False) -> Dict[str, Any]:
        return self.cleanup_old_versions(before_date=before_date, dry_run=dry_run)

    def get_version_diff(self, family_id: str, v1: str, v2: str) -> dict:
        from ...content_schema import parse_markdown_sections, compute_section_diff
        conn = self._connect()
        try:
            rows = conn.execute("SELECT uuid, content FROM entity WHERE uuid IN (?, ?)", (v1, v2)).fetchall()
        finally:
            conn.rollback()
        v1_content = ""
        v2_content = ""
        for r in rows:
            if r["uuid"] == v1:
                v1_content = r["content"] or ""
            else:
                v2_content = r["content"] or ""
        s1 = parse_markdown_sections(v1_content)
        s2 = parse_markdown_sections(v2_content)
        return compute_section_diff(s1, s2)
