import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...models import Relation
from ...perf import _perf_timer
from .helpers import RELATION_COLUMNS, _fmt_dt, _row_to_relation

logger = logging.getLogger(__name__)


class _RelationWriteMixin:

    def delete_relation_all_versions(self, family_id: str) -> int:
        return self.delete_relation_by_id(family_id)

    def delete_relation_by_absolute_id(self, absolute_id: str) -> bool:
        with self._relation_write_lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM relates_to WHERE relation_uuid = ? AND graph_id = ?", (absolute_id, self._graph_id))
                cursor = conn.execute("DELETE FROM relation WHERE uuid = ? AND graph_id = ?", (absolute_id, self._graph_id))
                conn.commit()
                deleted = cursor.rowcount > 0
            finally:
                conn.rollback()
        self._invalidate_relation_cache_bulk()
        return deleted

    def delete_relation_by_id(self, family_id: str) -> int:
        with self._relation_write_lock:
            conn = self._connect()
            try:
                abs_ids = [r["uuid"] for r in conn.execute(
                    "SELECT uuid FROM relation WHERE family_id = ? AND graph_id = ?", (family_id, self._graph_id),
                ).fetchall()]
                count = 0
                if abs_ids:
                    placeholders = ",".join("?" * len(abs_ids))
                    conn.execute(f"DELETE FROM relates_to WHERE relation_uuid IN ({placeholders}) AND graph_id = ?", abs_ids + [self._graph_id])
                    cursor = conn.execute("DELETE FROM relation WHERE family_id = ? AND graph_id = ?", (family_id, self._graph_id))
                    count = cursor.rowcount
                conn.commit()
            finally:
                conn.rollback()
        self._invalidate_relation_cache_bulk()
        return count

    def batch_delete_relation_versions_by_absolute_ids(self, absolute_ids: List[str]) -> int:
        if not absolute_ids:
            return 0
        with self._relation_write_lock:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(absolute_ids))
                conn.execute(f"DELETE FROM relates_to WHERE relation_uuid IN ({placeholders}) AND graph_id = ?", absolute_ids + [self._graph_id])
                cursor = conn.execute(f"DELETE FROM relation WHERE uuid IN ({placeholders}) AND graph_id = ?", absolute_ids + [self._graph_id])
                conn.commit()
                deleted = cursor.rowcount
            finally:
                conn.rollback()
        self._invalidate_relation_cache_bulk()
        return deleted

    def batch_delete_relations(self, family_ids: List[str]) -> int:
        if not family_ids:
            return 0
        with self._relation_write_lock:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(family_ids))
                all_uuids = [r["uuid"] for r in conn.execute(
                    f"SELECT uuid FROM relation WHERE family_id IN ({placeholders}) AND graph_id = ?",
                    family_ids + [self._graph_id],
                ).fetchall()]
                count = 0
                if all_uuids:
                    uuid_ph = ",".join("?" * len(all_uuids))
                    conn.execute(f"DELETE FROM relates_to WHERE relation_uuid IN ({uuid_ph}) AND graph_id = ?", all_uuids + [self._graph_id])
                cursor = conn.execute(
                    f"DELETE FROM relation WHERE family_id IN ({placeholders}) AND graph_id = ?",
                    family_ids + [self._graph_id],
                )
                count = cursor.rowcount
                conn.commit()
            finally:
                conn.rollback()
        self._invalidate_relation_cache_bulk()
        return count

    def batch_get_relations_referencing_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, List[Relation]]:
        if not absolute_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(absolute_ids))
            rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation "
                f"WHERE (entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})) AND graph_id = ?",
                absolute_ids + absolute_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        result_map: Dict[str, List[Relation]] = {aid: [] for aid in absolute_ids}
        for r in rows:
            rel = _row_to_relation(dict(r))
            if rel.entity1_absolute_id in result_map:
                result_map[rel.entity1_absolute_id].append(rel)
            if rel.entity2_absolute_id in result_map:
                result_map[rel.entity2_absolute_id].append(rel)
        return result_map

    def invalidate_relation(self, family_id: str, reason: str = "") -> int:
        """Delete all but the latest version of a relation."""
        conn = self._connect()
        try:
            # Delete old versions, keep latest per version_seq
            cursor = conn.execute(
                "DELETE FROM relation WHERE family_id = ? AND graph_id = ? "
                "AND uuid NOT IN (SELECT r2.uuid FROM relation r2 WHERE r2.family_id = ? AND r2.graph_id = ? ORDER BY r2.version_seq DESC LIMIT 1)",
                (family_id, self._graph_id, family_id, self._graph_id),
            )
            conn.commit()
        finally:
            conn.rollback()
        self._invalidate_relation_cache_bulk()
        return cursor.rowcount

    def redirect_relation(self, family_id: str, side: str, new_family_id: str) -> int:
        if side not in ("entity1", "entity2"):
            raise ValueError(f"side must be 'entity1' or 'entity2', got '{side}'")
        side_field = f"{side}_absolute_id"
        with self._relation_write_lock:
            conn = self._connect()
            try:
                target_row = conn.execute(
                    "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY processed_time DESC LIMIT 1",
                    (new_family_id, self._graph_id),
                ).fetchone()
                if not target_row:
                    return 0
                new_abs_id = target_row["uuid"]
                cursor = conn.execute(
                    f"UPDATE relation SET {side_field} = ? WHERE family_id = ? AND graph_id = ?",
                    (new_abs_id, family_id, self._graph_id),
                )
                conn.commit()
                count = cursor.rowcount
            finally:
                conn.rollback()
        self._invalidate_relation_cache_bulk()
        return count

    def update_relation_by_absolute_id(self, absolute_id: str, **fields) -> Optional[Relation]:
        valid_keys = {"content", "summary", "attributes", "confidence"}
        filtered = {k: v for k, v in fields.items() if k in valid_keys and v is not None}
        if not filtered:
            return None
        needs_emb_update = "content" in filtered
        _precomputed_emb = None
        if needs_emb_update and self.embedding_client and self.embedding_client.is_available():
            current = self.get_relation_by_absolute_id(absolute_id)
            if current:
                merged = Relation(
                    name="", content=filtered.get("content", current.content),
                    entity1_absolute_id=current.entity1_absolute_id,
                    entity2_absolute_id=current.entity2_absolute_id,
                )
                _precomputed_emb = self._compute_relation_embedding(merged)
        with self._relation_write_lock:
            conn = self._connect()
            try:
                set_parts = [f"{k} = ?" for k in filtered]
                params = list(filtered.values())
                if _precomputed_emb is not None:
                    set_parts.append("embedding = ?")
                    params.append(_precomputed_emb)
                params.extend([absolute_id, self._graph_id])
                cursor = conn.execute(
                    f"UPDATE relation SET {', '.join(set_parts)} WHERE uuid = ? AND graph_id = ?",
                    params,
                )
                conn.commit()
                if cursor.rowcount == 0:
                    return None
                row = conn.execute(
                    f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE uuid = ? AND graph_id = ?",
                    (absolute_id, self._graph_id),
                ).fetchone()
            finally:
                conn.rollback()
            if not row:
                return None
            relation = _row_to_relation(dict(row))
            if _precomputed_emb is not None:
                relation.embedding = _precomputed_emb
            self._invalidate_relation_cache_bulk()
        if _precomputed_emb is not None:
            emb_array = np.frombuffer(_precomputed_emb, dtype=np.float32)
            self._update_relation_emb_cache(relation, emb_array)
        elif needs_emb_update:
            self._update_relation_emb_cache(relation, None)
        return relation

    def update_relation_confidence(self, family_id: str, confidence: float):
        confidence = max(0.0, min(1.0, confidence))
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE relation SET confidence = ? WHERE family_id = ? AND graph_id = ? "
                "AND uuid = (SELECT uuid FROM relation WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1)",
                (confidence, family_id, self._graph_id, family_id, self._graph_id),
            )
            conn.commit()
        finally:
            conn.rollback()
        self._invalidate_relation_cache_bulk()

    def refresh_relates_to_edges(self, family_ids: List[str] = None):
        conn = self._connect()
        try:
            if family_ids:
                # Incremental: refresh edges for specified entity families
                placeholders = ",".join("?" * len(family_ids))
                # Delete stale edges
                conn.execute(
                    f"DELETE FROM relates_to WHERE graph_id = ? AND "
                    f"(entity1_uuid IN (SELECT uuid FROM entity WHERE family_id IN ({placeholders}) AND graph_id = ?) "
                    f"OR entity2_uuid IN (SELECT uuid FROM entity WHERE family_id IN ({placeholders}) AND graph_id = ?))",
                    [self._graph_id] + family_ids + [self._graph_id] + family_ids + [self._graph_id],
                )
                # Recreate edges for valid relations involving these families
                rows = conn.execute(
                    f"SELECT r.uuid AS r_uuid, r.content, r.entity1_absolute_id, r.entity2_absolute_id "
                    f"FROM relation r "
                    f"WHERE r.graph_id = ? "
                    f"AND (r.entity1_absolute_id IN (SELECT uuid FROM entity WHERE family_id IN ({placeholders}) AND graph_id = ?) "
                    f"OR r.entity2_absolute_id IN (SELECT uuid FROM entity WHERE family_id IN ({placeholders}) AND graph_id = ?))",
                    [self._graph_id] + family_ids + [self._graph_id] + family_ids + [self._graph_id],
                ).fetchall()
                for r in rows:
                    e1_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (r["entity1_absolute_id"], self._graph_id)).fetchone()
                    e2_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (r["entity2_absolute_id"], self._graph_id)).fetchone()
                    if e1_row and e2_row:
                        e1_latest = conn.execute(
                            "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                            (e1_row["family_id"], self._graph_id),
                        ).fetchone()
                        e2_latest = conn.execute(
                            "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                            (e2_row["family_id"], self._graph_id),
                        ).fetchone()
                        if e1_latest and e2_latest:
                            conn.execute(
                                "INSERT OR REPLACE INTO relates_to (entity1_uuid, entity2_uuid, relation_uuid, fact, graph_id) VALUES (?, ?, ?, ?, ?)",
                                (e1_latest["uuid"], e2_latest["uuid"], r["r_uuid"], r["content"], self._graph_id),
                            )
                refreshed = len(rows)
                conn.commit()
                if refreshed > 0:
                    logger.info("refresh_relates_to_edges: incremental refresh for %d families, %d edges", len(family_ids), refreshed)
                return {"refreshed": refreshed}
            else:
                # Full refresh
                conn.execute("DELETE FROM relates_to WHERE graph_id = ?", (self._graph_id,))
                rows = conn.execute(
                    "SELECT r.uuid AS r_uuid, r.content, r.entity1_absolute_id, r.entity2_absolute_id "
                    "FROM relation r WHERE r.graph_id = ?",
                    (self._graph_id,),
                ).fetchall()
                created = 0
                for r in rows:
                    e1_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (r["entity1_absolute_id"], self._graph_id)).fetchone()
                    e2_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (r["entity2_absolute_id"], self._graph_id)).fetchone()
                    if e1_row and e2_row:
                        e1_latest = conn.execute(
                            "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                            (e1_row["family_id"], self._graph_id),
                        ).fetchone()
                        e2_latest = conn.execute(
                            "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                            (e2_row["family_id"], self._graph_id),
                        ).fetchone()
                        if e1_latest and e2_latest:
                            conn.execute(
                                "INSERT OR REPLACE INTO relates_to (entity1_uuid, entity2_uuid, relation_uuid, fact, graph_id) VALUES (?, ?, ?, ?, ?)",
                                (e1_latest["uuid"], e2_latest["uuid"], r["r_uuid"], r["content"], self._graph_id),
                            )
                            created += 1
                conn.commit()
                if created > 0:
                    logger.info("refresh_relates_to_edges: full refresh, created=%d new edges", created)
                return {"deleted": 0, "created": created}
        finally:
            conn.rollback()

    def save_dream_relation(self, entity1_id: str, entity2_id: str,
                            content: str, confidence: float, reasoning: str,
                            dream_cycle_id: Optional[str] = None,
                            episode_id: Optional[str] = None) -> Dict[str, Any]:
        resolved_map = self.resolve_family_ids([entity1_id, entity2_id])
        resolved1 = resolved_map.get(entity1_id, entity1_id)
        resolved2 = resolved_map.get(entity2_id, entity2_id)
        if not resolved1:
            raise ValueError(f"Entity not found: {entity1_id}")
        if not resolved2:
            raise ValueError(f"Entity not found: {entity2_id}")
        entities_map = self.get_entities_by_family_ids([resolved1, resolved2])
        entity1 = entities_map.get(resolved1)
        entity2 = entities_map.get(resolved2)
        if not entity1:
            raise ValueError(f"Entity not found: {entity1_id}")
        if not entity2:
            raise ValueError(f"Entity not found: {entity2_id}")
        existing = self.get_relations_by_entities(resolved1, resolved2, include_candidates=True)
        source_doc = f"dream:{dream_cycle_id}" if dream_cycle_id else "dream"
        if existing:
            latest = existing[0]
            new_confidence = max(latest.confidence or 0, confidence)
            new_prov_entry = {"source": "dream", "dream_cycle_id": dream_cycle_id, "confidence": confidence, "reasoning": reasoning}
            try:
                old_prov = json.loads(latest.provenance) if latest.provenance else []
            except Exception:
                old_prov = []
            old_prov.append(new_prov_entry)
            now = datetime.now()
            record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            merged_content = f"{latest.content}\n[Dream update] {content}" if content != latest.content else latest.content
            try:
                merged_attrs = json.loads(latest.attributes) if latest.attributes else {}
            except (json.JSONDecodeError, TypeError):
                merged_attrs = {}
            if dream_cycle_id:
                merged_attrs.setdefault("additional_dream_cycles", [])
                merged_attrs["additional_dream_cycles"].append(dream_cycle_id)
            relation = Relation(
                absolute_id=record_id, family_id=latest.family_id,
                entity1_absolute_id=latest.entity1_absolute_id,
                entity2_absolute_id=latest.entity2_absolute_id,
                content=merged_content, event_time=now, processed_time=now,
                episode_id=episode_id or latest.episode_id or "",
                source_document=source_doc, confidence=new_confidence,
                provenance=json.dumps(old_prov, ensure_ascii=False),
                attributes=json.dumps(merged_attrs) if merged_attrs else latest.attributes,
            )
            self.save_relation(relation)
            return {"family_id": latest.family_id, "entity1_family_id": resolved1, "entity2_family_id": resolved2,
                    "entity1_name": entity1.name, "entity2_name": entity2.name, "action": "merged"}
        if entity1.name <= entity2.name:
            e1_abs, e2_abs = entity1.absolute_id, entity2.absolute_id
        else:
            e1_abs, e2_abs = entity2.absolute_id, entity1.absolute_id
        now = datetime.now()
        family_id = f"rel_{uuid.uuid4().hex[:12]}"
        record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        provenance_data = {"source": "dream", "dream_cycle_id": dream_cycle_id, "confidence": confidence, "reasoning": reasoning}
        relation = Relation(
            absolute_id=record_id, family_id=family_id,
            entity1_absolute_id=e1_abs, entity2_absolute_id=e2_abs,
            content=content, event_time=now, processed_time=now,
            episode_id=episode_id or "", source_document=source_doc,
            confidence=min(confidence, 0.5),
            provenance=json.dumps([provenance_data], ensure_ascii=False),
            attributes=json.dumps({
                "tier": "candidate", "status": "hypothesized", "corroboration_count": 0,
                "created_by_dream": dream_cycle_id or "unknown", "created_at": now.isoformat(),
            }),
        )
        self.save_relation(relation)
        return {"family_id": family_id, "entity1_family_id": resolved1, "entity2_family_id": resolved2,
                "entity1_name": entity1.name, "entity2_name": entity2.name, "action": "created"}
