from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...models import Relation
from ...perf import _perf_timer
from .helpers import RELATION_COLUMNS, _fmt_dt, _row_to_relation


class _RelationReadMixin:

    def get_relation_by_absolute_id(self, relation_absolute_id: str) -> Optional[Relation]:
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE uuid = ? AND graph_id = ?",
                (relation_absolute_id, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        return _row_to_relation(dict(row)) if row else None

    def get_relation_by_family_id(self, family_id: str) -> Optional[Relation]:
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                (family_id, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        return _row_to_relation(dict(row)) if row else None

    def get_relation_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT embedding FROM relation WHERE uuid = ? AND graph_id = ?", (absolute_id, self._graph_id)).fetchone()
        finally:
            conn.rollback()
        if row and row["embedding"]:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            return emb[:num_values].tolist()
        return None

    def get_relation_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
        if not family_ids:
            return {}
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = list({r for r in resolved_map.values() if r})
        if not canonical_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(canonical_ids))
            rows = conn.execute(
                f"SELECT family_id, COUNT(*) AS cnt FROM relation WHERE family_id IN ({placeholders}) AND graph_id = ? GROUP BY family_id",
                canonical_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        return {r["family_id"]: r["cnt"] for r in rows}

    def get_relation_versions(self, family_id: str) -> List[Relation]:
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE family_id = ? AND graph_id = ? ORDER BY processed_time ASC",
                (family_id, self._graph_id),
            ).fetchall()
        finally:
            conn.rollback()
        return [_row_to_relation(dict(r)) for r in rows]

    def get_relation_versions_batch(self, family_ids: List[str]) -> Dict[str, List[Relation]]:
        if not family_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(family_ids))
            rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE family_id IN ({placeholders}) AND graph_id = ? ORDER BY processed_time ASC",
                family_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        versions_map: Dict[str, List[Relation]] = {fid: [] for fid in family_ids}
        for r in rows:
            rel = _row_to_relation(dict(r))
            if rel.family_id in versions_map:
                versions_map[rel.family_id].append(rel)
        return versions_map

    def get_relations_by_absolute_ids(self, absolute_ids: List[str], valid_only: bool = False) -> List[Relation]:
        if not absolute_ids:
            return []
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(absolute_ids))
            extra = " AND version_seq = (SELECT MAX(r2.version_seq) FROM relation r2 WHERE r2.family_id = relation.family_id AND r2.graph_id = relation.graph_id)" if valid_only else ""
            rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE uuid IN ({placeholders}){extra}",
                absolute_ids,
            ).fetchall()
        finally:
            conn.rollback()
        return [_row_to_relation(dict(r)) for r in rows]

    def get_relations_by_entities(self, from_family_id: str, to_family_id: str, include_candidates: bool = False) -> List[Relation]:
        with _perf_timer("get_relations_by_entities"):
            result = self._get_relations_by_entities_impl(from_family_id, to_family_id)
            return self._filter_dream_candidates(result, include_candidates)

    def _get_relations_by_entities_impl(self, from_family_id: str, to_family_id: str) -> List[Relation]:
        from_family_id = self.resolve_family_id(from_family_id)
        to_family_id = self.resolve_family_id(to_family_id)
        if not from_family_id or not to_family_id:
            return []
        conn = self._connect()
        try:
            from_ids = [r["uuid"] for r in conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?", (from_family_id, self._graph_id),
            ).fetchall()]
            to_ids = [r["uuid"] for r in conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?", (to_family_id, self._graph_id),
            ).fetchall()]
            if not from_ids or not to_ids:
                return []
            all_ids = from_ids + to_ids
            placeholders = ",".join("?" * len(all_ids))
            from_ph = ",".join("?" * len(from_ids))
            to_ph = ",".join("?" * len(to_ids))
            rows = conn.execute(
                f"SELECT r.* FROM relation r "
                f"INNER JOIN ("
                f"  SELECT family_id, MAX(version_seq) AS max_vs FROM relation "
                f"  WHERE graph_id = ? GROUP BY family_id"
                f") latest ON r.family_id = latest.family_id AND r.version_seq = latest.max_vs "
                f"WHERE r.graph_id = ? "
                f"AND ((r.entity1_absolute_id IN ({from_ph}) AND r.entity2_absolute_id IN ({to_ph})) "
                f"  OR (r.entity1_absolute_id IN ({to_ph}) AND r.entity2_absolute_id IN ({from_ph}))) "
                f"ORDER BY r.processed_time DESC",
                [self._graph_id, self._graph_id] + from_ids + to_ids + to_ids + from_ids,
            ).fetchall()
        finally:
            conn.rollback()
        return [_row_to_relation(dict(r)) for r in rows]

    def get_relations_by_entity_absolute_ids(self, entity_absolute_ids: List[str], limit: Optional[int] = None,
                                              include_candidates: bool = False) -> List[Relation]:
        if not entity_absolute_ids:
            return []
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(entity_absolute_ids))
            query = (
                f"SELECT r.* FROM relation r "
                f"INNER JOIN ("
                f"  SELECT family_id, MAX(version_seq) AS max_vs FROM relation "
                f"  WHERE graph_id = ? GROUP BY family_id"
                f") latest ON r.family_id = latest.family_id AND r.version_seq = latest.max_vs "
                f"WHERE r.graph_id = ? "
                f"AND (r.entity1_absolute_id IN ({placeholders}) OR r.entity2_absolute_id IN ({placeholders})) "
                f"ORDER BY r.processed_time DESC"
            )
            params = [self._graph_id, self._graph_id] + entity_absolute_ids + entity_absolute_ids
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        relations = [_row_to_relation(dict(r)) for r in rows]
        return self._filter_dream_candidates(relations, include_candidates)

    def get_relations_by_entity_pairs(self, entity_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], List[Relation]]:
        if not entity_pairs:
            return {}
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)}, entity1_family_id, entity2_family_id "
                f"FROM relation WHERE graph_id = ?",
                (self._graph_id,),
            ).fetchall()
        finally:
            conn.rollback()
        _rel_by_pair: Dict[Tuple[str, str], List[Relation]] = defaultdict(list)
        seen_rel_fids: set = set()
        for r in rows:
            rel = _row_to_relation(dict(r))
            if rel.family_id in seen_rel_fids:
                continue
            seen_rel_fids.add(rel.family_id)
            f1 = r["entity1_family_id"]
            f2 = r["entity2_family_id"]
            if f1 and f2:
                pk = (f1, f2) if f1 <= f2 else (f2, f1)
                _rel_by_pair[pk].append(rel)
        results: Dict[Tuple[str, str], List[Relation]] = {}
        for e1, e2 in entity_pairs:
            pk = (e1, e2) if e1 <= e2 else (e2, e1)
            if pk not in results:
                results[pk] = _rel_by_pair.get(pk, [])
        return results

    def get_relation_embeddings(self, family_ids: List[str]) -> Dict[str, Any]:
        if not family_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(family_ids))
            rows = conn.execute(
                f"SELECT family_id, embedding FROM relation WHERE family_id IN ({placeholders}) AND embedding IS NOT NULL AND graph_id = ?",
                family_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        result = {}
        for r in rows:
            emb = r["embedding"]
            if emb:
                result[r["family_id"]] = np.frombuffer(emb, dtype=np.float32).copy()
        return result

    def get_relations_by_family_ids(self, family_ids: List[str], limit: int = 100, time_point: Optional[str] = None) -> List[Relation]:
        if not family_ids:
            return []
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(family_ids))
            tp_filter = ""
            tp_params = []
            if time_point:
                tp_filter = " AND (r.valid_at IS NULL OR r.valid_at <= ?)"
                tp_params.append(time_point)
            query = (
                f"SELECT DISTINCT r.* FROM entity e "
                f"INNER JOIN relation r ON (r.entity1_absolute_id = e.uuid OR r.entity2_absolute_id = e.uuid) "
                f"WHERE e.family_id IN ({placeholders}) AND e.graph_id = ? "
                f"AND r.graph_id = ?{tp_filter} "
                f"LIMIT ?"
            )
            params = family_ids + [self._graph_id, self._graph_id] + tp_params + [limit]
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [_row_to_relation(dict(r)) for r in rows]

    def get_relations_referencing_absolute_id(self, absolute_id: str) -> List[Relation]:
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation "
                f"WHERE (entity1_absolute_id = ? OR entity2_absolute_id = ?) AND graph_id = ?",
                (absolute_id, absolute_id, self._graph_id),
            ).fetchall()
        finally:
            conn.rollback()
        rels = [_row_to_relation(dict(r)) for r in rows]
        remap = self._build_entity_abs_id_remap()
        if remap:
            for rel in rels:
                rel.entity1_absolute_id = remap.get(rel.entity1_absolute_id, rel.entity1_absolute_id)
                rel.entity2_absolute_id = remap.get(rel.entity2_absolute_id, rel.entity2_absolute_id)
        return rels

    def get_all_relations(self, limit: Optional[int] = None, offset: Optional[int] = None,
                           exclude_embedding: bool = False, include_candidates: bool = False) -> List[Relation]:
        cols = [c for c in RELATION_COLUMNS if not (exclude_embedding and c == "embedding")]
        conn = self._connect()
        try:
            query = (
                f"SELECT r.* FROM relation r "
                f"INNER JOIN ("
                f"  SELECT family_id, MAX(version_seq) AS max_vs FROM relation "
                f"  WHERE graph_id = ? GROUP BY family_id"
                f") latest ON r.family_id = latest.family_id AND r.version_seq = latest.max_vs "
                f"WHERE r.graph_id = ? ORDER BY r.processed_time DESC"
            )
            params: list = [self._graph_id, self._graph_id]
            if offset:
                query += f" OFFSET {int(offset)}"
            if limit:
                query += f" LIMIT {int(limit)}"
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        relations = [_row_to_relation(dict(r)) for r in rows]
        return self._filter_dream_candidates(relations, include_candidates)

    def stream_all_relations(self, exclude_embedding: bool = True, include_candidates: bool = False, since: Optional[str] = None):
        relations = self.get_all_relations(exclude_embedding=exclude_embedding)
        remap = self._build_entity_abs_id_remap()
        for rel in relations:
            if since:
                pt = _fmt_dt(rel.processed_time)
                if pt and pt <= since:
                    continue
            if remap:
                rel.entity1_absolute_id = remap.get(rel.entity1_absolute_id, rel.entity1_absolute_id)
                rel.entity2_absolute_id = remap.get(rel.entity2_absolute_id, rel.entity2_absolute_id)
            if not self._is_dream_candidate(rel) or include_candidates:
                yield rel

    def count_relations_since(self, since: str) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT family_id) AS cnt FROM relation "
                "WHERE processed_time > ? AND graph_id = ?",
                (since, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def count_unique_relations(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT family_id) AS cnt FROM relation WHERE graph_id = ?",
                (self._graph_id,),
            ).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def get_old_relation_versions(self, limit: int = 100) -> List[Relation]:
        """Get relation versions that are not the latest per family_id."""
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE graph_id = ? "
                f"AND uuid NOT IN (SELECT r2.uuid FROM relation r2 "
                f"WHERE r2.family_id = relation.family_id AND r2.graph_id = relation.graph_id "
                f"ORDER BY r2.version_seq DESC LIMIT 1) "
                f"ORDER BY processed_time DESC LIMIT ?",
                (self._graph_id, limit),
            ).fetchall()
        finally:
            conn.rollback()
        return [_row_to_relation(dict(r)) for r in rows]
