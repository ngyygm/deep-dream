"""Entity read mixin for SQLiteGraphStorageManager — query methods for
fetching entities by absolute_id, family_id, versions, and time-point lookups."""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from ...models import Entity
from .helpers import ENTITY_COLUMNS, _row_to_entity

logger = logging.getLogger(__name__)


class _EntityReadMixin:
    """Entity query methods: get by absolute_id, family_id, versions, time-point."""

    def get_entity_by_absolute_id(self, absolute_id: str) -> Optional[Entity]:
        cache_key = f"entity:by_abs:{absolute_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE uuid = ? AND graph_id = ?",
                (absolute_id, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        if not row:
            return None
        entity = _row_to_entity(dict(row))
        self._cache.set(cache_key, entity, ttl=60)
        return entity

    def get_entity_by_family_id(self, family_id: str) -> Optional[Entity]:
        cache_key = f"entity:by_fid:{family_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        resolved_fid = self.resolve_family_id(family_id)
        if resolved_fid != family_id:
            cache_key2 = f"entity:by_fid:{resolved_fid}"
            cached = self._cache.get(cache_key2)
            if cached is not None:
                self._cache.set(cache_key, cached, ttl=60)
                return cached
        family_id = resolved_fid
        if not family_id:
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                (family_id, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        if not row:
            return None
        entity = _row_to_entity(dict(row))
        self._cache.set(cache_key, entity, ttl=60)
        return entity

    def get_entities_by_family_ids(self, family_ids: List[str]) -> Dict[str, Entity]:
        if not family_ids:
            return {}
        resolved_map = self.resolve_family_ids(list(family_ids))
        valid_fids = set(resolved_map.keys()) | set(resolved_map.values())
        if not valid_fids:
            return {}
        result: Dict[str, Entity] = {}
        uncached = set()
        for fid in valid_fids:
            cached = self._cache.get(f"entity:by_fid:{fid}")
            if cached is not None:
                result[fid] = cached
            else:
                uncached.add(fid)
        if uncached:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(uncached))
                rows = conn.execute(
                    f"SELECT e.* FROM entity e "
                    f"INNER JOIN ("
                    f"  SELECT family_id, MAX(version_seq) AS max_vs FROM entity "
                    f"  WHERE family_id IN ({placeholders}) AND graph_id = ? GROUP BY family_id"
                    f") latest ON e.family_id = latest.family_id AND e.version_seq = latest.max_vs "
                    f"WHERE e.graph_id = ?",
                    list(uncached) + [self._graph_id, self._graph_id],
                ).fetchall()
            finally:
                conn.rollback()
            for row in rows:
                entity = _row_to_entity(dict(row))
                result[entity.family_id] = entity
                self._cache.set(f"entity:by_fid:{entity.family_id}", entity, ttl=60)
        for orig_fid, resolved_fid in resolved_map.items():
            if resolved_fid in result and orig_fid not in result:
                result[orig_fid] = result[resolved_fid]
        return result

    def get_entities_by_absolute_ids(self, absolute_ids: List[str], valid_only: bool = False) -> List[Entity]:
        if not absolute_ids:
            return []
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(absolute_ids))
            extra = " AND version_seq = (SELECT MAX(e2.version_seq) FROM entity e2 WHERE e2.family_id = entity.family_id AND e2.graph_id = entity.graph_id)" if valid_only else ""
            rows = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE uuid IN ({placeholders}) AND graph_id = ?{extra}",
                absolute_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        return [_row_to_entity(dict(r)) for r in rows]

    def get_latest_absolute_ids_by_family_ids(self, family_ids: List[str]) -> Dict[str, str]:
        if not family_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(family_ids))
            rows = conn.execute(
                f"SELECT family_id, uuid FROM entity "
                f"WHERE family_id IN ({placeholders}) AND graph_id = ? "
                f"GROUP BY family_id HAVING version_seq = MAX(version_seq)",
                family_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        return {r["family_id"]: r["uuid"] for r in rows}

    def _get_all_absolute_ids_for_entity(self, family_id: str) -> List[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?",
                (family_id, self._graph_id),
            ).fetchall()
        finally:
            conn.rollback()
        return [r["uuid"] for r in rows]

    def get_entity_versions(self, family_id: str) -> List[Entity]:
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY processed_time ASC",
                (family_id, self._graph_id),
            ).fetchall()
        finally:
            conn.rollback()
        return [_row_to_entity(dict(r)) for r in rows]

    def get_entity_versions_batch(self, family_ids: List[str]) -> Dict[str, List[Entity]]:
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
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE family_id IN ({placeholders}) AND graph_id = ? ORDER BY processed_time ASC",
                canonical_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        versions_map: Dict[str, List[Entity]] = {fid: [] for fid in canonical_ids}
        for row in rows:
            entity = _row_to_entity(dict(row))
            if entity.family_id in versions_map:
                versions_map[entity.family_id].append(entity)
        return {orig: versions_map.get(resolved, []) for orig, resolved in resolved_map.items() if resolved}

    def get_entity_version_count(self, family_id: str) -> int:
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM entity WHERE family_id = ? AND graph_id = ?",
                (family_id, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def get_entity_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
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
                "SELECT family_id, COUNT(*) AS cnt FROM entity WHERE family_id IN ({}) AND graph_id = ? GROUP BY family_id".format(placeholders),
                canonical_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        counts = {r["family_id"]: r["cnt"] for r in rows}
        return {fid: counts.get(canonical, 0) for fid, canonical in resolved_map.items() if canonical}

    def get_entity_absolute_ids_up_to_version(self, family_id: str, max_absolute_id: str) -> List[str]:
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT processed_time FROM entity WHERE uuid = ? AND graph_id = ?",
                (max_absolute_id, self._graph_id),
            ).fetchone()
            if not row:
                return []
            max_pt = row["processed_time"]
            rows = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND processed_time <= ? AND graph_id = ? ORDER BY processed_time ASC",
                (family_id, max_pt, self._graph_id),
            ).fetchall()
        finally:
            conn.rollback()
        return [r["uuid"] for r in rows]

    def get_entity_version_at_time(self, family_id: str, time_point: datetime) -> Optional[Entity]:
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE family_id = ? AND event_time <= ? AND graph_id = ? ORDER BY processed_time DESC LIMIT 1",
                (family_id, time_point.isoformat(), self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        if not row:
            return None
        return _row_to_entity(dict(row))
