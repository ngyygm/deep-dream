"""Snapshot mixin — change tracking and point-in-time snapshots."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .helpers import ENTITY_COLUMNS, RELATION_COLUMNS, _row_to_entity, _row_to_relation


class _SnapshotMixin:

    def get_changes(self, since: datetime, until: Optional[datetime] = None) -> Dict[str, Any]:
        if until is None:
            until = datetime.now(timezone.utc)
        since_iso = since.isoformat()
        until_iso = until.isoformat()
        conn = self._connect()
        try:
            ent_rows = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE event_time >= ? AND event_time <= ? AND graph_id = ? ORDER BY event_time DESC",
                (since_iso, until_iso, self._graph_id),
            ).fetchall()
            entities = [_row_to_entity(dict(r)) for r in ent_rows]
            rel_rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE event_time >= ? AND event_time <= ? AND graph_id = ? ORDER BY event_time DESC",
                (since_iso, until_iso, self._graph_id),
            ).fetchall()
            relations = [_row_to_relation(dict(r)) for r in rel_rows]
        finally:
            conn.rollback()
        return {"entities": entities, "relations": relations}

    def get_snapshot(self, time_point: datetime, limit: Optional[int] = None) -> Dict[str, Any]:
        time_iso = time_point.isoformat()
        _limit = limit or 10000
        conn = self._connect()
        try:
            ent_rows = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity "
                f"WHERE (valid_at IS NULL OR valid_at <= ?) AND graph_id = ? "
                f"ORDER BY event_time DESC LIMIT ?",
                (time_iso, self._graph_id, _limit),
            ).fetchall()
            entities = [_row_to_entity(dict(r)) for r in ent_rows]
            rel_rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation "
                f"WHERE (valid_at IS NULL OR valid_at <= ?) AND graph_id = ? "
                f"ORDER BY event_time DESC LIMIT ?",
                (time_iso, self._graph_id, _limit),
            ).fetchall()
            relations = [_row_to_relation(dict(r)) for r in rel_rows]
        finally:
            conn.rollback()
        return {"entities": entities, "relations": relations}
