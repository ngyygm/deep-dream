"""Pipeline run tracking repository."""

import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)


def insert_pipeline_run(conn, run_id: str, run_type: str, status: str,
                        document_id: str = "", document_version_id: str = "",
                        episode_count: int = 0, entity_count: int = 0,
                        relation_count: int = 0,
                        started_at: str = "",
                        extra_json: str = "{}") -> None:
    conn.execute(
        """INSERT INTO pipeline_runs
           (run_id, run_type, status, document_id, document_version_id,
            episode_count, entity_count, relation_count,
            started_at, extra_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, run_type, status, document_id or None, document_version_id or None,
         episode_count, entity_count, relation_count,
         started_at, extra_json),
    )


def update_pipeline_run_status(conn, run_id: str, status: str,
                               finished_at: str = "",
                               error: str = "",
                               episode_count: Optional[int] = None,
                               entity_count: Optional[int] = None,
                               relation_count: Optional[int] = None) -> None:
    sets = ["status = ?"]
    params: list = [status]
    if finished_at:
        sets.append("finished_at = ?")
        params.append(finished_at)
    if error:
        sets.append("error = ?")
        params.append(error)
    if episode_count is not None:
        sets.append("episode_count = ?")
        params.append(episode_count)
    if entity_count is not None:
        sets.append("entity_count = ?")
        params.append(entity_count)
    if relation_count is not None:
        sets.append("relation_count = ?")
        params.append(relation_count)
    params.append(run_id)
    try:
        conn.execute(
            f"UPDATE pipeline_runs SET {', '.join(sets)} WHERE run_id = ?",
            params,
        )
    except sqlite3.IntegrityError:
        # Databases created before V1.5.1 have a CHECK constraint that only
        # knows running/succeeded/failed.  Keep pause/cancel durable on those
        # databases by recording a terminal failure with an explicit marker;
        # task-level state remains the source of truth for resumability.
        if status not in {"paused", "cancelled"}:
            raise
        fallback_sets = ["status = ?"]
        fallback_params = ["failed"]
        if finished_at:
            fallback_sets.append("finished_at = ?")
            fallback_params.append(finished_at)
        marker = f"[{status}]"
        fallback_error = f"{marker} {error}" if error else marker
        fallback_sets.append("error = ?")
        fallback_params.append(fallback_error)
        fallback_params.append(run_id)
        conn.execute(
            f"UPDATE pipeline_runs SET {', '.join(fallback_sets)} WHERE run_id = ?",
            fallback_params,
        )


def get_pipeline_run(conn, run_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM pipeline_runs LIMIT 0").description]
    return dict(zip(cols, row))
