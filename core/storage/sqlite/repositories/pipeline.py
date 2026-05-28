"""Pipeline run tracking repository."""

import logging
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
                               episode_count: int = 0,
                               entity_count: int = 0,
                               relation_count: int = 0) -> None:
    sets = ["status = ?"]
    params: list = [status]
    if finished_at:
        sets.append("finished_at = ?")
        params.append(finished_at)
    if error:
        sets.append("error = ?")
        params.append(error)
    if episode_count:
        sets.append("episode_count = ?")
        params.append(episode_count)
    if entity_count:
        sets.append("entity_count = ?")
        params.append(entity_count)
    if relation_count:
        sets.append("relation_count = ?")
        params.append(relation_count)
    params.append(run_id)
    conn.execute(
        f"UPDATE pipeline_runs SET {', '.join(sets)} WHERE run_id = ?",
        params,
    )


def get_pipeline_run(conn, run_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM pipeline_runs LIMIT 0").description]
    return dict(zip(cols, row))
