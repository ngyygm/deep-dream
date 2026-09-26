"""Episode and FTS lifecycle repository."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def insert_episode(conn, episode_id: str, episode_family_id: str,
                   document_id: str, document_version_id: str,
                   source_text: str = "", memory_text: str = "",
                   heading_path: str = "", start_offset: int = 0,
                   end_offset: int = 0, line_start: int = 0, line_end: int = 0,
                   chunk_index: int = 0, chunk_hash: str = "",
                   name: str = "", episode_type: str = "",
                   activity_type: str = "", event_time: str = "",
                   processed_at: str = "", run_id: str = "") -> None:
    conn.execute(
        """INSERT INTO episodes
           (episode_id, episode_family_id, document_id, document_version_id,
            source_text, memory_text, heading_path,
            start_offset, end_offset, line_start, line_end,
            chunk_index, chunk_hash, name, episode_type, activity_type,
            status, event_time, processed_at, run_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                   'active', ?, ?, ?)""",
        (episode_id, episode_family_id, document_id, document_version_id,
         source_text, memory_text, heading_path,
         start_offset, end_offset, line_start, line_end,
         chunk_index, chunk_hash, name, episode_type, activity_type,
         event_time, processed_at, run_id),
    )


def get_episode(conn, episode_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM episodes LIMIT 0").description]
    return dict(zip(cols, row))


def get_active_episodes_by_version(conn, document_version_id: str) -> list:
    cols = [d[0] for d in conn.execute("SELECT * FROM episodes LIMIT 0").description]
    rows = conn.execute(
        """SELECT * FROM episodes
           WHERE document_version_id = ? AND status = 'active'
           ORDER BY chunk_index""",
        (document_version_id,),
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def get_active_episodes_by_run(conn, document_version_id: str,
                               run_id: str) -> list:
    cols = [d[0] for d in conn.execute("SELECT * FROM episodes LIMIT 0").description]
    rows = conn.execute(
        """SELECT * FROM episodes
           WHERE document_version_id = ? AND run_id = ? AND status = 'active'
           ORDER BY chunk_index""",
        (document_version_id, run_id),
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def supersede_episodes_by_version(conn, document_version_id: str) -> list:
    """Supersede active episodes for a version. Returns episode IDs."""
    rows = conn.execute(
        "SELECT episode_id FROM episodes WHERE document_version_id = ? AND status = 'active'",
        (document_version_id,),
    ).fetchall()
    ep_ids = [r[0] for r in rows]
    if ep_ids:
        conn.execute(
            "UPDATE episodes SET status = 'superseded' WHERE document_version_id = ? AND status = 'active'",
            (document_version_id,),
        )
    return ep_ids


def reactivate_episodes_by_run(conn, document_version_id: str,
                                run_id: Optional[str] = None) -> list:
    """Reactivate superseded episodes for a specific run. Returns episode IDs."""
    if run_id is not None:
        rows = conn.execute(
            """SELECT episode_id FROM episodes
               WHERE document_version_id = ? AND run_id = ? AND status = 'superseded'
               ORDER BY chunk_index""",
            (document_version_id, run_id),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT episode_id FROM episodes
               WHERE document_version_id = ? AND run_id IS NULL AND status = 'superseded'
               ORDER BY chunk_index""",
            (document_version_id,),
        ).fetchall()
    ep_ids = [r[0] for r in rows]
    if ep_ids:
        placeholders = ",".join("?" for _ in ep_ids)
        conn.execute(
            f"UPDATE episodes SET status = 'active' WHERE episode_id IN ({placeholders})",
            ep_ids,
        )
    return ep_ids


def fts_sync_episode(conn, episode_id: str, document_id: str,
                     document_version_id: str, name: str = "",
                     heading_path: str = "", source_text: str = "",
                     memory_text: str = "") -> None:
    """Delete existing FTS row for episode_id, then insert new."""
    conn.execute("DELETE FROM episodes_fts WHERE episode_id = ?", (episode_id,))
    conn.execute(
        """INSERT INTO episodes_fts
           (episode_id, document_id, document_version_id,
            name, heading_path, source_text, memory_text)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (episode_id, document_id, document_version_id,
         name, heading_path, source_text, memory_text),
    )


def fts_delete_episodes(conn, episode_ids: list) -> None:
    if not episode_ids:
        return
    placeholders = ",".join("?" for _ in episode_ids)
    conn.execute(f"DELETE FROM episodes_fts WHERE episode_id IN ({placeholders})", episode_ids)


def rebuild_fts_all(conn) -> int:
    """Full rebuild of episodes_fts from active episodes. Returns count."""
    conn.execute("DELETE FROM episodes_fts")
    conn.execute("""
        INSERT INTO episodes_fts
            (episode_id, document_id, document_version_id,
             name, heading_path, source_text, memory_text)
        SELECT episode_id, document_id, document_version_id,
               name, heading_path, source_text, memory_text
        FROM episodes
        WHERE status = 'active'
    """)
    count = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE status = 'active'"
    ).fetchone()[0]
    conn.commit()
    return count


def get_latest_successful_run_id(conn, document_version_id: str) -> Optional[str]:
    """Find the latest succeeded chunk/reindex/remember run for a version."""
    row = conn.execute(
        """SELECT run_id FROM pipeline_runs
           WHERE document_version_id = ?
             AND status = 'succeeded'
             AND run_type IN ('chunk', 'reindex', 'remember')
           ORDER BY started_at DESC
           LIMIT 1""",
        (document_version_id,),
    ).fetchone()
    return row[0] if row else None
