"""Entity families, observations, and mentions repository."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def upsert_entity_family(conn, entity_family_id: str, canonical_name: str,
                         canonical_content: str = "", created_at: str = "",
                         updated_at: str = "") -> None:
    existing = conn.execute(
        "SELECT entity_family_id FROM entity_families WHERE entity_family_id = ?",
        (entity_family_id,),
    ).fetchone()
    if existing:
        conn.execute(
            """UPDATE entity_families
               SET canonical_name = ?, canonical_content = ?,
                   last_seen_at = ?, updated_at = ?
               WHERE entity_family_id = ?""",
            (canonical_name, canonical_content, updated_at, updated_at, entity_family_id),
        )
    else:
        conn.execute(
            """INSERT INTO entity_families
               (entity_family_id, canonical_name, canonical_content, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (entity_family_id, canonical_name, canonical_content, created_at, updated_at),
        )


def get_entity_family(conn, entity_family_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM entity_families WHERE entity_family_id = ?",
        (entity_family_id,),
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM entity_families LIMIT 0").description]
    return dict(zip(cols, row))


def find_entity_family_by_name(conn, canonical_name: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM entity_families WHERE canonical_name = ?",
        (canonical_name,),
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM entity_families LIMIT 0").description]
    return dict(zip(cols, row))


def insert_entity_observation(conn, entity_id: str, entity_family_id: str,
                              episode_id: str, name: str, content: str = "",
                              processed_at: str = "", run_id: str = "") -> None:
    conn.execute(
        """INSERT INTO entity_observations
           (entity_id, entity_family_id, episode_id, name, content,
            status, processed_at, run_id)
           VALUES (?, ?, ?, ?, ?, 'active', ?, ?)""",
        (entity_id, entity_family_id, episode_id, name, content,
         processed_at, run_id),
    )


def get_active_observation(conn, episode_id: str,
                           entity_family_id: str) -> Optional[dict]:
    row = conn.execute(
        """SELECT * FROM entity_observations
           WHERE episode_id = ? AND entity_family_id = ? AND status = 'active'""",
        (episode_id, entity_family_id),
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM entity_observations LIMIT 0").description]
    return dict(zip(cols, row))


def supersede_observations_by_episodes(conn, episode_ids: list) -> int:
    """Supersede active observations for given episodes. Returns count."""
    if not episode_ids:
        return 0
    placeholders = ",".join("?" for _ in episode_ids)
    cur = conn.execute(
        f"UPDATE entity_observations SET status = 'superseded' "
        f"WHERE episode_id IN ({placeholders}) AND status = 'active'",
        episode_ids,
    )
    return cur.rowcount


def reactivate_observations_by_episodes(conn, episode_ids: list) -> int:
    if not episode_ids:
        return 0
    placeholders = ",".join("?" for _ in episode_ids)
    cur = conn.execute(
        f"UPDATE entity_observations SET status = 'active' "
        f"WHERE episode_id IN ({placeholders}) AND status = 'superseded'",
        episode_ids,
    )
    return cur.rowcount


def insert_entity_mention(conn, mention_id: str, entity_id: str,
                          entity_family_id: str, episode_id: str,
                          surface_text: str, start_offset: int = 0,
                          end_offset: int = 0, line_start: int = 0,
                          line_end: int = 0, created_at: str = "") -> None:
    conn.execute(
        """INSERT INTO entity_mentions
           (mention_id, entity_id, entity_family_id, episode_id,
            surface_text, start_offset, end_offset, line_start, line_end, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (mention_id, entity_id, entity_family_id, episode_id,
         surface_text, start_offset, end_offset, line_start, line_end, created_at),
    )


def get_mentions_by_episode(conn, episode_id: str) -> list:
    cols = [d[0] for d in conn.execute("SELECT * FROM entity_mentions LIMIT 0").description]
    rows = conn.execute(
        "SELECT * FROM entity_mentions WHERE episode_id = ?",
        (episode_id,),
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def get_mentions_by_family(conn, entity_family_id: str) -> list:
    cols = [d[0] for d in conn.execute("SELECT * FROM entity_mentions LIMIT 0").description]
    rows = conn.execute(
        "SELECT * FROM entity_mentions WHERE entity_family_id = ?",
        (entity_family_id,),
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def list_entity_families(conn, limit: int = 100, offset: int = 0) -> list:
    cols = [d[0] for d in conn.execute("SELECT * FROM entity_families LIMIT 0").description]
    rows = conn.execute(
        "SELECT * FROM entity_families ORDER BY updated_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]
