"""Relation families and assertions repository."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def upsert_relation_family(conn, relation_family_id: str,
                           subject_entity_family_id: str,
                           object_entity_family_id: str,
                           predicate: str = "",
                           canonical_content: str = "",
                           created_at: str = "",
                           updated_at: str = "") -> None:
    predicate = predicate.strip().lower()
    existing = conn.execute(
        "SELECT relation_family_id FROM relation_families WHERE relation_family_id = ?",
        (relation_family_id,),
    ).fetchone()
    if existing:
        conn.execute(
            """UPDATE relation_families
               SET predicate = ?, canonical_content = ?,
                   last_seen_at = ?, updated_at = ?
               WHERE relation_family_id = ?""",
            (predicate, canonical_content, updated_at, updated_at, relation_family_id),
        )
    else:
        conn.execute(
            """INSERT INTO relation_families
               (relation_family_id, subject_entity_family_id, object_entity_family_id,
                predicate, canonical_content, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (relation_family_id, subject_entity_family_id, object_entity_family_id,
             predicate, canonical_content, created_at, updated_at),
        )


def get_relation_family(conn, relation_family_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM relation_families WHERE relation_family_id = ?",
        (relation_family_id,),
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM relation_families LIMIT 0").description]
    return dict(zip(cols, row))


def find_relation_family(conn, subject_family_id: str,
                         object_family_id: str,
                         predicate: str = "") -> Optional[dict]:
    predicate = predicate.strip().lower()
    row = conn.execute(
        """SELECT * FROM relation_families
           WHERE subject_entity_family_id = ?
             AND object_entity_family_id = ?
             AND predicate = ?""",
        (subject_family_id, object_family_id, predicate),
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM relation_families LIMIT 0").description]
    return dict(zip(cols, row))


def insert_relation_assertion(
    conn, relation_id: str, relation_family_id: str,
    episode_id: str, subject_entity_id: str, object_entity_id: str,
    subject_entity_family_id: str, object_entity_family_id: str,
    content: str = "", evidence_text: str = "",
    evidence_start_offset: int = 0, evidence_end_offset: int = 0,
    evidence_line_start: int = 0, evidence_line_end: int = 0,
    processed_at: str = "", run_id: str = ""
) -> None:
    conn.execute(
        """INSERT INTO relation_assertions
           (relation_id, relation_family_id, episode_id,
            subject_entity_id, object_entity_id,
            subject_entity_family_id, object_entity_family_id,
            content, evidence_text,
            evidence_start_offset, evidence_end_offset,
            evidence_line_start, evidence_line_end,
            status, processed_at, run_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
        (relation_id, relation_family_id, episode_id,
         subject_entity_id, object_entity_id,
         subject_entity_family_id, object_entity_family_id,
         content, evidence_text,
         evidence_start_offset, evidence_end_offset,
         evidence_line_start, evidence_line_end,
         processed_at, run_id),
    )


def supersede_assertions_by_episodes(conn, episode_ids: list) -> int:
    """Supersede active assertions for given episodes. Returns count."""
    if not episode_ids:
        return 0
    placeholders = ",".join("?" for _ in episode_ids)
    cur = conn.execute(
        f"UPDATE relation_assertions SET status = 'superseded' "
        f"WHERE episode_id IN ({placeholders}) AND status = 'active'",
        episode_ids,
    )
    return cur.rowcount


def reactivate_assertions_by_episodes(conn, episode_ids: list) -> int:
    if not episode_ids:
        return 0
    placeholders = ",".join("?" for _ in episode_ids)
    cur = conn.execute(
        f"UPDATE relation_assertions SET status = 'active' "
        f"WHERE episode_id IN ({placeholders}) AND status = 'superseded'",
        episode_ids,
    )
    return cur.rowcount


def get_active_assertions_by_episode(conn, episode_id: str) -> list:
    cols = [d[0] for d in conn.execute("SELECT * FROM relation_assertions LIMIT 0").description]
    rows = conn.execute(
        "SELECT * FROM relation_assertions WHERE episode_id = ? AND status = 'active'",
        (episode_id,),
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def validate_same_episode(conn, subject_entity_id: str,
                          object_entity_id: str,
                          episode_id: str) -> bool:
    sub = conn.execute(
        "SELECT episode_id FROM entity_observations WHERE entity_id = ?",
        (subject_entity_id,),
    ).fetchone()
    obj = conn.execute(
        "SELECT episode_id FROM entity_observations WHERE entity_id = ?",
        (object_entity_id,),
    ).fetchone()
    if sub is None or obj is None:
        return False
    return sub[0] == episode_id and obj[0] == episode_id


def list_relation_families(conn, limit: int = 100, offset: int = 0) -> list:
    cols = [d[0] for d in conn.execute("SELECT * FROM relation_families LIMIT 0").description]
    rows = conn.execute(
        "SELECT * FROM relation_families ORDER BY updated_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]
