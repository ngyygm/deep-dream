"""Entity merge and redirect operations for V1.5 schema."""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

from .repositories import entities as ent_repo, relations as rel_repo

logger = logging.getLogger(__name__)

_MAX_REDIRECT_DEPTH = 16


def _now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_redirect(conn: sqlite3.Connection,
                      source_family_id: str, target_family_id: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO entity_redirects (source_family_id, target_family_id, created_at) "
        "VALUES (?, ?, ?)",
        (source_family_id, target_family_id, _now_str()),
    )


def register_redirects_batch(conn: sqlite3.Connection,
                             redirects: Dict[str, str]) -> None:
    for source, target in redirects.items():
        register_redirect(conn, source, target)


def resolve_family_id(conn: sqlite3.Connection, family_id: str) -> str:
    visited = set()
    current = family_id
    for _ in range(_MAX_REDIRECT_DEPTH):
        if current in visited:
            break
        visited.add(current)
        row = conn.execute(
            "SELECT target_family_id FROM entity_redirects WHERE source_family_id = ?",
            (current,),
        ).fetchone()
        if row is None:
            break
        current = row[0]
    return current


def resolve_family_ids(conn: sqlite3.Connection,
                       family_ids: Iterable[str]) -> Dict[str, str]:
    return {fid: resolve_family_id(conn, fid) for fid in family_ids}


def merge_entity_families(conn: sqlite3.Connection,
                          target_family_id: str,
                          source_family_ids: List[str],
                          skip_name_check: bool = False) -> Dict[str, Any]:
    target_family_id = resolve_family_id(conn, target_family_id)
    merged = []
    for source_id in source_family_ids:
        source_id = resolve_family_id(conn, source_id)
        if source_id == target_family_id:
            continue
        # Reassign observations
        conn.execute(
            "UPDATE entity_observations SET entity_family_id = ? WHERE entity_family_id = ?",
            (target_family_id, source_id),
        )
        # Reassign mentions
        conn.execute(
            "UPDATE entity_mentions SET entity_family_id = ? WHERE entity_family_id = ?",
            (target_family_id, source_id),
        )
        # Update relation_families that reference source
        conn.execute(
            "UPDATE relation_families SET subject_entity_family_id = ? "
            "WHERE subject_entity_family_id = ?",
            (target_family_id, source_id),
        )
        conn.execute(
            "UPDATE relation_families SET object_entity_family_id = ? "
            "WHERE object_entity_family_id = ?",
            (target_family_id, source_id),
        )
        # Update relation_assertions cache
        conn.execute(
            "UPDATE relation_assertions SET subject_entity_family_id = ? "
            "WHERE subject_entity_family_id = ?",
            (target_family_id, source_id),
        )
        conn.execute(
            "UPDATE relation_assertions SET object_entity_family_id = ? "
            "WHERE object_entity_family_id = ?",
            (target_family_id, source_id),
        )
        # Delete source family
        conn.execute("DELETE FROM entity_families WHERE entity_family_id = ?", (source_id,))
        # Register redirect
        register_redirect(conn, source_id, target_family_id)
        merged.append(source_id)
    conn.commit()
    return {"merged": merged, "target": target_family_id}


def redirect_entity_relations(conn: sqlite3.Connection,
                              old_family_id: str, new_family_id: str) -> None:
    conn.execute(
        "UPDATE relation_families SET subject_entity_family_id = ? "
        "WHERE subject_entity_family_id = ?",
        (new_family_id, old_family_id),
    )
    conn.execute(
        "UPDATE relation_families SET object_entity_family_id = ? "
        "WHERE object_entity_family_id = ?",
        (new_family_id, old_family_id),
    )
    conn.execute(
        "UPDATE relation_assertions SET subject_entity_family_id = ? "
        "WHERE subject_entity_family_id = ?",
        (new_family_id, old_family_id),
    )
    conn.execute(
        "UPDATE relation_assertions SET object_entity_family_id = ? "
        "WHERE object_entity_family_id = ?",
        (new_family_id, old_family_id),
    )
    conn.commit()


def delete_entity_all_versions(conn: sqlite3.Connection, family_id: str) -> int:
    conn.execute("DELETE FROM entity_mentions WHERE entity_family_id = ?", (family_id,))
    cnt = conn.execute(
        "SELECT COUNT(*) FROM entity_observations WHERE entity_family_id = ?",
        (family_id,),
    ).fetchone()[0]
    conn.execute("DELETE FROM entity_observations WHERE entity_family_id = ?", (family_id,))
    conn.execute("DELETE FROM entity_families WHERE entity_family_id = ?", (family_id,))
    conn.commit()
    return cnt


def dedup_merge_batch(conn: sqlite3.Connection,
                      pairs: List[Tuple[str, str]]) -> int:
    total = 0
    for old_fid, new_fid in pairs:
        delete_entity_all_versions(conn, old_fid)
        register_redirect(conn, old_fid, new_fid)
        total += 1
    return total
