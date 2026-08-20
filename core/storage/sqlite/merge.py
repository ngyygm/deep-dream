"""Entity merge and redirect operations for V1.5 schema."""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, Iterable, List, Tuple

from .helpers import now_utc_str


logger = logging.getLogger(__name__)

_MAX_REDIRECT_DEPTH = 16


def register_redirect(conn: sqlite3.Connection,
                      source_family_id: str, target_family_id: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO entity_redirects (source_family_id, target_family_id, created_at) "
        "VALUES (?, ?, ?)",
        (source_family_id, target_family_id, now_utc_str()),
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


def _merge_duplicate_relation_families(conn: sqlite3.Connection,
                                       source_id: str,
                                       target_id: str) -> None:
    """Merge relation_families rows whose endpoints would collide after redirect.

    After redirecting *source_id* -> *target_id*, any relation_family row that
    now has the same (subject_entity_family_id, object_entity_family_id,
    is_directed) as an existing row must be merged.  We transfer all
    relation_assertions that point to the duplicate into the surviving row,
    then delete the duplicate.
    """
    # Find rows that currently reference source_id (either as subject or object)
    source_rows = conn.execute(
        "SELECT relation_family_id, subject_entity_family_id, "
        "object_entity_family_id, is_directed, canonical_content "
        "FROM relation_families "
        "WHERE subject_entity_family_id = ? OR object_entity_family_id = ?",
        (source_id, source_id),
    ).fetchall()

    for rel_fid, sub_fid, obj_fid, is_dir, content in source_rows:
        # Compute the post-redirect endpoint pair
        new_sub = target_id if sub_fid == source_id else sub_fid
        new_obj = target_id if obj_fid == source_id else obj_fid

        # Skip self-relations (A -> A) created by the redirect
        if new_sub == new_obj:
            conn.execute("DELETE FROM relation_families WHERE relation_family_id = ?",
                         (rel_fid,))
            conn.execute("DELETE FROM relation_assertions WHERE relation_family_id = ?",
                         (rel_fid,))
            continue

        # Check if a relation_family with the post-redirect pair already exists
        existing = conn.execute(
            "SELECT relation_family_id FROM relation_families "
            "WHERE subject_entity_family_id = ? AND object_entity_family_id = ? "
            "AND is_directed = ? AND relation_family_id != ?",
            (new_sub, new_obj, is_dir, rel_fid),
        ).fetchone()

        if existing:
            survivor_fid = existing[0]
            # Reassign all assertions from the duplicate to the survivor
            conn.execute(
                "UPDATE relation_assertions SET relation_family_id = ? "
                "WHERE relation_family_id = ?",
                (survivor_fid, rel_fid),
            )
            # Update the assertion endpoint family_ids as well
            conn.execute(
                "UPDATE relation_assertions SET subject_entity_family_id = ? "
                "WHERE relation_family_id = ? AND subject_entity_family_id = ?",
                (new_sub, survivor_fid, source_id),
            )
            conn.execute(
                "UPDATE relation_assertions SET object_entity_family_id = ? "
                "WHERE relation_family_id = ? AND object_entity_family_id = ?",
                (new_obj, survivor_fid, source_id),
            )
            # Delete the duplicate relation_family
            conn.execute("DELETE FROM relation_families WHERE relation_family_id = ?",
                         (rel_fid,))
        else:
            # No collision — safe to update endpoints in place
            conn.execute(
                "UPDATE relation_families SET subject_entity_family_id = ?, "
                "object_entity_family_id = ? WHERE relation_family_id = ?",
                (new_sub, new_obj, rel_fid),
            )


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
        # Reassign relation_assertions that reference source
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
        # Merge relation_families: delete duplicates first (source->X that
        # already exist as target->X), then update remaining.
        # Without this, a UNIQUE(subject, object, is_directed) violation
        # crashes the merge and leaves the DB in an inconsistent state.
        _merge_duplicate_relation_families(conn, source_id, target_family_id)
        # Delete source family
        conn.execute("DELETE FROM entity_families WHERE entity_family_id = ?", (source_id,))
        # Register redirect
        register_redirect(conn, source_id, target_family_id)
        merged.append(source_id)
    conn.commit()
    return {"merged": merged, "target": target_family_id}


def redirect_entity_relations(conn: sqlite3.Connection,
                              old_family_id: str, new_family_id: str) -> None:
    """Redirect all relation endpoints from old_family_id to new_family_id.

    Handles the UNIQUE constraint on (subject, object, is_directed) in
    relation_families by merging duplicate rows instead of crashing.
    """
    # Update assertions first (no unique constraint on these)
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
    # Merge relation_families to avoid UNIQUE constraint violation
    _merge_duplicate_relation_families(conn, old_family_id, new_family_id)


def delete_entity_all_versions(conn: sqlite3.Connection, family_id: str) -> int:
    conn.execute("DELETE FROM entity_mentions WHERE entity_family_id = ?", (family_id,))
    cnt = conn.execute(
        "SELECT COUNT(*) FROM entity_observations WHERE entity_family_id = ?",
        (family_id,),
    ).fetchone()[0]
    conn.execute("DELETE FROM entity_observations WHERE entity_family_id = ?", (family_id,))
    conn.execute("DELETE FROM entity_families WHERE entity_family_id = ?", (family_id,))
    return cnt


def dedup_merge_batch(conn: sqlite3.Connection,
                      pairs: List[Tuple[str, str]]) -> int:
    total = 0
    for old_fid, new_fid in pairs:
        redirect_entity_relations(conn, old_fid, new_fid)
        delete_entity_all_versions(conn, old_fid)
        register_redirect(conn, old_fid, new_fid)
        total += 1
    return total
