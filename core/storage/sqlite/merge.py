"""Entity merge and redirect operations for V1.5 schema."""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, Iterable, List, Tuple

from .helpers import now_utc_str


logger = logging.getLogger(__name__)

_MAX_REDIRECT_DEPTH = 16


def _retire_assertion_collisions(
    conn: sqlite3.Connection,
    relation_family_ids: List[str],
    final_family_id: str,
    source_entity_id: str,
    target_entity_id: str,
) -> None:
    """Supersede active assertions that collapse to the same final key.

    The active uniqueness constraint includes the episode, relation family and
    endpoint families.  Redirecting either an endpoint or a duplicate relation
    family can therefore make previously distinct rows collide before the SQL
    UPDATE finishes.  Select a deterministic survivor first and retire the rest
    before changing any key columns.
    """
    if not relation_family_ids:
        return
    placeholders = ",".join("?" for _ in relation_family_ids)
    rows = conn.execute(
        f"SELECT relation_id, relation_family_id, episode_id, "
        f"subject_entity_family_id, object_entity_family_id, processed_at "
        f"FROM relation_assertions WHERE status='active' "
        f"AND relation_family_id IN ({placeholders}) "
        f"ORDER BY CASE WHEN relation_family_id=? THEN 0 ELSE 1 END, "
        f"processed_at DESC, rowid DESC",
        [*relation_family_ids, final_family_id],
    ).fetchall()
    seen = set()
    retire = []
    for relation_id, _family_id, episode_id, subject_id, object_id, _processed_at in rows:
        # NULL episode IDs do not collide under SQLite UNIQUE semantics.
        if episode_id is None:
            continue
        final_subject = target_entity_id if subject_id == source_entity_id else subject_id
        final_object = target_entity_id if object_id == source_entity_id else object_id
        key = (episode_id, final_family_id, final_subject, final_object)
        if key in seen:
            retire.append(relation_id)
        else:
            seen.add(key)
    if retire:
        retire_ph = ",".join("?" for _ in retire)
        conn.execute(
            f"UPDATE relation_assertions SET status='superseded' "
            f"WHERE relation_id IN ({retire_ph})",
            retire,
        )


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
            assertion_ids = [row[0] for row in conn.execute(
                "SELECT relation_id FROM relation_assertions WHERE relation_family_id = ?",
                (rel_fid,),
            ).fetchall()]
            if assertion_ids:
                aph = ",".join("?" for _ in assertion_ids)
                conn.execute(
                    f"DELETE FROM embeddings WHERE owner_type = 'relation_assert' AND owner_id IN ({aph})",
                    assertion_ids,
                )
            conn.execute("DELETE FROM relation_assertions WHERE relation_family_id = ?",
                         (rel_fid,))
            conn.execute("DELETE FROM relation_families WHERE relation_family_id = ?",
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
            _retire_assertion_collisions(
                conn, [survivor_fid, rel_fid], survivor_fid,
                source_id, target_id,
            )
            # Normalize endpoints while the two families are still distinct;
            # collision rows have already left the active unique index.
            conn.execute(
                "UPDATE relation_assertions SET subject_entity_family_id = ? "
                "WHERE relation_family_id IN (?, ?) AND subject_entity_family_id = ?",
                (target_id, survivor_fid, rel_fid, source_id),
            )
            conn.execute(
                "UPDATE relation_assertions SET object_entity_family_id = ? "
                "WHERE relation_family_id IN (?, ?) AND object_entity_family_id = ?",
                (target_id, survivor_fid, rel_fid, source_id),
            )
            # Reassign all assertions from the duplicate to the survivor
            conn.execute(
                "UPDATE relation_assertions SET relation_family_id = ? "
                "WHERE relation_family_id = ?",
                (survivor_fid, rel_fid),
            )
            conn.execute(
                "UPDATE relation_mentions SET relation_family_id = ? "
                "WHERE relation_family_id = ?",
                (survivor_fid, rel_fid),
            )
            # Delete the duplicate relation_family only after its assertions
            # have been reassigned; FK enforcement otherwise rejects the
            # parent delete and leaves the merge half-applied.
            conn.execute("DELETE FROM relation_families WHERE relation_family_id = ?",
                         (rel_fid,))
        else:
            _retire_assertion_collisions(
                conn, [rel_fid], rel_fid, source_id, target_id,
            )
            conn.execute(
                "UPDATE relation_assertions SET subject_entity_family_id = ? "
                "WHERE relation_family_id = ? AND subject_entity_family_id = ?",
                (target_id, rel_fid, source_id),
            )
            conn.execute(
                "UPDATE relation_assertions SET object_entity_family_id = ? "
                "WHERE relation_family_id = ? AND object_entity_family_id = ?",
                (target_id, rel_fid, source_id),
            )
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
        # If target and source were both observed in one episode, moving both
        # active rows to the target violates idx_entityobs_unique_active.  Keep
        # the target observation active and retain the source observation as
        # superseded history before changing its family key.
        conn.execute(
            "UPDATE entity_observations AS src SET status='superseded' "
            "WHERE src.entity_family_id=? AND src.status='active' "
            "AND src.episode_id IS NOT NULL AND EXISTS ("
            "SELECT 1 FROM entity_observations AS dst "
            "WHERE dst.entity_family_id=? AND dst.episode_id=src.episode_id "
            "AND dst.status='active')",
            (source_id, target_family_id),
        )
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
        # Merge relation families and assertions together.  The helper retires
        # any active assertion collisions before changing endpoint/family keys.
        # Without this, a UNIQUE(subject, object, is_directed) violation
        # crashes the merge and leaves the DB in an inconsistent state.
        _merge_duplicate_relation_families(conn, source_id, target_family_id)
        # Delete source family
        conn.execute("DELETE FROM entity_families WHERE entity_family_id = ?", (source_id,))
        # Register redirect
        register_redirect(conn, source_id, target_family_id)
        merged.append(source_id)
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
    relation_fids = [row[0] for row in conn.execute(
        "SELECT relation_family_id FROM relation_families "
        "WHERE subject_entity_family_id = ? OR object_entity_family_id = ?",
        (family_id, family_id),
    ).fetchall()]
    if relation_fids:
        placeholders = ",".join("?" for _ in relation_fids)
        relation_ids = [row[0] for row in conn.execute(
            f"SELECT relation_id FROM relation_assertions WHERE relation_family_id IN ({placeholders})",
            relation_fids,
        ).fetchall()]
        if relation_ids:
            rph = ",".join("?" for _ in relation_ids)
            conn.execute(
                f"DELETE FROM embeddings WHERE owner_type = 'relation_assert' AND owner_id IN ({rph})",
                relation_ids,
            )
            conn.execute(f"DELETE FROM relation_assertions WHERE relation_id IN ({rph})", relation_ids)
        conn.execute(f"DELETE FROM relation_families WHERE relation_family_id IN ({placeholders})", relation_fids)

    entity_ids = [row[0] for row in conn.execute(
        "SELECT entity_id FROM entity_observations WHERE entity_family_id = ?", (family_id,)
    ).fetchall()]
    conn.execute("DELETE FROM entity_mentions WHERE entity_family_id = ?", (family_id,))
    cnt = conn.execute(
        "SELECT COUNT(*) FROM entity_observations WHERE entity_family_id = ?",
        (family_id,),
    ).fetchone()[0]
    if entity_ids:
        eph = ",".join("?" for _ in entity_ids)
        # 删除 observations 前先清掉指向它们的外键行（与 document 路径的
        # _delete_observations_safe 同款）。redirect_entity_relations 只按
        # family id 重指断言端点，其他 relation family 中断言的
        # subject/object_entity_id 观察锚点（对齐可把别的 family 的观察挂
        # 上去）以及锚定本 family 观察的跨 episode entity_mentions 仍指向
        # 待删观察——PRAGMA foreign_keys=ON 下直接 DELETE 会炸 FK、整个
        # 合并批次回滚（cross-window dedup 静默丢失全部工作）。
        # relation_mentions 对 relation_assertions 有 ON DELETE CASCADE。
        conn.execute(f"DELETE FROM entity_mentions WHERE entity_id IN ({eph})", entity_ids)
        cross_assert_ids = [row[0] for row in conn.execute(
            f"SELECT relation_id FROM relation_assertions "
            f"WHERE subject_entity_id IN ({eph}) OR object_entity_id IN ({eph})",
            entity_ids + entity_ids,
        ).fetchall()]
        if cross_assert_ids:
            cph = ",".join("?" for _ in cross_assert_ids)
            conn.execute(f"DELETE FROM relation_assertions WHERE relation_id IN ({cph})", cross_assert_ids)
            conn.execute(
                f"DELETE FROM embeddings WHERE owner_type = 'relation_assert' AND owner_id IN ({cph})",
                cross_assert_ids,
            )
        conn.execute(
            f"DELETE FROM embeddings WHERE owner_type = 'entity_obs' AND owner_id IN ({eph})",
            entity_ids,
        )
    conn.execute(
        "DELETE FROM embeddings WHERE owner_type = 'entity_family' AND owner_id = ?",
        (family_id,),
    )
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
