"""Document and version lifecycle repository."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def insert_document(conn, document_id: str, title: str, managed_path: str,
                    source_mode: str = "managed", absolute_path: str = "",
                    vault_root: str = "", relative_path: str = "",
                    created_at: str = "", updated_at: str = "") -> None:
    conn.execute(
        """INSERT INTO documents
           (document_id, title, source_mode, managed_path, absolute_path,
            vault_root, relative_path, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
           ON CONFLICT(document_id) DO UPDATE SET
             title = excluded.title,
             source_mode = excluded.source_mode,
             managed_path = excluded.managed_path,
             absolute_path = excluded.absolute_path,
             vault_root = COALESCE(excluded.vault_root, documents.vault_root),
             relative_path = excluded.relative_path,
             updated_at = excluded.updated_at""",
        (document_id, title, source_mode, managed_path, absolute_path or None,
         vault_root or None, relative_path, created_at, updated_at),
    )


def get_document(conn, document_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM documents WHERE document_id = ?", (document_id,)
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM documents LIMIT 0").description]
    return dict(zip(cols, row))


def soft_delete_document(conn, document_id: str, updated_at: str = "") -> None:
    conn.execute(
        "UPDATE documents SET status = 'deleted', updated_at = ? WHERE document_id = ?",
        (updated_at, document_id),
    )


def update_current_version(conn, document_id: str, version_id: str,
                           updated_at: str = "") -> None:
    conn.execute(
        "UPDATE documents SET current_version_id = ?, updated_at = ? WHERE document_id = ?",
        (version_id, updated_at, document_id),
    )


def insert_document_version(conn, document_version_id: str, document_id: str,
                            content_hash: str, version_content_path: str = "",
                            title: str = "", frontmatter_json: str = "{}",
                            tags_json: str = "[]", aliases_json: str = "[]",
                            char_count: int = 0, line_count: int = 0,
                            byte_size: int = 0, mtime: str = "",
                            processed_at: str = "") -> None:
    conn.execute(
        """INSERT INTO document_versions
           (document_version_id, document_id, content_hash, version_content_path,
            title, frontmatter_json, tags_json, aliases_json,
            char_count, line_count, byte_size, mtime,
            status, processed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)""",
        (document_version_id, document_id, content_hash, version_content_path,
         title, frontmatter_json, tags_json, aliases_json,
         char_count, line_count, byte_size, mtime, processed_at),
    )


def get_active_version(conn, document_id: str) -> Optional[dict]:
    row = conn.execute(
        """SELECT * FROM document_versions
           WHERE document_id = ? AND status = 'active'""",
        (document_id,),
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM document_versions LIMIT 0").description]
    return dict(zip(cols, row))


def get_version_by_hash(conn, document_id: str, content_hash: str) -> Optional[dict]:
    row = conn.execute(
        """SELECT * FROM document_versions
           WHERE document_id = ? AND content_hash = ?""",
        (document_id, content_hash),
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM document_versions LIMIT 0").description]
    return dict(zip(cols, row))


def supersede_version(conn, document_version_id: str) -> None:
    conn.execute(
        "UPDATE document_versions SET status = 'superseded' WHERE document_version_id = ? AND status = 'active'",
        (document_version_id,),
    )


def reactivate_version(conn, document_version_id: str) -> None:
    conn.execute(
        "UPDATE document_versions SET status = 'active' WHERE document_version_id = ?",
        (document_version_id,),
    )


def supersede_active_version_cascade(conn, document_id: str) -> list:
    """Supersede current active version + downstream episodes/observations/assertions.

    Returns list of superseded episode IDs (for FTS cleanup).
    """
    ver = get_active_version(conn, document_id)
    if ver is None:
        return []

    ver_id = ver["document_version_id"]
    conn.execute(
        "UPDATE document_versions SET status = 'superseded' WHERE document_version_id = ? AND status = 'active'",
        (ver_id,),
    )

    ep_ids = _supersede_episodes_by_version(conn, ver_id)
    if ep_ids:
        placeholders = ",".join("?" for _ in ep_ids)
        conn.execute(
            f"UPDATE entity_observations SET status = 'superseded' WHERE episode_id IN ({placeholders}) AND status = 'active'",
            ep_ids,
        )
        conn.execute(
            f"UPDATE relation_assertions SET status = 'superseded' WHERE episode_id IN ({placeholders}) AND status = 'active'",
            ep_ids,
        )

    return ep_ids


def _supersede_episodes_by_version(conn, document_version_id: str) -> list:
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


def list_documents(conn, status: str = "active", limit: int = 100,
                   offset: int = 0) -> list:
    cols = [d[0] for d in conn.execute("SELECT * FROM documents LIMIT 0").description]
    rows = conn.execute(
        """SELECT * FROM documents
           WHERE status = ?
           ORDER BY updated_at DESC
           LIMIT ? OFFSET ?""",
        (status, limit, offset),
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def get_document_stats(conn, document_id: str) -> Optional[dict]:
    doc = get_document(conn, document_id)
    if doc is None:
        return None

    ver_count = conn.execute(
        "SELECT COUNT(*) FROM document_versions WHERE document_id = ?",
        (document_id,),
    ).fetchone()[0]

    active_ver = get_active_version(conn, document_id)
    ep_count = 0
    if active_ver:
        ep_count = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE document_version_id = ? AND status = 'active'",
            (active_ver["document_version_id"],),
        ).fetchone()[0]

    return {
        **doc,
        "version_count": ver_count,
        "episode_count": ep_count,
    }


# ── Document links ──────────────────────────────────

def delete_document_links_by_version(conn, from_document_version_id: str) -> int:
    cur = conn.execute(
        "DELETE FROM document_links WHERE from_document_version_id = ?",
        (from_document_version_id,),
    )
    return cur.rowcount


def insert_document_link(conn, link_id: str, from_document_id: str,
                         to_document_id: str, from_document_version_id: str,
                         from_episode_id: str = "", link_text: str = "",
                         link_target: str = "", line_start: int = 0,
                         line_end: int = 0, created_at: str = "") -> None:
    conn.execute(
        """INSERT INTO document_links
           (link_id, from_document_id, to_document_id, from_document_version_id,
            from_episode_id, link_text, link_target, line_start, line_end, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (link_id, from_document_id, to_document_id, from_document_version_id,
         from_episode_id or None, link_text, link_target,
         line_start, line_end, created_at),
    )
