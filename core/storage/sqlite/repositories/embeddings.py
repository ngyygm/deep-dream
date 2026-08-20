"""Embedding write, read, and vacuum repository."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def insert_embedding(conn, embedding_id: str, owner_type: str,
                     owner_id: str, text_kind: str, text_hash: str,
                     embedding_model: str, dimensions: int,
                     vector: bytes, run_id: str = "",
                     created_at: str = "") -> None:
    conn.execute(
        """INSERT OR REPLACE INTO embeddings
           (embedding_id, owner_type, owner_id, text_kind, text_hash,
            embedding_model, dimensions, vector, run_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (embedding_id, owner_type, owner_id, text_kind, text_hash,
         embedding_model, dimensions, vector, run_id, created_at),
    )


def get_embedding(conn, owner_type: str, owner_id: str, text_kind: str,
                  embedding_model: str, text_hash: str) -> Optional[bytes]:
    row = conn.execute(
        """SELECT vector FROM embeddings
           WHERE owner_type = ? AND owner_id = ? AND text_kind = ?
             AND embedding_model = ? AND text_hash = ?""",
        (owner_type, owner_id, text_kind, embedding_model, text_hash),
    ).fetchone()
    return row[0] if row else None


def search_entity_embeddings(conn, query_vector: bytes,
                             embedding_model: str,
                             limit: int = 10) -> list:
    """Search entity observation embeddings, filtered to active documents."""
    rows = conn.execute("""
        SELECT e.embedding_id, e.owner_id, eo.entity_family_id, eo.name, e.vector
        FROM embeddings e
        JOIN entity_observations eo ON eo.entity_id = e.owner_id AND eo.status = 'active'
        JOIN episodes ep ON ep.episode_id = eo.episode_id AND ep.status = 'active'
        JOIN documents d ON d.document_id = ep.document_id AND d.status = 'active'
        JOIN document_versions dv
          ON dv.document_id = ep.document_id
         AND dv.document_version_id = ep.document_version_id
         AND dv.status = 'active'
        WHERE e.owner_type = 'entity_obs'
          AND e.embedding_model = ?
        ORDER BY e.created_at DESC, e.embedding_id DESC
        LIMIT ?
    """, (embedding_model, limit)).fetchall()

    return [{"embedding_id": r[0], "owner_id": r[1], "entity_id": r[1],
             "entity_family_id": r[2], "name": r[3], "vector": r[4]} for r in rows]


def search_relation_embeddings(conn, query_vector: bytes,
                               embedding_model: str,
                               limit: int = 10) -> list:
    """Search relation assertion embeddings, filtered to active documents."""
    rows = conn.execute("""
        SELECT e.embedding_id, e.owner_id, ra.relation_family_id,
               rf.canonical_content, e.vector
        FROM embeddings e
        JOIN relation_assertions ra ON ra.relation_id = e.owner_id AND ra.status = 'active'
        JOIN relation_families rf ON rf.relation_family_id = ra.relation_family_id
        JOIN episodes ep ON ep.episode_id = ra.episode_id AND ep.status = 'active'
        JOIN documents d ON d.document_id = ep.document_id AND d.status = 'active'
        JOIN document_versions dv
          ON dv.document_id = ep.document_id
         AND dv.document_version_id = ep.document_version_id
         AND dv.status = 'active'
        WHERE e.owner_type = 'relation_assert'
          AND e.embedding_model = ?
        ORDER BY e.created_at DESC, e.embedding_id DESC
        LIMIT ?
    """, (embedding_model, limit)).fetchall()

    return [{"embedding_id": r[0], "owner_id": r[1], "relation_id": r[1],
             "relation_family_id": r[2],
             "canonical_content": r[3], "vector": r[4]} for r in rows]


def vacuum_orphaned(conn, dry_run: bool = False) -> int:
    """Delete (or count) embeddings whose owner does not exist. Returns count."""
    owner_tables = {
        "episode": ("episodes", "episode_id"),
        "entity_obs": ("entity_observations", "entity_id"),
        "relation_assert": ("relation_assertions", "relation_id"),
        "entity_family": ("entity_families", "entity_family_id"),
        "document_version": ("document_versions", "document_version_id"),
    }
    total = 0
    for otype, (table, pk) in owner_tables.items():
        if dry_run:
            count = conn.execute(f"""
                SELECT COUNT(*) FROM embeddings
                WHERE owner_type = ?
                  AND owner_id NOT IN (SELECT {pk} FROM {table})
            """, (otype,)).fetchone()[0]
            total += count
        else:
            cur = conn.execute(f"""
                DELETE FROM embeddings
                WHERE owner_type = ?
                  AND owner_id NOT IN (SELECT {pk} FROM {table})
            """, (otype,))
            total += cur.rowcount
    if not dry_run:
        conn.commit()
    return total


def vacuum_deleted_documents(conn, dry_run: bool = False) -> int:
    """Delete (or count) embeddings linked to deleted documents. Returns count."""
    # Episode/observation/assertion embeddings join through to documents
    total = 0
    for otype, join_sql in [
        ("episode", """
            SELECT e.embedding_id FROM embeddings e
            JOIN episodes ep ON ep.episode_id = e.owner_id
            JOIN documents d ON d.document_id = ep.document_id
            WHERE e.owner_type = 'episode' AND d.status = 'deleted'
        """),
        ("entity_obs", """
            SELECT e.embedding_id FROM embeddings e
            JOIN entity_observations eo ON eo.entity_id = e.owner_id
            JOIN episodes ep ON ep.episode_id = eo.episode_id
            JOIN documents d ON d.document_id = ep.document_id
            WHERE e.owner_type = 'entity_obs' AND d.status = 'deleted'
        """),
        ("relation_assert", """
            SELECT e.embedding_id FROM embeddings e
            JOIN relation_assertions ra ON ra.relation_id = e.owner_id
            JOIN episodes ep ON ep.episode_id = ra.episode_id
            JOIN documents d ON d.document_id = ep.document_id
            WHERE e.owner_type = 'relation_assert' AND d.status = 'deleted'
        """),
        ("document_version", """
            SELECT e.embedding_id FROM embeddings e
            JOIN document_versions dv ON dv.document_version_id = e.owner_id
            JOIN documents d ON d.document_id = dv.document_id
            WHERE e.owner_type = 'document_version' AND d.status = 'deleted'
        """),
    ]:
        if dry_run:
            count_sql = join_sql.replace(
                "SELECT e.embedding_id", "SELECT COUNT(*)", 1
            )
            count = conn.execute(count_sql).fetchone()[0]
            total += count
        else:
            ids = [r[0] for r in conn.execute(join_sql).fetchall()]
            if ids:
                ph = ",".join("?" for _ in ids)
                cur = conn.execute(f"DELETE FROM embeddings WHERE embedding_id IN ({ph})", ids)
                total += cur.rowcount
    if not dry_run:
        conn.commit()
    return total


def vacuum_inactive(conn, dry_run: bool = False) -> int:
    """Delete embeddings for superseded/stale owners. Returns count.

    Cleans:
    - Episode embeddings for superseded/stale episodes
    - entity_obs embeddings whose observation is superseded
    - entity_family embeddings whose *only* observations are superseded
    - relation_assert embeddings whose assertion is superseded
    """
    if dry_run:
        count = conn.execute("""
            SELECT COUNT(*) FROM embeddings e
            JOIN episodes ep ON ep.episode_id = e.owner_id
            WHERE e.owner_type = 'episode' AND ep.status IN ('superseded', 'stale')
        """).fetchone()[0]
        # entity_obs embeddings pointing to superseded observations
        count += conn.execute("""
            SELECT COUNT(*) FROM embeddings e
            JOIN entity_observations eo ON eo.entity_id = e.owner_id
            WHERE e.owner_type = 'entity_obs' AND eo.status = 'superseded'
        """).fetchone()[0]
        # entity_family embeddings with no active observations
        count += conn.execute("""
            SELECT COUNT(*) FROM embeddings e
            WHERE e.owner_type = 'entity_family'
              AND NOT EXISTS (
                SELECT 1 FROM entity_observations eo
                WHERE eo.entity_family_id = e.owner_id AND eo.status = 'active'
              )
        """).fetchone()[0]
        # relation_assert embeddings pointing to superseded assertions
        count += conn.execute("""
            SELECT COUNT(*) FROM embeddings e
            JOIN relation_assertions ra ON ra.relation_id = e.owner_id
            WHERE e.owner_type = 'relation_assert' AND ra.status = 'superseded'
        """).fetchone()[0]
        return count

    total = 0
    for otype, join_sql in [
        ("episode", """
            SELECT e.embedding_id FROM embeddings e
            JOIN episodes ep ON ep.episode_id = e.owner_id
            WHERE e.owner_type = 'episode' AND ep.status IN ('superseded', 'stale')
        """),
        ("entity_obs (superseded)", """
            SELECT e.embedding_id FROM embeddings e
            JOIN entity_observations eo ON eo.entity_id = e.owner_id
            WHERE e.owner_type = 'entity_obs' AND eo.status = 'superseded'
        """),
        ("entity_family (no active obs)", """
            SELECT e.embedding_id FROM embeddings e
            WHERE e.owner_type = 'entity_family'
              AND NOT EXISTS (
                SELECT 1 FROM entity_observations eo
                WHERE eo.entity_family_id = e.owner_id AND eo.status = 'active'
              )
        """),
        ("relation_assert (superseded)", """
            SELECT e.embedding_id FROM embeddings e
            JOIN relation_assertions ra ON ra.relation_id = e.owner_id
            WHERE e.owner_type = 'relation_assert' AND ra.status = 'superseded'
        """),
    ]:
        ids = [r[0] for r in conn.execute(join_sql).fetchall()]
        if ids:
            ph = ",".join("?" for _ in ids)
            cur = conn.execute(f"DELETE FROM embeddings WHERE embedding_id IN ({ph})", ids)
            total += cur.rowcount
    conn.commit()
    return total


def count_embeddings(conn, owner_type: str = "") -> int:
    if owner_type:
        return conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE owner_type = ?",
            (owner_type,),
        ).fetchone()[0]
    return conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
