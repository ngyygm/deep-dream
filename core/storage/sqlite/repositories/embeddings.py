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


def search_episode_embeddings(conn, query_vector: bytes,
                              embedding_model: str,
                              limit: int = 10) -> list:
    """Search episode embeddings, filtered to active documents."""
    # SQLite doesn't have native vector search; this returns candidate rows
    # for Python-side cosine similarity. A proper vector index (e.g. sqlite-vec)
    # would be used in production.
    rows = conn.execute("""
        SELECT e.embedding_id, e.owner_id, e.text_hash, e.vector,
               ep.document_id, ep.episode_family_id
        FROM embeddings e
        JOIN episodes ep ON ep.episode_id = e.owner_id AND ep.status = 'active'
        JOIN documents d ON d.document_id = ep.document_id AND d.status = 'active'
        JOIN document_versions dv
          ON dv.document_id = ep.document_id
         AND dv.document_version_id = ep.document_version_id
         AND dv.status = 'active'
        WHERE e.owner_type = 'episode'
          AND e.embedding_model = ?
        ORDER BY e.created_at DESC
        LIMIT ?
    """, (embedding_model, limit * 3)).fetchall()

    results = []
    for row in rows:
        results.append({
            "embedding_id": row[0],
            "episode_id": row[1],
            "text_hash": row[2],
            "document_id": row[4],
            "episode_family_id": row[5],
        })
    return results[:limit]


def search_entity_embeddings(conn, query_vector: bytes,
                             embedding_model: str,
                             limit: int = 10) -> list:
    """Search entity observation embeddings, filtered to active documents."""
    rows = conn.execute("""
        SELECT e.embedding_id, e.owner_id, eo.entity_family_id, eo.name
        FROM embeddings e
        JOIN entity_observations eo ON eo.entity_id = e.owner_id AND eo.status = 'active'
        JOIN episodes ep ON ep.episode_id = eo.episode_id AND ep.status = 'active'
        JOIN documents d ON d.document_id = ep.document_id AND d.status = 'active'
        WHERE e.owner_type = 'entity_obs'
          AND e.embedding_model = ?
        LIMIT ?
    """, (embedding_model, limit)).fetchall()

    return [{"embedding_id": r[0], "entity_id": r[1],
             "entity_family_id": r[2], "name": r[3]} for r in rows]


def vacuum_orphaned(conn) -> int:
    """Delete embeddings whose owner does not exist. Returns count."""
    owner_tables = {
        "episode": ("episodes", "episode_id"),
        "entity_obs": ("entity_observations", "entity_id"),
        "relation_assert": ("relation_assertions", "relation_id"),
        "entity_family": ("entity_families", "entity_family_id"),
        "document_version": ("document_versions", "document_version_id"),
    }
    total = 0
    for otype, (table, pk) in owner_tables.items():
        cur = conn.execute(f"""
            DELETE FROM embeddings
            WHERE owner_type = ?
              AND owner_id NOT IN (SELECT {pk} FROM {table})
        """, (otype,))
        total += cur.rowcount
    conn.commit()
    return total


def vacuum_deleted_documents(conn) -> int:
    """Delete embeddings linked to deleted documents. Returns count."""
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
        ids = [r[0] for r in conn.execute(join_sql).fetchall()]
        if ids:
            ph = ",".join("?" for _ in ids)
            cur = conn.execute(f"DELETE FROM embeddings WHERE embedding_id IN ({ph})", ids)
            total += cur.rowcount
    conn.commit()
    return total


def vacuum_inactive(conn, dry_run: bool = False) -> int:
    """Delete embeddings for superseded/stale owners. Returns count."""
    if dry_run:
        count = conn.execute("""
            SELECT COUNT(*) FROM embeddings e
            JOIN episodes ep ON ep.episode_id = e.owner_id
            WHERE e.owner_type = 'episode' AND ep.status IN ('superseded', 'stale')
        """).fetchone()[0]
        return count

    total = 0
    for otype, join_sql in [
        ("episode", """
            SELECT e.embedding_id FROM embeddings e
            JOIN episodes ep ON ep.episode_id = e.owner_id
            WHERE e.owner_type = 'episode' AND ep.status IN ('superseded', 'stale')
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
