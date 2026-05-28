"""FTS search, embedding search, and graph_edges queries."""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _is_short_cjk(query: str) -> bool:
    """Check if query is short CJK (< 3 CJK chars)."""
    cjk_chars = len(re.findall(r'[一-鿿぀-ゟ゠-ヿ가-힯]', query))
    return 0 < cjk_chars < 3


def search_fts(conn, query: str, limit: int = 20,
               like_fallback: bool = False) -> list:
    """Search episodes_fts, joining to active documents/versions."""
    use_like = like_fallback or _is_short_cjk(query)

    results = []
    try:
        rows = conn.execute("""
            SELECT episodes_fts.episode_id,
                   episodes_fts.name,
                   episodes_fts.heading_path,
                   episodes_fts.source_text,
                   episodes_fts.memory_text,
                   e.document_id,
                   e.document_version_id,
                   e.episode_family_id,
                   bm25(episodes_fts) AS score
            FROM episodes_fts
            JOIN episodes e
              ON e.episode_id = episodes_fts.episode_id
             AND e.status = 'active'
            JOIN documents d
              ON d.document_id = e.document_id
             AND d.status = 'active'
            JOIN document_versions dv
              ON dv.document_id = e.document_id
             AND dv.document_version_id = e.document_version_id
             AND dv.status = 'active'
            WHERE episodes_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """, (query, limit)).fetchall()

        cols = ["episode_id", "name", "heading_path", "source_text",
                "memory_text", "document_id", "document_version_id",
                "episode_family_id", "score"]
        results = [dict(zip(cols, r)) for r in rows]
    except Exception as exc:
        logger.warning("FTS MATCH failed for query=%r: %s", query, exc)

    if use_like and len(results) < limit:
        existing_ids = {r["episode_id"] for r in results}
        like_pattern = f"%{query}%"
        like_rows = conn.execute("""
            SELECT ep.episode_id, ep.name, ep.heading_path,
                   ep.source_text, ep.memory_text,
                   ep.document_id, ep.document_version_id,
                   ep.episode_family_id, 0.16 AS score
            FROM episodes ep
            JOIN documents d ON d.document_id = ep.document_id AND d.status = 'active'
            JOIN document_versions dv
              ON dv.document_id = ep.document_id
             AND dv.document_version_id = ep.document_version_id
             AND dv.status = 'active'
            WHERE ep.status = 'active'
              AND (ep.source_text LIKE ? OR ep.memory_text LIKE ? OR ep.name LIKE ?)
            LIMIT ?
        """, (like_pattern, like_pattern, like_pattern, limit)).fetchall()

        cols = ["episode_id", "name", "heading_path", "source_text",
                "memory_text", "document_id", "document_version_id",
                "episode_family_id", "score"]
        for r in like_rows:
            d = dict(zip(cols, r))
            if d["episode_id"] not in existing_ids:
                results.append(d)
                existing_ids.add(d["episode_id"])

    return results[:limit]


def search_fts_by_document(conn, document_id: str, query: str,
                           limit: int = 20) -> list:
    rows = conn.execute("""
        SELECT episodes_fts.episode_id,
               episodes_fts.name,
               episodes_fts.heading_path,
               episodes_fts.source_text,
               episodes_fts.memory_text,
               e.episode_family_id,
               bm25(episodes_fts) AS score
        FROM episodes_fts
        JOIN episodes e
          ON e.episode_id = episodes_fts.episode_id
         AND e.status = 'active'
         AND e.document_id = ?
        JOIN document_versions dv
          ON dv.document_id = e.document_id
         AND dv.document_version_id = e.document_version_id
         AND dv.status = 'active'
        WHERE episodes_fts MATCH ?
        ORDER BY score
        LIMIT ?
    """, (document_id, query, limit)).fetchall()

    cols = ["episode_id", "name", "heading_path", "source_text",
            "memory_text", "episode_family_id", "score"]
    return [dict(zip(cols, r)) for r in rows]


def get_graph_edges(conn, source_id: str = "",
                    edge_type: str = "",
                    limit: int = 100) -> list:
    """Query graph_edges view, optionally filtered."""
    conditions = []
    params = []
    if source_id:
        conditions.append("source_id = ?")
        params.append(source_id)
    if edge_type:
        conditions.append("edge_type = ?")
        params.append(edge_type)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = conn.execute(
        f"SELECT * FROM graph_edges {where} LIMIT ?",
        params + [limit],
    ).fetchall()

    cols = ["edge_type", "source_id", "target_id", "target_family_id"]
    return [dict(zip(cols, r)) for r in rows]


def get_graph_neighbors(conn, family_id: str, limit: int = 50) -> list:
    """Get neighbor concepts from graph_edges for a given family.

    Handles both family IDs and observation IDs in source_id by
    also matching via entity_observations for the source side.
    """
    rows = conn.execute("""
        SELECT ge.edge_type, ge.source_id, ge.target_id, ge.target_family_id,
               COALESCE(eo.entity_family_id, '') AS source_family_id
        FROM graph_edges ge
        LEFT JOIN entity_observations eo ON eo.entity_id = ge.source_id AND eo.status = 'active'
        WHERE ge.target_family_id = ?
           OR ge.source_id = ?
           OR eo.entity_family_id = ?
        LIMIT ?
    """, (family_id, family_id, family_id, limit)).fetchall()

    cols = ["edge_type", "source_id", "target_id", "target_family_id", "source_family_id"]
    return [dict(zip(cols, r)) for r in rows]


def get_document_graph(conn, document_id: str) -> dict:
    """Get full graph slice for a document."""
    edges = get_graph_edges(conn, source_id=document_id, limit=500)
    family_ids = set()
    for e in edges:
        if e["target_family_id"]:
            family_ids.add(e["target_family_id"])

    entities = []
    if family_ids:
        ph = ",".join("?" for _ in family_ids)
        rows = conn.execute(f"""
            SELECT ef.entity_family_id, ef.canonical_name
            FROM entity_families ef
            WHERE ef.entity_family_id IN ({ph})
        """, list(family_ids)).fetchall()
        entities = [{"entity_family_id": r[0], "name": r[1]} for r in rows]

    return {
        "document_id": document_id,
        "edges": edges,
        "entities": entities,
    }
