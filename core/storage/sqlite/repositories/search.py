"""FTS search, embedding search, and graph_edges queries."""

import re
import logging

logger = logging.getLogger(__name__)


def _is_short_cjk(query: str) -> bool:
    """Check if query is short CJK (< 3 CJK chars)."""
    cjk_chars = len(re.findall(r'[一-鿿぀-ゟ゠-ヿ가-힯]', query))
    return 0 < cjk_chars < 3


def _fts5_query(query: str) -> str:
    """Convert untrusted natural language into a literal FTS5 AND query.

    FTS5 treats punctuation such as ``-``, ``.``, ``:`` and parentheses as
    query-language operators.  Benchmark questions and agent reformulations
    routinely contain those characters (for example ``self-care`` and
    ``Dr. Seuss``), so passing the raw text to ``MATCH`` can raise an
    OperationalError.  Quoting each tokenizer-sized term preserves the
    existing all-terms semantics without exposing FTS syntax.
    """
    terms = re.findall(r"\w+", query, flags=re.UNICODE)
    return " ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def _fts5_or_query(query: str) -> str:
    """Literal FTS5 OR query used only when strict all-term search is empty."""
    terms = re.findall(r"\w+", query, flags=re.UNICODE)
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def search_fts(conn, query: str, limit: int = 20,
               like_fallback: bool = False) -> list:
    """Search episodes_fts, joining to active documents/versions."""
    use_like = like_fallback or _is_short_cjk(query)
    match_query = _fts5_query(query)
    if not match_query:
        return []

    results = []
    try:
        sql = """
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
            LEFT JOIN document_ingestion_state dis
              ON dis.document_id = d.document_id
            JOIN document_versions dv
              ON dv.document_id = e.document_id
             AND dv.document_version_id = e.document_version_id
             AND dv.status = 'active'
            WHERE episodes_fts MATCH ?
              AND COALESCE(dis.state, 'active') = 'active'
            ORDER BY score
            LIMIT ?
        """
        rows = conn.execute(sql, (match_query, limit)).fetchall()

        # Natural-language questions often contain one harmless term absent
        # from the corpus. Preserve precise AND semantics first, then degrade
        # to literal any-term/min-match retrieval instead of returning nothing.
        match_mode = "and"
        if not rows:
            or_query = _fts5_or_query(query)
            if or_query and or_query != match_query:
                rows = conn.execute(sql, (or_query, limit)).fetchall()
                match_mode = "or"

        cols = ["episode_id", "name", "heading_path", "source_text",
                "memory_text", "document_id", "document_version_id",
                "episode_family_id", "score"]
        results = [dict(zip(cols, r)) for r in rows]
        for row in results:
            row["match_mode"] = match_mode
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
            LEFT JOIN document_ingestion_state dis ON dis.document_id = d.document_id
            JOIN document_versions dv
              ON dv.document_id = ep.document_id
             AND dv.document_version_id = ep.document_version_id
             AND dv.status = 'active'
            WHERE ep.status = 'active'
              AND COALESCE(dis.state, 'active') = 'active'
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
    match_query = _fts5_query(query)
    if not match_query:
        return []
    sql = """
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
        JOIN documents d
          ON d.document_id = e.document_id
         AND d.status = 'active'
        LEFT JOIN document_ingestion_state dis
          ON dis.document_id = d.document_id
        JOIN document_versions dv
          ON dv.document_id = e.document_id
         AND dv.document_version_id = e.document_version_id
         AND dv.status = 'active'
        WHERE episodes_fts MATCH ?
          AND COALESCE(dis.state, 'active') = 'active'
        ORDER BY score
        LIMIT ?
    """
    rows = conn.execute(sql, (document_id, match_query, limit)).fetchall()
    match_mode = "and"
    if not rows:
        or_query = _fts5_or_query(query)
        if or_query and or_query != match_query:
            rows = conn.execute(sql, (document_id, or_query, limit)).fetchall()
            match_mode = "or"

    cols = ["episode_id", "name", "heading_path", "source_text",
            "memory_text", "episode_family_id", "score"]
    results = [dict(zip(cols, r)) for r in rows]
    for row in results:
        row["match_mode"] = match_mode
    return results


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

    cols = ["edge_type", "source_id", "target_id", "target_family_id", "source_family_id"]
    return [dict(zip(cols, r)) for r in rows]


def get_graph_neighbors(conn, family_id: str, limit: int = 50) -> list:
    """Get neighbor concepts from graph_edges for a given family."""
    rows = conn.execute("""
        SELECT ge.edge_type, ge.source_id, ge.target_id,
               ge.target_family_id, ge.source_family_id
        FROM graph_edges ge
        WHERE ge.target_family_id = ?
           OR ge.source_family_id = ?
        LIMIT ?
    """, (family_id, family_id, limit)).fetchall()

    cols = ["edge_type", "source_id", "target_id", "target_family_id", "source_family_id"]
    return [dict(zip(cols, r)) for r in rows]


def get_document_graph(conn, document_id: str) -> dict:
    """Get full graph slice for a document."""
    edges = get_graph_edges(conn, source_id=document_id, limit=500)

    # Collect entity family IDs from RELATES edges (source_family_id and
    # target_family_id) and MENTIONS edges (target_family_id).
    family_ids = set()
    for e in edges:
        et = e.get("edge_type", "")
        if et == "RELATES":
            if e.get("source_family_id"):
                family_ids.add(e["source_family_id"])
            if e.get("target_family_id"):
                family_ids.add(e["target_family_id"])
        elif et in ("MENTIONS", "ASSERTS"):
            if e.get("target_family_id"):
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
