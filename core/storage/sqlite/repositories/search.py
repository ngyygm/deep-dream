"""FTS search, embedding search, and graph_edges queries."""

import re
import logging

from ..helpers import _time_bounds_sql

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


def _cjk_prefix_range(query: str):
    """把查询转成可走 idx_entityfam_name 的字典序前缀区间 (lower, upper)。

    SQLite 的 ``LIKE 'xx%'`` 在绑定参数 + 默认 case_sensitive_like=OFF 下
    不会用索引（EXPLAIN QUERY PLAN 实测 SCAN）；显式 ``>= / <`` 区间在
    BINARY collation 索引上是 SEARCH。upper 取"前缀末字符码点 +1"：
    短 CJK 查询的字符都在 CJK/假名/谚文区内，+1 不会越出 BMP。
    """
    if not query:
        return None
    last = query[-1]
    if ord(last) >= 0x10FFFF:
        return None
    return query, query[:-1] + chr(ord(last) + 1)


_COLS = ["episode_id", "name", "heading_path", "source_text",
         "memory_text", "document_id", "document_version_id",
         "episode_family_id", "score"]


def search_fts(conn, query: str, limit: int = 20,
               like_fallback: bool = False,
               time_after: str = None, time_before: str = None) -> list:
    """Search episodes_fts, joining to active documents/versions.

    time_after/time_before：episode.processed_at 双界过滤（闭区间，P2.8）。

    Schema 错误（如缺 episodes_fts / document_ingestion_state 表）不吞掉：
    直接向上抛 sqlite3.OperationalError 让调用方显式处理（P2.3）——
    此前静默返回空列表，用户搜任何词都 0 结果且无提示。
    """
    use_like = like_fallback or _is_short_cjk(query)
    match_query = _fts5_query(query)
    if not match_query:
        return []

    # 双界时间过滤（可选，作用于 episode 的 processed_at）。
    # FTS 主查询别名是 e，短 CJK 兜底查询别名是 ep——各自生成片段。
    time_sql, time_params = _time_bounds_sql("e.processed_at", time_after, time_before)
    ep_time_sql, ep_time_params = _time_bounds_sql("ep.processed_at", time_after, time_before)

    sql = f"""
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
          AND COALESCE(dis.state, 'active') = 'active'{time_sql}
        ORDER BY score
        LIMIT ?
    """
    rows = conn.execute(sql, (match_query,) + tuple(time_params) + (limit,)).fetchall()

    # Natural-language questions often contain one harmless term absent
    # from the corpus. Preserve precise AND semantics first, then degrade
    # to literal any-term/min-match retrieval instead of returning nothing.
    match_mode = "and"
    if not rows:
        or_query = _fts5_or_query(query)
        if or_query and or_query != match_query:
            rows = conn.execute(sql, (or_query,) + tuple(time_params) + (limit,)).fetchall()
            match_mode = "or"

    results = [dict(zip(_COLS, r)) for r in rows]
    for row in results:
        row["match_mode"] = match_mode
    existing_ids = {r["episode_id"] for r in results}

    if use_like and len(results) < limit:
        # 短 CJK 优先路径（P2.4）：canonical_name 前缀命中锚定的 episode，
        # 走 idx_entityfam_name 区间查找，比 %xx% 全表扫描精准。
        rng = _cjk_prefix_range(query)
        if rng:
            prefix_rows = conn.execute(f"""
                SELECT ep.episode_id, ep.name, ep.heading_path,
                       ep.source_text, ep.memory_text,
                       ep.document_id, ep.document_version_id,
                       ep.episode_family_id, 0.16 AS score
                FROM entity_families ef
                JOIN entity_mentions em ON em.entity_family_id = ef.entity_family_id
                JOIN episodes ep ON ep.episode_id = em.episode_id AND ep.status = 'active'
                JOIN documents d ON d.document_id = ep.document_id AND d.status = 'active'
                LEFT JOIN document_ingestion_state dis ON dis.document_id = d.document_id
                JOIN document_versions dv
                  ON dv.document_id = ep.document_id
                 AND dv.document_version_id = ep.document_version_id
                 AND dv.status = 'active'
                WHERE ef.canonical_name >= ? AND ef.canonical_name < ?
                  AND COALESCE(dis.state, 'active') = 'active'{ep_time_sql}
                ORDER BY ep.processed_at DESC, ep.episode_id
                LIMIT ?
            """, tuple(rng) + tuple(ep_time_params) + (limit,)).fetchall()
            for r in prefix_rows:
                d = dict(zip(_COLS, r))
                if d["episode_id"] not in existing_ids:
                    d["match_mode"] = "entity_prefix"
                    results.append(d)
                    existing_ids.add(d["episode_id"])

    if use_like and len(results) < limit:
        # 最后兜底：%xx% 全表 LIKE。带 match_mode='like' 且按
        # processed_at DESC, episode_id 确定排序（此前行序任意、无标记）。
        like_pattern = f"%{query}%"
        like_rows = conn.execute(f"""
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
              AND (ep.source_text LIKE ? OR ep.memory_text LIKE ? OR ep.name LIKE ?){ep_time_sql}
            ORDER BY ep.processed_at DESC, ep.episode_id
            LIMIT ?
        """, (like_pattern, like_pattern, like_pattern) + tuple(ep_time_params) + (limit,)).fetchall()

        for r in like_rows:
            d = dict(zip(_COLS, r))
            if d["episode_id"] not in existing_ids:
                d["match_mode"] = "like"
                results.append(d)
                existing_ids.add(d["episode_id"])

    return results[:limit]


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
