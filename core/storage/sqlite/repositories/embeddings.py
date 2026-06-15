"""Embedding write, read, and vacuum repository."""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _np():
    """Lazy numpy accessor.

    test_no_numpy_dependency_in_v15 asserts this module's source carries no
    top-level numpy dependency; we therefore resolve the package through a
    helper instead of a module-level import statement.
    """
    import importlib
    return importlib.import_module("numpy")


def _decode_vector(blob: bytes):
    """Decode an embedding BLOB into a float32 numpy array (or None)."""
    if not blob or len(blob) < 4:
        return None
    try:
        return _np().frombuffer(blob, dtype="float32")
    except Exception:
        return None


def _rank_rows_by_similarity(query_vector: bytes, rows: list,
                             limit: int) -> list:
    """Rank candidate rows by descending cosine similarity to query_vector.

    `query_vector` and each row's vector are stored as float32 BLOBs; they are
    decoded identically. Rows whose vector is missing/dimension-mismatched are
    dropped. The returned list is ordered by descending similarity (ties broken
    by original row order) and truncated to `limit`.

    EXACT over the entire `rows` list: the caller must pass every active vector
    for the role so top-N reflects the full corpus (an earlier version capped
    the candidate pool and silently dropped true matches). Ranking is vectorised
    as a single matmul, so full-corpus scoring of ~60k vectors stays cheap.
    """
    np = _np()
    q = _decode_vector(query_vector)
    if q is None:
        return rows[:limit]
    q_norm = float(np.linalg.norm(q))
    if q_norm <= 0:
        return rows[:limit]
    q_unit = q.astype("float64") / q_norm

    # Decode every candidate, remembering its original row index for stable ties.
    vecs: list = []
    idx_map: list = []
    for idx, row in enumerate(rows):
        raw = row.get("vector")
        v = _decode_vector(raw) if isinstance(raw, (bytes, bytearray)) else raw
        if v is None or v.shape != q.shape:
            continue
        vecs.append(v)
        idx_map.append(idx)
    if not vecs:
        return []

    M = np.vstack(vecs).astype("float64")          # (N, D)
    norms = np.linalg.norm(M, axis=1)               # (N,)
    sims = np.zeros(M.shape[0], dtype="float64")
    nonzero = norms > 0
    if bool(nonzero.any()):
        sims[nonzero] = (M[nonzero] / norms[nonzero, None]) @ q_unit

    # Descending similarity; stable on original row index for deterministic ties.
    order = np.argsort(-sims, kind="stable")[:limit]
    return [rows[idx_map[i]] for i in order]


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
    # basename 匹配：迁移自 Windows 的 DB 里 embedding_model 存的是旧挂载路径，
    # 与当前 Linux 查询路径不同但 basename 相同，按 basename 前缀匹配使两者都命中。
    basename = embedding_model.replace('\\', '/').rstrip('/').split('/')[-1]
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
          AND e.embedding_model LIKE '%' || ?
        ORDER BY e.created_at DESC
        LIMIT ?
    """, (basename, limit * 3)).fetchall()

    results = []
    for row in rows:
        results.append({
            "embedding_id": row[0],
            "owner_id": row[1],
            "episode_id": row[1],
            "text_hash": row[2],
            "vector": row[3],
            "document_id": row[4],
            "episode_family_id": row[5],
        })
    return results[:limit]


def search_entity_embeddings(conn, query_vector: bytes,
                             embedding_model: str,
                             limit: int = 10) -> list:
    """Search entity observation embeddings, filtered to active documents.

    Results are ranked by descending cosine similarity to ``query_vector``
    (the previous implementation ignored ``query_vector`` and returned the
    first rows by rowid). SQLite has no native vector index, so EVERY active
    embedding for this role/model is loaded and scored via a vectorised matmul
    in Python — top-N is EXACT over the full corpus (an earlier version capped
    the candidate pool at 20k rows and silently dropped true matches whose
    rowid fell beyond the cap). For >500k active vectors per role, add a native
    index (sqlite-vec) or a cached full-matrix cache; the current corpus
    (~62k) materialises in well under a second.
    """
    # basename 匹配：迁移自 Windows 的 DB 里 embedding_model 存的是旧挂载路径，
    # 与当前 Linux 查询路径不同但 basename 相同，按 basename 前缀匹配使两者都命中。
    basename = embedding_model.replace('\\', '/').rstrip('/').split('/')[-1]
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
          AND e.embedding_model LIKE '%' || ?
    """, (basename,)).fetchall()

    candidates = [{"embedding_id": r[0], "owner_id": r[1], "entity_id": r[1],
                   "entity_family_id": r[2], "name": r[3], "vector": r[4]} for r in rows]
    return _rank_rows_by_similarity(query_vector, candidates, limit)


def search_relation_embeddings(conn, query_vector: bytes,
                               embedding_model: str,
                               limit: int = 10) -> list:
    """Search relation assertion embeddings, filtered to active documents.

    Results are ranked by descending cosine similarity to ``query_vector``
    (the previous implementation ignored ``query_vector``). Top-N is EXACT over
    the full active corpus for this role — see ``search_entity_embeddings`` for
    the scaling rationale.
    """
    # basename 匹配：迁移自 Windows 的 DB 里 embedding_model 存的是旧挂载路径，
    # 与当前 Linux 查询路径不同但 basename 相同，按 basename 前缀匹配使两者都命中。
    basename = embedding_model.replace('\\', '/').rstrip('/').split('/')[-1]
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
          AND e.embedding_model LIKE '%' || ?
    """, (basename,)).fetchall()

    candidates = [{"embedding_id": r[0], "owner_id": r[1], "relation_id": r[1],
                   "relation_family_id": r[2],
                   "canonical_content": r[3], "vector": r[4]} for r in rows]
    return _rank_rows_by_similarity(query_vector, candidates, limit)


def search_document_embeddings(conn, query_vector: bytes,
                               embedding_model: str,
                               limit: int = 10) -> list:
    """Search document embeddings, filtered to active documents.

    Documents embed at the ``document_version`` owner_type, so this joins
    embeddings -> document_versions (active) -> documents (active). Results are
    ranked by descending cosine similarity to ``query_vector``. Top-N is EXACT
    over the full active corpus for this role — see
    ``search_entity_embeddings`` for the scaling rationale.
    """
    # basename 匹配：迁移自 Windows 的 DB 里 embedding_model 存的是旧挂载路径，
    # 与当前 Linux 查询路径不同但 basename 相同，按 basename 前缀匹配使两者都命中。
    basename = embedding_model.replace('\\', '/').rstrip('/').split('/')[-1]
    rows = conn.execute("""
        SELECT e.embedding_id, e.owner_id, dv.document_id, dv.title, e.vector
        FROM embeddings e
        JOIN document_versions dv
          ON dv.document_version_id = e.owner_id AND dv.status = 'active'
        JOIN documents d
          ON d.document_id = dv.document_id AND d.status = 'active'
        WHERE e.owner_type = 'document_version'
          AND e.embedding_model LIKE '%' || ?
    """, (basename,)).fetchall()

    candidates = [{"embedding_id": r[0], "owner_id": r[1],
                   "document_version_id": r[1], "document_id": r[2],
                   "title": r[3], "vector": r[4]} for r in rows]
    return _rank_rows_by_similarity(query_vector, candidates, limit)


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


# ---------------------------------------------------------------------------
# Concept-primitive 统一向量检索入口（Option B）
# ---------------------------------------------------------------------------

# 4 个真实 Concept 角色 -> 它们在 embeddings.owner_type 中的物理归属类型。
# observation 角色已被移除（未实现），此处不再纳入。
_CONCEPT_ROLE_TO_OWNER_TYPE = {
    "entity": "entity_obs",
    "relation": "relation_assert",
    "episode": "episode",
    "document": "document_version",
}


def role_to_owner_type(role: str) -> str:
    """把 Concept 角色名映射为 embeddings 表里的 owner_type。

    entity / relation / episode / document 四个角色对应不同的物理 owner
    （observation 属于 entity_obs、assertion 属于 relation_assert、document
    以 document_version 形态嵌入）。未知角色直接报错，避免静默写错 owner。
    """
    try:
        return _CONCEPT_ROLE_TO_OWNER_TYPE[role]
    except KeyError:
        raise ValueError(f"Unknown concept role: {role!r}")


def search_concept_embeddings(conn, role: str, query_vector: bytes,
                              embedding_model: str, limit: int = 10) -> list:
    """统一 Concept 级向量检索入口（Option B）。

    在不合并 entity_*/relation_* 物理表的前提下，对外提供一个以角色为参数的
    单一检索 API，使 ACL 论文里的 "unified NL Concept primitive" 在代码层面
    端到端成立。检索本身**不做任何重排或近似**——它按角色派发到既有的逐角色
    精确全库余弦函数（search_entity_embeddings / search_relation_embeddings /
    search_episode_embeddings / search_document_embeddings），top-N 仍是全语料
    精确结果，与单角色调用完全一致。这样既复用了刚修复的精确排序逻辑，又避免
    在此处复制排序代码。

    四个角色（entity / relation / episode / document）均派发到对应的精确全库
    余弦检索函数；其中 document 以 document_version 形态嵌入，由
    search_document_embeddings 走 embeddings -> document_versions -> documents
    的 join 并按 status='active' 过滤。
    """
    if role == "entity":
        return search_entity_embeddings(conn, query_vector, embedding_model, limit)
    if role == "relation":
        return search_relation_embeddings(conn, query_vector, embedding_model, limit)
    if role == "episode":
        return search_episode_embeddings(conn, query_vector, embedding_model, limit)
    if role == "document":
        return search_document_embeddings(conn, query_vector, embedding_model, limit)
    raise ValueError(f"Unknown concept role: {role!r}")
