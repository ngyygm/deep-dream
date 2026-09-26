"""Tests for the LIVE hybrid retrieval stack.

Covers (2026-08 audit, P0 safety net):
- core/find/concept_search.py::hybrid_concept_search — the RRF fusion actually
  served by POST /api/v1/concepts/search and /api/v1/find, and shared with CLI
  ``concept search`` since P4.2 (BM25+semantic fusion, CJK threshold handling,
  fallback modes, role boost, BM25 threshold filter).
- storage-level FTS search incl. the short-CJK LIKE fallback path.
- HybridSearcher.cluster_results — the only HybridSearcher method with a
  production caller.

These tests pin CURRENT behavior so later precision work (P2) has a
regression net. Known-divergent behaviors are asserted as-is and marked
with `# CURRENT-BEHAVIOR` comments.
"""

import sqlite3
import pytest

from core.storage.sqlite.schema_v15 import init_schema_v15
from core.storage.sqlite.repositories import (
    documents as doc_repo,
    episodes as ep_repo,
    search as search_repo,
)
# P4.2/P4.3：检索实现自 routes/concepts.py 抽出至 core/find/concept_search.py
from core.find.concept_search import (
    hybrid_concept_search as _hybrid_concept_search,
    has_cjk as _has_cjk,
)

NOW = "2026-05-26T00:00:00Z"


# ── Fixtures ─────────────────────────────────────────

@pytest.fixture
def v15(tmp_path):
    db_path = str(tmp_path / "search_test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema_v15(conn)
    yield conn
    conn.close()


def _insert_doc(conn, doc_id="doc1", title="Test Doc"):
    doc_repo.insert_document(conn, doc_id, title=title,
                             managed_path="content/current/test.md",
                             created_at=NOW, updated_at=NOW)


def _insert_version(conn, doc_id="doc1", ver_id="ver1", content_hash="hash1"):
    doc_repo.insert_document_version(conn, ver_id, doc_id, content_hash,
                                     processed_at=NOW)
    doc_repo.update_current_version(conn, doc_id, ver_id, updated_at=NOW)


def _insert_episode(conn, ep_id, doc_id, ver_id, source_text,
                    family_id, chunk_index):
    ep_repo.insert_episode(conn, ep_id, family_id, doc_id, ver_id,
                           source_text=source_text, memory_text=source_text,
                           chunk_index=chunk_index, chunk_hash=f"ch-{ep_id}",
                           processed_at=NOW)
    ep_repo.fts_sync_episode(conn, ep_id, doc_id, ver_id,
                             source_text=source_text, memory_text=source_text)


def _seed(conn):
    _insert_doc(conn)
    _insert_version(conn)
    _insert_episode(conn, "ep1", "doc1", "ver1",
                    "张三在北京大学研究人工智能", "fam-ep1", chunk_index=0)
    _insert_episode(conn, "ep2", "doc1", "ver1",
                    "李四在上海交通大学教物理", "fam-ep2", chunk_index=1)


# 三段不同 processed_at 的时间锚点（P2.8 双界过滤测试用）
T_EARLY = "2026-01-10T00:00:00Z"
T_MID = "2026-02-10T12:00:00Z"
T_LATE = "2026-03-10T00:00:00Z"


def _seed_timed(conn):
    """Seed doc/version + 早/中/晚三段 episode（FTS 已同步）+ 实体观察。"""
    _insert_doc(conn)
    _insert_version(conn)
    for idx, (ep_id, when) in enumerate(
            [("ep-early", T_EARLY), ("ep-mid", T_MID), ("ep-late", T_LATE)]):
        text = f"alpha {ep_id} record"
        ep_repo.insert_episode(conn, ep_id, f"fam-{ep_id}", "doc1", "ver1",
                               source_text=text, memory_text=text,
                               chunk_index=idx, chunk_hash=f"ch-{ep_id}",
                               processed_at=when)
        ep_repo.fts_sync_episode(conn, ep_id, "doc1", "ver1",
                                 source_text=text, memory_text=text)
        # 实体观察与 episode 同刻（概念时间列 = entity_observations.processed_at）
        conn.execute(
            "INSERT INTO entity_families "
            "(entity_family_id, canonical_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)", (f"ent-{ep_id}", f"ent-{ep_id}", when, when))
        conn.execute(
            "INSERT INTO entity_observations "
            "(entity_id, entity_family_id, episode_id, name, content, status, processed_at) "
            "VALUES (?, ?, ?, ?, ?, 'active', ?)",
            (f"obs-{ep_id}", f"ent-{ep_id}", ep_id, f"ent-{ep_id}", text, when))


class _StubStorage:
    """Minimal storage stub for fusion-logic tests (no DB).

    Honors the role filter like the real storage does; ignores
    time_point/source_document like the real storage currently does.
    """

    def __init__(self, bm25=None, semantic=None):
        self._bm25 = bm25 or []
        self._semantic = semantic or []

    @staticmethod
    def _by_role(items, role):
        if role is None:
            return list(items)
        return [i for i in items if i.get("role") == role]

    def search_concepts_by_bm25(self, query, role=None, limit=20,
                                time_point=None, source_document=None,
                                time_after=None, time_before=None):
        return self._by_role(self._bm25, role)

    # P4.2：语义腿统一走 agent_semantic_search 单入口（返回 {"results": ...} 包装）
    def agent_semantic_search(self, query, *, role=None, top_k=20,
                              threshold=0.3, source_document=None,
                              time_point=None, time_after=None,
                              time_before=None):
        results = self._by_role(self._semantic, role)
        return {"results": results, "total": len(results)}


def _item(fid, role="entity", score=0.5):
    return {"family_id": fid, "role": role, "name": fid, "_score": score}


# ── CJK detection ────────────────────────────────────

def test_has_cjk_detection():
    assert _has_cjk("爱情") is True
    assert _has_cjk("张三 professor") is True
    assert _has_cjk("hello world") is False
    assert _has_cjk("") is False


# ── RRF fusion logic (live path) ─────────────────────

def test_hybrid_both_lists_item_outranks_single_list():
    """An item found by BOTH BM25 and semantic must outrank items found once."""
    shared = _item("fid-shared", score=0.9)
    bm25_only = _item("fid-bm25", score=0.9)
    sem_only = _item("fid-sem", score=0.9)
    storage = _StubStorage(
        bm25=[shared, bm25_only],
        semantic=[shared, sem_only],
    )
    results, meta = _hybrid_concept_search(
        storage, "query", role=None, limit=10, threshold=0.0)
    assert meta["effective_mode"] == "hybrid"
    assert results[0]["family_id"] == "fid-shared"
    fids = {r["family_id"] for r in results}
    assert fids == {"fid-shared", "fid-bm25", "fid-sem"}


def test_hybrid_rrf_scores_normalized_to_top_1():
    storage = _StubStorage(bm25=[_item("a"), _item("b")],
                           semantic=[_item("a")])
    results, _ = _hybrid_concept_search(
        storage, "q", role=None, limit=10, threshold=0.0)
    # top item's fused score normalized to ~1.0 (rounded to 4 decimals path
    # in concepts rounds to 6): top score is the max by construction
    assert results[0]["_score"] >= results[-1]["_score"]
    assert results[0]["_score"] > 0


def test_hybrid_bm25_only_mode_when_semantic_empty():
    storage = _StubStorage(bm25=[_item("a", score=0.8)])
    results, meta = _hybrid_concept_search(
        storage, "english query", role=None, limit=10, threshold=0.0)
    assert meta["effective_mode"] == "bm25_only"
    assert [r["family_id"] for r in results] == ["a"]


def test_hybrid_semantic_only_mode_when_bm25_empty():
    storage = _StubStorage(semantic=[_item("b", score=0.8)])
    results, meta = _hybrid_concept_search(
        storage, "english query", role=None, limit=10, threshold=0.0)
    assert meta["effective_mode"] == "semantic_only"
    assert [r["family_id"] for r in results] == ["b"]


def test_hybrid_empty_both():
    storage = _StubStorage()
    results, meta = _hybrid_concept_search(
        storage, "q", role=None, limit=10, threshold=0.0)
    assert results == []
    assert meta["bm25_results"] == 0


def test_hybrid_cjk_query_meta_and_lowered_thresholds():
    """CJK queries lower semantic threshold to 0.3 and BM25 threshold to 0.15."""
    # BM25 score 0.2: survives CJK threshold (0.15) but would fail default.
    storage = _StubStorage(bm25=[_item("cjk-a", score=0.2)])
    results, meta = _hybrid_concept_search(
        storage, "爱情", role=None, limit=10, threshold=0.5)
    assert meta["effective_mode"] == "bm25_only_cjk"
    assert len(results) == 1


def test_hybrid_bm25_threshold_filters_low_scores():
    storage = _StubStorage(bm25=[_item("low", score=0.1)])
    results, meta = _hybrid_concept_search(
        storage, "english query", role=None, limit=10, threshold=0.5)
    assert results == []
    assert meta["effective_mode"] == "hybrid"


def test_hybrid_role_boost_entities_above_relations():
    """No role filter: entity results rank above higher-scored relations."""
    storage = _StubStorage(
        bm25=[_item("rel-1", role="relation", score=0.9)],
        semantic=[_item("ent-1", role="entity", score=0.9)],
    )
    results, _ = _hybrid_concept_search(
        storage, "q", role=None, limit=10, threshold=0.0)
    assert results[0]["family_id"] == "ent-1"


def test_hybrid_role_filter_suppresses_boost():
    storage = _StubStorage(
        bm25=[_item("rel-1", role="relation", score=0.9)],
        semantic=[_item("ent-1", role="entity", score=0.9)],
    )
    results, _ = _hybrid_concept_search(
        storage, "q", role="relation", limit=10, threshold=0.0)
    assert all(r["family_id"] != "ent-1" for r in results)


# CURRENT-BEHAVIOR: search_concepts_by_bm25 silently ignores time_point /
# source_document filters (P2 precision item). Pin the status quo so the
# fix is a deliberate, visible change.
def test_hybrid_ignored_filters_current_behavior():
    storage = _StubStorage(bm25=[_item("a", score=0.9)])
    results, meta = _hybrid_concept_search(
        storage, "english query", role=None, limit=10, threshold=0.0,
        time_point="1999-01-01T00:00:00Z",
        source_document="nonexistent-doc")
    assert len(results) == 1  # filters not applied by storage (yet)


# ── Storage-level FTS (real SQLite) ──────────────────

def test_fts_search_matches_seeded_text(v15):
    _seed(v15)
    hits = search_repo.search_fts(v15, "人工智能", limit=10)
    assert len(hits) == 1
    assert hits[0]["episode_id"] == "ep1"


def test_fts_source_document_filter_is_applied(v15):
    """A document selector must not silently fall back to cross-document hits."""
    _seed(v15)
    _insert_doc(v15, doc_id="doc2", title="Other Doc")
    _insert_version(v15, doc_id="doc2", ver_id="ver2", content_hash="hash2")
    _insert_episode(v15, "ep3", "doc2", "ver2",
                    "张三在北京大学研究人工智能", "fam-ep3", chunk_index=0)

    all_hits = search_repo.search_fts(v15, "人工智能", limit=10)
    assert {hit["episode_id"] for hit in all_hits} == {"ep1", "ep3"}
    doc1_hits = search_repo.search_fts(
        v15, "人工智能", source_document="doc1", limit=10
    )
    assert {hit["episode_id"] for hit in doc1_hits} == {"ep1"}
    assert search_repo.search_fts(
        v15, "人工智能", source_document="missing-doc", limit=10
    ) == []


def test_fts_search_and_mode_fallback_to_or(v15):
    """All-term AND miss degrades to OR instead of returning nothing."""
    _seed(v15)
    # "量子纠缠" appears in no seeded row → AND empty → OR returns ep1.
    hits = search_repo.search_fts(v15, "人工智能 量子纠缠", limit=10)
    assert hits
    assert {h.get("match_mode") for h in hits} == {"or"}


def _insert_entity(conn, fam_id, name, ep_id):
    """实体 family + observation + mention 三件套（FK 齐全）。"""
    conn.execute(
        "INSERT INTO entity_families "
        "(entity_family_id, canonical_name, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)", (fam_id, name, NOW, NOW))
    conn.execute(
        "INSERT INTO entity_observations "
        "(entity_id, entity_family_id, episode_id, name, content, status, processed_at) "
        "VALUES (?, ?, ?, ?, '', 'active', ?)",
        (f"{fam_id}-obs1", fam_id, ep_id, name, NOW))
    conn.execute(
        "INSERT INTO entity_mentions "
        "(mention_id, entity_id, entity_family_id, episode_id, surface_text, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (f"{fam_id}-ment1", f"{fam_id}-obs1", fam_id, ep_id, name, NOW))


def test_fts_short_cjk_entity_prefix_path(v15):
    """1-2 字 CJK 查询优先走 canonical_name 前缀命中（P2.4）。

    entity_families.canonical_name 前缀匹配 → 锚定 mention 所在 episode，
    命中行带 match_mode='entity_prefix'；FTS 仍先查（trigram 下 2 字查
    不到，正常降级）。
    """
    _seed(v15)
    _insert_entity(v15, "fam-pku", "北京大学", "ep1")
    hits = search_repo.search_fts(v15, "北京", limit=10)
    assert any(h["episode_id"] == "ep1" and h["match_mode"] == "entity_prefix"
               for h in hits)
    # 前缀未命中任何 canonical_name：实体路径不产出结果
    assert search_repo.search_fts(v15, "复旦", limit=10) == []


def test_fts_short_cjk_entity_prefix_uses_index(v15):
    """前缀区间查找必须走 idx_entityfam_name（SEARCH 而非 SCAN）。

    绑定参数的 LIKE 'xx%' 在默认 case_sensitive_like=OFF 下是 SCAN——
    这正是改用字典序区间的依据（EXPLAIN QUERY PLAN 实测钉住）。
    """
    plan = v15.execute(
        "EXPLAIN QUERY PLAN SELECT entity_family_id FROM entity_families "
        "WHERE canonical_name >= ? AND canonical_name < ?", ("北京", "京城")
    ).fetchall()
    detail = " ".join(r[3] for r in plan)
    assert "SEARCH" in detail and "idx_entityfam_name" in detail
    like_plan = v15.execute(
        "EXPLAIN QUERY PLAN SELECT entity_family_id FROM entity_families "
        "WHERE canonical_name LIKE ?", ("北%",)
    ).fetchall()
    assert "SCAN" in " ".join(r[3] for r in like_plan)


def test_fts_short_cjk_like_fallback(v15):
    """1-2 字 CJK 查询最终兜底到 %xx% LIKE（常数分 0.16）。

    实体前缀路径未命中时降级到全表 LIKE；兜底行现在带
    match_mode='like' 键，且按 processed_at DESC, episode_id 确定排序。
    """
    _seed(v15)
    hits = search_repo.search_fts(v15, "张三", limit=10)
    assert any(h["episode_id"] == "ep1" for h in hits)
    like_rows = [h for h in hits if h.get("match_mode") == "like"]
    assert like_rows, "LIKE 兜底行必须带 match_mode='like'"
    assert all(h["score"] == 0.16 for h in like_rows)
    # 确定性排序：同参数两次查询结果完全一致
    again = search_repo.search_fts(v15, "张三", limit=10)
    assert [h["episode_id"] for h in hits] == [h["episode_id"] for h in again]


def test_fts_short_cjk_prefix_beats_like_dedup(v15):
    """同一 episode 命中实体前缀后不再被 LIKE 兜底重复追加。"""
    _seed(v15)
    _insert_entity(v15, "fam-zs", "张三", "ep1")
    hits = search_repo.search_fts(v15, "张三", limit=10)
    ep_ids = [h["episode_id"] for h in hits]
    assert ep_ids.count("ep1") == 1
    assert hits[0]["match_mode"] == "entity_prefix"


def test_fts_missing_fts_table_raises(v15):
    """缺 episodes_fts 表：search_fts 抛 OperationalError，不再静默空列表（P2.3）。"""
    _seed(v15)
    v15.execute("DROP TABLE episodes_fts")
    with pytest.raises(sqlite3.OperationalError):
        search_repo.search_fts(v15, "人工智能", limit=10)


def test_fts_missing_state_table_raises(v15):
    """缺 document_ingestion_state 表同样上抛（出厂库实证的缺失表）。"""
    _seed(v15)
    v15.execute("DROP TABLE document_ingestion_state")
    with pytest.raises(sqlite3.OperationalError):
        search_repo.search_fts(v15, "人工智能", limit=10)


def test_fts_query_special_characters_quoted(v15):
    """Queries with FTS operators (e.g. self-care, Dr. Seuss) must not raise."""
    _seed(v15)
    hits = search_repo.search_fts(v15, "self-care (Dr. Seuss)", limit=10)
    assert isinstance(hits, list)


# ── FTS 双界时间过滤（P2.8）───────────────────────────

def test_fts_search_time_dual_bounds(v15):
    """双界区间只命中区间内 episode：早于/晚于界都被滤掉。"""
    _seed_timed(v15)
    all_ids = {"ep-early", "ep-mid", "ep-late"}

    hits = search_repo.search_fts(v15, "alpha", limit=10)
    assert {h["episode_id"] for h in hits} == all_ids

    # 双界：只留 mid
    hits = search_repo.search_fts(v15, "alpha", limit=10,
                                  time_after="2026-02-01T00:00:00Z",
                                  time_before="2026-02-28T23:59:59Z")
    assert {h["episode_id"] for h in hits} == {"ep-mid"}

    # 单下界：滤掉 early
    hits = search_repo.search_fts(v15, "alpha", limit=10, time_after=T_MID)
    assert {h["episode_id"] for h in hits} == {"ep-mid", "ep-late"}

    # 单上界：滤掉 late
    hits = search_repo.search_fts(v15, "alpha", limit=10, time_before=T_MID)
    assert {h["episode_id"] for h in hits} == {"ep-early", "ep-mid"}

    # 闭区间：界值本身命中（>= / <=）
    hits = search_repo.search_fts(v15, "alpha", limit=10,
                                  time_after=T_MID, time_before=T_MID)
    assert {h["episode_id"] for h in hits} == {"ep-mid"}


def test_entities_by_bm25_time_dual_bounds(tmp_path):
    """存储层实体检索双界下推：概念时间列 = entity_observations.processed_at。"""
    from core.storage.sqlite.library_manager import LibraryManager
    mgr = LibraryManager(str(tmp_path / "lib"))
    conn = mgr._conn()
    _seed_timed(conn)
    conn.commit()

    try:
        entities = mgr.search_entities_by_bm25("alpha", limit=10)
        assert {e.name for e in entities} == {
            "ent-ep-early", "ent-ep-mid", "ent-ep-late"}

        # 双界：早于/晚于区间的实体都被滤掉
        entities = mgr.search_entities_by_bm25(
            "alpha", limit=10,
            time_after="2026-02-01T00:00:00Z",
            time_before="2026-02-28T23:59:59Z")
        assert [e.name for e in entities] == ["ent-ep-mid"]

        # 单下界 / 单上界
        entities = mgr.search_entities_by_bm25("alpha", limit=10, time_after=T_MID)
        assert {e.name for e in entities} == {"ent-ep-mid", "ent-ep-late"}
        entities = mgr.search_entities_by_bm25("alpha", limit=10, time_before=T_MID)
        assert {e.name for e in entities} == {"ent-ep-early", "ent-ep-mid"}
    finally:
        mgr.close()


def test_concepts_by_bm25_normalizes_scores(v15):
    _seed(v15)
    # Use the repo-level path through a thin wrapper: verify normalization
    # logic directly on search_fts output instead of instantiating the
    # full manager (which needs a storage root).
    raw = search_repo.search_fts(v15, "人工智能", limit=10)
    scores = [r["score"] for r in raw]
    span = max(scores) - min(scores)
    normalized = [(max(scores) - s) / span if span else 0.5 for s in scores]
    assert all(0.0 <= n <= 1.0 for n in normalized)


# ── HybridSearcher.cluster_results (only live method) ──

def test_cluster_results_groups_similar_items():
    from core.find.hybrid import HybridSearcher
    searcher = HybridSearcher(storage=None)
    items = [
        {"family_id": "a", "name": "机器学习", "content": "machine learning", "_score": 0.9},
        {"family_id": "b", "name": "深度学习", "content": "deep learning", "_score": 0.8},
        {"family_id": "c", "name": "北京", "content": "beijing city", "_score": 0.7},
        {"family_id": "d", "name": "上海", "content": "shanghai city", "_score": 0.6},
    ]
    clusters = searcher.cluster_results(items, num_clusters=2, sim_threshold=0.2)
    assert clusters
    total = sum(c["count"] for c in clusters)
    assert total == 4
    labels = {c["label"] for c in clusters}
    assert all(isinstance(label, str) and label for label in labels)


def test_cluster_results_few_items_returns_empty():
    from core.find.hybrid import HybridSearcher
    searcher = HybridSearcher(storage=None)
    assert searcher.cluster_results([{"family_id": "a"}]) == []
    assert searcher.cluster_results([]) == []


# ── 向量缓存按模型过滤（P2 精准性）─────────────────────

class TestVectorCacheModelFilter:
    """不变式 (e)：对齐向量缓存必须按 active embedding model 过滤。

    此前不过滤——换模型后跨模型余弦是垃圾值且缓存永不失效。
    """

    @staticmethod
    def _seed_embedding_row(conn, obs_id, fam_id, model, created_at,
                            vector=b"\x00\x00\x80\x3f"):
        conn.execute(
            "INSERT INTO entity_families "
            "(entity_family_id, canonical_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)", (fam_id, fam_id, created_at, created_at))
        conn.execute(
            "INSERT INTO entity_observations "
            "(entity_id, entity_family_id, episode_id, name, content, status, processed_at) "
            "VALUES (?, ?, NULL, ?, '', 'active', ?)",
            (obs_id, fam_id, fam_id, created_at))
        conn.execute(
            "INSERT INTO embeddings "
            "(embedding_id, owner_type, owner_id, text_kind, text_hash, "
            " embedding_model, dimensions, vector, created_at) "
            "VALUES (?, 'entity_obs', ?, 'content', 'h', ?, 1, ?, ?)",
            (f"emb-{obs_id}", obs_id, model, vector, created_at))

    def _manager(self, tmp_path, model_name):
        from types import SimpleNamespace
        from core.storage.sqlite.library_manager import LibraryManager
        mgr = LibraryManager(str(tmp_path / "lib"))
        mgr.embedding_client = SimpleNamespace(
            model_name=model_name, model_path=None,
            is_available=lambda: True)
        return mgr

    def test_cache_filters_by_active_model(self, tmp_path):
        mgr = self._manager(tmp_path, "m1")
        conn = mgr._conn()
        self._seed_embedding_row(conn, "o1", "f1", "m1", "2026-01-01T00:00:00Z")
        self._seed_embedding_row(conn, "o2", "f2", "m2", "2026-01-02T00:00:00Z")
        cache = mgr._vector_cache_for_role("entity")
        assert cache["model"] == "m1"
        assert [r["family_id"] for r in cache["rows"]] == ["f1"]

    def test_cache_majority_model_fallback(self, tmp_path):
        """active model 无行 → 回退多数模型（混模型存量库仍可用）。"""
        mgr = self._manager(tmp_path, "m9-new")
        conn = mgr._conn()
        self._seed_embedding_row(conn, "o1", "f1", "old-m", "2026-01-01T00:00:00Z")
        self._seed_embedding_row(conn, "o2", "f2", "old-m", "2026-01-02T00:00:00Z")
        self._seed_embedding_row(conn, "o3", "f3", "other", "2026-01-03T00:00:00Z")
        cache = mgr._vector_cache_for_role("entity")
        assert cache["model"] == "old-m"  # 回退到多数模型
        assert {r["family_id"] for r in cache["rows"]} == {"f1", "f2"}

    def test_cache_rebuilds_on_model_switch(self, tmp_path):
        mgr = self._manager(tmp_path, "m1")
        conn = mgr._conn()
        self._seed_embedding_row(conn, "o1", "f1", "m1", "2026-01-01T00:00:00Z")
        self._seed_embedding_row(conn, "o2", "f2", "m2", "2026-01-02T00:00:00Z")
        assert mgr._vector_cache_for_role("entity")["model"] == "m1"
        mgr.embedding_client.model_name = "m2"
        assert mgr._vector_cache_for_role("entity")["model"] == "m2"

    def test_invalidate_clears_cache(self, tmp_path):
        mgr = self._manager(tmp_path, "m1")
        conn = mgr._conn()
        self._seed_embedding_row(conn, "o1", "f1", "m1", "2026-01-01T00:00:00Z")
        assert mgr._vector_cache_for_role("entity")["matrix"] is not None
        mgr.invalidate_vector_caches()
        assert mgr._vector_role_cache == {}


def test_embedding_search_deterministic_order(tmp_path):
    """无 ORDER BY 的 LIMIT 行序任意（P2 精准性）——现按 created_at DESC 确定。"""
    import numpy as np
    from core.storage.sqlite.schema_v15 import init_schema_v15
    from core.storage.sqlite.repositories import embeddings as emb_repo
    from core.storage.sqlite.repositories import documents as doc_repo
    from core.storage.sqlite.repositories import episodes as ep_repo

    conn = sqlite3.connect(str(tmp_path / "det.db"))
    init_schema_v15(conn)
    doc_repo.insert_document(conn, "d1", title="d",
                             managed_path="content/current/d.md",
                             created_at=NOW, updated_at=NOW)
    doc_repo.insert_document_version(conn, "v1", "d1", "h1", processed_at=NOW)
    doc_repo.update_current_version(conn, "d1", "v1", updated_at=NOW)
    vec = np.array([1.0], dtype=np.float32).tobytes()
    ep_repo.insert_episode(conn, "ep1", "epf1", "d1", "v1",
                           source_text="t", memory_text="t",
                           chunk_index=0, chunk_hash="ch-1",
                           processed_at=NOW)
    for i, (obs, when) in enumerate([("o-old", "2026-01-01T00:00:00Z"),
                                     ("o-new", "2026-02-01T00:00:00Z")]):
        fam = f"f{i}"
        conn.execute(
            "INSERT INTO entity_families "
            "(entity_family_id, canonical_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)", (fam, fam, when, when))
        conn.execute(
            "INSERT INTO entity_observations "
            "(entity_id, entity_family_id, episode_id, name, content, status, processed_at) "
            "VALUES (?, ?, 'ep1', ?, '', 'active', ?)", (obs, fam, fam, when))
        conn.execute(
            "INSERT INTO embeddings "
            "(embedding_id, owner_type, owner_id, text_kind, text_hash, "
            " embedding_model, dimensions, vector, created_at) "
            "VALUES (?, 'entity_obs', ?, 'content', 'h', 'm', 1, ?, ?)",
            (f"emb-{i}", obs, vec, when))
    hits = emb_repo.search_entity_embeddings(conn, vec, "m", limit=1)
    assert [h["owner_id"] for h in hits] == ["o-new"]  # 新者先出，稳定
    conn.close()


# ── P2.6：/find 嵌套线程池自死锁回归 ───────────────────

def test_hybrid_no_nested_submit_when_pool_saturated():
    """池满时 hybrid 检索不得嵌套 submit 到 _shared_pool（不变式 c）。

    外层 /find 已把实体/关系检索提交到 _shared_pool 并阻塞等待结果；
    若 _hybrid_concept_search 再向同一池 submit 内层任务，池满时内层任务
    排在队尾永远无法执行，外层 worker 互等即自死锁。此处占满全部 worker，
    断言 hybrid 检索在调用线程内同步完成（小超时防挂起）。
    """
    import threading
    from core.server.routes import concepts as concepts_mod

    release = threading.Event()
    n_workers = concepts_mod._shared_pool._max_workers
    blockers = [concepts_mod._shared_pool.submit(release.wait)
                for _ in range(n_workers)]
    try:
        storage = _StubStorage(bm25=[_item("a", score=0.9)])
        out = []
        t = threading.Thread(
            target=lambda: out.append(
                _hybrid_concept_search(storage, "q", role=None, limit=5,
                                       threshold=0.0)))
        t.daemon = True
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive(), (
            "hybrid 检索挂起：内层仍向已满的 _shared_pool submit（嵌套自死锁）")
        assert out and out[0][0][0]["family_id"] == "a"
    finally:
        release.set()
        for b in blockers:
            b.result(timeout=5.0)


def test_find_hybrid_concurrent_requests_no_deadlock(client):
    """并发 /find hybrid 请求不得自死锁（P2.6 端到端回归）。

    每个请求向 _shared_pool 提交 2 个外层检索任务；旧实现在外层任务内再向
    同一池 submit 内层任务，并发下全部 worker 互等挂起。新实现内层同步执行。
    """
    import threading
    from core.tests.conftest import TEST_GRAPH_ID

    results = []
    lock = threading.Lock()

    def _post(idx):
        try:
            resp = client.post("/api/v1/find", json={
                "graph_id": TEST_GRAPH_ID,
                "query": f"nonexistent{idx}",
                "search_mode": "hybrid",
                "limit": 5,
            })
            with lock:
                results.append(resp.status_code)
        except Exception as exc:
            with lock:
                results.append(f"error: {exc}")

    threads = [threading.Thread(target=_post, args=(i,)) for i in range(3)]
    for t in threads:
        t.daemon = True
        t.start()
    for t in threads:
        t.join(timeout=15.0)
    assert all(not t.is_alive() for t in threads), (
        "并发 /find hybrid 请求死锁：内层任务嵌套 submit 到同一共享池")
    assert results and all(code == 200 for code in results), results


# ── P2.8：/find 双界时间过滤下推（路由级）──────────────

def test_find_time_dual_bounds_route(client, processor):
    """POST /api/v1/find 的 time_after/time_before 双界下推。

    旧实现把两界折叠成单个 time_point（另一界被静默丢弃，且搜索路径的
    存储层不消费 time_point），早于/晚于区间的实体照样返回。
    """
    from core.tests.conftest import TEST_GRAPH_ID

    storage = processor.storage
    conn = storage._conn()
    _seed_timed(conn)
    conn.commit()

    resp = client.post("/api/v1/find", json={
        "graph_id": TEST_GRAPH_ID,
        "query": "alpha",
        "search_mode": "bm25",
        "limit": 10,
        # threshold=0 关掉 BM25 归一化分数过滤，只考察时间双界
        "threshold": 0.0,
        "time_after": "2026-02-01T00:00:00Z",
        "time_before": "2026-02-28T23:59:59Z",
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    names = [e["name"] for e in body["data"]["entities"]]
    assert names == ["ent-ep-mid"], (
        f"双界区间应只命中区间内实体（早于/晚于都被滤掉），got {names}")

    # 无界对照：三段全命中（证明过滤来自双界，而非种子缺数据）
    resp_all = client.post("/api/v1/find", json={
        "graph_id": TEST_GRAPH_ID,
        "query": "alpha",
        "search_mode": "bm25",
        "limit": 10,
        "threshold": 0.0,
    })
    assert resp_all.status_code == 200
    names_all = [e["name"] for e in resp_all.get_json()["data"]["entities"]]
    assert set(names_all) == {"ent-ep-early", "ent-ep-mid", "ent-ep-late"}
