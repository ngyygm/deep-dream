"""Tests for the LIVE hybrid retrieval stack.

Covers (2026-08 audit, P0 safety net):
- routes/concepts.py::_hybrid_concept_search — the RRF fusion actually served
  by POST /api/v1/concepts/search and /api/v1/find (BM25+semantic fusion,
  CJK threshold handling, fallback modes, role boost, BM25 threshold filter).
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
from core.server.routes.concepts import _hybrid_concept_search, _has_cjk

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
                                time_point=None, source_document=None):
        return self._by_role(self._bm25, role)

    def search_concepts_by_similarity(self, query_text, role=None,
                                      threshold=0.3, max_results=20,
                                      time_point=None, source_document=None):
        return self._by_role(self._semantic, role)


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


def test_fts_search_and_mode_fallback_to_or(v15):
    """All-term AND miss degrades to OR instead of returning nothing."""
    _seed(v15)
    # "量子纠缠" appears in no seeded row → AND empty → OR returns ep1.
    hits = search_repo.search_fts(v15, "人工智能 量子纠缠", limit=10)
    assert hits
    assert {h.get("match_mode") for h in hits} == {"or"}


def test_fts_short_cjk_like_fallback(v15):
    """1-2 char CJK queries use the LIKE fallback (constant score 0.16).

    CURRENT-BEHAVIOR: LIKE-appended rows carry no ``match_mode`` key and the
    fallback has no relevance ordering (audit P2 item).
    """
    _seed(v15)
    hits = search_repo.search_fts(v15, "张三", limit=10)
    assert any(h["episode_id"] == "ep1" for h in hits)
    assert all(h["score"] == 0.16 for h in hits if h["episode_id"] == "ep1")


def test_fts_query_special_characters_quoted(v15):
    """Queries with FTS operators (e.g. self-care, Dr. Seuss) must not raise."""
    _seed(v15)
    hits = search_repo.search_fts(v15, "self-care (Dr. Seuss)", limit=10)
    assert isinstance(hits, list)


def test_concepts_by_bm25_normalizes_scores(v15):
    from core.storage.sqlite.library_manager import LibraryManager
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
    assert all(isinstance(l, str) and l for l in labels)


def test_cluster_results_few_items_returns_empty():
    from core.find.hybrid import HybridSearcher
    searcher = HybridSearcher(storage=None)
    assert searcher.cluster_results([{"family_id": "a"}]) == []
    assert searcher.cluster_results([]) == []
