"""a-storage 组 review 修复回归测试。

覆盖四条 finding（均指向 core/storage/sqlite/library_manager.py）：
- f1  search_concepts_by_bm25 的 episode 兜底腿必须与实体/关系腿同款
     BM25 归一，否则负原始分被下游 0.5/0.15 阈值全部滤掉。
- f6  SQL 语义检索腿（实体/关系）增加多数模型回退，对齐
     _build_vector_cache_for_role：换模型未 backfill 时不再静默空结果。
- f15 delete_document_version 必须先删 from_episode_id 引用的
     document_links 行再删 episodes（FK 无级联）。
- f5  恢复 adjust_confidence_on_corroboration(_batch)：v1.5 schema 下
     置信度存最新 active 观测/断言的 extra_json["confidence"]。
"""

import json

import numpy as np
import pytest

from core.storage.sqlite.library_manager import LibraryManager
from core.storage.sqlite.repositories import documents as doc_repo
from core.storage.sqlite.repositories import episodes as ep_repo

NOW = "2026-05-26T00:00:00Z"


def _seed_doc_with_episode(mgr, doc_id="d1", ver_id="v1", episodes=()):
    """写入 active 文档 + 版本 + 若干（chunk_index, source_text）episode 并同步 FTS。"""
    conn = mgr._conn()
    doc_repo.insert_document(conn, doc_id, title=doc_id,
                             managed_path=f"content/current/{doc_id}.md",
                             created_at=NOW, updated_at=NOW)
    doc_repo.insert_document_version(conn, ver_id, doc_id, "hash1",
                                     processed_at=NOW)
    doc_repo.update_current_version(conn, doc_id, ver_id, updated_at=NOW)
    for chunk_index, text in episodes:
        ep_id = f"{doc_id}_ep{chunk_index}"
        ep_repo.insert_episode(conn, ep_id, f"epfam_{doc_id}", doc_id, ver_id,
                               source_text=text, memory_text=text,
                               chunk_index=chunk_index,
                               chunk_hash=f"ch-{chunk_index}",
                               processed_at=NOW)
        ep_repo.fts_sync_episode(conn, ep_id, doc_id, ver_id,
                                 source_text=text, memory_text=text)
    conn.commit()


def _seed_entity_observation(conn, obs_id, fam_id, ep_id, name,
                             extra_json="{}", processed_at=NOW):
    conn.execute(
        "INSERT OR IGNORE INTO entity_families "
        "(entity_family_id, canonical_name, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)", (fam_id, name, processed_at, processed_at))
    conn.execute(
        "INSERT INTO entity_observations "
        "(entity_id, entity_family_id, episode_id, name, content, status, "
        " processed_at, extra_json) "
        "VALUES (?, ?, ?, ?, '', 'active', ?, ?)",
        (obs_id, fam_id, ep_id, name, processed_at, extra_json))


# ── f1：episode 兜底腿 BM25 归一 ─────────────────────────

def test_f1_single_hit_passes_downstream_threshold(tmp_path):
    """单命中（span=0 → 0.5）必须能过 concept_search 的 0.5 阈值过滤。

    修复前：FTS5 原始负分（如 -1.57e-06）原样进 _score，
    `(item.get("_score") or 0.0) >= 0.5` 把唯一真命中滤成 0 结果。
    """
    mgr = LibraryManager(str(tmp_path / "lib"))
    try:
        _seed_doc_with_episode(
            mgr, episodes=[(0, "josephson junction quantum tunneling effect")])
        results = mgr.search_concepts_by_bm25("tunneling", limit=10)
        assert len(results) == 1
        assert results[0]["role"] == "episode"
        surviving = [r for r in results if (r.get("_score") or 0.0) >= 0.5]
        assert surviving == results  # 下游默认阈值过滤后不得为空
    finally:
        mgr.close()


def test_f1_multi_hit_scores_normalized_0_1(tmp_path):
    """多命中：与实体/关系腿同款 (max_s - score)/span 归一，最相关 → 1.0。"""
    mgr = LibraryManager(str(tmp_path / "lib"))
    try:
        _seed_doc_with_episode(mgr, episodes=[
            (0, "josephson junction josephson junction josephson junction"),
            (1, "a josephson note among many other unrelated words"),
        ])
        results = mgr.search_concepts_by_bm25("josephson", limit=10)
        assert len(results) == 2
        scores = [r["_score"] for r in results]
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert scores[0] == pytest.approx(1.0)  # 已按 _score 降序
        assert scores[0] > scores[1]
        assert results[0]["_score"] >= 0.5
    finally:
        mgr.close()


def test_f1_cjk_like_floor_survives(tmp_path):
    """短 CJK 走 LIKE 兜底（score=0.16 常量）：span=0 → 0.5 ≥ 0.15 地板。"""
    mgr = LibraryManager(str(tmp_path / "lib"))
    try:
        _seed_doc_with_episode(mgr, episodes=[(0, "量子隧道效应的实验记录")])
        results = mgr.search_concepts_by_bm25("量子", limit=10)
        assert results, "CJK LIKE 命中不应被阈值过滤吞掉"
        assert all((r.get("_score") or 0.0) >= 0.15 for r in results)
    finally:
        mgr.close()


# ── f6：SQL 语义检索腿多数模型回退 ───────────────────────

class _FakeEmbeddingClient:
    def __init__(self, model_name, vector):
        self.model_name = model_name
        self.model_path = None
        self._vector = vector

    def is_available(self):
        return True

    def encode(self, _text):
        return self._vector


def _seed_obs_embedding(conn, obs_id, fam_id, ep_id, model, vector,
                        processed_at=NOW):
    _seed_entity_observation(conn, obs_id, fam_id, ep_id, fam_id,
                             processed_at=processed_at)
    conn.execute(
        "INSERT INTO embeddings "
        "(embedding_id, owner_type, owner_id, text_kind, text_hash, "
        " embedding_model, dimensions, vector, created_at) "
        "VALUES (?, 'entity_obs', ?, 'content', 'h', ?, ?, ?, ?)",
        (f"emb-{obs_id}", obs_id, model, len(vector) // 4, vector,
         processed_at))


def test_f6_entity_semantic_search_falls_back_to_majority_model(tmp_path):
    """库内向量全是旧模型、active model 无行 → 回退多数模型而非静默空。"""
    mgr = LibraryManager(str(tmp_path / "lib"))
    try:
        vec = np.array([1.0], dtype=np.float32)
        mgr.embedding_client = _FakeEmbeddingClient("model-new", vec)
        _seed_doc_with_episode(mgr, episodes=[(0, "alpha beta")])
        conn = mgr._conn()
        _seed_obs_embedding(conn, "o1", "f1", "d1_ep0", "model-old", vec.tobytes(),
                            processed_at="2026-01-01T00:00:00Z")
        _seed_obs_embedding(conn, "o2", "f2", "d1_ep0", "model-old", vec.tobytes(),
                            processed_at="2026-01-02T00:00:00Z")
        conn.commit()

        entities = mgr.search_entities_by_similarity("query", threshold=0.3,
                                                     max_results=5)
        assert entities, "换模型未 backfill 时语义检索应回退多数模型"
        assert {e.family_id for e in entities} == {"f1", "f2"}
        assert all(e._score >= 0.3 for e in entities)
    finally:
        mgr.close()


def test_f6_relation_semantic_search_falls_back_to_majority_model(tmp_path):
    mgr = LibraryManager(str(tmp_path / "lib"))
    try:
        vec = np.array([1.0], dtype=np.float32)
        mgr.embedding_client = _FakeEmbeddingClient("model-new", vec)
        _seed_doc_with_episode(mgr, episodes=[(0, "alpha beta")])
        conn = mgr._conn()
        # 断言的 subject/object 观测（relation_assertions 外键要求）
        _seed_obs_embedding(conn, "o1", "f1", "d1_ep0", "model-old", vec.tobytes())
        _seed_obs_embedding(conn, "o2", "f2", "d1_ep0", "model-old", vec.tobytes())
        conn.execute(
            "INSERT INTO relation_families "
            "(relation_family_id, subject_entity_family_id, "
            " object_entity_family_id, canonical_content, created_at, updated_at) "
            "VALUES ('rf1', 'f1', 'f2', 'rel-content', ?, ?)", (NOW, NOW))
        conn.execute(
            "INSERT INTO relation_assertions "
            "(relation_id, relation_family_id, episode_id, subject_entity_id, "
            " object_entity_id, subject_entity_family_id, "
            " object_entity_family_id, content, status, processed_at) "
            "VALUES ('ra1', 'rf1', 'd1_ep0', 'o1', 'o2', 'f1', 'f2', "
            "        'rel-content', 'active', ?)", (NOW,))
        conn.execute(
            "INSERT INTO embeddings "
            "(embedding_id, owner_type, owner_id, text_kind, text_hash, "
            " embedding_model, dimensions, vector, created_at) "
            "VALUES ('emb-ra1', 'relation_assert', 'ra1', 'content', 'h', "
            "        'model-old', 1, ?, ?)", (vec.tobytes(), NOW))
        conn.commit()

        relations = mgr.search_relations_by_similarity("query", threshold=0.3,
                                                       max_results=5)
        assert relations, "关系腿同样需要多数模型回退"
        assert relations[0].family_id == "rf1"
        assert relations[0]._score >= 0.3
    finally:
        mgr.close()


def test_f6_active_model_present_no_fallback(tmp_path):
    """active model 有行时仍严格按它过滤（不引入跨模型混排）。"""
    mgr = LibraryManager(str(tmp_path / "lib"))
    try:
        vec = np.array([1.0], dtype=np.float32)
        mgr.embedding_client = _FakeEmbeddingClient("model-cur", vec)
        _seed_doc_with_episode(mgr, episodes=[(0, "alpha beta")])
        conn = mgr._conn()
        _seed_obs_embedding(conn, "o1", "f1", "d1_ep0", "model-cur", vec.tobytes())
        _seed_obs_embedding(conn, "o2", "f2", "d1_ep0", "model-other", vec.tobytes())
        conn.commit()

        entities = mgr.search_entities_by_similarity("query", threshold=0.3,
                                                     max_results=5)
        assert [e.family_id for e in entities] == ["f1"]
    finally:
        mgr.close()


# ── f15：document_links 先于 episodes 删除 ────────────────

def test_f15_delete_document_version_with_episode_anchored_links(tmp_path):
    """vault 索引写入 from_episode_id 后，删版本不得撞 FK / 留部分 DML。"""
    mgr = LibraryManager(str(tmp_path / "lib"))
    try:
        _seed_doc_with_episode(mgr, episodes=[(0, "see [[note-b]] for details")])
        conn = mgr._conn()
        doc_repo.insert_document_link(
            conn, "dl1", "d1", None, "v1",
            from_episode_id="d1_ep0", link_text="note-b", created_at=NOW)
        conn.commit()

        result = mgr.delete_document_version("v1")  # 修复前 IntegrityError
        assert result == {"deleted": True, "document_id": "d1"}
        assert conn.execute("SELECT COUNT(*) FROM document_links").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE episode_id = 'd1_ep0'"
        ).fetchone()[0] == 0
    finally:
        mgr.close()


# ── f5：印证置信度调整恢复 ────────────────────────────────

def test_f5_adjust_confidence_entity_single(tmp_path):
    mgr = LibraryManager(str(tmp_path / "lib"))
    try:
        _seed_doc_with_episode(mgr, episodes=[(0, "alpha")])
        conn = mgr._conn()
        _seed_entity_observation(conn, "o1", "f1", "d1_ep0", "北京",
                                 extra_json=json.dumps({"confidence": 0.6}))
        conn.commit()

        assert mgr.get_concept_by_family_id("f1")["confidence"] == pytest.approx(0.6)
        mgr.adjust_confidence_on_corroboration("f1", source_type="entity")
        assert mgr.get_concept_by_family_id("f1")["confidence"] == pytest.approx(0.65)
    finally:
        mgr.close()


def test_f5_adjust_confidence_entity_batch_and_cap(tmp_path):
    """批量腿 + 1.0 封顶 + 无置信度记录的 family 不动、未知 fid 不抛。"""
    mgr = LibraryManager(str(tmp_path / "lib"))
    try:
        _seed_doc_with_episode(mgr, episodes=[(0, "alpha")])
        conn = mgr._conn()
        _seed_entity_observation(conn, "o1", "f1", "d1_ep0", "上海",
                                 extra_json=json.dumps({"confidence": 0.9}))
        _seed_entity_observation(conn, "o2", "f2", "d1_ep0", "天津",
                                 extra_json=json.dumps({"confidence": 0.99}))
        _seed_entity_observation(conn, "o3", "f3", "d1_ep0", "重庆")
        conn.commit()

        mgr.adjust_confidence_on_corroboration_batch(
            ["f1", "f2", "f3", "missing"], source_type="entity")
        assert mgr.get_concept_by_family_id("f1")["confidence"] == pytest.approx(0.95)
        assert mgr.get_concept_by_family_id("f2")["confidence"] == pytest.approx(1.0)
        assert mgr.get_concept_by_family_id("f3")["confidence"] is None
    finally:
        mgr.close()


def test_f5_adjust_confidence_relation_batch(tmp_path):
    """关系腿：最新 active 断言的 extra_json["confidence"] 被提升。"""
    mgr = LibraryManager(str(tmp_path / "lib"))
    try:
        _seed_doc_with_episode(mgr, episodes=[(0, "alpha")])
        conn = mgr._conn()
        _seed_entity_observation(conn, "o1", "f1", "d1_ep0", "甲")
        _seed_entity_observation(conn, "o2", "f2", "d1_ep0", "乙")
        conn.execute(
            "INSERT INTO relation_families "
            "(relation_family_id, subject_entity_family_id, "
            " object_entity_family_id, created_at, updated_at) "
            "VALUES ('rf1', 'f1', 'f2', ?, ?)", (NOW, NOW))
        conn.execute(
            "INSERT INTO relation_assertions "
            "(relation_id, relation_family_id, episode_id, subject_entity_id, "
            " object_entity_id, subject_entity_family_id, "
            " object_entity_family_id, content, status, processed_at, extra_json) "
            "VALUES ('ra1', 'rf1', 'd1_ep0', 'o1', 'o2', 'f1', 'f2', '', "
            "        'active', ?, ?)",
            (NOW, json.dumps({"confidence": 0.7})))
        conn.commit()

        assert mgr.get_concept_by_family_id("rf1")["confidence"] == pytest.approx(0.7)
        mgr.adjust_confidence_on_corroboration_batch(["rf1"], source_type="relation")
        assert mgr.get_concept_by_family_id("rf1")["confidence"] == pytest.approx(0.75)
    finally:
        mgr.close()


def test_f5_adjust_confidence_targets_latest_active_row(tmp_path):
    """多版本观测时只提升最新 active 行（与读取口 get_concept_by_family_id 一致）。

    同 episode 同 family 只允许一条 active（唯一部分索引），故旧观测挂
    chunk 0、新观测挂 chunk 1。
    """
    mgr = LibraryManager(str(tmp_path / "lib"))
    try:
        _seed_doc_with_episode(mgr, episodes=[(0, "alpha"), (1, "alpha more")])
        conn = mgr._conn()
        _seed_entity_observation(conn, "o-old", "f1", "d1_ep0", "南京",
                                 extra_json=json.dumps({"confidence": 0.6}),
                                 processed_at="2026-01-01T00:00:00Z")
        _seed_entity_observation(conn, "o-new", "f1", "d1_ep1", "南京",
                                 extra_json=json.dumps({"confidence": 0.8}),
                                 processed_at="2026-02-01T00:00:00Z")
        conn.commit()

        mgr.adjust_confidence_on_corroboration("f1", source_type="entity")
        old = conn.execute(
            "SELECT extra_json FROM entity_observations WHERE entity_id = 'o-old'"
        ).fetchone()[0]
        assert json.loads(old)["confidence"] == pytest.approx(0.6)  # 旧行不动
        assert mgr.get_concept_by_family_id("f1")["confidence"] == pytest.approx(0.85)
    finally:
        mgr.close()
