"""Verify V1.5 repositories accept pipeline-style data (mock extraction, no real LLM)."""

import sqlite3
import struct
import pytest

from core.storage.sqlite.schema_v15 import init_schema_v15
from core.storage.sqlite.repositories import (
    documents as doc_repo,
    episodes as ep_repo,
    entities as ent_repo,
    relations as rel_repo,
    embeddings as emb_repo,
    search as search_repo,
)

NOW = "2026-05-26T00:00:00Z"


@pytest.fixture
def v15(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema_v15(conn)
    yield conn
    conn.close()


def _setup_doc_episode(conn):
    doc_repo.insert_document(conn, "doc1", title="Test", managed_path="test.md",
                             created_at=NOW, updated_at=NOW)
    doc_repo.insert_document_version(conn, "ver1", "doc1", "hash1", processed_at=NOW)
    doc_repo.update_current_version(conn, "doc1", "ver1", updated_at=NOW)
    ep_repo.insert_episode(
        conn, "ep1", "ep_fam1", "doc1", "ver1",
        source_text="Alice met Bob at the cafe. They discussed quantum physics.",
        memory_text="Alice and Bob discussed quantum physics at a cafe.",
        heading_path="# Meeting Notes",
        start_offset=0, end_offset=62,
        line_start=1, line_end=3,
        chunk_index=0, chunk_hash="chash1",
        episode_type="chunk", activity_type="meeting",
        processed_at=NOW, run_id="run1",
    )
    ep_repo.fts_sync_episode(
        conn, "ep1", "doc1", "ver1",
        source_text="Alice met Bob at the cafe. They discussed quantum physics.",
        memory_text="Alice and Bob discussed quantum physics at a cafe.",
    )


def test_episode_repo_accepts_pipeline_fields(v15):
    _setup_doc_episode(v15)
    ep = ep_repo.get_episode(v15, "ep1")
    assert ep["episode_id"] == "ep1"
    assert ep["episode_family_id"] == "ep_fam1"
    assert ep["source_text"] == "Alice met Bob at the cafe. They discussed quantum physics."
    assert ep["memory_text"] == "Alice and Bob discussed quantum physics at a cafe."
    assert ep["heading_path"] == "# Meeting Notes"
    assert ep["chunk_index"] == 0
    assert ep["episode_type"] == "chunk"
    assert ep["run_id"] == "run1"


def test_entity_repo_accepts_pipeline_fields(v15):
    _setup_doc_episode(v15)
    ent_repo.upsert_entity_family(v15, "fam_alice", "Alice", canonical_content="A person named Alice",
                                  created_at=NOW, updated_at=NOW)
    ent_repo.upsert_entity_family(v15, "fam_bob", "Bob", canonical_content="A person named Bob",
                                  created_at=NOW, updated_at=NOW)
    ent_repo.insert_entity_observation(
        v15, "obs_alice1", "fam_alice", "ep1", "Alice",
        content="Alice is a physicist who met Bob at a cafe",
        processed_at=NOW, run_id="run1",
    )
    ent_repo.insert_entity_observation(
        v15, "obs_bob1", "fam_bob", "ep1", "Bob",
        content="Bob discussed quantum physics with Alice",
        processed_at=NOW, run_id="run1",
    )
    ent_repo.insert_entity_mention(
        v15, "m1", "obs_alice1", "fam_alice", "ep1", "Alice",
        start_offset=0, end_offset=5, line_start=1, line_end=1,
        created_at=NOW,
    )
    ent_repo.insert_entity_mention(
        v15, "m2", "obs_bob1", "fam_bob", "ep1", "Bob",
        start_offset=10, end_offset=13, line_start=1, line_end=1,
        created_at=NOW,
    )

    alice_obs = ent_repo.get_active_observation(v15, "ep1", "fam_alice")
    assert alice_obs is not None
    assert alice_obs["content"] == "Alice is a physicist who met Bob at a cafe"
    assert alice_obs["run_id"] == "run1"

    mentions = ent_repo.get_mentions_by_episode(v15, "ep1")
    assert len(mentions) == 2


def test_relation_repo_accepts_pipeline_fields(v15):
    _setup_doc_episode(v15)
    ent_repo.upsert_entity_family(v15, "fam_alice", "Alice", created_at=NOW, updated_at=NOW)
    ent_repo.upsert_entity_family(v15, "fam_bob", "Bob", created_at=NOW, updated_at=NOW)
    ent_repo.insert_entity_observation(v15, "obs_a", "fam_alice", "ep1", "Alice",
                                       processed_at=NOW, run_id="run1")
    ent_repo.insert_entity_observation(v15, "obs_b", "fam_bob", "ep1", "Bob",
                                       processed_at=NOW, run_id="run1")

    rel_repo.upsert_relation_family(v15, "rfam1", "fam_alice", "fam_bob", "discussed with",
                                    canonical_content="Alice and Bob discussed quantum physics",
                                    created_at=NOW, updated_at=NOW)
    rel_repo.insert_relation_assertion(
        v15, "rel1", "rfam1", "ep1", "obs_a", "obs_b", "fam_alice", "fam_bob",
        content="Alice discussed quantum physics with Bob at the cafe",
        evidence_text="They discussed quantum physics",
        evidence_start_offset=23, evidence_end_offset=55,
        evidence_line_start=1, evidence_line_end=1,
        processed_at=NOW, run_id="run1",
    )

    assertions = rel_repo.get_active_assertions_by_episode(v15, "ep1")
    assert len(assertions) == 1
    a = assertions[0]
    assert a["content"] == "Alice discussed quantum physics with Bob at the cafe"
    assert a["evidence_text"] == "They discussed quantum physics"
    assert a["run_id"] == "run1"
    assert a["subject_entity_family_id"] == "fam_alice"
    assert a["object_entity_family_id"] == "fam_bob"


def test_embedding_blob_round_trip(v15):
    _setup_doc_episode(v15)
    vector = struct.pack("4f", 0.1, 0.2, 0.3, 0.4)
    emb_repo.insert_embedding(
        v15, "emb1", "episode", "ep1", "source_text", "thash1",
        "text-embedding-3-small", 4, vector, run_id="run1", created_at=NOW,
    )
    retrieved = emb_repo.get_embedding(v15, "episode", "ep1", "source_text",
                                       "text-embedding-3-small", "thash1")
    assert retrieved == vector


def test_search_fts_returns_pipeline_episodes(v15):
    _setup_doc_episode(v15)
    results = search_repo.search_fts(v15, "quantum physics", limit=5)
    assert len(results) > 0
    assert results[0]["episode_id"] == "ep1"
    assert results[0]["document_id"] == "doc1"


def test_search_fts_chinese(v15):
    _setup_doc_episode(v15)
    # Insert Chinese content
    doc_repo.insert_document(v15, "doc2", title="Chinese Test",
                             managed_path="cn.md", created_at=NOW, updated_at=NOW)
    doc_repo.insert_document_version(v15, "ver2", "doc2", "hash2", processed_at=NOW)
    doc_repo.update_current_version(v15, "doc2", "ver2", updated_at=NOW)
    ep_repo.insert_episode(
        v15, "ep2", "fam_cn", "doc2", "ver2",
        source_text="心理学是研究人类心理活动的科学",
        memory_text="心理学研究概述",
        chunk_index=0, chunk_hash="chash_cn", processed_at=NOW,
    )
    ep_repo.fts_sync_episode(
        v15, "ep2", "doc2", "ver2",
        source_text="心理学是研究人类心理活动的科学",
        memory_text="心理学研究概述",
    )
    results = search_repo.search_fts(v15, "心理学", limit=5)
    assert len(results) > 0
    assert any(r["episode_id"] == "ep2" for r in results)


def test_pipeline_run_tracking(v15):
    # Create doc/version first (FK dependencies)
    doc_repo.insert_document(v15, "doc1", title="Test", managed_path="t.md",
                             created_at=NOW, updated_at=NOW)
    doc_repo.insert_document_version(v15, "ver1", "doc1", "hash1", processed_at=NOW)
    # Manual insert since no repo function exists yet
    v15.execute(
        "INSERT INTO pipeline_runs (run_id, run_type, status, document_id, document_version_id, started_at) "
        "VALUES ('run1', 'remember', 'succeeded', 'doc1', 'ver1', ?)",
        (NOW,),
    )

    from core.storage.sqlite.repositories.episodes import get_latest_successful_run_id
    run_id = get_latest_successful_run_id(v15, "ver1")
    assert run_id == "run1"
