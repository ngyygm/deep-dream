"""Verify V1.5 write lifecycle: document -> version -> episode -> entity -> relation -> FTS -> graph_edges."""

import sqlite3
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
from core.storage.sqlite.integrity import validate_all

NOW = "2026-05-26T00:00:00Z"


@pytest.fixture
def v15(tmp_path):
    db_path = str(tmp_path / "test.db")
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


def _insert_episode(conn, ep_id="ep1", doc_id="doc1", ver_id="ver1",
                    source_text="hello world", memory_text="hello",
                    family_id="fam1", chunk_index=0, chunk_hash="chash1"):
    ep_repo.insert_episode(conn, ep_id, family_id, doc_id, ver_id,
                           source_text=source_text, memory_text=memory_text,
                           chunk_index=chunk_index, chunk_hash=chunk_hash,
                           processed_at=NOW)
    ep_repo.fts_sync_episode(conn, ep_id, doc_id, ver_id,
                             source_text=source_text, memory_text=memory_text)


# ── Document tests ──────────────────────────────────

def test_document_insert_and_read(v15):
    _insert_doc(v15)
    doc = doc_repo.get_document(v15, "doc1")
    assert doc is not None
    assert doc["document_id"] == "doc1"
    assert doc["title"] == "Test Doc"
    assert doc["status"] == "active"


def test_document_version_insert_and_read(v15):
    _insert_doc(v15)
    _insert_version(v15)
    ver = doc_repo.get_active_version(v15, "doc1")
    assert ver is not None
    assert ver["document_version_id"] == "ver1"
    assert ver["status"] == "active"
    assert ver["content_hash"] == "hash1"


def test_document_soft_delete(v15):
    _insert_doc(v15)
    doc_repo.soft_delete_document(v15, "doc1", updated_at=NOW)
    doc = doc_repo.get_document(v15, "doc1")
    assert doc["status"] == "deleted"
    active = doc_repo.list_documents(v15, status="active")
    assert all(d["document_id"] != "doc1" for d in active)


def test_version_supersede_cascade(v15):
    _insert_doc(v15)
    _insert_version(v15)
    _insert_episode(v15)
    doc_repo.supersede_active_version_cascade(v15, "doc1")
    ver = doc_repo.get_active_version(v15, "doc1")
    assert ver is None  # no active version
    ep = ep_repo.get_episode(v15, "ep1")
    assert ep["status"] == "superseded"


# ── Episode tests ───────────────────────────────────

def test_episode_insert_and_fts_sync(v15):
    _insert_doc(v15)
    _insert_version(v15)
    _insert_episode(v15, source_text="hello world")
    rows = v15.execute("SELECT * FROM episodes_fts WHERE episode_id = 'ep1'").fetchall()
    assert len(rows) == 1


def test_episode_supersede_and_reactivate(v15):
    _insert_doc(v15)
    _insert_version(v15)
    _insert_episode(v15)
    ep_repo.supersede_episodes_by_version(v15, "ver1")
    ep = ep_repo.get_episode(v15, "ep1")
    assert ep["status"] == "superseded"
    # Episodes inserted without run_id get run_id="" (empty string), not NULL
    ep_repo.reactivate_episodes_by_run(v15, "ver1", run_id="")
    ep = ep_repo.get_episode(v15, "ep1")
    assert ep["status"] == "active"


# ── Entity tests ────────────────────────────────────

def test_entity_family_upsert(v15):
    ent_repo.upsert_entity_family(v15, "fam1", "Alice", created_at=NOW, updated_at=NOW)
    ent_repo.upsert_entity_family(v15, "fam1", "Alice Updated", created_at=NOW, updated_at=NOW)
    fam = ent_repo.get_entity_family(v15, "fam1")
    assert fam["canonical_name"] == "Alice Updated"
    count = v15.execute("SELECT COUNT(*) FROM entity_families").fetchone()[0]
    assert count == 1  # no duplicate


def test_entity_observation_unique_per_episode_family(v15):
    _insert_doc(v15)
    _insert_version(v15)
    _insert_episode(v15)
    ent_repo.upsert_entity_family(v15, "fam1", "Alice", created_at=NOW, updated_at=NOW)
    ent_repo.insert_entity_observation(v15, "obs1", "fam1", "ep1", "Alice",
                                       processed_at=NOW)
    obs = ent_repo.get_active_observation(v15, "ep1", "fam1")
    assert obs is not None
    assert obs["entity_id"] == "obs1"


def test_entity_mention_insert_and_read(v15):
    _insert_doc(v15)
    _insert_version(v15)
    _insert_episode(v15, source_text="Alice went to the store")
    ent_repo.upsert_entity_family(v15, "fam1", "Alice", created_at=NOW, updated_at=NOW)
    ent_repo.insert_entity_observation(v15, "obs1", "fam1", "ep1", "Alice",
                                       processed_at=NOW)
    ent_repo.insert_entity_mention(v15, "m1", "obs1", "fam1", "ep1", "Alice",
                                   start_offset=0, end_offset=5, created_at=NOW)
    by_ep = ent_repo.get_mentions_by_episode(v15, "ep1")
    assert len(by_ep) == 1
    by_fam = ent_repo.get_mentions_by_family(v15, "fam1")
    assert len(by_fam) == 1


# ── Relation tests ──────────────────────────────────

def test_relation_family_upsert(v15):
    ent_repo.upsert_entity_family(v15, "fam1", "Alice", created_at=NOW, updated_at=NOW)
    ent_repo.upsert_entity_family(v15, "fam2", "Bob", created_at=NOW, updated_at=NOW)
    rel_repo.upsert_relation_family(v15, "rfam1", "fam1", "fam2", "knows",
                                    created_at=NOW, updated_at=NOW)
    fam = rel_repo.get_relation_family(v15, "rfam1")
    assert fam["predicate"] == "knows"
    assert fam["subject_entity_family_id"] == "fam1"


def test_relation_assertion_insert(v15):
    _insert_doc(v15)
    _insert_version(v15)
    _insert_episode(v15)
    ent_repo.upsert_entity_family(v15, "fam1", "Alice", created_at=NOW, updated_at=NOW)
    ent_repo.upsert_entity_family(v15, "fam2", "Bob", created_at=NOW, updated_at=NOW)
    ent_repo.insert_entity_observation(v15, "obs1", "fam1", "ep1", "Alice",
                                       processed_at=NOW)
    ent_repo.insert_entity_observation(v15, "obs2", "fam2", "ep1", "Bob",
                                       processed_at=NOW)
    rel_repo.upsert_relation_family(v15, "rfam1", "fam1", "fam2", "knows",
                                    created_at=NOW, updated_at=NOW)
    rel_repo.insert_relation_assertion(
        v15, "rel1", "rfam1", "ep1", "obs1", "obs2", "fam1", "fam2",
        content="Alice knows Bob", processed_at=NOW,
    )
    assertions = rel_repo.get_active_assertions_by_episode(v15, "ep1")
    assert len(assertions) == 1
    assert assertions[0]["content"] == "Alice knows Bob"


def test_relation_validate_same_episode(v15):
    _insert_doc(v15)
    _insert_version(v15)
    _insert_episode(v15, ep_id="ep1")
    _insert_episode(v15, ep_id="ep2", family_id="fam2", chunk_index=1, chunk_hash="chash2")
    ent_repo.upsert_entity_family(v15, "fam1", "Alice", created_at=NOW, updated_at=NOW)
    ent_repo.upsert_entity_family(v15, "fam2", "Bob", created_at=NOW, updated_at=NOW)
    ent_repo.insert_entity_observation(v15, "obs1", "fam1", "ep1", "Alice",
                                       processed_at=NOW)
    ent_repo.insert_entity_observation(v15, "obs2", "fam2", "ep2", "Bob",
                                       processed_at=NOW)
    assert rel_repo.validate_same_episode(v15, "obs1", "obs2", "ep1") is False
    ent_repo.insert_entity_observation(v15, "obs3", "fam2", "ep1", "Bob",
                                       processed_at=NOW)
    assert rel_repo.validate_same_episode(v15, "obs1", "obs3", "ep1") is True


# ── Full lifecycle ──────────────────────────────────

def test_full_lifecycle(v15):
    _insert_doc(v15)
    _insert_version(v15)
    _insert_episode(v15, source_text="Alice knows Bob well", memory_text="Alice knows Bob")
    ent_repo.upsert_entity_family(v15, "fam1", "Alice", created_at=NOW, updated_at=NOW)
    ent_repo.upsert_entity_family(v15, "fam2", "Bob", created_at=NOW, updated_at=NOW)
    ent_repo.insert_entity_observation(v15, "obs1", "fam1", "ep1", "Alice",
                                       processed_at=NOW)
    ent_repo.insert_entity_observation(v15, "obs2", "fam2", "ep1", "Bob",
                                       processed_at=NOW)
    ent_repo.insert_entity_mention(v15, "m1", "obs1", "fam1", "ep1", "Alice",
                                   start_offset=0, end_offset=5, created_at=NOW)
    ent_repo.insert_entity_mention(v15, "m2", "obs2", "fam2", "ep1", "Bob",
                                   start_offset=11, end_offset=14, created_at=NOW)
    rel_repo.upsert_relation_family(v15, "rfam1", "fam1", "fam2", "knows",
                                    created_at=NOW, updated_at=NOW)
    rel_repo.insert_relation_assertion(
        v15, "rel1", "rfam1", "ep1", "obs1", "obs2", "fam1", "fam2",
        content="Alice knows Bob", processed_at=NOW,
    )

    # Verify graph_edges view
    edges = search_repo.get_graph_edges(v15, source_id="doc1")
    edge_types = {e["edge_type"] for e in edges}
    assert "HAS_EPISODE" in edge_types

    mentions_edges = search_repo.get_graph_edges(v15, source_id="ep1")
    mention_types = {e["edge_type"] for e in mentions_edges}
    assert "MENTIONS" in mention_types

    # Verify FTS search
    results = search_repo.search_fts(v15, "Alice", limit=5)
    assert len(results) > 0

    # Validate
    violations = validate_all(v15)
    assert len(violations) == 0, f"Violations: {violations}"
