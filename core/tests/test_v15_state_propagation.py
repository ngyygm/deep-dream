"""Verify V1.5 state propagation: supersede, reactivate, soft delete, vacuum, hash."""

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
from core.storage.sqlite.integrity import validate_all, validate_fts_consistency
from core.storage.sqlite.content_fs import compute_content_hash

NOW = "2026-05-26T00:00:00Z"


@pytest.fixture
def v15(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema_v15(conn)
    yield conn
    conn.close()


def _full_setup(conn, doc_id="doc1", ver_id="ver1", ep_id="ep1"):
    doc_repo.insert_document(conn, doc_id, title="Test", managed_path="test.md",
                             created_at=NOW, updated_at=NOW)
    doc_repo.insert_document_version(conn, ver_id, doc_id, "hash1", processed_at=NOW)
    doc_repo.update_current_version(conn, doc_id, ver_id, updated_at=NOW)
    ep_repo.insert_episode(conn, ep_id, "fam_ep", doc_id, ver_id,
                           source_text="Alice knows Bob", memory_text="summary",
                           chunk_index=0, chunk_hash="chash1", processed_at=NOW)
    ep_repo.fts_sync_episode(conn, ep_id, doc_id, ver_id,
                             source_text="Alice knows Bob", memory_text="summary")
    ent_repo.upsert_entity_family(conn, "fam1", "Alice", created_at=NOW, updated_at=NOW)
    ent_repo.upsert_entity_family(conn, "fam2", "Bob", created_at=NOW, updated_at=NOW)
    ent_repo.insert_entity_observation(conn, "obs1", "fam1", ep_id, "Alice",
                                       processed_at=NOW)
    ent_repo.insert_entity_observation(conn, "obs2", "fam2", ep_id, "Bob",
                                       processed_at=NOW)
    rel_repo.upsert_relation_family(conn, "rfam1", "fam1", "fam2", "knows",
                                    created_at=NOW, updated_at=NOW)
    rel_repo.insert_relation_assertion(
        conn, "rel1", "rfam1", ep_id, "obs1", "obs2", "fam1", "fam2",
        content="Alice knows Bob", processed_at=NOW,
    )


def test_supersede_version_cascades_to_observations(v15):
    _full_setup(v15)
    doc_repo.supersede_active_version_cascade(v15, "doc1")
    obs1 = v15.execute("SELECT status FROM entity_observations WHERE entity_id='obs1'").fetchone()
    assert obs1[0] == "superseded"


def test_supersede_version_cascades_to_assertions(v15):
    _full_setup(v15)
    doc_repo.supersede_active_version_cascade(v15, "doc1")
    rel = v15.execute("SELECT status FROM relation_assertions WHERE relation_id='rel1'").fetchone()
    assert rel[0] == "superseded"


def test_graph_edges_excludes_superseded(v15):
    _full_setup(v15)
    edges_before = search_repo.get_graph_edges(v15, source_id="doc1")
    assert len(edges_before) > 0
    doc_repo.supersede_active_version_cascade(v15, "doc1")
    edges_after = search_repo.get_graph_edges(v15, source_id="doc1")
    assert len(edges_after) == 0


def test_graph_edges_excludes_deleted_document(v15):
    _full_setup(v15)
    edges_before = search_repo.get_graph_edges(v15, source_id="doc1")
    assert len(edges_before) > 0
    doc_repo.soft_delete_document(v15, "doc1", updated_at=NOW)
    edges_after = search_repo.get_graph_edges(v15, source_id="doc1")
    assert len(edges_after) == 0


def test_fts_consistency_detects_stale_rows(v15):
    _full_setup(v15)
    # FTS row exists but episode is now superseded
    doc_repo.supersede_active_version_cascade(v15, "doc1")
    violations = validate_fts_consistency(v15)
    assert len(violations) > 0, "Should detect stale FTS rows after supersede"


def test_embedding_vacuum_orphaned(v15):
    _full_setup(v15)
    # Insert embedding for episode
    emb_repo.insert_embedding(v15, "emb1", "episode", "ep1", "source_text",
                              "thash1", "model1", 128, b"\x00" * 16,
                              created_at=NOW)
    count_before = emb_repo.count_embeddings(v15)
    assert count_before == 1
    # Delete the episode (make it orphaned)
    v15.execute("DELETE FROM entity_mentions WHERE episode_id='ep1'")
    v15.execute("DELETE FROM relation_assertions WHERE episode_id='ep1'")
    v15.execute("DELETE FROM entity_observations WHERE episode_id='ep1'")
    v15.execute("DELETE FROM episodes WHERE episode_id='ep1'")
    removed = emb_repo.vacuum_orphaned(v15)
    assert removed == 1
    assert emb_repo.count_embeddings(v15) == 0


def test_embedding_vacuum_deleted_documents(v15):
    _full_setup(v15)
    emb_repo.insert_embedding(v15, "emb1", "episode", "ep1", "source_text",
                              "thash1", "model1", 128, b"\x00" * 16,
                              created_at=NOW)
    doc_repo.soft_delete_document(v15, "doc1", updated_at=NOW)
    removed = emb_repo.vacuum_deleted_documents(v15)
    assert removed == 1


def test_content_hash_deterministic():
    h1 = compute_content_hash("hello world")
    h2 = compute_content_hash("hello world")
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex

    # CRLF normalization
    h_crlf = compute_content_hash("hello\r\nworld")
    h_lf = compute_content_hash("hello\nworld")
    assert h_crlf == h_lf

    # Different content
    h3 = compute_content_hash("hello earth")
    assert h1 != h3


def test_a_b_a_reimport(v15):
    """Case B: hash matches a superseded version → reactivate that version."""
    doc_id = "doc1"
    hash_a = "hash_aaa"
    hash_b = "hash_bbb"

    # Import A
    doc_repo.insert_document(v15, doc_id, title="Test", managed_path="test.md",
                             created_at=NOW, updated_at=NOW)
    doc_repo.insert_document_version(v15, "ver_a", doc_id, hash_a, processed_at=NOW)
    doc_repo.update_current_version(v15, doc_id, "ver_a", updated_at=NOW)
    ep_repo.insert_episode(v15, "ep_a1", "fam_a", doc_id, "ver_a",
                           chunk_index=0, chunk_hash="chash_a",
                           processed_at=NOW)
    ep_repo.fts_sync_episode(v15, "ep_a1", doc_id, "ver_a")

    # Import B (supersedes A)
    doc_repo.supersede_active_version_cascade(v15, doc_id)
    doc_repo.insert_document_version(v15, "ver_b", doc_id, hash_b, processed_at=NOW)
    doc_repo.update_current_version(v15, doc_id, "ver_b", updated_at=NOW)
    ep_repo.insert_episode(v15, "ep_b1", "fam_b", doc_id, "ver_b",
                           chunk_index=0, chunk_hash="chash_b",
                           processed_at=NOW)
    ep_repo.fts_sync_episode(v15, "ep_b1", doc_id, "ver_b")

    # Reimport A (Case B: reactivate superseded ver_a)
    doc_repo.supersede_active_version_cascade(v15, doc_id)
    doc_repo.reactivate_version(v15, "ver_a")
    doc_repo.update_current_version(v15, doc_id, "ver_a", updated_at=NOW)
    ep_repo.reactivate_episodes_by_run(v15, "ver_a", run_id=None)

    ver = doc_repo.get_active_version(v15, doc_id)
    assert ver["document_version_id"] == "ver_a"
    assert ver["content_hash"] == hash_a
