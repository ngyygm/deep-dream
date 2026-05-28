"""Verify V1.5 schema initialization creates correct structure."""

import sqlite3
import pytest

from core.storage.sqlite.schema_v15 import init_schema_v15


@pytest.fixture
def v15_conn(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    yield conn
    conn.close()


EXPECTED_TABLES = {
    "documents", "document_versions", "episodes",
    "entity_families", "entity_observations", "entity_mentions",
    "relation_families", "relation_assertions",
    "embeddings", "pipeline_runs", "document_links",
}


def _get_tables(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r[0] for r in rows}


def test_init_schema_creates_all_tables(v15_conn):
    init_schema_v15(v15_conn)
    tables = _get_tables(v15_conn)
    for t in EXPECTED_TABLES:
        assert t in tables, f"Missing table: {t}"


def test_init_schema_creates_fts(v15_conn):
    init_schema_v15(v15_conn)
    rows = v15_conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name='episodes_fts'"
    ).fetchall()
    assert len(rows) == 1
    assert "fts5" in rows[0][1].lower()


def test_init_schema_creates_graph_edges_view(v15_conn):
    init_schema_v15(v15_conn)
    rows = v15_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name='graph_edges'"
    ).fetchall()
    assert len(rows) == 1
    v15_conn.execute("SELECT * FROM graph_edges LIMIT 0")


def test_init_schema_creates_indexes(v15_conn):
    init_schema_v15(v15_conn)
    rows = v15_conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
    ).fetchone()
    assert rows[0] >= 15, f"Expected >= 15 indexes, got {rows[0]}"


def test_init_schema_returns_capability_info(v15_conn):
    result = init_schema_v15(v15_conn)
    assert "fts_tokenizer" in result
    assert result["fts_tokenizer"] in ("trigram", "default")


def test_init_schema_idempotent(v15_conn):
    init_schema_v15(v15_conn)
    init_schema_v15(v15_conn)  # should not raise


def test_foreign_keys_enforced(v15_conn):
    init_schema_v15(v15_conn)
    fk = v15_conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1
    with pytest.raises(Exception):
        v15_conn.execute(
            "INSERT INTO document_versions (document_version_id, document_id, content_hash, processed_at) "
            "VALUES ('v1', 'nonexistent_doc', 'abc', '2026-01-01T00:00:00Z')"
        )


def test_status_check_constraints(v15_conn):
    init_schema_v15(v15_conn)
    with pytest.raises(Exception):
        v15_conn.execute(
            "INSERT INTO documents (document_id, status, created_at, updated_at) "
            "VALUES ('d1', 'invalid_status', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')"
        )


def test_json_check_constraints(v15_conn):
    init_schema_v15(v15_conn)
    v15_conn.execute(
        "INSERT INTO documents (document_id, status, created_at, updated_at) "
        "VALUES ('d1', 'active', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')"
    )
    with pytest.raises(Exception):
        v15_conn.execute(
            "INSERT INTO document_versions "
            "(document_version_id, document_id, content_hash, frontmatter_json, processed_at) "
            "VALUES ('v1', 'd1', 'abc', 'not valid json', '2026-01-01T00:00:00Z')"
        )


def test_partial_unique_index_one_active_version(v15_conn):
    init_schema_v15(v15_conn)
    v15_conn.execute(
        "INSERT INTO documents (document_id, status, created_at, updated_at) "
        "VALUES ('d1', 'active', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')"
    )
    v15_conn.execute(
        "INSERT INTO document_versions "
        "(document_version_id, document_id, content_hash, status, processed_at) "
        "VALUES ('v1', 'd1', 'hash1', 'active', '2026-01-01T00:00:00Z')"
    )
    with pytest.raises(Exception):
        v15_conn.execute(
            "INSERT INTO document_versions "
            "(document_version_id, document_id, content_hash, status, processed_at) "
            "VALUES ('v2', 'd1', 'hash2', 'active', '2026-01-01T00:00:00Z')"
        )


def test_wal_mode(v15_conn):
    init_schema_v15(v15_conn)
    journal = v15_conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert journal.lower() == "wal"
