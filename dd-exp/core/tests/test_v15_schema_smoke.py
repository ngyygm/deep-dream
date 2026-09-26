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


# ── 启动自愈（P2.3：出厂库缺 FTS 表 → 搜索静默 0 结果）────

NOW = "2026-01-01T00:00:00Z"


def _insert_active_episode(conn):
    """插入一条满足 FK 的 active episode（doc → version → episode）。"""
    conn.execute(
        "INSERT INTO documents (document_id, status, created_at, updated_at) "
        "VALUES ('d1', 'active', ?, ?)", (NOW, NOW))
    conn.execute(
        "INSERT INTO document_versions "
        "(document_version_id, document_id, content_hash, status, processed_at) "
        "VALUES ('v1', 'd1', 'h1', 'active', ?)", (NOW,))
    conn.execute(
        "INSERT INTO episodes "
        "(episode_id, episode_family_id, document_id, document_version_id, "
        " source_text, status, processed_at) "
        "VALUES ('ep1', 'f1', 'd1', 'v1', '心理学研究', 'active', ?)", (NOW,))
    conn.commit()


def test_init_schema_heals_missing_fts_table(v15_conn):
    """缺 episodes_fts：重新打开库时补建 + 从 episodes 重建索引。"""
    init_schema_v15(v15_conn)
    _insert_active_episode(v15_conn)
    v15_conn.execute("DROP TABLE episodes_fts")
    result = init_schema_v15(v15_conn)
    assert result["fts_rebuilt"] is True
    assert v15_conn.execute(
        "SELECT COUNT(*) FROM episodes_fts WHERE episode_id = 'ep1'"
    ).fetchone()[0] == 1


def test_init_schema_heals_missing_plain_table(v15_conn):
    """普通表缺失（出厂库实证缺 document_ingestion_state）：IF NOT EXISTS 补建。"""
    init_schema_v15(v15_conn)
    v15_conn.execute("DROP TABLE document_ingestion_state")
    init_schema_v15(v15_conn)
    assert v15_conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'document_ingestion_state'"
    ).fetchone() is not None


def test_init_schema_rebuilds_empty_fts_with_episodes(v15_conn):
    """FTS 表存在但为空而 active episodes 非空：视为损坏，自愈重建。"""
    init_schema_v15(v15_conn)
    _insert_active_episode(v15_conn)
    v15_conn.execute("DELETE FROM episodes_fts")
    result = init_schema_v15(v15_conn)
    assert result["fts_rebuilt"] is True
    assert v15_conn.execute("SELECT COUNT(*) FROM episodes_fts").fetchone()[0] == 1


def test_init_schema_no_rebuild_on_healthy_or_empty_library(v15_conn):
    """健康库/空库不触发重建（fts_rebuilt=False）。"""
    assert init_schema_v15(v15_conn)["fts_rebuilt"] is False
    _insert_active_episode(v15_conn)
    # 已 sync 的 FTS 行保持非空 → 第二次 init 不重建
    v15_conn.execute(
        "INSERT INTO episodes_fts (episode_id, document_id, document_version_id, "
        "name, heading_path, source_text, memory_text) "
        "VALUES ('ep1', 'd1', 'v1', '', '', '心理学研究', '')")
    v15_conn.commit()
    assert init_schema_v15(v15_conn)["fts_rebuilt"] is False


def test_schema_health_reports_missing_objects(v15_conn):
    """schema_health（doctor 数据源）：缺表报缺失，好库报 ok + user_version。"""
    from core.storage.sqlite.schema_v15 import schema_health
    health = schema_health(v15_conn)  # 空库：什么都不存在
    assert health["ok"] is False
    assert "episodes_fts" in health["missing_tables"]
    assert "documents" in health["missing_tables"]

    init_schema_v15(v15_conn)
    health2 = schema_health(v15_conn)
    assert health2["ok"] is True
    assert health2["missing_tables"] == []
    assert isinstance(health2["user_version"], int)
