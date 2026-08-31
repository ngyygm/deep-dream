"""e-cli-vault 组 code review 修复的回归测试。

覆盖四条 finding：
  f4  db backfill-embeddings 落库向量 L2 归一化（与管线写路径一致）
  f12 vault 重索引 revert 回旧内容时复用既有 episode 行（UNIQUE 三元组）
  f13 config set 类型强转从 DEFAULTS 推导（布尔/数值不再落库为字符串）
  f14 resolve_concept_id 跳过 role=episode 命中（episode_id 不是 family id）
"""
import json
import sqlite3

import numpy as np
import pytest
from click.testing import CliRunner

from core.cli._helpers import resolve_concept_id
from core.cli._main import cli
from core.cli.cmd_config import _coerce_value
from core.cli.cmd_db import _backfill_embeddings_run
from core.storage.sqlite.schema_v15 import init_schema_v15
from core.storage.sqlite.vault_indexer import index_vault


# ------------------------------------------------------------------
# f4: backfill-embeddings 归一化
# ------------------------------------------------------------------

class _FakeEmbeddingClient:
    """返回固定非单位向量（|v| = 5）的假 client。"""

    model_name = "fake-model"

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([[3.0, 4.0] for _ in texts], dtype=np.float32)


def _seed_entity_for_backfill(conn):
    conn.execute(
        "INSERT INTO entity_families (entity_family_id, canonical_name, "
        "canonical_content, created_at, updated_at) "
        "VALUES ('fam_1', 'Alpha', 'alpha content', '2026-01-01T00:00:00', "
        "'2026-01-01T00:00:00')")
    conn.execute(
        "INSERT INTO entity_observations (entity_id, entity_family_id, name, "
        "status, processed_at) VALUES ('ent_1', 'fam_1', 'Alpha', 'active', "
        "'2026-01-01T00:00:00')")
    conn.commit()


def test_backfill_embeddings_normalizes_vectors_before_store(tmp_path):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema_v15(conn)
    _seed_entity_for_backfill(conn)

    stats = _backfill_embeddings_run(
        conn, _FakeEmbeddingClient(), batch_size=8, progress=lambda *_: None)

    assert stats["entity"]["stored"] == 1
    blob = conn.execute(
        "SELECT vector FROM embeddings WHERE owner_type = 'entity_obs'"
    ).fetchone()[0]
    vec = np.frombuffer(blob, dtype=np.float32)
    # 落库前归一化：|v| ≈ 1.0，且方向与原始 [3, 4] 一致
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-6)
    assert vec == pytest.approx(np.array([0.6, 0.8], dtype=np.float32))


# ------------------------------------------------------------------
# f12: vault revert 后重索引
# ------------------------------------------------------------------

def test_vault_reindex_reverted_content_reuses_episode_rows(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    v1_text = "# Note\n\n" + ("alpha paragraph. " * 420)  # >4000 字符，多 chunk
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema_v15(conn)
    library = tmp_path / "library"

    # 1) 索引 v1
    note.write_text(v1_text, encoding="utf-8")
    assert index_vault(conn, library, str(vault))["errors"] == 0
    v1_rows = conn.execute(
        "SELECT episode_id, document_version_id, chunk_index, chunk_hash "
        "FROM episodes WHERE status = 'active'").fetchall()
    assert len(v1_rows) >= 1

    # 2) 编辑 → reindex（v2 active，v1 superseded，行保留）
    note.write_text("# Note\n\n" + ("beta paragraph. " * 420), encoding="utf-8")
    assert index_vault(conn, library, str(vault))["errors"] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE status = 'active'").fetchone()[0] >= 1

    # 3) revert 回 v1 原文 → reindex：不得撞 UNIQUE(document_version_id,
    #    chunk_index, chunk_hash)，且复用既有 episode 行而非重新 INSERT
    note.write_text(v1_text, encoding="utf-8")
    result = index_vault(conn, library, str(vault))
    assert result["errors"] == 0
    assert result["indexed"] == 1

    v1_version_ids = {row["document_version_id"] for row in v1_rows}
    assert len(v1_version_ids) == 1
    ver_id = v1_version_ids.pop()
    # 同一 (version, chunk_index) 只有一行（复用而非重复插入）
    dup = conn.execute(
        "SELECT chunk_index, COUNT(*) FROM episodes "
        "WHERE document_version_id = ? GROUP BY chunk_index HAVING COUNT(*) > 1",
        (ver_id,)).fetchall()
    assert dup == []
    # v1 的 episode 行复活为 active，且沿用原 episode_id
    active_rows = conn.execute(
        "SELECT episode_id FROM episodes "
        "WHERE document_version_id = ? AND status = 'active'", (ver_id,)).fetchall()
    assert {r["episode_id"] for r in active_rows} == {r["episode_id"] for r in v1_rows}
    # 当前版本指回 v1
    doc_row = conn.execute(
        "SELECT current_version_id FROM documents WHERE status = 'active'").fetchone()
    assert doc_row["current_version_id"] == ver_id

    # 4) 再次编辑 + 再次 revert 仍可重复索引（不产生累积冲突）
    note.write_text("# Note\n\n" + ("gamma paragraph. " * 420), encoding="utf-8")
    assert index_vault(conn, library, str(vault))["errors"] == 0
    note.write_text(v1_text, encoding="utf-8")
    assert index_vault(conn, library, str(vault))["errors"] == 0


# ------------------------------------------------------------------
# f13: config set 类型强转从 DEFAULTS 推导
# ------------------------------------------------------------------

def test_coerce_value_derives_types_from_defaults():
    # 布尔 key（DEFAULTS 默认 True）解析为 bool
    assert _coerce_value("pipeline.remember.family_write_gate_enabled", "false") is False
    assert _coerce_value("pipeline.remember.family_write_gate_enabled", "true") is True
    assert _coerce_value("pipeline.remember.fallback_cooccurrence_relations", "on") is True
    # 数值 key 解析为 int（不再落库字符串 "900"）
    assert _coerce_value("runtime.task.stall_timeout_seconds", "900") == 900
    assert isinstance(_coerce_value("runtime.task.stall_timeout_seconds", "900"), int)
    assert isinstance(_coerce_value("runtime.task.queue_max_size", "2000"), int)
    # 字符串 key 保持字符串（不因字面量误转）
    assert _coerce_value("host", "1.2.3.4") == "1.2.3.4"
    # None 默认 key：仅 "null" 解析为 None；数字串保持原串（防误伤 API key）
    assert _coerce_value("llm.api_key", "null") is None
    assert _coerce_value("llm.api_key", "12345") == "12345"
    # DEFAULTS 未收录的 key：true/false/null 字面量解析为对应类型
    assert _coerce_value("llm.alignment.enabled", "false") is False
    assert _coerce_value("llm.alignment.enabled", "true") is True
    assert _coerce_value("llm.alignment.base_url", "null") is None
    assert _coerce_value("llm.alignment.base_url", "http://x") == "http://x"
    # 旧三张表语义保留（auth.enabled 默认 None 但语义是布尔）
    assert _coerce_value("auth.enabled", "false") is False
    assert _coerce_value("port", "17000") == 17000
    # 布尔 key 给非布尔值报错
    with pytest.raises(Exception):
        _coerce_value("pipeline.remember.family_write_gate_enabled", "maybe")
    with pytest.raises(Exception):
        _coerce_value("runtime.task.stall_timeout_seconds", "fast")


def _write_config(path, payload=None):
    path.write_text(json.dumps(payload or {"llm": {"mock": True}}), encoding="utf-8")


def test_config_set_writes_typed_bool_and_int(tmp_path):
    from core.server.config import load_config

    config_path = tmp_path / "config.json"
    _write_config(config_path)

    result = CliRunner().invoke(
        cli,
        ["--config", str(config_path), "config", "set",
         "pipeline.remember.family_write_gate_enabled", "false", "--yes"],
    )
    assert result.exit_code == 0, result.output
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["pipeline"]["remember"]["family_write_gate_enabled"] is False

    result = CliRunner().invoke(
        cli,
        ["--config", str(config_path), "config", "set",
         "runtime.task.stall_timeout_seconds", "900", "--yes"],
    )
    assert result.exit_code == 0, result.output
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["runtime"]["task"]["stall_timeout_seconds"] == 900
    assert isinstance(saved["runtime"]["task"]["stall_timeout_seconds"], int)

    # 服务侧加载该配置：不再出现 max(60.0, "900") 之类的 TypeError
    cfg = load_config(str(config_path))
    stall = cfg["runtime"]["task"]["stall_timeout_seconds"]
    assert stall == 900
    assert max(60.0, stall) == 900.0


# ------------------------------------------------------------------
# f14: resolve_concept_id 跳过 episode 命中
# ------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, like_row=None):
        self._like_row = like_row

    def execute(self, sql, params=()):
        return _FakeCursor(self._like_row)


class _FakeStorage:
    def __init__(self, matches, like_row=None):
        self._matches = matches
        self._like_row = like_row

    def get_concept_by_family_id(self, value):
        return None

    def search_concepts_by_bm25(self, query, limit=1, **kwargs):
        return self._matches[:limit]

    def _conn(self):
        return _FakeConn(self._like_row)


def test_resolve_concept_id_skips_episode_role_hit():
    # 仅 episode 命中：episode_id 不能当 family_id 返回，LIKE 兜底也无命中
    storage = _FakeStorage([
        {"family_id": "ep_deadbeef", "id": "ep_deadbeef", "role": "episode",
         "name": "some source text", "_score": 3.0},
    ])
    assert resolve_concept_id(storage, "some source text") is None


def test_resolve_concept_id_falls_through_to_entity_hit_after_episode():
    # episode 排第一、实体排第二：跳过 episode，返回真正的实体 family id
    storage = _FakeStorage([
        {"family_id": "ep_deadbeef", "id": "ep_deadbeef", "role": "episode",
         "name": "Alpha appears here", "_score": 3.0},
        {"family_id": "fam_alpha", "id": "ent_alpha", "role": "entity",
         "name": "Alpha", "_score": 2.0},
    ])
    assert resolve_concept_id(storage, "Alpha") == "fam_alpha"


def test_resolve_concept_id_keeps_entity_hit_without_role_key():
    # 兼容不带 role 键的旧 DTO/假存储：照常返回 family_id
    storage = _FakeStorage([
        {"family_id": "fam_beta", "id": "ent_beta", "name": "Beta"},
    ])
    assert resolve_concept_id(storage, "Beta") == "fam_beta"


def test_resolve_concept_id_episode_only_still_uses_like_fallback():
    # episode 命中被跳过后，LIKE 兜底找到实体仍然生效
    storage = _FakeStorage(
        [{"family_id": "ep_deadbeef", "role": "episode", "_score": 3.0}],
        like_row=("fam_gamma",),
    )
    assert resolve_concept_id(storage, "Gamma") == "fam_gamma"
