"""db backfill-embeddings：批量补算缺失向量（P5，原根目录脚本移植为 CLI）。

Stub embedding client，不发真实请求、不加载模型。
"""
import json
import sqlite3
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from core.cli._main import cli
from core.storage import embedding as emb_mod


class _StubClient:
    model_name = "stub-model"
    model_path = None

    def __init__(self, *args, **kwargs):
        pass

    def is_available(self):
        return True

    def encode(self, texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(8)
            out.append(v / np.linalg.norm(v))
        return out


@pytest.fixture()
def seeded_library(tmp_path):
    """init-v15 + 2 个实体家族 + 1 个关系家族（均有 active 版本）。"""
    lib = tmp_path / "library"
    lib.mkdir()
    cfg = tmp_path / "sc.json"
    cfg.write_text(json.dumps({"storage_path": str(lib), "llm": {"mock": True}}))
    runner = CliRunner()
    r = runner.invoke(cli, ["--config", str(cfg), "db", "init-v15"])
    assert r.exit_code == 0, r.output

    conn = sqlite3.connect(lib / "library.db")
    now = "2026-08-20T00:00:00"
    for fid, name in (("ef1", "Alice"), ("ef2", "Bob")):
        conn.execute(
            "INSERT INTO entity_families VALUES (?,?,?,?,?,?)",
            (fid, name, f"{name} 的简介", now, now, now))
        conn.execute(
            "INSERT INTO entity_observations "
            "(entity_id, entity_family_id, name, status, content, processed_at) "
            "VALUES (?,?,?,'active',?,?)",
            (f"eo_{fid}", fid, name, f"{name} 的简介", now))
    conn.execute(
        "INSERT INTO relation_families VALUES (?,?,?,1,?,?,?,?)",
        ("rf1", "ef1", "ef2", "knows", now, now, now))
    conn.execute(
        "INSERT INTO relation_assertions "
        "(relation_id, relation_family_id, subject_entity_id, object_entity_id, "
        " subject_entity_family_id, object_entity_family_id, status, processed_at) "
        "VALUES (?,?,?,?,?,?,'active',?)",
        ("ra1", "rf1", "eo_ef1", "eo_ef2", "ef1", "ef2", now))
    conn.commit()
    conn.close()
    return cfg, lib


def _run(runner, cfg, args):
    with patch.object(emb_mod, "EmbeddingClient", _StubClient):
        r = runner.invoke(cli, ["--json", "--config", str(cfg), "db", "backfill-embeddings"] + args)
    assert r.exit_code == 0, r.output
    return json.loads(r.output)


def test_backfill_stores_and_is_idempotent(seeded_library):
    cfg, lib = seeded_library
    runner = CliRunner()

    data = _run(runner, cfg, [])
    assert data["success"] is True
    assert data["data"]["entity"]["stored"] == 2
    assert data["data"]["relation"]["stored"] == 1

    conn = sqlite3.connect(lib / "library.db")
    rows = conn.execute(
        "SELECT owner_type, embedding_model, dimensions FROM embeddings").fetchall()
    conn.close()
    # embedding_model 与管线同源（client.model_name）——P2 按模型过滤的
    # 向量缓存能命中回填行
    assert {x[1] for x in rows} == {"stub-model"}
    assert all(x[2] == 8 for x in rows)
    assert len(rows) == 3

    # 幂等：已有 embedding 的版本不再重复编码
    data2 = _run(runner, cfg, [])
    assert data2["data"]["entity"]["stored"] == 0
    assert data2["data"]["relation"]["stored"] == 0


def test_backfill_human_mode_prints_progress(seeded_library, capsys):
    cfg, _ = seeded_library
    runner = CliRunner()
    with patch.object(emb_mod, "EmbeddingClient", _StubClient):
        r = runner.invoke(cli, ["--config", str(cfg), "db", "backfill-embeddings"])
    assert r.exit_code == 0, r.output
    assert "Entity" in r.output or "backfill complete" in r.output
