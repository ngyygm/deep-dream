"""P6.1：embedding 模型一致性校验。

纯函数判定 + LibraryManager 报告 + doctor 只读检查三层，
不加载真实 embedding 模型、不发网络请求。
"""
from types import SimpleNamespace

from core.storage.sqlite.helpers import embedding_consistency


def _mgr(tmp_path):
    from core.storage.sqlite.library_manager import LibraryManager
    return LibraryManager(str(tmp_path / "lib"))


def _seed_embeddings(conn, rows):
    """rows: [(model, count)] —— owner/text 字段取最小合法值。"""
    from core.storage.sqlite.helpers import now_utc_str

    ts = now_utc_str()
    for model, count in rows:
        for i in range(count):
            conn.execute(
                "INSERT INTO embeddings (owner_type, owner_id, text_kind, text_hash,"
                " embedding_model, dimensions, vector, created_at)"
                " VALUES ('entity_obs', ?, 'memory_text', ?, ?, 4, X'00000000', ?)",
                (f"e-{model}-{i}", f"h-{model}-{i}", model, ts),
            )
    conn.commit()


# ------------------------------------------------------------------
# 纯函数
# ------------------------------------------------------------------

def test_consistency_empty_library():
    ok, warn = embedding_consistency("m-a", {})
    assert ok is True and warn is None


def test_consistency_active_present():
    ok, warn = embedding_consistency("m-a", {"m-a": 10})
    assert ok is True and warn is None


def test_consistency_active_missing_points_to_backfill():
    ok, warn = embedding_consistency("m-b", {"m-a": 7, "m-old": 2})
    assert ok is False
    assert "backfill-embeddings" in warn
    assert "m-a" in warn  # 指明存量多数模型


def test_consistency_fragmentation_warns_but_consistent():
    ok, warn = embedding_consistency("m-a", {"m-a": 5, "m-old": 3})
    assert ok is True
    assert warn is not None and "m-old" in warn


# ------------------------------------------------------------------
# LibraryManager.embedding_model_report
# ------------------------------------------------------------------

def test_report_consistent(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.embedding_client = SimpleNamespace(model_name="m-a", model_path=None)
    _seed_embeddings(mgr._conn(), [("m-a", 3)])
    report = mgr.embedding_model_report()
    assert report["active"] == "m-a"
    assert report["models"] == {"m-a": 3}
    assert report["consistent"] is True and report["warning"] is None


def test_report_mismatch(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.embedding_client = SimpleNamespace(model_name=None, model_path="m-b")
    _seed_embeddings(mgr._conn(), [("m-a", 4), ("m-old", 1)])
    report = mgr.embedding_model_report()
    # model_name 优先、model_path 兜底——与写入路径同源
    assert report["active"] == "m-b"
    assert report["consistent"] is False
    assert "backfill-embeddings" in report["warning"]


def test_report_empty_library(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.embedding_client = None
    report = mgr.embedding_model_report()
    assert report["active"] == "unknown"
    assert report["consistent"] is True and report["warning"] is None


# ------------------------------------------------------------------
# doctor 只读检查
# ------------------------------------------------------------------

def test_doctor_check_embedding_models(tmp_path):
    from core.cli.cmd_doctor import _check_embedding_models

    config = {"embedding": {"model": "m-b"}}
    result = _check_embedding_models(tmp_path, config)
    # 库文件不存在：不报错、库视为一致
    assert result["active"] == "m-b"
    assert result["consistent"] is True and result["error"] is None

    # 建库并种入另一模型的向量 → 不一致警告
    mgr = _mgr(tmp_path)
    mgr.embedding_client = SimpleNamespace(model_name="m-a", model_path=None)
    _seed_embeddings(mgr._conn(), [("m-a", 2)])
    mgr.close()
    result = _check_embedding_models(tmp_path / "lib", config)
    assert result["models"] == {"m-a": 2}
    assert result["consistent"] is False
    assert "backfill-embeddings" in result["warning"]
