"""Regression tests for the security/reliability fixes from the code review."""

import json
import sqlite3

import pytest
from datetime import datetime, timezone
from click.testing import CliRunner

from core.exceptions import ConfigError
from core.models import Episode
from core.server.api import create_app
from core.server.config import _validate_config, resolve_embedding_model
from core.server.config import _normalize_runtime_config
from core.cli.cmd_config import _coerce_value
from core.cli._main import cli
from core.storage.sqlite.library_manager import LibraryManager
from core.storage.sqlite.schema_v15 import init_schema_v15
from core.storage.sqlite.vault_indexer import index_vault, index_markdown_file


def test_config_validation_uses_nested_search_and_rejects_non_finite_values():
    with pytest.raises(ConfigError):
        _validate_config({
            "llm": {"mock": True},
            "pipeline": {"search": {"similarity_threshold": 1.1}},
        })
    with pytest.raises(ConfigError):
        _validate_config({"llm": {"mock": True}, "rate_limit_per_minute": "oops"})


def test_embedding_model_name_defaults_to_local_loading():
    assert resolve_embedding_model({"model": "Qwen/Qwen3-Embedding-0.6B"}) == (
        None, "Qwen/Qwen3-Embedding-0.6B", True
    )
    assert resolve_embedding_model({"model": "remote-name", "use_local": False}) == (
        None, "remote-name", False
    )
    with pytest.raises(ConfigError):
        _validate_config({
            "llm": {"mock": True},
            "pipeline": {"search": {"similarity_threshold": float("nan")}},
        })


def test_strict_auth_does_not_make_every_route_public():
    app = create_app(
        object(),
        config={"host": "127.0.0.1", "auth": {"enabled": True, "strict_mode": True}},
    )
    response = app.test_client().get("/api/v1/system/config")
    assert response.status_code == 401
    # Client-side SPA deep links are HTML-only and must remain refreshable;
    # their API calls are still protected separately.
    assert app.test_client().get("/settings").status_code == 200


def test_non_loopback_listener_forces_fail_closed_authentication():
    app = create_app(object(), config={"host": "0.0.0.0"})
    client = app.test_client()
    assert client.get("/api/v1/system/config").status_code == 401
    assert client.get(
        "/api/v1/system/config", headers={"X-API-Key": "dev-key-insecure"}
    ).status_code == 401


def test_graph_id_rejects_non_string_json_before_route_dispatch():
    app = create_app(object(), config={"host": "127.0.0.1", "auth": {"enabled": False}})
    response = app.test_client().post(
        "/api/v1/remember", json={"graph_id": 123, "text": "hello"}
    )
    assert response.status_code == 400
    assert response.is_json


def test_create_graph_rejects_non_object_json():
    app = create_app(object(), config={"host": "127.0.0.1", "auth": {"enabled": False}})
    response = app.test_client().post("/api/v1/graphs", json=[])
    assert response.status_code == 400
    assert response.is_json


def test_config_worker_auto_is_not_parsed_as_integer():
    assert _coerce_value("runtime.concurrency.queue_workers", "auto") == "auto"
    assert _coerce_value("runtime.concurrency.window_workers", "AUTO") == "auto"


def test_config_normalization_does_not_truncate_invalid_fractional_workers():
    normalized = _normalize_runtime_config({
        "runtime": {"concurrency": {"queue_workers": 1.5}},
        "llm": {"mock": True},
    })
    assert normalized["runtime"]["concurrency"]["queue_workers"] == 1.5
    with pytest.raises(ConfigError):
        _validate_config(normalized)


def test_config_set_rejects_invalid_candidate_without_replacing_file(tmp_path):
    config_path = tmp_path / "config.json"
    original = {"llm": {"mock": True}, "port": 16200}
    config_path.write_text(json.dumps(original), encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        ["--json", "--config", str(config_path), "config", "set", "port", "0", "--yes"],
    )
    assert result.exit_code == 2
    assert json.loads(config_path.read_text(encoding="utf-8")) == original
    assert json.loads(result.output)["success"] is False


def test_json_db_error_has_nonzero_exit_code(tmp_path):
    storage_path = tmp_path / "library"
    storage_path.mkdir()
    (storage_path / "library.db").write_bytes(b"")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "llm": {"mock": True},
        "storage_path": str(storage_path),
    }), encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        ["--json", "--config", str(config_path), "db", "quality"],
    )
    assert result.exit_code == 1
    assert json.loads(result.output)["success"] is False


def test_task_cli_health_url_uses_v1_path():
    from core.cli import cmd_task

    cmd_task._resolve_api_base._cached = "http://127.0.0.1:16200/api/v1"
    assert cmd_task._api_url("/health") == "http://127.0.0.1:16200/api/v1/health"


def test_configured_api_key_file_failure_does_not_enable_dev_key(tmp_path):
    from core.server import auth

    key_file = tmp_path / "keys.json"
    key_file.write_text("{bad", encoding="utf-8")
    old_keys = dict(auth._API_KEYS)
    old_allow_dev = auth._ALLOW_DEV_KEY
    try:
        auth.init_auth({"host": "127.0.0.1", "auth": {"api_keys_file": str(key_file)}})
        assert auth._validate_api_key("dev-key-insecure")[0] is False
    finally:
        auth._API_KEYS.clear()
        auth._API_KEYS.update(old_keys)
        auth._ALLOW_DEV_KEY = old_allow_dev


def test_cross_site_mutation_is_rejected_before_destructive_route():
    class Registry:
        def __init__(self):
            self.cleared = False

        def clear_graph(self, _graph_id):
            self.cleared = True

    registry = Registry()
    app = create_app(registry, config={"host": "127.0.0.1", "auth": {"enabled": False}})
    client = app.test_client()
    response = client.post(
        "/api/v1/graphs/library/clear",
        json={"confirm_graph_id": "library"},
        headers={"Origin": "https://attacker.example", "Sec-Fetch-Site": "cross-site"},
    )
    assert response.status_code == 403
    assert registry.cleared is False

    response = client.post(
        "/api/v1/graphs/library/clear",
        json={"confirm_graph_id": "library"},
        headers={"Origin": "http://localhost:3000", "Sec-Fetch-Site": "same-site"},
    )
    assert response.status_code == 403
    assert registry.cleared is False

    response = client.post("/api/v1/graphs/library/clear", json={})
    assert response.status_code == 400
    assert registry.cleared is False


def test_config_patch_preserves_redacted_secret_and_is_atomic(tmp_path):
    config_path = tmp_path / "service.json"
    config = {
        "_config_path": str(config_path),
        "auth": {"enabled": False},
        "llm": {"api_key": "sk-real-secret", "model": "demo"},
        "runtime": {"concurrency": {"queue_workers": 1}},
    }
    app = create_app(object(), config=config)
    client = app.test_client()

    response = client.patch(
        "/api/v1/system/config",
        json={"config": {
            "llm": {"api_key": "••••••••"},
            "runtime": {"concurrency": {"queue_workers": 2}},
        }},
    )
    assert response.status_code == 200
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["llm"]["api_key"] == "sk-real-secret"
    assert saved["runtime"]["concurrency"]["queue_workers"] == 2
    assert response.get_json()["data"]["config"]["llm"]["api_key"] == "••••••••"


def test_vault_identity_is_path_stable_and_force_reindex_is_safe(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    first = vault / "a.md"
    second = vault / "b.md"
    first.write_text("# Same\nbody", encoding="utf-8")
    second.write_text("# Same\nbody", encoding="utf-8")
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema_v15(conn)

    assert index_vault(conn, tmp_path / "library", str(vault))["indexed"] == 2
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
    # Repeating the same bytes with force must not violate the unique
    # (document_id, content_hash) constraint.
    assert index_vault(conn, tmp_path / "library", str(vault), force=True)["errors"] == 0

    first.write_text("# Changed\nnew body", encoding="utf-8")
    assert index_vault(conn, tmp_path / "library", str(vault))["errors"] == 0
    assert conn.execute("SELECT COUNT(*) FROM document_versions WHERE document_id = (SELECT document_id FROM documents WHERE absolute_path = ?)", (str(first.resolve()),)).fetchone()[0] == 2


def test_single_file_index_uses_absolute_identity(tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    (a_dir / "README.md").write_text("same", encoding="utf-8")
    (b_dir / "README.md").write_text("same", encoding="utf-8")
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema_v15(conn)
    library = tmp_path / "library"
    first = index_markdown_file(conn, library, str(a_dir / "README.md"))
    second = index_markdown_file(conn, library, str(b_dir / "README.md"))
    assert first["status"] == second["status"] == "indexed"
    assert first["document_id"] != second["document_id"]


def test_targeted_repair_preserves_full_document_and_retires_old_window(tmp_path):
    manager = LibraryManager(str(tmp_path / "library"))
    full_text = "A" * 1200 + "B" * 1200
    now = datetime.now(timezone.utc)
    first = Episode("ep-original", full_text[:1200], now, "repair.md", now)
    manager.save_episode(first, text=full_text, doc_hash="window-0", chunk_index=0)
    doc = manager.list_documents(limit=1)[0]
    version_id = doc["document_version_id"]

    replacement = Episode("ep-repaired", full_text[1200:], now, "repair.md", now)
    manager.save_episode(
        replacement,
        text=full_text[1200:],
        doc_hash="window-0-repair",
        chunk_index=0,
        override_doc_id=doc["document_id"],
    )

    assert manager.get_document_content(version_id)["content"] == full_text
    assert manager._conn().execute(
        "SELECT status FROM episodes WHERE episode_id='ep-original'"
    ).fetchone()[0] == "superseded"
    assert manager._conn().execute(
        "SELECT COUNT(*) FROM document_versions WHERE document_id=?",
        (doc["document_id"],),
    ).fetchone()[0] == 1
    manager.close()


def test_retrieval_slices_use_disjoint_indexes_and_rollback_safe_window_writes(tmp_path):
    manager = LibraryManager(str(tmp_path / "library"))
    now = datetime.now(timezone.utc)
    try:
        first = "aaaa\nbbbb\ncccc\ndddd"
        second = "eeee\nffff\ngggg\nhhhh"
        manager.save_episode(
            Episode("slice-ep-0", first, now, "slices.md", now),
            text=first, doc_hash="slice-0", chunk_index=0,
            retrieval_slice_chars=10, run_id="slice-run",
        )
        manager.save_episode(
            Episode("slice-ep-1", second, now, "slices.md", now),
            text=second, doc_hash="slice-1", chunk_index=1,
            retrieval_slice_chars=10, run_id="slice-run",
        )
        rows = manager._conn().execute(
            "SELECT chunk_index, episode_type, status FROM episodes "
            "WHERE document_version_id=(SELECT document_version_id FROM episodes WHERE episode_id=?)",
            ("slice-ep-0",),
        ).fetchall()
        assert [r[0] for r in rows if r[1] == "" and r[2] == "active"] == [0, 1]
        assert all(r[0] < 0 for r in rows if r[1] == "retrieval_slice" and r[2] == "active")
        assert manager._conn().in_transaction is False
    finally:
        manager.close()


def test_merge_same_episode_observation_keeps_target_and_rolls_back_on_error(tmp_path):
    manager = LibraryManager(str(tmp_path / "library"))
    now = datetime.now(timezone.utc)
    ts = "2026-01-01T00:00:00+00:00"
    try:
        manager.save_episode(Episode("merge-ep", "x", now, "merge.md", now), text="x", doc_hash="merge", chunk_index=0)
        conn = manager._conn()
        conn.executemany(
            "INSERT INTO entity_families(entity_family_id,canonical_name,created_at,updated_at) VALUES(?,?,?,?)",
            [("merge-target", "target", ts, ts), ("merge-source", "source", ts, ts)],
        )
        conn.executemany(
            "INSERT INTO entity_observations(entity_id,entity_family_id,episode_id,name,processed_at) VALUES(?,?,?,?,?)",
            [("merge-target-obs", "merge-target", "merge-ep", "target", ts),
             ("merge-source-obs", "merge-source", "merge-ep", "source", ts)],
        )
        conn.commit()
        result = manager.merge_entity_families("merge-target", ["merge-source"])
        assert result["merged"] == ["merge-source"]
        rows = conn.execute(
            "SELECT entity_id, entity_family_id, status FROM entity_observations ORDER BY entity_id"
        ).fetchall()
        assert [tuple(r) for r in rows] == [
            ("merge-source-obs", "merge-target", "superseded"),
            ("merge-target-obs", "merge-target", "active"),
        ]
        assert conn.in_transaction is False
    finally:
        manager.close()


def test_relation_mentions_are_stored_without_entity_fk_mismatch(tmp_path):
    manager = LibraryManager(str(tmp_path / "library"))
    now = datetime.now(timezone.utc)
    ts = "2026-01-01T00:00:00+00:00"
    try:
        manager.save_episode(Episode("rel-ep", "x", now, "rel.md", now), text="x", doc_hash="rel", chunk_index=0)
        conn = manager._conn()
        conn.executemany(
            "INSERT INTO entity_families(entity_family_id,canonical_name,created_at,updated_at) VALUES(?,?,?,?)",
            [("rel-a", "a", ts, ts), ("rel-b", "b", ts, ts)],
        )
        conn.executemany(
            "INSERT INTO entity_observations(entity_id,entity_family_id,episode_id,name,processed_at) VALUES(?,?,?,?,?)",
            [("rel-oa", "rel-a", "rel-ep", "a", ts), ("rel-ob", "rel-b", "rel-ep", "b", ts)],
        )
        conn.execute(
            "INSERT INTO relation_families(relation_family_id,subject_entity_family_id,object_entity_family_id,canonical_content,created_at,updated_at) VALUES(?,?,?,?,?,?)",
            ("rel-fam", "rel-a", "rel-b", "a-b", ts, ts),
        )
        conn.execute(
            "INSERT INTO relation_assertions(relation_id,relation_family_id,episode_id,subject_entity_id,object_entity_id,subject_entity_family_id,object_entity_family_id,content,processed_at) VALUES(?,?,?,?,?,?,?,?,?)",
            ("rel-assert", "rel-fam", "rel-ep", "rel-oa", "rel-ob", "rel-a", "rel-b", "a-b", ts),
        )
        conn.commit()
        assert manager.save_episode_mentions("rel-ep", ["rel-assert"], target_type="relation") == 1
        assert conn.execute("SELECT COUNT(*) FROM relation_mentions").fetchone()[0] == 1
    finally:
        manager.close()


def test_stale_task_generation_cannot_publish_progress():
    from core.server.task_journal import RememberTask
    from core.server.task_queue import RememberTaskQueue
    import threading

    task = RememberTask(
        task_id="generation-task", text="x", source_name="x.md",
        load_cache=False, control_action=None, event_time=None, original_path="",
    )
    queue = object.__new__(RememberTaskQueue)
    queue._lock = threading.RLock()
    queue._tasks = {task.task_id: task}
    queue._update_task_progress(task, status="running", execution_generation=0)
    task.execution_generation = 1
    queue._update_task_progress(task, status="completed", execution_generation=0)
    assert task.status == "running"


def test_content_patches_are_persisted_idempotently(tmp_path):
    from core.models import ContentPatch
    from core.storage.sqlite.library_manager import LibraryManager

    manager = LibraryManager(str(tmp_path / "library"))
    patch = ContentPatch(
        uuid="patch-1",
        target_type="Entity",
        target_absolute_id="entity-2",
        target_family_id="family-1",
        section_key="facts",
        change_type="modified",
        old_hash="old",
        new_hash="new",
        diff_summary="facts changed",
        source_document="note.md",
        event_time=datetime.now(timezone.utc),
    )
    assert manager.save_content_patches([patch]) == 1
    assert manager.save_content_patches([patch]) == 0
    rows = manager.get_content_patches(target_family_id="family-1")
    assert len(rows) == 1
    assert rows[0]["new_hash"] == "new"
    manager.close()
