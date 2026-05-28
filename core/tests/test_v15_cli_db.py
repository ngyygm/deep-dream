"""Verify V1.5 CLI db subcommands."""

import json
import os
import sqlite3
import pytest

from core.cli import main


@pytest.fixture
def test_library(tmp_path):
    lib_dir = str(tmp_path / "library")
    os.makedirs(lib_dir)
    config = {
        "storage_path": lib_dir,
        "llm": {"api_key": "test"},
    }
    config_path = str(tmp_path / "service_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)
    return lib_dir, config_path


def _run_cli(config_path, *args):
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["deep-dream", "--config", config_path] + list(args)
        rc = main()
        return rc
    finally:
        sys.argv = old_argv


def _parse_last_json(stdout: str):
    """Parse the last JSON object from CLI output (may have warnings before it)."""
    # Find the last complete JSON object by finding the last top-level {
    text = stdout.strip()
    idx = text.rfind("\n{")
    if idx == -1 and text.startswith("{"):
        return json.loads(text)
    if idx == -1:
        raise ValueError(f"No JSON found in output: {text[:200]}")
    return json.loads(text[idx + 1:])


def _run_cli_capture(config_path, *args, capsys):
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["deep-dream", "--config", config_path] + list(args)
        main()
        captured = capsys.readouterr()
        return _parse_last_json(captured.out)
    finally:
        sys.argv = old_argv


def test_db_init_v15(test_library, capsys):
    lib_dir, config_path = test_library
    result = _run_cli_capture(config_path, "db", "init-v15", capsys=capsys)
    assert result["success"] is True
    assert result["action"] == "init-v15"
    db_path = os.path.join(lib_dir, "graph.db")
    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()}
    assert "documents" in tables
    assert "entity_families" in tables
    conn.close()


def test_db_validate_empty_db(test_library, capsys):
    lib_dir, config_path = test_library
    _run_cli(config_path, "db", "init-v15")
    result = _run_cli_capture(config_path, "db", "validate", capsys=capsys)
    assert result["success"] is True
    assert result["violations"] == 0


def test_db_rebuild_fts(test_library, capsys):
    lib_dir, config_path = test_library
    _run_cli(config_path, "db", "init-v15")
    result = _run_cli_capture(config_path, "db", "rebuild-fts", capsys=capsys)
    assert result["success"] is True


def test_db_compact(test_library, capsys):
    lib_dir, config_path = test_library
    _run_cli(config_path, "db", "init-v15")
    result = _run_cli_capture(config_path, "db", "compact", capsys=capsys)
    assert result["success"] is True


def test_db_vacuum_embeddings_empty(test_library, capsys):
    lib_dir, config_path = test_library
    _run_cli(config_path, "db", "init-v15")
    result = _run_cli_capture(config_path, "db", "vacuum-embeddings", capsys=capsys)
    assert result["success"] is True


def test_db_reset_v15_backup_old(test_library, capsys):
    lib_dir, config_path = test_library
    _run_cli(config_path, "db", "init-v15")
    db_path = os.path.join(lib_dir, "graph.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO documents (document_id, status, created_at, updated_at) "
        "VALUES ('d1', 'active', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')"
    )
    conn.commit()
    conn.close()

    result = _run_cli_capture(config_path, "db", "reset-v15", "--backup-old", capsys=capsys)
    assert result["success"] is True
    assert "backup" in result

    backup_file = result["backup"]
    assert os.path.exists(backup_file)

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert count == 0
    conn.close()
