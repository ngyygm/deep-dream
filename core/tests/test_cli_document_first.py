import json
from pathlib import Path

import pytest

from core.cli import main


def _write_config(tmp_path: Path) -> Path:
    config = {
        "storage_path": str(tmp_path / "memory"),
        "storage": {"backend": "sqlite", "vector_dim": 8},
        "llm": {
            "api_key": "test",
            "model": "test",
            "base_url": "http://localhost:1/v1",
            "context_window_tokens": 1024,
        },
        "embedding": {"model": None, "device": "cpu"},
    }
    path = tmp_path / "service_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _run_cli(capsys, *args):
    code = main(list(args))
    captured = capsys.readouterr()
    assert code == 0
    return json.loads(captured.out)


def test_docs_map_and_search_external_file(tmp_path, capsys):
    config = _write_config(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("# Note\n\nAlpha memory line\n\n## Detail\nBeta concept line\n", encoding="utf-8")

    _run_cli(capsys, "--config", str(config), "graph", "create", "test")
    _run_cli(capsys, "--config", str(config), "vault", "index", str(vault), "--graph", "test")

    mapped = _run_cli(capsys, "--config", str(config), "docs", "map", str(note), "--graph", "test")
    assert mapped["success"] is True
    assert mapped["data"]["total"] == 1
    assert mapped["data"]["documents"][0]["source_mode"] == "external"
    assert mapped["data"]["documents"][0]["resolved_path"] == str(note.resolve())

    searched = _run_cli(capsys, "--config", str(config), "docs", "search", "Beta", "--graph", "test")
    assert searched["used"]["raw_files"] is True
    assert searched["data"]["total"] == 1
    assert searched["data"]["hits"][0]["document"]["line_start"] == 6
    assert searched["data"]["hits"][0]["verification"] == "raw_file"

    note.unlink()
    fallback = _run_cli(capsys, "--config", str(config), "docs", "search", "Beta", "--graph", "test")
    assert fallback["data"]["total"] == 1
    assert fallback["data"]["hits"][0]["verification"] == "snapshot"


def test_episode_from_file_and_episode_concepts_shape(tmp_path, capsys):
    config = _write_config(tmp_path)
    note = tmp_path / "doc.md"
    note.write_text("# Title\n\nIntro line\n\n## Section\nTarget line\n", encoding="utf-8")

    _run_cli(capsys, "--config", str(config), "graph", "create", "test")
    _run_cli(capsys, "--config", str(config), "vault", "index", str(note), "--graph", "test")

    episodes = _run_cli(
        capsys,
        "--config",
        str(config),
        "episode",
        "from-file",
        str(note),
        "--line",
        "6",
        "--graph",
        "test",
    )
    assert episodes["success"] is True
    assert episodes["data"]["total"] >= 1
    episode_id = episodes["data"]["episodes"][0]["episode_version_id"]

    concepts = _run_cli(capsys, "--config", str(config), "episode", "concepts", episode_id, "--graph", "test")
    assert concepts["success"] is True
    assert concepts["data"]["episode_id"] == episode_id
    assert isinstance(concepts["data"]["concepts"], list)


def test_sql_is_read_only(tmp_path, capsys):
    config = _write_config(tmp_path)
    _run_cli(capsys, "--config", str(config), "graph", "create", "test")

    rc = main(["--config", str(config), "sql", "--query", "DELETE FROM concept_family", "--graph", "test"])
    assert rc == 1


def test_explore_uses_agent_provided_terms_and_returns_evidence_cards(tmp_path, capsys):
    config = _write_config(tmp_path)
    note = tmp_path / "logic.md"
    note.write_text(
        "# Logic\n\n"
        "保持一致能让想法、信念和行为对齐。\n"
        "前后矛盾会削弱理性判断。\n",
        encoding="utf-8",
    )

    _run_cli(capsys, "--config", str(config), "graph", "create", "test")
    _run_cli(capsys, "--config", str(config), "vault", "index", str(note), "--graph", "test")

    result = _run_cli(
        capsys,
        "--config",
        str(config),
        "explore",
        "逻辑自洽",
        "--graph",
        "test",
        "--terms",
        "保持一致,前后矛盾",
        "--limit",
        "5",
        "--file-limit",
        "5",
    )

    terms = [item["term"] for item in result["data"]["query_terms"]]
    assert "保持一致" in terms
    assert result["data"]["coverage"]["file_hits"] >= 1
    assert result["data"]["coverage"]["evidence_cards"] >= 1
    assert result["data"]["evidence_cards"][0]["verification"] in {"raw_file", "source_text"}
