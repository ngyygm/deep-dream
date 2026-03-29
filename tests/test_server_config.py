from __future__ import annotations

import json
from pathlib import Path

from server.config import load_config


def test_load_config_normalizes_new_grouped_schema(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "host": "0.0.0.0",
        "port": 16200,
        "storage_path": "./graph",
        "llm": {
            "api_key": "test-key",
            "model": "gpt-4",
            "base_url": "http://127.0.0.1:9999/v1",
        },
        "runtime": {
            "concurrency": {
                "queue_workers": 2,
                "window_workers": 5,
            },
            "task": {
                "load_cache_memory": True,
            },
            "retry": {
                "queue_max_retries": 7,
                "queue_retry_delay_seconds": 3,
            },
        },
        "pipeline": {
            "search": {
                "similarity_threshold": 0.6,
                "max_similar_entities": 8,
            },
            "extraction": {
                "extraction_rounds": 2,
                "entity_post_enhancement": True,
                "prompt_memory_cache_max_chars": 1234,
            },
            "debug": {
                "distill_data_dir": "distill_out",
            },
        },
    }, ensure_ascii=False), encoding="utf-8")

    cfg = load_config(str(config_path))

    assert cfg["llm"]["context_window_tokens"] == 8000
    assert cfg["runtime"]["concurrency"]["queue_workers"] == 2
    assert cfg["runtime"]["concurrency"]["window_workers"] == 5
    assert cfg["runtime"]["task"]["load_cache_memory"] is True
    assert cfg["pipeline"]["search"]["similarity_threshold"] == 0.6
    assert cfg["pipeline"]["extraction"]["entity_post_enhancement"] is True
    assert cfg["pipeline"]["extraction"]["prompt_memory_cache_max_chars"] == 1234
    assert cfg["pipeline"]["debug"]["distill_data_dir"] == "distill_out"

    # backward-compatible flat keys
    assert cfg["remember_workers"] == 2
    assert cfg["pipeline"]["max_concurrent_windows"] == 5
    assert cfg["pipeline"]["load_cache_memory"] is True
    assert cfg["pipeline"]["similarity_threshold"] == 0.6


def test_runtime_workers_capped_by_llm_max_concurrency(tmp_path: Path):
    config_path = tmp_path / "config_cap.json"
    config_path.write_text(json.dumps({
        "llm": {
            "api_key": "test-key",
            "model": "gpt-4",
            "base_url": "http://127.0.0.1:9999/v1",
            "max_concurrency": 2,
        },
        "runtime": {
            "concurrency": {
                "queue_workers": 8,
                "window_workers": 16,
            },
        },
    }, ensure_ascii=False), encoding="utf-8")

    cfg = load_config(str(config_path))

    assert cfg["runtime"]["concurrency"]["queue_workers"] == 2
    assert cfg["runtime"]["concurrency"]["window_workers"] == 2
    assert cfg["remember_workers"] == 2
    assert cfg["pipeline"]["max_concurrent_windows"] == 2


def test_load_config_normalizes_legacy_flat_schema(tmp_path: Path):
    config_path = tmp_path / "config_legacy.json"
    config_path.write_text(json.dumps({
        "host": "0.0.0.0",
        "port": 16200,
        "storage_path": "./graph",
        "llm": {
            "api_key": "test-key",
            "model": "gpt-4",
            "base_url": "http://127.0.0.1:9999/v1",
        },
        "remember_workers": 3,
        "remember_max_retries": 4,
        "remember_retry_delay_seconds": 5,
        "pipeline": {
            "max_concurrent_windows": 9,
            "load_cache_memory": False,
            "similarity_threshold": 0.75,
            "entity_post_enhancement": True,
            "prompt_memory_cache_max_chars": 1500,
            "distill_data_dir": "legacy_distill",
        },
    }, ensure_ascii=False), encoding="utf-8")

    cfg = load_config(str(config_path))

    assert cfg["runtime"]["concurrency"]["queue_workers"] == 3
    assert cfg["runtime"]["concurrency"]["window_workers"] == 9
    assert cfg["runtime"]["retry"]["queue_max_retries"] == 4
    assert cfg["runtime"]["retry"]["queue_retry_delay_seconds"] == 5.0
    assert cfg["runtime"]["task"]["load_cache_memory"] is False
    assert cfg["pipeline"]["search"]["similarity_threshold"] == 0.75
    assert cfg["pipeline"]["extraction"]["entity_post_enhancement"] is True
    assert cfg["pipeline"]["extraction"]["prompt_memory_cache_max_chars"] == 1500
    assert cfg["pipeline"]["debug"]["distill_data_dir"] == "legacy_distill"
