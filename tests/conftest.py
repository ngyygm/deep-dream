"""
Shared fixtures for Deep Dream tests.

When USE_REAL_LLM=1 is set, the real LLM backend from service_config.json
is used for pipeline tests.  A module-scoped EmbeddingClient is shared so
the model is loaded only once per session.
"""

import json
import os
from pathlib import Path

import pytest


_CONFIG_PATH = Path(__file__).resolve().parent.parent / "service_config.json"


def _load_real_llm_config():
    """Read llm block from service_config.json."""
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    llm = cfg.get("llm", {})
    out = {}
    if llm.get("api_key"):
        out["llm_api_key"] = llm["api_key"]
    if llm.get("model"):
        out["llm_model"] = llm["model"]
    if llm.get("base_url"):
        out["llm_base_url"] = llm["base_url"]
    if llm.get("think") is not None:
        out["llm_think_mode"] = llm["think"]
    if llm.get("max_tokens"):
        out["llm_max_tokens"] = llm["max_tokens"]
    if llm.get("context_window_tokens"):
        out["llm_context_window_tokens"] = llm["context_window_tokens"]
    return out


def pytest_configure(config):
    """Register the real_llm marker."""
    config.addinivalue_line(
        "markers", "real_llm: test requires USE_REAL_LLM=1"
    )


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.real_llm tests when USE_REAL_LLM is not set."""
    if os.environ.get("USE_REAL_LLM") == "1":
        return
    skip_real = pytest.mark.skip(reason="Set USE_REAL_LLM=1 to run")
    for item in items:
        if "real_llm" in item.keywords:
            item.add_marker(skip_real)


@pytest.fixture(scope="session")
def real_llm_config():
    """Return real LLM config dict (empty when USE_REAL_LLM is unset)."""
    if os.environ.get("USE_REAL_LLM") != "1":
        return {}
    return _load_real_llm_config()


@pytest.fixture(scope="session")
def shared_embedding_client():
    """Load embedding model once per test session (GPU if configured).

    Only loads the model when USE_REAL_LLM=1 — otherwise returns None.
    This prevents blocking the entire test session on GPU model loading.
    """
    if os.environ.get("USE_REAL_LLM") != "1":
        return None
    from processor.storage.embedding import EmbeddingClient
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        emb = cfg.get("embedding", {})
        model_path = emb.get("model")
        device = emb.get("device", "cpu")
        if model_path:
            client = EmbeddingClient(model_path=model_path, device=device, use_local=True)
            return client
    client = EmbeddingClient(use_local=False)
    return client
