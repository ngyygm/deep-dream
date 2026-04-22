"""
Shared helpers for V2 remember pipeline quality tests.

Provides:
- _create_llm_client: Create LLMClient with real config
- _make_v2_processor: Create TemporalMemoryGraphProcessor in V2 mode
- _run_v2_extraction: Run V2 extraction only (steps 2-8)
- _fuzzy_match: Check if expected entity is in actual names
- _count_matches: Count how many expected entities match
"""

import pytest
from datetime import datetime


def _create_llm_client(real_llm_config, shared_embedding_client):
    """Create LLMClient with real config from service_config.json.

    Args:
        real_llm_config: Dict with keys llm_api_key, llm_model, etc.
        shared_embedding_client: EmbeddingClient instance

    Returns:
        LLMClient configured for real LLM calls.
    """
    from processor.llm.client import LLMClient
    kwargs = {
        "api_key": real_llm_config.get("llm_api_key"),
        "model_name": real_llm_config.get("llm_model"),
        "base_url": real_llm_config.get("llm_base_url"),
        "think_mode": real_llm_config.get("llm_think_mode", False),
        "max_tokens": real_llm_config.get("llm_max_tokens"),
        "context_window_tokens": real_llm_config.get("llm_context_window_tokens"),
        "max_llm_concurrency": 2,
        "embedding_client": shared_embedding_client,
        "content_snippet_length": 300,
        "relation_content_snippet_length": 200,
    }
    return LLMClient(**kwargs)


def _make_v2_processor(tmp_path, real_llm_config, shared_embedding_client, **overrides):
    """Create a V2-mode TemporalMemoryGraphProcessor with isolated storage.

    Args:
        tmp_path: Temporary directory for storage
        real_llm_config: Dict with LLM config keys
        shared_embedding_client: EmbeddingClient instance
        **overrides: Additional kwargs to override defaults

    Returns:
        TemporalMemoryGraphProcessor configured for V2 extraction mode.
    """
    from processor.pipeline.orchestrator import TemporalMemoryGraphProcessor
    defaults = dict(
        storage_path=str(tmp_path),
        window_size=1000,        # Large to avoid multi-window for most texts
        overlap=200,
        embedding_use_local=False,
        embedding_client=shared_embedding_client,
        max_llm_concurrency=1,   # Serial for deterministic testing
        remember_config={"mode": "v2"},
    )
    # real_llm_config keys (llm_api_key, llm_model, etc.) match __init__ param names
    defaults.update(real_llm_config)
    defaults.update(overrides)
    return TemporalMemoryGraphProcessor(**defaults)


def _run_v2_extraction(processor, text):
    """Run V2 extraction only (steps 2-8), bypass alignment/storage.

    Args:
        processor: TemporalMemoryGraphProcessor with V2 mode
        text: Input text to extract from

    Returns:
        (entities, relations) — lists of dicts
    """
    from processor.models import Episode
    episode = Episode(
        absolute_id="test_ep",
        content=text,
        event_time=datetime.now(),
        source_document="test_doc",
        activity_type="文档处理",
    )
    processor.remember_mode = "v2"
    return processor._extract_only_v2(
        episode, text, "test_doc",
        verbose=False, verbose_steps=False,
    )


def _fuzzy_match(expected: str, actual_names: set) -> bool:
    """Check if expected entity name is fuzzily matched in actual names.

    Uses substring matching (case-insensitive) to handle:
    - "Google" matching "Google Inc."
    - "TCP/IP" matching "TCP/IP协议"
    - "Kubernetes" matching "kubernetes"
    """
    expected_lower = expected.lower()
    for name in actual_names:
        name_lower = name.lower()
        if expected_lower in name_lower or name_lower in expected_lower:
            return True
    return False


def _count_matches(expected_list, actual_names):
    """Count how many expected entities are matched in actual names.

    Returns (matched_count, total_count, missing_list).
    """
    matched = 0
    missing = []
    for expected in expected_list:
        if _fuzzy_match(expected, actual_names):
            matched += 1
        else:
            missing.append(expected)
    return matched, len(expected_list), missing
