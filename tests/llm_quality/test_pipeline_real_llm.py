"""
Real LLM pipeline tests.

These tests call the actual LLM backend (GLM-4-Flash) and use GPU embedding.
Run with:  USE_REAL_LLM=1 pytest tests/test_pipeline_real_llm.py -v

Without USE_REAL_LLM=1, all tests in this file are automatically skipped.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from processor.pipeline.orchestrator import TemporalMemoryGraphProcessor
from processor.models import Episode, Entity


def _make_real_processor(tmp_path, real_llm_config, shared_embedding_client, **overrides):
    """Create a processor with real LLM + shared embedding client.

    Uses a larger context_window_tokens (32k) to avoid budget-exceeded errors
    during multi-round extraction with real models.
    """
    defaults = dict(
        storage_path=str(tmp_path),
        window_size=100,
        overlap=20,
        embedding_use_local=False,
        embedding_client=shared_embedding_client,
        max_llm_concurrency=2,
        llm_context_window_tokens=128000,
    )
    defaults.update(real_llm_config)
    # Force large context window regardless of service_config's 7000
    defaults["llm_context_window_tokens"] = 128000
    defaults.update(overrides)
    return TemporalMemoryGraphProcessor(**defaults)


# ===================================================================
# 1. remember_text with short text (single window) — real LLM
# ===================================================================

@pytest.mark.real_llm
class TestRememberTextShortReal:

    def test_returns_dict_with_expected_keys(self, tmp_path, real_llm_config, shared_embedding_client):
        proc = _make_real_processor(tmp_path, real_llm_config, shared_embedding_client)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        result = proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        assert isinstance(result, dict)
        assert "episode_id" in result
        assert "chunks_processed" in result
        assert "storage_path" in result

    def test_single_chunk_processed(self, tmp_path, real_llm_config, shared_embedding_client):
        proc = _make_real_processor(tmp_path, real_llm_config, shared_embedding_client)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        result = proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        assert result["chunks_processed"] == 1

    def test_episode_id_is_string(self, tmp_path, real_llm_config, shared_embedding_client):
        proc = _make_real_processor(tmp_path, real_llm_config, shared_embedding_client)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        result = proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        assert isinstance(result["episode_id"], str)
        assert len(result["episode_id"]) > 0

    def test_storage_path_matches(self, tmp_path, real_llm_config, shared_embedding_client):
        proc = _make_real_processor(tmp_path, real_llm_config, shared_embedding_client)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        result = proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        assert result["storage_path"] == str(proc.storage.storage_path)

    def test_entities_extracted(self, tmp_path, real_llm_config, shared_embedding_client):
        proc = _make_real_processor(tmp_path, real_llm_config, shared_embedding_client)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        entities = proc.storage.get_all_entities()
        assert isinstance(entities, list)
        # Real LLM should extract at least one entity from this text
        assert len(entities) > 0

    def test_relations_query_works_after_processing(self, tmp_path, real_llm_config, shared_embedding_client):
        proc = _make_real_processor(tmp_path, real_llm_config, shared_embedding_client)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        relations = proc.storage.get_all_relations()
        assert isinstance(relations, list)


# ===================================================================
# 2. remember_text with multi-window text — real LLM
# ===================================================================

@pytest.mark.real_llm
class TestRememberTextMultiWindowReal:

    def test_multiple_chunks_processed(self, tmp_path, real_llm_config, shared_embedding_client):
        proc = _make_real_processor(tmp_path, real_llm_config, shared_embedding_client)
        text = (
            "Alice walked through the bustling market on a sunny morning. "
            "She greeted Bob who was selling fresh oranges at his fruit stand. "
            "The market was filled with colorful stalls and cheerful vendors. "
            "Alice picked up some ripe tomatoes and a bag of green apples. "
            "Bob recommended the local honey from the nearby beekeeper Charlie."
        )
        assert len(text) > 100
        result = proc.remember_text(text, doc_name="test_multi", verbose=False, verbose_steps=False)
        assert result["chunks_processed"] > 1

    def test_entities_extracted_from_multi_window(self, tmp_path, real_llm_config, shared_embedding_client):
        proc = _make_real_processor(tmp_path, real_llm_config, shared_embedding_client)
        text = (
            "Alice walked through the bustling market on a sunny morning. "
            "She greeted Bob who was selling fresh oranges at his fruit stand. "
            "The market was filled with colorful stalls and cheerful vendors. "
            "Alice picked up some ripe tomatoes and a bag of green apples. "
            "Bob recommended the local honey from the nearby beekeeper Charlie."
        )
        proc.remember_text(text, doc_name="test_multi", verbose=False, verbose_steps=False)
        entities = proc.storage.get_all_entities()
        assert isinstance(entities, list)
        # Real LLM should extract named entities (Alice, Bob, Charlie, etc.)
        assert len(entities) > 0


# ===================================================================
# 3. Edge cases — real LLM
# ===================================================================

@pytest.mark.real_llm
class TestEdgeCasesReal:

    def test_empty_string(self, tmp_path, real_llm_config, shared_embedding_client):
        proc = _make_real_processor(tmp_path, real_llm_config, shared_embedding_client)
        result = proc.remember_text("", doc_name="empty", verbose=False, verbose_steps=False)
        assert result["chunks_processed"] == 1

    def test_whitespace_only(self, tmp_path, real_llm_config, shared_embedding_client):
        proc = _make_real_processor(tmp_path, real_llm_config, shared_embedding_client)
        text = "   \n\t  \n   "
        result = proc.remember_text(text, doc_name="whitespace", verbose=False, verbose_steps=False)
        assert result["chunks_processed"] >= 1

    def test_unicode_and_emoji(self, tmp_path, real_llm_config, shared_embedding_client):
        proc = _make_real_processor(tmp_path, real_llm_config, shared_embedding_client)
        text = "Hello world! This is a test with unicode characters."
        result = proc.remember_text(text, doc_name="unicode", verbose=False, verbose_steps=False)
        assert result["chunks_processed"] >= 1
        assert isinstance(result["episode_id"], str)


# ===================================================================
# 4. remember_phase2_windows — real LLM
# ===================================================================

@pytest.mark.real_llm
class TestRememberPhase2WindowsReal:

    def test_returns_dict_with_expected_keys(self, tmp_path, real_llm_config, shared_embedding_client):
        proc = _make_real_processor(tmp_path, real_llm_config, shared_embedding_client)
        text = "Bob visited the science museum and learned about quantum physics experiments."
        result = proc.remember_phase2_windows(
            text, doc_name="phase2_test", verbose=False,
        )
        assert isinstance(result, dict)
        assert "episode_id" in result
        assert "chunks_processed" in result
        assert "storage_path" in result

    def test_chunks_processed_at_least_one(self, tmp_path, real_llm_config, shared_embedding_client):
        proc = _make_real_processor(tmp_path, real_llm_config, shared_embedding_client)
        text = "Bob visited the science museum and learned about quantum physics experiments."
        result = proc.remember_phase2_windows(
            text, doc_name="phase2_test", verbose=False,
        )
        assert result["chunks_processed"] >= 1

    def test_episode_id_is_string(self, tmp_path, real_llm_config, shared_embedding_client):
        proc = _make_real_processor(tmp_path, real_llm_config, shared_embedding_client)
        text = "Bob visited the science museum and learned about quantum physics experiments."
        result = proc.remember_phase2_windows(
            text, doc_name="phase2_test", verbose=False,
        )
        assert isinstance(result["episode_id"], str)
        assert len(result["episode_id"]) > 0

    def test_with_overall_cache(self, tmp_path, real_llm_config, shared_embedding_client):
        from datetime import datetime
        proc = _make_real_processor(tmp_path, real_llm_config, shared_embedding_client)
        overall = Episode(
            absolute_id="overall_test_001",
            content="Test overall memory content for phase2.",
            event_time=datetime.now(),
            source_document="test_doc",
            activity_type="文档整体",
        )
        text = "The quick brown fox jumps over the lazy dog in the garden."
        result = proc.remember_phase2_windows(
            text, doc_name="phase2_with_cache",
            verbose=False, overall_cache=overall,
        )
        assert result["chunks_processed"] >= 1
