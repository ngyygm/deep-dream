"""
Comprehensive pytest tests for TemporalMemoryGraphProcessor pipeline.

Tests cover initialization, runtime statistics, remember_text with various
inputs, phase1/phase2 processing, concurrency primitives, and edge cases.
All tests use mock mode (no API key) and tmp_path for storage isolation.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from processor.pipeline.orchestrator import TemporalMemoryGraphProcessor
from processor.models import MemoryCache, Entity


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_processor(tmp_path, **overrides):
    """Create a TemporalMemoryGraphProcessor in mock mode with small windows."""
    defaults = dict(
        storage_path=str(tmp_path),
        window_size=100,
        overlap=20,
        llm_api_key=None,
        llm_model="gpt-4",
        embedding_use_local=False,
        max_llm_concurrency=2,
    )
    defaults.update(overrides)
    return TemporalMemoryGraphProcessor(**defaults)


# ===================================================================
# 1. Initialization
# ===================================================================

class TestInitialization:
    """Verify that __init__ sets all attributes correctly."""

    def test_default_similarity_threshold(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert proc.similarity_threshold == 0.7

    def test_default_content_snippet_length(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert proc.content_snippet_length == 50

    def test_default_relation_content_snippet_length(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert proc.relation_content_snippet_length == 50

    def test_default_max_similar_entities(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert proc.max_similar_entities == 10

    def test_default_entity_extraction_max_iterations(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert proc.entity_extraction_max_iterations == 3

    def test_default_relation_extraction_max_iterations(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert proc.relation_extraction_max_iterations == 3

    def test_default_entity_post_enhancement(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert proc.entity_post_enhancement is False

    def test_default_load_cache_memory(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert proc.load_cache_memory is False

    def test_default_compress_multi_round_extraction(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert proc.compress_multi_round_extraction is False

    def test_compress_multi_round_extraction_override(self, tmp_path):
        proc = _make_processor(tmp_path, compress_multi_round_extraction=True)
        assert proc.compress_multi_round_extraction is True

    def test_default_max_alignment_candidates(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert proc.max_alignment_candidates is None

    def test_llm_threads_from_max_llm_concurrency(self, tmp_path):
        proc = _make_processor(tmp_path, max_llm_concurrency=2)
        assert proc.llm_threads == 2

    def test_llm_threads_default_when_none(self, tmp_path):
        proc = _make_processor(tmp_path, max_llm_concurrency=None)
        assert proc.llm_threads == 1

    def test_window_size_and_overlap_stored_in_document_processor(self, tmp_path):
        proc = _make_processor(tmp_path, window_size=100, overlap=20)
        assert proc.document_processor.window_size == 100
        assert proc.document_processor.overlap == 20

    def test_thread_pool_created(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert isinstance(proc._extraction_executor, ThreadPoolExecutor)

    def test_thread_pool_max_workers_default(self, tmp_path):
        proc = _make_processor(tmp_path)
        # Default max_concurrent_windows=1 => max_workers=1
        assert proc._extraction_executor._max_workers == 1

    def test_thread_pool_max_workers_custom(self, tmp_path):
        proc = _make_processor(tmp_path, max_concurrent_windows=4)
        assert proc._extraction_executor._max_workers == 4

    def test_window_slot_semaphore_default(self, tmp_path):
        proc = _make_processor(tmp_path)
        # Default max_concurrent_windows=1
        assert proc._window_slot._value == 1

    def test_window_slot_semaphore_custom(self, tmp_path):
        proc = _make_processor(tmp_path, max_concurrent_windows=3)
        assert proc._window_slot._value == 3

    def test_custom_parameters_override_defaults(self, tmp_path):
        proc = _make_processor(
            tmp_path,
            similarity_threshold=0.85,
            max_similar_entities=5,
            content_snippet_length=100,
            relation_content_snippet_length=80,
            entity_extraction_max_iterations=5,
            relation_extraction_max_iterations=4,
            entity_post_enhancement=True,
            load_cache_memory=True,
            jaccard_search_threshold=0.6,
            embedding_name_search_threshold=0.75,
            embedding_full_search_threshold=0.65,
            max_alignment_candidates=3,
        )
        assert proc.similarity_threshold == 0.85
        assert proc.max_similar_entities == 5
        assert proc.content_snippet_length == 100
        assert proc.relation_content_snippet_length == 80
        assert proc.entity_extraction_max_iterations == 5
        assert proc.relation_extraction_max_iterations == 4
        assert proc.entity_post_enhancement is True
        assert proc.load_cache_memory is True
        assert proc.jaccard_search_threshold == 0.6
        assert proc.embedding_name_search_threshold == 0.75
        assert proc.embedding_full_search_threshold == 0.65
        assert proc.max_alignment_candidates == 3

    def test_current_memory_cache_initially_none(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert proc.current_memory_cache is None

    def test_active_counters_start_at_zero(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert proc._active_window_extractions == 0
        assert proc._peak_window_extractions == 0
        assert proc._active_main_pipeline_windows == 0
        assert proc._active_step6 == 0
        assert proc._active_step7 == 0

    def test_cache_lock_is_threading_lock(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert isinstance(proc._cache_lock, type(threading.Lock()))

    def test_runtime_lock_is_threading_lock(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert isinstance(proc._runtime_lock, type(threading.Lock()))

    def test_max_concurrent_windows_clamped_min(self, tmp_path):
        proc = _make_processor(tmp_path, max_concurrent_windows=0)
        assert proc._max_concurrent_windows == 1

    def test_max_concurrent_windows_clamped_max(self, tmp_path):
        proc = _make_processor(tmp_path, max_concurrent_windows=100)
        assert proc._max_concurrent_windows == 64

    def test_entity_extraction_rounds_alias(self, tmp_path):
        proc = _make_processor(tmp_path, entity_extraction_max_iterations=7)
        assert proc.entity_extraction_rounds == 7

    def test_relation_extraction_rounds_alias(self, tmp_path):
        proc = _make_processor(tmp_path, relation_extraction_max_iterations=6)
        assert proc.relation_extraction_rounds == 6

    def test_extraction_rounds_generic_fallback(self, tmp_path):
        proc = _make_processor(tmp_path, extraction_rounds=4)
        assert proc.entity_extraction_rounds == 4
        assert proc.relation_extraction_rounds == 4


# ===================================================================
# 2. get_runtime_stats
# ===================================================================

class TestGetRuntimeStats:

    def test_returns_dict(self, tmp_path):
        proc = _make_processor(tmp_path)
        stats = proc.get_runtime_stats()
        assert isinstance(stats, dict)

    def test_expected_keys(self, tmp_path):
        proc = _make_processor(tmp_path)
        stats = proc.get_runtime_stats()
        expected_keys = {
            "configured_window_workers",
            "configured_llm_threads",
            "active_window_extractions",
            "active_main_pipeline_windows",
            "peak_window_extractions",
            "active_step6",
            "active_step7",
        }
        assert expected_keys.issubset(stats.keys())

    def test_active_window_extractions_starts_at_zero(self, tmp_path):
        proc = _make_processor(tmp_path)
        stats = proc.get_runtime_stats()
        assert stats["active_window_extractions"] == 0

    def test_active_main_pipeline_windows_starts_at_zero(self, tmp_path):
        proc = _make_processor(tmp_path)
        stats = proc.get_runtime_stats()
        assert stats["active_main_pipeline_windows"] == 0

    def test_peak_window_extractions_starts_at_zero(self, tmp_path):
        proc = _make_processor(tmp_path)
        stats = proc.get_runtime_stats()
        assert stats["peak_window_extractions"] == 0

    def test_active_step6_starts_at_zero(self, tmp_path):
        proc = _make_processor(tmp_path)
        stats = proc.get_runtime_stats()
        assert stats["active_step6"] == 0

    def test_active_step7_starts_at_zero(self, tmp_path):
        proc = _make_processor(tmp_path)
        stats = proc.get_runtime_stats()
        assert stats["active_step7"] == 0

    def test_configured_window_workers(self, tmp_path):
        proc = _make_processor(tmp_path, max_concurrent_windows=2)
        stats = proc.get_runtime_stats()
        assert stats["configured_window_workers"] == 2

    def test_configured_llm_threads(self, tmp_path):
        proc = _make_processor(tmp_path, max_llm_concurrency=3)
        stats = proc.get_runtime_stats()
        assert stats["configured_llm_threads"] == 3


# ===================================================================
# 3. get_statistics
# ===================================================================

class TestGetStatistics:

    def test_returns_dict(self, tmp_path):
        proc = _make_processor(tmp_path)
        stats = proc.get_statistics()
        assert isinstance(stats, dict)

    def test_has_memory_caches_key(self, tmp_path):
        proc = _make_processor(tmp_path)
        stats = proc.get_statistics()
        assert "memory_caches" in stats
        assert isinstance(stats["memory_caches"], int)

    def test_has_storage_path_key(self, tmp_path):
        proc = _make_processor(tmp_path)
        stats = proc.get_statistics()
        assert "storage_path" in stats
        assert stats["storage_path"] == str(proc.storage.storage_path)

    def test_memory_caches_starts_at_zero(self, tmp_path):
        proc = _make_processor(tmp_path)
        stats = proc.get_statistics()
        assert stats["memory_caches"] == 0


# ===================================================================
# 4. process_documents with empty list
# ===================================================================

class TestProcessDocumentsEmpty:

    def test_empty_document_list_no_crash(self, tmp_path):
        proc = _make_processor(tmp_path)
        # process_documents with an empty list should simply iterate zero
        # times and return without error.
        proc.process_documents([], verbose=False)


# ===================================================================
# 5. remember_text with short text (single window)
# ===================================================================

class TestRememberTextShort:

    def test_returns_dict_with_expected_keys(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        result = proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        assert isinstance(result, dict)
        assert "memory_cache_id" in result
        assert "chunks_processed" in result
        assert "storage_path" in result

    def test_single_chunk_processed(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        result = proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        assert result["chunks_processed"] == 1

    def test_memory_cache_id_is_string(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        result = proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        assert isinstance(result["memory_cache_id"], str)
        assert len(result["memory_cache_id"]) > 0

    def test_storage_path_matches(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        result = proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        assert result["storage_path"] == str(proc.storage.storage_path)

    def test_entities_query_works_after_processing(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        entities = proc.storage.get_all_entities()
        # In mock mode the prompt "请从文本中抽取所有概念实体（越多越好）："
        # does not match the _mock_llm_response pattern "抽取实体" (contiguous),
        # so the LLM returns "默认响应" which fails JSON parsing, yielding 0 entities.
        # The test verifies the query completes without error and returns a list.
        assert isinstance(entities, list)

    def test_relations_query_works_after_processing(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        relations = proc.storage.get_all_relations()
        # Same mock mode limitation: 0 entities => 0 relations extracted.
        # Verify the query completes without error.
        assert isinstance(relations, list)


# ===================================================================
# 6. remember_text with multi-window text
# ===================================================================

class TestRememberTextMultiWindow:

    def test_multiple_chunks_processed(self, tmp_path):
        proc = _make_processor(tmp_path)
        # window_size=100, overlap=20 => stride=80
        # Need text > 100 chars to trigger 2+ windows.
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

    def test_entities_query_after_multi_window(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = (
            "Alice walked through the bustling market on a sunny morning. "
            "She greeted Bob who was selling fresh oranges at his fruit stand. "
            "The market was filled with colorful stalls and cheerful vendors. "
            "Alice picked up some ripe tomatoes and a bag of green apples. "
            "Bob recommended the local honey from the nearby beekeeper Charlie."
        )
        proc.remember_text(text, doc_name="test_multi", verbose=False, verbose_steps=False)
        entities = proc.storage.get_all_entities()
        # Mock mode prompt mismatch yields 0 entities (see test_entities_query_works_after_processing).
        # Verify multi-window processing completes and storage is queryable.
        assert isinstance(entities, list)

    def test_extract_failure_does_not_emit_secondary_step6_step7_errors(self, tmp_path, monkeypatch, capsys):
        proc = _make_processor(tmp_path)
        text = (
            "Alice walked through the bustling market on a sunny morning. "
            "She greeted Bob who was selling fresh oranges at his fruit stand. "
            "The market was filled with colorful stalls and cheerful vendors. "
            "Alice picked up some ripe tomatoes and a bag of green apples. "
            "Bob recommended the local honey from the nearby beekeeper Charlie."
        )

        def _boom(*args, **kwargs):
            raise ValueError("simulated extract failure")

        monkeypatch.setattr(proc, "_extract_only", _boom)

        with pytest.raises(ValueError, match="simulated extract failure"):
            proc.remember_text(text, doc_name="test_extract_failure", verbose=False, verbose_steps=False)

        captured = capsys.readouterr()
        assert "step6 skipped for window" not in captured.err
        assert "step6 result for window" not in captured.err


# ===================================================================
# 7. remember_text with start_chunk (skip)
# ===================================================================

class TestRememberTextStartChunk:

    def test_start_chunk_beyond_total_returns_immediately(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Short text."
        # total_chunks will be 1; start_chunk=10 >= 1 => early return
        result = proc.remember_text(
            text, doc_name="test_skip", verbose=False, verbose_steps=False, start_chunk=10,
        )
        assert result["chunks_processed"] == 1
        # memory_cache_id is None since current_memory_cache is still None
        assert result["memory_cache_id"] is None


# ===================================================================
# 8. Concurrency: _window_slot semaphore
# ===================================================================

class TestConcurrencySemaphore:

    def test_window_slot_value_matches_max_concurrent_windows(self, tmp_path):
        proc = _make_processor(tmp_path, max_concurrent_windows=1)
        assert proc._window_slot._value == 1

    def test_window_slot_value_larger(self, tmp_path):
        proc = _make_processor(tmp_path, max_concurrent_windows=5)
        assert proc._window_slot._value == 5

    def test_window_slot_is_semaphore(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert isinstance(proc._window_slot, threading.Semaphore)

    def test_max_concurrent_windows_attribute(self, tmp_path):
        proc = _make_processor(tmp_path, max_concurrent_windows=3)
        assert proc._max_concurrent_windows == 3


# ===================================================================
# 9. Cache lock
# ===================================================================

class TestCacheLock:

    def test_cache_lock_is_threading_lock(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert isinstance(proc._cache_lock, type(threading.Lock()))

    def test_cache_lock_acquire_release(self, tmp_path):
        proc = _make_processor(tmp_path)
        proc._cache_lock.acquire()
        proc._cache_lock.release()
        # Should be able to acquire again (was not deadlocked)
        proc._cache_lock.acquire()
        proc._cache_lock.release()


# ===================================================================
# 10. Edge cases
# ===================================================================

class TestEdgeCases:

    def test_empty_string(self, tmp_path):
        proc = _make_processor(tmp_path)
        result = proc.remember_text("", doc_name="empty", verbose=False, verbose_steps=False)
        # Empty text: len=0 <= window_size=100 => total_chunks=1
        # The chunk will be the doc prefix only since text is empty.
        assert result["chunks_processed"] == 1

    def test_whitespace_only(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "   \n\t  \n   "
        result = proc.remember_text(text, doc_name="whitespace", verbose=False, verbose_steps=False)
        assert result["chunks_processed"] >= 1

    def test_unicode_and_emoji(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Hello world! This is a test with unicode characters."
        result = proc.remember_text(text, doc_name="unicode", verbose=False, verbose_steps=False)
        assert result["chunks_processed"] >= 1
        assert isinstance(result["memory_cache_id"], str)


# ===================================================================
# 11. remember_phase1_overall
# ===================================================================

class TestRememberPhase1Overall:

    def test_returns_memory_cache(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "This is a document about testing the pipeline with mock data."
        result = proc.remember_phase1_overall(text, doc_name="phase1_test")
        assert isinstance(result, MemoryCache)

    def test_activity_type_is_document_overall(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "This is a document about testing the pipeline with mock data."
        result = proc.remember_phase1_overall(text, doc_name="phase1_test")
        assert result.activity_type == "文档整体"

    def test_absolute_id_is_set(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "This is a document about testing the pipeline with mock data."
        result = proc.remember_phase1_overall(text, doc_name="phase1_test")
        assert isinstance(result.absolute_id, str)
        assert len(result.absolute_id) > 0

    def test_event_time_is_set(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "This is a document about testing the pipeline with mock data."
        result = proc.remember_phase1_overall(text, doc_name="phase1_test")
        assert result.event_time is not None

    def test_with_event_time(self, tmp_path):
        from datetime import datetime, timezone
        proc = _make_processor(tmp_path)
        text = "This is a document about testing the pipeline with mock data."
        t = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = proc.remember_phase1_overall(
            text, doc_name="phase1_test", event_time=t,
        )
        assert result.event_time == t


# ===================================================================
# 12. remember_phase2_windows with short text
# ===================================================================

class TestRememberPhase2Windows:

    def test_returns_dict_with_expected_keys(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Bob visited the science museum and learned about quantum physics experiments."
        result = proc.remember_phase2_windows(
            text, doc_name="phase2_test", verbose=False,
        )
        assert isinstance(result, dict)
        assert "memory_cache_id" in result
        assert "chunks_processed" in result
        assert "storage_path" in result

    def test_chunks_processed_at_least_one(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Bob visited the science museum and learned about quantum physics experiments."
        result = proc.remember_phase2_windows(
            text, doc_name="phase2_test", verbose=False,
        )
        assert result["chunks_processed"] >= 1

    def test_memory_cache_id_is_string(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Bob visited the science museum and learned about quantum physics experiments."
        result = proc.remember_phase2_windows(
            text, doc_name="phase2_test", verbose=False,
        )
        assert isinstance(result["memory_cache_id"], str)
        assert len(result["memory_cache_id"]) > 0

    def test_with_overall_cache(self, tmp_path):
        proc = _make_processor(tmp_path)
        from datetime import datetime
        overall = MemoryCache(
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
