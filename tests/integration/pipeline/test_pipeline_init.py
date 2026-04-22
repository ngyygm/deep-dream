"""
Tests for TemporalMemoryGraphProcessor initialization, runtime stats,
statistics query, concurrency semaphore, and cache lock.
"""

import threading

import pytest

from processor.pipeline.orchestrator import TemporalMemoryGraphProcessor


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_processor(tmp_path, **overrides):
    """Create a TemporalMemoryGraphProcessor with small windows.

    In mock mode (default) no API key is provided.  When called from a
    @pytest.mark.real_llm test that injects real_llm_config / shared_embedding_client
    via fixtures, pass them in overrides.
    """
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
        assert proc.content_snippet_length == 300

    def test_default_relation_content_snippet_length(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert proc.relation_content_snippet_length == 200

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
        from concurrent.futures import ThreadPoolExecutor
        assert isinstance(proc._extraction_executor, ThreadPoolExecutor)

    def test_thread_pool_max_workers_default(self, tmp_path):
        proc = _make_processor(tmp_path)
        # max_llm_concurrency=2, max_concurrent_windows=None => auto-derived = 2
        assert proc._extraction_executor._max_workers == 2

    def test_thread_pool_max_workers_custom(self, tmp_path):
        proc = _make_processor(tmp_path, max_concurrent_windows=4)
        assert proc._extraction_executor._max_workers == 4

    def test_window_slot_semaphore_default(self, tmp_path):
        proc = _make_processor(tmp_path)
        # max_llm_concurrency=2, max_concurrent_windows=None => auto-derived = 2
        assert proc._window_slot._value == 2

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

    def test_current_episode_initially_none(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert proc.current_episode is None

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


    def test_default_remember_mode_multi_step(self, tmp_path):
        proc = _make_processor(tmp_path)
        assert proc.remember_mode == "multi_step"
        assert proc.remember_anchor_recall_rounds == 1
        assert proc.remember_named_entity_recall_rounds == 1
        assert proc.remember_entity_content_batch_size == 6
        assert proc.remember_relation_hint_rounds == 1
        assert proc.remember_relation_expand_rounds == 1
        assert proc.remember_pre_alignment_validation_retries == 2

    def test_remember_config_override_enables_legacy_mode(self, tmp_path):
        proc = _make_processor(
            tmp_path,
            remember_config={"mode": "legacy", "alignment_policy": "default"},
        )
        assert proc.remember_mode == "legacy"
        assert proc.remember_alignment_conservative is False
        assert proc.relation_processor.preserve_distinct_relations_per_pair is False

    def test_conservative_remember_config_sets_relation_preservation(self, tmp_path):
        proc = _make_processor(
            tmp_path,
            remember_config={"mode": "multi_step", "alignment_policy": "conservative"},
        )
        assert proc.remember_alignment_conservative is True
        assert proc.relation_processor.preserve_distinct_relations_per_pair is True


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

    def test_has_episodes_key(self, tmp_path):
        proc = _make_processor(tmp_path)
        stats = proc.get_statistics()
        assert "episodes" in stats
        assert isinstance(stats["episodes"], int)

    def test_has_storage_path_key(self, tmp_path):
        proc = _make_processor(tmp_path)
        stats = proc.get_statistics()
        assert "storage_path" in stats
        assert stats["storage_path"] == str(proc.storage.storage_path)

    def test_episodes_starts_at_zero(self, tmp_path):
        proc = _make_processor(tmp_path)
        stats = proc.get_statistics()
        assert stats["episodes"] == 0


# ===================================================================
# 4. Concurrency: _window_slot semaphore
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
# 5. Cache lock
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
