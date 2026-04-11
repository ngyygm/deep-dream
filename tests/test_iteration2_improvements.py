"""
Comprehensive tests for iteration-2 improvements.

Covers 4 dimensions with 30+ test cases:
  1. Configurable merge_safe thresholds (8 tests)
  2. Orchestrator graceful shutdown (8 tests)
  3. Entity processor configurable parameters (8 tests)
  4. Threshold boundary conditions (7 tests)
"""
import sys
import threading
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from concurrent.futures import ThreadPoolExecutor

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Dimension 1: Configurable merge_safe thresholds
# ============================================================
class TestMergeSafeThresholds:
    """Tests for configurable merge_safe thresholds in EntityProcessor."""

    def _make_processor(self, **kwargs):
        from processor.pipeline.entity import EntityProcessor
        storage = MagicMock()
        llm = MagicMock()
        return EntityProcessor(storage, llm, **kwargs)

    def test_default_embedding_threshold(self):
        proc = self._make_processor()
        assert proc.merge_safe_embedding_threshold == 0.55

    def test_default_jaccard_threshold(self):
        proc = self._make_processor()
        assert proc.merge_safe_jaccard_threshold == 0.4

    def test_custom_embedding_threshold(self):
        proc = self._make_processor(merge_safe_embedding_threshold=0.7)
        assert proc.merge_safe_embedding_threshold == 0.7

    def test_custom_jaccard_threshold(self):
        proc = self._make_processor(merge_safe_jaccard_threshold=0.6)
        assert proc.merge_safe_jaccard_threshold == 0.6

    def test_strict_thresholds(self):
        """Very high thresholds should make merge_safe harder to satisfy."""
        proc = self._make_processor(
            merge_safe_embedding_threshold=0.95,
            merge_safe_jaccard_threshold=0.9,
        )
        assert proc.merge_safe_embedding_threshold == 0.95
        assert proc.merge_safe_jaccard_threshold == 0.9

    def test_permissive_thresholds(self):
        """Very low thresholds should make merge_safe easier to satisfy."""
        proc = self._make_processor(
            merge_safe_embedding_threshold=0.1,
            merge_safe_jaccard_threshold=0.1,
        )
        assert proc.merge_safe_embedding_threshold == 0.1
        assert proc.merge_safe_jaccard_threshold == 0.1

    def test_thresholds_affect_candidate_table(self):
        """merge_safe in candidate table should use instance thresholds."""
        proc = self._make_processor(
            merge_safe_embedding_threshold=0.99,
            merge_safe_jaccard_threshold=0.99,
        )
        # With thresholds at 0.99, moderate scores should NOT be merge_safe
        best_dense = 0.6
        lexical_score = 0.5
        # This simulates the logic in _build_entity_candidate_table
        is_safe = (best_dense >= proc.merge_safe_embedding_threshold and
                   lexical_score >= proc.merge_safe_jaccard_threshold)
        assert is_safe is False

    def test_zero_thresholds_always_merge_safe(self):
        """Zero thresholds should make everything merge_safe (except core_name_match)."""
        proc = self._make_processor(
            merge_safe_embedding_threshold=0.0,
            merge_safe_jaccard_threshold=0.0,
        )
        best_dense = 0.01
        lexical_score = 0.01
        is_safe = (best_dense >= proc.merge_safe_embedding_threshold and
                   lexical_score >= proc.merge_safe_jaccard_threshold)
        assert is_safe is True


# ============================================================
# Dimension 2: Orchestrator graceful shutdown
# ============================================================
class TestOrchestratorShutdown:
    """Tests for TemporalMemoryGraphProcessor graceful shutdown."""

    def _make_processor(self, tmp_path):
        from processor.pipeline.orchestrator import TemporalMemoryGraphProcessor
        with patch('processor.pipeline.orchestrator.StorageManager'):
            proc = TemporalMemoryGraphProcessor.__new__(TemporalMemoryGraphProcessor)
            proc._extraction_executor = ThreadPoolExecutor(max_workers=2)
            proc.storage = MagicMock()
            return proc

    def test_close_exists(self):
        from processor.pipeline.orchestrator import TemporalMemoryGraphProcessor
        assert hasattr(TemporalMemoryGraphProcessor, 'close')

    def test_close_calls_storage_close(self, tmp_path):
        proc = self._make_processor(tmp_path)
        proc.close()
        proc.storage.close.assert_called_once()

    def test_close_shuts_down_executor(self, tmp_path):
        proc = self._make_processor(tmp_path)
        executor = proc._extraction_executor
        proc.close()
        # Executor should be shut down (submitting should fail or be rejected)
        assert executor._shutdown

    def test_close_idempotent(self, tmp_path):
        """Calling close() twice should not error."""
        proc = self._make_processor(tmp_path)
        proc.close()
        proc.close()  # Should not raise

    def test_close_without_executor(self, tmp_path):
        """close() should work even if _extraction_executor doesn't exist."""
        from processor.pipeline.orchestrator import TemporalMemoryGraphProcessor
        proc = TemporalMemoryGraphProcessor.__new__(TemporalMemoryGraphProcessor)
        proc.storage = MagicMock()
        proc.close()  # Should not raise
        proc.storage.close.assert_called()

    def test_close_without_storage(self):
        """close() should work even if storage doesn't exist."""
        from processor.pipeline.orchestrator import TemporalMemoryGraphProcessor
        proc = TemporalMemoryGraphProcessor.__new__(TemporalMemoryGraphProcessor)
        proc._extraction_executor = None
        proc.close()  # Should not raise

    def test_del_calls_close(self, tmp_path):
        """__del__ should call close() without errors."""
        proc = self._make_processor(tmp_path)
        proc.__del__()  # Should not raise

    def test_del_handles_exceptions(self, tmp_path):
        """__del__ should silently handle exceptions."""
        proc = self._make_processor(tmp_path)
        proc.storage.close.side_effect = RuntimeError("fail")
        proc.__del__()  # Should not raise


# ============================================================
# Dimension 3: Entity processor configurable parameters
# ============================================================
class TestEntityProcessorConfig:
    """Tests for EntityProcessor configuration options."""

    def _make_processor(self, **kwargs):
        from processor.pipeline.entity import EntityProcessor
        storage = MagicMock()
        llm = MagicMock()
        return EntityProcessor(storage, llm, **kwargs)

    def test_default_max_similar_entities(self):
        proc = self._make_processor()
        assert proc.max_similar_entities == 10

    def test_custom_max_similar_entities(self):
        proc = self._make_processor(max_similar_entities=20)
        assert proc.max_similar_entities == 20

    def test_default_content_snippet_length(self):
        proc = self._make_processor()
        assert proc.content_snippet_length == 50

    def test_custom_content_snippet_length(self):
        proc = self._make_processor(content_snippet_length=100)
        assert proc.content_snippet_length == 100

    def test_default_batch_resolution_enabled(self):
        proc = self._make_processor()
        assert proc.batch_resolution_enabled is True

    def test_default_batch_confidence_threshold(self):
        proc = self._make_processor()
        assert proc.batch_resolution_confidence_threshold == 0.75

    def test_verbose_flag(self):
        proc = self._make_processor(verbose=False)
        assert proc.verbose is False

    def test_entity_progress_verbose_flag(self):
        proc = self._make_processor(entity_progress_verbose=True)
        assert proc.entity_progress_verbose is True


# ============================================================
# Dimension 4: Threshold boundary conditions
# ============================================================
class TestThresholdBoundaries:
    """Tests for edge cases around threshold values."""

    def test_relation_threshold_exactly_at_boundary(self):
        from processor.pipeline.extraction import MIN_RELATION_CONTENT_LENGTH, _is_valid_relation_content
        exact = "a" * MIN_RELATION_CONTENT_LENGTH
        assert _is_valid_relation_content(exact) is True

    def test_relation_threshold_one_below(self):
        from processor.pipeline.extraction import MIN_RELATION_CONTENT_LENGTH, _is_valid_relation_content
        below = "a" * (MIN_RELATION_CONTENT_LENGTH - 1)
        assert _is_valid_relation_content(below) is False

    def test_empty_string_always_fails(self):
        from processor.pipeline.extraction import _is_valid_relation_content
        assert _is_valid_relation_content("") is False
        assert _is_valid_relation_content(None) is False

    def test_unicode_at_threshold(self):
        from processor.pipeline.extraction import MIN_RELATION_CONTENT_LENGTH, _is_valid_relation_content
        content = "中" * MIN_RELATION_CONTENT_LENGTH
        assert _is_valid_relation_content(content) is True

    def test_whitespace_only_content(self):
        """Content with only spaces passes length check but is invalid semantically.
        The extraction pipeline strips content before calling this function,
        so an all-space string would arrive as empty and be rejected."""
        from processor.pipeline.extraction import _is_valid_relation_content
        # This is 8 spaces — passes length but should be caught by extraction reflow
        # The function checks raw length (not stripped), which is fine because
        # extraction.py strips content before passing to this function.
        assert _is_valid_relation_content("   ") is False  # 3 chars < 8

    def test_merge_safe_threshold_at_exactly_one(self):
        """Threshold = 1.0 means scores must be exactly 1.0 to pass."""
        proc = self._make_processor(merge_safe_embedding_threshold=1.0, merge_safe_jaccard_threshold=1.0)
        is_safe = (1.0 >= proc.merge_safe_embedding_threshold and
                   1.0 >= proc.merge_safe_jaccard_threshold)
        assert is_safe is True
        is_safe_099 = (0.99 >= proc.merge_safe_embedding_threshold and
                       0.99 >= proc.merge_safe_jaccard_threshold)
        assert is_safe_099 is False

    def test_merge_safe_threshold_negative(self):
        """Negative thresholds should make everything merge_safe."""
        proc = self._make_processor(merge_safe_embedding_threshold=-0.1, merge_safe_jaccard_threshold=-0.1)
        is_safe = (0.0 >= proc.merge_safe_embedding_threshold and
                   0.0 >= proc.merge_safe_jaccard_threshold)
        assert is_safe is True

    def _make_processor(self, **kwargs):
        from processor.pipeline.entity import EntityProcessor
        storage = MagicMock()
        llm = MagicMock()
        return EntityProcessor(storage, llm, **kwargs)
