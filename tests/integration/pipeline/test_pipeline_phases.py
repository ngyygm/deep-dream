"""
Tests for TemporalMemoryGraphProcessor phase1 (overall) and phase2 (windows)
remember methods.
"""

from datetime import datetime, timezone

import pytest

from processor.pipeline.orchestrator import TemporalMemoryGraphProcessor
from processor.models import Episode


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
# 1. remember_phase1_overall
# ===================================================================

class TestRememberPhase1Overall:

    def test_returns_memory_cache(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "This is a document about testing the pipeline with mock data."
        result = proc.remember_phase1_overall(text, doc_name="phase1_test")
        assert isinstance(result, Episode)

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
        proc = _make_processor(tmp_path)
        text = "This is a document about testing the pipeline with mock data."
        t = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = proc.remember_phase1_overall(
            text, doc_name="phase1_test", event_time=t,
        )
        assert result.event_time == t


# ===================================================================
# 2. remember_phase2_windows with short text
# ===================================================================

class TestRememberPhase2Windows:

    def test_returns_dict_with_expected_keys(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Bob visited the science museum and learned about quantum physics experiments."
        result = proc.remember_phase2_windows(
            text, doc_name="phase2_test", verbose=False,
        )
        assert isinstance(result, dict)
        assert "episode_id" in result
        assert "chunks_processed" in result
        assert "storage_path" in result

    def test_chunks_processed_at_least_one(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Bob visited the science museum and learned about quantum physics experiments."
        result = proc.remember_phase2_windows(
            text, doc_name="phase2_test", verbose=False,
        )
        assert result["chunks_processed"] >= 1

    def test_episode_id_is_string(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Bob visited the science museum and learned about quantum physics experiments."
        result = proc.remember_phase2_windows(
            text, doc_name="phase2_test", verbose=False,
        )
        assert isinstance(result["episode_id"], str)
        assert len(result["episode_id"]) > 0

    def test_with_overall_cache(self, tmp_path):
        proc = _make_processor(tmp_path)
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
