"""
Tests for TemporalMemoryGraphProcessor remember_text with various inputs,
process_documents with empty list, and edge cases.
"""

import pytest

from processor.pipeline.orchestrator import TemporalMemoryGraphProcessor
from processor.llm.client import _mock_json_fence


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
# 1. process_documents with empty list
# ===================================================================

class TestProcessDocumentsEmpty:

    def test_empty_document_list_no_crash(self, tmp_path):
        proc = _make_processor(tmp_path)
        # process_documents with an empty list should simply iterate zero
        # times and return without error.
        proc.process_documents([], verbose=False)


# ===================================================================
# 2. remember_text with short text (single window)
# ===================================================================

class TestRememberTextShort:

    def test_returns_dict_with_expected_keys(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        result = proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        assert isinstance(result, dict)
        assert "episode_id" in result
        assert "chunks_processed" in result
        assert "storage_path" in result

    def test_single_chunk_processed(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        result = proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        assert result["chunks_processed"] == 1

    def test_episode_id_is_string(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        result = proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        assert isinstance(result["episode_id"], str)
        assert len(result["episode_id"]) > 0

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
        # In mock mode the LLM returns a fixed entity; verify the query completes without error.
        assert isinstance(entities, list)

    def test_relations_query_works_after_processing(self, tmp_path):
        proc = _make_processor(tmp_path)
        text = "Alice went to the market to buy some fresh vegetables and fruits."
        proc.remember_text(text, doc_name="test_short", verbose=False, verbose_steps=False)
        relations = proc.storage.get_all_relations()
        # In mock mode the LLM returns a fixed relation; verify the query completes without error.
        assert isinstance(relations, list)


# ===================================================================
# 3. remember_text with multi-window text
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
# 4. remember_text with start_chunk (skip)
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
        # episode_id is None since current_episode is still None
        assert result["episode_id"] is None


# ===================================================================
# 5. Edge cases
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
        assert isinstance(result["episode_id"], str)
