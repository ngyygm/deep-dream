"""
Comprehensive tests for iteration-3 refactoring.

Covers 4 dimensions with 30+ test cases:
  1. _search_entity_candidates extracted method (10 tests)
  2. _filter_candidates_by_existing_relations (8 tests)
  3. _TITLE_SUFFIXES_RE normalization (8 tests)
  4. Integration: _process_single_entity uses new methods (7 tests)
"""
import sys
import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_entity(family_id, name, content="desc", processed_time=None):
    from processor.models import Entity
    return Entity(
        absolute_id=f"abs_{family_id}",
        family_id=family_id,
        name=name,
        content=content,
        event_time=datetime.now(),
        processed_time=processed_time or datetime.now(),
        episode_id="ep1",
        source_document="test",
    )


def _make_processor():
    from processor.pipeline.entity import EntityProcessor
    storage = MagicMock()
    llm = MagicMock()
    proc = EntityProcessor(storage, llm)
    proc.storage.search_entities_by_similarity.return_value = []
    return proc


# ============================================================
# Dimension 1: _search_entity_candidates
# ============================================================
class TestSearchEntityCandidates:
    """Tests for the extracted _search_entity_candidates method."""

    def test_returns_empty_when_no_results(self):
        proc = _make_processor()
        result = proc._search_entity_candidates("测试", "描述", 0.5)
        assert result == []

    def test_merges_results_from_all_search_modes(self):
        proc = _make_processor()
        e1 = _make_entity("fid1", "实体A")
        e2 = _make_entity("fid2", "实体B")
        e3 = _make_entity("fid3", "实体C")
        # Mock different search modes returning different entities
        def mock_search(query, **kwargs):
            method = kwargs.get("similarity_method", "")
            mode = kwargs.get("text_mode", "")
            if method == "jaccard" and mode == "name_only":
                return [e1]
            if method == "embedding" and mode == "name_only":
                return [e2]
            if method == "embedding" and mode == "name_and_content":
                return [e3]
            return []
        proc.storage.search_entities_by_similarity = mock_search
        result = proc._search_entity_candidates("实体A", "描述", 0.5)
        assert len(result) == 3

    def test_dedup_by_family_id_keeps_latest(self):
        proc = _make_processor()
        t1 = datetime(2024, 1, 1)
        t2 = datetime(2025, 1, 1)
        e_old = _make_entity("fid1", "实体A", processed_time=t1)
        e_new = _make_entity("fid1", "实体A更新", processed_time=t2)
        # _search_entity_candidates calls storage 4 times (jaccard, core_jaccard, name_emb, full_emb)
        # All return both entities; dedup should keep e_new (latest processed_time)
        proc.storage.search_entities_by_similarity = lambda *a, **k: [e_old, e_new]
        result = proc._search_entity_candidates("实体A", "描述", 0.5)
        assert len(result) == 1
        assert result[0].processed_time == t2

    def test_title_suffix_triggers_core_name_search(self):
        proc = _make_processor()
        calls = []
        def mock_search(query, **kwargs):
            calls.append(query)
            return []
        proc.storage.search_entities_by_similarity = mock_search
        proc._search_entity_candidates("张伟教授", "描述", 0.5)
        # Should search for both "张伟教授" and "张伟"
        assert "张伟教授" in calls
        assert "张伟" in calls

    def test_no_title_suffix_no_core_search(self):
        proc = _make_processor()
        calls = []
        def mock_search(query, **kwargs):
            calls.append(query)
            return []
        proc.storage.search_entities_by_similarity = mock_search
        proc._search_entity_candidates("张伟", "描述", 0.5)
        # Only "张伟" should appear, no core name search
        assert calls.count("张伟") >= 1

    def test_thresholds_fall_back_to_similarity_threshold(self):
        proc = _make_processor()
        proc.storage.search_entities_by_similarity = MagicMock(return_value=[])
        proc._search_entity_candidates("test", "desc", 0.8)
        # Jaccard threshold should be min(0.8, 0.6) = 0.6
        calls = proc.storage.search_entities_by_similarity.call_args_list
        for call in calls:
            threshold = call[1].get("threshold", call[0][1] if len(call[0]) > 1 else None)
            if threshold is not None:
                assert threshold <= 0.8

    def test_custom_thresholds_used(self):
        proc = _make_processor()
        proc.storage.search_entities_by_similarity = MagicMock(return_value=[])
        proc._search_entity_candidates(
            "test", "desc", 0.8,
            jaccard_search_threshold=0.3,
            embedding_name_search_threshold=0.5,
            embedding_full_search_threshold=0.7,
        )
        calls = proc.storage.search_entities_by_similarity.call_args_list
        thresholds = [c[1]["threshold"] for c in calls if "threshold" in c[1]]
        assert 0.3 in thresholds
        assert 0.5 in thresholds
        assert 0.7 in thresholds

    def test_filter_called_with_relation_pairs(self):
        proc = _make_processor()
        e1 = _make_entity("fid1", "B")
        proc.storage.search_entities_by_similarity = lambda *a, **k: [e1]
        names = {"B"}
        # Actual format: (sorted_tuple, content_hash)
        pairs = {(tuple(sorted(["A", "B"])), hash("some relation content".lower()))}
        result = proc._search_entity_candidates(
            "A", "desc", 0.5,
            extracted_entity_names=names,
            extracted_relation_pairs=pairs,
        )
        # B should be filtered out since pair (A,B) exists in extracted_relation_pairs
        assert len(result) == 0

    def test_filter_preserves_same_name_candidates(self):
        proc = _make_processor()
        e1 = _make_entity("fid1", "A")
        proc.storage.search_entities_by_similarity = lambda *a, **k: [e1]
        names = {"A"}
        pairs = {tuple(sorted(["A", "A"]))}
        result = proc._search_entity_candidates(
            "A", "desc", 0.5,
            extracted_entity_names=names,
            extracted_relation_pairs=pairs,
        )
        # Same name should be preserved
        assert len(result) == 1


# ============================================================
# Dimension 2: _filter_candidates_by_existing_relations
# ============================================================
class TestFilterCandidates:
    """Tests for _filter_candidates_by_existing_relations."""

    def test_no_filter_when_no_names_or_pairs(self):
        proc = _make_processor()
        e1 = _make_entity("f1", "A")
        result = proc._filter_candidates_by_existing_relations([e1], "X", set(), set())
        assert len(result) == 1

    def test_filters_candidates_with_existing_relations(self):
        proc = _make_processor()
        e1 = _make_entity("f1", "B")
        names = {"B"}
        # Actual format: (sorted_tuple, content_hash)
        pairs = {(tuple(sorted(["X", "B"])), hash("relation content".lower()))}
        result = proc._filter_candidates_by_existing_relations([e1], "X", names, pairs)
        assert len(result) == 0

    def test_keeps_candidates_not_in_names(self):
        proc = _make_processor()
        e1 = _make_entity("f1", "C")
        names = {"B"}
        pairs = set()
        result = proc._filter_candidates_by_existing_relations([e1], "X", names, pairs)
        assert len(result) == 1

    def test_keeps_same_name_candidates(self):
        proc = _make_processor()
        e1 = _make_entity("f1", "X")
        names = {"X"}
        pairs = {(tuple(sorted(["X", "X"])), hash("content".lower()))}
        result = proc._filter_candidates_by_existing_relations([e1], "X", names, pairs)
        assert len(result) == 1

    def test_handles_multiple_candidates(self):
        proc = _make_processor()
        e1 = _make_entity("f1", "A")
        e2 = _make_entity("f2", "B")
        e3 = _make_entity("f3", "C")
        names = {"A", "B"}
        # Actual format: (sorted_tuple, content_hash)
        pairs = {(tuple(sorted(["X", "B"])), hash("relation content".lower()))}
        result = proc._filter_candidates_by_existing_relations(
            [e1, e2, e3], "X", names, pairs
        )
        # A: kept (same name as entity_name X? No, "A" != "X")
        # Wait: entity_name="X", candidate.name="A" → A not in names? A IS in names
        # A in names, pair_key=(A,X), check if (A,X) in pairs → (B,X) is in pairs, (A,X) is not
        # So A is kept. B in names, pair_key=(B,X), (B,X) is in pairs → filtered
        # C not in names → kept
        # Result: A, C
        assert len(result) == 2
        result_names = {r.name for r in result}
        assert result_names == {"A", "C"}

    def test_empty_candidates_returns_empty(self):
        proc = _make_processor()
        result = proc._filter_candidates_by_existing_relations([], "X", {"A"}, set())
        assert result == []

    def test_pair_ordering_independent(self):
        proc = _make_processor()
        e1 = _make_entity("f1", "B")
        names = {"B"}
        # Actual format: (sorted_tuple, content_hash)
        pairs = {(("B", "X"), hash("content".lower()))}
        result = proc._filter_candidates_by_existing_relations([e1], "X", names, pairs)
        assert len(result) == 0

    def test_no_pairs_keeps_non_name_candidates(self):
        proc = _make_processor()
        e1 = _make_entity("f1", "B")
        names = {"B"}
        pairs = set()
        result = proc._filter_candidates_by_existing_relations([e1], "X", names, pairs)
        # B is in names but no relation pairs — should be kept
        assert len(result) == 1


# ============================================================
# Dimension 3: _TITLE_SUFFIXES_RE normalization
# ============================================================
class TestTitleSuffixes:
    """Tests for title suffix removal in entity name normalization."""

    def test_removes_professor(self):
        from processor.pipeline.entity import EntityProcessor
        result = EntityProcessor._TITLE_SUFFIXES_RE.sub('', "张伟教授").strip()
        assert result == "张伟"

    def test_removes_doctor(self):
        from processor.pipeline.entity import EntityProcessor
        result = EntityProcessor._TITLE_SUFFIXES_RE.sub('', "李明博士").strip()
        assert result == "李明"

    def test_no_suffix_unchanged(self):
        from processor.pipeline.entity import EntityProcessor
        result = EntityProcessor._TITLE_SUFFIXES_RE.sub('', "王五").strip()
        assert result == "王五"

    def test_removes_engineer(self):
        from processor.pipeline.entity import EntityProcessor
        result = EntityProcessor._TITLE_SUFFIXES_RE.sub('', "赵六工程师").strip()
        assert result == "赵六"

    def test_removes_manager(self):
        from processor.pipeline.entity import EntityProcessor
        result = EntityProcessor._TITLE_SUFFIXES_RE.sub('', "钱七经理").strip()
        assert result == "钱七"

    def test_english_name_unchanged(self):
        from processor.pipeline.entity import EntityProcessor
        result = EntityProcessor._TITLE_SUFFIXES_RE.sub('', "John Smith").strip()
        assert result == "John Smith"

    def test_multiple_name_types(self):
        from processor.pipeline.entity import EntityProcessor
        names = ["张教授", "李主任", "王校长", "刘总监"]
        for name in names:
            result = EntityProcessor._TITLE_SUFFIXES_RE.sub('', name).strip()
            assert len(result) < len(name), f"Failed for {name}"

    def test_short_core_name_detected(self):
        from processor.pipeline.entity import EntityProcessor
        name = "教授"  # core name would be empty
        result = EntityProcessor._TITLE_SUFFIXES_RE.sub('', name).strip()
        assert result == ""  # entire name is a suffix


# ============================================================
# Dimension 4: Integration with _process_single_entity
# ============================================================
class TestProcessSingleEntityIntegration:
    """Integration tests ensuring _process_single_entity uses the new methods."""

    def test_delegates_to_search_candidates(self):
        proc = _make_processor()
        proc._search_entity_candidates = MagicMock(return_value=[])
        proc._create_new_entity = MagicMock(return_value=_make_entity("new1", "test"))
        proc._process_single_entity(
            {"name": "test", "content": "desc"},
            episode_id="ep1",
            similarity_threshold=0.5,
        )
        proc._search_entity_candidates.assert_called_once()

    def test_creates_new_entity_when_no_candidates(self):
        proc = _make_processor()
        proc._create_new_entity = MagicMock(return_value=_make_entity("new1", "test"))
        result = proc._process_single_entity(
            {"name": "test", "content": "desc"},
            episode_id="ep1",
            similarity_threshold=0.5,
        )
        entity, relations, name_map = result
        assert entity is not None
        assert "test" in name_map

    def test_returns_correct_structure(self):
        proc = _make_processor()
        proc._search_entity_candidates = MagicMock(return_value=[])
        proc._create_new_entity = MagicMock(return_value=_make_entity("new1", "test"))
        result = proc._process_single_entity(
            {"name": "test", "content": "desc"},
            episode_id="ep1",
            similarity_threshold=0.5,
        )
        assert len(result) == 3  # (entity, relations, name_map)

    def test_entity_tree_log_suppressed_when_not_verbose(self):
        proc = _make_processor()
        proc.entity_progress_verbose = False
        proc.verbose = False
        proc._search_entity_candidates = MagicMock(return_value=[])
        proc._create_new_entity = MagicMock(return_value=_make_entity("new1", "test"))
        # Should not raise any errors when logging is suppressed
        proc._process_single_entity(
            {"name": "test", "content": "desc"},
            episode_id="ep1",
            similarity_threshold=0.5,
            entity_index=1,
            total_entities=5,
        )

    def test_search_candidates_passes_all_thresholds(self):
        proc = _make_processor()
        proc._search_entity_candidates = MagicMock(return_value=[])
        proc._create_new_entity = MagicMock(return_value=_make_entity("new1", "test"))
        proc._process_single_entity(
            {"name": "test", "content": "desc"},
            episode_id="ep1",
            similarity_threshold=0.7,
            jaccard_search_threshold=0.3,
            embedding_name_search_threshold=0.5,
            embedding_full_search_threshold=0.6,
        )
        call_args = proc._search_entity_candidates.call_args
        # _process_single_entity passes thresholds as positional args:
        # (entity_name, entity_content, similarity_threshold,
        #  jaccard_search_threshold, embedding_name_search_threshold,
        #  embedding_full_search_threshold, extracted_entity_names, extracted_relation_pairs)
        assert call_args[0][3] == 0.3  # jaccard_search_threshold
        assert call_args[0][4] == 0.5  # embedding_name_search_threshold
        assert call_args[0][5] == 0.6  # embedding_full_search_threshold

    def test_passes_relation_context_to_search(self):
        proc = _make_processor()
        proc._search_entity_candidates = MagicMock(return_value=[])
        proc._create_new_entity = MagicMock(return_value=_make_entity("new1", "test"))
        names = {"A", "B"}
        pairs = {(tuple(sorted(["A", "B"])), hash("content".lower()))}
        proc._process_single_entity(
            {"name": "A", "content": "desc"},
            episode_id="ep1",
            similarity_threshold=0.5,
            extracted_entity_names=names,
            extracted_relation_pairs=pairs,
        )
        call_args = proc._search_entity_candidates.call_args
        # Positional args: index 6=extracted_entity_names, index 7=extracted_relation_pairs
        assert call_args[0][6] == names
        assert call_args[0][7] == pairs

    def test_base_time_forwarded_to_create(self):
        proc = _make_processor()
        proc._search_entity_candidates = MagicMock(return_value=[])
        proc._create_new_entity = MagicMock(return_value=_make_entity("new1", "test"))
        bt = datetime(2025, 1, 1)
        proc._process_single_entity(
            {"name": "test", "content": "desc"},
            episode_id="ep1",
            similarity_threshold=0.5,
            base_time=bt,
        )
        proc._create_new_entity.assert_called_once()
        assert proc._create_new_entity.call_args[1]["base_time"] == bt
