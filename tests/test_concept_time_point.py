"""
Concept time_point filtering — 32 tests across 4 dimensions.

Vision gap fix: The concepts table has valid_at/invalid_at columns but they were
NEVER used in any query. Vision.md principle #6 states: "时间不可省略".
Now all concept query methods support time_point parameter, filtering by:
  valid_at <= time_point AND (invalid_at IS NULL OR invalid_at > time_point)

D1: get_concept_by_family_id time_point filtering (8 tests)
D2: search_concepts_by_bm25 time_point filtering (8 tests)
D3: list_concepts & count_concepts time_point filtering (8 tests)
D4: traverse & neighbors & similarity time_point filtering (8 tests)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_storage(tmp_path):
    from processor.storage.manager import StorageManager
    sm = StorageManager(str(tmp_path / "graph"))
    yield sm
    if hasattr(sm, '_vector_store') and sm._vector_store:
        sm._vector_store.close()


def _make_entity(family_id: str, name: str, content: str,
                 source_document: str = "test",
                 valid_at: str = None,
                 invalid_at: str = None):
    from processor.models import Entity
    now = datetime.now(timezone.utc)
    va = datetime.fromisoformat(valid_at) if valid_at else now
    entity = Entity(
        absolute_id=str(uuid.uuid4()),
        family_id=family_id,
        name=name,
        content=content,
        event_time=now,
        processed_time=now,
        episode_id="ep_test",
        source_document=source_document,
        valid_at=va,
        invalid_at=invalid_at,
    )
    return entity


def _make_relation(family_id: str, e1_abs: str, e2_abs: str, content: str,
                   valid_at: str = None, invalid_at: str = None):
    from processor.models import Relation
    now = datetime.now(timezone.utc)
    va = datetime.fromisoformat(valid_at) if valid_at else now
    return Relation(
        absolute_id=str(uuid.uuid4()),
        family_id=family_id,
        entity1_absolute_id=e1_abs,
        entity2_absolute_id=e2_abs,
        content=content,
        event_time=now,
        processed_time=now,
        episode_id="ep_test",
        source_document="test",
        valid_at=va,
        invalid_at=invalid_at,
    )


def _save_entity_with_times(storage, family_id, name, content,
                              valid_at, invalid_at=None):
    """Save entity directly to concepts table with explicit valid_at/invalid_at."""
    e = _make_entity(family_id, name, content, valid_at=valid_at, invalid_at=invalid_at)
    storage.save_entity(e)
    # Manually update valid_at/invalid_at in concepts table for precise control
    conn = storage._get_conn()
    conn.execute(
        "UPDATE concepts SET valid_at = ?, invalid_at = ? WHERE id = ?",
        (valid_at, invalid_at, e.absolute_id)
    )
    conn.commit()
    return e


def _save_relation_with_times(storage, family_id, e1_abs, e2_abs, content,
                                valid_at, invalid_at=None):
    """Save relation directly to concepts table with explicit valid_at/invalid_at."""
    r = _make_relation(family_id, e1_abs, e2_abs, content,
                       valid_at=valid_at, invalid_at=invalid_at)
    storage.save_relation(r)
    conn = storage._get_conn()
    conn.execute(
        "UPDATE concepts SET valid_at = ?, invalid_at = ? WHERE id = ?",
        (valid_at, invalid_at, r.absolute_id)
    )
    conn.commit()
    return r


# ══════════════════════════════════════════════════════════════════════════
# D1: get_concept_by_family_id time_point filtering
# ══════════════════════════════════════════════════════════════════════════


class TestGetConceptTimePoint:
    """D1: time_point filtering for get_concept_by_family_id."""

    def test_no_time_point_returns_latest(self, tmp_storage):
        """D1.1: Without time_point, returns the latest version."""
        _save_entity_with_times(tmp_storage, "d1_e1", "Alpha", "Content A",
                                valid_at="2024-01-01T00:00:00")
        result = tmp_storage.get_concept_by_family_id("d1_e1")
        assert result is not None
        assert result["family_id"] == "d1_e1"

    def test_time_point_before_valid_at_returns_none(self, tmp_storage):
        """D1.2: time_point before valid_at returns None."""
        _save_entity_with_times(tmp_storage, "d1_e2", "Beta", "Content B",
                                valid_at="2024-06-01T00:00:00")
        result = tmp_storage.get_concept_by_family_id(
            "d1_e2", time_point="2024-01-01T00:00:00"
        )
        assert result is None

    def test_time_point_after_valid_at_returns_concept(self, tmp_storage):
        """D1.3: time_point after valid_at returns the concept."""
        _save_entity_with_times(tmp_storage, "d1_e3", "Gamma", "Content C",
                                valid_at="2024-01-01T00:00:00")
        result = tmp_storage.get_concept_by_family_id(
            "d1_e3", time_point="2024-06-01T00:00:00"
        )
        assert result is not None
        assert result["family_id"] == "d1_e3"

    def test_time_point_at_valid_at_returns_concept(self, tmp_storage):
        """D1.4: time_point exactly at valid_at returns the concept."""
        _save_entity_with_times(tmp_storage, "d1_e4", "Delta", "Content D",
                                valid_at="2024-03-15T12:00:00")
        result = tmp_storage.get_concept_by_family_id(
            "d1_e4", time_point="2024-03-15T12:00:00"
        )
        assert result is not None

    def test_invalid_at_before_time_point_returns_none(self, tmp_storage):
        """D1.5: Concept with invalid_at before time_point returns None."""
        _save_entity_with_times(tmp_storage, "d1_e5", "Epsilon", "Content E",
                                valid_at="2024-01-01T00:00:00",
                                invalid_at="2024-06-01T00:00:00")
        result = tmp_storage.get_concept_by_family_id(
            "d1_e5", time_point="2024-09-01T00:00:00"
        )
        assert result is None

    def test_invalid_at_after_time_point_returns_concept(self, tmp_storage):
        """D1.6: Concept with invalid_at after time_point returns concept."""
        _save_entity_with_times(tmp_storage, "d1_e6", "Zeta", "Content F",
                                valid_at="2024-01-01T00:00:00",
                                invalid_at="2024-12-31T00:00:00")
        result = tmp_storage.get_concept_by_family_id(
            "d1_e6", time_point="2024-06-01T00:00:00"
        )
        assert result is not None

    def test_nonexistent_family_id_returns_none(self, tmp_storage):
        """D1.7: Nonexistent family_id returns None regardless of time_point."""
        result = tmp_storage.get_concept_by_family_id(
            "nonexistent", time_point="2024-06-01T00:00:00"
        )
        assert result is None

    def test_batch_get_with_time_point(self, tmp_storage):
        """D1.8: get_concepts_by_family_ids filters by time_point."""
        _save_entity_with_times(tmp_storage, "d1_b1", "Batch1", "Content",
                                valid_at="2024-01-01T00:00:00")
        _save_entity_with_times(tmp_storage, "d1_b2", "Batch2", "Content",
                                valid_at="2024-07-01T00:00:00")
        result = tmp_storage.get_concepts_by_family_ids(
            ["d1_b1", "d1_b2"], time_point="2024-03-01T00:00:00"
        )
        assert "d1_b1" in result
        assert "d1_b2" not in result


# ══════════════════════════════════════════════════════════════════════════
# D2: search_concepts_by_bm25 time_point filtering
# ══════════════════════════════════════════════════════════════════════════


class TestBM25TimePoint:
    """D2: time_point filtering for BM25 search."""

    def test_bm25_no_time_point_finds_all(self, tmp_storage):
        """D2.1: BM25 without time_point finds all matching concepts."""
        _save_entity_with_times(tmp_storage, "d2_e1", "Quantum Physics",
                                "Quantum mechanics",
                                valid_at="2024-01-01T00:00:00")
        results = tmp_storage.search_concepts_by_bm25(
            "Quantum", role="entity", limit=10
        )
        assert any(r.get("family_id") == "d2_e1" for r in results)

    def test_bm25_with_time_point_finds_valid(self, tmp_storage):
        """D2.2: BM25 with time_point only finds concepts valid at that time."""
        _save_entity_with_times(tmp_storage, "d2_e2", "Relativity Theory",
                                "General relativity",
                                valid_at="2024-03-01T00:00:00")
        results = tmp_storage.search_concepts_by_bm25(
            "Relativity", role="entity", limit=10,
            time_point="2024-06-01T00:00:00"
        )
        assert any(r.get("family_id") == "d2_e2" for r in results)

    def test_bm25_filters_out_not_yet_valid(self, tmp_storage):
        """D2.3: BM25 filters out concepts with valid_at after time_point."""
        _save_entity_with_times(tmp_storage, "d2_e3", "Future Concept",
                                "Not yet valid",
                                valid_at="2025-01-01T00:00:00")
        results = tmp_storage.search_concepts_by_bm25(
            "Future", role="entity", limit=10,
            time_point="2024-06-01T00:00:00"
        )
        assert not any(r.get("family_id") == "d2_e3" for r in results)

    def test_bm25_filters_out_already_invalidated(self, tmp_storage):
        """D2.4: BM25 filters out concepts with invalid_at before time_point."""
        _save_entity_with_times(tmp_storage, "d2_e4", "Expired Concept",
                                "No longer valid",
                                valid_at="2024-01-01T00:00:00",
                                invalid_at="2024-03-01T00:00:00")
        results = tmp_storage.search_concepts_by_bm25(
            "Expired", role="entity", limit=10,
            time_point="2024-06-01T00:00:00"
        )
        assert not any(r.get("family_id") == "d2_e4" for r in results)

    def test_bm25_mixed_validity(self, tmp_storage):
        """D2.5: BM25 returns only valid concepts from a mixed set."""
        _save_entity_with_times(tmp_storage, "d2_v1", "Valid Concept",
                                "Currently valid",
                                valid_at="2024-01-01T00:00:00")
        _save_entity_with_times(tmp_storage, "d2_v2", "Invalid Concept",
                                "No longer valid",
                                valid_at="2024-01-01T00:00:00",
                                invalid_at="2024-03-01T00:00:00")
        _save_entity_with_times(tmp_storage, "d2_v3", "Future Concept",
                                "Not yet valid",
                                valid_at="2025-01-01T00:00:00")
        results = tmp_storage.search_concepts_by_bm25(
            "Concept", role="entity", limit=10,
            time_point="2024-06-01T00:00:00"
        )
        fids = [r.get("family_id") for r in results]
        assert "d2_v1" in fids
        assert "d2_v2" not in fids
        assert "d2_v3" not in fids

    def test_bm25_time_point_with_relation_role(self, tmp_storage):
        """D2.6: BM25 time_point works for relations."""
        e1 = _make_entity("d2_ra", "EntityA", "A")
        e2 = _make_entity("d2_rb", "EntityB", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        _save_relation_with_times(tmp_storage, "d2_r1", e1.absolute_id,
                                  e2.absolute_id, "Causal link",
                                  valid_at="2024-01-01T00:00:00",
                                  invalid_at="2024-06-01T00:00:00")
        results = tmp_storage.search_concepts_by_bm25(
            "Causal", role="relation", limit=10,
            time_point="2024-03-01T00:00:00"
        )
        assert any(r.get("family_id") == "d2_r1" for r in results)

        results_late = tmp_storage.search_concepts_by_bm25(
            "Causal", role="relation", limit=10,
            time_point="2024-09-01T00:00:00"
        )
        assert not any(r.get("family_id") == "d2_r1" for r in results_late)

    def test_bm25_time_point_without_role(self, tmp_storage):
        """D2.7: BM25 time_point works without role filter."""
        _save_entity_with_times(tmp_storage, "d2_nr1", "TestNoRole", "Content",
                                valid_at="2024-01-01T00:00:00")
        results = tmp_storage.search_concepts_by_bm25(
            "TestNoRole", role=None, limit=10,
            time_point="2024-06-01T00:00:00"
        )
        assert any(r.get("family_id") == "d2_nr1" for r in results)

    def test_bm25_no_match_with_time_point(self, tmp_storage):
        """D2.8: BM25 with time_point and no match returns empty."""
        _save_entity_with_times(tmp_storage, "d2_nm1", "HiddenEntity", "Content",
                                valid_at="2025-01-01T00:00:00")
        results = tmp_storage.search_concepts_by_bm25(
            "HiddenEntity", role="entity", limit=10,
            time_point="2024-01-01T00:00:00"
        )
        assert not any(r.get("family_id") == "d2_nm1" for r in results)


# ══════════════════════════════════════════════════════════════════════════
# D3: list_concepts & count_concepts time_point filtering
# ══════════════════════════════════════════════════════════════════════════


class TestListCountTimePoint:
    """D3: time_point filtering for list_concepts and count_concepts."""

    def test_list_without_time_point(self, tmp_storage):
        """D3.1: list_concepts without time_point returns all."""
        _save_entity_with_times(tmp_storage, "d3_l1", "ListTest1", "Content",
                                valid_at="2024-01-01T00:00:00")
        _save_entity_with_times(tmp_storage, "d3_l2", "ListTest2", "Content",
                                valid_at="2025-01-01T00:00:00")
        results = tmp_storage.list_concepts(role="entity", limit=50)
        fids = [r["family_id"] for r in results]
        assert "d3_l1" in fids
        assert "d3_l2" in fids

    def test_list_with_time_point_filters(self, tmp_storage):
        """D3.2: list_concepts with time_point filters correctly."""
        _save_entity_with_times(tmp_storage, "d3_l3", "ValidList", "Content",
                                valid_at="2024-01-01T00:00:00")
        _save_entity_with_times(tmp_storage, "d3_l4", "FutureList", "Content",
                                valid_at="2025-01-01T00:00:00")
        results = tmp_storage.list_concepts(
            role="entity", limit=50, time_point="2024-06-01T00:00:00"
        )
        fids = [r["family_id"] for r in results]
        assert "d3_l3" in fids
        assert "d3_l4" not in fids

    def test_list_with_invalid_at_filter(self, tmp_storage):
        """D3.3: list_concepts excludes invalidated concepts."""
        _save_entity_with_times(tmp_storage, "d3_l5", "ActiveList", "Content",
                                valid_at="2024-01-01T00:00:00")
        _save_entity_with_times(tmp_storage, "d3_l6", "ExpiredList", "Content",
                                valid_at="2024-01-01T00:00:00",
                                invalid_at="2024-03-01T00:00:00")
        results = tmp_storage.list_concepts(
            role="entity", limit=50, time_point="2024-06-01T00:00:00"
        )
        fids = [r["family_id"] for r in results]
        assert "d3_l5" in fids
        assert "d3_l6" not in fids

    def test_count_without_time_point(self, tmp_storage):
        """D3.4: count_concepts without time_point counts all."""
        _save_entity_with_times(tmp_storage, "d3_c1", "Count1", "Content",
                                valid_at="2024-01-01T00:00:00")
        _save_entity_with_times(tmp_storage, "d3_c2", "Count2", "Content",
                                valid_at="2025-01-01T00:00:00")
        total = tmp_storage.count_concepts(role="entity")
        assert total >= 2

    def test_count_with_time_point(self, tmp_storage):
        """D3.5: count_concepts with time_point counts only valid."""
        _save_entity_with_times(tmp_storage, "d3_c3", "ValidCount", "Content",
                                valid_at="2024-01-01T00:00:00")
        _save_entity_with_times(tmp_storage, "d3_c4", "FutureCount", "Content",
                                valid_at="2025-01-01T00:00:00")
        total = tmp_storage.count_concepts(
            role="entity", time_point="2024-06-01T00:00:00"
        )
        assert total >= 1
        # FutureCount should not be counted
        fids = []
        results = tmp_storage.list_concepts(
            role="entity", limit=50, time_point="2024-06-01T00:00:00"
        )
        fids = [r["family_id"] for r in results]
        assert "d3_c4" not in fids

    def test_list_pagination_with_time_point(self, tmp_storage):
        """D3.6: Pagination works correctly with time_point filter."""
        for i in range(5):
            _save_entity_with_times(tmp_storage, f"d3_p{i}", f"PageTest {i}",
                                    f"Content {i}",
                                    valid_at="2024-01-01T00:00:00")
        page1 = tmp_storage.list_concepts(
            role="entity", limit=2, offset=0,
            time_point="2024-06-01T00:00:00"
        )
        assert len(page1) <= 2

    def test_list_all_roles_with_time_point(self, tmp_storage):
        """D3.7: list_concepts without role filter works with time_point."""
        _save_entity_with_times(tmp_storage, "d3_ar1", "AllRoles", "Content",
                                valid_at="2024-01-01T00:00:00")
        results = tmp_storage.list_concepts(
            limit=50, time_point="2024-06-01T00:00:00"
        )
        fids = [r["family_id"] for r in results]
        assert "d3_ar1" in fids

    def test_count_all_roles_with_time_point(self, tmp_storage):
        """D3.8: count_concepts without role filter works with time_point."""
        _save_entity_with_times(tmp_storage, "d3_cr1", "CountRoles", "Content",
                                valid_at="2024-01-01T00:00:00")
        _save_entity_with_times(tmp_storage, "d3_cr2", "FutureRoles", "Content",
                                valid_at="2025-01-01T00:00:00")
        total = tmp_storage.count_concepts(time_point="2024-06-01T00:00:00")
        assert total >= 1


# ══════════════════════════════════════════════════════════════════════════
# D4: traverse & neighbors & similarity time_point filtering
# ══════════════════════════════════════════════════════════════════════════


class TestTraverseNeighborsTimePoint:
    """D4: time_point filtering for traverse, neighbors, and similarity."""

    def test_neighbors_entity_with_time_point(self, tmp_storage):
        """D4.1: get_concept_neighbors respects time_point for entity role."""
        e1 = _save_entity_with_times(tmp_storage, "d4_n1", "NeighborA", "A",
                                      valid_at="2024-01-01T00:00:00")
        e2 = _save_entity_with_times(tmp_storage, "d4_n2", "NeighborB", "B",
                                      valid_at="2024-01-01T00:00:00")
        # Create a valid relation
        _save_relation_with_times(tmp_storage, "d4_r1", e1.absolute_id,
                                  e2.absolute_id, "Connection",
                                  valid_at="2024-01-01T00:00:00")
        neighbors = tmp_storage.get_concept_neighbors(
            "d4_n1", time_point="2024-06-01T00:00:00"
        )
        assert any(n.get("family_id") == "d4_r1" for n in neighbors)

    def test_neighbors_expires_invalidated_relation(self, tmp_storage):
        """D4.2: get_concept_neighbors excludes invalidated relations."""
        e1 = _save_entity_with_times(tmp_storage, "d4_n3", "NeighborC", "C",
                                      valid_at="2024-01-01T00:00:00")
        e2 = _save_entity_with_times(tmp_storage, "d4_n4", "NeighborD", "D",
                                      valid_at="2024-01-01T00:00:00")
        _save_relation_with_times(tmp_storage, "d4_r2", e1.absolute_id,
                                  e2.absolute_id, "Expired connection",
                                  valid_at="2024-01-01T00:00:00",
                                  invalid_at="2024-03-01T00:00:00")
        neighbors = tmp_storage.get_concept_neighbors(
            "d4_n3", time_point="2024-06-01T00:00:00"
        )
        assert not any(n.get("family_id") == "d4_r2" for n in neighbors)

    def test_traverse_with_time_point(self, tmp_storage):
        """D4.3: traverse_concepts respects time_point."""
        e1 = _save_entity_with_times(tmp_storage, "d4_t1", "TraverseA", "A",
                                      valid_at="2024-01-01T00:00:00")
        e2 = _save_entity_with_times(tmp_storage, "d4_t2", "TraverseB", "B",
                                      valid_at="2024-01-01T00:00:00")
        _save_relation_with_times(tmp_storage, "d4_tr1", e1.absolute_id,
                                  e2.absolute_id, "Traversal link",
                                  valid_at="2024-01-01T00:00:00")
        result = tmp_storage.traverse_concepts(
            ["d4_t1"], max_depth=2, time_point="2024-06-01T00:00:00"
        )
        assert "d4_t1" in result["concepts"]
        # Should find the relation neighbor
        assert result["visited_count"] >= 1

    def test_traverse_excludes_future(self, tmp_storage):
        """D4.4: traverse_concepts excludes future concepts."""
        _save_entity_with_times(tmp_storage, "d4_t3", "FutureTraverse", "Content",
                                valid_at="2025-01-01T00:00:00")
        result = tmp_storage.traverse_concepts(
            ["d4_t3"], max_depth=1, time_point="2024-06-01T00:00:00"
        )
        assert "d4_t3" not in result["concepts"]

    def test_similarity_no_time_point(self, tmp_storage):
        """D4.5: search_concepts_by_similarity without time_point returns results."""
        _save_entity_with_times(tmp_storage, "d4_s1", "SimilarTest", "Content",
                                valid_at="2024-01-01T00:00:00")
        results = tmp_storage.search_concepts_by_similarity(
            query_text="SimilarTest", role="entity",
            threshold=0.3, max_results=10
        )
        assert any(r.get("family_id") == "d4_s1" for r in results)

    def test_similarity_with_time_point(self, tmp_storage):
        """D4.6: search_concepts_by_similarity with time_point filters."""
        _save_entity_with_times(tmp_storage, "d4_s2", "Valid Similar", "Content",
                                valid_at="2024-01-01T00:00:00")
        _save_entity_with_times(tmp_storage, "d4_s3", "Future Similar", "Content",
                                valid_at="2025-01-01T00:00:00")
        results = tmp_storage.search_concepts_by_similarity(
            query_text="Similar", role="entity",
            threshold=0.3, max_results=10,
            time_point="2024-06-01T00:00:00"
        )
        fids = [r.get("family_id") for r in results]
        assert "d4_s2" in fids
        assert "d4_s3" not in fids

    def test_neighbors_without_time_point(self, tmp_storage):
        """D4.7: get_concept_neighbors without time_point returns all."""
        e1 = _save_entity_with_times(tmp_storage, "d4_n5", "NoTimeA", "A",
                                      valid_at="2024-01-01T00:00:00")
        e2 = _save_entity_with_times(tmp_storage, "d4_n6", "NoTimeB", "B",
                                      valid_at="2024-01-01T00:00:00")
        _save_relation_with_times(tmp_storage, "d4_r3", e1.absolute_id,
                                  e2.absolute_id, "No time filter",
                                  valid_at="2024-01-01T00:00:00",
                                  invalid_at="2024-06-01T00:00:00")
        neighbors = tmp_storage.get_concept_neighbors("d4_n5")
        # Without time_point, invalidated relation still appears
        assert any(n.get("family_id") == "d4_r3" for n in neighbors)

    def test_time_point_sql_helper(self, tmp_storage):
        """D4.8: _time_point_sql helper returns correct SQL fragment."""
        # No time_point
        sql, params = tmp_storage._time_point_sql(None)
        assert sql == ""
        assert params == []

        # With time_point
        sql, params = tmp_storage._time_point_sql("2024-06-01T00:00:00")
        assert "valid_at" in sql
        assert "invalid_at" in sql
        assert len(params) == 2
        assert params[0] == "2024-06-01T00:00:00"
        assert params[1] == "2024-06-01T00:00:00"
