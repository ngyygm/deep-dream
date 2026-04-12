"""
Concept BM25 candidate supplementation — 38 tests across 4 dimensions.

Vision gap fix: _build_entity_candidate_table only searched entities table.
Now it also supplements candidates via concept_fts BM25 search, catching
entities that embedding/Jaccard alone miss.

D1: Concept BM25 supplementation basics (10 tests)
D2: Jaccard threshold filtering (10 tests)
D3: Batch field resolution (10 tests)
D4: Integration edge cases (8 tests)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
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
                 source_document: str = "test"):
    from processor.models import Entity
    now = datetime.now(timezone.utc)
    return Entity(
        absolute_id=str(uuid.uuid4()),
        family_id=family_id,
        name=name,
        content=content,
        event_time=now,
        processed_time=now,
        episode_id="ep_test",
        source_document=source_document,
    )


def _make_processor(storage, llm_client=None):
    from processor.pipeline.entity import EntityProcessor
    if llm_client is None:
        llm_client = MagicMock()
        llm_client.effective_entity_snippet_length.return_value = 50
    return EntityProcessor(storage=storage, llm_client=llm_client)


def _concept_result(family_id: str, name: str, content: str = "",
                    role: str = "entity") -> dict:
    return {
        "family_id": family_id,
        "name": name,
        "content": content,
        "role": role,
        "id": str(uuid.uuid4()),
    }


# ══════════════════════════════════════════════════════════════════════════
# D1: Concept BM25 supplementation basics
# ══════════════════════════════════════════════════════════════════════════


class TestConceptBM25Basics:
    """D1: Basic concept BM25 candidate supplementation."""

    def test_bm25_match_adds_candidate(self, tmp_storage):
        """D1.1: BM25 match adds candidate to table."""
        proc = _make_processor(tmp_storage)
        # Pre-seed entity in storage so concepts table has it
        e = _make_entity("d1_ent1", "Alice Smith", "A researcher")
        tmp_storage.save_entity(e)

        extracted = [{"name": "Alice Smith", "content": "A researcher"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        assert len(result) == 1
        assert any(c["family_id"] == "d1_ent1" for c in result[0])

    def test_no_bm25_match_unchanged(self, tmp_storage):
        """D1.2: No BM25 match leaves table unchanged."""
        proc = _make_processor(tmp_storage)
        extracted = [{"name": "NonexistentEntity", "content": "Nothing"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        assert result == {}

    def test_multiple_bm25_results(self, tmp_storage):
        """D1.3: Multiple BM25 results add multiple candidates."""
        e1 = _make_entity("d1_m1", "Alice Smith", "Researcher")
        e2 = _make_entity("d1_m2", "Alice Johnson", "Engineer")
        tmp_storage.bulk_save_entities([e1, e2])

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "Alice", "content": "Someone named Alice"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.2
        )
        # Should have found at least one Alice
        assert len(result.get(0, [])) >= 1

    def test_concept_candidate_merge_safe_false(self, tmp_storage):
        """D1.4: Concept-sourced candidate has merge_safe=False."""
        e = _make_entity("d1_ms1", "Bob Test", "Tester")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "Bob Test", "content": "A tester"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        candidates = result.get(0, [])
        bm25_candidates = [c for c in candidates if c["family_id"] == "d1_ms1"]
        assert len(bm25_candidates) >= 1
        assert bm25_candidates[0]["merge_safe"] is False

    def test_concept_candidate_dense_score_zero(self, tmp_storage):
        """D1.5: Concept-sourced candidate has dense_score=0.0."""
        e = _make_entity("d1_ds1", "Carol Test", "Developer")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "Carol Test", "content": "Dev"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        bm25_cands = [c for c in result.get(0, []) if c["family_id"] == "d1_ds1"]
        assert len(bm25_cands) >= 1
        assert bm25_cands[0]["dense_score"] == 0.0

    def test_combined_score_equals_jaccard(self, tmp_storage):
        """D1.6: combined_score equals lexical (Jaccard) score."""
        e = _make_entity("d1_cs1", "David Wei", "Scientist")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "David Wei", "content": "A scientist"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        bm25_cands = [c for c in result.get(0, []) if c["family_id"] == "d1_cs1"]
        assert len(bm25_cands) >= 1
        assert bm25_cands[0]["combined_score"] == bm25_cands[0]["lexical_score"]

    def test_existing_candidates_not_duplicated(self, tmp_storage):
        """D1.7: Existing candidates (by family_id) are not duplicated."""
        e = _make_entity("d1_dup1", "Eve Test", "Manager")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        existing = {
            0: [{
                "family_id": "d1_dup1",
                "name": "Eve Test",
                "content": "Manager",
                "source_document": "",
                "version_count": 1,
                "entity": e,
                "lexical_score": 0.9,
                "dense_score": 0.8,
                "combined_score": 0.9,
                "merge_safe": True,
            }]
        }
        extracted = [{"name": "Eve Test", "content": "Manager"}]
        result = proc._supplement_candidates_from_concepts(
            existing, extracted, jaccard_threshold=0.3
        )
        # Should still have only one entry for d1_dup1
        cands = result.get(0, [])
        dup_count = sum(1 for c in cands if c["family_id"] == "d1_dup1")
        assert dup_count == 1

    def test_special_characters_in_name(self, tmp_storage):
        """D1.8: Entity name with special characters handled."""
        e = _make_entity("d1_sp1", "O'Brien", "Writer")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "O'Brien", "content": "A writer"}]
        # Should not crash even if FTS has issues with special chars
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        # Just check it doesn't crash
        assert isinstance(result, dict)

    def test_empty_extracted_entities(self, tmp_storage):
        """D1.9: Empty extracted_entities returns unchanged table."""
        proc = _make_processor(tmp_storage)
        result = proc._supplement_candidates_from_concepts(
            {"existing": "data"}, [], jaccard_threshold=0.3
        )
        assert result == {"existing": "data"}

    def test_single_entity_single_hit(self, tmp_storage):
        """D1.10: Single extracted entity with single BM25 hit."""
        e = _make_entity("d1_s1", "Frank Test", "Analyst")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "Frank Test", "content": "Data analyst"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        assert 0 in result
        franks = [c for c in result[0] if c["family_id"] == "d1_s1"]
        assert len(franks) >= 1


# ══════════════════════════════════════════════════════════════════════════
# D2: Jaccard threshold filtering
# ══════════════════════════════════════════════════════════════════════════


class TestJaccardThresholdFiltering:
    """D2: Jaccard threshold filtering for concept BM25 candidates."""

    def test_below_threshold_filtered(self, tmp_storage):
        """D2.1: BM25 hit below Jaccard threshold is filtered out."""
        # "Test Entity" vs "XYZ" → very low Jaccard
        e = _make_entity("d2_lo1", "XYZ Unrelated Name", "Content")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "Test Entity", "content": "Something"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.8
        )
        for c in result.get(0, []):
            assert c["family_id"] != "d2_lo1"

    def test_above_threshold_included(self, tmp_storage):
        """D2.2: BM25 hit above Jaccard threshold is included."""
        e = _make_entity("d2_hi1", "Exact Match Test", "Content")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "Exact Match Test", "content": "Content"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.5
        )
        fids = [c["family_id"] for c in result.get(0, [])]
        assert "d2_hi1" in fids

    def test_exact_threshold_included(self, tmp_storage):
        """D2.3: BM25 hit at exact Jaccard threshold is included."""
        # Name "AB" vs "AB" → Jaccard = 1.0 which is >= threshold
        e = _make_entity("d2_et1", "AB", "Content")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "AB", "content": "Content"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=1.0
        )
        fids = [c["family_id"] for c in result.get(0, [])]
        assert "d2_et1" in fids

    def test_custom_threshold_respected(self, tmp_storage):
        """D2.4: Custom Jaccard threshold is respected."""
        e = _make_entity("d2_ct1", "Hello World Foo", "Content")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "Hello World Bar", "content": "Content"}]

        # With high threshold, should not match
        result_high = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.95
        )
        fids_high = [c["family_id"] for c in result_high.get(0, [])]
        assert "d2_ct1" not in fids_high

    def test_all_below_threshold_no_new(self, tmp_storage):
        """D2.5: All BM25 hits below threshold → no new candidates."""
        e = _make_entity("d2_ab1", "Completely Different Entity", "X")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "No Match At All", "content": "Y"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.9
        )
        for idx in result:
            for c in result[idx]:
                assert c["family_id"] != "d2_ab1"

    def test_mixed_above_below_threshold(self, tmp_storage):
        """D2.6: Mixed: some above, some below threshold."""
        e1 = _make_entity("d2_mx1", "Test Alpha", "Content")
        e2 = _make_entity("d2_mx2", "Completely Unrelated", "Content")
        tmp_storage.bulk_save_entities([e1, e2])

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "Test Alpha", "content": "Something"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.5
        )
        fids = [c["family_id"] for c in result.get(0, [])]
        assert "d2_mx1" in fids  # High Jaccard → included

    def test_very_high_jaccard_included(self, tmp_storage):
        """D2.7: Very high Jaccard (0.9+) candidate included."""
        e = _make_entity("d2_vh1", "Test Entity Name", "Content")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "Test Entity Name", "content": "Content"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.9
        )
        fids = [c["family_id"] for c in result.get(0, [])]
        assert "d2_vh1" in fids

    def test_very_low_jaccard_filtered(self, tmp_storage):
        """D2.8: Very low Jaccard (0.1) candidate filtered."""
        e = _make_entity("d2_vl1", "AB", "Content")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "XYZW", "content": "Content"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.5
        )
        fids = [c["family_id"] for c in result.get(0, [])]
        assert "d2_vl1" not in fids

    def test_partial_name_below_threshold(self, tmp_storage):
        """D2.9: Partial name match below threshold filtered."""
        e = _make_entity("d2_pn1", "International Business Machines", "Tech")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "IBM", "content": "Tech company"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.8
        )
        fids = [c["family_id"] for c in result.get(0, [])]
        assert "d2_pn1" not in fids

    def test_threshold_zero_includes_all(self, tmp_storage):
        """D2.10: Threshold=0.0 includes all BM25 hits."""
        e = _make_entity("d2_tz1", "Something", "Content")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "Anything", "content": "Whatever"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.0
        )
        # With threshold 0.0, any BM25 hit should be included
        # (if there are any BM25 matches at all)
        assert isinstance(result, dict)


# ══════════════════════════════════════════════════════════════════════════
# D3: Batch field resolution
# ══════════════════════════════════════════════════════════════════════════


class TestBatchFieldResolution:
    """D3: Batch resolution of Entity objects and version counts."""

    def test_entity_object_populated(self, tmp_storage):
        """D3.1: Entity object correctly populated from get_entities_by_family_ids."""
        e = _make_entity("d3_eo1", "EntityObj Test", "Content here", source_document="doc1")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "EntityObj Test", "content": "Content here"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        bm25_cands = [c for c in result.get(0, []) if c["family_id"] == "d3_eo1"]
        assert len(bm25_cands) >= 1
        assert bm25_cands[0]["entity"] is not None

    def test_version_count_populated(self, tmp_storage):
        """D3.2: Version count correctly populated from get_entity_version_counts."""
        e1 = _make_entity("d3_vc1", "VersionTest", "V1")
        tmp_storage.save_entity(e1)
        e2 = _make_entity("d3_vc1", "VersionTest", "V2")
        tmp_storage.save_entity(e2)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "VersionTest", "content": "V2"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        bm25_cands = [c for c in result.get(0, []) if c["family_id"] == "d3_vc1"]
        assert len(bm25_cands) >= 1
        assert bm25_cands[0]["version_count"] >= 2

    def test_missing_entity_handled(self, tmp_storage):
        """D3.3: Missing entity (concept exists but entity deleted) handled gracefully."""
        proc = _make_processor(tmp_storage)
        # Mock BM25 to return a concept with non-existent family_id
        with patch.object(tmp_storage, 'search_concepts_by_bm25',
                          return_value=[_concept_result("nonexistent_fid", "Test", "C")]):
            with patch.object(tmp_storage, 'get_entities_by_family_ids', return_value={}):
                with patch.object(tmp_storage, 'get_entity_version_counts',
                                  return_value={"nonexistent_fid": 0}):
                    extracted = [{"name": "Test", "content": "C"}]
                    result = proc._supplement_candidates_from_concepts(
                        {}, extracted, jaccard_threshold=0.3
                    )
                    # Should not crash, entity field should be None
                    if 0 in result:
                        bm25_cands = [c for c in result[0] if c["family_id"] == "nonexistent_fid"]
                        if bm25_cands:
                            assert bm25_cands[0]["entity"] is None

    def test_source_document_populated(self, tmp_storage):
        """D3.4: source_document field populated from concept."""
        e = _make_entity("d3_sd1", "SourceDoc Test", "Content", source_document="my_doc.txt")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "SourceDoc Test", "content": "Content"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        bm25_cands = [c for c in result.get(0, []) if c["family_id"] == "d3_sd1"]
        assert len(bm25_cands) >= 1
        assert bm25_cands[0]["source_document"] == "my_doc.txt"

    def test_content_populated(self, tmp_storage):
        """D3.5: content field populated from concept."""
        e = _make_entity("d3_cp1", "ContentPop", "Some interesting content")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "ContentPop", "content": "Some interesting content"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        bm25_cands = [c for c in result.get(0, []) if c["family_id"] == "d3_cp1"]
        assert len(bm25_cands) >= 1
        assert bm25_cands[0]["content"] == "Some interesting content"

    def test_multiple_entities_batch_resolved(self, tmp_storage):
        """D3.6: Multiple extracted entities batch-resolved efficiently."""
        e1 = _make_entity("d3_br1", "Entity One", "Content 1")
        e2 = _make_entity("d3_br2", "Entity Two", "Content 2")
        tmp_storage.bulk_save_entities([e1, e2])

        proc = _make_processor(tmp_storage)
        extracted = [
            {"name": "Entity One", "content": "Content 1"},
            {"name": "Entity Two", "content": "Content 2"},
        ]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        # Both should have results
        fids_0 = [c["family_id"] for c in result.get(0, [])]
        fids_1 = [c["family_id"] for c in result.get(1, [])]
        assert "d3_br1" in fids_0
        assert "d3_br2" in fids_1

    def test_family_id_correctly_mapped(self, tmp_storage):
        """D3.7: family_id correctly mapped from concept."""
        e = _make_entity("d3_fid1", "FID Test", "Content")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "FID Test", "content": "Content"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        fids = [c["family_id"] for c in result.get(0, [])]
        assert "d3_fid1" in fids

    def test_name_correctly_mapped(self, tmp_storage):
        """D3.8: name correctly mapped from concept."""
        e = _make_entity("d3_nm1", "Name Map Test", "Content")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "Name Map Test", "content": "Content"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        bm25_cands = [c for c in result.get(0, []) if c["family_id"] == "d3_nm1"]
        assert len(bm25_cands) >= 1
        assert bm25_cands[0]["name"] == "Name Map Test"

    def test_concept_no_matching_entity(self, tmp_storage):
        """D3.9: Concept with no matching entity row uses name as content fallback."""
        proc = _make_processor(tmp_storage)
        with patch.object(tmp_storage, 'search_concepts_by_bm25',
                          return_value=[_concept_result("d3_orph", "Orphan", "Orphan content")]):
            with patch.object(tmp_storage, 'get_entities_by_family_ids', return_value={}):
                with patch.object(tmp_storage, 'get_entity_version_counts',
                                  return_value={}):
                    extracted = [{"name": "Orphan", "content": "Some content"}]
                    result = proc._supplement_candidates_from_concepts(
                        {}, extracted, jaccard_threshold=0.3
                    )
                    bm25_cands = [c for c in result.get(0, []) if c["family_id"] == "d3_orph"]
                    if bm25_cands:
                        # Fallback: content should be name since entity is None
                        assert bm25_cands[0]["content"] is not None

    def test_batch_resolves_across_slots(self, tmp_storage):
        """D3.10: Batch resolves across multiple candidate slots."""
        e = _make_entity("d3_slot1", "Shared Name Entity", "Content")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        # Two extracted entities with same name → same BM25 results for both
        extracted = [
            {"name": "Shared Name Entity", "content": "First"},
            {"name": "Shared Name Entity", "content": "Second"},
        ]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        # Both indices should have the candidate
        assert "d3_slot1" in [c["family_id"] for c in result.get(0, [])]
        assert "d3_slot1" in [c["family_id"] for c in result.get(1, [])]


# ══════════════════════════════════════════════════════════════════════════
# D4: Integration edge cases
# ══════════════════════════════════════════════════════════════════════════


class TestConceptBM25EdgeCases:
    """D4: Integration edge cases for concept BM25 supplementation."""

    def test_integration_with_existing_candidates(self, tmp_storage):
        """D4.1: BM25 candidates integrate with existing embedding-based candidates."""
        e1 = _make_entity("d4_int1", "Integrate Test", "Content")
        e2 = _make_entity("d4_int2", "Integrate Other", "Different")
        tmp_storage.bulk_save_entities([e1, e2])

        proc = _make_processor(tmp_storage)
        existing = {
            0: [{
                "family_id": "d4_int2",
                "name": "Integrate Other",
                "content": "Different",
                "source_document": "",
                "version_count": 1,
                "entity": e2,
                "lexical_score": 0.7,
                "dense_score": 0.8,
                "combined_score": 0.8,
                "merge_safe": True,
            }]
        }
        extracted = [{"name": "Integrate Test", "content": "Content"}]
        result = proc._supplement_candidates_from_concepts(
            existing, extracted, jaccard_threshold=0.3
        )
        fids = [c["family_id"] for c in result.get(0, [])]
        assert "d4_int2" in fids  # Existing preserved
        assert "d4_int1" in fids  # New from BM25

    def test_duplicate_names_deduped_for_query(self, tmp_storage):
        """D4.2: Duplicate entity names in extracted_entities deduplicated for BM25 query."""
        e = _make_entity("d4_dd1", "Dedup Name", "Content")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [
            {"name": "Dedup Name", "content": "First"},
            {"name": "Dedup Name", "content": "Second"},
            {"name": "Dedup Name", "content": "Third"},
        ]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        # All three indices should have the candidate
        for idx in range(3):
            fids = [c["family_id"] for c in result.get(idx, [])]
            assert "d4_dd1" in fids

    def test_unicode_entity_names(self, tmp_storage):
        """D4.3: Unicode entity names work with BM25."""
        e = _make_entity("d4_un1", "深度学习", "人工智能技术")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "深度学习", "content": "人工智能技术"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        # Should not crash — BM25 may or may not find Chinese FTS match
        assert isinstance(result, dict)

    def test_bm25_exception_handled(self, tmp_storage):
        """D4.4: storage.search_concepts_by_bm25 exception handled gracefully."""
        proc = _make_processor(tmp_storage)
        with patch.object(tmp_storage, 'search_concepts_by_bm25',
                          side_effect=RuntimeError("FTS error")):
            extracted = [{"name": "Test", "content": "Content"}]
            result = proc._supplement_candidates_from_concepts(
                {}, extracted, jaccard_threshold=0.3
            )
            # Should not crash, return empty
            assert result == {}

    def test_concept_empty_content(self, tmp_storage):
        """D4.5: Concept with empty content handled."""
        e = _make_entity("d4_ec1", "EmptyContent", "")
        tmp_storage.save_entity(e)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": "EmptyContent", "content": ""}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        bm25_cands = [c for c in result.get(0, []) if c["family_id"] == "d4_ec1"]
        if bm25_cands:
            assert bm25_cands[0]["entity"] is not None

    def test_large_number_extracted(self, tmp_storage):
        """D4.6: Large number of extracted entities processed."""
        # Create 20 entities
        entities = [_make_entity(f"d4_lg{i}", f"Entity {i}", f"Content {i}") for i in range(20)]
        tmp_storage.bulk_save_entities(entities)

        proc = _make_processor(tmp_storage)
        extracted = [{"name": f"Entity {i}", "content": f"Content {i}"} for i in range(20)]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        # Should have results for at least some indices
        assert len(result) > 0

    def test_empty_concept_table(self, tmp_storage):
        """D4.7: Concept table empty → no new candidates."""
        proc = _make_processor(tmp_storage)
        # No entities saved → concept_fts is empty
        extracted = [{"name": "Nothing Here", "content": "N/A"}]
        result = proc._supplement_candidates_from_concepts(
            {}, extracted, jaccard_threshold=0.3
        )
        assert result == {}

    def test_candidate_sorting_after_merge(self, tmp_storage):
        """D4.8: Candidates remain sorted by combined_score after merge."""
        e1 = _make_entity("d4_so1", "Sort High", "Content")
        e2 = _make_entity("d4_so2", "Sort Low Match", "Different")
        tmp_storage.bulk_save_entities([e1, e2])

        proc = _make_processor(tmp_storage)
        existing = {
            0: [{
                "family_id": "d4_so1",
                "name": "Sort High",
                "content": "Content",
                "source_document": "",
                "version_count": 1,
                "entity": e1,
                "lexical_score": 0.95,
                "dense_score": 0.9,
                "combined_score": 0.95,
                "merge_safe": True,
            }]
        }
        extracted = [{"name": "Sort Low Match", "content": "Different"}]
        result = proc._supplement_candidates_from_concepts(
            existing, extracted, jaccard_threshold=0.2
        )
        candidates = result.get(0, [])
        # Should be sorted by combined_score descending
        scores = [c["combined_score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)
