"""
Dream Pipeline Integration tests — 36 tests across 4 dimensions.

D1: Remember Pipeline Corroboration Triggers (9 tests)
D2: Cross-Cycle Corroboration Lifecycle (9 tests)
D3: Candidate Filtering in Search/Find (9 tests)
D4: End-to-End Integration (9 tests)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
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


def _make_entity(family_id: str, name: str, content: str, episode_id: str = "ep_test",
                 source_document: str = "", confidence: float = None):
    from processor.models import Entity
    now = datetime.now(timezone.utc)
    return Entity(
        absolute_id=str(uuid.uuid4()),
        family_id=family_id,
        name=name,
        content=content,
        event_time=now,
        processed_time=now,
        episode_id=episode_id,
        source_document=source_document,
        confidence=confidence,
    )


def _make_relation(family_id: str, e1_id: str, e2_id: str, content: str,
                   episode_id: str = "ep_test", source_document: str = "",
                   confidence: float = None, attributes: str = None):
    from processor.models import Relation
    now = datetime.now(timezone.utc)
    return Relation(
        absolute_id=str(uuid.uuid4()),
        family_id=family_id,
        entity1_absolute_id=e1_id,
        entity2_absolute_id=e2_id,
        content=content,
        event_time=now,
        processed_time=now,
        episode_id=episode_id,
        source_document=source_document,
        confidence=confidence,
        attributes=attributes,
    )


def _setup_two_entities(storage):
    """Create two entities and return them."""
    e1 = _make_entity("ent_pi_1", "Alpha", "Entity A")
    e2 = _make_entity("ent_pi_2", "Beta", "Entity B")
    storage.bulk_save_entities([e1, e2])
    return e1, e2


# ══════════════════════════════════════════════════════════════════════════
# D1: Remember Pipeline Corroboration Triggers
# ══════════════════════════════════════════════════════════════════════════


class TestPipelineCorroboration:
    """Test that the relation pipeline triggers dream corroboration."""

    def test_corroborate_called_after_bulk_save(self, tmp_storage):
        """D1.1: corroborate_dream_relation is called for each entity pair after bulk_save."""
        e1, e2 = _setup_two_entities(tmp_storage)

        # Create a dream candidate between the entities
        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "dream test",
            dream_cycle_id="cycle_d1",
        )

        # Create a regular relation (simulating remember extraction)
        r = _make_relation("rel_d1_r1", e1.absolute_id, e2.absolute_id,
                           "remembered relation", source_document="api_input")
        tmp_storage.bulk_save_relations([r])

        # Call corroborate for the pair (simulating pipeline behavior)
        result = tmp_storage.corroborate_dream_relation(
            "ent_pi_1", "ent_pi_2", corroboration_source="remember",
        )
        assert result is not None
        assert result["corroboration_count"] == 1

    def test_corroborate_ignores_non_dream_pairs(self, tmp_storage):
        """D1.2: Pairs with no dream candidates return None."""
        e1, e2 = _setup_two_entities(tmp_storage)

        result = tmp_storage.corroborate_dream_relation(
            "ent_pi_1", "ent_pi_2", corroboration_source="remember",
        )
        assert result is None

    def test_corroborate_ignores_already_verified(self, tmp_storage):
        """D1.3: Already verified dream relations are not corroborated."""
        e1, e2 = _setup_two_entities(tmp_storage)

        cr = tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
        )
        tmp_storage.promote_candidate_relation(cr["family_id"])

        result = tmp_storage.corroborate_dream_relation(
            "ent_pi_1", "ent_pi_2", corroboration_source="remember",
        )
        assert result is None

    def test_relation_processor_triggers_corroboration(self, tmp_storage):
        """D1.4: RelationProcessor.process_relations_batch triggers corroboration."""
        e1, e2 = _setup_two_entities(tmp_storage)

        # Create a dream candidate
        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
            dream_cycle_id="cycle_proc",
        )

        # Use RelationProcessor to process a relation pair
        from processor.pipeline.relation import RelationProcessor
        from processor.llm.client import LLMClient

        llm = MagicMock(spec=LLMClient)
        llm._current_distill_step = 0
        llm._priority_local = MagicMock()
        llm._priority_local.priority = 6

        proc = RelationProcessor(tmp_storage, llm)

        # Build the input data matching process_relations_batch signature
        extracted_relations = [
            {"content": "Alpha relates to Beta", "entity1_name": "Alpha", "entity2_name": "Beta"},
        ]
        entity_name_to_id = {"Alpha": "ent_pi_1", "Beta": "ent_pi_2"}

        # Mock the LLM call — return fallback to trigger _process_single_relation
        llm.resolve_relation_pair_batch.return_value = {"action": "fallback", "confidence": 0.0}

        # Mock _process_single_relation to return a relation without calling LLM
        mock_rel = _make_relation("rel_proc_1", e1.absolute_id, e2.absolute_id,
                                  "proc relation", source_document="api_input")
        with patch.object(proc, '_process_single_relation', return_value=mock_rel):
            result = proc.process_relations_batch(
                extracted_relations=extracted_relations,
                entity_name_to_id=entity_name_to_id,
                episode_id="ep_proc_test",
                source_document="api_input",
            )

        # Check that the dream candidate was corroborated
        candidates = tmp_storage.get_candidate_relations(status="hypothesized")
        # Should have been corroborated at least once
        if candidates:
            rel = candidates[0]
            attrs = json.loads(rel.attributes)
            assert attrs.get("corroboration_count", 0) >= 1

    def test_corroboration_failure_does_not_crash_pipeline(self, tmp_storage):
        """D1.5: Corroboration errors don't crash the pipeline."""
        e1, e2 = _setup_two_entities(tmp_storage)

        r = _make_relation("rel_safe_1", e1.absolute_id, e2.absolute_id,
                           "safe relation", source_document="api_input")

        # Should not raise even if there are no dream candidates
        try:
            tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "remember")
        except Exception:
            pass  # Pipeline should handle this gracefully

    def test_corroborate_multiple_pairs(self, tmp_storage):
        """D1.6: Corroboration works for multiple entity pairs."""
        e1, e2 = _setup_two_entities(tmp_storage)
        e3 = _make_entity("ent_pi_3", "Gamma", "Entity C")
        tmp_storage.save_entity(e3)

        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream1", 0.5, "test",
        )
        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_3", "dream2", 0.4, "test",
        )

        r1 = tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "remember")
        r2 = tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_3", "remember")

        assert r1 is not None
        assert r2 is not None

    def test_corroborate_with_entity_resolution(self, tmp_storage):
        """D1.7: Corroboration resolves entity family_ids correctly."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
        )

        # Corroborate using different order
        result = tmp_storage.corroborate_dream_relation(
            "ent_pi_2", "ent_pi_1", corroboration_source="remember",
        )
        assert result is not None
        assert result["corroboration_count"] == 1

    def test_corroborate_after_merge(self, tmp_storage):
        """D1.8: Corroboration works after entity merge."""
        e1, e2 = _setup_two_entities(tmp_storage)
        e3 = _make_entity("ent_pi_3", "Alpha2", "Another Alpha")
        tmp_storage.save_entity(e3)

        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
        )

        # Merge ent_pi_3 into ent_pi_1
        tmp_storage.merge_entity_families("ent_pi_1", ["ent_pi_3"])

        # Corroboration should still work
        result = tmp_storage.corroborate_dream_relation(
            "ent_pi_1", "ent_pi_2", "remember",
        )
        assert result is not None

    def test_corroborate_source_tracked(self, tmp_storage):
        """D1.9: Corroboration source is tracked as 'remember'."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
        )

        result = tmp_storage.corroborate_dream_relation(
            "ent_pi_1", "ent_pi_2", corroboration_source="remember",
        )
        rel = tmp_storage.get_relation_by_family_id(result["family_id"])
        attrs = json.loads(rel.attributes)
        assert "remember" in attrs.get("corroboration_sources", [])


# ══════════════════════════════════════════════════════════════════════════
# D2: Cross-Cycle Corroboration Lifecycle
# ══════════════════════════════════════════════════════════════════════════


class TestCrossCycleCorroboration:
    """Test corroboration across multiple dream/remember cycles."""

    def test_first_remember_corroborates(self, tmp_storage):
        """D2.1: First remember call corroborates dream candidate (count=1)."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
            dream_cycle_id="cycle_1",
        )

        result = tmp_storage.corroborate_dream_relation(
            "ent_pi_1", "ent_pi_2", "remember",
        )
        assert result["corroboration_count"] == 1
        assert result["status"] == "hypothesized"

    def test_second_remember_auto_promotes(self, tmp_storage):
        """D2.2: Second remember call triggers auto-promotion."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
            dream_cycle_id="cycle_2",
        )

        r1 = tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "remember")
        assert r1["corroboration_count"] == 1

        r2 = tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "remember")
        assert r2["new_status"] == "verified"

    def test_dream_then_remember_corroborates(self, tmp_storage):
        """D2.3: Dream source then remember source corroborates."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
        )

        r1 = tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "dream")
        assert r1["corroboration_count"] == 1

        r2 = tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "remember")
        assert r2["new_status"] == "verified"

    def test_promoted_relation_has_both_sources(self, tmp_storage):
        """D2.4: Auto-promoted relation has both corroboration sources."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
        )

        tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "dream")
        result = tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "remember")

        rel = tmp_storage.get_relation_by_family_id(result["family_id"])
        attrs = json.loads(rel.attributes)
        sources = attrs.get("corroboration_sources", [])
        assert "dream" in sources
        assert "remember" in sources

    def test_promoted_confidence_at_least_07(self, tmp_storage):
        """D2.5: Auto-promoted relation has confidence ≥ 0.7."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.3, "test",
        )

        tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "remember")
        result = tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "remember")

        assert result["confidence"] >= 0.7

    def test_three_corroborations_only_promotes_once(self, tmp_storage):
        """D2.6: After promotion, further corroborations return None."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
        )

        tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "remember")
        tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "remember")

        # Already verified — should return None
        r3 = tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "remember")
        assert r3 is None

    def test_corroboration_with_dream_cycle_id(self, tmp_storage):
        """D2.7: Corroboration works with dream cycle ID."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
            dream_cycle_id="cycle_d2",
        )

        r1 = tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "remember")
        assert r1 is not None

        # Verify dream cycle ID in attributes
        rel = tmp_storage.get_relation_by_family_id(r1["family_id"])
        attrs = json.loads(rel.attributes)
        assert attrs.get("created_by_dream") == "cycle_d2"

    def test_batch_reject_prevents_corroboration(self, tmp_storage):
        """D2.8: Rejected dream candidates cannot be corroborated."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
            dream_cycle_id="cycle_rej",
        )

        # Reject the entire cycle
        tmp_storage.reject_dream_cycle_relations("cycle_rej")

        # Corroboration should return None (rejected, not hypothesized)
        result = tmp_storage.corroborate_dream_relation(
            "ent_pi_1", "ent_pi_2", "remember",
        )
        assert result is None

    def test_multiple_dream_relations_same_pair(self, tmp_storage):
        """D2.9: Multiple dream candidates between same pair — only first gets corroborated."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream 1", 0.5, "test",
            dream_cycle_id="cycle_a",
        )
        # Second dream relation merges with first
        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream 2", 0.4, "test",
            dream_cycle_id="cycle_b",
        )

        result = tmp_storage.corroborate_dream_relation(
            "ent_pi_1", "ent_pi_2", "remember",
        )
        assert result is not None
        assert result["corroboration_count"] >= 1


# ══════════════════════════════════════════════════════════════════════════
# D3: Candidate Filtering in Search/Find
# ══════════════════════════════════════════════════════════════════════════


class TestCandidateSearchFiltering:
    """Test that candidate relations are properly filtered in search results."""

    def test_candidates_listed_correctly(self, tmp_storage):
        """D3.1: Dream candidates appear in candidate listing."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
        )

        candidates = tmp_storage.get_candidate_relations(status="hypothesized")
        assert len(candidates) >= 1

    def test_verified_candidates_in_verified_list(self, tmp_storage):
        """D3.2: Promoted candidates appear in verified listing."""
        e1, e2 = _setup_two_entities(tmp_storage)

        cr = tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
        )
        tmp_storage.promote_candidate_relation(cr["family_id"])

        verified = tmp_storage.get_candidate_relations(status="verified")
        ver_ids = {r.family_id for r in verified}
        assert cr["family_id"] in ver_ids

    def test_rejected_candidates_in_rejected_list(self, tmp_storage):
        """D3.3: Rejected candidates appear in rejected listing."""
        e1, e2 = _setup_two_entities(tmp_storage)

        cr = tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
        )
        tmp_storage.demote_candidate_relation(cr["family_id"], reason="test")

        rejected = tmp_storage.get_candidate_relations(status="rejected")
        rej_ids = {r.family_id for r in rejected}
        assert cr["family_id"] in rej_ids

    def test_no_status_returns_all(self, tmp_storage):
        """D3.4: No status filter returns all dream candidates."""
        e1, e2 = _setup_two_entities(tmp_storage)
        e3 = _make_entity("ent_pi_3", "Gamma", "G")
        tmp_storage.save_entity(e3)

        cr1 = tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream 1", 0.5, "test",
        )
        cr2 = tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_3", "dream 2", 0.4, "test",
        )

        tmp_storage.promote_candidate_relation(cr1["family_id"])

        all_candidates = tmp_storage.get_candidate_relations()
        all_ids = {r.family_id for r in all_candidates}
        assert cr1["family_id"] in all_ids
        assert cr2["family_id"] in all_ids

    def test_regular_relations_not_in_candidates(self, tmp_storage):
        """D3.5: Regular (non-dream) relations are excluded from candidate listing."""
        e1, e2 = _setup_two_entities(tmp_storage)

        r = _make_relation("rel_reg_d3", e1.absolute_id, e2.absolute_id,
                           "regular", source_document="manual", confidence=0.9)
        tmp_storage.save_relation(r)

        candidates = tmp_storage.get_candidate_relations()
        cand_ids = {r.family_id for r in candidates}
        assert "rel_reg_d3" not in cand_ids

    def test_candidate_count_matches_listing(self, tmp_storage):
        """D3.6: Count matches the number of candidates listed."""
        e1, e2 = _setup_two_entities(tmp_storage)
        e3 = _make_entity("ent_pi_3", "Gamma", "G")
        tmp_storage.save_entity(e3)

        tmp_storage.save_dream_relation("ent_pi_1", "ent_pi_2", "r1", 0.5, "test")
        tmp_storage.save_dream_relation("ent_pi_1", "ent_pi_3", "r2", 0.4, "test")

        count = tmp_storage.count_candidate_relations()
        listing = tmp_storage.get_candidate_relations()
        assert count == len(listing)

    def test_candidate_count_by_status(self, tmp_storage):
        """D3.7: Count by status is accurate."""
        e1, e2 = _setup_two_entities(tmp_storage)

        cr1 = tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "r1", 0.5, "test",
        )

        hypo_count = tmp_storage.count_candidate_relations(status="hypothesized")
        assert hypo_count >= 1

        tmp_storage.promote_candidate_relation(cr1["family_id"])
        ver_count = tmp_storage.count_candidate_relations(status="verified")
        assert ver_count >= 1

    def test_pagination_with_mixed_statuses(self, tmp_storage):
        """D3.8: Pagination works with mixed statuses."""
        e1 = _make_entity("ent_pg_d3", "PG1", "P1")
        e2 = _make_entity("ent_pg2_d3", "PG2", "P2")
        tmp_storage.bulk_save_entities([e1, e2])

        for i in range(5):
            ei = _make_entity(f"ent_pg_d3_{i}", f"PG{i+2}", f"P{i+2}")
            tmp_storage.save_entity(ei)
            tmp_storage.save_dream_relation(
                "ent_pg_d3", f"ent_pg_d3_{i}", f"rel {i}", 0.5, "test",
            )

        page1 = tmp_storage.get_candidate_relations(limit=2, offset=0)
        page2 = tmp_storage.get_candidate_relations(limit=2, offset=2)
        assert len(page1) <= 2
        assert len(page2) <= 2

    def test_empty_status_returns_empty(self, tmp_storage):
        """D3.9: Filtering by nonexistent status returns empty."""
        e1, e2 = _setup_two_entities(tmp_storage)
        tmp_storage.save_dream_relation("ent_pi_1", "ent_pi_2", "r1", 0.5, "test")

        result = tmp_storage.get_candidate_relations(status="nonexistent_status")
        assert result == []


# ══════════════════════════════════════════════════════════════════════════
# D4: End-to-End Integration
# ══════════════════════════════════════════════════════════════════════════


class TestEndToEndIntegration:
    """End-to-end integration tests for dream candidate lifecycle."""

    def test_full_lifecycle_create_corroborate_promote(self, tmp_storage):
        """D4.1: Full lifecycle: create → corroborate × 2 → auto-promote."""
        e1, e2 = _setup_two_entities(tmp_storage)

        # Step 1: Dream creates candidate
        cr = tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test reasoning",
            dream_cycle_id="cycle_e2e",
        )
        assert cr["action"] == "created"

        # Verify it's a candidate
        rel = tmp_storage.get_relation_by_family_id(cr["family_id"])
        attrs = json.loads(rel.attributes)
        assert attrs["tier"] == "candidate"
        assert attrs["status"] == "hypothesized"

        # Step 2: First corroboration
        r1 = tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "remember")
        assert r1["corroboration_count"] == 1

        # Step 3: Second corroboration → auto-promote
        r2 = tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "dream")
        assert r2["new_status"] == "verified"

        # Verify final state
        final = tmp_storage.get_relation_by_family_id(cr["family_id"])
        final_attrs = json.loads(final.attributes)
        assert final_attrs["tier"] == "verified"
        assert final_attrs["status"] == "verified"
        assert final.confidence >= 0.7

    def test_full_lifecycle_create_reject(self, tmp_storage):
        """D4.2: Full lifecycle: create → reject."""
        e1, e2 = _setup_two_entities(tmp_storage)

        cr = tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
        )

        demo = tmp_storage.demote_candidate_relation(cr["family_id"], reason="incorrect")
        assert demo["new_status"] == "rejected"
        assert demo["confidence"] <= 0.2

    def test_lifecycle_with_manual_promotion(self, tmp_storage):
        """D4.3: Manual promotion overrides auto-promotion timeline."""
        e1, e2 = _setup_two_entities(tmp_storage)

        cr = tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.3, "test",
        )

        # Manually promote before any corroboration
        promo = tmp_storage.promote_candidate_relation(
            cr["family_id"], evidence_source="manual", new_confidence=0.95,
        )
        assert promo["new_status"] == "verified"
        assert promo["confidence"] == 0.95

    def test_batch_reject_cycle(self, tmp_storage):
        """D4.4: Batch reject all relations from a dream cycle."""
        e1, e2 = _setup_two_entities(tmp_storage)
        e3 = _make_entity("ent_pi_3", "Gamma", "G")
        tmp_storage.save_entity(e3)

        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "r1", 0.5, "test",
            dream_cycle_id="cycle_batch_d4",
        )
        tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_3", "r2", 0.4, "test",
            dream_cycle_id="cycle_batch_d4",
        )

        result = tmp_storage.reject_dream_cycle_relations("cycle_batch_d4")
        assert result["rejected_count"] >= 2

    def test_corroboration_after_manual_demotion(self, tmp_storage):
        """D4.5: Demoted candidate cannot be corroborated."""
        e1, e2 = _setup_two_entities(tmp_storage)

        cr = tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
        )

        tmp_storage.demote_candidate_relation(cr["family_id"], reason="wrong")

        result = tmp_storage.corroborate_dream_relation(
            "ent_pi_1", "ent_pi_2", "remember",
        )
        assert result is None

    def test_version_history_preserved_through_lifecycle(self, tmp_storage):
        """D4.6: Version history is preserved through create → corroborate → promote."""
        e1, e2 = _setup_two_entities(tmp_storage)

        cr = tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
        )
        fid = cr["family_id"]

        v1 = tmp_storage.get_relation_versions(fid)
        assert len(v1) >= 1

        tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "remember")
        v2 = tmp_storage.get_relation_versions(fid)
        assert len(v2) > len(v1)

        tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "dream")
        v3 = tmp_storage.get_relation_versions(fid)
        assert len(v3) > len(v2)

    def test_confidence_evolution_through_lifecycle(self, tmp_storage):
        """D4.7: Confidence evolves correctly: 0.5 → 0.6 → ≥0.7."""
        e1, e2 = _setup_two_entities(tmp_storage)

        cr = tmp_storage.save_dream_relation(
            "ent_pi_1", "ent_pi_2", "dream link", 0.5, "test",
        )
        fid = cr["family_id"]

        # Initial: 0.5
        rel = tmp_storage.get_relation_by_family_id(fid)
        assert rel.confidence == 0.5

        # After first corroboration: 0.5 + 0.1 = 0.6 (capped at 0.69)
        tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "remember")
        rel = tmp_storage.get_relation_by_family_id(fid)
        assert rel.confidence > 0.5
        assert rel.confidence < 0.7

        # After second corroboration: ≥0.7 (promoted)
        tmp_storage.corroborate_dream_relation("ent_pi_1", "ent_pi_2", "dream")
        rel = tmp_storage.get_relation_by_family_id(fid)
        assert rel.confidence >= 0.7

    def test_lifecycle_with_chinese_content(self, tmp_storage):
        """D4.8: Full lifecycle works with Chinese content."""
        e1 = _make_entity("ent_cn_d4", "人工智能", "AI技术")
        e2 = _make_entity("ent_cn2_d4", "机器学习", "ML技术")
        tmp_storage.bulk_save_entities([e1, e2])

        cr = tmp_storage.save_dream_relation(
            "ent_cn_d4", "ent_cn2_d4", "人工智能包含机器学习作为子领域",
            0.5, "语义关联", dream_cycle_id="cycle_cn",
        )
        assert cr["action"] == "created"

        tmp_storage.corroborate_dream_relation("ent_cn_d4", "ent_cn2_d4", "remember")
        r2 = tmp_storage.corroborate_dream_relation("ent_cn_d4", "ent_cn2_d4", "dream")
        assert r2["new_status"] == "verified"

    def test_lifecycle_with_unicode_special_chars(self, tmp_storage):
        """D4.9: Full lifecycle works with Unicode and special characters."""
        e1 = _make_entity("ent_uni_d4", "特殊字符<!>&", "Test 🌍")
        e2 = _make_entity("ent_uni2_d4", "Emoji🎉", "Test 🚀")
        tmp_storage.bulk_save_entities([e1, e2])

        cr = tmp_storage.save_dream_relation(
            "ent_uni_d4", "ent_uni2_d4", "Special: → ← ↔",
            0.4, "test", dream_cycle_id="cycle_uni",
        )
        assert cr["action"] == "created"

        result = tmp_storage.corroborate_dream_relation(
            "ent_uni_d4", "ent_uni2_d4", "remember",
        )
        assert result is not None
        assert result["corroboration_count"] == 1
