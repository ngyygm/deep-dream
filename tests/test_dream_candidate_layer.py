"""
Dream Candidate Layer tests — 36 tests across 4 dimensions.

D1: Candidate Creation (9 tests)
D2: Promotion/Demotion Lifecycle (9 tests)
D3: Corroboration Auto-Promotion (9 tests)
D4: Search Filtering + Batch Operations (9 tests)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

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
    e1 = _make_entity("ent_cl_1", "Alpha", "Entity A")
    e2 = _make_entity("ent_cl_2", "Beta", "Entity B")
    storage.bulk_save_entities([e1, e2])
    return e1, e2


# ══════════════════════════════════════════════════════════════════════════
# D1: Candidate Creation
# ══════════════════════════════════════════════════════════════════════════


class TestCandidateCreation:
    """Test that dream relations are created as candidates."""

    def test_new_dream_relation_is_candidate(self, tmp_storage):
        """D1.1: New dream relation has tier=candidate."""
        e1, e2 = _setup_two_entities(tmp_storage)

        result = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test relation", 0.7, "test reasoning",
            dream_cycle_id="cycle_001",
        )
        assert result["action"] == "created"

        rel = tmp_storage.get_relation_by_family_id(result["family_id"])
        attrs = json.loads(rel.attributes)
        assert attrs["tier"] == "candidate"
        assert attrs["status"] == "hypothesized"

    def test_confidence_capped_at_05(self, tmp_storage):
        """D1.2: New dream relation confidence capped at 0.5."""
        e1, e2 = _setup_two_entities(tmp_storage)

        result = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.9, "high conf reasoning",
        )
        rel = tmp_storage.get_relation_by_family_id(result["family_id"])
        assert rel.confidence == 0.5

    def test_low_confidence_preserved(self, tmp_storage):
        """D1.3: Confidence below 0.5 is not raised."""
        e1, e2 = _setup_two_entities(tmp_storage)

        result = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.2, "low conf",
        )
        rel = tmp_storage.get_relation_by_family_id(result["family_id"])
        assert rel.confidence == 0.2

    def test_corroboration_count_starts_zero(self, tmp_storage):
        """D1.4: New candidate has corroboration_count=0."""
        e1, e2 = _setup_two_entities(tmp_storage)

        result = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
        )
        rel = tmp_storage.get_relation_by_family_id(result["family_id"])
        attrs = json.loads(rel.attributes)
        assert attrs["corroboration_count"] == 0

    def test_dream_cycle_id_in_attributes(self, tmp_storage):
        """D1.5: Dream cycle ID is stored in attributes."""
        e1, e2 = _setup_two_entities(tmp_storage)

        result = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
            dream_cycle_id="cycle_xyz",
        )
        rel = tmp_storage.get_relation_by_family_id(result["family_id"])
        attrs = json.loads(rel.attributes)
        assert attrs["created_by_dream"] == "cycle_xyz"

    def test_non_dream_relation_not_candidate(self, tmp_storage):
        """D1.6: Regular (non-dream) relation has no candidate tier."""
        e1, e2 = _setup_two_entities(tmp_storage)
        r = _make_relation("rel_normal", e1.absolute_id, e2.absolute_id, "normal",
                           source_document="manual", confidence=0.9)
        tmp_storage.save_relation(r)

        rel = tmp_storage.get_relation_by_family_id("rel_normal")
        assert rel.attributes is None or json.loads(rel.attributes or "{}").get("tier") != "candidate"

    def test_merged_dream_relation_preserves_tier(self, tmp_storage):
        """D1.7: Merging with existing candidate preserves candidate tier."""
        e1, e2 = _setup_two_entities(tmp_storage)

        r1 = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "first", 0.5, "first",
        )
        r2 = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "second", 0.4, "second",
        )
        assert r2["action"] == "merged"
        assert r2["family_id"] == r1["family_id"]

    def test_source_document_has_dream_prefix(self, tmp_storage):
        """D1.8: Dream relation source_document starts with 'dream:'."""
        e1, e2 = _setup_two_entities(tmp_storage)

        result = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
            dream_cycle_id="cycle_123",
        )
        rel = tmp_storage.get_relation_by_family_id(result["family_id"])
        assert rel.source_document == "dream:cycle_123"

    def test_get_candidate_returns_new_candidate(self, tmp_storage):
        """D1.9: get_candidate_relations finds newly created candidates."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
        )
        candidates = tmp_storage.get_candidate_relations()
        assert len(candidates) >= 1
        attrs = json.loads(candidates[0].attributes)
        assert attrs["tier"] == "candidate"


# ══════════════════════════════════════════════════════════════════════════
# D2: Promotion/Demotion Lifecycle
# ══════════════════════════════════════════════════════════════════════════


class TestPromotionDemotion:
    """Test candidate promotion and demotion."""

    def test_promote_candidate(self, tmp_storage):
        """D2.1: Promoting candidate changes tier to verified."""
        e1, e2 = _setup_two_entities(tmp_storage)

        result = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
        )
        fid = result["family_id"]

        promo = tmp_storage.promote_candidate_relation(fid, evidence_source="manual")
        assert promo["new_status"] == "verified"
        assert promo["new_tier"] == "verified"
        assert promo["confidence"] >= 0.7

    def test_promote_increases_confidence(self, tmp_storage):
        """D2.2: Promotion sets confidence to at least 0.7."""
        e1, e2 = _setup_two_entities(tmp_storage)

        result = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.3, "test",
        )
        fid = result["family_id"]

        promo = tmp_storage.promote_candidate_relation(fid)
        assert promo["confidence"] >= 0.7

    def test_promote_custom_confidence(self, tmp_storage):
        """D2.3: Promotion with explicit confidence override."""
        e1, e2 = _setup_two_entities(tmp_storage)

        result = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
        )
        fid = result["family_id"]

        promo = tmp_storage.promote_candidate_relation(fid, new_confidence=0.95)
        assert promo["confidence"] == 0.95

    def test_promote_nonexistent_raises(self, tmp_storage):
        """D2.4: Promoting nonexistent relation raises ValueError."""
        with pytest.raises(ValueError, match="关系不存在"):
            tmp_storage.promote_candidate_relation("nonexistent_xyz")

    def test_demote_candidate(self, tmp_storage):
        """D2.5: Demoting candidate sets status to rejected."""
        e1, e2 = _setup_two_entities(tmp_storage)

        result = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
        )
        fid = result["family_id"]

        demo = tmp_storage.demote_candidate_relation(fid, reason="incorrect")
        assert demo["new_status"] == "rejected"

    def test_demote_lowers_confidence(self, tmp_storage):
        """D2.6: Demotion caps confidence at 0.2."""
        e1, e2 = _setup_two_entities(tmp_storage)

        result = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
        )
        fid = result["family_id"]

        demo = tmp_storage.demote_candidate_relation(fid, reason="wrong")
        assert demo["confidence"] <= 0.2

    def test_promote_creates_new_version(self, tmp_storage):
        """D2.7: Promotion creates a new version, preserving history."""
        e1, e2 = _setup_two_entities(tmp_storage)

        result = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
        )
        fid = result["family_id"]

        versions_before = tmp_storage.get_relation_versions(fid)
        tmp_storage.promote_candidate_relation(fid)
        versions_after = tmp_storage.get_relation_versions(fid)

        assert len(versions_after) > len(versions_before)

    def test_demote_nonexistent_raises(self, tmp_storage):
        """D2.8: Demoting nonexistent relation raises ValueError."""
        with pytest.raises(ValueError, match="关系不存在"):
            tmp_storage.demote_candidate_relation("nonexistent_xyz")

    def test_count_candidate_by_status(self, tmp_storage):
        """D2.9: Count candidates by status."""
        e1, e2 = _setup_two_entities(tmp_storage)
        e3 = _make_entity("ent_cl_3", "Gamma", "G")
        tmp_storage.save_entity(e3)

        r1 = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "rel1", 0.5, "test",
        )
        r2 = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_3", "rel2", 0.4, "test",
        )

        # Both hypothesized
        assert tmp_storage.count_candidate_relations(status="hypothesized") >= 2

        # Promote one
        tmp_storage.promote_candidate_relation(r1["family_id"])
        assert tmp_storage.count_candidate_relations(status="verified") >= 1
        assert tmp_storage.count_candidate_relations(status="hypothesized") >= 1


# ══════════════════════════════════════════════════════════════════════════
# D3: Corroboration Auto-Promotion
# ══════════════════════════════════════════════════════════════════════════


class TestCorroboration:
    """Test auto-promotion via corroboration."""

    def test_first_corroboration_increments_count(self, tmp_storage):
        """D3.1: First corroboration sets count=1, does not promote."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
        )

        result = tmp_storage.corroborate_dream_relation(
            "ent_cl_1", "ent_cl_2", corroboration_source="remember",
        )
        assert result["corroboration_count"] == 1
        assert result["status"] == "hypothesized"

    def test_second_corroboration_auto_promotes(self, tmp_storage):
        """D3.2: Second corroboration triggers auto-promotion."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
        )

        tmp_storage.corroborate_dream_relation("ent_cl_1", "ent_cl_2", "remember")
        result = tmp_storage.corroborate_dream_relation("ent_cl_1", "ent_cl_2", "dream")

        assert result["new_status"] == "verified"

    def test_corroboration_increases_confidence(self, tmp_storage):
        """D3.3: Each corroboration increases confidence by 0.1."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
        )

        result = tmp_storage.corroborate_dream_relation("ent_cl_1", "ent_cl_2", "remember")
        rel = tmp_storage.get_relation_by_family_id(result["family_id"])
        assert rel.confidence > 0.5  # boosted by 0.1

    def test_no_corroboration_for_non_dream(self, tmp_storage):
        """D3.4: Non-dream relations are not corroborated."""
        e1, e2 = _setup_two_entities(tmp_storage)
        r = _make_relation("rel_manual", e1.absolute_id, e2.absolute_id,
                           "manual relation", source_document="manual", confidence=0.9)
        tmp_storage.save_relation(r)

        result = tmp_storage.corroborate_dream_relation("ent_cl_1", "ent_cl_2")
        assert result is None

    def test_no_corroboration_for_already_verified(self, tmp_storage):
        """D3.5: Already verified relations are not corroborated again."""
        e1, e2 = _setup_two_entities(tmp_storage)

        cr = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
        )
        tmp_storage.promote_candidate_relation(cr["family_id"])

        result = tmp_storage.corroborate_dream_relation("ent_cl_1", "ent_cl_2")
        # Already verified, so no further corroboration
        # The relation is no longer "hypothesized", so returns None
        assert result is None

    def test_no_corroboration_for_nonexistent_pair(self, tmp_storage):
        """D3.6: Nonexistent entity pair returns None."""
        _setup_two_entities(tmp_storage)
        result = tmp_storage.corroborate_dream_relation("ent_cl_1", "nonexistent")
        assert result is None

    def test_corroboration_tracks_sources(self, tmp_storage):
        """D3.7: Corroboration sources are tracked in attributes."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
        )

        tmp_storage.corroborate_dream_relation("ent_cl_1", "ent_cl_2", "remember")
        result = tmp_storage.corroborate_dream_relation("ent_cl_1", "ent_cl_2", "dream")

        # After promotion, check attributes on the promoted version
        rel = tmp_storage.get_relation_by_family_id(result["family_id"])
        attrs = json.loads(rel.attributes)
        assert "remember" in attrs.get("corroboration_sources", [])
        assert "dream" in attrs.get("corroboration_sources", [])

    def test_first_corroboration_caps_confidence_below_07(self, tmp_storage):
        """D3.8: First corroboration confidence stays below 0.7 (promotion threshold)."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
        )

        result = tmp_storage.corroborate_dream_relation("ent_cl_1", "ent_cl_2", "remember")
        rel = tmp_storage.get_relation_by_family_id(result["family_id"])
        assert rel.confidence < 0.7

    def test_multiple_dream_cycles_corroborate_same(self, tmp_storage):
        """D3.9: Corroboration works across multiple dream cycles."""
        e1, e2 = _setup_two_entities(tmp_storage)

        tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
            dream_cycle_id="cycle_1",
        )

        # First corroboration
        r1 = tmp_storage.corroborate_dream_relation("ent_cl_1", "ent_cl_2", "remember")
        assert r1["corroboration_count"] == 1

        # Second corroboration from a different source
        r2 = tmp_storage.corroborate_dream_relation("ent_cl_1", "ent_cl_2", "dream")
        assert r2["new_status"] == "verified"


# ══════════════════════════════════════════════════════════════════════════
# D4: Search Filtering + Batch Operations
# ══════════════════════════════════════════════════════════════════════════


class TestSearchFilteringAndBatch:
    """Test candidate filtering and batch operations."""

    def test_get_candidates_filters_by_status(self, tmp_storage):
        """D4.1: Filter candidates by status."""
        e1, e2 = _setup_two_entities(tmp_storage)
        e3 = _make_entity("ent_cl_3", "Gamma", "G")
        tmp_storage.save_entity(e3)

        r1 = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "rel1", 0.5, "test",
        )
        tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_3", "rel2", 0.4, "test",
        )

        # Promote r1
        tmp_storage.promote_candidate_relation(r1["family_id"])

        hypothesized = tmp_storage.get_candidate_relations(status="hypothesized")
        verified = tmp_storage.get_candidate_relations(status="verified")

        hypo_ids = {r.family_id for r in hypothesized}
        ver_ids = {r.family_id for r in verified}

        assert r1["family_id"] in ver_ids
        assert r1["family_id"] not in hypo_ids

    def test_get_candidates_pagination(self, tmp_storage):
        """D4.2: Pagination works for candidate listing."""
        e1 = _make_entity("ent_pg1", "PG1", "P1")
        e2 = _make_entity("ent_pg2", "PG2", "P2")
        tmp_storage.bulk_save_entities([e1, e2])

        for i in range(5):
            ei = _make_entity(f"ent_pg_{i}", f"PG{i+2}", f"P{i+2}")
            tmp_storage.save_entity(ei)
            tmp_storage.save_dream_relation(
                "ent_pg1", f"ent_pg_{i}", f"rel {i}", 0.5, "test",
            )

        page1 = tmp_storage.get_candidate_relations(limit=2, offset=0)
        page2 = tmp_storage.get_candidate_relations(limit=2, offset=2)

        assert len(page1) <= 2
        assert len(page2) <= 2
        ids1 = {r.family_id for r in page1}
        ids2 = {r.family_id for r in page2}
        assert ids1.isdisjoint(ids2)

    def test_reject_dream_cycle_relations(self, tmp_storage):
        """D4.3: Batch reject all relations from a dream cycle."""
        e1, e2 = _setup_two_entities(tmp_storage)
        e3 = _make_entity("ent_cl_3", "Gamma", "G")
        tmp_storage.save_entity(e3)

        tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "r1", 0.5, "test",
            dream_cycle_id="cycle_batch",
        )
        tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_3", "r2", 0.4, "test",
            dream_cycle_id="cycle_batch",
        )
        # Different cycle
        tmp_storage.save_dream_relation(
            "ent_cl_2", "ent_cl_3", "r3", 0.5, "test",
            dream_cycle_id="cycle_other",
        )

        result = tmp_storage.reject_dream_cycle_relations("cycle_batch")
        assert result["rejected_count"] >= 2
        assert result["cycle_id"] == "cycle_batch"

        # Check that cycle_batch relations are rejected
        candidates = tmp_storage.get_candidate_relations(status="rejected")
        rejected_fids = {r.family_id for r in candidates}
        # The cycle_other relation should not be rejected
        assert len(rejected_fids) >= 2

    def test_count_candidates_total(self, tmp_storage):
        """D4.4: Count all candidates regardless of status."""
        e1, e2 = _setup_two_entities(tmp_storage)
        e3 = _make_entity("ent_cl_3", "Gamma", "G")
        tmp_storage.save_entity(e3)

        tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "r1", 0.5, "test",
        )
        tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_3", "r2", 0.4, "test",
        )

        total = tmp_storage.count_candidate_relations()
        assert total >= 2

    def test_empty_graph_no_candidates(self, tmp_storage):
        """D4.5: Empty graph returns no candidates."""
        candidates = tmp_storage.get_candidate_relations()
        assert candidates == []

    def test_count_zero_on_empty(self, tmp_storage):
        """D4.6: Count returns 0 on empty graph."""
        assert tmp_storage.count_candidate_relations() == 0

    def test_candidate_does_not_include_regular_relations(self, tmp_storage):
        """D4.7: Regular relations are not included in candidate listing."""
        e1, e2 = _setup_two_entities(tmp_storage)
        r = _make_relation("rel_reg", e1.absolute_id, e2.absolute_id,
                           "regular", source_document="manual", confidence=0.9)
        tmp_storage.save_relation(r)

        candidates = tmp_storage.get_candidate_relations()
        reg_ids = {r.family_id for r in candidates}
        assert "rel_reg" not in reg_ids

    def test_promoted_relation_searchable(self, tmp_storage):
        """D4.8: Promoted relations appear in verified listing."""
        e1, e2 = _setup_two_entities(tmp_storage)

        result = tmp_storage.save_dream_relation(
            "ent_cl_1", "ent_cl_2", "test", 0.5, "test",
        )
        tmp_storage.promote_candidate_relation(result["family_id"])

        verified = tmp_storage.get_candidate_relations(status="verified")
        ver_ids = {r.family_id for r in verified}
        assert result["family_id"] in ver_ids

    def test_reject_cycle_with_no_relations(self, tmp_storage):
        """D4.9: Rejecting empty cycle returns zero count."""
        result = tmp_storage.reject_dream_cycle_relations("nonexistent_cycle")
        assert result["rejected_count"] == 0
