"""
Confidence Engine — 30+ test cases across 4 data dimensions:

  Dimension 1: Corroboration (evidence-driven confidence increase)
  Dimension 2: Contradiction (evidence-driven confidence decrease)
  Dimension 3: Manual override (direct confidence set via API)
  Dimension 4: Dream source weighting + edge cases
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processor.storage.manager import StorageManager
from processor.models import Entity, Relation
from processor.pipeline.entity import EntityProcessor
from processor.pipeline.relation import RelationProcessor


def _now() -> datetime:
    return datetime.now()


def _make_entity(
    family_id="fam_1",
    name="TestEntity",
    content="A test entity.",
    confidence=0.7,
    processed_time=None,
) -> Entity:
    return Entity(
        absolute_id=f"{family_id}_abs_{datetime.now().strftime('%H%M%S_%f')}",
        family_id=family_id,
        name=name,
        content=content,
        event_time=_now(),
        processed_time=processed_time or _now(),
        episode_id="ep_test",
        source_document="test.md",
        confidence=confidence,
        content_format="markdown",
        summary=content[:80],
    )


def _make_relation(
    family_id="r_fam_1",
    content="test relation",
    confidence=0.7,
    processed_time=None,
) -> Relation:
    return Relation(
        absolute_id=f"{family_id}_abs_{datetime.now().strftime('%H%M%S_%f')}",
        family_id=family_id,
        entity1_absolute_id="ea_1",
        entity2_absolute_id="ea_2",
        content=content,
        event_time=_now(),
        processed_time=processed_time or _now(),
        episode_id="ep_test",
        source_document="test.md",
        confidence=confidence,
        content_format="markdown",
        summary=content[:80],
    )


@pytest.fixture
def storage(tmp_path):
    s = StorageManager(storage_path=str(tmp_path / "test_conf.db"))
    yield s
    s.close()


# ═══════════════════════════════════════════════════════════════════
# Dimension 1: Corroboration — confidence increase
# ═══════════════════════════════════════════════════════════════════

class TestCorroborationIncrease:
    """Evidence-driven confidence increase via adjust_confidence_on_corroboration."""

    def test_entity_single_corroboration(self, storage):
        """Single corroboration should increase entity confidence by 0.05."""
        e = _make_entity(family_id="fam_c1", confidence=0.7)
        storage.save_entity(e)
        storage.adjust_confidence_on_corroboration("fam_c1", source_type="entity")
        updated = storage.get_entity_by_family_id("fam_c1")
        assert abs(updated.confidence - 0.75) < 0.01

    def test_entity_multiple_corroborations(self, storage):
        """Multiple corroboration calls should accumulate."""
        e = _make_entity(family_id="fam_c2", confidence=0.5)
        storage.save_entity(e)
        for _ in range(3):
            storage.adjust_confidence_on_corroboration("fam_c2", source_type="entity")
        updated = storage.get_entity_by_family_id("fam_c2")
        assert abs(updated.confidence - 0.65) < 0.01  # 0.5 + 3*0.05

    def test_entity_confidence_ceiling_at_1(self, storage):
        """Confidence should cap at 1.0 even with many corroboration calls."""
        e = _make_entity(family_id="fam_c3", confidence=0.95)
        storage.save_entity(e)
        storage.adjust_confidence_on_corroboration("fam_c3", source_type="entity")
        storage.adjust_confidence_on_corroboration("fam_c3", source_type="entity")
        updated = storage.get_entity_by_family_id("fam_c3")
        assert updated.confidence <= 1.0
        assert abs(updated.confidence - 1.0) < 0.01

    def test_relation_single_corroboration(self, storage):
        """Single corroboration should increase relation confidence by 0.05."""
        r = _make_relation(family_id="r_c4", confidence=0.6)
        storage.save_relation(r)
        storage.adjust_confidence_on_corroboration("r_c4", source_type="relation")
        updated = storage.get_relation_by_family_id("r_c4")
        assert abs(updated.confidence - 0.65) < 0.01

    def test_relation_multiple_corroborations(self, storage):
        """Multiple relation corroboration calls should accumulate."""
        r = _make_relation(family_id="r_c5", confidence=0.3)
        storage.save_relation(r)
        for _ in range(5):
            storage.adjust_confidence_on_corroboration("r_c5", source_type="relation")
        updated = storage.get_relation_by_family_id("r_c5")
        assert abs(updated.confidence - 0.55) < 0.01

    def test_entity_starting_from_zero(self, storage):
        """Entity with confidence 0 should still increase."""
        e = _make_entity(family_id="fam_c6", confidence=0.0)
        storage.save_entity(e)
        storage.adjust_confidence_on_corroboration("fam_c6", source_type="entity")
        updated = storage.get_entity_by_family_id("fam_c6")
        assert abs(updated.confidence - 0.05) < 0.01

    def test_corroboration_on_nonexistent_entity(self, storage):
        """Corroboration on nonexistent family_id should not raise."""
        storage.adjust_confidence_on_corroboration("nonexistent", source_type="entity")
        # Should silently pass without error

    def test_corroboration_on_nonexistent_relation(self, storage):
        """Corroboration on nonexistent relation family_id should not raise."""
        storage.adjust_confidence_on_corroboration("nonexistent_r", source_type="relation")


# ═══════════════════════════════════════════════════════════════════
# Dimension 2: Contradiction — confidence decrease
# ═══════════════════════════════════════════════════════════════════

class TestContradictionDecrease:
    """Evidence-driven confidence decrease via adjust_confidence_on_contradiction."""

    def test_entity_single_contradiction(self, storage):
        """Single contradiction should decrease entity confidence by 0.1."""
        e = _make_entity(family_id="fam_d1", confidence=0.7)
        storage.save_entity(e)
        storage.adjust_confidence_on_contradiction("fam_d1", source_type="entity")
        updated = storage.get_entity_by_family_id("fam_d1")
        assert abs(updated.confidence - 0.6) < 0.01

    def test_entity_multiple_contradictions(self, storage):
        """Multiple contradiction calls should accumulate."""
        e = _make_entity(family_id="fam_d2", confidence=0.9)
        storage.save_entity(e)
        for _ in range(4):
            storage.adjust_confidence_on_contradiction("fam_d2", source_type="entity")
        updated = storage.get_entity_by_family_id("fam_d2")
        assert abs(updated.confidence - 0.5) < 0.01  # 0.9 - 4*0.1

    def test_entity_confidence_floor_at_0(self, storage):
        """Confidence should floor at 0.0 even with many contradictions."""
        e = _make_entity(family_id="fam_d3", confidence=0.15)
        storage.save_entity(e)
        for _ in range(3):
            storage.adjust_confidence_on_contradiction("fam_d3", source_type="entity")
        updated = storage.get_entity_by_family_id("fam_d3")
        assert updated.confidence >= 0.0
        assert abs(updated.confidence - 0.0) < 0.01

    def test_relation_single_contradiction(self, storage):
        """Single contradiction should decrease relation confidence by 0.1."""
        r = _make_relation(family_id="r_d4", confidence=0.8)
        storage.save_relation(r)
        storage.adjust_confidence_on_contradiction("r_d4", source_type="relation")
        updated = storage.get_relation_by_family_id("r_d4")
        assert abs(updated.confidence - 0.7) < 0.01

    def test_relation_multiple_contradictions(self, storage):
        """Multiple contradiction calls should accumulate for relations."""
        r = _make_relation(family_id="r_d5", confidence=0.6)
        storage.save_relation(r)
        for _ in range(3):
            storage.adjust_confidence_on_contradiction("r_d5", source_type="relation")
        updated = storage.get_relation_by_family_id("r_d5")
        assert abs(updated.confidence - 0.3) < 0.01

    def test_contradiction_on_nonexistent_entity(self, storage):
        """Contradiction on nonexistent family_id should not raise."""
        storage.adjust_confidence_on_contradiction("ghost", source_type="entity")

    def test_contradiction_on_nonexistent_relation(self, storage):
        """Contradiction on nonexistent relation should not raise."""
        storage.adjust_confidence_on_contradiction("ghost_r", source_type="relation")

    def test_mixed_corroboration_and_contradiction(self, storage):
        """Corroboration then contradiction should net correctly."""
        e = _make_entity(family_id="fam_mix", confidence=0.5)
        storage.save_entity(e)
        storage.adjust_confidence_on_corroboration("fam_mix", source_type="entity")  # 0.55
        storage.adjust_confidence_on_corroboration("fam_mix", source_type="entity")  # 0.60
        storage.adjust_confidence_on_contradiction("fam_mix", source_type="entity")  # 0.50
        updated = storage.get_entity_by_family_id("fam_mix")
        assert abs(updated.confidence - 0.50) < 0.01


# ═══════════════════════════════════════════════════════════════════
# Dimension 3: Manual override (direct confidence set)
# ═══════════════════════════════════════════════════════════════════

class TestManualOverride:
    """Direct confidence setting via update_entity/relation_confidence."""

    def test_entity_set_confidence_high(self, storage):
        """Set entity confidence to 0.95."""
        e = _make_entity(family_id="fam_m1", confidence=0.5)
        storage.save_entity(e)
        storage.update_entity_confidence("fam_m1", 0.95)
        updated = storage.get_entity_by_family_id("fam_m1")
        assert abs(updated.confidence - 0.95) < 0.01

    def test_entity_set_confidence_zero(self, storage):
        """Set entity confidence to 0.0."""
        e = _make_entity(family_id="fam_m2", confidence=0.8)
        storage.save_entity(e)
        storage.update_entity_confidence("fam_m2", 0.0)
        updated = storage.get_entity_by_family_id("fam_m2")
        assert abs(updated.confidence - 0.0) < 0.01

    def test_entity_confidence_clamped_above_1(self, storage):
        """Values > 1.0 should be clamped to 1.0."""
        e = _make_entity(family_id="fam_m3", confidence=0.5)
        storage.save_entity(e)
        storage.update_entity_confidence("fam_m3", 1.5)
        updated = storage.get_entity_by_family_id("fam_m3")
        assert abs(updated.confidence - 1.0) < 0.01

    def test_entity_confidence_clamped_below_0(self, storage):
        """Values < 0.0 should be clamped to 0.0."""
        e = _make_entity(family_id="fam_m4", confidence=0.5)
        storage.save_entity(e)
        storage.update_entity_confidence("fam_m4", -0.3)
        updated = storage.get_entity_by_family_id("fam_m4")
        assert abs(updated.confidence - 0.0) < 0.01

    def test_relation_set_confidence(self, storage):
        """Set relation confidence to 0.88."""
        r = _make_relation(family_id="r_m5", confidence=0.5)
        storage.save_relation(r)
        storage.update_relation_confidence("r_m5", 0.88)
        updated = storage.get_relation_by_family_id("r_m5")
        assert abs(updated.confidence - 0.88) < 0.01

    def test_relation_confidence_clamped_above_1(self, storage):
        """Relation values > 1.0 should be clamped to 1.0."""
        r = _make_relation(family_id="r_m6", confidence=0.5)
        storage.save_relation(r)
        storage.update_relation_confidence("r_m6", 2.0)
        updated = storage.get_relation_by_family_id("r_m6")
        assert abs(updated.confidence - 1.0) < 0.01

    def test_relation_confidence_clamped_below_0(self, storage):
        """Relation values < 0.0 should be clamped to 0.0."""
        r = _make_relation(family_id="r_m7", confidence=0.5)
        storage.save_relation(r)
        storage.update_relation_confidence("r_m7", -1.0)
        updated = storage.get_relation_by_family_id("r_m7")
        assert abs(updated.confidence - 0.0) < 0.01

    def test_override_then_corroboration(self, storage):
        """After manual override, corroboration should continue from new value."""
        e = _make_entity(family_id="fam_m8", confidence=0.3)
        storage.save_entity(e)
        storage.update_entity_confidence("fam_m8", 0.9)
        storage.adjust_confidence_on_corroboration("fam_m8", source_type="entity")
        updated = storage.get_entity_by_family_id("fam_m8")
        assert abs(updated.confidence - 0.95) < 0.01


# ═══════════════════════════════════════════════════════════════════
# Dimension 4: Dream source weighting + multi-version + pipeline integration
# ═══════════════════════════════════════════════════════════════════

class TestDreamWeightingAndEdgeCases:
    """Dream sources get half weight; multi-version behavior; pipeline integration."""

    def test_dream_source_half_weight_entity(self, storage):
        """Dream source corroboration should add only 0.025 to entity."""
        e = _make_entity(family_id="fam_dw1", confidence=0.5)
        storage.save_entity(e)
        storage.adjust_confidence_on_corroboration("fam_dw1", source_type="entity", is_dream=True)
        updated = storage.get_entity_by_family_id("fam_dw1")
        assert abs(updated.confidence - 0.525) < 0.01

    def test_dream_source_multiple(self, storage):
        """Multiple dream corroboration calls should accumulate at half rate."""
        e = _make_entity(family_id="fam_dw2", confidence=0.4)
        storage.save_entity(e)
        for _ in range(4):
            storage.adjust_confidence_on_corroboration("fam_dw2", source_type="entity", is_dream=True)
        updated = storage.get_entity_by_family_id("fam_dw2")
        assert abs(updated.confidence - 0.5) < 0.01  # 0.4 + 4*0.025

    def test_dream_vs_entity_corroboration_mixed(self, storage):
        """Mixing dream and entity corroboration should use correct deltas."""
        e = _make_entity(family_id="fam_dw3", confidence=0.5)
        storage.save_entity(e)
        storage.adjust_confidence_on_corroboration("fam_dw3", source_type="entity", is_dream=True)   # +0.025
        storage.adjust_confidence_on_corroboration("fam_dw3", source_type="entity", is_dream=False)  # +0.05
        updated = storage.get_entity_by_family_id("fam_dw3")
        assert abs(updated.confidence - 0.575) < 0.01

    def test_multi_version_updates_latest_only(self, storage):
        """Confidence update should only affect the latest version."""
        t1 = _now() - timedelta(minutes=10)
        t2 = _now()
        e1 = _make_entity(family_id="fam_mv1", confidence=0.5, processed_time=t1)
        e1.absolute_id = "abs_mv1_v1"
        storage.save_entity(e1)
        e2 = _make_entity(family_id="fam_mv1", confidence=0.6, processed_time=t2)
        e2.absolute_id = "abs_mv1_v2"
        storage.save_entity(e2)

        storage.update_entity_confidence("fam_mv1", 0.9)
        versions = storage.get_entity_versions("fam_mv1")
        # Latest version should be 0.9
        assert abs(versions[0].confidence - 0.9) < 0.01
        # Older version should be unchanged
        assert abs(versions[1].confidence - 0.5) < 0.01

    def test_entity_version_creation_increases_confidence(self, storage):
        """Creating a new entity version via pipeline should increase confidence."""
        e = _make_entity(family_id="fam_pipe1", confidence=0.7)
        storage.save_entity(e)

        llm = MagicMock()
        llm.effective_entity_snippet_length.return_value = 50
        llm.judge_content_need_update.return_value = True
        proc = EntityProcessor(storage=storage, llm_client=llm)

        new_entity = proc._create_entity_version(
            family_id="fam_pipe1",
            name="TestEntity",
            content="Updated content for confidence test",
            episode_id="ep_2",
            source_document="new_doc.md",
            base_time=_now(),
            old_content="A test entity.",
            old_content_format="plain",
            skip_if_unchanged=True,
        )
        # The confidence should have been boosted by adjust_confidence_on_corroboration
        assert new_entity is not None
        updated = storage.get_entity_by_family_id("fam_pipe1")
        # 0.7 + 0.05 = 0.75
        assert updated.confidence >= 0.75 - 0.01

    def test_relation_version_creation_increases_confidence(self, storage):
        """Creating a new relation version via pipeline should increase confidence."""
        # Need to seed entities for relation creation
        e1 = _make_entity(family_id="fam_rp1", name="EntityA")
        e2 = _make_entity(family_id="fam_rp2", name="EntityB")
        storage.save_entity(e1)
        storage.save_entity(e2)

        r = _make_relation(family_id="r_pipe1", confidence=0.6)
        storage.save_relation(r)

        llm = MagicMock()
        llm.effective_entity_snippet_length.return_value = 50
        proc = RelationProcessor(storage=storage, llm_client=llm)

        new_rel = proc._create_relation_version(
            family_id="r_pipe1",
            entity1_id="fam_rp1",
            entity2_id="fam_rp2",
            content="Updated relation content for confidence test",
            episode_id="ep_3",
            source_document="new_doc.md",
        )
        assert new_rel is not None
        updated = storage.get_relation_by_family_id("r_pipe1")
        # 0.6 + 0.05 = 0.65
        assert updated.confidence >= 0.65 - 0.01

    def test_confidence_precision_float(self, storage):
        """Multiple adjustments should maintain float precision."""
        e = _make_entity(family_id="fam_prec", confidence=0.7000001)
        storage.save_entity(e)
        for _ in range(7):
            storage.adjust_confidence_on_corroboration("fam_prec", source_type="entity")
        updated = storage.get_entity_by_family_id("fam_prec")
        # 0.7 + 7*0.05 = 1.0 (capped)
        assert updated.confidence <= 1.0
        assert updated.confidence >= 0.99

    def test_confidence_after_many_rapid_adjustments(self, storage):
        """Rapid sequential adjustments should be consistent."""
        e = _make_entity(family_id="fam_rapid", confidence=0.5)
        storage.save_entity(e)
        # 5 corroboration (+0.25) then 2 contradiction (-0.20) = 0.55
        for _ in range(5):
            storage.adjust_confidence_on_corroboration("fam_rapid", source_type="entity")
        for _ in range(2):
            storage.adjust_confidence_on_contradiction("fam_rapid", source_type="entity")
        updated = storage.get_entity_by_family_id("fam_rapid")
        assert abs(updated.confidence - 0.55) < 0.01

    def test_relation_dream_corroboration(self, storage):
        """Dream corroboration on relation should use half delta."""
        r = _make_relation(family_id="r_dream", confidence=0.5)
        storage.save_relation(r)
        storage.adjust_confidence_on_corroboration("r_dream", source_type="relation", is_dream=True)
        updated = storage.get_relation_by_family_id("r_dream")
        assert abs(updated.confidence - 0.525) < 0.01

    def test_confidence_default_on_new_entity(self, storage):
        """Newly created entity should have default confidence 0.7."""
        llm = MagicMock()
        llm.effective_entity_snippet_length.return_value = 50
        proc = EntityProcessor(storage=storage, llm_client=llm)
        entity = proc._build_new_entity("TestDefault", "Content", "ep_new")
        assert abs(entity.confidence - 0.7) < 0.01

    def test_confidence_default_on_new_relation(self, storage):
        """Newly created relation should have default confidence 0.7."""
        e1 = _make_entity(family_id="fam_nd1", name="A")
        e2 = _make_entity(family_id="fam_nd2", name="B")
        storage.save_entity(e1)
        storage.save_entity(e2)

        llm = MagicMock()
        proc = RelationProcessor(storage=storage, llm_client=llm)
        rel = proc._build_new_relation(
            "fam_nd1", "fam_nd2", "A relates B", "ep_new",
            entity1_name="A", entity2_name="B",
        )
        assert rel is not None
        assert abs(rel.confidence - 0.7) < 0.01
