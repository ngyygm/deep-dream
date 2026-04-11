"""
Concept sync on entity/relation updates — 36+ tests across 4 dimensions.

Vision Gap 2 fix: When entity/relation fields are updated (summary, attributes,
confidence, name, content), the changes must propagate to the concepts table.

D1: Entity update methods sync to concepts (10 tests)
D2: Relation update methods sync to concepts (10 tests)
D3: Confidence adjustment methods sync for both entity and relation (10 tests)
D4: Edge cases — versioning, missing concepts, concurrent updates, FTS sync (8 tests)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

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
                 summary: Optional[str] = None, attributes=None,
                 confidence: Optional[float] = None):
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
        source_document="test",
        summary=summary,
        attributes=attributes,
        confidence=confidence,
    )


def _make_relation(family_id: str, e1_abs: str, e2_abs: str, content: str,
                   summary: Optional[str] = None, confidence: Optional[float] = None):
    from processor.models import Relation
    now = datetime.now(timezone.utc)
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
        summary=summary,
        confidence=confidence,
    )


def _get_concept(storage, absolute_id: str) -> Optional[dict]:
    """Get a concept row by absolute_id."""
    conn = storage._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM concepts WHERE id = ?", (absolute_id,))
    row = cursor.fetchone()
    if not row:
        return None
    cols = [desc[0] for desc in cursor.description]
    return dict(zip(cols, row))


def _count_concept_fts(storage, query: str) -> int:
    """Count FTS matches for a query in concept_fts."""
    conn = storage._get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT COUNT(*) FROM concept_fts WHERE concept_fts MATCH ?", (query,)
        )
        return cursor.fetchone()[0]
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════
# D1: Entity update methods sync to concepts
# ══════════════════════════════════════════════════════════════════════════


class TestEntityUpdateConceptSync:
    """D1: Entity field updates propagate to concepts table."""

    def test_update_entity_by_absolute_id_name(self, tmp_storage):
        """D1.1: update_entity_by_absolute_id(name=...) syncs to concepts."""
        e = _make_entity("d1_e1", "Original", "Content")
        tmp_storage.save_entity(e)

        tmp_storage.update_entity_by_absolute_id(e.absolute_id, name="Updated")

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept is not None
        assert concept["name"] == "Updated"

    def test_update_entity_by_absolute_id_content(self, tmp_storage):
        """D1.2: update_entity_by_absolute_id(content=...) syncs to concepts."""
        e = _make_entity("d1_e2", "Test", "Original content")
        tmp_storage.save_entity(e)

        tmp_storage.update_entity_by_absolute_id(e.absolute_id, content="New content")

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept["content"] == "New content"

    def test_update_entity_by_absolute_id_summary(self, tmp_storage):
        """D1.3: update_entity_by_absolute_id(summary=...) syncs to concepts."""
        e = _make_entity("d1_e3", "Test", "Content")
        tmp_storage.save_entity(e)

        tmp_storage.update_entity_by_absolute_id(e.absolute_id, summary="A summary")

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept["summary"] == "A summary"

    def test_update_entity_by_absolute_id_attributes(self, tmp_storage):
        """D1.4: update_entity_by_absolute_id(attributes=...) syncs to concepts."""
        e = _make_entity("d1_e4", "Test", "Content")
        tmp_storage.save_entity(e)

        attrs = json.dumps({"color": "blue"})
        tmp_storage.update_entity_by_absolute_id(e.absolute_id, attributes=attrs)

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept["attributes"] == attrs

    def test_update_entity_by_absolute_id_confidence(self, tmp_storage):
        """D1.5: update_entity_by_absolute_id(confidence=...) syncs to concepts."""
        e = _make_entity("d1_e5", "Test", "Content")
        tmp_storage.save_entity(e)

        tmp_storage.update_entity_by_absolute_id(e.absolute_id, confidence=0.9)

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept["confidence"] == 0.9

    def test_update_entity_summary_method(self, tmp_storage):
        """D1.6: update_entity_summary syncs to concepts."""
        e = _make_entity("d1_e6", "Test", "Content")
        tmp_storage.save_entity(e)

        tmp_storage.update_entity_summary("d1_e6", "Summary via method")

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept["summary"] == "Summary via method"

    def test_update_entity_attributes_method(self, tmp_storage):
        """D1.7: update_entity_attributes syncs to concepts."""
        e = _make_entity("d1_e7", "Test", "Content")
        tmp_storage.save_entity(e)

        tmp_storage.update_entity_attributes("d1_e7", '{"key": "val"}')

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept["attributes"] == '{"key": "val"}'

    def test_update_entity_confidence_method(self, tmp_storage):
        """D1.8: update_entity_confidence syncs to concepts."""
        e = _make_entity("d1_e8", "Test", "Content")
        tmp_storage.save_entity(e)

        tmp_storage.update_entity_confidence("d1_e8", 0.75)

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept["confidence"] == 0.75

    def test_update_multiple_fields_at_once(self, tmp_storage):
        """D1.9: Multiple fields updated in one call all sync to concepts."""
        e = _make_entity("d1_e9", "Old", "Old content")
        tmp_storage.save_entity(e)

        tmp_storage.update_entity_by_absolute_id(
            e.absolute_id, name="New", content="New content", summary="S", confidence=0.8
        )

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept["name"] == "New"
        assert concept["content"] == "New content"
        assert concept["summary"] == "S"
        assert concept["confidence"] == 0.8

    def test_update_entity_fts_sync(self, tmp_storage):
        """D1.10: Name/content update also updates concept_fts."""
        e = _make_entity("d1_e10", "UniqueAlpha", "BetaGamma content")
        tmp_storage.save_entity(e)

        # After update, new name should be findable in FTS
        tmp_storage.update_entity_by_absolute_id(e.absolute_id, name="OmegaDelta")

        count = _count_concept_fts(tmp_storage, "OmegaDelta")
        assert count >= 1


# ══════════════════════════════════════════════════════════════════════════
# D2: Relation update methods sync to concepts
# ══════════════════════════════════════════════════════════════════════════


class TestRelationUpdateConceptSync:
    """D2: Relation field updates propagate to concepts table."""

    def test_update_relation_by_absolute_id_content(self, tmp_storage):
        """D2.1: update_relation_by_absolute_id(content=...) syncs to concepts."""
        e1 = _make_entity("d2_e1", "A", "A")
        e2 = _make_entity("d2_e2", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("d2_r1", e1.absolute_id, e2.absolute_id, "Original")
        tmp_storage.save_relation(r)

        tmp_storage.update_relation_by_absolute_id(r.absolute_id, content="Updated")

        concept = _get_concept(tmp_storage, r.absolute_id)
        assert concept is not None
        assert concept["content"] == "Updated"

    def test_update_relation_by_absolute_id_summary(self, tmp_storage):
        """D2.2: update_relation_by_absolute_id(summary=...) syncs to concepts."""
        e1 = _make_entity("d2_e3", "A", "A")
        e2 = _make_entity("d2_e4", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("d2_r2", e1.absolute_id, e2.absolute_id, "Rel")
        tmp_storage.save_relation(r)

        tmp_storage.update_relation_by_absolute_id(r.absolute_id, summary="Rel summary")

        concept = _get_concept(tmp_storage, r.absolute_id)
        assert concept["summary"] == "Rel summary"

    def test_update_relation_by_absolute_id_confidence(self, tmp_storage):
        """D2.3: update_relation_by_absolute_id(confidence=...) syncs to concepts."""
        e1 = _make_entity("d2_e5", "A", "A")
        e2 = _make_entity("d2_e6", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("d2_r3", e1.absolute_id, e2.absolute_id, "Rel")
        tmp_storage.save_relation(r)

        tmp_storage.update_relation_by_absolute_id(r.absolute_id, confidence=0.88)

        concept = _get_concept(tmp_storage, r.absolute_id)
        assert concept["confidence"] == 0.88

    def test_update_relation_by_absolute_id_attributes(self, tmp_storage):
        """D2.4: update_relation_by_absolute_id(attributes=...) syncs to concepts."""
        e1 = _make_entity("d2_e7", "A", "A")
        e2 = _make_entity("d2_e8", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("d2_r4", e1.absolute_id, e2.absolute_id, "Rel")
        tmp_storage.save_relation(r)

        attrs = '{"type": "causal"}'
        tmp_storage.update_relation_by_absolute_id(r.absolute_id, attributes=attrs)

        concept = _get_concept(tmp_storage, r.absolute_id)
        assert concept["attributes"] == attrs

    def test_update_relation_confidence_method(self, tmp_storage):
        """D2.5: update_relation_confidence syncs to concepts."""
        e1 = _make_entity("d2_e9", "A", "A")
        e2 = _make_entity("d2_e10", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("d2_r5", e1.absolute_id, e2.absolute_id, "Rel")
        tmp_storage.save_relation(r)

        tmp_storage.update_relation_confidence("d2_r5", 0.65)

        concept = _get_concept(tmp_storage, r.absolute_id)
        assert concept["confidence"] == 0.65

    def test_relation_update_no_fields_no_crash(self, tmp_storage):
        """D2.6: update_relation_by_absolute_id with no valid fields doesn't crash."""
        e1 = _make_entity("d2_e11", "A", "A")
        e2 = _make_entity("d2_e12", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("d2_r6", e1.absolute_id, e2.absolute_id, "Rel")
        tmp_storage.save_relation(r)

        result = tmp_storage.update_relation_by_absolute_id(r.absolute_id, unknown_field="x")
        assert result is not None

    def test_relation_multiple_fields_update(self, tmp_storage):
        """D2.7: Multiple relation fields updated at once all sync."""
        e1 = _make_entity("d2_e13", "A", "A")
        e2 = _make_entity("d2_e14", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("d2_r7", e1.absolute_id, e2.absolute_id, "Original")
        tmp_storage.save_relation(r)

        tmp_storage.update_relation_by_absolute_id(
            r.absolute_id, content="Updated", summary="Sum", confidence=0.5
        )

        concept = _get_concept(tmp_storage, r.absolute_id)
        assert concept["content"] == "Updated"
        assert concept["summary"] == "Sum"
        assert concept["confidence"] == 0.5

    def test_relation_fts_sync_on_content_update(self, tmp_storage):
        """D2.8: Content update also updates concept_fts for relation."""
        e1 = _make_entity("d2_e15", "A", "A")
        e2 = _make_entity("d2_e16", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("d2_r8", e1.absolute_id, e2.absolute_id, "ZetaThetaRel")
        tmp_storage.save_relation(r)

        tmp_storage.update_relation_by_absolute_id(r.absolute_id, content="AlphaOmegaRelation")

        count = _count_concept_fts(tmp_storage, "AlphaOmegaRelation")
        assert count >= 1

    def test_relation_confidence_clamped_to_range(self, tmp_storage):
        """D2.9: Confidence is clamped to [0.0, 1.0] and syncs to concepts."""
        e1 = _make_entity("d2_e17", "A", "A")
        e2 = _make_entity("d2_e18", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("d2_r9", e1.absolute_id, e2.absolute_id, "Rel", confidence=0.5)
        tmp_storage.save_relation(r)

        tmp_storage.update_relation_confidence("d2_r9", 1.5)

        concept = _get_concept(tmp_storage, r.absolute_id)
        assert concept["confidence"] == 1.0

    def test_relation_confidence_negative_clamped(self, tmp_storage):
        """D2.10: Negative confidence is clamped to 0.0 and syncs to concepts."""
        e1 = _make_entity("d2_e19", "A", "A")
        e2 = _make_entity("d2_e20", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("d2_r10", e1.absolute_id, e2.absolute_id, "Rel", confidence=0.5)
        tmp_storage.save_relation(r)

        tmp_storage.update_relation_confidence("d2_r10", -0.5)

        concept = _get_concept(tmp_storage, r.absolute_id)
        assert concept["confidence"] == 0.0


# ══════════════════════════════════════════════════════════════════════════
# D3: Confidence adjustment methods sync for both entity and relation
# ══════════════════════════════════════════════════════════════════════════


class TestConfidenceAdjustmentConceptSync:
    """D3: Confidence adjustment methods sync to concepts table."""

    def test_corroboration_entity(self, tmp_storage):
        """D3.1: adjust_confidence_on_corroboration (entity) syncs to concepts."""
        e = _make_entity("d3_e1", "Test", "Content", confidence=0.5)
        tmp_storage.save_entity(e)

        tmp_storage.adjust_confidence_on_corroboration("d3_e1", source_type="entity")

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept["confidence"] == pytest.approx(0.55)

    def test_corroboration_relation(self, tmp_storage):
        """D3.2: adjust_confidence_on_corroboration (relation) syncs to concepts."""
        e1 = _make_entity("d3_e2", "A", "A")
        e2 = _make_entity("d3_e3", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("d3_r1", e1.absolute_id, e2.absolute_id, "Rel", confidence=0.6)
        tmp_storage.save_relation(r)

        tmp_storage.adjust_confidence_on_corroboration("d3_r1", source_type="relation")

        concept = _get_concept(tmp_storage, r.absolute_id)
        assert concept["confidence"] == pytest.approx(0.65)

    def test_corroboration_dream_halved(self, tmp_storage):
        """D3.3: Dream corroboration uses halved delta and syncs."""
        e = _make_entity("d3_e4", "Test", "Content", confidence=0.5)
        tmp_storage.save_entity(e)

        tmp_storage.adjust_confidence_on_corroboration("d3_e4", source_type="entity", is_dream=True)

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept["confidence"] == pytest.approx(0.525)

    def test_corroboration_caps_at_one(self, tmp_storage):
        """D3.4: Corroboration caps confidence at 1.0 and syncs."""
        e = _make_entity("d3_e5", "Test", "Content", confidence=0.99)
        tmp_storage.save_entity(e)

        tmp_storage.adjust_confidence_on_corroboration("d3_e5", source_type="entity")

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept["confidence"] == 1.0

    def test_contradiction_entity(self, tmp_storage):
        """D3.5: adjust_confidence_on_contradiction (entity) syncs to concepts."""
        e = _make_entity("d3_e6", "Test", "Content", confidence=0.8)
        tmp_storage.save_entity(e)

        tmp_storage.adjust_confidence_on_contradiction("d3_e6", source_type="entity")

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept["confidence"] == pytest.approx(0.7)

    def test_contradiction_relation(self, tmp_storage):
        """D3.6: adjust_confidence_on_contradiction (relation) syncs to concepts."""
        e1 = _make_entity("d3_e7", "A", "A")
        e2 = _make_entity("d3_e8", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("d3_r2", e1.absolute_id, e2.absolute_id, "Rel", confidence=0.3)
        tmp_storage.save_relation(r)

        tmp_storage.adjust_confidence_on_contradiction("d3_r2", source_type="relation")

        concept = _get_concept(tmp_storage, r.absolute_id)
        assert concept["confidence"] == pytest.approx(0.2)

    def test_contradiction_floors_at_zero(self, tmp_storage):
        """D3.7: Contradiction floors confidence at 0.0 and syncs."""
        e = _make_entity("d3_e9", "Test", "Content", confidence=0.05)
        tmp_storage.save_entity(e)

        tmp_storage.adjust_confidence_on_contradiction("d3_e9", source_type="entity")

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept["confidence"] == 0.0

    def test_corroboration_skips_no_confidence(self, tmp_storage):
        """D3.8: Corroboration skips entities without confidence set."""
        e = _make_entity("d3_e10", "Test", "Content", confidence=None)
        tmp_storage.save_entity(e)

        # Should not crash
        tmp_storage.adjust_confidence_on_corroboration("d3_e10", source_type="entity")

    def test_contradiction_skips_no_confidence(self, tmp_storage):
        """D3.9: Contradiction skips entities without confidence set."""
        e = _make_entity("d3_e11", "Test", "Content", confidence=None)
        tmp_storage.save_entity(e)

        # Should not crash
        tmp_storage.adjust_confidence_on_contradiction("d3_e11", source_type="entity")

    def test_sequential_corroboration_contradiction(self, tmp_storage):
        """D3.10: Sequential corroboration then contradiction updates correctly."""
        e = _make_entity("d3_e12", "Test", "Content", confidence=0.5)
        tmp_storage.save_entity(e)

        tmp_storage.adjust_confidence_on_corroboration("d3_e12", source_type="entity")  # 0.55
        tmp_storage.adjust_confidence_on_contradiction("d3_e12", source_type="entity")  # 0.45

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept["confidence"] == pytest.approx(0.45)


# ══════════════════════════════════════════════════════════════════════════
# D4: Edge cases — versioning, missing concepts, concurrent, FTS
# ══════════════════════════════════════════════════════════════════════════


class TestConceptSyncEdgeCases:
    """D4: Edge cases for concept sync on updates."""

    def test_update_entity_nonexistent_absolute_id(self, tmp_storage):
        """D4.1: Updating nonexistent absolute_id doesn't crash."""
        result = tmp_storage.update_entity_by_absolute_id(
            "nonexistent-id-12345", name="Test"
        )
        assert result is None

    def test_update_relation_nonexistent_absolute_id(self, tmp_storage):
        """D4.2: Updating nonexistent relation absolute_id doesn't crash."""
        result = tmp_storage.update_relation_by_absolute_id(
            "nonexistent-rel-id-12345", content="Test"
        )
        assert result is None

    def test_entity_update_only_latest_version_concept(self, tmp_storage):
        """D4.3: Update syncs only the updated version's concept row."""
        e1 = _make_entity("d4_e1", "V1", "Version 1", confidence=0.5)
        tmp_storage.save_entity(e1)
        e2 = _make_entity("d4_e1", "V2", "Version 2", confidence=0.7)
        tmp_storage.save_entity(e2)

        # Update V2's confidence
        tmp_storage.update_entity_by_absolute_id(e2.absolute_id, confidence=0.9)

        # V2 concept should have new confidence
        concept_v2 = _get_concept(tmp_storage, e2.absolute_id)
        assert concept_v2["confidence"] == 0.9

        # V1 concept should be unchanged
        concept_v1 = _get_concept(tmp_storage, e1.absolute_id)
        assert concept_v1["confidence"] == pytest.approx(0.5)

    def test_relation_update_preserves_connects_field(self, tmp_storage):
        """D4.4: Relation update doesn't overwrite connects field in concepts."""
        e1 = _make_entity("d4_e2", "A", "A")
        e2 = _make_entity("d4_e3", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("d4_r1", e1.absolute_id, e2.absolute_id, "Rel")
        tmp_storage.save_relation(r)

        tmp_storage.update_relation_by_absolute_id(r.absolute_id, content="Updated")

        concept = _get_concept(tmp_storage, r.absolute_id)
        assert concept["connects"] is not None
        connects = json.loads(concept["connects"])
        assert e1.absolute_id in connects
        assert e2.absolute_id in connects

    def test_entity_update_with_unicode(self, tmp_storage):
        """D4.5: Unicode entity name/content updates sync correctly."""
        e = _make_entity("d4_e4", "Test", "Content")
        tmp_storage.save_entity(e)

        tmp_storage.update_entity_by_absolute_id(
            e.absolute_id, name="深度学习 🚀", content="包含中文字符"
        )

        concept = _get_concept(tmp_storage, e.absolute_id)
        assert concept["name"] == "深度学习 🚀"
        assert concept["content"] == "包含中文字符"

    def test_update_entity_summary_then_read_via_concept(self, tmp_storage):
        """D4.6: Updated summary is readable via concept API."""
        e = _make_entity("d4_e5", "Test", "Content")
        tmp_storage.save_entity(e)

        tmp_storage.update_entity_summary("d4_e5", "Conceptual summary")

        concept = tmp_storage.get_concept_by_family_id("d4_e5")
        assert concept is not None
        assert concept["summary"] == "Conceptual summary"

    def test_corroboration_relation_then_read_via_concept(self, tmp_storage):
        """D4.7: Corroboration-adjusted relation confidence readable via concept."""
        e1 = _make_entity("d4_e6", "A", "A")
        e2 = _make_entity("d4_e7", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("d4_r2", e1.absolute_id, e2.absolute_id, "Rel", confidence=0.4)
        tmp_storage.save_relation(r)

        tmp_storage.adjust_confidence_on_corroboration("d4_r2", source_type="relation")

        concept = tmp_storage.get_concept_by_family_id("d4_r2")
        assert concept is not None
        assert concept["confidence"] == pytest.approx(0.45)

    def test_entity_name_update_fts_searchable(self, tmp_storage):
        """D4.8: Updated entity name is searchable via concept BM25."""
        e = _make_entity("d4_e8", "BeforeUpdate", "Some content")
        tmp_storage.save_entity(e)

        tmp_storage.update_entity_by_absolute_id(e.absolute_id, name="AfterUpdate")

        results = tmp_storage.search_concepts_by_bm25("AfterUpdate", role="entity", limit=5)
        assert any(c.get("family_id") == "d4_e8" for c in results)
