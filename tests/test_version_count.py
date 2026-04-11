"""
Version Count Propagation tests — 36 tests across 4 dimensions.

D1: entity_to_dict / relation_to_dict version_count parameter (9 tests)
D2: Storage batch version count methods (9 tests)
D3: enrich_entity/relation_version_counts helpers (9 tests)
D4: Endpoint-level version_count in search results (9 tests)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict

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
                 episode_id: str = "ep_test", source_document: str = "",
                 confidence: float = None):
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
                   confidence: float = None):
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
    )


# ══════════════════════════════════════════════════════════════════════════
# D1: entity_to_dict / relation_to_dict version_count parameter
# ══════════════════════════════════════════════════════════════════════════


class TestSerializationVersionCount:
    """Test that entity_to_dict and relation_to_dict handle version_count."""

    def test_entity_dict_no_vc_omits_field(self, tmp_storage):
        """D1.1: entity_to_dict without version_count does not include it."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ent_vc1", "Test", "Content")
        d = entity_to_dict(e)
        assert "version_count" not in d

    def test_entity_dict_with_vc_includes_field(self, tmp_storage):
        """D1.2: entity_to_dict with version_count includes it."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ent_vc2", "Test", "Content")
        d = entity_to_dict(e, version_count=3)
        assert d["version_count"] == 3

    def test_entity_vc_zero(self, tmp_storage):
        """D1.3: version_count=0 is included."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ent_vc3", "Test", "Content")
        d = entity_to_dict(e, version_count=0)
        assert "version_count" in d
        assert d["version_count"] == 0

    def test_entity_vc_with_score(self, tmp_storage):
        """D1.4: version_count coexists with _score."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ent_vc4", "Test", "Content")
        d = entity_to_dict(e, _score=0.85, version_count=5)
        assert d["_score"] == 0.85
        assert d["version_count"] == 5

    def test_relation_dict_no_vc_omits_field(self, tmp_storage):
        """D1.5: relation_to_dict without version_count does not include it."""
        from server.blueprints.helpers import relation_to_dict
        r = _make_relation("rel_vc5", "a1", "a2", "test")
        d = relation_to_dict(r)
        assert "version_count" not in d

    def test_relation_dict_with_vc_includes_field(self, tmp_storage):
        """D1.6: relation_to_dict with version_count includes it."""
        from server.blueprints.helpers import relation_to_dict
        r = _make_relation("rel_vc6", "a1", "a2", "test")
        d = relation_to_dict(r, version_count=2)
        assert d["version_count"] == 2

    def test_relation_vc_with_score(self, tmp_storage):
        """D1.7: relation version_count coexists with _score."""
        from server.blueprints.helpers import relation_to_dict
        r = _make_relation("rel_vc7", "a1", "a2", "test")
        d = relation_to_dict(r, _score=0.75, version_count=4)
        assert d["_score"] == 0.75
        assert d["version_count"] == 4

    def test_entity_vc_preserves_all_fields(self, tmp_storage):
        """D1.8: Adding version_count doesn't remove other fields."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ent_vc8", "TestName", "TestContent", confidence=0.9)
        d = entity_to_dict(e, version_count=7, _score=0.88)
        assert d["name"] == "TestName"
        assert d["family_id"] == "ent_vc8"
        assert d["confidence"] == 0.9
        assert d["version_count"] == 7
        assert d["_score"] == 0.88

    def test_entity_vc_one(self, tmp_storage):
        """D1.9: version_count=1 is included."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ent_vc9", "Test", "Content")
        d = entity_to_dict(e, version_count=1)
        assert d["version_count"] == 1


# ══════════════════════════════════════════════════════════════════════════
# D2: Storage batch version count methods
# ══════════════════════════════════════════════════════════════════════════


class TestBatchVersionCounts:
    """Test batch version count storage methods."""

    def test_entity_single_family_one_version(self, tmp_storage):
        """D2.1: Single entity family with one version."""
        e = _make_entity("ent_bv1", "Test", "Content")
        tmp_storage.save_entity(e)
        counts = tmp_storage.get_entity_version_counts(["ent_bv1"])
        assert counts["ent_bv1"] == 1

    def test_entity_single_family_multiple_versions(self, tmp_storage):
        """D2.2: Entity family with 3 versions."""
        fid = "ent_bv2"
        for i in range(3):
            e = _make_entity(fid, f"Name{i}", f"Content{i}")
            tmp_storage.save_entity(e)
        counts = tmp_storage.get_entity_version_counts([fid])
        assert counts[fid] == 3

    def test_entity_multiple_families(self, tmp_storage):
        """D2.3: Multiple entity families with different version counts."""
        for i in range(2):
            tmp_storage.save_entity(_make_entity("fam_a", f"A{i}", f"Ca{i}"))
        for i in range(4):
            tmp_storage.save_entity(_make_entity("fam_b", f"B{i}", f"Cb{i}"))
        counts = tmp_storage.get_entity_version_counts(["fam_a", "fam_b"])
        assert counts["fam_a"] == 2
        assert counts["fam_b"] == 4

    def test_entity_empty_list(self, tmp_storage):
        """D2.4: Empty family_ids returns empty dict."""
        counts = tmp_storage.get_entity_version_counts([])
        assert counts == {}

    def test_entity_nonexistent_family(self, tmp_storage):
        """D2.5: Non-existent family_id returns empty dict."""
        counts = tmp_storage.get_entity_version_counts(["nonexistent_123"])
        assert counts == {}

    def test_relation_single_family_one_version(self, tmp_storage):
        """D2.6: Single relation family with one version."""
        e1 = _make_entity("ent_rbv1", "A", "A")
        e2 = _make_entity("ent_rbv2", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        r = _make_relation("rel_bv1", e1.absolute_id, e2.absolute_id, "test")
        tmp_storage.save_relation(r)
        counts = tmp_storage.get_relation_version_counts(["rel_bv1"])
        assert counts["rel_bv1"] == 1

    def test_relation_single_family_multiple_versions(self, tmp_storage):
        """D2.7: Relation family with 3 versions."""
        e1 = _make_entity("ent_rbv3", "A", "A")
        e2 = _make_entity("ent_rbv4", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        fid = "rel_bv2"
        for i in range(3):
            r = _make_relation(fid, e1.absolute_id, e2.absolute_id, f"test{i}")
            tmp_storage.save_relation(r)
        counts = tmp_storage.get_relation_version_counts([fid])
        assert counts[fid] == 3

    def test_relation_multiple_families(self, tmp_storage):
        """D2.8: Multiple relation families with different counts."""
        e1 = _make_entity("ent_rbv5", "A", "A")
        e2 = _make_entity("ent_rbv6", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        for i in range(2):
            tmp_storage.save_relation(_make_relation("rfam_a", e1.absolute_id, e2.absolute_id, f"Ra{i}"))
        for i in range(5):
            tmp_storage.save_relation(_make_relation("rfam_b", e1.absolute_id, e2.absolute_id, f"Rb{i}"))
        counts = tmp_storage.get_relation_version_counts(["rfam_a", "rfam_b"])
        assert counts["rfam_a"] == 2
        assert counts["rfam_b"] == 5

    def test_relation_nonexistent_family(self, tmp_storage):
        """D2.9: Non-existent relation family returns empty dict."""
        counts = tmp_storage.get_relation_version_counts(["nonexistent_rel_123"])
        assert counts == {}


# ══════════════════════════════════════════════════════════════════════════
# D3: enrich_entity/relation_version_counts helpers
# ══════════════════════════════════════════════════════════════════════════


class TestEnrichVersionCounts:
    """Test enrich_entity_version_counts and enrich_relation_version_counts."""

    def test_enrich_entity_basic(self, tmp_storage):
        """D3.1: Enrich entity dicts with version counts."""
        from server.blueprints.helpers import entity_to_dict, enrich_entity_version_counts
        for i in range(3):
            tmp_storage.save_entity(_make_entity("ent_enr1", f"Name{i}", f"Content{i}"))
        e = tmp_storage.get_entity_by_family_id("ent_enr1")
        dicts = [entity_to_dict(e)]
        enrich_entity_version_counts(dicts, tmp_storage)
        assert dicts[0]["version_count"] == 3

    def test_enrich_entity_empty_list(self, tmp_storage):
        """D3.2: Empty list is a no-op."""
        from server.blueprints.helpers import enrich_entity_version_counts
        result = enrich_entity_version_counts([], tmp_storage)
        assert result == []

    def test_enrich_entity_no_family_id(self, tmp_storage):
        """D3.3: Dicts without family_id are skipped."""
        from server.blueprints.helpers import enrich_entity_version_counts
        dicts = [{"name": "test"}]
        enrich_entity_version_counts(dicts, tmp_storage)
        assert "version_count" not in dicts[0]

    def test_enrich_entity_nonexistent_family(self, tmp_storage):
        """D3.4: Non-existent family_id doesn't add version_count."""
        from server.blueprints.helpers import entity_to_dict, enrich_entity_version_counts
        e = _make_entity("ent_ghost", "Ghost", "Content")
        dicts = [entity_to_dict(e)]
        enrich_entity_version_counts(dicts, tmp_storage)
        assert "version_count" not in dicts[0]

    def test_enrich_entity_multiple_families(self, tmp_storage):
        """D3.5: Multiple entity families enriched correctly."""
        from server.blueprints.helpers import entity_to_dict, enrich_entity_version_counts
        for i in range(2):
            tmp_storage.save_entity(_make_entity("fam_x", f"X{i}", f"Cx{i}"))
        for i in range(4):
            tmp_storage.save_entity(_make_entity("fam_y", f"Y{i}", f"Cy{i}"))
        ex = tmp_storage.get_entity_by_family_id("fam_x")
        ey = tmp_storage.get_entity_by_family_id("fam_y")
        dicts = [entity_to_dict(ex), entity_to_dict(ey)]
        enrich_entity_version_counts(dicts, tmp_storage)
        assert dicts[0]["version_count"] == 2
        assert dicts[1]["version_count"] == 4

    def test_enrich_relation_basic(self, tmp_storage):
        """D3.6: Enrich relation dicts with version counts."""
        from server.blueprints.helpers import relation_to_dict, enrich_relation_version_counts
        e1 = _make_entity("ent_er1", "A", "A")
        e2 = _make_entity("ent_er2", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        for i in range(3):
            tmp_storage.save_relation(_make_relation("rel_enr1", e1.absolute_id, e2.absolute_id, f"T{i}"))
        r = tmp_storage.get_relation_by_family_id("rel_enr1")
        dicts = [relation_to_dict(r)]
        enrich_relation_version_counts(dicts, tmp_storage)
        assert dicts[0]["version_count"] == 3

    def test_enrich_relation_empty_list(self, tmp_storage):
        """D3.7: Empty relation list is a no-op."""
        from server.blueprints.helpers import enrich_relation_version_counts
        result = enrich_relation_version_counts([], tmp_storage)
        assert result == []

    def test_enrich_relation_preserves_score(self, tmp_storage):
        """D3.8: Enrich preserves existing _score."""
        from server.blueprints.helpers import relation_to_dict, enrich_relation_version_counts
        e1 = _make_entity("ent_er3", "A", "A")
        e2 = _make_entity("ent_er4", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(_make_relation("rel_enr2", e1.absolute_id, e2.absolute_id, "T"))
        r = tmp_storage.get_relation_by_family_id("rel_enr2")
        dicts = [relation_to_dict(r, _score=0.77)]
        enrich_relation_version_counts(dicts, tmp_storage)
        assert dicts[0]["_score"] == 0.77
        assert dicts[0]["version_count"] == 1

    def test_enrich_entity_preserves_score(self, tmp_storage):
        """D3.9: Enrich preserves existing _score on entity dicts."""
        from server.blueprints.helpers import entity_to_dict, enrich_entity_version_counts
        tmp_storage.save_entity(_make_entity("ent_er5", "A", "A"))
        e = tmp_storage.get_entity_by_family_id("ent_er5")
        dicts = [entity_to_dict(e, _score=0.92)]
        enrich_entity_version_counts(dicts, tmp_storage)
        assert dicts[0]["_score"] == 0.92
        assert dicts[0]["version_count"] == 1


# ══════════════════════════════════════════════════════════════════════════
# D4: Endpoint-level version_count in search results (simulated)
# ══════════════════════════════════════════════════════════════════════════


class TestEndpointVersionCount:
    """Test version_count propagation through endpoint patterns."""

    def test_entity_search_single_version(self, tmp_storage):
        """D4.1: Entity search result with single version gets version_count=1."""
        from server.blueprints.helpers import entity_to_dict, enrich_entity_version_counts
        tmp_storage.save_entity(_make_entity("ent_ep1", "Alpha", "Alpha content"))
        e = tmp_storage.get_entity_by_family_id("ent_ep1")
        dicts = [entity_to_dict(e)]
        enrich_entity_version_counts(dicts, tmp_storage)
        assert dicts[0]["version_count"] == 1

    def test_entity_search_multi_version(self, tmp_storage):
        """D4.2: Entity search result with 4 versions gets version_count=4."""
        from server.blueprints.helpers import entity_to_dict, enrich_entity_version_counts
        for i in range(4):
            tmp_storage.save_entity(_make_entity("ent_ep2", f"V{i}", f"C{i}"))
        e = tmp_storage.get_entity_by_family_id("ent_ep2")
        dicts = [entity_to_dict(e)]
        enrich_entity_version_counts(dicts, tmp_storage)
        assert dicts[0]["version_count"] == 4

    def test_entity_search_with_score_and_vc(self, tmp_storage):
        """D4.3: Entity result has both _score and version_count."""
        from server.blueprints.helpers import entity_to_dict, enrich_entity_version_counts
        tmp_storage.save_entity(_make_entity("ent_ep3", "T", "C"))
        e = tmp_storage.get_entity_by_family_id("ent_ep3")
        dicts = [entity_to_dict(e, _score=0.65)]
        enrich_entity_version_counts(dicts, tmp_storage)
        assert dicts[0]["_score"] == 0.65
        assert dicts[0]["version_count"] == 1

    def test_relation_search_single_version(self, tmp_storage):
        """D4.4: Relation search result with single version gets version_count=1."""
        from server.blueprints.helpers import relation_to_dict, enrich_relation_version_counts
        e1 = _make_entity("ent_ep4", "A", "A")
        e2 = _make_entity("ent_ep5", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(_make_relation("rel_ep1", e1.absolute_id, e2.absolute_id, "link"))
        r = tmp_storage.get_relation_by_family_id("rel_ep1")
        dicts = [relation_to_dict(r)]
        enrich_relation_version_counts(dicts, tmp_storage)
        assert dicts[0]["version_count"] == 1

    def test_relation_search_multi_version(self, tmp_storage):
        """D4.5: Relation search result with 3 versions gets version_count=3."""
        from server.blueprints.helpers import relation_to_dict, enrich_relation_version_counts
        e1 = _make_entity("ent_ep6", "A", "A")
        e2 = _make_entity("ent_ep7", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        for i in range(3):
            tmp_storage.save_relation(_make_relation("rel_ep2", e1.absolute_id, e2.absolute_id, f"link{i}"))
        r = tmp_storage.get_relation_by_family_id("rel_ep2")
        dicts = [relation_to_dict(r)]
        enrich_relation_version_counts(dicts, tmp_storage)
        assert dicts[0]["version_count"] == 3

    def test_unified_search_mixed_entities(self, tmp_storage):
        """D4.6: Unified search — different version counts per entity."""
        from server.blueprints.helpers import entity_to_dict, enrich_entity_version_counts
        for i in range(2):
            tmp_storage.save_entity(_make_entity("fam_u1", f"U1_{i}", f"C1_{i}"))
        for i in range(5):
            tmp_storage.save_entity(_make_entity("fam_u2", f"U2_{i}", f"C2_{i}"))
        e1 = tmp_storage.get_entity_by_family_id("fam_u1")
        e2 = tmp_storage.get_entity_by_family_id("fam_u2")
        dicts = [entity_to_dict(e1), entity_to_dict(e2)]
        enrich_entity_version_counts(dicts, tmp_storage)
        assert dicts[0]["version_count"] == 2
        assert dicts[1]["version_count"] == 5

    def test_unified_search_mixed_relations(self, tmp_storage):
        """D4.7: Unified search — different version counts per relation."""
        from server.blueprints.helpers import relation_to_dict, enrich_relation_version_counts
        e1 = _make_entity("ent_ep8", "A", "A")
        e2 = _make_entity("ent_ep9", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        for i in range(3):
            tmp_storage.save_relation(_make_relation("ru1", e1.absolute_id, e2.absolute_id, f"L1_{i}"))
        for i in range(1):
            tmp_storage.save_relation(_make_relation("ru2", e1.absolute_id, e2.absolute_id, f"L2_{i}"))
        r1 = tmp_storage.get_relation_by_family_id("ru1")
        r2 = tmp_storage.get_relation_by_family_id("ru2")
        dicts = [relation_to_dict(r1), relation_to_dict(r2)]
        enrich_relation_version_counts(dicts, tmp_storage)
        assert dicts[0]["version_count"] == 3
        assert dicts[1]["version_count"] == 1

    def test_large_batch_version_counts(self, tmp_storage):
        """D4.8: Large batch of entities with version counts."""
        from server.blueprints.helpers import entity_to_dict, enrich_entity_version_counts
        fids = [f"ent_batch_{i}" for i in range(20)]
        for fid in fids:
            tmp_storage.save_entity(_make_entity(fid, fid, fid))
            # Give some entities multiple versions
            if fid.endswith("0") or fid.endswith("5"):
                tmp_storage.save_entity(_make_entity(fid, fid + "v2", fid + "v2"))

        entities = []
        for fid in fids:
            e = tmp_storage.get_entity_by_family_id(fid)
            if e:
                entities.append(e)

        dicts = [entity_to_dict(e) for e in entities]
        enrich_entity_version_counts(dicts, tmp_storage)
        assert len(dicts) == 20
        # Every entity should have version_count
        for d in dicts:
            assert "version_count" in d
            assert d["version_count"] >= 1
        # Entities ending in 0 or 5 should have 2 versions
        for d in dicts:
            fid = d["family_id"]
            if fid.endswith("0") or fid.endswith("5"):
                assert d["version_count"] == 2

    def test_entity_profile_pattern(self, tmp_storage):
        """D4.9: entity_profile pattern — entity_to_dict with version_count."""
        from server.blueprints.helpers import entity_to_dict
        for i in range(3):
            tmp_storage.save_entity(_make_entity("ent_profile", f"P{i}", f"C{i}"))
        e = tmp_storage.get_entity_by_family_id("ent_profile")
        vc = tmp_storage.get_entity_version_count("ent_profile")
        d = entity_to_dict(e, version_count=vc)
        assert d["version_count"] == 3
        assert d["family_id"] == "ent_profile"
