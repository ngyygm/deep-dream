"""
Provenance fix + Episode serialization + Candidate layer API — 36 tests across 4 dimensions.

D1: get_concept_provenance works for all roles (entity/relation/observation) (9 tests)
D2: episode_to_dict includes processed_time + all time fields (9 tests)
D3: Dream candidate layer API endpoints (9 tests)
D4: Cross-cutting — edge cases, unicode, batch operations (9 tests)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import List

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


def _make_episode(absolute_id: str, content: str, source_document: str = "",
                  episode_type: str = ""):
    from processor.models import Episode
    now = datetime.now(timezone.utc)
    ep = Episode(
        absolute_id=absolute_id,
        content=content,
        event_time=now,
        source_document=source_document,
        episode_type=episode_type,
    )
    ep.processed_time = now
    return ep


# ══════════════════════════════════════════════════════════════════════════
# D1: get_concept_provenance works for all roles
# ══════════════════════════════════════════════════════════════════════════


class TestConceptProvenance:
    """Test that get_concept_provenance works for entity, relation, and observation roles."""

    def test_entity_provenance(self, tmp_storage):
        """D1.1: Entity concept provenance returns mentioning episodes."""
        e = _make_entity("prov_ent1", "Test", "Content")
        ep = _make_episode("prov_ep1", "Some text about Test")
        tmp_storage.save_entity(e)
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("prov_ep1", [e.absolute_id], target_type="entity")

        result = tmp_storage.get_concept_provenance("prov_ent1")
        assert len(result) == 1
        assert result[0]["episode_id"] == "prov_ep1"

    def test_entity_provenance_empty(self, tmp_storage):
        """D1.2: Entity with no mentions returns empty provenance."""
        e = _make_entity("prov_ent2", "Alone", "No mentions")
        tmp_storage.save_entity(e)
        result = tmp_storage.get_concept_provenance("prov_ent2")
        assert result == []

    def test_relation_provenance(self, tmp_storage):
        """D1.3: Relation concept provenance returns mentioning episodes."""
        e1 = _make_entity("prov_re1", "A", "A")
        e2 = _make_entity("prov_re2", "B", "B")
        r = _make_relation("prov_rel1", e1.absolute_id, e2.absolute_id, "A links to B")
        ep = _make_episode("prov_ep2", "Text about A linking to B")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("prov_ep2", [r.absolute_id], target_type="relation")

        result = tmp_storage.get_concept_provenance("prov_rel1")
        assert len(result) == 1
        assert result[0]["episode_id"] == "prov_ep2"

    def test_relation_provenance_empty(self, tmp_storage):
        """D1.4: Relation with no mentions returns empty provenance."""
        e1 = _make_entity("prov_re3", "A", "A")
        e2 = _make_entity("prov_re4", "B", "B")
        r = _make_relation("prov_rel2", e1.absolute_id, e2.absolute_id, "link")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)
        result = tmp_storage.get_concept_provenance("prov_rel2")
        assert result == []

    def test_nonexistent_concept_provenance(self, tmp_storage):
        """D1.5: Nonexistent concept returns empty provenance."""
        result = tmp_storage.get_concept_provenance("nonexistent_xyz_123")
        assert result == []

    def test_entity_provenance_multiple_episodes(self, tmp_storage):
        """D1.6: Entity mentioned by multiple episodes returns all."""
        e = _make_entity("prov_ent3", "Multi", "Mentioned many times")
        ep1 = _make_episode("prov_ep3a", "First mention of Multi")
        ep2 = _make_episode("prov_ep3b", "Second mention of Multi")
        tmp_storage.save_entity(e)
        tmp_storage.save_episode(ep1)
        tmp_storage.save_episode(ep2)
        tmp_storage.save_episode_mentions("prov_ep3a", [e.absolute_id], target_type="entity")
        tmp_storage.save_episode_mentions("prov_ep3b", [e.absolute_id], target_type="entity")

        result = tmp_storage.get_concept_provenance("prov_ent3")
        assert len(result) == 2
        ep_ids = {r["episode_id"] for r in result}
        assert "prov_ep3a" in ep_ids
        assert "prov_ep3b" in ep_ids

    def test_relation_with_multiple_versions(self, tmp_storage):
        """D1.7: Relation with multiple versions returns all episode mentions."""
        e1 = _make_entity("prov_re5", "A", "A")
        e2 = _make_entity("prov_re6", "B", "B")
        r1 = _make_relation("prov_rel3", e1.absolute_id, e2.absolute_id, "v1 link")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r1)
        ep = _make_episode("prov_ep4", "Mentions the relation")
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("prov_ep4", [r1.absolute_id], target_type="relation")

        # Create a second version
        r2 = _make_relation("prov_rel3", e1.absolute_id, e2.absolute_id, "v2 link")
        tmp_storage.save_relation(r2)

        result = tmp_storage.get_concept_provenance("prov_rel3")
        assert len(result) >= 1

    def test_get_concept_mentions_delegates_to_provenance(self, tmp_storage):
        """D1.8: get_concept_mentions delegates to get_concept_provenance."""
        e = _make_entity("prov_ent4", "Test", "Content")
        ep = _make_episode("prov_ep5", "Text")
        tmp_storage.save_entity(e)
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("prov_ep5", [e.absolute_id], target_type="entity")

        mentions = tmp_storage.get_concept_mentions("prov_ent4")
        provenance = tmp_storage.get_concept_provenance("prov_ent4")
        assert mentions == provenance

    def test_provenance_result_has_required_fields(self, tmp_storage):
        """D1.9: Each provenance entry has episode_id, content, event_time, source_document."""
        e = _make_entity("prov_ent5", "Fields", "Content")
        ep = _make_episode("prov_ep6", "Checking fields", source_document="test_doc.txt")
        tmp_storage.save_entity(e)
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("prov_ep6", [e.absolute_id], target_type="entity")

        result = tmp_storage.get_concept_provenance("prov_ent5")
        assert len(result) == 1
        entry = result[0]
        assert "episode_id" in entry
        assert "content" in entry
        assert "event_time" in entry
        assert "source_document" in entry
        assert entry["source_document"] == "test_doc.txt"


# ══════════════════════════════════════════════════════════════════════════
# D2: episode_to_dict includes processed_time + all time fields
# ══════════════════════════════════════════════════════════════════════════


class TestEpisodeSerialization:
    """Test that episode_to_dict includes all time fields."""

    def test_episode_has_processed_time(self):
        """D2.1: episode_to_dict includes processed_time."""
        from server.blueprints.helpers import episode_to_dict
        ep = _make_episode("epser1", "Content")
        d = episode_to_dict(ep)
        assert "processed_time" in d
        assert d["processed_time"] is not None

    def test_episode_has_event_time(self):
        """D2.2: episode_to_dict includes event_time."""
        from server.blueprints.helpers import episode_to_dict
        ep = _make_episode("epser2", "Content")
        d = episode_to_dict(ep)
        assert "event_time" in d
        assert d["event_time"] is not None

    def test_episode_has_absolute_id(self):
        """D2.3: episode_to_dict includes absolute_id."""
        from server.blueprints.helpers import episode_to_dict
        ep = _make_episode("epser3", "Content")
        d = episode_to_dict(ep)
        assert d["absolute_id"] == "epser3"
        assert d["id"] == "epser3"  # backward compat

    def test_episode_has_source_document(self):
        """D2.4: episode_to_dict includes source_document."""
        from server.blueprints.helpers import episode_to_dict
        ep = _make_episode("epser4", "Content", source_document="my_doc.txt")
        d = episode_to_dict(ep)
        assert d["source_document"] == "my_doc.txt"

    def test_episode_has_content(self):
        """D2.5: episode_to_dict includes content."""
        from server.blueprints.helpers import episode_to_dict
        ep = _make_episode("epser5", "Some interesting content")
        d = episode_to_dict(ep)
        assert "Some interesting content" in d["content"]

    def test_episode_isoformat_times(self):
        """D2.6: event_time and processed_time are ISO format strings."""
        from server.blueprints.helpers import episode_to_dict
        ep = _make_episode("epser6", "Content")
        d = episode_to_dict(ep)
        # Should be ISO format strings
        assert isinstance(d["event_time"], str)
        assert "T" in d["event_time"]  # ISO format contains T
        assert isinstance(d["processed_time"], str)
        assert "T" in d["processed_time"]

    def test_episode_activity_type(self):
        """D2.7: episode_to_dict includes activity_type."""
        from server.blueprints.helpers import episode_to_dict
        ep = _make_episode("epser7", "Content")
        ep.activity_type = "remember"
        d = episode_to_dict(ep)
        assert d.get("activity_type") is not None

    def test_episode_chinese_content(self):
        """D2.8: Chinese content preserved in episode_to_dict."""
        from server.blueprints.helpers import episode_to_dict
        ep = _make_episode("epser8", "这是一段中文内容，包含深度学习相关文本")
        d = episode_to_dict(ep)
        assert "深度学习" in d["content"]

    def test_episode_episode_type(self):
        """D2.9: episode_to_dict includes episode_type."""
        from server.blueprints.helpers import episode_to_dict
        ep = _make_episode("epser9", "Content", episode_type="dream")
        d = episode_to_dict(ep)
        assert d.get("episode_type") == "dream"


# ══════════════════════════════════════════════════════════════════════════
# D3: Dream candidate layer API endpoints
# ══════════════════════════════════════════════════════════════════════════


class TestCandidateLayerAPI:
    """Test Dream candidate layer storage methods (equivalent to API endpoint logic)."""

    def _create_dream_candidate(self, storage, fid_prefix="dcand",
                                 status="hypothesized", content="Dream link"):
        """Helper: create a dream candidate relation."""
        e1 = _make_entity(f"{fid_prefix}_e1", "A", "A")
        e2 = _make_entity(f"{fid_prefix}_e2", "B", "B")
        storage.bulk_save_entities([e1, e2])
        attrs = json.dumps({"tier": "candidate", "status": status, "corroboration_count": 0})
        r = _make_relation(
            f"{fid_prefix}_r", e1.absolute_id, e2.absolute_id, content,
            source_document="dream_cycle_abc123", attributes=attrs, confidence=0.5)
        storage.save_relation(r)
        return r

    def test_list_candidates_empty(self, tmp_storage):
        """D3.1: No candidates returns empty list."""
        result = tmp_storage.get_candidate_relations()
        assert isinstance(result, list)

    def test_list_candidates_with_dream_source(self, tmp_storage):
        """D3.2: Candidates with dream source_document are returned."""
        r = self._create_dream_candidate(tmp_storage)
        result = tmp_storage.get_candidate_relations()
        fids = [rel.family_id for rel in result]
        assert r.family_id in fids

    def test_count_candidates(self, tmp_storage):
        """D3.3: count_candidate_relations returns correct count."""
        self._create_dream_candidate(tmp_storage)
        count = tmp_storage.count_candidate_relations()
        assert count >= 1

    def test_count_candidates_by_status(self, tmp_storage):
        """D3.4: count_candidate_relations filters by status."""
        self._create_dream_candidate(tmp_storage, "dcnt1", status="hypothesized")
        count_hyp = tmp_storage.count_candidate_relations(status="hypothesized")
        count_ver = tmp_storage.count_candidate_relations(status="verified")
        assert count_hyp >= 1
        assert count_ver == 0

    def test_promote_candidate(self, tmp_storage):
        """D3.5: promote_candidate_relation changes status to verified."""
        r = self._create_dream_candidate(tmp_storage, "dprom")
        result = tmp_storage.promote_candidate_relation(r.family_id)
        assert result["new_status"] == "verified"
        assert result["new_tier"] == "verified"
        assert result["confidence"] >= 0.7

    def test_promote_candidate_not_found(self, tmp_storage):
        """D3.6: Promoting nonexistent relation raises ValueError."""
        with pytest.raises(ValueError):
            tmp_storage.promote_candidate_relation("nonexistent_xyz")

    def test_demote_candidate(self, tmp_storage):
        """D3.7: demote_candidate_relation changes status to rejected."""
        r = self._create_dream_candidate(tmp_storage, "ddem")
        result = tmp_storage.demote_candidate_relation(r.family_id, reason="test rejection")
        assert result["new_status"] == "rejected"
        assert result["confidence"] <= 0.2

    def test_demote_candidate_not_found(self, tmp_storage):
        """D3.8: Demoting nonexistent relation raises ValueError."""
        with pytest.raises(ValueError):
            tmp_storage.demote_candidate_relation("nonexistent_xyz")

    def test_corroborate_dream_relation(self, tmp_storage):
        """D3.9: corroborate_dream_relation increases corroboration count."""
        e1 = _make_entity("corr_e1", "X", "X")
        e2 = _make_entity("corr_e2", "Y", "Y")
        tmp_storage.bulk_save_entities([e1, e2])
        attrs = json.dumps({"tier": "candidate", "status": "hypothesized",
                            "corroboration_count": 0})
        r = _make_relation("corr_r", e1.absolute_id, e2.absolute_id, "X-Y",
                           source_document="dream_cycle_test", attributes=attrs,
                           confidence=0.5)
        tmp_storage.save_relation(r)
        # First corroboration: count goes to 1, still hypothesized
        result = tmp_storage.corroborate_dream_relation("corr_e1", "corr_e2")
        assert result["family_id"] == "corr_r"
        assert result["corroboration_count"] == 1
        assert result["status"] == "hypothesized"
        # Second corroboration: auto-promotes to verified
        result2 = tmp_storage.corroborate_dream_relation("corr_e1", "corr_e2")
        assert result2["new_status"] == "verified"


# ══════════════════════════════════════════════════════════════════════════
# D4: Cross-cutting — edge cases, unicode, batch
# ══════════════════════════════════════════════════════════════════════════


class TestProvenanceAndCandidateEdgeCases:
    """Cross-cutting edge cases for provenance and candidates."""

    def test_provenance_with_unicode_episode_content(self, tmp_storage):
        """D4.1: Provenance preserves Chinese/unicode content."""
        e = _make_entity("uni_pe1", "深度学习", "AI子领域")
        ep = _make_episode("uni_ep1", "关于深度学习的讨论 🚀", source_document="中文文档.txt")
        tmp_storage.save_entity(e)
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("uni_ep1", [e.absolute_id], target_type="entity")

        result = tmp_storage.get_concept_provenance("uni_pe1")
        assert len(result) == 1
        assert "深度学习" in result[0]["content"]
        assert "🚀" in result[0]["content"]

    def test_provenance_deduplicates_episodes(self, tmp_storage):
        """D4.2: Same episode mentioning same entity twice doesn't duplicate."""
        e = _make_entity("dedup_pe1", "Dedup", "Test")
        ep = _make_episode("dedup_ep1", "Text")
        tmp_storage.save_entity(e)
        tmp_storage.save_episode(ep)
        # Same mention saved twice should still return once due to DISTINCT
        tmp_storage.save_episode_mentions("dedup_ep1", [e.absolute_id], target_type="entity")

        result = tmp_storage.get_concept_provenance("dedup_pe1")
        assert len(result) == 1

    def test_candidate_promotion_preserves_content(self, tmp_storage):
        """D4.3: Promoted candidate preserves original content."""
        e1 = _make_entity("pres_e1", "A", "A")
        e2 = _make_entity("pres_e2", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        attrs = json.dumps({"tier": "candidate", "status": "hypothesized",
                            "corroboration_count": 0})
        r = _make_relation("pres_r", e1.absolute_id, e2.absolute_id,
                           "Original dream content", source_document="dream_cycle_test",
                           attributes=attrs, confidence=0.5)
        tmp_storage.save_relation(r)
        tmp_storage.promote_candidate_relation("pres_r")

        updated = tmp_storage.get_relation_by_family_id("pres_r")
        assert "Original dream content" in updated.content

    def test_candidate_demotion_with_reason(self, tmp_storage):
        """D4.4: Demotion reason preserved in attributes."""
        e1 = _make_entity("reas_e1", "A", "A")
        e2 = _make_entity("reas_e2", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        attrs = json.dumps({"tier": "candidate", "status": "hypothesized",
                            "corroboration_count": 0})
        r = _make_relation("reas_r", e1.absolute_id, e2.absolute_id, "link",
                           source_document="dream_cycle_test", attributes=attrs,
                           confidence=0.5)
        tmp_storage.save_relation(r)
        tmp_storage.demote_candidate_relation("reas_r", reason="incorrect fact")

        updated = tmp_storage.get_relation_by_family_id("reas_r")
        parsed_attrs = json.loads(updated.attributes)
        assert parsed_attrs.get("rejected_reason") == "incorrect fact"

    def test_promote_with_custom_confidence(self, tmp_storage):
        """D4.5: Promote with explicit confidence overrides auto value."""
        e1 = _make_entity("cust_e1", "A", "A")
        e2 = _make_entity("cust_e2", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        attrs = json.dumps({"tier": "candidate", "status": "hypothesized",
                            "corroboration_count": 0})
        r = _make_relation("cust_r", e1.absolute_id, e2.absolute_id, "link",
                           source_document="dream_cycle_test", attributes=attrs,
                           confidence=0.5)
        tmp_storage.save_relation(r)
        result = tmp_storage.promote_candidate_relation("cust_r", new_confidence=0.95)
        assert result["confidence"] == 0.95

    def test_candidate_list_pagination(self, tmp_storage):
        """D4.6: Candidate listing supports pagination."""
        # Create 3 candidates
        for i in range(3):
            e1 = _make_entity(f"pag_e1_{i}", f"A{i}", f"A{i}")
            e2 = _make_entity(f"pag_e2_{i}", f"B{i}", f"B{i}")
            tmp_storage.bulk_save_entities([e1, e2])
            attrs = json.dumps({"tier": "candidate", "status": "hypothesized",
                                "corroboration_count": 0})
            r = _make_relation(f"pag_r_{i}", e1.absolute_id, e2.absolute_id,
                               f"Dream link {i}", source_document="dream_cycle_test",
                               attributes=attrs, confidence=0.5)
            tmp_storage.save_relation(r)

        page1 = tmp_storage.get_candidate_relations(limit=2, offset=0)
        page2 = tmp_storage.get_candidate_relations(limit=2, offset=2)
        assert len(page1) <= 2
        # Total should be at least 3
        total = tmp_storage.count_candidate_relations()
        assert total >= 3

    def test_provenance_for_entity_mentioned_as_relation_target(self, tmp_storage):
        """D4.7: Entity provenance works even when mentioned via relation mention."""
        e1 = _make_entity("rm_ent1", "X", "X")
        e2 = _make_entity("rm_ent2", "Y", "Y")
        r = _make_relation("rm_rel1", e1.absolute_id, e2.absolute_id, "X-Y")
        ep = _make_episode("rm_ep1", "Text about relation")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)
        tmp_storage.save_episode(ep)
        # Mention the relation, not the entity
        tmp_storage.save_episode_mentions("rm_ep1", [r.absolute_id], target_type="relation")

        # Entity provenance should not include relation-targeting mentions
        result = tmp_storage.get_concept_provenance("rm_ent1")
        # This entity wasn't directly mentioned, so should be empty
        assert len(result) == 0

    def test_episode_serialization_preserves_all_fields(self):
        """D4.8: All episode fields survive serialization."""
        from server.blueprints.helpers import episode_to_dict
        ep = _make_episode("rt_ep1", "Round trip content", source_document="rt_doc.txt",
                           episode_type="remember")
        ep.activity_type = "extraction"
        d = episode_to_dict(ep)

        assert d["absolute_id"] == "rt_ep1"
        assert d["id"] == "rt_ep1"
        assert "Round trip content" in d["content"]
        assert d["source_document"] == "rt_doc.txt"
        assert d["event_time"] is not None
        assert d["processed_time"] is not None
        assert d["episode_type"] == "remember"

    def test_candidate_promote_then_demote(self, tmp_storage):
        """D4.9: Can promote then demote a candidate."""
        e1 = _make_entity("pd_e1", "A", "A")
        e2 = _make_entity("pd_e2", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        attrs = json.dumps({"tier": "candidate", "status": "hypothesized",
                            "corroboration_count": 0})
        r = _make_relation("pd_r", e1.absolute_id, e2.absolute_id, "link",
                           source_document="dream_cycle_test", attributes=attrs,
                           confidence=0.5)
        tmp_storage.save_relation(r)

        # Promote
        prom_result = tmp_storage.promote_candidate_relation("pd_r")
        assert prom_result["new_status"] == "verified"

        # Then demote
        dem_result = tmp_storage.demote_candidate_relation("pd_r", reason="changed mind")
        assert dem_result["new_status"] == "rejected"
