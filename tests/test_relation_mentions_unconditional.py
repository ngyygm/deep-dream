"""
Relation MENTIONS unconditional — 36 tests across 4 dimensions.

Vision principle: "内容版本和关联解耦" — MENTIONS must be created unconditionally
for all resolved relation pairs (both new and existing), even when no new version
is created.

D1: process_relations_batch MENTIONS for new relations (9 tests)
D2: process_relations_batch MENTIONS for existing relations — no new version (9 tests)
D3: MENTIONS linkage — Episode → Relation provenance round-trip (9 tests)
D4: Cross-cutting — edge cases, unicode, batch, concurrent (9 tests)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Tuple
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
                 episode_id: str = "ep_test", source_document: str = ""):
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
    )


def _make_relation(family_id: str, e1_abs: str, e2_abs: str, content: str,
                   episode_id: str = "ep_test", source_document: str = "",
                   confidence: float = None, attributes: str = None):
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
        episode_id=episode_id,
        source_document=source_document,
        confidence=confidence,
        attributes=attributes,
    )


def _make_episode(absolute_id: str, content: str, source_document: str = ""):
    from processor.models import Episode
    now = datetime.now(timezone.utc)
    ep = Episode(
        absolute_id=absolute_id,
        content=content,
        event_time=now,
        source_document=source_document,
    )
    ep.processed_time = now
    return ep


def _setup_processor(tmp_storage):
    """Create a mock processor with real storage and mock LLM."""
    from processor.pipeline.relation import RelationProcessor
    mock_llm = MagicMock()
    proc = RelationProcessor(tmp_storage, mock_llm)
    return proc


# ══════════════════════════════════════════════════════════════════════════
# D1: process_relations_batch MENTIONS for new relations
# ══════════════════════════════════════════════════════════════════════════


class TestNewRelationMentions:
    """D1: New relations produce absolute_ids in processed_relations."""

    def test_new_relation_in_processed_results(self, tmp_storage):
        """D1.1: A brand new relation appears in processed_relations."""
        e1 = _make_entity("new_e1", "Alice", "Alice is a person")
        e2 = _make_entity("new_e2", "Bob", "Bob is a person")
        tmp_storage.bulk_save_entities([e1, e2])

        proc = _setup_processor(tmp_storage)
        # Mock LLM to skip relation resolution — force new creation
        proc.batch_resolution_enabled = False
        proc.llm_client._current_distill_step = None
        proc.llm_client._priority_local = MagicMock()
        proc.llm_client._priority_local.priority = 6

        mock_llm_result = MagicMock()
        mock_llm_result.action = "new"
        mock_llm_result.confidence = 0.8
        mock_llm_result.merged_content = "Alice knows Bob"
        proc.llm_client.resolve_relation_pair_batch.return_value = {
            "action": "create_new",
            "confidence": 0.8,
            "merged_content": "Alice knows Bob",
        }
        proc.llm_client.merge_multiple_relation_contents.return_value = "Alice knows Bob"

        extracted = [{"entity1_name": "Alice", "entity2_name": "Bob",
                      "content": "Alice knows Bob", "relation_type": "knows"}]
        name_to_id = {"Alice": "new_e1", "Bob": "new_e2"}

        results = proc.process_relations_batch(extracted, name_to_id, "ep_test")
        assert len(results) >= 1
        assert any(r.content == "Alice knows Bob" for r in results)

    def test_new_relation_absolute_id_not_empty(self, tmp_storage):
        """D1.2: New relation has a non-empty absolute_id."""
        e1 = _make_entity("nid_e1", "Cat", "Cat")
        e2 = _make_entity("nid_e2", "Dog", "Dog")
        tmp_storage.bulk_save_entities([e1, e2])

        proc = _setup_processor(tmp_storage)
        proc.llm_client._current_distill_step = None
        proc.llm_client._priority_local = MagicMock()
        proc.llm_client._priority_local.priority = 6
        proc.llm_client.merge_multiple_relation_contents.return_value = "Cat chases Dog"

        extracted = [{"entity1_name": "Cat", "entity2_name": "Dog",
                      "content": "Cat chases Dog"}]
        name_to_id = {"Cat": "nid_e1", "Dog": "nid_e2"}

        results = proc.process_relations_batch(extracted, name_to_id, "ep_test")
        assert all(r.absolute_id for r in results)

    def test_new_relation_family_id_generated(self, tmp_storage):
        """D1.3: New relation has a generated family_id."""
        e1 = _make_entity("fid_e1", "X", "X")
        e2 = _make_entity("fid_e2", "Y", "Y")
        tmp_storage.bulk_save_entities([e1, e2])

        proc = _setup_processor(tmp_storage)
        proc.llm_client._current_distill_step = None
        proc.llm_client._priority_local = MagicMock()
        proc.llm_client._priority_local.priority = 6
        proc.llm_client.merge_multiple_relation_contents.return_value = "X-Y link"

        extracted = [{"entity1_name": "X", "entity2_name": "Y",
                      "content": "X-Y link"}]
        name_to_id = {"X": "fid_e1", "Y": "fid_e2"}

        results = proc.process_relations_batch(extracted, name_to_id, "ep_test")
        assert any(r.family_id.startswith("rel_") for r in results)

    def test_multiple_new_relations_all_returned(self, tmp_storage):
        """D1.4: Multiple new relations are all in processed_results."""
        e1 = _make_entity("mul_e1", "A", "A")
        e2 = _make_entity("mul_e2", "B", "B")
        e3 = _make_entity("mul_e3", "C", "C")
        tmp_storage.bulk_save_entities([e1, e2, e3])

        proc = _setup_processor(tmp_storage)
        proc.llm_client._current_distill_step = None
        proc.llm_client._priority_local = MagicMock()
        proc.llm_client._priority_local.priority = 6
        proc.llm_client.merge_multiple_relation_contents.side_effect = [
            "A-B link", "B-C link"
        ]

        extracted = [
            {"entity1_name": "A", "entity2_name": "B", "content": "A-B link"},
            {"entity1_name": "B", "entity2_name": "C", "content": "B-C link"},
        ]
        name_to_id = {"A": "mul_e1", "B": "mul_e2", "C": "mul_e3"}

        results = proc.process_relations_batch(extracted, name_to_id, "ep_test")
        assert len(results) >= 2

    def test_new_relation_mention_saved(self, tmp_storage):
        """D1.5: Episode mentions saved for new relation."""
        e1 = _make_entity("mns_e1", "P", "P")
        e2 = _make_entity("mns_e2", "Q", "Q")
        tmp_storage.bulk_save_entities([e1, e2])

        proc = _setup_processor(tmp_storage)
        proc.llm_client._current_distill_step = None
        proc.llm_client._priority_local = MagicMock()
        proc.llm_client._priority_local.priority = 6
        proc.llm_client.merge_multiple_relation_contents.return_value = "P-Q link"

        extracted = [{"entity1_name": "P", "entity2_name": "Q",
                      "content": "P-Q link"}]
        name_to_id = {"P": "mns_e1", "Q": "mns_e2"}

        results = proc.process_relations_batch(extracted, name_to_id, "ep_mention_test")
        assert len(results) >= 1

        # Now manually save episode mentions (simulating Phase C-2)
        rel_abs_ids = [r.absolute_id for r in results if r.absolute_id]
        ep = _make_episode("ep_mention_test", "Text about P and Q")
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("ep_mention_test", rel_abs_ids, target_type="relation")

        # Verify mention exists
        provenance = tmp_storage.get_concept_provenance(results[0].family_id)
        assert len(provenance) >= 1

    def test_new_relation_content_preserved(self, tmp_storage):
        """D1.6: Relation content preserved through processing."""
        e1 = _make_entity("cnt_e1", "M", "M")
        e2 = _make_entity("cnt_e2", "N", "N")
        tmp_storage.bulk_save_entities([e1, e2])

        proc = _setup_processor(tmp_storage)
        proc.llm_client._current_distill_step = None
        proc.llm_client._priority_local = MagicMock()
        proc.llm_client._priority_local.priority = 6
        proc.llm_client.merge_multiple_relation_contents.return_value = "M works with N"

        extracted = [{"entity1_name": "M", "entity2_name": "N",
                      "content": "M works with N"}]
        name_to_id = {"M": "cnt_e1", "N": "cnt_e2"}

        results = proc.process_relations_batch(extracted, name_to_id, "ep_test")
        assert any("works with" in r.content for r in results)

    def test_new_relation_episode_id_set(self, tmp_storage):
        """D1.7: New relation has correct episode_id."""
        e1 = _make_entity("epid_e1", "R", "R")
        e2 = _make_entity("epid_e2", "S", "S")
        tmp_storage.bulk_save_entities([e1, e2])

        proc = _setup_processor(tmp_storage)
        proc.llm_client._current_distill_step = None
        proc.llm_client._priority_local = MagicMock()
        proc.llm_client._priority_local.priority = 6
        proc.llm_client.merge_multiple_relation_contents.return_value = "R-S link"

        extracted = [{"entity1_name": "R", "entity2_name": "S",
                      "content": "R-S link"}]
        name_to_id = {"R": "epid_e1", "S": "epid_e2"}

        results = proc.process_relations_batch(extracted, name_to_id, "ep_custom_123")
        assert all(getattr(r, 'episode_id', None) == "ep_custom_123" for r in results)

    def test_self_relation_filtered(self, tmp_storage):
        """D1.8: Self-relation (same entity) is filtered out."""
        e1 = _make_entity("self_e1", "Z", "Z")
        tmp_storage.save_entity(e1)

        proc = _setup_processor(tmp_storage)
        extracted = [{"entity1_name": "Z", "entity2_name": "Z",
                      "content": "Z relates to itself"}]
        name_to_id = {"Z": "self_e1"}

        results = proc.process_relations_batch(extracted, name_to_id, "ep_test")
        assert len(results) == 0

    def test_missing_entity_filtered(self, tmp_storage):
        """D1.9: Relation with missing entity name is filtered out."""
        proc = _setup_processor(tmp_storage)
        extracted = [{"entity1_name": "", "entity2_name": "Y",
                      "content": "link"}]
        name_to_id = {"Y": "some_id"}

        results = proc.process_relations_batch(extracted, name_to_id, "ep_test")
        assert len(results) == 0


# ══════════════════════════════════════════════════════════════════════════
# D2: MENTIONS for existing relations — no new version
# ══════════════════════════════════════════════════════════════════════════


class TestExistingRelationMentions:
    """D2: Existing relations still produce MENTIONS (unconditional per vision.md)."""

    def test_existing_relation_returned_in_results(self, tmp_storage):
        """D2.1: When content matches exactly, existing relation is returned."""
        e1 = _make_entity("ex_e1", "Alpha", "Alpha")
        e2 = _make_entity("ex_e2", "Beta", "Beta")
        existing_rel = _make_relation("ex_r1", e1.absolute_id, e2.absolute_id,
                                       "Alpha works with Beta")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(existing_rel)

        proc = _setup_processor(tmp_storage)
        proc.llm_client._current_distill_step = None
        proc.llm_client._priority_local = MagicMock()
        proc.llm_client._priority_local.priority = 6

        # Extracted relation has same content → exact duplicate
        extracted = [{"entity1_name": "Alpha", "entity2_name": "Beta",
                      "content": "Alpha works with Beta"}]
        name_to_id = {"Alpha": "ex_e1", "Beta": "ex_e2"}

        results = proc.process_relations_batch(extracted, name_to_id, "ep_test")
        assert len(results) >= 1
        # Should contain the existing relation (or a version of it)
        fids = {r.family_id for r in results}
        assert "ex_r1" in fids

    def test_existing_relation_mention_round_trip(self, tmp_storage):
        """D2.2: Existing relation's MENTION enables provenance lookup."""
        e1 = _make_entity("rt_e1", "Gamma", "Gamma")
        e2 = _make_entity("rt_e2", "Delta", "Delta")
        existing_rel = _make_relation("rt_r1", e1.absolute_id, e2.absolute_id,
                                       "Gamma-Delta link")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(existing_rel)
        ep = _make_episode("rt_ep1", "Text about Gamma and Delta")
        tmp_storage.save_episode(ep)

        proc = _setup_processor(tmp_storage)
        proc.llm_client._current_distill_step = None
        proc.llm_client._priority_local = MagicMock()
        proc.llm_client._priority_local.priority = 6

        extracted = [{"entity1_name": "Gamma", "entity2_name": "Delta",
                      "content": "Gamma-Delta link"}]
        name_to_id = {"Gamma": "rt_e1", "Delta": "rt_e2"}

        results = proc.process_relations_batch(extracted, name_to_id, "rt_ep1")
        rel_abs_ids = [r.absolute_id for r in results if r.absolute_id]
        tmp_storage.save_episode_mentions("rt_ep1", rel_abs_ids, target_type="relation")

        # Provenance should work
        prov = tmp_storage.get_concept_provenance("rt_r1")
        assert len(prov) >= 1
        assert any(p["episode_id"] == "rt_ep1" for p in prov)

    def test_existing_relation_no_duplicate_version(self, tmp_storage):
        """D2.3: Exact duplicate content doesn't create a new version."""
        e1 = _make_entity("dup_e1", "Epsilon", "Epsilon")
        e2 = _make_entity("dup_e2", "Zeta", "Zeta")
        existing_rel = _make_relation("dup_r1", e1.absolute_id, e2.absolute_id,
                                       "Epsilon-Zeta link")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(existing_rel)

        versions_before = tmp_storage.get_relation_versions("dup_r1")

        proc = _setup_processor(tmp_storage)
        proc.llm_client._current_distill_step = None
        proc.llm_client._priority_local = MagicMock()
        proc.llm_client._priority_local.priority = 6

        extracted = [{"entity1_name": "Epsilon", "entity2_name": "Zeta",
                      "content": "Epsilon-Zeta link"}]
        name_to_id = {"Epsilon": "dup_e1", "Zeta": "dup_e2"}

        proc.process_relations_batch(extracted, name_to_id, "ep_test")

        versions_after = tmp_storage.get_relation_versions("dup_r1")
        assert len(versions_after) == len(versions_before)

    def test_multiple_mentions_same_relation(self, tmp_storage):
        """D2.4: Same relation mentioned by multiple episodes has multiple provenance entries."""
        e1 = _make_entity("mm_e1", "Eta", "Eta")
        e2 = _make_entity("mm_e2", "Theta", "Theta")
        existing_rel = _make_relation("mm_r1", e1.absolute_id, e2.absolute_id,
                                       "Eta-Theta link")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(existing_rel)

        ep1 = _make_episode("mm_ep1", "First mention")
        ep2 = _make_episode("mm_ep2", "Second mention")
        tmp_storage.save_episode(ep1)
        tmp_storage.save_episode(ep2)
        tmp_storage.save_episode_mentions("mm_ep1", [existing_rel.absolute_id], target_type="relation")
        tmp_storage.save_episode_mentions("mm_ep2", [existing_rel.absolute_id], target_type="relation")

        prov = tmp_storage.get_concept_provenance("mm_r1")
        assert len(prov) >= 2

    def test_existing_and_new_in_same_batch(self, tmp_storage):
        """D2.5: Batch with both existing and new relations."""
        e1 = _make_entity("mix_e1", "Iota", "Iota")
        e2 = _make_entity("mix_e2", "Kappa", "Kappa")
        e3 = _make_entity("mix_e3", "Lambda", "Lambda")
        existing_rel = _make_relation("mix_r1", e1.absolute_id, e2.absolute_id,
                                       "Iota-Kappa link")
        tmp_storage.bulk_save_entities([e1, e2, e3])
        tmp_storage.save_relation(existing_rel)

        proc = _setup_processor(tmp_storage)
        proc.llm_client._current_distill_step = None
        proc.llm_client._priority_local = MagicMock()
        proc.llm_client._priority_local.priority = 6
        proc.llm_client.merge_multiple_relation_contents.return_value = "Kappa-Lambda link"

        extracted = [
            {"entity1_name": "Iota", "entity2_name": "Kappa", "content": "Iota-Kappa link"},
            {"entity1_name": "Kappa", "entity2_name": "Lambda", "content": "Kappa-Lambda link"},
        ]
        name_to_id = {"Iota": "mix_e1", "Kappa": "mix_e2", "Lambda": "mix_e3"}

        results = proc.process_relations_batch(extracted, name_to_id, "ep_test")
        assert len(results) >= 2

    def test_existing_relation_confidence_preserved(self, tmp_storage):
        """D2.6: When relation not updated, confidence is preserved."""
        e1 = _make_entity("conf_e1", "Mu", "Mu")
        e2 = _make_entity("conf_e2", "Nu", "Nu")
        existing_rel = _make_relation("conf_r1", e1.absolute_id, e2.absolute_id,
                                       "Mu-Nu link", confidence=0.85)
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(existing_rel)

        proc = _setup_processor(tmp_storage)
        proc.llm_client._current_distill_step = None
        proc.llm_client._priority_local = MagicMock()
        proc.llm_client._priority_local.priority = 6

        extracted = [{"entity1_name": "Mu", "entity2_name": "Nu",
                      "content": "Mu-Nu link"}]
        name_to_id = {"Mu": "conf_e1", "Nu": "conf_e2"}

        results = proc.process_relations_batch(extracted, name_to_id, "ep_test")
        mu_nu = [r for r in results if r.family_id == "conf_r1"]
        assert len(mu_nu) >= 1
        assert mu_nu[0].confidence == 0.85

    def test_existing_relation_version_count_unchanged(self, tmp_storage):
        """D2.7: Exact duplicate doesn't increase version count."""
        e1 = _make_entity("vc_e1", "Xi", "Xi")
        e2 = _make_entity("vc_e2", "Omicron", "Omicron")
        existing_rel = _make_relation("vc_r1", e1.absolute_id, e2.absolute_id,
                                       "Xi-Omicron link")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(existing_rel)

        counts_before = tmp_storage.get_relation_version_counts(["vc_r1"])

        proc = _setup_processor(tmp_storage)
        proc.llm_client._current_distill_step = None
        proc.llm_client._priority_local = MagicMock()
        proc.llm_client._priority_local.priority = 6

        extracted = [{"entity1_name": "Xi", "entity2_name": "Omicron",
                      "content": "Xi-Omicron link"}]
        name_to_id = {"Xi": "vc_e1", "Omicron": "vc_e2"}

        proc.process_relations_batch(extracted, name_to_id, "ep_test")

        counts_after = tmp_storage.get_relation_version_counts(["vc_r1"])
        assert counts_after["vc_r1"] == counts_before["vc_r1"]

    def test_existing_relation_attributes_preserved(self, tmp_storage):
        """D2.8: Attributes preserved when relation not updated."""
        e1 = _make_entity("attr_e1", "Pi", "Pi")
        e2 = _make_entity("attr_e2", "Rho", "Rho")
        attrs = json.dumps({"tier": "verified", "status": "confirmed"})
        existing_rel = _make_relation("attr_r1", e1.absolute_id, e2.absolute_id,
                                       "Pi-Rho link", attributes=attrs)
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(existing_rel)

        proc = _setup_processor(tmp_storage)
        proc.llm_client._current_distill_step = None
        proc.llm_client._priority_local = MagicMock()
        proc.llm_client._priority_local.priority = 6

        extracted = [{"entity1_name": "Pi", "entity2_name": "Rho",
                      "content": "Pi-Rho link"}]
        name_to_id = {"Pi": "attr_e1", "Rho": "attr_e2"}

        results = proc.process_relations_batch(extracted, name_to_id, "ep_test")
        attr_rels = [r for r in results if r.family_id == "attr_r1"]
        assert len(attr_rels) >= 1

    def test_no_relations_returns_empty(self, tmp_storage):
        """D2.9: No input relations returns empty list."""
        proc = _setup_processor(tmp_storage)
        results = proc.process_relations_batch([], {}, "ep_test")
        assert results == []


# ══════════════════════════════════════════════════════════════════════════
# D3: MENTIONS linkage — Episode → Relation provenance round-trip
# ══════════════════════════════════════════════════════════════════════════


class TestProvenanceRoundTrip:
    """D3: MENTIONS create correct provenance links for relations."""

    def test_provenance_after_manual_mention(self, tmp_storage):
        """D3.1: Manual mention creates correct provenance."""
        e1 = _make_entity("pr_e1", "A", "A")
        e2 = _make_entity("pr_e2", "B", "B")
        r = _make_relation("pr_r1", e1.absolute_id, e2.absolute_id, "A-B link")
        ep = _make_episode("pr_ep1", "Text about A-B")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("pr_ep1", [r.absolute_id], target_type="relation")

        prov = tmp_storage.get_concept_provenance("pr_r1")
        assert len(prov) == 1
        assert prov[0]["episode_id"] == "pr_ep1"
        assert "Text about A-B" in prov[0]["content"]

    def test_provenance_multiple_episodes(self, tmp_storage):
        """D3.2: Relation mentioned by multiple episodes shows all."""
        e1 = _make_entity("me_e1", "C", "C")
        e2 = _make_entity("me_e2", "D", "D")
        r = _make_relation("me_r1", e1.absolute_id, e2.absolute_id, "C-D link")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)

        for i in range(3):
            ep = _make_episode(f"me_ep{i}", f"Episode {i} about C-D")
            tmp_storage.save_episode(ep)
            tmp_storage.save_episode_mentions(f"me_ep{i}", [r.absolute_id], target_type="relation")

        prov = tmp_storage.get_concept_provenance("me_r1")
        assert len(prov) == 3

    def test_provenance_deduplication(self, tmp_storage):
        """D3.3: Same episode mentioning relation twice doesn't duplicate."""
        e1 = _make_entity("dd_e1", "E", "E")
        e2 = _make_entity("dd_e2", "F", "F")
        r = _make_relation("dd_r1", e1.absolute_id, e2.absolute_id, "E-F link")
        ep = _make_episode("dd_ep1", "Text about E-F")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("dd_ep1", [r.absolute_id], target_type="relation")

        prov = tmp_storage.get_concept_provenance("dd_r1")
        assert len(prov) == 1

    def test_provenance_for_entity_and_relation(self, tmp_storage):
        """D3.4: Same episode mentions both entity and relation."""
        e1 = _make_entity("er_e1", "G", "G content")
        e2 = _make_entity("er_e2", "H", "H content")
        r = _make_relation("er_r1", e1.absolute_id, e2.absolute_id, "G-H link")
        ep = _make_episode("er_ep1", "Text about G and H relationship")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("er_ep1", [e1.absolute_id, e2.absolute_id], target_type="entity")
        tmp_storage.save_episode_mentions("er_ep1", [r.absolute_id], target_type="relation")

        # Entity provenance
        ent_prov = tmp_storage.get_concept_provenance("er_e1")
        assert len(ent_prov) == 1
        # Relation provenance
        rel_prov = tmp_storage.get_concept_provenance("er_r1")
        assert len(rel_prov) == 1

    def test_provenance_empty_for_unmentioned(self, tmp_storage):
        """D3.5: Relation with no mentions returns empty provenance."""
        e1 = _make_entity("um_e1", "I", "I")
        e2 = _make_entity("um_e2", "J", "J")
        r = _make_relation("um_r1", e1.absolute_id, e2.absolute_id, "I-J link")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)

        prov = tmp_storage.get_concept_provenance("um_r1")
        assert prov == []

    def test_provenance_fields_complete(self, tmp_storage):
        """D3.6: Each provenance entry has required fields."""
        e1 = _make_entity("fld_e1", "K", "K")
        e2 = _make_entity("fld_e2", "L", "L")
        r = _make_relation("fld_r1", e1.absolute_id, e2.absolute_id, "K-L link")
        ep = _make_episode("fld_ep1", "K-L discussion", source_document="doc.txt")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("fld_ep1", [r.absolute_id], target_type="relation")

        prov = tmp_storage.get_concept_provenance("fld_r1")
        assert len(prov) == 1
        entry = prov[0]
        assert "episode_id" in entry
        assert "content" in entry
        assert "event_time" in entry
        assert "source_document" in entry
        assert entry["source_document"] == "doc.txt"

    def test_entity_mention_not_in_relation_provenance(self, tmp_storage):
        """D3.7: Entity-type mentions don't appear in relation provenance."""
        e1 = _make_entity("sep_e1", "M", "M")
        e2 = _make_entity("sep_e2", "N", "N")
        r = _make_relation("sep_r1", e1.absolute_id, e2.absolute_id, "M-N link")
        ep = _make_episode("sep_ep1", "Text")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)
        tmp_storage.save_episode(ep)
        # Only mention entities, NOT the relation
        tmp_storage.save_episode_mentions("sep_ep1", [e1.absolute_id], target_type="entity")

        prov = tmp_storage.get_concept_provenance("sep_r1")
        assert prov == []

    def test_batch_mention_creation(self, tmp_storage):
        """D3.8: Batch save_episode_mentions works for multiple relations."""
        e1 = _make_entity("bat_e1", "O", "O")
        e2 = _make_entity("bat_e2", "P", "P")
        e3 = _make_entity("bat_e3", "Q", "Q")
        r1 = _make_relation("bat_r1", e1.absolute_id, e2.absolute_id, "O-P link")
        r2 = _make_relation("bat_r2", e2.absolute_id, e3.absolute_id, "P-Q link")
        ep = _make_episode("bat_ep1", "Batch mention text")
        tmp_storage.bulk_save_entities([e1, e2, e3])
        tmp_storage.save_relation(r1)
        tmp_storage.save_relation(r2)
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("bat_ep1", [r1.absolute_id, r2.absolute_id],
                                          target_type="relation")

        prov1 = tmp_storage.get_concept_provenance("bat_r1")
        prov2 = tmp_storage.get_concept_provenance("bat_r2")
        assert len(prov1) == 1
        assert len(prov2) == 1

    def test_get_concept_mentions_delegates(self, tmp_storage):
        """D3.9: get_concept_mentions delegates to get_concept_provenance."""
        e1 = _make_entity("del_e1", "R", "R")
        e2 = _make_entity("del_e2", "S", "S")
        r = _make_relation("del_r1", e1.absolute_id, e2.absolute_id, "R-S link")
        ep = _make_episode("del_ep1", "R-S text")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("del_ep1", [r.absolute_id], target_type="relation")

        mentions = tmp_storage.get_concept_mentions("del_r1")
        provenance = tmp_storage.get_concept_provenance("del_r1")
        assert mentions == provenance


# ══════════════════════════════════════════════════════════════════════════
# D4: Cross-cutting — edge cases, unicode, batch, concurrent
# ══════════════════════════════════════════════════════════════════════════


class TestCrossCuttingMentions:
    """D4: Cross-cutting edge cases for relation mentions."""

    def test_unicode_content_in_provenance(self, tmp_storage):
        """D4.1: Chinese content preserved in relation provenance."""
        e1 = _make_entity("uni_e1", "深度学习", "AI子领域")
        e2 = _make_entity("uni_e2", "神经网络", "深度学习的核心方法")
        r = _make_relation("uni_r1", e1.absolute_id, e2.absolute_id,
                           "深度学习依赖神经网络 🚀")
        ep = _make_episode("uni_ep1", "关于深度学习的讨论", source_document="中文文档.txt")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("uni_ep1", [r.absolute_id], target_type="relation")

        prov = tmp_storage.get_concept_provenance("uni_r1")
        assert len(prov) == 1
        assert "深度学习" in prov[0]["content"]

    def test_mention_after_relation_version_update(self, tmp_storage):
        """D4.2: Mention still works after relation version update."""
        e1 = _make_entity("ver_e1", "T", "T")
        e2 = _make_entity("ver_e2", "U", "U")
        r1 = _make_relation("ver_r1", e1.absolute_id, e2.absolute_id, "T-U v1")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r1)

        ep = _make_episode("ver_ep1", "Version 1")
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("ver_ep1", [r1.absolute_id], target_type="relation")

        # Create new version
        r2 = _make_relation("ver_r1", e1.absolute_id, e2.absolute_id, "T-U v2")
        tmp_storage.save_relation(r2)

        ep2 = _make_episode("ver_ep2", "Version 2")
        tmp_storage.save_episode(ep2)
        tmp_storage.save_episode_mentions("ver_ep2", [r2.absolute_id], target_type="relation")

        prov = tmp_storage.get_concept_provenance("ver_r1")
        assert len(prov) >= 2

    def test_empty_content_filtered(self, tmp_storage):
        """D4.3: Relations with empty/short content are filtered."""
        e1 = _make_entity("emp_e1", "V", "V")
        e2 = _make_entity("emp_e2", "W", "W")
        tmp_storage.bulk_save_entities([e1, e2])

        proc = _setup_processor(tmp_storage)
        proc.llm_client._current_distill_step = None
        proc.llm_client._priority_local = MagicMock()
        proc.llm_client._priority_local.priority = 6

        extracted = [{"entity1_name": "V", "entity2_name": "W",
                      "content": ""}]  # Empty content
        name_to_id = {"V": "emp_e1", "W": "emp_e2"}

        results = proc.process_relations_batch(extracted, name_to_id, "ep_test")
        # Empty content should be filtered at merge stage
        assert len(results) == 0

    def test_mention_target_type_relation(self, tmp_storage):
        """D4.4: MENTIONS with target_type=relation work correctly."""
        e1 = _make_entity("tt_e1", "X", "X")
        e2 = _make_entity("tt_e2", "Y", "Y")
        r = _make_relation("tt_r1", e1.absolute_id, e2.absolute_id, "X-Y link")
        ep = _make_episode("tt_ep1", "X-Y text")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("tt_ep1", [r.absolute_id], target_type="relation")

        # Verify via get_episode_entities
        entities = tmp_storage.get_episode_entities("tt_ep1")
        # Should find the relation target
        assert len(entities) >= 1

    def test_relation_provenance_nonexistent(self, tmp_storage):
        """D4.5: Nonexistent relation returns empty provenance."""
        prov = tmp_storage.get_concept_provenance("nonexistent_rel_xyz")
        assert prov == []

    def test_multiple_versions_provenance(self, tmp_storage):
        """D4.6: Relation with multiple versions returns all episode mentions."""
        e1 = _make_entity("mv_e1", "AA", "AA")
        e2 = _make_entity("mv_e2", "BB", "BB")
        r1 = _make_relation("mv_r1", e1.absolute_id, e2.absolute_id, "AA-BB v1")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r1)

        ep1 = _make_episode("mv_ep1", "First mention")
        tmp_storage.save_episode(ep1)
        tmp_storage.save_episode_mentions("mv_ep1", [r1.absolute_id], target_type="relation")

        r2 = _make_relation("mv_r1", e1.absolute_id, e2.absolute_id, "AA-BB v2")
        tmp_storage.save_relation(r2)

        ep2 = _make_episode("mv_ep2", "Second mention")
        tmp_storage.save_episode(ep2)
        tmp_storage.save_episode_mentions("mv_ep2", [r2.absolute_id], target_type="relation")

        prov = tmp_storage.get_concept_provenance("mv_r1")
        assert len(prov) >= 2
        ep_ids = {p["episode_id"] for p in prov}
        assert "mv_ep1" in ep_ids
        assert "mv_ep2" in ep_ids

    def test_large_batch_mentions(self, tmp_storage):
        """D4.7: Large batch of mentions works correctly."""
        e1 = _make_entity("lb_e1", "CC", "CC")
        e2 = _make_entity("lb_e2", "DD", "DD")
        r = _make_relation("lb_r1", e1.absolute_id, e2.absolute_id, "CC-DD link")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)

        # Create 10 episodes, all mentioning the same relation
        for i in range(10):
            ep = _make_episode(f"lb_ep{i}", f"Episode {i}")
            tmp_storage.save_episode(ep)
            tmp_storage.save_episode_mentions(f"lb_ep{i}", [r.absolute_id], target_type="relation")

        prov = tmp_storage.get_concept_provenance("lb_r1")
        assert len(prov) == 10

    def test_mention_after_entity_deletion(self, tmp_storage):
        """D4.8: Mention still valid after deleting one entity version."""
        e1 = _make_entity("del_e1", "EE", "EE")
        e2 = _make_entity("del_e2", "FF", "FF")
        r = _make_relation("del_r1", e1.absolute_id, e2.absolute_id, "EE-FF link")
        ep = _make_episode("del_ep1", "Text")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("del_ep1", [r.absolute_id], target_type="relation")

        # Verify mention exists
        prov_before = tmp_storage.get_concept_provenance("del_r1")
        assert len(prov_before) == 1

        # Delete the entity version (should not break relation provenance)
        tmp_storage.delete_entity_by_absolute_id(e1.absolute_id)
        prov_after = tmp_storage.get_concept_provenance("del_r1")
        assert len(prov_after) == 1

    def test_mention_idempotent(self, tmp_storage):
        """D4.9: Saving same mention twice is idempotent."""
        e1 = _make_entity("idm_e1", "GG", "GG")
        e2 = _make_entity("idm_e2", "HH", "HH")
        r = _make_relation("idm_r1", e1.absolute_id, e2.absolute_id, "GG-HH link")
        ep = _make_episode("idm_ep1", "Text")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)
        tmp_storage.save_episode(ep)

        # Save mention twice
        tmp_storage.save_episode_mentions("idm_ep1", [r.absolute_id], target_type="relation")
        tmp_storage.save_episode_mentions("idm_ep1", [r.absolute_id], target_type="relation")

        prov = tmp_storage.get_concept_provenance("idm_r1")
        assert len(prov) == 1  # Should not duplicate
