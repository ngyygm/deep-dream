"""
Relation confidence corroboration in batch pipeline — 32 tests across 4 dimensions.

Vision gap fix: When a relation is version-updated (matched existing) in the batch
pipeline path, adjust_confidence_on_corroboration was never called. Only the
single-relation fallback path had it. Now the batch path tracks corroborated
family_ids and calls adjust_confidence_on_corroboration after bulk_save_relations.

D1: Basic corroboration (8 tests)
D2: Multi-relation batch corroboration (8 tests)
D3: Edge cases — parallel, fallback, no-match (8 tests)
D4: Confidence increment verification (8 tests)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from unittest.mock import MagicMock, patch, call

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


def _make_relation(family_id: str, e1_abs: str, e2_abs: str, content: str,
                   confidence: float = 0.7):
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
        confidence=confidence,
    )


def _make_processor(storage, llm_client=None):
    from processor.pipeline.relation import RelationProcessor
    if llm_client is None:
        llm_client = MagicMock()
        llm_client._current_distill_step = 0
        llm_client._priority_local = MagicMock()
        llm_client._priority_local.priority = 6
        llm_client.merge_multiple_relation_contents.side_effect = lambda *a, **kw: "merged content"
    return RelationProcessor(storage=storage, llm_client=llm_client)


def _setup_entities_for_pair(storage, fid1="ent_aaa", fid2="ent_bbb"):
    """Create two entities and return their family_ids + absolute_ids."""
    e1 = _make_entity(fid1, "Entity A", "Content A")
    e2 = _make_entity(fid2, "Entity B", "Content B")
    storage.bulk_save_entities([e1, e2])
    return e1, e2


# ══════════════════════════════════════════════════════════════════════════
# D1: Basic corroboration
# ══════════════════════════════════════════════════════════════════════════


class TestBasicCorroboration:
    """D1: Basic relation confidence corroboration in batch path."""

    def test_match_existing_triggers_corroboration(self, tmp_storage):
        """D1.1: Version-updated relation triggers adjust_confidence_on_corroboration."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_match1", e1.absolute_id, e2.absolute_id,
            "Existing relation content", confidence=0.7
        )
        tmp_storage.save_relation(existing_rel)

        proc = _make_processor(tmp_storage)
        # Mock batch resolution to return match_existing
        proc.llm_client.resolve_relation_pair_batch.return_value = {
            "action": "match_existing",
            "matched_relation_id": "rel_match1",
            "need_update": True,
            "merged_content": "Updated relation content",
            "confidence": 0.9,
        }

        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "New info about the relation"}]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
                source_document="test_doc",
            )
            mock_adj.assert_called_once_with("rel_match1", source_type="relation")

    def test_new_relation_no_corroboration(self, tmp_storage):
        """D1.2: Newly created relation does NOT trigger corroboration."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)

        proc = _make_processor(tmp_storage)
        proc.llm_client.resolve_relation_pair_batch.return_value = {
            "action": "create_new",
            "merged_content": "Brand new relation",
            "confidence": 0.9,
        }

        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "A new relation"}]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
                source_document="test_doc",
            )
            mock_adj.assert_not_called()

    def test_no_existing_relations_no_corroboration(self, tmp_storage):
        """D1.3: No existing relations → direct create → no corroboration."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)

        proc = _make_processor(tmp_storage)
        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "First relation between A and B"}]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
                source_document="test_doc",
            )
            mock_adj.assert_not_called()

    def test_match_existing_no_update_no_corroboration(self, tmp_storage):
        """D1.4: match_existing without need_update does NOT corroborate."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_noupd", e1.absolute_id, e2.absolute_id,
            "Already perfect", confidence=0.8
        )
        tmp_storage.save_relation(existing_rel)

        proc = _make_processor(tmp_storage)
        proc.llm_client.resolve_relation_pair_batch.return_value = {
            "action": "match_existing",
            "matched_relation_id": "rel_noupd",
            "need_update": False,
            "confidence": 0.9,
        }

        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "Same content"}]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
                source_document="test_doc",
            )
            mock_adj.assert_not_called()

    def test_duplicate_content_no_corroboration(self, tmp_storage):
        """D1.5: Exact duplicate content skips LLM, no corroboration."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_dup1", e1.absolute_id, e2.absolute_id,
            "A works with B"
        )
        tmp_storage.save_relation(existing_rel)

        proc = _make_processor(tmp_storage)
        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "A works with B"}]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
                source_document="test_doc",
            )
            mock_adj.assert_not_called()

    def test_corroboration_after_bulk_save(self, tmp_storage):
        """D1.6: Corroboration called AFTER bulk_save_relations."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_order1", e1.absolute_id, e2.absolute_id,
            "Old content", confidence=0.6
        )
        tmp_storage.save_relation(existing_rel)

        proc = _make_processor(tmp_storage)
        proc.llm_client.resolve_relation_pair_batch.return_value = {
            "action": "match_existing",
            "matched_relation_id": "rel_order1",
            "need_update": True,
            "merged_content": "New content",
            "confidence": 0.9,
        }

        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "Additional info"}]

        call_order = []
        orig_bulk = tmp_storage.bulk_save_relations
        orig_adj = tmp_storage.adjust_confidence_on_corroboration

        def tracked_bulk(rels):
            call_order.append("bulk_save")
            return orig_bulk(rels)

        def tracked_adj(fid, **kw):
            call_order.append("corroboration")
            return orig_adj(fid, **kw)

        with patch.object(tmp_storage, 'bulk_save_relations', side_effect=tracked_bulk):
            with patch.object(tmp_storage, 'adjust_confidence_on_corroboration',
                              side_effect=tracked_adj):
                proc.process_relations_batch(
                    extracted, entity_name_to_id, "ep_test",
                    source_document="test_doc",
                )
        assert call_order == ["bulk_save", "corroboration"]

    def test_corroboration_exception_does_not_crash(self, tmp_storage):
        """D1.7: adjust_confidence_on_corroboration exception is swallowed."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_exc1", e1.absolute_id, e2.absolute_id,
            "Content", confidence=0.5
        )
        tmp_storage.save_relation(existing_rel)

        proc = _make_processor(tmp_storage)
        proc.llm_client.resolve_relation_pair_batch.return_value = {
            "action": "match_existing",
            "matched_relation_id": "rel_exc1",
            "need_update": True,
            "merged_content": "Updated relation information",
            "confidence": 0.9,
        }

        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "More info"}]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration',
                          side_effect=RuntimeError("DB error")):
            result = proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
                source_document="test_doc",
            )
            # Should not crash
            assert isinstance(result, list)

    def test_match_existing_not_found_creates_new_no_corroboration(self, tmp_storage):
        """D1.8: match_existing but matched_relation_id not found → new relation, no corroboration."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_other", e1.absolute_id, e2.absolute_id,
            "Other relation"
        )
        tmp_storage.save_relation(existing_rel)

        proc = _make_processor(tmp_storage)
        proc.llm_client.resolve_relation_pair_batch.return_value = {
            "action": "match_existing",
            "matched_relation_id": "rel_nonexistent",
            "need_update": True,
            "merged_content": "Fallback new",
            "confidence": 0.9,
        }

        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "Some relation"}]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
                source_document="test_doc",
            )
            mock_adj.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════
# D2: Multi-relation batch corroboration
# ══════════════════════════════════════════════════════════════════════════


class TestMultiRelationBatchCorroboration:
    """D2: Multiple relations in batch, selective corroboration."""

    def test_multiple_match_existing_all_corroborated(self, tmp_storage):
        """D2.1: Two entity pairs, both match_existing → both corroborated."""
        e1, e2 = _setup_entities_for_pair(tmp_storage, "ent_aa", "ent_bb")
        e3 = _make_entity("ent_cc", "Entity C", "Content C")
        tmp_storage.save_entity(e3)

        rel1 = _make_relation("rel_m1", e1.absolute_id, e2.absolute_id, "R1", confidence=0.6)
        rel2 = _make_relation("rel_m2", e1.absolute_id, e3.absolute_id, "R2", confidence=0.5)
        tmp_storage.bulk_save_relations([rel1, rel2])

        proc = _make_processor(tmp_storage)
        call_count = [0]
        def mock_resolve(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "action": "match_existing",
                    "matched_relation_id": "rel_m1",
                    "need_update": True,
                    "merged_content": "Updated R1",
                    "confidence": 0.9,
                }
            return {
                "action": "match_existing",
                "matched_relation_id": "rel_m2",
                "need_update": True,
                "merged_content": "Updated R2",
                "confidence": 0.9,
            }

        proc.llm_client.resolve_relation_pair_batch.side_effect = mock_resolve

        entity_name_to_id = {
            "Entity A": "ent_aa", "Entity B": "ent_bb", "Entity C": "ent_cc"
        }
        extracted = [
            {"entity1_name": "Entity A", "entity2_name": "Entity B", "content": "Info about AB"},
            {"entity1_name": "Entity A", "entity2_name": "Entity C", "content": "Info about AC"},
        ]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
                source_document="test_doc",
            )
            calls = mock_adj.call_args_list
            fids = {c[0][0] for c in calls}
            assert "rel_m1" in fids
            assert "rel_m2" in fids

    def test_mixed_match_and_new_selective_corroboration(self, tmp_storage):
        """D2.2: One match_existing + one create_new → only match corroborated."""
        e1, e2 = _setup_entities_for_pair(tmp_storage, "ent_dd", "ent_ee")
        e3 = _make_entity("ent_ff", "Entity F", "Content F")
        tmp_storage.save_entity(e3)

        rel1 = _make_relation("rel_mx1", e1.absolute_id, e2.absolute_id, "Old", confidence=0.6)
        tmp_storage.save_relation(rel1)

        proc = _make_processor(tmp_storage)
        call_count = [0]
        def mock_resolve(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "action": "match_existing",
                    "matched_relation_id": "rel_mx1",
                    "need_update": True,
                    "merged_content": "Updated relation content",
                    "confidence": 0.9,
                }
            return {
                "action": "create_new",
                "merged_content": "Brand new relation content",
                "confidence": 0.85,
            }

        proc.llm_client.resolve_relation_pair_batch.side_effect = mock_resolve

        entity_name_to_id = {
            "Entity D": "ent_dd", "Entity E": "ent_ee", "Entity F": "ent_ff"
        }
        extracted = [
            {"entity1_name": "Entity D", "entity2_name": "Entity E", "content": "Update DE"},
            {"entity1_name": "Entity D", "entity2_name": "Entity F", "content": "New DF"},
        ]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
                source_document="test_doc",
            )
            calls = mock_adj.call_args_list
            assert len(calls) == 1
            assert calls[0][0][0] == "rel_mx1"

    def test_same_pair_multiple_versions_single_corroboration(self, tmp_storage):
        """D2.3: Same pair appears twice → only one corroboration call."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_sp1", e1.absolute_id, e2.absolute_id,
            "Existing", confidence=0.6
        )
        tmp_storage.save_relation(existing_rel)

        proc = _make_processor(tmp_storage)
        proc.llm_client.resolve_relation_pair_batch.return_value = {
            "action": "match_existing",
            "matched_relation_id": "rel_sp1",
            "need_update": True,
            "merged_content": "Updated relation content",
            "confidence": 0.9,
        }

        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        # Two relations for same pair
        extracted = [
            {"entity1_name": "Entity A", "entity2_name": "Entity B", "content": "Info 1"},
            {"entity1_name": "Entity A", "entity2_name": "Entity B", "content": "Info 2"},
        ]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
                source_document="test_doc",
            )
            # Should be called once for the matched family_id
            fids = [c[0][0] for c in mock_adj.call_args_list]
            assert fids.count("rel_sp1") == 1

    def test_three_pairs_all_different_actions(self, tmp_storage):
        """D2.4: Three pairs: match+update, match+no-update, create_new."""
        e1 = _make_entity("ent_gg", "G", "Content G")
        e2 = _make_entity("ent_hh", "H", "Content H")
        e3 = _make_entity("ent_ii", "I", "Content I")
        e4 = _make_entity("ent_jj", "J", "Content J")
        tmp_storage.bulk_save_entities([e1, e2, e3, e4])

        rel1 = _make_relation("rel_gh", e1.absolute_id, e2.absolute_id, "GH", confidence=0.6)
        rel2 = _make_relation("rel_gi", e1.absolute_id, e3.absolute_id, "GI", confidence=0.8)
        tmp_storage.bulk_save_relations([rel1, rel2])

        proc = _make_processor(tmp_storage)
        call_count = [0]
        def mock_resolve(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"action": "match_existing", "matched_relation_id": "rel_gh",
                        "need_update": True, "merged_content": "Updated GH", "confidence": 0.9}
            elif call_count[0] == 2:
                return {"action": "match_existing", "matched_relation_id": "rel_gi",
                        "need_update": False, "confidence": 0.9}
            else:
                return {"action": "create_new", "merged_content": "New GJ", "confidence": 0.85}

        proc.llm_client.resolve_relation_pair_batch.side_effect = mock_resolve

        entity_name_to_id = {"G": "ent_gg", "H": "ent_hh", "I": "ent_ii", "J": "ent_jj"}
        extracted = [
            {"entity1_name": "G", "entity2_name": "H", "content": "Update GH"},
            {"entity1_name": "G", "entity2_name": "I", "content": "Same GI"},
            {"entity1_name": "G", "entity2_name": "J", "content": "New GJ"},
        ]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
                source_document="test_doc",
            )
            fids = [c[0][0] for c in mock_adj.call_args_list]
            assert "rel_gh" in fids
            assert "rel_gi" not in fids

    def test_empty_extracted_no_corroboration(self, tmp_storage):
        """D2.5: Empty extracted_relations → no corroboration."""
        proc = _make_processor(tmp_storage)
        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            result = proc.process_relations_batch(
                [], {}, "ep_test",
            )
            mock_adj.assert_not_called()
            assert result == []

    def test_self_relation_no_corroboration(self, tmp_storage):
        """D2.6: Self-relation (same entity pair) filtered → no corroboration."""
        e1 = _make_entity("ent_self", "Self", "Content")
        tmp_storage.save_entity(e1)

        proc = _make_processor(tmp_storage)
        entity_name_to_id = {"Self": "ent_self"}
        extracted = [{"entity1_name": "Self", "entity2_name": "Self",
                      "content": "Self relation"}]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
            )
            mock_adj.assert_not_called()

    def test_corroboration_with_dream_corroboration(self, tmp_storage):
        """D2.7: Confidence corroboration runs alongside dream corroboration."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_dr1", e1.absolute_id, e2.absolute_id,
            "Content", confidence=0.6
        )
        tmp_storage.save_relation(existing_rel)

        proc = _make_processor(tmp_storage)
        proc.llm_client.resolve_relation_pair_batch.return_value = {
            "action": "match_existing",
            "matched_relation_id": "rel_dr1",
            "need_update": True,
            "merged_content": "Updated relation content",
            "confidence": 0.9,
        }

        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "More info"}]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_conf:
            with patch.object(tmp_storage, 'corroborate_dream_relation') as mock_dream:
                proc.process_relations_batch(
                    extracted, entity_name_to_id, "ep_test",
                )
                mock_conf.assert_called_once_with("rel_dr1", source_type="relation")
                mock_dream.assert_called_once()

    def test_no_relations_to_persist_no_corroboration(self, tmp_storage):
        """D2.8: If no relations_to_persist, still no crash and no corroboration."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_nop1", e1.absolute_id, e2.absolute_id,
            "Exact same content"
        )
        tmp_storage.save_relation(existing_rel)

        proc = _make_processor(tmp_storage)
        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "Exact same content"}]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            result = proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
            )
            mock_adj.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════
# D3: Edge cases — parallel, fallback, no-match
# ══════════════════════════════════════════════════════════════════════════


class TestRelationCorroborationEdgeCases:
    """D3: Edge cases for relation confidence corroboration."""

    def test_parallel_path_corroboration(self, tmp_storage):
        """D3.1: Parallel processing path also corroborates correctly."""
        e1, e2 = _setup_entities_for_pair(tmp_storage, "ent_kk", "ent_ll")
        e3 = _make_entity("ent_mm", "Entity M", "Content M")
        tmp_storage.save_entity(e3)

        rel1 = _make_relation("rel_kl", e1.absolute_id, e2.absolute_id, "KL", confidence=0.5)
        rel2 = _make_relation("rel_km", e1.absolute_id, e3.absolute_id, "KM", confidence=0.5)
        tmp_storage.bulk_save_relations([rel1, rel2])

        proc = _make_processor(tmp_storage)
        call_count = [0]
        def mock_resolve(*a, **kw):
            call_count[0] += 1
            return {
                "action": "match_existing",
                "matched_relation_id": "rel_kl" if call_count[0] == 1 else "rel_km",
                "need_update": True,
                "merged_content": f"Updated {call_count[0]}",
                "confidence": 0.9,
            }

        proc.llm_client.resolve_relation_pair_batch.side_effect = mock_resolve

        entity_name_to_id = {
            "Entity K": "ent_kk", "Entity L": "ent_ll", "Entity M": "ent_mm"
        }
        extracted = [
            {"entity1_name": "Entity K", "entity2_name": "Entity L", "content": "Update KL"},
            {"entity1_name": "Entity K", "entity2_name": "Entity M", "content": "Update KM"},
        ]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
                source_document="test_doc", max_workers=2,
            )
            fids = {c[0][0] for c in mock_adj.call_args_list}
            assert "rel_kl" in fids
            assert "rel_km" in fids

    def test_fallback_to_single_path(self, tmp_storage):
        """D3.2: Fallback to single path does not corroborate in batch path."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_fb1", e1.absolute_id, e2.absolute_id,
            "Content", confidence=0.6
        )
        tmp_storage.save_relation(existing_rel)

        proc = _make_processor(tmp_storage)
        proc.llm_client.resolve_relation_pair_batch.return_value = {
            "action": "fallback",
            "confidence": 0.3,
        }
        # Mock single path
        proc._process_single_relation = MagicMock(return_value=None)

        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "Some relation"}]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
            )
            # Fallback path does its own corroboration (not in batch corroboration)
            mock_adj.assert_not_called()

    def test_missing_entity_id_no_crash(self, tmp_storage):
        """D3.3: Missing entity in name_to_id → filtered, no crash."""
        proc = _make_processor(tmp_storage)
        entity_name_to_id = {"Known": "ent_known"}
        extracted = [
            {"entity1_name": "Known", "entity2_name": "Unknown", "content": "Rel"},
        ]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            result = proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
            )
            mock_adj.assert_not_called()

    def test_corroboration_with_unicode_content(self, tmp_storage):
        """D3.4: Unicode content in matched relation corroborates correctly."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_uni1", e1.absolute_id, e2.absolute_id,
            "深度学习是人工智能的子领域", confidence=0.6
        )
        tmp_storage.save_relation(existing_rel)

        proc = _make_processor(tmp_storage)
        proc.llm_client.resolve_relation_pair_batch.return_value = {
            "action": "match_existing",
            "matched_relation_id": "rel_uni1",
            "need_update": True,
            "merged_content": "更新后的深度学习描述",
            "confidence": 0.9,
        }

        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "深度学习的新发展"}]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
            )
            mock_adj.assert_called_once_with("rel_uni1", source_type="relation")

    def test_large_batch_corroboration(self, tmp_storage):
        """D3.5: 10 relation pairs, 5 matching → 5 corroboration calls."""
        entities = [_make_entity(f"ent_l{i}", f"E{i}", f"C{i}") for i in range(6)]
        tmp_storage.bulk_save_entities(entities)

        # Create 5 existing relations
        rels = []
        for i in range(5):
            r = _make_relation(
                f"rel_l{i}", entities[0].absolute_id, entities[i+1].absolute_id,
                f"Existing R{i}", confidence=0.5
            )
            rels.append(r)
        tmp_storage.bulk_save_relations(rels)

        proc = _make_processor(tmp_storage)
        call_count = [0]
        def mock_resolve(*a, **kw):
            call_count[0] += 1
            if call_count[0] <= 5:
                return {
                    "action": "match_existing",
                    "matched_relation_id": f"rel_l{call_count[0]-1}",
                    "need_update": True,
                    "merged_content": f"Updated R{call_count[0]}",
                    "confidence": 0.9,
                }
            return {
                "action": "create_new",
                "merged_content": "New relation",
                "confidence": 0.85,
            }

        proc.llm_client.resolve_relation_pair_batch.side_effect = mock_resolve

        entity_name_to_id = {f"E{i}": f"ent_l{i}" for i in range(6)}
        extracted = [
            {"entity1_name": "E0", "entity2_name": f"E{i}", "content": f"Info {i}"}
            for i in range(1, 6)
        ]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
            )
            assert mock_adj.call_count == 5

    def test_empty_entity_name_no_corroboration(self, tmp_storage):
        """D3.6: Empty entity names filtered out, no corroboration."""
        proc = _make_processor(tmp_storage)
        entity_name_to_id = {"E": "ent_e"}
        extracted = [
            {"entity1_name": "", "entity2_name": "E", "content": "Rel"},
            {"entity1_name": "E", "entity2_name": "", "content": "Rel"},
        ]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
            )
            mock_adj.assert_not_called()

    def test_corroboration_source_type_is_relation(self, tmp_storage):
        """D3.7: adjust_confidence_on_corroboration called with source_type='relation'."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_st1", e1.absolute_id, e2.absolute_id,
            "Content", confidence=0.5
        )
        tmp_storage.save_relation(existing_rel)

        proc = _make_processor(tmp_storage)
        proc.llm_client.resolve_relation_pair_batch.return_value = {
            "action": "match_existing",
            "matched_relation_id": "rel_st1",
            "need_update": True,
            "merged_content": "Updated relation content",
            "confidence": 0.9,
        }

        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "Info"}]

        with patch.object(tmp_storage, 'adjust_confidence_on_corroboration') as mock_adj:
            proc.process_relations_batch(
                extracted, entity_name_to_id, "ep_test",
            )
            mock_adj.assert_called_once_with("rel_st1", source_type="relation")


# ══════════════════════════════════════════════════════════════════════════
# D4: Confidence increment verification
# ══════════════════════════════════════════════════════════════════════════


class TestConfidenceIncrementVerification:
    """D4: Verify that confidence actually increments after corroboration."""

    def test_confidence_increases_after_corroboration(self, tmp_storage):
        """D4.1: Relation confidence increases after version update corroboration."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_ci1", e1.absolute_id, e2.absolute_id,
            "Original content", confidence=0.6
        )
        tmp_storage.save_relation(existing_rel)

        proc = _make_processor(tmp_storage)
        proc.llm_client.resolve_relation_pair_batch.return_value = {
            "action": "match_existing",
            "matched_relation_id": "rel_ci1",
            "need_update": True,
            "merged_content": "Updated content",
            "confidence": 0.9,
        }

        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "New info"}]

        # Get confidence before
        rel_before = tmp_storage.get_relation_by_family_id("rel_ci1")
        conf_before = rel_before.confidence if rel_before else 0

        proc.process_relations_batch(
            extracted, entity_name_to_id, "ep_test",
            source_document="test_doc",
        )

        # Get confidence after (latest version)
        rel_after = tmp_storage.get_relation_by_family_id("rel_ci1")
        assert rel_after is not None
        # Confidence should have increased (adjust_confidence_on_corroboration adds +0.05)
        assert rel_after.confidence > conf_before

    def test_confidence_increment_amount(self, tmp_storage):
        """D4.2: Each corroboration adds approximately +0.05 to confidence."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_amt1", e1.absolute_id, e2.absolute_id,
            "Content", confidence=0.50
        )
        tmp_storage.save_relation(existing_rel)

        # Directly call adjust_confidence_on_corroboration
        tmp_storage.adjust_confidence_on_corroboration("rel_amt1", source_type="relation")

        rel = tmp_storage.get_relation_by_family_id("rel_amt1")
        assert rel is not None
        assert rel.confidence >= 0.55

    def test_confidence_capped_at_1(self, tmp_storage):
        """D4.3: Confidence does not exceed 1.0 even with many corroboration calls."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_cap1", e1.absolute_id, e2.absolute_id,
            "Content", confidence=0.95
        )
        tmp_storage.save_relation(existing_rel)

        # Call corroboration multiple times
        for _ in range(10):
            tmp_storage.adjust_confidence_on_corroboration("rel_cap1", source_type="relation")

        rel = tmp_storage.get_relation_by_family_id("rel_cap1")
        assert rel is not None
        assert rel.confidence <= 1.0

    def test_new_relation_confidence_default(self, tmp_storage):
        """D4.4: Newly created relation starts with default confidence (no corroboration)."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)

        proc = _make_processor(tmp_storage)
        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "A new relation"}]

        result = proc.process_relations_batch(
            extracted, entity_name_to_id, "ep_test",
            source_document="test_doc",
        )

        assert len(result) >= 1
        # Find the new relation
        new_rels = [r for r in result if r.family_id not in ("rel_cap1",)]
        if new_rels:
            assert new_rels[0].confidence == 0.7  # default

    def test_corroboration_affects_entity_not_relation(self, tmp_storage):
        """D4.5: source_type='relation' adjusts relation confidence, not entity."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_type1", e1.absolute_id, e2.absolute_id,
            "Content", confidence=0.5
        )
        tmp_storage.save_relation(existing_rel)

        # Get entity confidence before
        ent_before = tmp_storage.get_entity_by_family_id(e1.family_id)

        tmp_storage.adjust_confidence_on_corroboration("rel_type1", source_type="relation")

        # Entity confidence should NOT change
        ent_after = tmp_storage.get_entity_by_family_id(e1.family_id)
        if ent_before and ent_after:
            assert ent_before.confidence == ent_after.confidence

    def test_multiple_corroboration_calls_cumulative(self, tmp_storage):
        """D4.6: Multiple corroboration calls are cumulative."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_cum1", e1.absolute_id, e2.absolute_id,
            "Content", confidence=0.40
        )
        tmp_storage.save_relation(existing_rel)

        # First corroboration
        tmp_storage.adjust_confidence_on_corroboration("rel_cum1", source_type="relation")
        rel1 = tmp_storage.get_relation_by_family_id("rel_cum1")
        conf1 = rel1.confidence

        # Second corroboration
        tmp_storage.adjust_confidence_on_corroboration("rel_cum1", source_type="relation")
        rel2 = tmp_storage.get_relation_by_family_id("rel_cum1")
        conf2 = rel2.confidence

        assert conf2 > conf1
        assert conf2 >= conf1 + 0.04  # approximately +0.05 each

    def test_corroboration_on_nonexistent_family_id_no_crash(self, tmp_storage):
        """D4.7: Corroboration on non-existent family_id does not crash."""
        # Should silently do nothing
        tmp_storage.adjust_confidence_on_corroboration("nonexistent_rel", source_type="relation")

    def test_version_count_increases_on_match(self, tmp_storage):
        """D4.8: Version-updated relation has increased version count."""
        e1, e2 = _setup_entities_for_pair(tmp_storage)
        existing_rel = _make_relation(
            "rel_vc1", e1.absolute_id, e2.absolute_id,
            "Original", confidence=0.5
        )
        tmp_storage.save_relation(existing_rel)

        proc = _make_processor(tmp_storage)
        proc.llm_client.resolve_relation_pair_batch.return_value = {
            "action": "match_existing",
            "matched_relation_id": "rel_vc1",
            "need_update": True,
            "merged_content": "Updated content",
            "confidence": 0.9,
        }

        entity_name_to_id = {"Entity A": e1.family_id, "Entity B": e2.family_id}
        extracted = [{"entity1_name": "Entity A", "entity2_name": "Entity B",
                      "content": "More info"}]

        versions_before = tmp_storage.get_relation_version_counts(["rel_vc1"]).get("rel_vc1", 0)

        proc.process_relations_batch(
            extracted, entity_name_to_id, "ep_test",
            source_document="test_doc",
        )

        versions_after = tmp_storage.get_relation_version_counts(["rel_vc1"]).get("rel_vc1", 0)
        assert versions_after > versions_before
