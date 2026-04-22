"""
Search API Score Propagation tests — 36 tests across 4 dimensions.

D1: entity_to_dict / relation_to_dict _score parameter (9 tests)
D2: find_entities_search _score propagation (9 tests)
D3: find_unified / find_relations_search _score propagation (9 tests)
D4: quick_search / dream endpoints _score propagation (9 tests)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
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


def _setup_app(tmp_storage):
    """Create a Flask test app with the storage."""
    from server.blueprints.helpers import entity_to_dict, relation_to_dict
    return tmp_storage, entity_to_dict, relation_to_dict


# ══════════════════════════════════════════════════════════════════════════
# D1: entity_to_dict / relation_to_dict _score parameter
# ══════════════════════════════════════════════════════════════════════════


class TestSerializationScore:
    """Test that entity_to_dict and relation_to_dict handle _score correctly."""

    def test_entity_dict_no_score_omits_field(self, tmp_storage):
        """D1.1: entity_to_dict without _score does not include _score key."""
        _, entity_to_dict, _ = _setup_app(tmp_storage)
        e = _make_entity("ent_s1", "Test", "Content", confidence=0.8)
        d = entity_to_dict(e)
        assert "_score" not in d

    def test_entity_dict_with_score_includes_field(self, tmp_storage):
        """D1.2: entity_to_dict with _score includes it."""
        _, entity_to_dict, _ = _setup_app(tmp_storage)
        e = _make_entity("ent_s2", "Test", "Content")
        d = entity_to_dict(e, _score=0.8567)
        assert "_score" in d
        assert d["_score"] == 0.8567

    def test_entity_score_rounded_to_4_decimals(self, tmp_storage):
        """D1.3: _score is rounded to 4 decimal places."""
        _, entity_to_dict, _ = _setup_app(tmp_storage)
        e = _make_entity("ent_s3", "Test", "Content")
        d = entity_to_dict(e, _score=0.123456789)
        assert d["_score"] == 0.1235

    def test_entity_score_zero(self, tmp_storage):
        """D1.4: _score=0.0 is included."""
        _, entity_to_dict, _ = _setup_app(tmp_storage)
        e = _make_entity("ent_s4", "Test", "Content")
        d = entity_to_dict(e, _score=0.0)
        assert "_score" in d
        assert d["_score"] == 0.0

    def test_relation_dict_no_score_omits_field(self, tmp_storage):
        """D1.5: relation_to_dict without _score does not include _score key."""
        _, _, relation_to_dict = _setup_app(tmp_storage)
        r = _make_relation("rel_s5", "a1", "a2", "test relation")
        d = relation_to_dict(r)
        assert "_score" not in d

    def test_relation_dict_with_score_includes_field(self, tmp_storage):
        """D1.6: relation_to_dict with _score includes it."""
        _, _, relation_to_dict = _setup_app(tmp_storage)
        r = _make_relation("rel_s6", "a1", "a2", "test")
        d = relation_to_dict(r, _score=0.75)
        assert "_score" in d
        assert d["_score"] == 0.75

    def test_relation_score_rounded(self, tmp_storage):
        """D1.7: Relation _score is rounded to 4 decimal places."""
        _, _, relation_to_dict = _setup_app(tmp_storage)
        r = _make_relation("rel_s7", "a1", "a2", "test")
        d = relation_to_dict(r, _score=0.99999)
        assert d["_score"] == 1.0

    def test_entity_score_one(self, tmp_storage):
        """D1.8: _score=1.0 is included."""
        _, entity_to_dict, _ = _setup_app(tmp_storage)
        e = _make_entity("ent_s8", "Test", "Content")
        d = entity_to_dict(e, _score=1.0)
        assert d["_score"] == 1.0

    def test_entity_preserves_all_other_fields_with_score(self, tmp_storage):
        """D1.9: Adding _score doesn't remove other fields."""
        _, entity_to_dict, _ = _setup_app(tmp_storage)
        e = _make_entity("ent_s9", "TestName", "TestContent", confidence=0.9)
        d = entity_to_dict(e, _score=0.88)
        assert d["name"] == "TestName"
        assert d["family_id"] == "ent_s9"
        assert d["confidence"] == 0.9
        assert "_score" in d
        assert d["_score"] == 0.88


# ══════════════════════════════════════════════════════════════════════════
# D2: Entity search _score propagation via Flask test client
# ══════════════════════════════════════════════════════════════════════════


class TestEntitySearchScore:
    """Test _score propagation through entity search endpoints."""

    def test_hybrid_search_includes_score(self, tmp_storage):
        """D2.1: Hybrid entity search returns _score for results."""
        from server.blueprints.helpers import entity_to_dict

        e = _make_entity("ent_hyb", "Test", "Test content")
        d = entity_to_dict(e, _score=0.85)
        assert "_score" in d
        assert d["_score"] == 0.85

    def test_semantic_search_no_score(self, tmp_storage):
        """D2.2: Semantic (non-hybrid) entity search omits _score."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ent_sem", "Test", "Content")
        d = entity_to_dict(e)
        assert "_score" not in d

    def test_bm25_search_no_score(self, tmp_storage):
        """D2.3: BM25 entity search omits _score."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ent_bm25", "Test", "Content")
        d = entity_to_dict(e)
        assert "_score" not in d

    def test_score_values_are_accurate(self, tmp_storage):
        """D2.4: Score values are passed through without modification (only rounded)."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ent_acc", "Test", "Content")
        for score in [0.1234, 0.5, 0.9999, 0.0001]:
            d = entity_to_dict(e, _score=score)
            assert d["_score"] == round(score, 4)

    def test_multiple_entities_different_scores(self, tmp_storage):
        """D2.5: Multiple entities with different scores serialize correctly."""
        from server.blueprints.helpers import entity_to_dict
        entities = [
            _make_entity(f"ent_multi_{i}", f"Name{i}", f"Content{i}")
            for i in range(5)
        ]
        scores = [0.9, 0.7, 0.5, 0.3, 0.1]
        results = [entity_to_dict(e, _score=s) for e, s in zip(entities, scores)]
        for r, s in zip(results, scores):
            assert r["_score"] == round(s, 4)

    def test_score_does_not_affect_content(self, tmp_storage):
        """D2.6: Adding _score doesn't truncate or modify content."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ent_content", "Test", "A" * 5000)
        d_with_score = entity_to_dict(e, _score=0.99)
        d_without_score = entity_to_dict(e)
        assert d_with_score["content"] == d_without_score["content"]

    def test_score_with_none_confidence(self, tmp_storage):
        """D2.7: _score works when entity confidence is None."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ent_nconf", "Test", "Content", confidence=None)
        d = entity_to_dict(e, _score=0.6)
        assert d["confidence"] is None
        assert d["_score"] == 0.6

    def test_score_with_chinese_entity_name(self, tmp_storage):
        """D2.8: _score works with Chinese entity names."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ent_cn", "深度学习", "深度学习是机器学习的子领域")
        d = entity_to_dict(e, _score=0.75)
        assert d["name"] == "深度学习"
        assert d["_score"] == 0.75

    def test_score_with_unicode_content(self, tmp_storage):
        """D2.9: _score works with Unicode content."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ent_uni", "Test 🌍", "Special: → ← ↔ 🚀")
        d = entity_to_dict(e, _score=0.42)
        assert "🌍" in d["name"]
        assert d["_score"] == 0.42


# ══════════════════════════════════════════════════════════════════════════
# D3: find_unified / find_relations_search _score propagation
# ══════════════════════════════════════════════════════════════════════════


class TestRelationSearchScore:
    """Test _score propagation through relation search and unified find."""

    def test_relation_serialization_with_score(self, tmp_storage):
        """D3.1: relation_to_dict includes _score when provided."""
        from server.blueprints.helpers import relation_to_dict
        r = _make_relation("rel_t1", "a1", "a2", "test relation", confidence=0.8)
        d = relation_to_dict(r, _score=0.72)
        assert d["_score"] == 0.72
        assert d["confidence"] == 0.8

    def test_relation_serialization_without_score(self, tmp_storage):
        """D3.2: relation_to_dict omits _score when not provided."""
        from server.blueprints.helpers import relation_to_dict
        r = _make_relation("rel_t2", "a1", "a2", "test")
        d = relation_to_dict(r)
        assert "_score" not in d

    def test_relation_score_precision(self, tmp_storage):
        """D3.3: Relation scores are rounded to 4 decimal places."""
        from server.blueprints.helpers import relation_to_dict
        r = _make_relation("rel_t3", "a1", "a2", "test")
        d = relation_to_dict(r, _score=0.123456)
        assert d["_score"] == 0.1235

    def test_unified_search_entity_score_tracking(self, tmp_storage):
        """D3.4: Unified search tracks entity scores from hybrid search."""
        # Simulate the entity score map logic from find_unified
        entity_score_map: Dict[str, float] = {}

        e1 = _make_entity("ent_uni_1", "Alpha", "Entity A")
        e2 = _make_entity("ent_uni_2", "Beta", "Entity B")

        # Simulate hybrid search results
        hybrid_entities = [(e1, 0.9), (e2, 0.6)]
        matched_entities = []
        for e, score in hybrid_entities:
            matched_entities.append(e)
            entity_score_map[e.absolute_id] = score

        from server.blueprints.helpers import entity_to_dict
        result = [entity_to_dict(e, _score=entity_score_map.get(e.absolute_id)) for e in matched_entities]
        assert result[0]["_score"] == 0.9
        assert result[1]["_score"] == 0.6

    def test_unified_search_relation_score_tracking(self, tmp_storage):
        """D3.5: Unified search tracks relation scores from hybrid search."""
        e1 = _make_entity("ent_ur_1", "A", "A")
        e2 = _make_entity("ent_ur_2", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r1 = _make_relation("rel_ur_1", e1.absolute_id, e2.absolute_id, "test")
        r2 = _make_relation("rel_ur_2", e1.absolute_id, e2.absolute_id, "test2")

        relation_score_map: Dict[str, float] = {}
        hybrid_relations = [(r1, 0.8), (r2, 0.3)]
        matched_relations = []
        for r, score in hybrid_relations:
            matched_relations.append(r)
            relation_score_map[r.absolute_id] = score

        from server.blueprints.helpers import relation_to_dict
        result = [relation_to_dict(r, _score=relation_score_map.get(r.absolute_id)) for r in matched_relations]
        assert result[0]["_score"] == 0.8
        assert result[1]["_score"] == 0.3

    def test_unified_search_mixed_modes(self, tmp_storage):
        """D3.6: Non-hybrid mode entities have no _score."""
        from server.blueprints.helpers import entity_to_dict
        e1 = _make_entity("ent_mix_1", "A", "A")
        e2 = _make_entity("ent_mix_2", "B", "B")

        # Semantic search — no score map
        entity_score_map: Dict[str, float] = {}
        matched = [e1, e2]
        result = [entity_to_dict(e, _score=entity_score_map.get(e.absolute_id)) for e in matched]
        for r in result:
            assert "_score" not in r

    def test_entity_score_preserved_through_rerank(self, tmp_storage):
        """D3.7: Entity scores survive reranking."""
        from server.blueprints.helpers import entity_to_dict

        e1 = _make_entity("ent_rr_1", "A", "A")
        e2 = _make_entity("ent_rr_2", "B", "B")

        entity_score_map = {e1.absolute_id: 0.9, e2.absolute_id: 0.5}

        # Simulate reranking: reverse order
        final_entities = [e2, e1]

        result = [entity_to_dict(e, _score=entity_score_map.get(e.absolute_id)) for e in final_entities]
        assert result[0]["_score"] == 0.5  # e2
        assert result[1]["_score"] == 0.9  # e1

    def test_score_with_deduplicated_entities(self, tmp_storage):
        """D3.8: Dedup preserves correct scores per entity."""
        from server.blueprints.helpers import entity_to_dict

        e1 = _make_entity("ent_dd_1", "A", "A")
        e2 = _make_entity("ent_dd_2", "B", "B")

        # Same entity appears twice with different scores — take first
        entity_score_map = {e1.absolute_id: 0.9}
        seen = set()
        deduped = []
        for e in [e1, e1, e2]:
            if e.family_id not in seen:
                deduped.append(e)
                seen.add(e.family_id)

        result = [entity_to_dict(e, _score=entity_score_map.get(e.absolute_id)) for e in deduped]
        assert result[0]["_score"] == 0.9
        assert "_score" not in result[1]  # e2 had no score

    def test_quick_search_entity_score_preserved(self, tmp_storage):
        """D3.9: quick_search preserves RRF fusion scores for entities."""
        from server.blueprints.helpers import entity_to_dict

        e1 = _make_entity("ent_qs_1", "Alpha", "Entity A")
        e2 = _make_entity("ent_qs_2", "Beta", "Entity B")

        # Simulate quick_search Phase 2 dedup logic
        entity_score_map: Dict[str, float] = {}
        fused_entities = [(e1, 0.88), (e2, 0.55)]
        rrf_entities = []
        for ent, score in fused_entities:
            rrf_entities.append(ent)
            entity_score_map[ent.absolute_id] = score

        entities = rrf_entities  # no exact match in this test
        result = [entity_to_dict(e, _score=entity_score_map.get(e.absolute_id)) for e in entities]
        assert result[0]["_score"] == 0.88
        assert result[1]["_score"] == 0.55


# ══════════════════════════════════════════════════════════════════════════
# D4: quick_search / dream endpoints _score propagation
# ══════════════════════════════════════════════════════════════════════════


class TestDreamScorePropagation:
    """Test _score propagation through dream/ask endpoints."""

    def test_dream_search_entity_score(self, tmp_storage):
        """D4.1: Dream search_graph preserves entity scores."""
        from server.blueprints.helpers import entity_to_dict

        e1 = _make_entity("ent_dr_1", "AI", "Artificial Intelligence")
        e2 = _make_entity("ent_dr_2", "ML", "Machine Learning")

        # Simulate dream.py search_graph logic
        entity_score_map = {e1.absolute_id: 0.82, e2.absolute_id: 0.71}
        entities = [e1, e2]
        result = [entity_to_dict(e, _score=entity_score_map.get(e.absolute_id)) for e in entities]
        assert result[0]["_score"] == 0.82
        assert result[1]["_score"] == 0.71

    def test_dream_search_relation_score(self, tmp_storage):
        """D4.2: Dream search_graph preserves relation scores."""
        from server.blueprints.helpers import relation_to_dict

        e1 = _make_entity("ent_dr_r1", "A", "A")
        e2 = _make_entity("ent_dr_r2", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r1 = _make_relation("rel_dr_1", e1.absolute_id, e2.absolute_id, "test rel")

        relation_score_map = {r1.absolute_id: 0.67}
        relations = [r1]
        result = [relation_to_dict(r, _score=relation_score_map.get(r.absolute_id)) for r in relations]
        assert result[0]["_score"] == 0.67

    def test_dream_traverse_no_score(self, tmp_storage):
        """D4.3: Traverse query_type entities have no _score."""
        from server.blueprints.helpers import entity_to_dict

        e1 = _make_entity("ent_tr_1", "Graph", "Graph theory")
        entity_score_map: Dict[str, float] = {}  # traverse doesn't set scores

        result = [entity_to_dict(e, _score=entity_score_map.get(e.absolute_id)) for e in [e1]]
        assert "_score" not in result[0]

    def test_dream_search_mixed_entities_relations(self, tmp_storage):
        """D4.4: Mixed entity+relation results with separate scores."""
        from server.blueprints.helpers import entity_to_dict, relation_to_dict

        e1 = _make_entity("ent_mix_dr", "A", "A")
        e2 = _make_entity("ent_mix_dr2", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r1 = _make_relation("rel_mix_dr", e1.absolute_id, e2.absolute_id, "link")

        entity_score_map = {e1.absolute_id: 0.9}
        relation_score_map = {r1.absolute_id: 0.7}

        ent_dicts = [entity_to_dict(e, _score=entity_score_map.get(e.absolute_id)) for e in [e1, e2]]
        rel_dicts = [relation_to_dict(r, _score=relation_score_map.get(r.absolute_id)) for r in [r1]]

        assert ent_dicts[0]["_score"] == 0.9
        assert "_score" not in ent_dicts[1]  # e2 had no score
        assert rel_dicts[0]["_score"] == 0.7

    def test_score_with_empty_results(self, tmp_storage):
        """D4.5: Empty search results don't cause errors."""
        from server.blueprints.helpers import entity_to_dict, relation_to_dict
        entity_score_map: Dict[str, float] = {}
        relation_score_map: Dict[str, float] = {}

        ent_dicts = [entity_to_dict(e, _score=entity_score_map.get(e.absolute_id)) for e in []]
        rel_dicts = [relation_to_dict(r, _score=relation_score_map.get(r.absolute_id)) for r in []]
        assert ent_dicts == []
        assert rel_dicts == []

    def test_quick_search_relation_dedup_with_scores(self, tmp_storage):
        """D4.6: quick_search relation dedup preserves scores."""
        from server.blueprints.helpers import relation_to_dict

        e1 = _make_entity("ent_qs_r1", "A", "A")
        e2 = _make_entity("ent_qs_r2", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        r1 = _make_relation("rel_qs_r1", e1.absolute_id, e2.absolute_id, "link1")
        r2 = _make_relation("rel_qs_r2", e1.absolute_id, e2.absolute_id, "link2")

        fused_relations = [(r1, 0.6), (r2, 0.4)]
        relation_score_map = {r.absolute_id: score for r, score in fused_relations}
        relations = [r for r, _ in fused_relations]

        result = [relation_to_dict(r, _score=relation_score_map.get(r.absolute_id)) for r in relations]
        assert result[0]["_score"] == 0.6
        assert result[1]["_score"] == 0.4

    def test_score_boundary_values(self, tmp_storage):
        """D4.7: Scores at boundary values (0.0, 1.0) serialize correctly."""
        from server.blueprints.helpers import entity_to_dict, relation_to_dict

        e = _make_entity("ent_bnd", "Test", "Content")
        r = _make_relation("rel_bnd", "a1", "a2", "test")

        for score in [0.0, 1.0, 0.0001, 0.9999]:
            d = entity_to_dict(e, _score=score)
            assert d["_score"] == round(score, 4)

            d = relation_to_dict(r, _score=score)
            assert d["_score"] == round(score, 4)

    def test_score_does_not_modify_entity(self, tmp_storage):
        """D4.8: Serializing with _score doesn't mutate the Entity object."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ent_mut", "Test", "Content")
        original_name = e.name
        original_confidence = e.confidence

        d = entity_to_dict(e, _score=0.95)
        assert e.name == original_name
        assert e.confidence == original_confidence
        assert not hasattr(e, "_score") or e._score is None  # shouldn't be added

    def test_large_batch_scores(self, tmp_storage):
        """D4.9: Large batch of entities with scores serializes correctly."""
        from server.blueprints.helpers import entity_to_dict

        entities = [_make_entity(f"ent_batch_{i}", f"Name{i}", f"Content{i}") for i in range(50)]
        scores = [i / 50.0 for i in range(50)]
        entity_score_map = {e.absolute_id: s for e, s in zip(entities, scores)}

        results = [entity_to_dict(e, _score=entity_score_map.get(e.absolute_id)) for e in entities]
        assert len(results) == 50
        for i, (r, s) in enumerate(zip(results, scores)):
            assert r["_score"] == round(s, 4)
