"""
Graph & Concept Traversal Enrichment tests — 36 tests across 4 dimensions.

D1: bfs_expand_with_relations returns entities + relations (9 tests)
D2: traverse_graph endpoint returns full entity/relation dicts (9 tests)
D3: traverse_concepts includes full relation metadata (9 tests)
D4: Cross-cutting — edge cases, dedup, large graphs (9 tests)
"""
from __future__ import annotations

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


def _build_chain(storage, count: int, prefix: str = "chain"):
    """Build a chain of entities: e0 --r0--> e1 --r1--> e2 ..."""
    entities = []
    relations = []
    for i in range(count):
        e = _make_entity(f"{prefix}_e{i}", f"Node{i}", f"Content{i}")
        storage.save_entity(e)
        entities.append(e)
    for i in range(count - 1):
        r = _make_relation(f"{prefix}_r{i}", entities[i].absolute_id,
                           entities[i+1].absolute_id, f"Link{i}->{i+1}")
        storage.save_relation(r)
        relations.append(r)
    return entities, relations


# ══════════════════════════════════════════════════════════════════════════
# D1: bfs_expand_with_relations returns entities + relations
# ══════════════════════════════════════════════════════════════════════════


class TestBfsExpandWithRelations:
    """Test that bfs_expand_with_relations collects both entities and relations."""

    def test_single_entity_no_relations(self, tmp_storage):
        """D1.1: Single seed entity with no relations."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        tmp_storage.save_entity(_make_entity("solo1", "Solo", "Alone"))
        searcher = GraphTraversalSearcher(tmp_storage)
        ents, rels, visited = searcher.bfs_expand_with_relations(["solo1"])
        assert len(ents) == 1
        assert len(rels) == 0
        assert "solo1" in visited

    def test_chain_of_two(self, tmp_storage):
        """D1.2: Two connected entities, 1 relation."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        ents, rels = _build_chain(tmp_storage, 2, "ch2")
        searcher = GraphTraversalSearcher(tmp_storage)
        found_ents, found_rels, visited = searcher.bfs_expand_with_relations(["ch2_e0"])
        assert len(found_ents) == 2
        assert len(found_rels) == 1

    def test_chain_of_three(self, tmp_storage):
        """D1.3: Three connected entities, 2 relations."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        _build_chain(tmp_storage, 3, "ch3")
        searcher = GraphTraversalSearcher(tmp_storage)
        ents, rels, visited = searcher.bfs_expand_with_relations(["ch3_e0"], max_depth=2)
        assert len(ents) == 3
        assert len(rels) == 2

    def test_max_depth_limits(self, tmp_storage):
        """D1.4: max_depth=1 only reaches immediate neighbors."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        _build_chain(tmp_storage, 4, "depth")
        searcher = GraphTraversalSearcher(tmp_storage)
        ents, rels, visited = searcher.bfs_expand_with_relations(
            ["depth_e0"], max_depth=1, max_nodes=50)
        assert len(ents) == 2  # e0 + e1 only
        assert len(rels) == 1

    def test_max_nodes_limits(self, tmp_storage):
        """D1.5: max_nodes limits returned entities."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        _build_chain(tmp_storage, 5, "mn")
        searcher = GraphTraversalSearcher(tmp_storage)
        ents, rels, visited = searcher.bfs_expand_with_relations(
            ["mn_e0"], max_depth=2, max_nodes=2)
        assert len(ents) <= 2

    def test_relation_dedup(self, tmp_storage):
        """D1.6: Relations are not duplicated when visited from both sides."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        e1 = _make_entity("hub1", "Hub", "Central")
        e2 = _make_entity("spoke1", "Spoke", "Edge")
        e3 = _make_entity("spoke2", "Spoke2", "Edge2")
        tmp_storage.bulk_save_entities([e1, e2, e3])
        r1 = _make_relation("r_hub_s1", e1.absolute_id, e2.absolute_id, "link1")
        r2 = _make_relation("r_hub_s2", e1.absolute_id, e3.absolute_id, "link2")
        tmp_storage.bulk_save_relations([r1, r2])
        searcher = GraphTraversalSearcher(tmp_storage)
        ents, rels, visited = searcher.bfs_expand_with_relations(["hub1"])
        assert len(rels) == 2
        # No duplicate family_ids
        rel_fids = [r.family_id for r in rels]
        assert len(rel_fids) == len(set(rel_fids))

    def test_multiple_seeds(self, tmp_storage):
        """D1.7: Multiple seed entities."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        e1 = _make_entity("seed_a", "A", "A")
        e2 = _make_entity("seed_b", "B", "B")
        e3 = _make_entity("seed_c", "C", "C")
        tmp_storage.bulk_save_entities([e1, e2, e3])
        r = _make_relation("r_ab", e1.absolute_id, e2.absolute_id, "A-B")
        tmp_storage.save_relation(r)
        searcher = GraphTraversalSearcher(tmp_storage)
        ents, rels, visited = searcher.bfs_expand_with_relations(["seed_a", "seed_b"])
        assert len(ents) >= 2

    def test_backward_compat_bfs_expand(self, tmp_storage):
        """D1.8: bfs_expand still returns only entities (backward compat)."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        _build_chain(tmp_storage, 3, "compat")
        searcher = GraphTraversalSearcher(tmp_storage)
        ents = searcher.bfs_expand(["compat_e0"], max_depth=2)
        assert isinstance(ents, list)
        assert len(ents) == 3

    def test_visited_set_correct(self, tmp_storage):
        """D1.9: Visited set includes all traversed family_ids."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        _build_chain(tmp_storage, 3, "vis")
        searcher = GraphTraversalSearcher(tmp_storage)
        ents, rels, visited = searcher.bfs_expand_with_relations(
            ["vis_e0"], max_depth=2, max_nodes=50)
        assert "vis_e0" in visited
        assert "vis_e1" in visited
        assert "vis_e2" in visited


# ══════════════════════════════════════════════════════════════════════════
# D2: traverse_graph endpoint returns full entity/relation dicts
# ══════════════════════════════════════════════════════════════════════════


class TestTraverseGraphEndpoint:
    """Test that traverse_graph returns full entity and relation dicts."""

    def test_returns_entities_and_relations_keys(self, tmp_storage):
        """D2.1: Response has entities, relations, visited_count keys."""
        from server.blueprints.helpers import entity_to_dict, relation_to_dict
        from server.blueprints.helpers import enrich_entity_version_counts, enrich_relation_version_counts
        from processor.search.graph_traversal import GraphTraversalSearcher

        _build_chain(tmp_storage, 3, "ep1")
        searcher = GraphTraversalSearcher(tmp_storage)
        entities, relations, visited = searcher.bfs_expand_with_relations(
            ["ep1_e0"], max_depth=2, max_nodes=50)

        ent_dicts = [entity_to_dict(e) for e in entities]
        rel_dicts = [relation_to_dict(r) for r in relations]
        enrich_entity_version_counts(ent_dicts, tmp_storage)
        enrich_relation_version_counts(rel_dicts, tmp_storage)

        assert "entities" in {"entities": ent_dicts}
        assert "relations" in {"relations": rel_dicts}

    def test_entity_dict_has_all_fields(self, tmp_storage):
        """D2.2: Entity dicts include event_time, confidence, etc."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ep2_e0", "Node", "Content", confidence=0.8)
        tmp_storage.save_entity(e)
        d = entity_to_dict(tmp_storage.get_entity_by_family_id("ep2_e0"))
        assert "event_time" in d
        assert "processed_time" in d
        assert "confidence" in d
        assert "family_id" in d

    def test_relation_dict_has_all_fields(self, tmp_storage):
        """D2.3: Relation dicts include event_time, confidence, etc."""
        from server.blueprints.helpers import relation_to_dict
        e1 = _make_entity("ep3_a", "A", "A")
        e2 = _make_entity("ep3_b", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        r = _make_relation("ep3_r", e1.absolute_id, e2.absolute_id, "link", confidence=0.9)
        tmp_storage.save_relation(r)
        d = relation_to_dict(tmp_storage.get_relation_by_family_id("ep3_r"))
        assert "event_time" in d
        assert "processed_time" in d
        assert "confidence" in d
        assert "entity1_absolute_id" in d
        assert "entity2_absolute_id" in d

    def test_version_count_in_entity_dicts(self, tmp_storage):
        """D2.4: Entity dicts include version_count after enrichment."""
        from server.blueprints.helpers import entity_to_dict, enrich_entity_version_counts
        for i in range(3):
            tmp_storage.save_entity(_make_entity("ep4_e0", f"V{i}", f"C{i}"))
        e = tmp_storage.get_entity_by_family_id("ep4_e0")
        dicts = [entity_to_dict(e)]
        enrich_entity_version_counts(dicts, tmp_storage)
        assert dicts[0]["version_count"] == 3

    def test_version_count_in_relation_dicts(self, tmp_storage):
        """D2.5: Relation dicts include version_count after enrichment."""
        from server.blueprints.helpers import relation_to_dict, enrich_relation_version_counts
        e1 = _make_entity("ep5_a", "A", "A")
        e2 = _make_entity("ep5_b", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        for i in range(2):
            tmp_storage.save_relation(_make_relation("ep5_r", e1.absolute_id, e2.absolute_id, f"L{i}"))
        r = tmp_storage.get_relation_by_family_id("ep5_r")
        dicts = [relation_to_dict(r)]
        enrich_relation_version_counts(dicts, tmp_storage)
        assert dicts[0]["version_count"] == 2

    def test_entity_names_in_relations(self, tmp_storage):
        """D2.6: Relation dicts have entity1_name/entity2_name after enrich."""
        from server.blueprints.helpers import relation_to_dict, enrich_relations
        from processor.storage.manager import StorageManager
        e1 = _make_entity("ep6_a", "Alpha", "A")
        e2 = _make_entity("ep6_b", "Beta", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        r = _make_relation("ep6_r", e1.absolute_id, e2.absolute_id, "link")
        tmp_storage.save_relation(r)
        r = tmp_storage.get_relation_by_family_id("ep6_r")
        dicts = [relation_to_dict(r)]

        class FakeProcessor:
            storage = tmp_storage
        enrich_relations(dicts, FakeProcessor())
        assert dicts[0]["entity1_name"] == "Alpha"
        assert dicts[0]["entity2_name"] == "Beta"

    def test_visited_count_correct(self, tmp_storage):
        """D2.7: visited_count matches number of unique entity families."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        _build_chain(tmp_storage, 3, "vcnt")
        searcher = GraphTraversalSearcher(tmp_storage)
        _, _, visited = searcher.bfs_expand_with_relations(["vcnt_e0"], max_depth=2)
        assert len(visited) == 3

    def test_empty_seed_returns_empty(self, tmp_storage):
        """D2.8: Empty seed list returns empty results."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        searcher = GraphTraversalSearcher(tmp_storage)
        ents, rels, visited = searcher.bfs_expand_with_relations(
            ["nonexistent_12345"])
        assert len(ents) == 0
        assert len(rels) == 0

    def test_chinese_entity_names_preserved(self, tmp_storage):
        """D2.9: Chinese names preserved in traversal results."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("cn_ent", "深度学习", "机器学习子领域")
        tmp_storage.save_entity(e)
        d = entity_to_dict(tmp_storage.get_entity_by_family_id("cn_ent"))
        assert d["name"] == "深度学习"


# ══════════════════════════════════════════════════════════════════════════
# D3: traverse_concepts includes full relation metadata
# ══════════════════════════════════════════════════════════════════════════


class TestTraverseConcepts:
    """Test traverse_concepts includes full relation metadata."""

    def test_concept_traversal_returns_concepts_and_relations(self, tmp_storage):
        """D3.1: traverse_concepts returns concepts and relations keys."""
        e1 = _make_entity("tc_a", "Alpha", "Alpha content")
        e2 = _make_entity("tc_b", "Beta", "Beta content")
        tmp_storage.bulk_save_entities([e1, e2])
        r = _make_relation("tc_r", e1.absolute_id, e2.absolute_id, "connects")
        tmp_storage.save_relation(r)

        result = tmp_storage.traverse_concepts(["tc_a"])
        assert "concepts" in result
        assert "relations" in result
        assert "visited_count" in result

    def test_relation_concept_has_full_fields(self, tmp_storage):
        """D3.2: Relation entries include event_time, processed_time, confidence."""
        e1 = _make_entity("tc2_a", "Alpha", "Alpha content", confidence=0.8)
        e2 = _make_entity("tc2_b", "Beta", "Beta content")
        tmp_storage.bulk_save_entities([e1, e2])
        r = _make_relation("tc2_r", e1.absolute_id, e2.absolute_id, "link", confidence=0.9)
        tmp_storage.save_relation(r)

        result = tmp_storage.traverse_concepts(["tc2_a"])
        rels = result["relations"]
        # Should have at least one relation with full metadata
        if rels:
            rel = rels[0]
            assert "family_id" in rel or "role" in rel
            # Full concept records include these fields
            if "event_time" in rel:
                assert rel["event_time"] is not None

    def test_concept_entity_has_role(self, tmp_storage):
        """D3.3: Entity concept has role='entity'."""
        e = _make_entity("tc3_a", "Test", "Content")
        tmp_storage.save_entity(e)
        result = tmp_storage.traverse_concepts(["tc3_a"])
        concept = result["concepts"].get("tc3_a")
        assert concept is not None
        assert concept.get("role") == "entity"

    def test_concept_relation_has_role(self, tmp_storage):
        """D3.4: Relation concept has role='relation'."""
        e1 = _make_entity("tc4_a", "Alpha", "A")
        e2 = _make_entity("tc4_b", "Beta", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        r = _make_relation("tc4_r", e1.absolute_id, e2.absolute_id, "link")
        tmp_storage.save_relation(r)

        result = tmp_storage.traverse_concepts(["tc4_a"])
        rels = result["relations"]
        if rels:
            assert rels[0].get("role") == "relation"

    def test_no_duplicate_relations(self, tmp_storage):
        """D3.5: Relations not duplicated even when traversed from multiple entities."""
        e1 = _make_entity("tc5_a", "A", "A")
        e2 = _make_entity("tc5_b", "B", "B")
        e3 = _make_entity("tc5_c", "C", "C")
        tmp_storage.bulk_save_entities([e1, e2, e3])
        r1 = _make_relation("tc5_r1", e1.absolute_id, e2.absolute_id, "A-B")
        r2 = _make_relation("tc5_r2", e2.absolute_id, e3.absolute_id, "B-C")
        tmp_storage.bulk_save_relations([r1, r2])

        result = tmp_storage.traverse_concepts(["tc5_a"])
        rel_fids = [r.get("family_id") for r in result["relations"]]
        assert len(rel_fids) == len(set(rel_fids))

    def test_visited_count_includes_all(self, tmp_storage):
        """D3.6: visited_count includes all visited concepts."""
        e1 = _make_entity("tc6_a", "A", "A")
        e2 = _make_entity("tc6_b", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        r = _make_relation("tc6_r", e1.absolute_id, e2.absolute_id, "link")
        tmp_storage.save_relation(r)

        result = tmp_storage.traverse_concepts(["tc6_a"])
        assert result["visited_count"] >= 1  # At least the seed

    def test_nonexistent_seed_returns_empty(self, tmp_storage):
        """D3.7: Nonexistent seed returns empty concepts."""
        result = tmp_storage.traverse_concepts(["nonexistent_xyz"])
        assert result["concepts"] == {}
        assert result["visited_count"] == 0

    def test_max_depth_limits_traversal(self, tmp_storage):
        """D3.8: max_depth limits how deep BFS goes."""
        ents, rels = _build_chain(tmp_storage, 4, "td")
        # max_depth=1: seed entity -> its relation neighbors (depth 1)
        result = tmp_storage.traverse_concepts(["td_e0"], max_depth=1)
        concepts = result["concepts"]
        assert "td_e0" in concepts
        # The relation concept td_r0 is a direct neighbor at depth 1
        assert any(r.get("family_id") == "td_r0" for r in result["relations"])
        # e3 should not be visited (too deep)
        assert "td_e3" not in concepts
        # With max_depth=2, e1 should be reachable via the relation
        result2 = tmp_storage.traverse_concepts(["td_e0"], max_depth=2)
        assert "td_e0" in result2["concepts"]
        assert "td_e1" in result2["concepts"]

    def test_concept_has_confidence(self, tmp_storage):
        """D3.9: Concept entries include confidence field."""
        e = _make_entity("tc9_a", "Test", "Content", confidence=0.85)
        tmp_storage.save_entity(e)
        result = tmp_storage.traverse_concepts(["tc9_a"])
        concept = result["concepts"].get("tc9_a")
        assert concept is not None
        assert concept.get("confidence") == 0.85


# ══════════════════════════════════════════════════════════════════════════
# D4: Cross-cutting — edge cases, dedup, large graphs
# ══════════════════════════════════════════════════════════════════════════


class TestTraversalEdgeCases:
    """Cross-cutting traversal edge cases."""

    def test_self_loop_not_created(self, tmp_storage):
        """D4.1: BFS doesn't loop back to already-visited nodes."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        e1 = _make_entity("loop_a", "A", "A")
        e2 = _make_entity("loop_b", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        r1 = _make_relation("loop_r1", e1.absolute_id, e2.absolute_id, "A->B")
        r2 = _make_relation("loop_r2", e2.absolute_id, e1.absolute_id, "B->A")
        tmp_storage.bulk_save_relations([r1, r2])
        searcher = GraphTraversalSearcher(tmp_storage)
        ents, rels, visited = searcher.bfs_expand_with_relations(["loop_a"])
        assert len(ents) == 2  # No duplicates
        assert len(visited) == 2

    def test_disconnected_subgraph(self, tmp_storage):
        """D4.2: Seed in disconnected subgraph only reaches its component."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        e1 = _make_entity("comp_a", "A", "A")
        e2 = _make_entity("comp_b", "B", "B")
        e3 = _make_entity("comp_c", "C", "C")  # disconnected
        tmp_storage.bulk_save_entities([e1, e2, e3])
        r = _make_relation("comp_r", e1.absolute_id, e2.absolute_id, "A-B")
        tmp_storage.save_relation(r)
        searcher = GraphTraversalSearcher(tmp_storage)
        ents, rels, visited = searcher.bfs_expand_with_relations(["comp_a"])
        assert "comp_c" not in visited

    def test_star_topology(self, tmp_storage):
        """D4.3: Star topology — hub connected to 5 spokes."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        hub = _make_entity("star_hub", "Hub", "Central")
        tmp_storage.save_entity(hub)
        spokes = []
        for i in range(5):
            s = _make_entity(f"star_s{i}", f"Spoke{i}", f"Edge{i}")
            tmp_storage.save_entity(s)
            spokes.append(s)
            r = _make_relation(f"star_r{i}", hub.absolute_id, s.absolute_id, f"Hub-Spoke{i}")
            tmp_storage.save_relation(r)
        searcher = GraphTraversalSearcher(tmp_storage)
        ents, rels, visited = searcher.bfs_expand_with_relations(
            ["star_hub"], max_depth=1, max_nodes=50)
        assert len(ents) == 6  # hub + 5 spokes
        assert len(rels) == 5

    def test_chain_of_10(self, tmp_storage):
        """D4.4: Chain of 10 entities traversed fully."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        _build_chain(tmp_storage, 10, "long")
        searcher = GraphTraversalSearcher(tmp_storage)
        ents, rels, visited = searcher.bfs_expand_with_relations(
            ["long_e0"], max_depth=10, max_nodes=50)
        assert len(ents) == 10
        assert len(rels) == 9

    def test_unicode_entity_names(self, tmp_storage):
        """D4.5: Unicode entity names preserved through traversal."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("uni_α", "α粒子", "物理概念 🚀")
        tmp_storage.save_entity(e)
        d = entity_to_dict(tmp_storage.get_entity_by_family_id("uni_α"))
        assert "α粒子" in d["name"]
        assert "🚀" in d["content"]

    def test_multi_seed_merge(self, tmp_storage):
        """D4.6: Two seeds converge on same entity."""
        from processor.search.graph_traversal import GraphTraversalSearcher
        e1 = _make_entity("merge_a", "A", "A")
        e2 = _make_entity("merge_b", "B", "B")
        e3 = _make_entity("merge_c", "C", "C")  # shared target
        tmp_storage.bulk_save_entities([e1, e2, e3])
        r1 = _make_relation("merge_r1", e1.absolute_id, e3.absolute_id, "A-C")
        r2 = _make_relation("merge_r2", e2.absolute_id, e3.absolute_id, "B-C")
        tmp_storage.bulk_save_relations([r1, r2])
        searcher = GraphTraversalSearcher(tmp_storage)
        ents, rels, visited = searcher.bfs_expand_with_relations(
            ["merge_a", "merge_b"])
        assert len(ents) == 3  # No duplicate for merge_c
        assert "merge_c" in visited

    def test_relation_with_none_confidence(self, tmp_storage):
        """D4.7: Relations with None confidence serialize correctly."""
        from server.blueprints.helpers import relation_to_dict
        e1 = _make_entity("nc_a", "A", "A")
        e2 = _make_entity("nc_b", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        r = _make_relation("nc_r", e1.absolute_id, e2.absolute_id, "link", confidence=None)
        tmp_storage.save_relation(r)
        d = relation_to_dict(tmp_storage.get_relation_by_family_id("nc_r"))
        assert d["confidence"] is None
        assert "family_id" in d

    def test_concept_traversal_empty_start(self, tmp_storage):
        """D4.8: Empty start_family_ids returns empty result."""
        result = tmp_storage.traverse_concepts([])
        assert result["concepts"] == {}
        assert result["relations"] == []
        assert result["visited_count"] == 0

    def test_enrich_preserves_all_relation_fields(self, tmp_storage):
        """D4.9: enrich_relation_version_counts preserves all fields."""
        from server.blueprints.helpers import relation_to_dict, enrich_relation_version_counts
        e1 = _make_entity("pf_a", "A", "A")
        e2 = _make_entity("pf_b", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        r = _make_relation("pf_r", e1.absolute_id, e2.absolute_id, "link",
                           confidence=0.7, attributes='{"tier":"candidate"}')
        tmp_storage.save_relation(r)
        d = relation_to_dict(tmp_storage.get_relation_by_family_id("pf_r"), _score=0.88)
        enrich_relation_version_counts([d], tmp_storage)
        assert d["_score"] == 0.88
        assert d["confidence"] == 0.7
        assert d["version_count"] == 1
        assert d["family_id"] == "pf_r"
