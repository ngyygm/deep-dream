"""
Comprehensive tests for iteration-6: bounded BFS shortest paths.

Covers 4 dimensions with 35+ test cases:
  1. Basic path finding correctness (9 tests)
  2. Multi-path & diamond topology (9 tests)
  3. Edge cases & boundary conditions (10 tests)
  4. Invalidated relations & versioned entities (9 tests)
"""
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_real_storage(tmp_path):
    """Create a fully functional StorageManager (not mocked)."""
    from processor.storage.manager import StorageManager
    return StorageManager(str(tmp_path / "storage"))


def _entity(**kwargs):
    """Helper to build an Entity with sensible defaults."""
    from processor.models import Entity
    defaults = dict(
        event_time=datetime.now(),
        processed_time=datetime.now(),
        episode_id="ep1",
        source_document="",
    )
    defaults.update(kwargs)
    return Entity(**defaults)


def _relation(**kwargs):
    """Helper to build a Relation with sensible defaults."""
    from processor.models import Relation
    defaults = dict(
        content="test relation",
        event_time=datetime.now(),
        processed_time=datetime.now(),
        episode_id="ep1",
        source_document="",
    )
    defaults.update(kwargs)
    return Relation(**defaults)


def _save_entity(sm, fid, name=None, abs_id=None):
    """Save an entity and return (entity, absolute_id)."""
    name = name or fid
    abs_id = abs_id or f"{fid}_v1"
    e = _entity(absolute_id=abs_id, family_id=fid, name=name, content=f"{name} content")
    sm.save_entity(e)
    return e


def _save_relation(sm, fid, e1_abs, e2_abs, content="relates to"):
    """Save a relation."""
    r = _relation(
        absolute_id=f"{fid}_v1", family_id=fid,
        entity1_absolute_id=e1_abs, entity2_absolute_id=e2_abs,
        content=content,
    )
    sm.save_relation(r)
    return r


# ============================================================
# Dimension 1: Basic path finding correctness
# ============================================================
class TestBasicPathFinding:
    """Tests for basic single-path BFS correctness."""

    def test_direct_neighbor(self, tmp_path):
        """Two directly connected entities → path_length=1."""
        sm = _make_real_storage(tmp_path)
        e1 = _save_entity(sm, "ent_a", "Alice")
        e2 = _save_entity(sm, "ent_b", "Bob")
        _save_relation(sm, "rel_ab", e1.absolute_id, e2.absolute_id, "Alice knows Bob")
        result = sm.find_shortest_paths("ent_a", "ent_b")
        assert result["path_length"] == 1
        assert result["total_shortest_paths"] == 1
        assert len(result["paths"]) == 1
        assert result["source_entity"].family_id == "ent_a"
        assert result["target_entity"].family_id == "ent_b"
        sm.close()

    def test_path_length_2(self, tmp_path):
        """A→B→C → path_length=2."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        ec = _save_entity(sm, "ent_c", "Charlie")
        _save_relation(sm, "rel_ab", ea.absolute_id, eb.absolute_id)
        _save_relation(sm, "rel_bc", eb.absolute_id, ec.absolute_id)
        result = sm.find_shortest_paths("ent_a", "ent_c")
        assert result["path_length"] == 2
        assert result["total_shortest_paths"] == 1
        assert len(result["paths"][0]["entities"]) >= 2
        assert len(result["paths"][0]["relations"]) == 2
        sm.close()

    def test_path_length_3_chain(self, tmp_path):
        """A→B→C→D → path_length=3."""
        sm = _make_real_storage(tmp_path)
        entities = [_save_entity(sm, f"ent_{c}", c) for c in "ABCD"]
        for i in range(3):
            _save_relation(sm, f"rel_{i}", entities[i].absolute_id, entities[i + 1].absolute_id)
        result = sm.find_shortest_paths("ent_A", "ent_D")
        assert result["path_length"] == 3
        sm.close()

    def test_same_entity_returns_zero(self, tmp_path):
        """Source == target → path_length=0."""
        sm = _make_real_storage(tmp_path)
        _save_entity(sm, "ent_a", "Alice")
        result = sm.find_shortest_paths("ent_a", "ent_a")
        assert result["path_length"] == 0
        assert result["total_shortest_paths"] == 1
        assert len(result["paths"]) == 1
        assert result["paths"][0]["length"] == 0
        sm.close()

    def test_no_path_returns_negative_one(self, tmp_path):
        """Disconnected entities → path_length=-1."""
        sm = _make_real_storage(tmp_path)
        _save_entity(sm, "ent_a", "Alice")
        _save_entity(sm, "ent_b", "Bob")
        result = sm.find_shortest_paths("ent_a", "ent_b")
        assert result["path_length"] == -1
        assert result["total_shortest_paths"] == 0
        assert result["paths"] == []
        sm.close()

    def test_nonexistent_source(self, tmp_path):
        """Non-existent source → path_length=-1, source_entity=None."""
        sm = _make_real_storage(tmp_path)
        _save_entity(sm, "ent_b", "Bob")
        result = sm.find_shortest_paths("ent_nonexistent", "ent_b")
        assert result["path_length"] == -1
        assert result["source_entity"] is None
        sm.close()

    def test_nonexistent_target(self, tmp_path):
        """Non-existent target → path_length=-1, target_entity=None."""
        sm = _make_real_storage(tmp_path)
        _save_entity(sm, "ent_a", "Alice")
        result = sm.find_shortest_paths("ent_a", "ent_nonexistent")
        assert result["path_length"] == -1
        assert result["target_entity"] is None
        sm.close()

    def test_both_nonexistent(self, tmp_path):
        """Both non-existent → path_length=-1."""
        sm = _make_real_storage(tmp_path)
        result = sm.find_shortest_paths("ent_x", "ent_y")
        assert result["path_length"] == -1
        assert result["source_entity"] is None
        assert result["target_entity"] is None
        sm.close()

    def test_path_entities_are_entity_objects(self, tmp_path):
        """Path entities should be Entity objects with proper fields."""
        from processor.models import Entity
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        _save_relation(sm, "rel_ab", ea.absolute_id, eb.absolute_id)
        result = sm.find_shortest_paths("ent_a", "ent_b")
        for ent in result["paths"][0]["entities"]:
            assert isinstance(ent, Entity)
            assert ent.family_id is not None
            assert ent.name is not None
        sm.close()


# ============================================================
# Dimension 2: Multi-path & diamond topology
# ============================================================
class TestMultiPathDiamond:
    """Tests for multi-path and diamond-shaped topologies."""

    def test_diamond_two_paths(self, tmp_path):
        """Diamond: A→B→D and A→C→D → 2 shortest paths."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        ec = _save_entity(sm, "ent_c", "Charlie")
        ed = _save_entity(sm, "ent_d", "Diana")
        _save_relation(sm, "rel_ab", ea.absolute_id, eb.absolute_id)
        _save_relation(sm, "rel_ac", ea.absolute_id, ec.absolute_id)
        _save_relation(sm, "rel_bd", eb.absolute_id, ed.absolute_id)
        _save_relation(sm, "rel_cd", ec.absolute_id, ed.absolute_id)
        result = sm.find_shortest_paths("ent_a", "ent_d")
        assert result["path_length"] == 2
        assert result["total_shortest_paths"] == 2
        assert len(result["paths"]) == 2
        sm.close()

    def test_diamond_with_max_paths_limit(self, tmp_path):
        """max_paths=1 should return only 1 path even when 2 exist."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        ec = _save_entity(sm, "ent_c", "Charlie")
        ed = _save_entity(sm, "ent_d", "Diana")
        _save_relation(sm, "rel_ab", ea.absolute_id, eb.absolute_id)
        _save_relation(sm, "rel_ac", ea.absolute_id, ec.absolute_id)
        _save_relation(sm, "rel_bd", eb.absolute_id, ed.absolute_id)
        _save_relation(sm, "rel_cd", ec.absolute_id, ed.absolute_id)
        result = sm.find_shortest_paths("ent_a", "ent_d", max_paths=1)
        assert result["total_shortest_paths"] == 2
        assert len(result["paths"]) == 1
        sm.close()

    def test_three_way_parallel_paths(self, tmp_path):
        """A→D via B, C, E → 3 parallel paths of length 2."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        intermediates = [_save_entity(sm, f"ent_{c}", c) for c in ("Bob", "Charlie", "Eve")]
        ed = _save_entity(sm, "ent_d", "Diana")
        for i, mid in enumerate(intermediates):
            _save_relation(sm, f"rel_a{i}", ea.absolute_id, mid.absolute_id)
            _save_relation(sm, f"rel_{i}d", mid.absolute_id, ed.absolute_id)
        result = sm.find_shortest_paths("ent_a", "ent_d")
        assert result["path_length"] == 2
        assert result["total_shortest_paths"] == 3
        assert len(result["paths"]) == 3
        sm.close()

    def test_longer_path_not_returned(self, tmp_path):
        """Direct A→D path should be preferred over A→B→C→D."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        ec = _save_entity(sm, "ent_c", "Charlie")
        ed = _save_entity(sm, "ent_d", "Diana")
        # Direct path
        _save_relation(sm, "rel_ad", ea.absolute_id, ed.absolute_id)
        # Longer path
        _save_relation(sm, "rel_ab", ea.absolute_id, eb.absolute_id)
        _save_relation(sm, "rel_bc", eb.absolute_id, ec.absolute_id)
        _save_relation(sm, "rel_cd", ec.absolute_id, ed.absolute_id)
        result = sm.find_shortest_paths("ent_a", "ent_d")
        assert result["path_length"] == 1
        assert result["total_shortest_paths"] == 1
        sm.close()

    def test_triangle_three_edges(self, tmp_path):
        """Triangle A-B-C: all pairs have path_length=1."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        ec = _save_entity(sm, "ent_c", "Charlie")
        _save_relation(sm, "rel_ab", ea.absolute_id, eb.absolute_id)
        _save_relation(sm, "rel_bc", eb.absolute_id, ec.absolute_id)
        _save_relation(sm, "rel_ac", ea.absolute_id, ec.absolute_id)
        for src, tgt in [("ent_a", "ent_b"), ("ent_b", "ent_c"), ("ent_a", "ent_c")]:
            result = sm.find_shortest_paths(src, tgt)
            assert result["path_length"] == 1
        sm.close()

    def test_star_topology(self, tmp_path):
        """Star: center→all leaves; leaves reach each other via center (length=2)."""
        sm = _make_real_storage(tmp_path)
        center = _save_entity(sm, "ent_center", "Hub")
        leaves = [_save_entity(sm, f"ent_l{i}", f"Leaf{i}") for i in range(5)]
        for i, leaf in enumerate(leaves):
            _save_relation(sm, f"rel_hub{i}", center.absolute_id, leaf.absolute_id)
        # Any two leaves should have path_length=2
        result = sm.find_shortest_paths("ent_l0", "ent_l4")
        assert result["path_length"] == 2
        sm.close()

    def test_bidirectional_traversal(self, tmp_path):
        """A→B should work same as B→A."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        _save_relation(sm, "rel_ab", ea.absolute_id, eb.absolute_id)
        forward = sm.find_shortest_paths("ent_a", "ent_b")
        backward = sm.find_shortest_paths("ent_b", "ent_a")
        assert forward["path_length"] == backward["path_length"] == 1
        sm.close()

    def test_self_loop_relation_ignored(self, tmp_path):
        """Self-loop relations (entity1==entity2) should be ignored."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        # Self-loop on A (should be ignored)
        _save_relation(sm, "rel_self", ea.absolute_id, ea.absolute_id, "self-ref")
        _save_relation(sm, "rel_ab", ea.absolute_id, eb.absolute_id)
        result = sm.find_shortest_paths("ent_a", "ent_b")
        assert result["path_length"] == 1
        sm.close()

    def test_path_relations_are_relation_objects(self, tmp_path):
        """Path relations should be Relation objects with proper content."""
        from processor.models import Relation
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        _save_relation(sm, "rel_ab", ea.absolute_id, eb.absolute_id, "Alice mentors Bob")
        result = sm.find_shortest_paths("ent_a", "ent_b")
        assert len(result["paths"][0]["relations"]) == 1
        rel = result["paths"][0]["relations"][0]
        assert isinstance(rel, Relation)
        assert "mentors" in rel.content
        sm.close()


# ============================================================
# Dimension 3: Edge cases & boundary conditions
# ============================================================
class TestEdgeCases:
    """Tests for edge cases, boundary conditions, and special inputs."""

    def test_max_depth_1(self, tmp_path):
        """max_depth=1 should only find direct neighbors."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        ec = _save_entity(sm, "ent_c", "Charlie")
        _save_relation(sm, "rel_ab", ea.absolute_id, eb.absolute_id)
        _save_relation(sm, "rel_bc", eb.absolute_id, ec.absolute_id)
        # depth=1: A→B reachable, A→C not reachable
        result_ab = sm.find_shortest_paths("ent_a", "ent_b", max_depth=1)
        result_ac = sm.find_shortest_paths("ent_a", "ent_c", max_depth=1)
        assert result_ab["path_length"] == 1
        assert result_ac["path_length"] == -1
        sm.close()

    def test_max_depth_0(self, tmp_path):
        """max_depth=0 should only find same-entity paths."""
        sm = _make_real_storage(tmp_path)
        _save_entity(sm, "ent_a", "Alice")
        _save_entity(sm, "ent_b", "Bob")
        result = sm.find_shortest_paths("ent_a", "ent_a", max_depth=0)
        assert result["path_length"] == 0
        sm.close()

    def test_max_depth_prevents_deeper_search(self, tmp_path):
        """max_depth=2 should not find paths of length 3."""
        sm = _make_real_storage(tmp_path)
        entities = [_save_entity(sm, f"ent_{c}", c) for c in "ABCD"]
        for i in range(3):
            _save_relation(sm, f"rel_{i}", entities[i].absolute_id, entities[i + 1].absolute_id)
        result = sm.find_shortest_paths("ent_A", "ent_D", max_depth=2)
        assert result["path_length"] == -1
        sm.close()

    def test_empty_database(self, tmp_path):
        """Empty database → path_length=-1."""
        sm = _make_real_storage(tmp_path)
        result = sm.find_shortest_paths("ent_a", "ent_b")
        assert result["path_length"] == -1
        sm.close()

    def test_single_entity_no_relations(self, tmp_path):
        """Single entity, no relations → self path works, others don't."""
        sm = _make_real_storage(tmp_path)
        _save_entity(sm, "ent_a", "Alice")
        result_self = sm.find_shortest_paths("ent_a", "ent_a")
        assert result_self["path_length"] == 0
        # Create another entity with no relations
        _save_entity(sm, "ent_b", "Bob")
        result = sm.find_shortest_paths("ent_a", "ent_b")
        assert result["path_length"] == -1
        sm.close()

    def test_multiple_relations_same_pair(self, tmp_path):
        """Multiple relations between same pair → path_length=1, uses first relation."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        _save_relation(sm, "rel_ab1", ea.absolute_id, eb.absolute_id, "colleagues")
        _save_relation(sm, "rel_ab2", ea.absolute_id, eb.absolute_id, "friends")
        result = sm.find_shortest_paths("ent_a", "ent_b")
        assert result["path_length"] == 1
        # Should have at least one relation
        assert len(result["paths"][0]["relations"]) >= 1
        sm.close()

    def test_entity_with_multiple_versions(self, tmp_path):
        """Entity with multiple versions should use absolute_ids correctly."""
        sm = _make_real_storage(tmp_path)
        # Save two versions of entity A
        ea_v1 = _save_entity(sm, "ent_a", "Alice", abs_id="ent_a_v1")
        ea_v2 = _save_entity(sm, "ent_a", "Alice Updated", abs_id="ent_a_v2")
        eb = _save_entity(sm, "ent_b", "Bob")
        # Relation uses v2 of A
        _save_relation(sm, "rel_ab", ea_v2.absolute_id, eb.absolute_id)
        result = sm.find_shortest_paths("ent_a", "ent_b")
        assert result["path_length"] == 1
        sm.close()

    def test_deep_chain_max_depth_6(self, tmp_path):
        """Chain of 7 entities (6 hops) with max_depth=6 should find path."""
        sm = _make_real_storage(tmp_path)
        entities = [_save_entity(sm, f"ent_{i}", f"Node{i}") for i in range(7)]
        for i in range(6):
            _save_relation(sm, f"rel_{i}", entities[i].absolute_id, entities[i + 1].absolute_id)
        result = sm.find_shortest_paths("ent_0", "ent_6")
        assert result["path_length"] == 6
        sm.close()

    def test_deep_chain_exceeds_max_depth(self, tmp_path):
        """Chain of 7 entities with max_depth=5 should NOT find path."""
        sm = _make_real_storage(tmp_path)
        entities = [_save_entity(sm, f"ent_{i}", f"Node{i}") for i in range(7)]
        for i in range(6):
            _save_relation(sm, f"rel_{i}", entities[i].absolute_id, entities[i + 1].absolute_id)
        result = sm.find_shortest_paths("ent_0", "ent_6", max_depth=5)
        assert result["path_length"] == -1
        sm.close()

    def test_unicode_entity_names_in_paths(self, tmp_path):
        """Unicode entity names should work in path finding."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_zhang", "张三")
        eb = _save_entity(sm, "ent_li", "李四")
        _save_relation(sm, "rel_zl", ea.absolute_id, eb.absolute_id, "张三是李四的同事")
        result = sm.find_shortest_paths("ent_zhang", "ent_li")
        assert result["path_length"] == 1
        assert result["paths"][0]["relations"][0].content == "张三是李四的同事"
        sm.close()

    def test_large_graph_linear(self, tmp_path):
        """Linear graph with 50 nodes should find path correctly."""
        sm = _make_real_storage(tmp_path)
        entities = [_save_entity(sm, f"ent_{i:03d}", f"Node{i}") for i in range(50)]
        for i in range(49):
            _save_relation(sm, f"rel_{i:03d}", entities[i].absolute_id, entities[i + 1].absolute_id)
        result = sm.find_shortest_paths("ent_000", "ent_049", max_depth=50)
        assert result["path_length"] == 49
        sm.close()


# ============================================================
# Dimension 4: Invalidated relations & versioned entities
# ============================================================
class TestInvalidatedAndVersioned:
    """Tests for invalidated relations and versioned entities handling."""

    def test_invalidated_relation_excluded(self, tmp_path):
        """Soft-deleted (invalidated) relations should not be traversed."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        ec = _save_entity(sm, "ent_c", "Charlie")
        # Direct but invalidated
        r_invalid = _relation(
            absolute_id="rel_ab_inv", family_id="rel_ab",
            entity1_absolute_id=ea.absolute_id, entity2_absolute_id=eb.absolute_id,
            content="invalidated link",
        )
        sm.save_relation(r_invalid)
        # Manually invalidate
        conn = sm._get_conn()
        conn.execute("UPDATE relations SET invalid_at = ? WHERE id = ?",
                     (datetime.now(timezone.utc).isoformat(), "rel_ab_inv"))
        conn.commit()
        # Valid path via C
        _save_relation(sm, "rel_ac", ea.absolute_id, ec.absolute_id)
        _save_relation(sm, "rel_cb", ec.absolute_id, eb.absolute_id)
        result = sm.find_shortest_paths("ent_a", "ent_b")
        assert result["path_length"] == 2  # A→C→B, not direct
        sm.close()

    def test_all_paths_invalidated(self, tmp_path):
        """All relations invalidated → no path."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        r = _relation(
            absolute_id="rel_ab_inv", family_id="rel_ab",
            entity1_absolute_id=ea.absolute_id, entity2_absolute_id=eb.absolute_id,
            content="will be invalidated",
        )
        sm.save_relation(r)
        conn = sm._get_conn()
        conn.execute("UPDATE relations SET invalid_at = ? WHERE id = ?",
                     (datetime.now(timezone.utc).isoformat(), "rel_ab_inv"))
        conn.commit()
        result = sm.find_shortest_paths("ent_a", "ent_b")
        assert result["path_length"] == -1
        sm.close()

    def test_entity_with_old_and_new_version(self, tmp_path):
        """Relation using old version should still be traversed."""
        sm = _make_real_storage(tmp_path)
        ea_v1 = _save_entity(sm, "ent_a", "Alice v1", abs_id="a_v1")
        ea_v2 = _save_entity(sm, "ent_a", "Alice v2", abs_id="a_v2")
        eb = _save_entity(sm, "ent_b", "Bob")
        # Relation uses old version
        _save_relation(sm, "rel_ab", "a_v1", eb.absolute_id)
        result = sm.find_shortest_paths("ent_a", "ent_b")
        assert result["path_length"] == 1
        sm.close()

    def test_relation_between_different_versions_different_families(self, tmp_path):
        """Relation between versions of different families."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice", abs_id="a_v1")
        eb = _save_entity(sm, "ent_b", "Bob", abs_id="b_v1")
        # Both have second versions
        _save_entity(sm, "ent_a", "Alice v2", abs_id="a_v2")
        _save_entity(sm, "ent_b", "Bob v2", abs_id="b_v2")
        # Relations from different versions
        _save_relation(sm, "rel_ab1", "a_v1", "b_v1", "v1 link")
        _save_relation(sm, "rel_ab2", "a_v2", "b_v2", "v2 link")
        result = sm.find_shortest_paths("ent_a", "ent_b")
        assert result["path_length"] == 1
        sm.close()

    def test_multiple_version_chain(self, tmp_path):
        """Chain where each entity has multiple versions."""
        sm = _make_real_storage(tmp_path)
        for c in "ABCD":
            _save_entity(sm, f"ent_{c}", f"{c} v1", abs_id=f"{c}_v1")
            _save_entity(sm, f"ent_{c}", f"{c} v2", abs_id=f"{c}_v2")
        for i, (a, b) in enumerate([("A", "B"), ("B", "C"), ("C", "D")]):
            _save_relation(sm, f"rel_{i}", f"{a}_v2", f"{b}_v2")
        result = sm.find_shortest_paths("ent_A", "ent_D")
        assert result["path_length"] == 3
        sm.close()

    def test_invalidated_bypass_longer_path(self, tmp_path):
        """Invalidated shortcut forces longer path."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        ec = _save_entity(sm, "ent_c", "Charlie")
        # Direct but invalidated
        r = _relation(absolute_id="rel_ab", family_id="rel_ab",
                       entity1_absolute_id=ea.absolute_id, entity2_absolute_id=eb.absolute_id)
        sm.save_relation(r)
        conn = sm._get_conn()
        conn.execute("UPDATE relations SET invalid_at = ? WHERE id = ?",
                     (datetime.now(timezone.utc).isoformat(), "rel_ab"))
        conn.commit()
        # Valid path: A→C→B
        _save_relation(sm, "rel_ac", ea.absolute_id, ec.absolute_id)
        _save_relation(sm, "rel_cb", ec.absolute_id, eb.absolute_id)
        result = sm.find_shortest_paths("ent_a", "ent_b")
        assert result["path_length"] == 2
        sm.close()

    def test_diamond_with_one_invalidated(self, tmp_path):
        """Diamond with one arm invalidated → only one path remains."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        ec = _save_entity(sm, "ent_c", "Charlie")
        ed = _save_entity(sm, "ent_d", "Diana")
        _save_relation(sm, "rel_ab", ea.absolute_id, eb.absolute_id)
        _save_relation(sm, "rel_ac", ea.absolute_id, ec.absolute_id)
        _save_relation(sm, "rel_bd", eb.absolute_id, ed.absolute_id)
        # Invalidate C→D
        r_cd = _relation(absolute_id="rel_cd", family_id="rel_cd",
                          entity1_absolute_id=ec.absolute_id, entity2_absolute_id=ed.absolute_id)
        sm.save_relation(r_cd)
        conn = sm._get_conn()
        conn.execute("UPDATE relations SET invalid_at = ? WHERE id = ?",
                     (datetime.now(timezone.utc).isoformat(), "rel_cd"))
        conn.commit()
        result = sm.find_shortest_paths("ent_a", "ent_d")
        assert result["path_length"] == 2
        assert result["total_shortest_paths"] == 1
        sm.close()

    def test_cross_version_relation(self, tmp_path):
        """Relation connecting v1 of A to v2 of B."""
        sm = _make_real_storage(tmp_path)
        _save_entity(sm, "ent_a", "A", abs_id="a_v1")
        _save_entity(sm, "ent_a", "A v2", abs_id="a_v2")
        _save_entity(sm, "ent_b", "B", abs_id="b_v1")
        _save_entity(sm, "ent_b", "B v2", abs_id="b_v2")
        _save_relation(sm, "rel_cross", "a_v1", "b_v2", "cross-version link")
        result = sm.find_shortest_paths("ent_a", "ent_b")
        assert result["path_length"] == 1
        sm.close()

    def test_result_structure_completeness(self, tmp_path):
        """Result should have all required keys."""
        sm = _make_real_storage(tmp_path)
        ea = _save_entity(sm, "ent_a", "Alice")
        eb = _save_entity(sm, "ent_b", "Bob")
        _save_relation(sm, "rel_ab", ea.absolute_id, eb.absolute_id)
        result = sm.find_shortest_paths("ent_a", "ent_b")
        assert "source_entity" in result
        assert "target_entity" in result
        assert "path_length" in result
        assert "total_shortest_paths" in result
        assert "paths" in result
        # Each path should have entities, relations, length
        for path in result["paths"]:
            assert "entities" in path
            assert "relations" in path
            assert "length" in path
        sm.close()
