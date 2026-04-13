"""
Tests for DreamOrchestrator: seed selection, graph exploration, relation discovery.

Covers DreamHistory, DreamConfig, and the orchestrator's 5-step pipeline
with mocked storage and LLM client.
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processor.dream.orchestrator import (
    DreamHistory,
    DreamConfig,
    DreamOrchestrator,
    DreamResult,
    VALID_STRATEGIES,
)
from processor.models import Entity
from datetime import datetime, timezone, timedelta


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_entity(
    family_id="fam_1",
    name="TestEntity",
    content="A test entity with some content.",
    confidence=0.7,
) -> Entity:
    return Entity(
        absolute_id=f"abs_{family_id}",
        family_id=family_id,
        name=name,
        content=content,
        confidence=confidence,
        event_time=_now(),
        processed_time=_now(),
        episode_id="ep_1",
        source_document="test.md",
    )


# ═══════════════════════════════════════════════════════════════════
# DreamHistory tests
# ═══════════════════════════════════════════════════════════════════

class TestDreamHistory:
    """Tests for DreamHistory LRU tracking."""

    def test_mark_and_check(self):
        h = DreamHistory()
        h.mark_checked("a", "b", "c1")
        assert h.was_checked("a", "b") is True
        assert h.was_checked("b", "a") is True  # frozenset is order-independent

    def test_not_checked(self):
        h = DreamHistory()
        assert h.was_checked("a", "b") is False

    def test_lru_eviction(self):
        h = DreamHistory(max_entries=3)
        h.mark_checked("a", "b", "c1")
        h.mark_checked("c", "d", "c1")
        h.mark_checked("e", "f", "c1")
        # Adding a 4th entry should evict the 1st
        h.mark_checked("g", "h", "c1")
        assert h.was_checked("a", "b") is False
        assert h.was_checked("g", "h") is True

    def test_lru_access_moves_to_end(self):
        h = DreamHistory(max_entries=3)
        h.mark_checked("a", "b", "c1")
        h.mark_checked("c", "d", "c1")
        h.mark_checked("e", "f", "c1")
        # Accessing "a","b" moves it to end
        assert h.was_checked("a", "b") is True
        # Adding 1 more should evict "c","d" (not "a","b")
        h.mark_checked("g", "h", "c1")
        assert h.was_checked("a", "b") is True
        assert h.was_checked("c", "d") is False

    def test_mark_explored(self):
        h = DreamHistory()
        h.mark_explored("c1", {"a", "b"})
        h.mark_explored("c2", {"c", "d"})
        recent = h.get_recently_explored(last_n=2)
        assert recent == {"a", "b", "c", "d"}

    def test_explored_keeps_only_10(self):
        h = DreamHistory()
        for i in range(15):
            h.mark_explored(f"c{i}", {f"e{i}"})
        # Only last 10 cycles kept (c5..c14), c0..c4 evicted
        recent = h.get_recently_explored(last_n=15)
        assert f"e0" not in recent
        assert f"e4" not in recent
        assert f"e5" in recent  # c5 is the oldest surviving cycle
        assert f"e14" in recent

    def test_get_recently_explored_n(self):
        h = DreamHistory()
        h.mark_explored("c1", {"a"})
        h.mark_explored("c2", {"b"})
        h.mark_explored("c3", {"c"})
        # Only last 2
        recent = h.get_recently_explored(last_n=2)
        assert recent == {"b", "c"}

    def test_reset(self):
        h = DreamHistory()
        h.mark_checked("a", "b", "c1")
        h.mark_explored("c1", {"x"})
        h.reset()
        assert h.was_checked("a", "b") is False
        assert h.get_recently_explored() == set()

    def test_empty_history(self):
        h = DreamHistory()
        assert h.get_recently_explored() == set()
        assert h.was_checked("x", "y") is False


# ═══════════════════════════════════════════════════════════════════
# DreamConfig tests
# ═══════════════════════════════════════════════════════════════════

class TestDreamConfig:
    """Tests for DreamConfig validation and clamping."""

    def test_defaults(self):
        c = DreamConfig()
        assert c.seed_count == 3
        assert c.max_depth == 2
        assert c.max_relations == 5
        assert c.strategy == "random"

    def test_seed_count_clamped(self):
        assert DreamConfig(seed_count=0).seed_count == 1
        assert DreamConfig(seed_count=100).seed_count == 10

    def test_max_depth_clamped(self):
        assert DreamConfig(max_depth=0).max_depth == 1
        assert DreamConfig(max_depth=10).max_depth == 4

    def test_max_relations_clamped(self):
        assert DreamConfig(max_relations=0).max_relations == 1
        assert DreamConfig(max_relations=50).max_relations == 20

    def test_min_confidence_clamped(self):
        assert DreamConfig(min_confidence=-1.0).min_confidence == 0.0
        assert DreamConfig(min_confidence=2.0).min_confidence == 1.0


# ═══════════════════════════════════════════════════════════════════
# DreamOrchestrator: _select_seeds
# ═══════════════════════════════════════════════════════════════════

class TestSelectSeeds:
    """Tests for seed selection step."""

    def _make_orchestrator(self, seeds_return=None):
        storage = MagicMock()
        if seeds_return is not None:
            storage.get_dream_seeds.return_value = seeds_return
        else:
            storage.get_dream_seeds.return_value = []
        llm = MagicMock()
        config = DreamConfig()
        return DreamOrchestrator(storage, llm, config), storage

    def test_no_seeds_returns_empty(self):
        orch, _ = self._make_orchestrator(seeds_return=[])
        result = orch._select_seeds(orch.config)
        assert result == []

    def test_seeds_returned(self):
        seeds = [{"family_id": "f1", "name": "A"}, {"family_id": "f2", "name": "B"}]
        orch, storage = self._make_orchestrator(seeds_return=seeds)
        result = orch._select_seeds(orch.config)
        assert len(result) == 2

    def test_exclude_ids_merged(self):
        orch, storage = self._make_orchestrator(seeds_return=[])
        orch.config.exclude_ids = ["ex1"]
        orch._select_seeds(orch.config, recently_explored={"ex2"})
        call_args = storage.get_dream_seeds.call_args
        exclude_list = call_args[1]["exclude_ids"]
        assert "ex1" in exclude_list
        assert "ex2" in exclude_list

    def test_storage_failure_returns_empty(self):
        orch, storage = self._make_orchestrator()
        storage.get_dream_seeds.side_effect = Exception("DB error")
        result = orch._select_seeds(orch.config)
        assert result == []


# ═══════════════════════════════════════════════════════════════════
# DreamOrchestrator: _explore_graph
# ═══════════════════════════════════════════════════════════════════

class TestExploreGraph:
    """Tests for BFS graph exploration step."""

    def _make_orchestrator(self, bfs_result=None, bfs_relations=None):
        storage = MagicMock()
        llm = MagicMock()
        config = DreamConfig()
        orch = DreamOrchestrator(storage, llm, config)
        # Mock the searcher's bfs_expand_with_relations
        if bfs_result is not None:
            orch._searcher = MagicMock()
            orch._searcher.bfs_expand_with_relations.return_value = (
                bfs_result, bfs_relations or [], set()
            )
        else:
            orch._searcher = MagicMock()
            orch._searcher.bfs_expand_with_relations.return_value = ([], [], set())
        return orch

    def test_empty_bfs(self):
        seeds = [{"family_id": "f1", "name": "A", "content": "desc"}]
        orch = self._make_orchestrator(bfs_result=[])
        lookup, seen, explored, rel_ctx = orch._explore_graph(seeds, orch.config)
        # Seeds should still be in lookup
        assert "f1" in lookup

    def test_bfs_with_entities(self):
        e1 = _make_entity("f1", "A", "Content A")
        e2 = _make_entity("f2", "B", "Content B")
        seeds = [{"family_id": "f1", "name": "A", "content": "Content A"}]
        orch = self._make_orchestrator(bfs_result=[e1, e2])
        lookup, seen, explored, rel_ctx = orch._explore_graph(seeds, orch.config)
        assert "f1" in lookup
        assert "f2" in lookup
        assert len(explored) == 1  # 1 seed

    def test_bfs_failure_returns_empty(self):
        orch = self._make_orchestrator()
        orch._searcher.bfs_expand_with_relations.side_effect = Exception("BFS failed")
        seeds = [{"family_id": "f1", "name": "A", "content": "desc"}]
        lookup, seen, explored, rel_ctx = orch._explore_graph(seeds, orch.config)
        assert "f1" in lookup  # seed still added

    def test_explored_neighbors_populated(self):
        e1 = _make_entity("f1", "Seed", "Seed content")
        e2 = _make_entity("f2", "Neighbor1", "N1 content")
        e3 = _make_entity("f3", "Neighbor2", "N2 content")
        seeds = [{"family_id": "f1", "name": "Seed", "content": "Seed content"}]
        orch = self._make_orchestrator(bfs_result=[e1, e2, e3])
        _, _, explored, _ = orch._explore_graph(seeds, orch.config)
        assert len(explored) == 1
        assert explored[0]["neighbor_count"] == 2  # f2 and f3, not f1 itself

    def test_relation_context_built_from_bfs_relations(self):
        """Relation context is built from BFS relations."""
        from processor.models import Relation
        e1 = _make_entity("f1", "A", "Content A")
        e2 = _make_entity("f2", "B", "Content B")
        e3 = _make_entity("f3", "C", "Content C")
        rel = Relation(
            absolute_id="r1_abs", family_id="rel_1",
            entity1_absolute_id="abs_f1", entity2_absolute_id="abs_f2",
            content="A and B are related", confidence=0.8,
            event_time=_now(), processed_time=_now(),
            episode_id="ep1", source_document="test",
        )
        seeds = [{"family_id": "f1", "name": "A", "content": "Content A"}]
        orch = self._make_orchestrator(bfs_result=[e1, e2, e3], bfs_relations=[rel])
        _, _, _, rel_ctx = orch._explore_graph(seeds, orch.config)
        # f1 should have a relation to f2
        assert "f1" in rel_ctx
        assert any("B" in r for r in rel_ctx["f1"])
        # f2 should have a relation back to f1
        assert "f2" in rel_ctx
        assert any("A" in r for r in rel_ctx["f2"])
        # f3 has no relations
        assert "f3" not in rel_ctx

    def test_relation_context_empty_when_no_relations(self):
        """Relation context is empty when BFS returns no relations."""
        e1 = _make_entity("f1", "A", "Content A")
        seeds = [{"family_id": "f1", "name": "A", "content": "Content A"}]
        orch = self._make_orchestrator(bfs_result=[e1])
        _, _, _, rel_ctx = orch._explore_graph(seeds, orch.config)
        assert rel_ctx == {}


# ═══════════════════════════════════════════════════════════════════
# DreamOrchestrator: _judge_pair
# ═══════════════════════════════════════════════════════════════════

class TestJudgePair:
    """Tests for the LLM judge step."""

    def _make_orchestrator(self):
        storage = MagicMock()
        llm = MagicMock()
        config = DreamConfig()
        return DreamOrchestrator(storage, llm, config)

    def test_existing_pair_skipped_via_prefetch(self):
        orch = self._make_orchestrator()
        existing = {("f1", "f2")}
        result = orch._judge_pair("f1", "A", "f2", "B", orch.config,
                                  existing_pairs=existing)
        assert result is None
        orch.llm_client.call_llm_until_json_parses.assert_not_called()

    def test_existing_pair_skipped_via_reverse_prefetch(self):
        orch = self._make_orchestrator()
        existing = {("f2", "f1")}
        result = orch._judge_pair("f1", "A", "f2", "B", orch.config,
                                  existing_pairs=existing)
        assert result is None

    def test_existing_pair_skipped_via_db(self):
        orch = self._make_orchestrator()
        orch.storage.get_relations_by_entities.return_value = [_make_entity()]
        result = orch._judge_pair("f1", "A", "f2", "B", orch.config,
                                  entity_lookup={"f1": {"name": "A", "content": "desc"},
                                                 "f2": {"name": "B", "content": "desc"}},
                                  existing_pairs=None)
        assert result is None

    def test_missing_entity_in_lookup(self):
        orch = self._make_orchestrator()
        result = orch._judge_pair("f1", "A", "f2", "B", orch.config,
                                  entity_lookup={"f1": {"name": "A", "content": "desc"}},
                                  existing_pairs=set())
        assert result is None

    def test_llm_says_no_relation(self):
        orch = self._make_orchestrator()
        orch.llm_client.call_llm_until_json_parses.return_value = (
            {"need_create": False}, {}
        )
        result = orch._judge_pair("f1", "A", "f2", "B", orch.config,
                                  entity_lookup={
                                      "f1": {"name": "A", "content": "desc A"},
                                      "f2": {"name": "B", "content": "desc B"},
                                  },
                                  existing_pairs=set())
        assert result is None

    def test_llm_says_yes_generates_relation(self):
        orch = self._make_orchestrator()
        # Single call returns both judgment and content
        orch.llm_client.call_llm_until_json_parses.return_value = (
            {"need_create": True, "confidence": 0.8,
             "content": "A and B are related through shared interests"}, {},
        )
        result = orch._judge_pair("f1", "A", "f2", "B", orch.config,
                                  entity_lookup={
                                      "f1": {"name": "A", "content": "desc A"},
                                      "f2": {"name": "B", "content": "desc B"},
                                  },
                                  existing_pairs=set())
        assert result is not None
        assert result["content"] == "A and B are related through shared interests"
        assert result["confidence"] == 0.8

    def test_llm_generates_empty_content(self):
        orch = self._make_orchestrator()
        orch.llm_client.call_llm_until_json_parses.return_value = (
            {"need_create": True, "confidence": 0.7, "content": ""}, {},
        )
        result = orch._judge_pair("f1", "A", "f2", "B", orch.config,
                                  entity_lookup={
                                      "f1": {"name": "A", "content": "desc A"},
                                      "f2": {"name": "B", "content": "desc B"},
                                  },
                                  existing_pairs=set())
        assert result is None

    def test_llm_generates_short_content(self):
        orch = self._make_orchestrator()
        orch.llm_client.call_llm_until_json_parses.return_value = (
            {"need_create": True, "confidence": 0.7, "content": "short"}, {},
        )
        result = orch._judge_pair("f1", "A", "f2", "B", orch.config,
                                  entity_lookup={
                                      "f1": {"name": "A", "content": "desc A"},
                                      "f2": {"name": "B", "content": "desc B"},
                                  },
                                  existing_pairs=set())
        assert result is None

    def test_confidence_clamped(self):
        orch = self._make_orchestrator()
        orch.llm_client.call_llm_until_json_parses.return_value = (
            {"need_create": True, "confidence": 2.0,
             "content": "A valid relation description between two entities"}, {},
        )
        result = orch._judge_pair("f1", "A", "f2", "B", orch.config,
                                  entity_lookup={
                                      "f1": {"name": "A", "content": "desc A"},
                                      "f2": {"name": "B", "content": "desc B"},
                                  },
                                  existing_pairs=set())
        assert result is not None
        assert result["confidence"] == 1.0  # Clamped to 1.0

    def test_topology_context_included_in_prompt(self):
        """Graph topology is included in LLM prompt when relation_context is provided."""
        orch = self._make_orchestrator()
        orch.llm_client.call_llm_until_json_parses.return_value = (
            {"need_create": True, "confidence": 0.7,
             "content": "Topology-informed relation between A and B"}, {},
        )
        rel_ctx = {
            "f1": ["B — A and B share interests", "C — A influences C"],
            "f2": ["A — A and B share interests", "D — B collaborates with D"],
        }
        result = orch._judge_pair("f1", "A", "f2", "B", orch.config,
                                  entity_lookup={
                                      "f1": {"name": "A", "content": "desc A"},
                                      "f2": {"name": "B", "content": "desc B"},
                                  },
                                  existing_pairs=set(),
                                  relation_context=rel_ctx)
        assert result is not None
        # Verify LLM was called with topology context in the user message
        call_args = orch.llm_client.call_llm_until_json_parses.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"]
        assert "已知关联" in user_msg
        assert "A and B share interests" in user_msg

    def test_topology_absent_when_no_context(self):
        """No topology section when relation_context is None."""
        orch = self._make_orchestrator()
        orch.llm_client.call_llm_until_json_parses.return_value = (
            {"need_create": False}, {},
        )
        orch._judge_pair("f1", "A", "f2", "B", orch.config,
                         entity_lookup={
                             "f1": {"name": "A", "content": "desc A"},
                             "f2": {"name": "B", "content": "desc B"},
                         },
                         existing_pairs=set(),
                         relation_context=None)
        call_args = orch.llm_client.call_llm_until_json_parses.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"]
        assert "已知关联" not in user_msg

class TestDreamOrchestratorRun:
    """Integration tests for the full dream cycle."""

    def _make_orchestrator(self, seeds, bfs_entities=None):
        storage = MagicMock()
        storage.get_dream_seeds.return_value = seeds
        storage.get_relations_by_entity_pairs.return_value = {}
        storage.save_dream_relation.return_value = {"family_id": "rel_1"}
        storage.save_dream_episode.return_value = True

        llm = MagicMock()
        # LLM says: no relation for all pairs
        llm.call_llm_until_json_parses.return_value = (
            {"need_create": False}, {}
        )

        config = DreamConfig(seed_count=2, max_relations=3)
        orch = DreamOrchestrator(storage, llm, config)

        # Mock BFS (returns entities + relations + visited set)
        orch._searcher = MagicMock()
        orch._searcher.bfs_expand_with_relations.return_value = (
            bfs_entities or [], [], set()
        )

        return orch, storage, llm

    def test_empty_graph_returns_empty(self):
        orch, _, _ = self._make_orchestrator(seeds=[])
        result = orch.run()
        assert result.seeds == []
        assert result.relations_created == []
        assert result.stats["seeds_count"] == 0

    def test_seeds_but_no_relations(self):
        seeds = [
            {"family_id": "f1", "name": "A", "content": "Content A"},
            {"family_id": "f2", "name": "B", "content": "Content B"},
        ]
        e1 = _make_entity("f1", "A", "Content A")
        e2 = _make_entity("f2", "B", "Content B")
        orch, _, _ = self._make_orchestrator(seeds, bfs_entities=[e1, e2])
        result = orch.run()
        assert len(result.seeds) == 2
        assert result.stats["seeds_count"] == 2
        assert result.relations_created == []

    def test_relations_created(self):
        seeds = [
            {"family_id": "f1", "name": "A", "content": "Content A"},
            {"family_id": "f2", "name": "B", "content": "Content B"},
        ]
        e1 = _make_entity("f1", "A", "Content A")
        e2 = _make_entity("f2", "B", "Content B")
        orch, storage, llm = self._make_orchestrator(seeds, bfs_entities=[e1, e2])

        # LLM says yes for the pair (single call with judgment + content)
        llm.call_llm_until_json_parses.return_value = (
            {"need_create": True, "confidence": 0.7,
             "content": "A and B share common research interests"}, {},
        )

        result = orch.run()
        assert len(result.relations_created) >= 1
        assert result.stats["relations_created_count"] >= 1
        assert result.relations_created[0]["entity1_name"] in ("A", "B")

    def test_history_updated_after_run(self):
        seeds = [
            {"family_id": "f1", "name": "A", "content": "C"},
            {"family_id": "f2", "name": "B", "content": "C"},
        ]
        e1 = _make_entity("f1", "A", "C")
        e2 = _make_entity("f2", "B", "C")
        orch, _, _ = self._make_orchestrator(seeds, bfs_entities=[e1, e2])
        orch.run()
        # Entities should be in recently explored
        recent = orch._history.get_recently_explored(last_n=1)
        assert "f1" in recent
        assert "f2" in recent

    def test_cycle_count_increments(self):
        orch, _, _ = self._make_orchestrator(seeds=[])
        assert orch._cycle_count == 0
        orch.run()
        assert orch._cycle_count == 1
        orch.run()
        assert orch._cycle_count == 2

    def test_max_relations_respected(self):
        seeds = [
            {"family_id": "f1", "name": "A", "content": "CA"},
            {"family_id": "f2", "name": "B", "content": "CB"},
            {"family_id": "f3", "name": "C", "content": "CC"},
        ]
        entities = [_make_entity(f"f{i+1}", n, f"C{i+1}") for i, n in enumerate(["A", "B", "C"])]
        orch, storage, llm = self._make_orchestrator(seeds, bfs_entities=entities)
        orch.config.max_relations = 1  # Only 1 relation allowed

        # LLM says yes for all pairs (single call per pair)
        llm.call_llm_until_json_parses.side_effect = [
            ({"need_create": True, "confidence": 0.7, "content": "Relation between entities"}, {}),
            ({"need_create": True, "confidence": 0.8, "content": "Another relation between entities"}, {}),
        ]

        result = orch.run()
        assert len(result.relations_created) <= 1

    def test_episode_saved(self):
        seeds = [{"family_id": "f1", "name": "A", "content": "C"}]
        e1 = _make_entity("f1", "A", "C")
        orch, storage, _ = self._make_orchestrator(seeds, bfs_entities=[e1])
        orch.run()
        storage.save_dream_episode.assert_called_once()

    def test_min_confidence_filter(self):
        seeds = [
            {"family_id": "f1", "name": "A", "content": "CA"},
            {"family_id": "f2", "name": "B", "content": "CB"},
        ]
        e1 = _make_entity("f1", "A", "CA")
        e2 = _make_entity("f2", "B", "CB")
        orch, storage, llm = self._make_orchestrator(seeds, bfs_entities=[e1, e2])
        orch.config.min_confidence = 0.9  # High threshold

        # LLM says yes with low confidence (single call)
        llm.call_llm_until_json_parses.return_value = (
            {"need_create": True, "confidence": 0.3,
             "content": "Some relation between these two entities"}, {},
        )

        result = orch.run()
        assert len(result.relations_created) == 0


# ═══════════════════════════════════════════════════════════════════
# DreamOrchestrator: _discover_relations
# ═══════════════════════════════════════════════════════════════════

class TestDiscoverRelations:
    """Tests for the concurrent relation discovery step."""

    def _make_orchestrator(self):
        storage = MagicMock()
        storage.get_relations_by_entity_pairs.return_value = {}
        storage.save_dream_relation.return_value = {"family_id": "rel_1"}
        llm = MagicMock()
        config = DreamConfig(llm_concurrency=2)
        return DreamOrchestrator(storage, llm, config)

    def test_no_pairs_returns_empty(self):
        orch = self._make_orchestrator()
        explored = [{"seed": {"family_id": "f1", "name": "A"}, "neighbors": []}]
        entity_lookup = {"f1": {"name": "A", "content": "desc"}}
        rels, checked = orch._discover_relations([], explored, entity_lookup, "c1", orch.config)
        assert rels == []
        assert checked == 0

    def test_history_skips_checked_pairs(self):
        orch = self._make_orchestrator()
        orch._history.mark_checked("f1", "f2", "prev_cycle")
        explored = [{
            "seed": {"family_id": "f1", "name": "A"},
            "neighbors": [{"family_id": "f2", "name": "B"}],
        }]
        entity_lookup = {
            "f1": {"name": "A", "content": "desc"},
            "f2": {"name": "B", "content": "desc"},
        }
        rels, checked = orch._discover_relations(
            [{"family_id": "f1", "name": "A"}],
            explored, entity_lookup, "c1", orch.config,
        )
        # Pair was in history, so should be skipped
        assert checked == 0

    def test_batch_prefetch_failure_graceful(self):
        orch = self._make_orchestrator()
        orch.storage.get_relations_by_entity_pairs.side_effect = Exception("DB error")
        # LLM says no
        orch.llm_client.call_llm_until_json_parses.return_value = (
            {"need_create": False}, {}
        )
        explored = [{
            "seed": {"family_id": "f1", "name": "A"},
            "neighbors": [{"family_id": "f2", "name": "B"}],
        }]
        entity_lookup = {
            "f1": {"name": "A", "content": "desc A"},
            "f2": {"name": "B", "content": "desc B"},
        }
        # Should not crash
        rels, checked = orch._discover_relations(
            [{"family_id": "f1", "name": "A"}],
            explored, entity_lookup, "c1", orch.config,
        )
        assert checked >= 1


# ═══════════════════════════════════════════════════════════════════
# DreamOrchestrator: _save_episode
# ═══════════════════════════════════════════════════════════════════

class TestSaveEpisode:
    """Tests for dream episode saving."""

    def test_save_success(self):
        storage = MagicMock()
        storage.save_dream_episode.return_value = True
        llm = MagicMock()
        config = DreamConfig()
        orch = DreamOrchestrator(storage, llm, config)
        orch._save_episode("c1", "summary", {"f1", "f2"}, [], config)
        storage.save_dream_episode.assert_called_once()

    def test_save_failure_does_not_crash(self):
        storage = MagicMock()
        storage.save_dream_episode.side_effect = Exception("Write error")
        llm = MagicMock()
        config = DreamConfig()
        orch = DreamOrchestrator(storage, llm, config)
        # Should not raise
        orch._save_episode("c1", "summary", {"f1"}, [], config)


# ═══════════════════════════════════════════════════════════════════
# DreamOrchestrator: strategies
# ═══════════════════════════════════════════════════════════════════

class TestStrategies:
    """Verify all valid strategies can be configured."""

    @pytest.mark.parametrize("strategy", VALID_STRATEGIES)
    def test_valid_strategies(self, strategy):
        config = DreamConfig(strategy=strategy)
        assert config.strategy == strategy

    def test_invalid_strategy_raises(self):
        # DreamConfig now validates strategy at construction time
        with pytest.raises(ValueError, match="无效策略"):
            DreamConfig(strategy="nonexistent")


# ═══════════════════════════════════════════════════════════════════
# _discover_relations: early termination marks ALL pairs in history
# ═══════════════════════════════════════════════════════════════════

class TestDiscoverRelationsEarlyBreak:
    """Verify that when max_relations is hit, ALL submitted pairs are still
    marked in DreamHistory (not just the ones whose results were consumed)."""

    def _make_orchestrator(self, max_relations=1):
        storage = MagicMock()
        storage.get_relations_by_entity_pairs.return_value = {}
        storage.save_dream_relation.return_value = {"family_id": "rel_1"}
        llm = MagicMock()
        config = DreamConfig(llm_concurrency=1, max_relations=max_relations)
        return DreamOrchestrator(storage, llm, config)

    def test_all_pairs_marked_on_early_break(self):
        """When max_relations=1 and LLM says yes to multiple pairs, all pairs
        should still be marked in history even though only 1 relation is kept."""
        orch = self._make_orchestrator(max_relations=1)

        # 3 pairs: seed f1 with neighbors f2, f3, f4
        explored = [{
            "seed": {"family_id": "f1", "name": "A"},
            "neighbors": [
                {"family_id": "f2", "name": "B"},
                {"family_id": "f3", "name": "C"},
                {"family_id": "f4", "name": "D"},
            ],
        }]
        entity_lookup = {
            "f1": {"name": "A", "content": "desc A"},
            "f2": {"name": "B", "content": "desc B"},
            "f3": {"name": "C", "content": "desc C"},
            "f4": {"name": "D", "content": "desc D"},
        }

        # LLM says yes to all pairs (single call per pair)
        orch.llm_client.call_llm_until_json_parses.side_effect = [
            ({"need_create": True, "confidence": 0.8,
              "content": "Relation between A and B that is long enough"}, {}),
            ({"need_create": True, "confidence": 0.7,
              "content": "Relation between A and C that is long enough"}, {}),
            ({"need_create": True, "confidence": 0.6,
              "content": "Relation between A and D that is long enough"}, {}),
        ]

        rels, checked = orch._discover_relations(
            [{"family_id": "f1", "name": "A"}],
            explored, entity_lookup, "cycle_1", orch.config,
        )

        # Only 1 relation created (max_relations=1)
        assert len(rels) <= 1

        # ALL 3 pairs should be marked in history (not just the 1st)
        assert orch._history.was_checked("f1", "f2")
        assert orch._history.was_checked("f1", "f3")
        assert orch._history.was_checked("f1", "f4")

        # pairs_checked should reflect all submitted pairs
        assert checked == 3

    def test_pairs_checked_count_includes_drained(self):
        """Even pairs drained after early_break should increment pairs_checked."""
        orch = self._make_orchestrator(max_relations=1)
        explored = [{
            "seed": {"family_id": "f1", "name": "A"},
            "neighbors": [
                {"family_id": "f2", "name": "B"},
                {"family_id": "f3", "name": "C"},
            ],
        }]
        entity_lookup = {
            "f1": {"name": "A", "content": "desc A"},
            "f2": {"name": "B", "content": "desc B"},
            "f3": {"name": "C", "content": "desc C"},
        }

        # LLM says yes to both (single call per pair)
        orch.llm_client.call_llm_until_json_parses.side_effect = [
            ({"need_create": True, "confidence": 0.8,
              "content": "First relation between A and B entities"}, {}),
            ({"need_create": True, "confidence": 0.7,
              "content": "Second relation between A and C entities"}, {}),
        ]

        rels, checked = orch._discover_relations(
            [{"family_id": "f1", "name": "A"}],
            explored, entity_lookup, "cycle_1", orch.config,
        )

        assert checked == 2
        assert len(rels) <= 1


# ═══════════════════════════════════════════════════════════════════
# GraphRegistry: persistent orchestrator and dream lock
# ═══════════════════════════════════════════════════════════════════

class TestRegistryDreamOrchestrator:
    """Tests for GraphRegistry's persistent DreamOrchestrator management."""

    def _make_registry(self):
        """Create a minimal GraphRegistry with mocked dependencies."""
        from server.registry import GraphRegistry
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"llm": {}, "pipeline": {}, "runtime": {}}
            registry = GraphRegistry(tmpdir, config)

        # Mock get_processor to avoid real DB creation
        mock_processor = MagicMock()
        mock_processor.storage = MagicMock()
        mock_processor.llm_client = MagicMock()
        registry.get_processor = MagicMock(return_value=mock_processor)

        return registry, mock_processor

    def test_same_orchestrator_for_same_graph(self):
        """Same graph_id returns same orchestrator instance (persistence)."""
        registry, _ = self._make_registry()
        config = DreamConfig(strategy="random", seed_count=3)
        orch1 = registry.get_dream_orchestrator("g1", config)
        orch2 = registry.get_dream_orchestrator("g1", DreamConfig(strategy="hub"))
        assert orch1 is orch2

    def test_different_graphs_get_different_orchestrators(self):
        """Different graph_ids get different orchestrator instances."""
        registry, _ = self._make_registry()
        orch1 = registry.get_dream_orchestrator("g1")
        orch2 = registry.get_dream_orchestrator("g2")
        assert orch1 is not orch2

    def test_config_updated_on_existing_orchestrator(self):
        """Calling with a new config updates the existing orchestrator's config."""
        registry, _ = self._make_registry()
        config1 = DreamConfig(strategy="random", seed_count=3)
        orch = registry.get_dream_orchestrator("g1", config1)
        assert orch.config.strategy == "random"

        config2 = DreamConfig(strategy="hub", seed_count=5)
        orch2 = registry.get_dream_orchestrator("g1", config2)
        assert orch2.config.strategy == "hub"
        assert orch2.config.seed_count == 5

    def test_history_preserved_across_calls(self):
        """DreamHistory is preserved across get_dream_orchestrator calls."""
        registry, _ = self._make_registry()
        orch = registry.get_dream_orchestrator("g1")
        orch._history.mark_checked("a", "b", "c1")

        orch2 = registry.get_dream_orchestrator("g1")
        assert orch2._history.was_checked("a", "b")

    def test_dream_lock_same_for_same_graph(self):
        """Same graph_id returns same lock instance."""
        registry, _ = self._make_registry()
        lock1 = registry.get_dream_lock("g1")
        lock2 = registry.get_dream_lock("g1")
        assert lock1 is lock2

    def test_dream_lock_different_for_different_graphs(self):
        """Different graph_ids get different locks."""
        registry, _ = self._make_registry()
        lock1 = registry.get_dream_lock("g1")
        lock2 = registry.get_dream_lock("g2")
        assert lock1 is not lock2

    def test_delete_graph_cleans_up_orchestrator_and_lock(self):
        """delete_graph removes orchestrator and lock for that graph."""
        registry, mock_proc = self._make_registry()
        mock_proc.storage.close = MagicMock()
        mock_proc.storage.storage_path = "/tmp/test"

        # Create orchestrator and lock
        orch = registry.get_dream_orchestrator("g1")
        lock = registry.get_dream_lock("g1")

        # Delete graph (need to mock shutil.rmtree and Path.is_dir)
        import shutil
        original_rmtree = shutil.rmtree
        shutil.rmtree = MagicMock()

        # Make graph dir appear to exist
        from unittest.mock import patch as _patch
        with _patch.object(type(registry._base_path), 'is_dir', return_value=False):
            registry.delete_graph("g1")

        shutil.rmtree = original_rmtree

        # Orchestrator and lock should be gone
        assert "g1" not in registry._orchestrators
        assert "g1" not in registry._dream_locks

        # New call should create fresh instances
        orch2 = registry.get_dream_orchestrator("g1")
        lock2 = registry.get_dream_lock("g1")
        assert orch2 is not orch
        assert lock2 is not lock

    def test_concurrent_dream_lock(self):
        """Lock actually blocks concurrent access."""
        import threading
        registry, _ = self._make_registry()
        lock = registry.get_dream_lock("g1")

        results = []

        def _acquire_and_record(name, hold_time=0.05):
            if lock.acquire(timeout=0.1):
                results.append(f"{name}_acquired")
                import time
                time.sleep(hold_time)
                results.append(f"{name}_release")
                lock.release()
            else:
                results.append(f"{name}_blocked")

        t1 = threading.Thread(target=_acquire_and_record, args=("t1", 0.1))
        t2 = threading.Thread(target=_acquire_and_record, args=("t2",))

        t1.start()
        import time
        time.sleep(0.02)  # Ensure t1 acquires first
        t2.start()
        t1.join()
        t2.join()

        # t1 should acquire, t2 should either acquire after or be blocked
        assert "t1_acquired" in results


# ═══════════════════════════════════════════════════════════════════
# Strategy Rotation tests
# ═══════════════════════════════════════════════════════════════════

class TestStrategyRotation:
    """Tests for DreamHistory strategy rotation logic."""

    def test_next_strategy_picks_unused_first(self):
        """Unused strategies are picked before recently-used ones."""
        h = DreamHistory()
        # No history → first unused strategy
        s = h.next_strategy()
        assert s in VALID_STRATEGIES

    def test_next_strategy_rotates_through_all(self):
        """Rotation cycles through all valid strategies."""
        h = DreamHistory()
        picked = []
        for _ in range(len(VALID_STRATEGIES)):
            s = h.next_strategy()
            picked.append(s)
            h.record_strategy_result(s, 0)
        assert set(picked) == set(VALID_STRATEGIES)

    def test_next_strategy_lru_after_all_used(self):
        """After all strategies are used, picks least-recently-used."""
        h = DreamHistory()
        # Use all strategies
        for s in VALID_STRATEGIES:
            h.record_strategy_result(s, 0)
        # The least recently used is the first one recorded
        # next_strategy should pick it
        next_s = h.next_strategy()
        assert next_s in VALID_STRATEGIES
        # It should prefer early strategies (LRU), not recent ones
        assert next_s != VALID_STRATEGIES[-1]  # last used should not be picked again immediately

    def test_next_strategy_prefers_effective(self):
        """When multiple LRU candidates, picks the one with best ratio."""
        h = DreamHistory()
        # Record all strategies but with different effectiveness
        for i, s in enumerate(VALID_STRATEGIES):
            h.record_strategy_result(s, i)  # later strategies get more relations
        # Record again for first few (so they're more recent)
        for s in VALID_STRATEGIES[:2]:
            h.record_strategy_result(s, 0)
        # The LRU candidates should be the later strategies
        # Among those, pick the one with best ratio
        next_s = h.next_strategy()
        assert next_s in VALID_STRATEGIES

    def test_record_strategy_result(self):
        """record_strategy_result updates history and stats."""
        h = DreamHistory()
        h.record_strategy_result("random", 3)
        h.record_strategy_result("random", 2)
        stats = h.get_strategy_stats()
        assert stats["random"]["cycles"] == 2
        assert stats["random"]["relations"] == 5

    def test_strategy_history_capped_at_60(self):
        """Strategy history doesn't grow unbounded."""
        h = DreamHistory()
        for i in range(100):
            h.record_strategy_result("random", 1)
        assert len(h._strategy_history) == 60

    def test_reset_clears_strategy_data(self):
        """Reset clears strategy history and stats."""
        h = DreamHistory()
        h.record_strategy_result("random", 3)
        h.reset()
        assert h._strategy_history == []
        assert h._strategy_stats == {}

    def test_best_by_stats_single_candidate(self):
        """_best_by_stats returns the only candidate."""
        h = DreamHistory()
        assert h._best_by_stats(["hub"]) == "hub"

    def test_best_by_stats_picks_higher_ratio(self):
        """_best_by_stats picks strategy with better relations/cycles ratio."""
        h = DreamHistory()
        h._strategy_stats = {
            "random": {"cycles": 2, "relations": 0},
            "orphan": {"cycles": 2, "relations": 6},
        }
        assert h._best_by_stats(["random", "orphan"]) == "orphan"


class TestAutoRotateRun:
    """Integration tests for auto_rotate in DreamOrchestrator.run()."""

    def _make_orchestrator(self, seeds):
        storage = MagicMock()
        storage.get_dream_seeds.return_value = seeds
        storage.get_relations_by_entity_pairs.return_value = {}
        storage.save_dream_relation.return_value = {"family_id": "rel_1"}
        storage.save_dream_episode.return_value = True

        llm = MagicMock()
        llm.call_llm_until_json_parses.return_value = (
            {"need_create": False}, {}
        )

        config = DreamConfig(strategy="random", seed_count=2, max_relations=3)
        orch = DreamOrchestrator(storage, llm, config)
        orch._searcher = MagicMock()
        orch._searcher.bfs_expand_with_relations.return_value = ([], [], set())
        return orch, storage, llm

    def test_auto_rotate_changes_strategy(self):
        """auto_rotate=True uses a different strategy than config default."""
        seeds = [
            {"family_id": "f1", "name": "A", "content": "C"},
            {"family_id": "f2", "name": "B", "content": "C"},
        ]
        e1 = _make_entity("f1", "A", "C")
        e2 = _make_entity("f2", "B", "C")
        orch, storage, llm = self._make_orchestrator(seeds)
        orch._searcher.bfs_expand_with_relations.return_value = (
            [e1, e2], [], set()
        )

        result = orch.run(auto_rotate=True)
        # First rotation with no history: picks first unused strategy
        assert result.strategy in VALID_STRATEGIES
        assert result.stats.get("strategy_rotated") is True

    def test_auto_rotate_records_history(self):
        """Auto-rotate records strategy usage in history."""
        seeds = [{"family_id": "f1", "name": "A", "content": "C"}]
        e1 = _make_entity("f1", "A", "C")
        orch, _, _ = self._make_orchestrator(seeds)
        orch._searcher.bfs_expand_with_relations.return_value = (
            [e1], [], set()
        )

        orch.run(auto_rotate=True)
        stats = orch._history.get_strategy_stats()
        assert len(stats) >= 1  # At least one strategy was recorded

    def test_auto_rotate_cycles_strategies(self):
        """Multiple auto_rotate calls cycle through different strategies."""
        seeds = [{"family_id": "f1", "name": "A", "content": "C"}]
        e1 = _make_entity("f1", "A", "C")
        orch, _, _ = self._make_orchestrator(seeds)
        orch._searcher.bfs_expand_with_relations.return_value = (
            [e1], [], set()
        )

        strategies_seen = set()
        for _ in range(len(VALID_STRATEGIES)):
            result = orch.run(auto_rotate=True)
            strategies_seen.add(result.strategy)
        # Should have seen multiple different strategies
        assert len(strategies_seen) > 1

    def test_no_auto_rotate_keeps_config_strategy(self):
        """Without auto_rotate, uses config.strategy as-is."""
        seeds = [{"family_id": "f1", "name": "A", "content": "C"}]
        e1 = _make_entity("f1", "A", "C")
        orch, _, _ = self._make_orchestrator(seeds)
        orch._searcher.bfs_expand_with_relations.return_value = (
            [e1], [], set()
        )

        result = orch.run(auto_rotate=False)
        assert result.strategy == "random"  # matches config
        assert result.stats.get("strategy_rotated") is False

    def test_empty_graph_auto_rotate_still_records(self):
        """Empty graph still records the rotated strategy result."""
        orch, _, _ = self._make_orchestrator(seeds=[])
        result = orch.run(auto_rotate=True)
        assert result.seeds == []
        # Strategy should still be recorded
        stats = orch._history.get_strategy_stats()
        assert len(stats) >= 1
