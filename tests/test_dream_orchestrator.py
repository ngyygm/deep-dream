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

    def _make_orchestrator(self, bfs_result=None):
        storage = MagicMock()
        llm = MagicMock()
        config = DreamConfig()
        orch = DreamOrchestrator(storage, llm, config)
        # Mock the searcher's bfs_expand
        if bfs_result is not None:
            orch._searcher = MagicMock()
            orch._searcher.bfs_expand.return_value = bfs_result
        else:
            orch._searcher = MagicMock()
            orch._searcher.bfs_expand.return_value = []
        return orch

    def test_empty_bfs(self):
        seeds = [{"family_id": "f1", "name": "A", "content": "desc"}]
        orch = self._make_orchestrator(bfs_result=[])
        lookup, seen, explored = orch._explore_graph(seeds, orch.config)
        # Seeds should still be in lookup
        assert "f1" in lookup

    def test_bfs_with_entities(self):
        e1 = _make_entity("f1", "A", "Content A")
        e2 = _make_entity("f2", "B", "Content B")
        seeds = [{"family_id": "f1", "name": "A", "content": "Content A"}]
        orch = self._make_orchestrator(bfs_result=[e1, e2])
        lookup, seen, explored = orch._explore_graph(seeds, orch.config)
        assert "f1" in lookup
        assert "f2" in lookup
        assert len(explored) == 1  # 1 seed

    def test_bfs_failure_returns_empty(self):
        orch = self._make_orchestrator()
        orch._searcher.bfs_expand.side_effect = Exception("BFS failed")
        seeds = [{"family_id": "f1", "name": "A", "content": "desc"}]
        lookup, seen, explored = orch._explore_graph(seeds, orch.config)
        assert "f1" in lookup  # seed still added

    def test_explored_neighbors_populated(self):
        e1 = _make_entity("f1", "Seed", "Seed content")
        e2 = _make_entity("f2", "Neighbor1", "N1 content")
        e3 = _make_entity("f3", "Neighbor2", "N2 content")
        seeds = [{"family_id": "f1", "name": "Seed", "content": "Seed content"}]
        orch = self._make_orchestrator(bfs_result=[e1, e2, e3])
        _, _, explored = orch._explore_graph(seeds, orch.config)
        assert len(explored) == 1
        assert explored[0]["neighbor_count"] == 2  # f2 and f3, not f1 itself


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
        # First call: judge → yes
        # Second call: generate content
        orch.llm_client.call_llm_until_json_parses.side_effect = [
            ({"need_create": True, "confidence": 0.8}, {}),
            ({"content": "A and B are related through shared interests"}, {}),
        ]
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
        orch.llm_client.call_llm_until_json_parses.side_effect = [
            ({"need_create": True, "confidence": 0.7}, {}),
            ({"content": ""}, {}),  # Empty content
        ]
        result = orch._judge_pair("f1", "A", "f2", "B", orch.config,
                                  entity_lookup={
                                      "f1": {"name": "A", "content": "desc A"},
                                      "f2": {"name": "B", "content": "desc B"},
                                  },
                                  existing_pairs=set())
        assert result is None

    def test_llm_generates_short_content(self):
        orch = self._make_orchestrator()
        orch.llm_client.call_llm_until_json_parses.side_effect = [
            ({"need_create": True, "confidence": 0.7}, {}),
            ({"content": "short"}, {}),  # < 10 chars
        ]
        result = orch._judge_pair("f1", "A", "f2", "B", orch.config,
                                  entity_lookup={
                                      "f1": {"name": "A", "content": "desc A"},
                                      "f2": {"name": "B", "content": "desc B"},
                                  },
                                  existing_pairs=set())
        assert result is None

    def test_confidence_clamped(self):
        orch = self._make_orchestrator()
        orch.llm_client.call_llm_until_json_parses.side_effect = [
            ({"need_create": True, "confidence": 2.0}, {}),  # Over 1.0
            ({"content": "A valid relation description between two entities"}, {}),
        ]
        result = orch._judge_pair("f1", "A", "f2", "B", orch.config,
                                  entity_lookup={
                                      "f1": {"name": "A", "content": "desc A"},
                                      "f2": {"name": "B", "content": "desc B"},
                                  },
                                  existing_pairs=set())
        assert result is not None
        assert result["confidence"] == 1.0  # Clamped to 1.0


# ═══════════════════════════════════════════════════════════════════
# DreamOrchestrator: full run
# ═══════════════════════════════════════════════════════════════════

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

        # Mock BFS
        orch._searcher = MagicMock()
        orch._searcher.bfs_expand.return_value = bfs_entities or []

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

        # LLM says yes for the pair
        llm.call_llm_until_json_parses.side_effect = [
            ({"need_create": True, "confidence": 0.7}, {}),
            ({"content": "A and B share common research interests"}, {}),
        ]

        result = orch.run()
        assert len(result.relations_created) == 1
        assert result.stats["relations_created_count"] == 1
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

        # LLM says yes for all pairs
        llm.call_llm_until_json_parses.side_effect = [
            ({"need_create": True, "confidence": 0.7}, {}),
            ({"content": "Relation between entities"}, {}),
            ({"need_create": True, "confidence": 0.8}, {}),
            ({"content": "Another relation between entities"}, {}),
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

        # LLM says yes with low confidence
        llm.call_llm_until_json_parses.side_effect = [
            ({"need_create": True, "confidence": 0.3}, {}),  # Below threshold
            ({"content": "Some relation between these two entities"}, {}),
        ]

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

    def test_invalid_strategy_accepted_as_is(self):
        # DreamConfig doesn't validate strategy; the storage layer does
        config = DreamConfig(strategy="nonexistent")
        assert config.strategy == "nonexistent"
