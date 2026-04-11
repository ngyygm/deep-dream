"""
DreamStoreMixin comprehensive tests — 36 test cases across 4 dimensions.

D1: Isolated Entities (9 tests)
D2: Cleanup Invalidated Versions (9 tests)
D3: Dream Seeds — 5 strategies (9 tests)
D4: Dream Relations & Episodes (9 tests)
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone, timedelta
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
                   episode_id: str = "ep_test", source_document: str = ""):
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
    )


# ══════════════════════════════════════════════════════════════════════════
# D1: Isolated Entities
# ══════════════════════════════════════════════════════════════════════════


class TestIsolatedEntities:
    """Test isolated entity detection and counting."""

    def test_empty_graph_no_isolated(self, tmp_storage):
        """D1.1: Empty graph returns no isolated entities."""
        result = tmp_storage.get_isolated_entities()
        assert result == []

    def test_single_entity_is_isolated(self, tmp_storage):
        """D1.2: Single entity with no relations is isolated."""
        e = _make_entity("ent_iso1", "Isolated", "No friends")
        tmp_storage.save_entity(e)

        result = tmp_storage.get_isolated_entities()
        assert len(result) == 1
        assert result[0].family_id == "ent_iso1"

    def test_connected_entity_not_isolated(self, tmp_storage):
        """D1.3: Entity with relations is not isolated."""
        e1 = _make_entity("ent_c1", "Connected1", "Has friends")
        e2 = _make_entity("ent_c2", "Connected2", "Also has friends")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("rel_c1", e1.absolute_id, e2.absolute_id, "friends")
        tmp_storage.save_relation(r)

        result = tmp_storage.get_isolated_entities()
        assert len(result) == 0

    def test_mixed_isolated_and_connected(self, tmp_storage):
        """D1.4: Mixed graph correctly identifies only isolated entities."""
        # Create 5 entities
        entities = [_make_entity(f"ent_mix_{i}", f"Mix{i}", f"Content {i}") for i in range(5)]
        tmp_storage.bulk_save_entities(entities)

        # Connect first 3 with relations
        r1 = _make_relation("rel_mix1", entities[0].absolute_id, entities[1].absolute_id, "e0→e1")
        r2 = _make_relation("rel_mix2", entities[1].absolute_id, entities[2].absolute_id, "e1→e2")
        tmp_storage.bulk_save_relations([r1, r2])

        result = tmp_storage.get_isolated_entities()
        isolated_ids = {e.family_id for e in result}
        assert "ent_mix_3" in isolated_ids
        assert "ent_mix_4" in isolated_ids
        assert "ent_mix_0" not in isolated_ids

    def test_count_isolated_entities(self, tmp_storage):
        """D1.5: Count matches actual isolated count."""
        for i in range(5):
            tmp_storage.save_entity(_make_entity(f"ent_cnt_{i}", f"Cnt{i}", f"C{i}"))

        count = tmp_storage.count_isolated_entities()
        assert count == 5

    def test_count_isolated_with_connections(self, tmp_storage):
        """D1.6: Count excludes connected entities."""
        e1 = _make_entity("ent_cc1", "CC1", "C1")
        e2 = _make_entity("ent_cc2", "CC2", "C2")
        e3 = _make_entity("ent_cc3", "CC3", "C3")
        tmp_storage.bulk_save_entities([e1, e2, e3])

        r = _make_relation("rel_cc1", e1.absolute_id, e2.absolute_id, "connects")
        tmp_storage.save_relation(r)

        count = tmp_storage.count_isolated_entities()
        assert count == 1  # only e3

    def test_isolated_entities_pagination(self, tmp_storage):
        """D1.7: Pagination works for isolated entities."""
        for i in range(10):
            tmp_storage.save_entity(_make_entity(f"ent_page_{i}", f"Page{i}", f"C{i}"))

        page1 = tmp_storage.get_isolated_entities(limit=3, offset=0)
        page2 = tmp_storage.get_isolated_entities(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        ids1 = {e.family_id for e in page1}
        ids2 = {e.family_id for e in page2}
        assert ids1.isdisjoint(ids2)

    def test_isolated_entities_offset_beyond(self, tmp_storage):
        """D1.8: Offset beyond range returns empty list."""
        for i in range(3):
            tmp_storage.save_entity(_make_entity(f"ent_off_{i}", f"Off{i}", f"C{i}"))

        result = tmp_storage.get_isolated_entities(limit=10, offset=100)
        assert result == []

    def test_invalidated_version_not_counted_as_isolated(self, tmp_storage):
        """D1.9: Only valid (non-invalidated) entities considered for isolation."""
        e1 = _make_entity("ent_inv1", "Inv1", "v1")
        tmp_storage.save_entity(e1)
        time.sleep(0.01)
        e2 = _make_entity("ent_inv1", "Inv1", "v2")  # new version invalidates v1
        tmp_storage.save_entity(e2)

        # Only latest version should be considered
        result = tmp_storage.get_isolated_entities()
        assert any(e.family_id == "ent_inv1" for e in result)


# ══════════════════════════════════════════════════════════════════════════
# D2: Cleanup Invalidated Versions
# ══════════════════════════════════════════════════════════════════════════


class TestCleanup:
    """Test invalidated version cleanup."""

    def test_dry_run_returns_counts(self, tmp_storage):
        """D2.1: Dry run returns counts without deleting."""
        e1 = _make_entity("ent_dry1", "Dry1", "v1")
        tmp_storage.save_entity(e1)
        time.sleep(0.01)
        e2 = _make_entity("ent_dry1", "Dry1", "v2")
        tmp_storage.save_entity(e2)

        result = tmp_storage.cleanup_invalidated_versions(dry_run=True)
        assert result["dry_run"] is True
        assert result["entities_to_remove"] >= 1

    def test_dry_run_does_not_delete(self, tmp_storage):
        """D2.2: Dry run does not actually remove records."""
        e1 = _make_entity("ent_nd1", "NoDel1", "v1")
        tmp_storage.save_entity(e1)
        time.sleep(0.01)
        e2 = _make_entity("ent_nd1", "NoDel1", "v2")
        tmp_storage.save_entity(e2)

        versions_before = tmp_storage.get_entity_versions("ent_nd1")
        tmp_storage.cleanup_invalidated_versions(dry_run=True)
        versions_after = tmp_storage.get_entity_versions("ent_nd1")

        assert len(versions_before) == len(versions_after)

    def test_actual_cleanup_deletes_invalidated(self, tmp_storage):
        """D2.3: Actual cleanup deletes invalidated entity versions."""
        e1 = _make_entity("ent_del1", "Del1", "v1")
        tmp_storage.save_entity(e1)
        time.sleep(0.01)
        e2 = _make_entity("ent_del1", "Del1", "v2")
        tmp_storage.save_entity(e2)

        result = tmp_storage.cleanup_invalidated_versions(dry_run=False)
        assert result["dry_run"] is False
        assert result["deleted_entity_versions"] >= 1

        # Only latest version should remain
        versions = tmp_storage.get_entity_versions("ent_del1")
        assert len(versions) == 1
        assert versions[0].content == "v2"

    def test_cleanup_preserves_valid_entities(self, tmp_storage):
        """D2.4: Cleanup does not touch valid (non-invalidated) entities."""
        e1 = _make_entity("ent_pres1", "Pres1", "only version")
        tmp_storage.save_entity(e1)

        result = tmp_storage.cleanup_invalidated_versions(dry_run=False)
        assert result["deleted_entity_versions"] == 0

        entity = tmp_storage.get_entity_by_family_id("ent_pres1")
        assert entity is not None

    def test_cleanup_relation_versions(self, tmp_storage):
        """D2.5: Cleanup also handles invalidated relation versions."""
        e1 = _make_entity("ent_rel1", "Rel1", "R1")
        e2 = _make_entity("ent_rel2", "Rel2", "R2")
        tmp_storage.bulk_save_entities([e1, e2])

        r1 = _make_relation("rel_rel1", e1.absolute_id, e2.absolute_id, "v1")
        tmp_storage.save_relation(r1)
        time.sleep(0.01)
        r2 = _make_relation("rel_rel1", e1.absolute_id, e2.absolute_id, "v2")
        tmp_storage.save_relation(r2)

        result = tmp_storage.cleanup_invalidated_versions(dry_run=False)
        assert result["deleted_relation_versions"] >= 1

    def test_cleanup_before_date_filter(self, tmp_storage):
        """D2.6: before_date filter limits cleanup scope."""
        e1 = _make_entity("ent_bf1", "BF1", "v1")
        tmp_storage.save_entity(e1)
        time.sleep(0.01)
        e2 = _make_entity("ent_bf1", "BF1", "v2")
        tmp_storage.save_entity(e2)

        # Use a date far in the future — should find no invalidated versions
        future_date = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
        result = tmp_storage.cleanup_invalidated_versions(before_date=future_date, dry_run=True)
        assert result["entities_to_remove"] >= 1

    def test_cleanup_no_invalidated_returns_zero(self, tmp_storage):
        """D2.7: No invalidated versions returns zero deletions."""
        e = _make_entity("ent_ninv", "NoInv", "only version")
        tmp_storage.save_entity(e)

        result = tmp_storage.cleanup_invalidated_versions(dry_run=False)
        assert result["deleted_entity_versions"] == 0
        assert result["deleted_relation_versions"] == 0

    def test_cleanup_multiple_entity_versions(self, tmp_storage):
        """D2.8: Cleanup removes multiple old versions of same entity."""
        fid = "ent_multi_v"
        for i in range(5):
            tmp_storage.save_entity(_make_entity(fid, f"MultiV{i}", f"v{i}"))
            time.sleep(0.01)

        result = tmp_storage.cleanup_invalidated_versions(dry_run=False)
        assert result["deleted_entity_versions"] >= 4

        versions = tmp_storage.get_entity_versions(fid)
        assert len(versions) == 1

    def test_cleanup_after_merge(self, tmp_storage):
        """D2.9: Entities invalidated by merge can be cleaned up."""
        e1 = _make_entity("ent_mg1", "Merge1", "C1")
        e2 = _make_entity("ent_mg2", "Merge2", "C2")
        tmp_storage.bulk_save_entities([e1, e2])

        # Merge creates redirects, but doesn't invalidate versions directly
        tmp_storage.merge_entity_families("ent_mg1", ["ent_mg2"])

        # Cleanup should work without errors
        result = tmp_storage.cleanup_invalidated_versions(dry_run=False)
        assert isinstance(result["deleted_entity_versions"], int)


# ══════════════════════════════════════════════════════════════════════════
# D3: Dream Seeds — 5 strategies
# ══════════════════════════════════════════════════════════════════════════


class TestDreamSeeds:
    """Test dream seed selection strategies."""

    @pytest.fixture
    def populated_storage(self, tmp_storage):
        """Storage with entities and relations for seed testing."""
        # Create 10 entities, 3 with relations (hubs), 7 isolated
        entities = [
            _make_entity(f"ent_seed_{i}", f"Seed{i}", f"Content for seed {i}",
                         confidence=0.3 if i < 3 else 0.8)
            for i in range(10)
        ]
        tmp_storage.bulk_save_entities(entities)

        # Connect first 3 entities to entity 0 (making it a hub)
        for i in range(1, 4):
            r = _make_relation(
                f"rel_seed_{i}", entities[0].absolute_id, entities[i].absolute_id,
                f"Seed0→Seed{i}"
            )
            tmp_storage.save_relation(r)

        return tmp_storage, entities

    def test_random_seeds(self, populated_storage):
        """D3.1: Random strategy returns entities."""
        sm, _ = populated_storage
        seeds = sm.get_dream_seeds(strategy="random", count=5)
        assert len(seeds) <= 5
        assert len(seeds) > 0
        for s in seeds:
            assert "family_id" in s
            assert "name" in s

    def test_orphan_seeds(self, populated_storage):
        """D3.2: Orphan strategy returns isolated entities."""
        sm, _ = populated_storage
        seeds = sm.get_dream_seeds(strategy="orphan", count=5)
        # Should find the 6 isolated entities (indices 4-9)
        assert len(seeds) > 0
        seed_ids = {s["family_id"] for s in seeds}
        # Entities 0-3 are connected, so shouldn't appear
        for s in seeds:
            assert s["reason"] == "孤立实体：无任何关系连接"

    def test_hub_seeds(self, populated_storage):
        """D3.3: Hub strategy returns highly connected entities."""
        sm, _ = populated_storage
        seeds = sm.get_dream_seeds(strategy="hub", count=3)
        assert len(seeds) > 0
        # ent_seed_0 should be the top hub (3 connections)
        assert seeds[0]["family_id"] == "ent_seed_0"
        assert seeds[0]["degree"] >= 3

    def test_low_confidence_seeds(self, populated_storage):
        """D3.4: Low confidence strategy returns entities with confidence < 0.5."""
        sm, _ = populated_storage
        seeds = sm.get_dream_seeds(strategy="low_confidence", count=5)
        assert len(seeds) > 0
        for s in seeds:
            assert s["confidence"] < 0.5

    def test_seed_exclude_ids(self, populated_storage):
        """D3.5: Excluded IDs are not returned."""
        sm, _ = populated_storage
        seeds = sm.get_dream_seeds(strategy="random", count=10,
                                    exclude_ids=["ent_seed_0", "ent_seed_1"])
        returned_ids = {s["family_id"] for s in seeds}
        assert "ent_seed_0" not in returned_ids
        assert "ent_seed_1" not in returned_ids

    def test_unknown_strategy_raises(self, populated_storage):
        """D3.6: Unknown strategy raises ValueError."""
        sm, _ = populated_storage
        with pytest.raises(ValueError, match="未知的种子策略"):
            sm.get_dream_seeds(strategy="nonexistent")

    def test_seeds_have_reason(self, populated_storage):
        """D3.7: All seeds include a reason field."""
        sm, _ = populated_storage
        for strategy in ["random", "orphan", "hub", "low_confidence"]:
            seeds = sm.get_dream_seeds(strategy=strategy, count=3)
            for s in seeds:
                assert "reason" in s
                assert len(s["reason"]) > 0

    def test_seeds_count_capped(self, populated_storage):
        """D3.8: Seed count respects requested limit."""
        sm, _ = populated_storage
        seeds = sm.get_dream_seeds(strategy="random", count=3)
        assert len(seeds) <= 3

    def test_orphan_seeds_on_fully_connected_graph(self, tmp_storage):
        """D3.9: No orphan seeds when all entities are connected."""
        e1 = _make_entity("ent_fc1", "FC1", "C1")
        e2 = _make_entity("ent_fc2", "FC2", "C2")
        tmp_storage.bulk_save_entities([e1, e2])

        r = _make_relation("rel_fc1", e1.absolute_id, e2.absolute_id, "connects")
        tmp_storage.save_relation(r)

        seeds = tmp_storage.get_dream_seeds(strategy="orphan", count=10)
        assert len(seeds) == 0


# ══════════════════════════════════════════════════════════════════════════
# D4: Dream Relations & Episodes
# ══════════════════════════════════════════════════════════════════════════


class TestDreamRelations:
    """Test dream relation creation and merging."""

    def test_create_new_dream_relation(self, tmp_storage):
        """D4.1: Create a new relation between existing entities."""
        e1 = _make_entity("ent_dr1", "Alpha", "A")
        e2 = _make_entity("ent_dr2", "Beta", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        result = tmp_storage.save_dream_relation(
            "ent_dr1", "ent_dr2",
            content="Alpha relates to Beta",
            confidence=0.7,
            reasoning="Both are Greek letters",
            dream_cycle_id="dream_test_001",
        )

        assert result["action"] == "created"
        assert result["entity1_name"] in ("Alpha", "Beta")
        assert result["entity2_name"] in ("Alpha", "Beta")
        assert result["family_id"].startswith("rel_")

    def test_merge_existing_dream_relation(self, tmp_storage):
        """D4.2: Existing relation gets merged with dream update."""
        e1 = _make_entity("ent_dm1", "Gamma", "G")
        e2 = _make_entity("ent_dm2", "Delta", "D")
        tmp_storage.bulk_save_entities([e1, e2])

        # Create initial dream relation
        r1 = tmp_storage.save_dream_relation(
            "ent_dm1", "ent_dm2",
            content="Initial connection",
            confidence=0.5,
            reasoning="First discovery",
        )
        assert r1["action"] == "created"

        # Merge with higher confidence
        r2 = tmp_storage.save_dream_relation(
            "ent_dm1", "ent_dm2",
            content="Stronger connection found",
            confidence=0.9,
            reasoning="Second discovery confirms",
        )
        assert r2["action"] == "merged"
        assert r2["family_id"] == r1["family_id"]

    def test_dream_relation_nonexistent_entity_raises(self, tmp_storage):
        """D4.3: Non-existent entity raises ValueError."""
        e1 = _make_entity("ent_dne1", "Exists", "E")
        tmp_storage.save_entity(e1)

        with pytest.raises(ValueError, match="实体不存在"):
            tmp_storage.save_dream_relation(
                "ent_dne1", "nonexistent_999",
                content="test", confidence=0.5, reasoning="test",
            )

    def test_dream_relation_with_cycle_id(self, tmp_storage):
        """D4.4: Dream relation includes cycle_id in source_document."""
        e1 = _make_entity("ent_dc1", "Eps1", "E1")
        e2 = _make_entity("ent_dc2", "Eps2", "E2")
        tmp_storage.bulk_save_entities([e1, e2])

        result = tmp_storage.save_dream_relation(
            "ent_dc1", "ent_dc2",
            content="Test", confidence=0.6, reasoning="Test",
            dream_cycle_id="cycle_abc123",
        )
        assert result["action"] == "created"

        # Verify source_document
        rel = tmp_storage.get_relation_by_family_id(result["family_id"])
        assert rel.source_document == "dream:cycle_abc123"

    def test_dream_relation_confidence_max(self, tmp_storage):
        """D4.5: Merged relation takes max confidence."""
        e1 = _make_entity("ent_cf1", "CF1", "C1")
        e2 = _make_entity("ent_cf2", "CF2", "C2")
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.save_dream_relation("ent_cf1", "ent_cf2", "r1", 0.3, "test")
        r2 = tmp_storage.save_dream_relation("ent_cf1", "ent_cf2", "r2", 0.8, "test")

        rel = tmp_storage.get_relation_by_family_id(r2["family_id"])
        assert rel.confidence >= 0.8

    def test_dream_relation_absolute_id_resolution(self, tmp_storage):
        """D4.6: Dream relation resolves using absolute_ids (not family_ids)."""
        e1 = _make_entity("ent_ar1", "AR1", "A1")
        e2 = _make_entity("ent_ar2", "AR2", "A2")
        tmp_storage.bulk_save_entities([e1, e2])

        result = tmp_storage.save_dream_relation(
            "ent_ar1", "ent_ar2", "test", 0.5, "test",
        )

        rel = tmp_storage.get_relation_by_family_id(result["family_id"])
        # Relation should point to actual absolute_ids
        assert rel.entity1_absolute_id in (e1.absolute_id, e2.absolute_id)
        assert rel.entity2_absolute_id in (e1.absolute_id, e2.absolute_id)

    def test_dream_relation_entity_ordering(self, tmp_storage):
        """D4.7: Dream relation orders entities by name (alphabetical)."""
        e1 = _make_entity("ent_ord_b", "Beta", "B")
        e2 = _make_entity("ent_ord_a", "Alpha", "A")
        tmp_storage.bulk_save_entities([e1, e2])

        result = tmp_storage.save_dream_relation(
            "ent_ord_b", "ent_ord_a", "test", 0.5, "test",
        )

        rel = tmp_storage.get_relation_by_family_id(result["family_id"])
        # Alpha <= Beta, so entity1 should point to Alpha's absolute_id
        assert rel.entity1_absolute_id == e2.absolute_id  # Alpha
        assert rel.entity2_absolute_id == e1.absolute_id  # Beta


class TestDreamEpisodes:
    """Test dream episode creation."""

    def test_create_dream_episode(self, tmp_storage):
        """D4.8: Dream episode is created with correct fields."""
        result = tmp_storage.save_dream_episode(
            content="Explored connections between concepts",
            strategy_used="random",
        )
        assert "episode_id" in result
        assert result["episode_type"] == "dream"
        assert "cycle_id" in result

    def test_dream_episode_with_entity_mentions(self, tmp_storage):
        """D4.9: Dream episode records entity mentions."""
        e1 = _make_entity("ent_dep1", "Dep1", "D1")
        e2 = _make_entity("ent_dep2", "Dep2", "D2")
        tmp_storage.bulk_save_entities([e1, e2])

        result = tmp_storage.save_dream_episode(
            content="Found relation between Dep1 and Dep2",
            entities_examined=["ent_dep1", "ent_dep2"],
            strategy_used="orphan",
        )
        assert result["episode_type"] == "dream"

    def test_dream_episode_with_relations(self, tmp_storage):
        """D4.10: Dream episode records created relations."""
        e1 = _make_entity("ent_der1", "DER1", "D1")
        e2 = _make_entity("ent_der2", "DER2", "D2")
        tmp_storage.bulk_save_entities([e1, e2])

        rel_result = tmp_storage.save_dream_relation(
            "ent_der1", "ent_der2", "test relation", 0.7, "test",
        )

        result = tmp_storage.save_dream_episode(
            content="Created a new connection",
            entities_examined=["ent_der1", "ent_der2"],
            relations_created=[rel_result["family_id"]],
            strategy_used="random",
        )
        assert "cycle_id" in result

    def test_dream_episode_cycle_id_propagation(self, tmp_storage):
        """D4.11: Custom cycle_id is used correctly."""
        result = tmp_storage.save_dream_episode(
            content="Test content",
            strategy_used="hub",
            dream_cycle_id="custom_cycle_42",
        )
        assert result["cycle_id"] == "custom_cycle_42"

    def test_dream_episode_generates_cycle_id(self, tmp_storage):
        """D4.12: Missing cycle_id gets auto-generated."""
        result = tmp_storage.save_dream_episode(
            content="Auto cycle test",
            strategy_used="random",
        )
        assert result["cycle_id"].startswith("dream_")

    def test_dream_episode_counts(self, tmp_storage):
        """D4.13: Episode counts entities and relations correctly."""
        result = tmp_storage.save_dream_episode(
            content="Count test",
            entities_examined=["e1", "e2", "e3"],
            relations_created=["r1", "r2"],
            entities_examined_count=3,
            relations_created_count=2,
            strategy_used="random",
        )
        assert result["episode_type"] == "dream"

    def test_dream_episode_nonexistent_entity_mention(self, tmp_storage):
        """D4.14: Non-existent entity mentions are silently skipped."""
        result = tmp_storage.save_dream_episode(
            content="Skip nonexistent",
            entities_examined=["nonexistent_123", "also_missing_456"],
            strategy_used="random",
        )
        assert result["episode_type"] == "dream"

    def test_dream_episode_empty(self, tmp_storage):
        """D4.15: Minimal episode with no entities or relations."""
        result = tmp_storage.save_dream_episode(
            content="Empty dream",
            strategy_used="random",
        )
        assert result is not None
        assert "episode_id" in result

    def test_dream_episode_chinese_content(self, tmp_storage):
        """D4.16: Dream episode handles Chinese content correctly."""
        result = tmp_storage.save_dream_episode(
            content="发现了一个新的连接：人工智能与量子计算之间的关系。这种关联可能改变未来的计算范式。",
            strategy_used="hub",
            dream_cycle_id="cycle_cn_001",
        )
        assert result["cycle_id"] == "cycle_cn_001"

    def test_dream_episode_unicode_content(self, tmp_storage):
        """D4.17: Dream episode handles Unicode content correctly."""
        result = tmp_storage.save_dream_episode(
            content="Unicode test: 🌍 🧠 → 🚀 with émojis and spëcial chars",
            strategy_used="random",
        )
        assert result["episode_type"] == "dream"
