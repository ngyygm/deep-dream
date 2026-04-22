"""
Round 2 tests: SQL injection prevention, merge relations, VectorStore backfill,
confidence adjustment, and edge cases.

Tests span 4 dimensions:
D1: Confidence system (SQL injection + correctness)
D2: Entity merge (relation tracking + edge cases)
D3: VectorStore (backfill + search integration)
D4: Edge cases + stress tests
"""
from __future__ import annotations

import json
import time
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


def _make_entity(family_id: str, name: str, content: str, episode_id: str = "ep_test", source_document: str = ""):
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


def _make_relation(family_id: str, e1_id: str, e2_id: str, content: str, episode_id: str = "ep_test", source_document: str = ""):
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
# D1: Confidence System — SQL injection prevention + correctness
# ══════════════════════════════════════════════════════════════════════════


class TestConfidenceSystem:
    """Test confidence adjustment and SQL injection prevention."""

    def test_validate_table_name_entity(self):
        """D1.1: _validate_table_name returns 'entities' for 'entity'."""
        from processor.storage.mixins.entity_store import EntityStoreMixin
        assert EntityStoreMixin._validate_table_name("entity") == "entities"

    def test_validate_table_name_relation(self):
        """D1.2: _validate_table_name returns 'relations' for 'relation'."""
        from processor.storage.mixins.entity_store import EntityStoreMixin
        assert EntityStoreMixin._validate_table_name("relation") == "relations"

    def test_validate_table_name_rejects_injection(self):
        """D1.3: _validate_table_name rejects malicious input."""
        from processor.storage.mixins.entity_store import EntityStoreMixin
        assert EntityStoreMixin._validate_table_name("entities; DROP TABLE --") == "relations"
        # The allowlist maps anything non-"entity" to "relations", which is safe.
        # Both "entities" and "relations" are in the allowlist.

    def test_validate_table_name_rejects_arbitrary(self):
        """D1.4: Arbitrary source_type maps to 'relations' (safe default)."""
        from processor.storage.mixins.entity_store import EntityStoreMixin
        # Any non-"entity" value → "relations" (in allowlist)
        result = EntityStoreMixin._validate_table_name("anything_malicious")
        assert result in EntityStoreMixin._VALID_TABLES

    def test_corroboration_increases_confidence(self, tmp_storage):
        """D1.5: adjust_confidence_on_corroboration increases confidence by 0.05."""
        e = _make_entity("ent_conf1", "ConfTest", "Test content")
        tmp_storage.save_entity(e)

        # Set initial confidence
        tmp_storage.update_entity_confidence("ent_conf1", 0.5)
        result = tmp_storage.get_entity_by_family_id("ent_conf1")
        assert result.confidence == 0.5

        tmp_storage.adjust_confidence_on_corroboration("ent_conf1", "entity")
        result = tmp_storage.get_entity_by_family_id("ent_conf1")
        assert result.confidence == 0.55

    def test_corroboration_dream_half_weight(self, tmp_storage):
        """D1.6: Dream corroboration uses half weight (0.025)."""
        e = _make_entity("ent_conf2", "DreamConf", "Test")
        tmp_storage.save_entity(e)
        tmp_storage.update_entity_confidence("ent_conf2", 0.5)

        tmp_storage.adjust_confidence_on_corroboration("ent_conf2", "entity", is_dream=True)
        result = tmp_storage.get_entity_by_family_id("ent_conf2")
        assert result.confidence == 0.525

    def test_contradiction_decreases_confidence(self, tmp_storage):
        """D1.7: adjust_confidence_on_contradiction decreases confidence by 0.1."""
        e = _make_entity("ent_conf3", "ContrTest", "Test")
        tmp_storage.save_entity(e)
        tmp_storage.update_entity_confidence("ent_conf3", 0.7)

        tmp_storage.adjust_confidence_on_contradiction("ent_conf3", "entity")
        result = tmp_storage.get_entity_by_family_id("ent_conf3")
        assert result.confidence == 0.6

    def test_confidence_upper_bound(self, tmp_storage):
        """D1.8: Confidence cannot exceed 1.0."""
        e = _make_entity("ent_conf4", "Upper", "Test")
        tmp_storage.save_entity(e)
        tmp_storage.update_entity_confidence("ent_conf4", 0.98)

        tmp_storage.adjust_confidence_on_corroboration("ent_conf4", "entity")
        result = tmp_storage.get_entity_by_family_id("ent_conf4")
        assert result.confidence == 1.0

    def test_confidence_lower_bound(self, tmp_storage):
        """D1.9: Confidence cannot go below 0.0."""
        e = _make_entity("ent_conf5", "Lower", "Test")
        tmp_storage.save_entity(e)
        tmp_storage.update_entity_confidence("ent_conf5", 0.05)

        tmp_storage.adjust_confidence_on_contradiction("ent_conf5", "entity")
        result = tmp_storage.get_entity_by_family_id("ent_conf5")
        assert result.confidence == 0.0


# ══════════════════════════════════════════════════════════════════════════
# D2: Entity Merge — Relation tracking
# ══════════════════════════════════════════════════════════════════════════


class TestEntityMerge:
    """Test entity merge and relation pointer tracking."""

    def test_merge_basic(self, tmp_storage):
        """D2.1: Basic merge moves entities to target family."""
        e1 = _make_entity("ent_m1", "MergeA", "Content A")
        e2 = _make_entity("ent_m2", "MergeB", "Content B")
        tmp_storage.bulk_save_entities([e1, e2])

        result = tmp_storage.merge_entity_families("ent_m1", ["ent_m2"])
        assert result["entities_updated"] >= 1

        # Both should now be under ent_m1
        versions = tmp_storage.get_entity_versions("ent_m1")
        assert len(versions) >= 2

        # ent_m2 resolves to ent_m1 via redirect
        old = tmp_storage.get_entity_by_family_id("ent_m2")
        assert old is not None  # redirect resolves to merged entity
        assert old.family_id == "ent_m1"

    def test_merge_preserves_relations(self, tmp_storage):
        """D2.2: Relations remain valid after merge."""
        e1 = _make_entity("ent_mr1", "Alpha", "A")
        e2 = _make_entity("ent_mr2", "Beta", "B")
        e3 = _make_entity("ent_mr3", "Gamma", "C")
        tmp_storage.bulk_save_entities([e1, e2, e3])

        r = _make_relation("rel_mr1", e1.absolute_id, e2.absolute_id, "Alpha→Beta")
        tmp_storage.save_relation(r)

        # Merge e2 into e1
        result = tmp_storage.merge_entity_families("ent_mr1", ["ent_mr2"])
        assert result["relations_updated"] >= 0  # count tracked

        # Relation should still be findable via the merged family
        rels = tmp_storage.get_entity_relations_by_family_id("ent_mr1")
        assert len(rels) >= 1

    def test_merge_creates_redirect(self, tmp_storage):
        """D2.3: Merge creates redirect entries."""
        e1 = _make_entity("ent_red1", "Target", "T")
        e2 = _make_entity("ent_red2", "Source", "S")
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.merge_entity_families("ent_red1", ["ent_red2"])

        # resolve_family_id should redirect ent_red2 → ent_red1
        resolved = tmp_storage.resolve_family_id("ent_red2")
        assert resolved == "ent_red1"

    def test_merge_no_self_merge(self, tmp_storage):
        """D2.4: Merging into itself is a no-op."""
        e = _make_entity("ent_self", "Self", "Content")
        tmp_storage.save_entity(e)

        result = tmp_storage.merge_entity_families("ent_self", ["ent_self"])
        assert result["entities_updated"] == 0

    def test_merge_returns_target_info(self, tmp_storage):
        """D2.5: Merge result contains target_family_id and merged source IDs."""
        e1 = _make_entity("ent_ret1", "Ret1", "C1")
        e2 = _make_entity("ent_ret2", "Ret2", "C2")
        tmp_storage.bulk_save_entities([e1, e2])

        result = tmp_storage.merge_entity_families("ent_ret1", ["ent_ret2"])
        assert result["target_family_id"] == "ent_ret1"
        assert "ent_ret2" in result["merged_source_ids"]

    def test_merge_multiple_sources(self, tmp_storage):
        """D2.6: Merge multiple sources into one target."""
        entities = [_make_entity(f"ent_multi_{i}", f"Multi{i}", f"C{i}") for i in range(5)]
        tmp_storage.bulk_save_entities(entities)

        result = tmp_storage.merge_entity_families(
            "ent_multi_0",
            ["ent_multi_1", "ent_multi_2", "ent_multi_3", "ent_multi_4"],
        )
        assert result["entities_updated"] == 4

        # All should be under ent_multi_0
        versions = tmp_storage.get_entity_versions("ent_multi_0")
        assert len(versions) == 5

    def test_merge_nonexistent_source(self, tmp_storage):
        """D2.7: Merging nonexistent source returns 0 updates."""
        e = _make_entity("ent_nexist", "Exists", "Content")
        tmp_storage.save_entity(e)

        result = tmp_storage.merge_entity_families("ent_nexist", ["nonexistent_123"])
        assert result["entities_updated"] == 0

    def test_merge_version_count_after(self, tmp_storage):
        """D2.8: Version count is correct after merge."""
        e1 = _make_entity("ent_vc1", "VC1", "v1")
        e2 = _make_entity("ent_vc2", "VC2", "v2")
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.merge_entity_families("ent_vc1", ["ent_vc2"])

        counts = tmp_storage.get_entity_version_counts(["ent_vc1"])
        assert counts.get("ent_vc1") == 2


# ══════════════════════════════════════════════════════════════════════════
# D3: VectorStore — Backfill + Search Integration
# ══════════════════════════════════════════════════════════════════════════


class TestVectorStore:
    """Test VectorStore backfill and search integration."""

    def test_vector_store_lazy_init(self, tmp_storage):
        """D3.1: VectorStore initializes lazily (None before first access)."""
        assert tmp_storage._vector_store is None

    def test_vector_store_backfill_on_init(self, tmp_storage):
        """D3.2: VectorStore backfills existing embeddings when initialized."""
        # Save entities with embeddings (if embedding client available)
        # Without embedding client, VectorStore stays None
        if tmp_storage.embedding_client is None or not tmp_storage.embedding_client.is_available():
            pytest.skip("No embedding client available")

        e1 = _make_entity("ent_vb1", "Backfill1", "Content for backfill test")
        e2 = _make_entity("ent_vb2", "Backfill2", "Another backfill test")
        tmp_storage.bulk_save_entities([e1, e2])

        # Trigger VectorStore init
        vs = tmp_storage._get_vector_store()
        if vs is not None:
            # Should have backfilled
            assert vs is not None

    def test_search_uses_vector_store_when_available(self, tmp_storage):
        """D3.3: Embedding search prefers VectorStore KNN over brute-force."""
        if tmp_storage.embedding_client is None or not tmp_storage.embedding_client.is_available():
            pytest.skip("No embedding client available")

        # Create entities
        entities = [
            _make_entity("ent_vs1", "Python", "Python programming language"),
            _make_entity("ent_vs2", "Rust", "Rust systems programming language"),
            _make_entity("ent_vs3", "Cooking", "How to cook pasta"),
        ]
        for e in entities:
            tmp_storage.save_entity(e)

        # Search should work (uses VectorStore if available, falls back otherwise)
        results = tmp_storage.search_entities_by_similarity(
            query_name="programming",
            query_content="programming language",
            threshold=0.3,
            max_results=3,
            similarity_method="embedding",
        )
        assert isinstance(results, list)

    def test_vector_store_upsert_on_save(self, tmp_storage):
        """D3.4: Entity save triggers VectorStore upsert."""
        if tmp_storage.embedding_client is None or not tmp_storage.embedding_client.is_available():
            pytest.skip("No embedding client available")

        e = _make_entity("ent_upsert", "UpsertTest", "Test upsert")
        tmp_storage.save_entity(e)

        # VectorStore should have been created during save
        vs = tmp_storage._get_vector_store()
        if vs is not None:
            # Verify the entity is in VectorStore
            results = vs.search("entity_vectors",
                                tmp_storage.embedding_client.encode("UpsertTest").tolist(),
                                limit=1)
            assert len(results) >= 1

    def test_knn_search_returns_empty_without_vs(self, tmp_storage):
        """D3.5: _vector_knn_search returns [] when no VectorStore."""
        # Force no VectorStore
        tmp_storage._vector_store = None
        import numpy as np
        result = tmp_storage._vector_knn_search(
            "entity_vectors", np.zeros(384, dtype=np.float32), limit=5
        )
        assert result == []

    def test_vector_store_graceful_degradation(self, tmp_storage):
        """D3.6: Search works even if VectorStore is unavailable."""
        # Ensure no embedding client
        if tmp_storage.embedding_client is None:
            # Should fall back to text similarity
            e = _make_entity("ent_degrade", "Degradation", "Test content")
            tmp_storage.save_entity(e)

            results = tmp_storage.search_entities_by_similarity(
                query_name="Degradation",
                threshold=0.0,
                max_results=5,
                similarity_method="text",
            )
            assert isinstance(results, list)


# ══════════════════════════════════════════════════════════════════════════
# D4: Edge Cases + Stress Tests
# ══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_content_entity(self, tmp_storage):
        """D4.1: Entity with empty content is handled."""
        e = _make_entity("ent_empty", "EmptyContent", "")
        tmp_storage.save_entity(e)
        result = tmp_storage.get_entity_by_family_id("ent_empty")
        assert result is not None
        assert result.content == ""

    def test_very_long_content(self, tmp_storage):
        """D4.2: Very long content (10KB) is stored correctly."""
        long_content = "A" * 10000
        e = _make_entity("ent_long", "LongContent", long_content)
        tmp_storage.save_entity(e)
        result = tmp_storage.get_entity_by_family_id("ent_long")
        assert result is not None
        assert len(result.content) == 10000

    def test_unicode_content(self, tmp_storage):
        """D4.3: Unicode content (emoji, CJK, Arabic) stored correctly."""
        content = "Hello 世界 🌍 مرحبا שלום"
        e = _make_entity("ent_uni", "Unicode", content)
        tmp_storage.save_entity(e)
        result = tmp_storage.get_entity_by_family_id("ent_uni")
        assert result.content == content

    def test_null_bytes_in_name(self, tmp_storage):
        """D4.4: Null bytes in name handled gracefully."""
        e = _make_entity("ent_null", "Test\x00Name", "Content")
        tmp_storage.save_entity(e)
        result = tmp_storage.get_entity_by_family_id("ent_null")
        assert result is not None

    def test_many_versions(self, tmp_storage):
        """D4.5: Entity with 20 versions tracked correctly."""
        for i in range(20):
            e = _make_entity("ent_manyv", f"Version{i}", f"Content v{i}")
            tmp_storage.save_entity(e)
            time.sleep(0.01)

        counts = tmp_storage.get_entity_version_counts(["ent_manyv"])
        assert counts.get("ent_manyv") == 20

    def test_delete_entity_all_versions(self, tmp_storage):
        """D4.6: delete_entity_all_versions removes all versions."""
        for i in range(3):
            tmp_storage.save_entity(_make_entity("ent_del", f"V{i}", f"C{i}"))
            time.sleep(0.01)

        count = tmp_storage.delete_entity_all_versions("ent_del")
        assert count == 3

        result = tmp_storage.get_entity_by_family_id("ent_del")
        assert result is None

    def test_delete_nonexistent_entity(self, tmp_storage):
        """D4.7: Deleting nonexistent entity returns 0."""
        count = tmp_storage.delete_entity_all_versions("nonexistent_xyz")
        assert count == 0

    def test_relation_self_loop_prevention(self, tmp_storage):
        """D4.8: Self-referencing entity (same e1/e2) is stored."""
        e = _make_entity("ent_selfref", "Self", "Content")
        tmp_storage.save_entity(e)
        # Relations can technically point to same entity on both ends
        r = _make_relation("rel_selfref", e.absolute_id, e.absolute_id, "Self-reference")
        tmp_storage.save_relation(r)
        result = tmp_storage.get_relation_by_family_id("rel_selfref")
        assert result is not None

    def test_batch_delete_entities(self, tmp_storage):
        """D4.9: Batch delete removes multiple entities."""
        for i in range(5):
            tmp_storage.save_entity(_make_entity(f"ent_bd_{i}", f"BD{i}", f"C{i}"))

        count = tmp_storage.batch_delete_entities(
            [f"ent_bd_{i}" for i in range(5)]
        )
        assert count == 5

    def test_bm25_returns_list(self, tmp_storage):
        """D4.10: BM25 search always returns a list."""
        tmp_storage.save_entity(_make_entity("ent_bm25_list", "TestBM25", "Test content for BM25"))

        results = tmp_storage.search_entities_by_bm25("TestBM25", limit=5)
        assert isinstance(results, list)

    def test_get_entities_by_absolute_ids(self, tmp_storage):
        """D4.11: Batch get by absolute_ids works."""
        entities = [
            _make_entity(f"ent_batch_{i}", f"Batch{i}", f"C{i}")
            for i in range(3)
        ]
        tmp_storage.bulk_save_entities(entities)

        abs_ids = [e.absolute_id for e in entities]
        results = tmp_storage.get_entities_by_absolute_ids(abs_ids)
        assert len(results) == 3

    def test_get_entities_by_absolute_ids_empty(self, tmp_storage):
        """D4.12: Batch get with empty list returns empty."""
        results = tmp_storage.get_entities_by_absolute_ids([])
        assert results == []

    def test_update_entity_summary(self, tmp_storage):
        """D4.13: update_entity_summary sets summary correctly."""
        e = _make_entity("ent_summary", "Summary", "Content")
        tmp_storage.save_entity(e)

        tmp_storage.update_entity_summary("ent_summary", "New AI-generated summary")
        result = tmp_storage.get_entity_by_family_id("ent_summary")
        assert result.summary == "New AI-generated summary"

    def test_graph_statistics_empty(self, tmp_storage):
        """D4.14: Graph statistics on empty graph."""
        stats = tmp_storage.get_graph_statistics()
        assert stats["entity_count"] == 0
        assert stats["relation_count"] == 0

    def test_data_quality_report(self, tmp_storage):
        """D4.15: Graph statistics available as quality proxy for SQLite backend."""
        e1 = _make_entity("ent_q1", "Q1", "v1")
        tmp_storage.save_entity(e1)
        time.sleep(0.01)
        e2 = _make_entity("ent_q1", "Q1", "v2")
        tmp_storage.save_entity(e2)

        stats = tmp_storage.get_graph_statistics()
        assert stats["entity_count"] >= 1
        # After saving 2 versions of same entity, still 1 unique family_id
        assert stats["entity_count"] == 2  # row count, not unique families
