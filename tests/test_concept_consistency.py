"""
Concept table consistency — 36 tests across 4 dimensions.

Ensures merge and delete operations keep the concepts table in sync with
the primary tables (entities, relations, episodes).

D1: merge_entity_families updates concepts table (9 tests)
D2: delete_episode cleans concepts + mentions (9 tests)
D3: delete_entity/relation cleans concepts (9 tests)
D4: Cross-cutting — merge + search + traversal consistency (9 tests)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_storage(tmp_path):
    from processor.storage.manager import StorageManager
    sm = StorageManager(str(tmp_path / "graph"))
    yield sm
    if hasattr(sm, '_vector_store') and sm._vector_store:
        sm._vector_store.close()


def _make_entity(family_id: str, name: str, content: str, source_document: str = ""):
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


def _make_relation(family_id: str, e1_abs: str, e2_abs: str, content: str):
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
        source_document="",
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


# ══════════════════════════════════════════════════════════════════════════
# D1: merge_entity_families updates concepts table
# ══════════════════════════════════════════════════════════════════════════


class TestMergeConceptsConsistency:
    """D1: merge_entity_families propagates family_id changes to concepts table."""

    def test_merge_updates_concepts_family_id(self, tmp_storage):
        """D1.1: After merge, source concept rows have target family_id."""
        e1 = _make_entity("merge_tgt", "Target", "Target entity")
        e2 = _make_entity("merge_src", "Source", "Source entity")
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.merge_entity_families("merge_tgt", ["merge_src"])

        # Source's concepts should now have target family_id
        conn = tmp_storage._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT family_id FROM concepts WHERE role='entity' AND id = ?",
            (e2.absolute_id,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "merge_tgt"

    def test_merge_no_orphaned_concepts(self, tmp_storage):
        """D1.2: After merge, no concept rows with source family_id remain."""
        e1 = _make_entity("orphan_tgt", "T", "T")
        e2 = _make_entity("orphan_src", "S", "S")
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.merge_entity_families("orphan_tgt", ["orphan_src"])

        # No concepts should have the old source family_id
        conn = tmp_storage._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM concepts WHERE family_id = 'orphan_src'"
        )
        assert cursor.fetchone()[0] == 0

    def test_merge_target_concepts_unchanged(self, tmp_storage):
        """D1.3: Target entity concepts still exist after merge."""
        e1 = _make_entity("tgt_unch", "T", "Target content")
        e2 = _make_entity("src_unch", "S", "Source content")
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.merge_entity_families("tgt_unch", ["src_unch"])

        concept = tmp_storage.get_concept_by_family_id("tgt_unch")
        assert concept is not None
        # After merge, both target and source concepts share the family_id
        # get_concept_by_family_id returns latest version, which may be the source

    def test_merge_multiple_sources(self, tmp_storage):
        """D1.4: Merging multiple sources updates all concepts."""
        e1 = _make_entity("multi_tgt", "T", "T")
        e2 = _make_entity("multi_src1", "S1", "S1")
        e3 = _make_entity("multi_src2", "S2", "S2")
        tmp_storage.bulk_save_entities([e1, e2, e3])

        tmp_storage.merge_entity_families("multi_tgt", ["multi_src1", "multi_src2"])

        conn = tmp_storage._get_conn()
        cursor = conn.cursor()
        for src_fid in ["multi_src1", "multi_src2"]:
            cursor.execute("SELECT COUNT(*) FROM concepts WHERE family_id = ?", (src_fid,))
            assert cursor.fetchone()[0] == 0

    def test_merge_concept_search_uses_new_family(self, tmp_storage):
        """D1.5: After merge, concept search finds entity under target family_id."""
        e1 = _make_entity("search_tgt", "Quantum", "Quantum physics")
        e2 = _make_entity("search_src", "量子", "量子力学")
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.merge_entity_families("search_tgt", ["search_src"])

        # get_concept_by_family_id with source should not find (redirected)
        concept_src = tmp_storage.get_concept_by_family_id("search_src")
        # It should find the target (or nothing if no redirect resolution)
        concept_tgt = tmp_storage.get_concept_by_family_id("search_tgt")
        assert concept_tgt is not None

    def test_merge_entities_table_consistent(self, tmp_storage):
        """D1.6: entities table and concepts table stay consistent after merge."""
        e1 = _make_entity("cons_tgt", "T", "T")
        e2 = _make_entity("cons_src", "S", "S")
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.merge_entity_families("cons_tgt", ["cons_src"])

        conn = tmp_storage._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT family_id FROM entities WHERE id = ?", (e2.absolute_id,))
        ent_row = cursor.fetchone()
        cursor.execute("SELECT family_id FROM concepts WHERE id = ?", (e2.absolute_id,))
        cpt_row = cursor.fetchone()
        assert ent_row[0] == cpt_row[0] == "cons_tgt"

    def test_merge_result_includes_concepts_count(self, tmp_storage):
        """D1.7: merge result reflects the operation."""
        e1 = _make_entity("res_tgt", "T", "T")
        e2 = _make_entity("res_src", "S", "S")
        tmp_storage.bulk_save_entities([e1, e2])

        result = tmp_storage.merge_entity_families("res_tgt", ["res_src"])
        assert result["entities_updated"] >= 1

    def test_merge_idempotent(self, tmp_storage):
        """D1.8: Merging same source twice doesn't break."""
        e1 = _make_entity("idem_tgt", "T", "T")
        e2 = _make_entity("idem_src", "S", "S")
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.merge_entity_families("idem_tgt", ["idem_src"])
        result = tmp_storage.merge_entity_families("idem_tgt", ["idem_src"])
        # Second merge should not error, may report 0 updates
        assert isinstance(result, dict)

    def test_merge_empty_source_list(self, tmp_storage):
        """D1.9: Empty source list returns zero updates."""
        result = tmp_storage.merge_entity_families("any_target", [])
        assert result["entities_updated"] == 0


# ══════════════════════════════════════════════════════════════════════════
# D2: delete_episode cleans concepts + mentions
# ══════════════════════════════════════════════════════════════════════════


class TestDeleteEpisodeConceptCleanup:
    """D2: delete_episode removes concept and mention rows from DB."""

    def test_delete_removes_concept(self, tmp_storage):
        """D2.1: Deleting episode removes its concept row."""
        ep = _make_episode("del_ep_1", "Content to delete", source_document="del_doc.txt")
        tmp_storage.save_episode(ep)

        # Verify concept exists
        concept_before = tmp_storage.get_concept_by_family_id("del_ep_1")
        assert concept_before is not None

        tmp_storage.delete_episode("del_ep_1")

        # Concept should be gone
        concept_after = tmp_storage.get_concept_by_family_id("del_ep_1")
        assert concept_after is None

    def test_delete_removes_mentions(self, tmp_storage):
        """D2.2: Deleting episode removes its episode_mentions."""
        e1 = _make_entity("dm_e1", "A", "A")
        tmp_storage.bulk_save_entities([e1])

        ep = _make_episode("dm_ep1", "Content about A", source_document="dm_doc.txt")
        tmp_storage.save_episode(ep)
        tmp_storage.save_episode_mentions("dm_ep1", [e1.absolute_id], target_type="entity")

        tmp_storage.delete_episode("dm_ep1")

        # Mentions should be gone
        mentions = tmp_storage.get_episode_entities("dm_ep1")
        assert mentions == []

    def test_delete_removes_episode_row(self, tmp_storage):
        """D2.3: Deleting episode removes its episodes table row."""
        ep = _make_episode("dr_ep1", "Row to delete", source_document="dr_doc.txt")
        tmp_storage.save_episode(ep)

        assert tmp_storage.get_episode_from_db("dr_ep1") is not None
        tmp_storage.delete_episode("dr_ep1")
        assert tmp_storage.get_episode_from_db("dr_ep1") is None

    def test_delete_nonexistent_returns_zero(self, tmp_storage):
        """D2.4: Deleting nonexistent episode returns 0."""
        result = tmp_storage.delete_episode("nonexistent_ep_xyz")
        assert result == 0

    def test_delete_preserves_other_episodes(self, tmp_storage):
        """D2.5: Deleting one episode doesn't affect others."""
        ep1 = _make_episode("pr_ep1", "Keep this", source_document="pr_doc1.txt")
        ep2 = _make_episode("pr_ep2", "Delete this", source_document="pr_doc2.txt")
        tmp_storage.save_episode(ep1)
        tmp_storage.save_episode(ep2)

        tmp_storage.delete_episode("pr_ep2")

        assert tmp_storage.get_episode_from_db("pr_ep1") is not None
        assert tmp_storage.get_episode_from_db("pr_ep2") is None

    def test_delete_versioned_episode_removes_one(self, tmp_storage):
        """D2.6: Deleting one version of a versioned episode removes only that row."""
        ep1 = _make_episode("vdel_ep1", "V1", source_document="vdel_doc.txt")
        tmp_storage.save_episode(ep1)
        ep2 = _make_episode("vdel_ep2", "V2", source_document="vdel_doc.txt")
        tmp_storage.save_episode(ep2)

        tmp_storage.delete_episode("vdel_ep2")

        # vdel_ep1 should still exist, vdel_ep2 should be gone
        assert tmp_storage.get_episode_from_db("vdel_ep1") is not None
        assert tmp_storage.get_episode_from_db("vdel_ep2") is None

    def test_delete_reduces_episode_count(self, tmp_storage):
        """D2.7: Deleting episode reduces count."""
        ep = _make_episode("cnt_ep1", "To delete", source_document="cnt_doc.txt")
        tmp_storage.save_episode(ep)
        count_before = tmp_storage.count_episodes()
        tmp_storage.delete_episode("cnt_ep1")
        count_after = tmp_storage.count_episodes()
        assert count_after == count_before - 1

    def test_delete_concept_count_reduced(self, tmp_storage):
        """D2.8: Deleting episode reduces observation concept count."""
        ep = _make_episode("ccnt_ep1", "Observation", source_document="ccnt_doc.txt")
        tmp_storage.save_episode(ep)
        count_before = tmp_storage.count_concepts(role="observation")
        tmp_storage.delete_episode("ccnt_ep1")
        count_after = tmp_storage.count_concepts(role="observation")
        assert count_after == count_before - 1

    def test_delete_episode_with_unicode_content(self, tmp_storage):
        """D2.9: Deleting episode with unicode content works."""
        ep = _make_episode("uni_ep1", "内容包含中文和emoji 🚀", source_document="uni_doc.txt")
        tmp_storage.save_episode(ep)
        assert tmp_storage.delete_episode("uni_ep1") == 1


# ══════════════════════════════════════════════════════════════════════════
# D3: delete_entity/relation cleans concepts
# ══════════════════════════════════════════════════════════════════════════


class TestDeleteEntityRelationConcepts:
    """D3: delete_entity and delete_relation clean concept rows."""

    def test_delete_entity_removes_concept(self, tmp_storage):
        """D3.1: delete_entity_by_absolute_id removes concept row."""
        e = _make_entity("dec_e1", "Entity", "Content")
        tmp_storage.save_entity(e)

        # Verify concept exists
        conn = tmp_storage._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM concepts WHERE id = ?", (e.absolute_id,))
        assert cursor.fetchone()[0] == 1

        tmp_storage.delete_entity_by_absolute_id(e.absolute_id)

        cursor.execute("SELECT COUNT(*) FROM concepts WHERE id = ?", (e.absolute_id,))
        assert cursor.fetchone()[0] == 0

    def test_delete_entity_family_removes_all_concepts(self, tmp_storage):
        """D3.2: delete_entity_by_id (all versions) removes all concept rows."""
        e1 = _make_entity("def_e1", "E", "V1")
        e2 = _make_entity("def_e1", "E", "V2")
        tmp_storage.save_entity(e1)
        tmp_storage.save_entity(e2)

        tmp_storage.delete_entity_by_id("def_e1")

        conn = tmp_storage._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM concepts WHERE family_id = 'def_e1'")
        assert cursor.fetchone()[0] == 0

    def test_delete_relation_removes_concept(self, tmp_storage):
        """D3.3: delete_relation_by_absolute_id removes concept row."""
        e1 = _make_entity("drc_e1", "A", "A")
        e2 = _make_entity("drc_e2", "B", "B")
        r = _make_relation("drc_r1", e1.absolute_id, e2.absolute_id, "A-B")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)

        tmp_storage.delete_relation_by_absolute_id(r.absolute_id)

        conn = tmp_storage._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM concepts WHERE id = ?", (r.absolute_id,))
        assert cursor.fetchone()[0] == 0

    def test_delete_relation_family_removes_all_concepts(self, tmp_storage):
        """D3.4: delete_relation (all versions) removes all concept rows."""
        e1 = _make_entity("drf_e1", "X", "X")
        e2 = _make_entity("drf_e2", "Y", "Y")
        r1 = _make_relation("drf_r1", e1.absolute_id, e2.absolute_id, "X-Y v1")
        r2 = _make_relation("drf_r1", e1.absolute_id, e2.absolute_id, "X-Y v2")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r1)
        tmp_storage.save_relation(r2)

        tmp_storage.delete_relation_by_id("drf_r1")

        conn = tmp_storage._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM concepts WHERE family_id = 'drf_r1'")
        assert cursor.fetchone()[0] == 0

    def test_delete_entity_preserves_others(self, tmp_storage):
        """D3.5: Deleting one entity doesn't affect others' concepts."""
        e1 = _make_entity("p_e1", "Keep", "Keep")
        e2 = _make_entity("p_e2", "Remove", "Remove")
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.delete_entity_by_absolute_id(e2.absolute_id)

        concept1 = tmp_storage.get_concept_by_family_id("p_e1")
        assert concept1 is not None

    def test_delete_relation_preserves_entities(self, tmp_storage):
        """D3.6: Deleting relation doesn't affect connected entity concepts."""
        e1 = _make_entity("rpe_e1", "A", "A")
        e2 = _make_entity("rpe_e2", "B", "B")
        r = _make_relation("rpe_r1", e1.absolute_id, e2.absolute_id, "A-B")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)

        tmp_storage.delete_relation_by_absolute_id(r.absolute_id)

        assert tmp_storage.get_concept_by_family_id("rpe_e1") is not None
        assert tmp_storage.get_concept_by_family_id("rpe_e2") is not None

    def test_delete_entity_concept_count_reduced(self, tmp_storage):
        """D3.7: Deleting entity reduces entity concept count."""
        e = _make_entity("cntd_e1", "ToDel", "Content")
        tmp_storage.save_entity(e)
        count_before = tmp_storage.count_concepts(role="entity")
        tmp_storage.delete_entity_by_id("cntd_e1")
        count_after = tmp_storage.count_concepts(role="entity")
        assert count_after == count_before - 1

    def test_delete_relation_concept_count_reduced(self, tmp_storage):
        """D3.8: Deleting relation reduces relation concept count."""
        e1 = _make_entity("cntr_e1", "A", "A")
        e2 = _make_entity("cntr_e2", "B", "B")
        r = _make_relation("cntr_r1", e1.absolute_id, e2.absolute_id, "A-B")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)
        count_before = tmp_storage.count_concepts(role="relation")
        tmp_storage.delete_relation_by_id("cntr_r1")
        count_after = tmp_storage.count_concepts(role="relation")
        assert count_after == count_before - 1

    def test_delete_nonexistent_entity_no_error(self, tmp_storage):
        """D3.9: Deleting nonexistent entity doesn't error."""
        tmp_storage.delete_entity_by_id("nonexistent_entity_xyz")
        # Should not raise


# ══════════════════════════════════════════════════════════════════════════
# D4: Cross-cutting — merge + search + traversal consistency
# ══════════════════════════════════════════════════════════════════════════


class TestCrossCuttingConsistency:
    """D4: Cross-cutting consistency tests for concept table."""

    def test_merge_then_search_finds_target(self, tmp_storage):
        """D4.1: After merge, BM25 search finds content under target family_id."""
        e1 = _make_entity("ms_tgt", "Python", "Python programming")
        e2 = _make_entity("ms_src", "Java", "Java programming")
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.merge_entity_families("ms_tgt", ["ms_src"])

        # BM25 search for "Java" should still find it (under merged family_id)
        results = tmp_storage.search_concepts_by_bm25("Java", role="entity", limit=10)
        assert len(results) >= 1
        assert any(r["family_id"] == "ms_tgt" for r in results)

    def test_merge_then_traverse(self, tmp_storage):
        """D4.2: After merge, concept traversal works correctly."""
        e1 = _make_entity("trv_tgt", "Alpha", "Alpha entity")
        e2 = _make_entity("trv_src", "Beta", "Beta entity")
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.merge_entity_families("trv_tgt", ["trv_src"])

        # Traversal from target should work
        result = tmp_storage.traverse_concepts(["trv_tgt"], max_depth=1)
        assert "trv_tgt" in result["concepts"]

    def test_delete_then_list(self, tmp_storage):
        """D4.3: After delete, list_concepts doesn't show deleted."""
        e = _make_entity("dlist_e1", "ToDelete", "Content")
        tmp_storage.save_entity(e)

        tmp_storage.delete_entity_by_id("dlist_e1")

        concepts = tmp_storage.list_concepts(role="entity", limit=100)
        assert not any(c["family_id"] == "dlist_e1" for c in concepts)

    def test_merge_preserves_observation_concepts(self, tmp_storage):
        """D4.4: Merging entities doesn't affect observation concepts."""
        ep = _make_episode("mp_ep1", "An observation", source_document="mp_doc.txt")
        tmp_storage.save_episode(ep)
        e1 = _make_entity("mp_e1", "A", "A")
        e2 = _make_entity("mp_e2", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.merge_entity_families("mp_e1", ["mp_e2"])

        # Observation should be untouched
        concept = tmp_storage.get_concept_by_family_id("mp_ep1")
        assert concept is not None
        assert concept["role"] == "observation"

    def test_delete_all_of_role(self, tmp_storage):
        """D4.5: Deleting all entities leaves relations and observations."""
        e1 = _make_entity("dar_e1", "A", "A")
        e2 = _make_entity("dar_e2", "B", "B")
        r = _make_relation("dar_r1", e1.absolute_id, e2.absolute_id, "A-B")
        ep = _make_episode("dar_ep1", "Obs", source_document="dar_doc.txt")
        tmp_storage.bulk_save_entities([e1, e2])
        tmp_storage.save_relation(r)
        tmp_storage.save_episode(ep)

        tmp_storage.delete_entity_by_id("dar_e1")
        tmp_storage.delete_entity_by_id("dar_e2")

        # Relations and observations still exist as concepts
        assert tmp_storage.count_concepts(role="relation") >= 1
        assert tmp_storage.count_concepts(role="observation") >= 1

    def test_merge_versioned_entities(self, tmp_storage):
        """D4.6: Merging entity with multiple versions updates all concept rows."""
        e1 = _make_entity("mv_tgt", "T", "T")
        e2 = _make_entity("mv_src", "S", "S v1")
        e3 = _make_entity("mv_src", "S", "S v2")
        tmp_storage.save_entity(e1)
        tmp_storage.save_entity(e2)
        tmp_storage.save_entity(e3)

        tmp_storage.merge_entity_families("mv_tgt", ["mv_src"])

        conn = tmp_storage._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM concepts WHERE family_id = 'mv_src'")
        assert cursor.fetchone()[0] == 0
        cursor.execute("SELECT COUNT(*) FROM concepts WHERE family_id = 'mv_tgt'")
        assert cursor.fetchone()[0] >= 3  # 1 target + 2 merged versions

    def test_unicode_merge_concepts(self, tmp_storage):
        """D4.7: Merge with unicode content preserves concept data."""
        e1 = _make_entity("uni_tgt", "深度学习", "深度学习是AI的核心 🚀")
        e2 = _make_entity("uni_src", "机器学习", "机器学习是AI的基础方法")
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.merge_entity_families("uni_tgt", ["uni_src"])

        concept = tmp_storage.get_concept_by_family_id("uni_tgt")
        assert concept is not None
        # Either target or source concept should be found under the merged family_id

    def test_rapid_create_delete_cycle(self, tmp_storage):
        """D4.8: Rapid create-delete cycle doesn't leak concept rows."""
        for i in range(5):
            e = _make_entity(f"cycle_e{i}", f"E{i}", f"Content {i}")
            tmp_storage.save_entity(e)
            tmp_storage.delete_entity_by_id(f"cycle_e{i}")

        # All entity concepts should be gone
        assert tmp_storage.count_concepts(role="entity") == 0

    def test_merge_then_concept_stats(self, tmp_storage):
        """D4.9: Concept counts are consistent after merge."""
        e1 = _make_entity("stat_tgt", "T", "T")
        e2 = _make_entity("stat_src", "S", "S")
        tmp_storage.bulk_save_entities([e1, e2])

        concept_count_before = tmp_storage.count_concepts(role="entity")
        tmp_storage.merge_entity_families("stat_tgt", ["stat_src"])
        concept_count_after = tmp_storage.count_concepts(role="entity")

        # Concept count unchanged (just moved family_id, not added/removed)
        assert concept_count_after == concept_count_before
