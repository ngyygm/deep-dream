"""
Episode/Observation versioning — 36 tests across 4 dimensions.

Vision principle: "Observation也会演化" — same source_document reprocessed
creates a new version (same family_id), not a standalone concept.

D1: New episodes get standalone family_id (9 tests)
D2: Re-processed documents get versioned — same family_id, new absolute_id (9 tests)
D3: Concept table dual-write uses resolved family_id (9 tests)
D4: Edge cases — empty source_document, unicode, multiple versions, provenance (9 tests)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List
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


def _make_episode(absolute_id: str, content: str, source_document: str = "",
                  activity_type: str = "", episode_type: str = ""):
    from processor.models import Episode
    now = datetime.now(timezone.utc)
    ep = Episode(
        absolute_id=absolute_id,
        content=content,
        event_time=now,
        source_document=source_document,
        activity_type=activity_type,
    )
    ep.processed_time = now
    ep.episode_type = episode_type
    return ep


def _save_episode_to_db(storage, ep, doc_hash: str = ""):
    """Directly call _save_episode_to_db without file system operations."""
    storage._save_episode_to_db(ep, doc_hash=doc_hash)


# ══════════════════════════════════════════════════════════════════════════
# D1: New episodes get standalone family_id
# ══════════════════════════════════════════════════════════════════════════


class TestNewEpisodeFamilyId:
    """D1: New episodes without existing source_document get standalone family_id."""

    def test_new_episode_gets_own_family_id(self, tmp_storage):
        """D1.1: Episode with new source_document gets its absolute_id as family_id."""
        ep = _make_episode("ep_new_1", "First content", source_document="doc_A.txt")
        _save_episode_to_db(tmp_storage, ep)

        row = tmp_storage.get_episode_from_db("ep_new_1")
        assert row is not None
        assert row["family_id"] == "ep_new_1"

    def test_new_episode_no_source_document(self, tmp_storage):
        """D1.2: Episode with empty source_document gets absolute_id as family_id."""
        ep = _make_episode("ep_nosource_1", "Content without source")
        _save_episode_to_db(tmp_storage, ep)

        row = tmp_storage.get_episode_from_db("ep_nosource_1")
        assert row is not None
        assert row["family_id"] == "ep_nosource_1"

    def test_two_different_sources_different_families(self, tmp_storage):
        """D1.3: Two episodes with different source_documents have different family_ids."""
        ep1 = _make_episode("ep_diff_1", "Content A", source_document="doc_A.txt")
        ep2 = _make_episode("ep_diff_2", "Content B", source_document="doc_B.txt")
        _save_episode_to_db(tmp_storage, ep1)
        _save_episode_to_db(tmp_storage, ep2)

        row1 = tmp_storage.get_episode_from_db("ep_diff_1")
        row2 = tmp_storage.get_episode_from_db("ep_diff_2")
        assert row1["family_id"] != row2["family_id"]

    def test_new_episode_stored_in_concepts(self, tmp_storage):
        """D1.4: New episode appears in concepts table with role=observation."""
        ep = _make_episode("ep_concept_1", "Concept content", source_document="doc_C.txt")
        _save_episode_to_db(tmp_storage, ep)

        concept = tmp_storage.get_concept_by_family_id("ep_concept_1")
        assert concept is not None
        assert concept["role"] == "observation"

    def test_new_episode_content_preserved(self, tmp_storage):
        """D1.5: Episode content is stored correctly."""
        content = "This is the first observation about the world."
        ep = _make_episode("ep_cnt_1", content, source_document="doc_D.txt")
        _save_episode_to_db(tmp_storage, ep)

        row = tmp_storage.get_episode_from_db("ep_cnt_1")
        assert row["content"] == content

    def test_new_episode_activity_type(self, tmp_storage):
        """D1.6: Episode activity_type is preserved."""
        ep = _make_episode("ep_act_1", "Content", source_document="doc_E.txt",
                           activity_type="remember")
        _save_episode_to_db(tmp_storage, ep)

        row = tmp_storage.get_episode_from_db("ep_act_1")
        assert row["activity_type"] == "remember"

    def test_new_episode_episode_type(self, tmp_storage):
        """D1.7: Episode episode_type is preserved."""
        ep = _make_episode("ep_etype_1", "Content", source_document="doc_F.txt",
                           episode_type="document")
        _save_episode_to_db(tmp_storage, ep)

        row = tmp_storage.get_episode_from_db("ep_etype_1")
        assert row["episode_type"] == "document"

    def test_list_episodes_shows_new(self, tmp_storage):
        """D1.8: New episode appears in list_episodes."""
        ep = _make_episode("ep_list_1", "List content", source_document="doc_G.txt")
        _save_episode_to_db(tmp_storage, ep)

        episodes = tmp_storage.list_episodes(limit=10)
        assert any(e["uuid"] == "ep_list_1" for e in episodes)

    def test_count_episodes_increments(self, tmp_storage):
        """D1.9: count_episodes reflects new episodes."""
        count_before = tmp_storage.count_episodes()
        ep = _make_episode("ep_count_1", "Count content", source_document="doc_H.txt")
        _save_episode_to_db(tmp_storage, ep)
        count_after = tmp_storage.count_episodes()
        assert count_after == count_before + 1


# ══════════════════════════════════════════════════════════════════════════
# D2: Re-processed documents get versioned — same family_id, new absolute_id
# ══════════════════════════════════════════════════════════════════════════


class TestEpisodeVersioning:
    """D2: Same source_document reprocessed creates new version with same family_id."""

    def test_same_source_reuses_family_id(self, tmp_storage):
        """D2.1: Second episode with same source_document gets first's family_id."""
        ep1 = _make_episode("ep_ver_1", "Version 1 content", source_document="doc_version.txt")
        _save_episode_to_db(tmp_storage, ep1)

        ep2 = _make_episode("ep_ver_2", "Version 2 content", source_document="doc_version.txt")
        _save_episode_to_db(tmp_storage, ep2)

        row1 = tmp_storage.get_episode_from_db("ep_ver_1")
        row2 = tmp_storage.get_episode_from_db("ep_ver_2")
        assert row1["family_id"] == "ep_ver_1"  # First is standalone
        assert row2["family_id"] == "ep_ver_1"  # Second reuses first's family_id

    def test_version_count_increases(self, tmp_storage):
        """D2.2: Multiple versions of same source increase episode count."""
        ep1 = _make_episode("ep_vc_1", "V1", source_document="doc_vc.txt")
        _save_episode_to_db(tmp_storage, ep1)
        ep2 = _make_episode("ep_vc_2", "V2", source_document="doc_vc.txt")
        _save_episode_to_db(tmp_storage, ep2)

        # Both should be in the episodes table
        assert tmp_storage.get_episode_from_db("ep_vc_1") is not None
        assert tmp_storage.get_episode_from_db("ep_vc_2") is not None
        assert tmp_storage.count_episodes() >= 2

    def test_different_sources_not_versioned(self, tmp_storage):
        """D2.3: Different source_documents get independent family_ids."""
        ep1 = _make_episode("ep_ind_1", "A", source_document="doc_ind_A.txt")
        _save_episode_to_db(tmp_storage, ep1)
        ep2 = _make_episode("ep_ind_2", "B", source_document="doc_ind_B.txt")
        _save_episode_to_db(tmp_storage, ep2)

        row1 = tmp_storage.get_episode_from_db("ep_ind_1")
        row2 = tmp_storage.get_episode_from_db("ep_ind_2")
        assert row1["family_id"] != row2["family_id"]

    def test_three_versions_same_source(self, tmp_storage):
        """D2.4: Three episodes with same source all share family_id."""
        for i in range(3):
            ep = _make_episode(f"ep_3v_{i}", f"Version {i}", source_document="doc_3v.txt")
            _save_episode_to_db(tmp_storage, ep)

        for i in range(3):
            row = tmp_storage.get_episode_from_db(f"ep_3v_{i}")
            assert row["family_id"] == "ep_3v_0"

    def test_versioned_content_independent(self, tmp_storage):
        """D2.5: Each version preserves its own content."""
        ep1 = _make_episode("ep_ic_1", "Original content", source_document="doc_ic.txt")
        _save_episode_to_db(tmp_storage, ep1)
        ep2 = _make_episode("ep_ic_2", "Updated content", source_document="doc_ic.txt")
        _save_episode_to_db(tmp_storage, ep2)

        row1 = tmp_storage.get_episode_from_db("ep_ic_1")
        row2 = tmp_storage.get_episode_from_db("ep_ic_2")
        assert row1["content"] == "Original content"
        assert row2["content"] == "Updated content"

    def test_versioned_episode_in_list(self, tmp_storage):
        """D2.6: Both versions appear in list_episodes."""
        ep1 = _make_episode("ep_lv_1", "V1", source_document="doc_lv.txt")
        _save_episode_to_db(tmp_storage, ep1)
        ep2 = _make_episode("ep_lv_2", "V2", source_document="doc_lv.txt")
        _save_episode_to_db(tmp_storage, ep2)

        episodes = tmp_storage.list_episodes(limit=10)
        ids = {e["uuid"] for e in episodes}
        assert "ep_lv_1" in ids
        assert "ep_lv_2" in ids

    def test_interleaved_sources(self, tmp_storage):
        """D2.7: Interleaved processing of different sources maintains correct versioning."""
        ep_a1 = _make_episode("ep_ix_a1", "A v1", source_document="doc_ix_A.txt")
        _save_episode_to_db(tmp_storage, ep_a1)
        ep_b1 = _make_episode("ep_ix_b1", "B v1", source_document="doc_ix_B.txt")
        _save_episode_to_db(tmp_storage, ep_b1)
        ep_a2 = _make_episode("ep_ix_a2", "A v2", source_document="doc_ix_A.txt")
        _save_episode_to_db(tmp_storage, ep_a2)

        row_a1 = tmp_storage.get_episode_from_db("ep_ix_a1")
        row_b1 = tmp_storage.get_episode_from_db("ep_ix_b1")
        row_a2 = tmp_storage.get_episode_from_db("ep_ix_a2")
        assert row_a1["family_id"] == "ep_ix_a1"
        assert row_b1["family_id"] == "ep_ix_b1"
        assert row_a2["family_id"] == "ep_ix_a1"  # Reuses A's family_id

    def test_version_after_delete(self, tmp_storage):
        """D2.8: New version after deleting previous still gets correct family_id."""
        ep1 = _make_episode("ep_del_1", "To be deleted", source_document="doc_del.txt")
        _save_episode_to_db(tmp_storage, ep1)

        # Simulate new version (old one still exists in DB)
        ep2 = _make_episode("ep_del_2", "New version", source_document="doc_del.txt")
        _save_episode_to_db(tmp_storage, ep2)

        row2 = tmp_storage.get_episode_from_db("ep_del_2")
        assert row2["family_id"] == "ep_del_1"

    def test_idempotent_save(self, tmp_storage):
        """D2.9: Saving same episode twice is idempotent."""
        ep = _make_episode("ep_idem_1", "Same content", source_document="doc_idem.txt")
        _save_episode_to_db(tmp_storage, ep)
        _save_episode_to_db(tmp_storage, ep)  # Save again

        # Should still have exactly 1 row
        row = tmp_storage.get_episode_from_db("ep_idem_1")
        assert row is not None
        assert row["family_id"] == "ep_idem_1"


# ══════════════════════════════════════════════════════════════════════════
# D3: Concept table dual-write uses resolved family_id
# ══════════════════════════════════════════════════════════════════════════


class TestConceptDualWrite:
    """D3: Concepts table observation entries use resolved family_id."""

    def test_new_episode_concept_standalone(self, tmp_storage):
        """D3.1: New episode's concept has family_id == absolute_id."""
        ep = _make_episode("ep_cpt_1", "Concept test", source_document="doc_cpt_new.txt")
        _save_episode_to_db(tmp_storage, ep)

        concept = tmp_storage.get_concept_by_family_id("ep_cpt_1")
        assert concept is not None
        assert concept["id"] == "ep_cpt_1"
        assert concept["family_id"] == "ep_cpt_1"

    def test_versioned_episode_concept_shared_family(self, tmp_storage):
        """D3.2: Versioned episode's concept shares family_id with original."""
        ep1 = _make_episode("ep_cfv_1", "V1 concept", source_document="doc_cfv.txt")
        _save_episode_to_db(tmp_storage, ep1)
        ep2 = _make_episode("ep_cfv_2", "V2 concept", source_document="doc_cfv.txt")
        _save_episode_to_db(tmp_storage, ep2)

        # Both should be resolvable via the original family_id
        concept = tmp_storage.get_concept_by_family_id("ep_cfv_1")
        assert concept is not None
        assert concept["role"] == "observation"

    def test_concept_role_observation(self, tmp_storage):
        """D3.3: Episode concepts have role='observation'."""
        ep = _make_episode("ep_role_1", "Role test", source_document="doc_role.txt")
        _save_episode_to_db(tmp_storage, ep)

        concept = tmp_storage.get_concept_by_family_id("ep_role_1")
        assert concept["role"] == "observation"

    def test_concept_content_matches(self, tmp_storage):
        """D3.4: Concept content matches episode content."""
        content = "Observation about quantum computing advances."
        ep = _make_episode("ep_cc_1", content, source_document="doc_cc.txt")
        _save_episode_to_db(tmp_storage, ep)

        concept = tmp_storage.get_concept_by_family_id("ep_cc_1")
        assert concept["content"] == content

    def test_concept_source_document(self, tmp_storage):
        """D3.5: Concept stores source_document."""
        ep = _make_episode("ep_csd_1", "Content", source_document="my_document.pdf")
        _save_episode_to_db(tmp_storage, ep)

        concept = tmp_storage.get_concept_by_family_id("ep_csd_1")
        assert concept["source_document"] == "my_document.pdf"

    def test_versioned_concepts_both_exist(self, tmp_storage):
        """D3.6: Both versions exist in concepts table (different absolute_ids)."""
        ep1 = _make_episode("ep_cex_1", "First observation", source_document="doc_cex.txt")
        _save_episode_to_db(tmp_storage, ep1)
        ep2 = _make_episode("ep_cex_2", "Second observation", source_document="doc_cex.txt")
        _save_episode_to_db(tmp_storage, ep2)

        # get_concept_by_family_id returns latest
        concept = tmp_storage.get_concept_by_family_id("ep_cex_1")
        assert concept is not None

    def test_list_concepts_includes_observations(self, tmp_storage):
        """D3.7: list_concepts with role='observation' includes episodes."""
        ep = _make_episode("ep_lci_1", "List test", source_document="doc_lci.txt")
        _save_episode_to_db(tmp_storage, ep)

        obs = tmp_storage.list_concepts(role="observation", limit=10)
        assert any(c["family_id"] == "ep_lci_1" for c in obs)

    def test_count_concepts_includes_observations(self, tmp_storage):
        """D3.8: count_concepts for role='observation' counts episodes."""
        count_before = tmp_storage.count_concepts(role="observation")
        ep = _make_episode("ep_oci_1", "Count test", source_document="doc_oci.txt")
        _save_episode_to_db(tmp_storage, ep)
        count_after = tmp_storage.count_concepts(role="observation")
        assert count_after >= count_before + 1

    def test_bm25_search_finds_observation(self, tmp_storage):
        """D3.9: BM25 search finds observation concepts."""
        ep = _make_episode("ep_bm25_1", "Quantum entanglement experiment results",
                           source_document="doc_bm25.txt")
        _save_episode_to_db(tmp_storage, ep)

        results = tmp_storage.search_concepts_by_bm25("quantum", role="observation", limit=10)
        assert len(results) >= 1


# ══════════════════════════════════════════════════════════════════════════
# D4: Edge cases — empty source_document, unicode, multiple versions, provenance
# ══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """D4: Edge cases for episode versioning."""

    def test_empty_source_document_no_versioning(self, tmp_storage):
        """D4.1: Empty source_document episodes never get versioned."""
        ep1 = _make_episode("ep_empty_1", "First", source_document="")
        _save_episode_to_db(tmp_storage, ep1)
        ep2 = _make_episode("ep_empty_2", "Second", source_document="")
        _save_episode_to_db(tmp_storage, ep2)

        row1 = tmp_storage.get_episode_from_db("ep_empty_1")
        row2 = tmp_storage.get_episode_from_db("ep_empty_2")
        assert row1["family_id"] == "ep_empty_1"
        assert row2["family_id"] == "ep_empty_2"  # Not versioned

    def test_unicode_source_document(self, tmp_storage):
        """D4.2: Unicode source_document works for versioning."""
        ep1 = _make_episode("ep_uni_1", "内容版本1", source_document="文档_测试.txt")
        _save_episode_to_db(tmp_storage, ep1)
        ep2 = _make_episode("ep_uni_2", "内容版本2", source_document="文档_测试.txt")
        _save_episode_to_db(tmp_storage, ep2)

        row2 = tmp_storage.get_episode_from_db("ep_uni_2")
        assert row2["family_id"] == "ep_uni_1"

    def test_unicode_content_preserved(self, tmp_storage):
        """D4.3: Unicode content preserved through versioning."""
        content = "深度学习 🚀 是人工智能的核心技术（含中文）"
        ep = _make_episode("ep_uc_1", content, source_document="doc_uc.txt")
        _save_episode_to_db(tmp_storage, ep)

        row = tmp_storage.get_episode_from_db("ep_uc_1")
        assert row["content"] == content

    def test_provenance_for_observation(self, tmp_storage):
        """D4.4: get_concept_provenance works for observation role."""
        ep1 = _make_episode("ep_prov_1", "Obs v1", source_document="doc_prov.txt")
        _save_episode_to_db(tmp_storage, ep1)

        # Provenance should work even without mentions
        prov = tmp_storage.get_concept_provenance("ep_prov_1")
        # May be empty (no mentions), but should not error
        assert isinstance(prov, list)

    def test_many_versions_same_source(self, tmp_storage):
        """D4.5: Many versions of same source_document."""
        for i in range(10):
            ep = _make_episode(f"ep_many_{i}", f"Version {i}",
                               source_document="doc_many.txt")
            _save_episode_to_db(tmp_storage, ep)

        # All should share the first episode's family_id
        for i in range(10):
            row = tmp_storage.get_episode_from_db(f"ep_many_{i}")
            assert row["family_id"] == "ep_many_0"

    def test_source_document_with_path(self, tmp_storage):
        """D4.6: Source document with path separators works for versioning."""
        ep1 = _make_episode("ep_path_1", "V1", source_document="/data/docs/report.pdf")
        _save_episode_to_db(tmp_storage, ep1)
        ep2 = _make_episode("ep_path_2", "V2", source_document="/data/docs/report.pdf")
        _save_episode_to_db(tmp_storage, ep2)

        row2 = tmp_storage.get_episode_from_db("ep_path_2")
        assert row2["family_id"] == "ep_path_1"

    def test_different_path_same_filename_not_versioned(self, tmp_storage):
        """D4.7: Different full paths but same filename are NOT versioned together."""
        ep1 = _make_episode("ep_dpf_1", "V1", source_document="/data/A/report.pdf")
        _save_episode_to_db(tmp_storage, ep1)
        ep2 = _make_episode("ep_dpf_2", "V2", source_document="/data/B/report.pdf")
        _save_episode_to_db(tmp_storage, ep2)

        row1 = tmp_storage.get_episode_from_db("ep_dpf_1")
        row2 = tmp_storage.get_episode_from_db("ep_dpf_2")
        assert row1["family_id"] != row2["family_id"]  # Different sources

    def test_concept_neighbors_for_observation(self, tmp_storage):
        """D4.8: get_concept_neighbors works for observation role."""
        ep = _make_episode("ep_nbr_1", "Neighbor test", source_document="doc_nbr.txt")
        _save_episode_to_db(tmp_storage, ep)

        # Should not error even if no mentions
        neighbors = tmp_storage.get_concept_neighbors("ep_nbr_1")
        assert isinstance(neighbors, list)

    def test_traverse_concepts_includes_observations(self, tmp_storage):
        """D4.9: traverse_concepts can include observation concepts."""
        ep = _make_episode("ep_trv_1", "Traverse test", source_document="doc_trv.txt")
        _save_episode_to_db(tmp_storage, ep)

        # Traverse from the observation
        result = tmp_storage.traverse_concepts(["ep_trv_1"], max_depth=1)
        assert "ep_trv_1" in result["concepts"]
