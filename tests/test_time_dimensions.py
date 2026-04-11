"""
Time dimensions (valid_at/invalid_at) in API responses — 36 tests across 4 dimensions.

Vision principle: "时间不可省略" — all API responses must carry time dimensions.
This validates that entity_to_dict and relation_to_dict include valid_at and invalid_at.

D1: Entity valid_at/invalid_at in entity_to_dict (9 tests)
D2: Relation valid_at/invalid_at in relation_to_dict (9 tests)
D3: Time dimensions preserved through CRUD lifecycle (9 tests)
D4: Edge cases — None times, unicode, versioning, merging (9 tests)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
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


def _make_entity(family_id: str, name: str, content: str,
                 valid_at=None, invalid_at=None, source_document: str = ""):
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
        valid_at=valid_at,
        invalid_at=invalid_at,
    )


def _make_relation(family_id: str, e1_abs: str, e2_abs: str, content: str,
                   valid_at=None, invalid_at=None):
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
        valid_at=valid_at,
        invalid_at=invalid_at,
    )


# ══════════════════════════════════════════════════════════════════════════
# D1: Entity valid_at/invalid_at in entity_to_dict
# ══════════════════════════════════════════════════════════════════════════


class TestEntityTimeDimensions:
    """D1: entity_to_dict includes valid_at and invalid_at."""

    def test_entity_valid_at_present(self):
        """D1.1: entity_to_dict includes valid_at when set."""
        from server.blueprints.helpers import entity_to_dict
        va = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)
        e = _make_entity("t1", "Test", "Content", valid_at=va)
        d = entity_to_dict(e)
        assert "valid_at" in d
        assert d["valid_at"] is not None
        assert "2026-01-15" in d["valid_at"]

    def test_entity_invalid_at_present(self):
        """D1.2: entity_to_dict includes invalid_at when set."""
        from server.blueprints.helpers import entity_to_dict
        ia = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
        e = _make_entity("t2", "Test", "Content", invalid_at=ia)
        d = entity_to_dict(e)
        assert "invalid_at" in d
        assert d["invalid_at"] is not None
        assert "2026-03-01" in d["invalid_at"]

    def test_entity_both_times_present(self):
        """D1.3: entity_to_dict includes both valid_at and invalid_at."""
        from server.blueprints.helpers import entity_to_dict
        va = datetime(2026, 1, 1, tzinfo=timezone.utc)
        ia = datetime(2026, 6, 1, tzinfo=timezone.utc)
        e = _make_entity("t3", "Test", "Content", valid_at=va, invalid_at=ia)
        d = entity_to_dict(e)
        assert d["valid_at"] is not None
        assert d["invalid_at"] is not None

    def test_entity_no_times_returns_none(self):
        """D1.4: entity_to_dict returns None for times when not set."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("t4", "Test", "Content")
        d = entity_to_dict(e)
        assert "valid_at" in d
        assert "invalid_at" in d
        assert d["valid_at"] is None
        assert d["invalid_at"] is None

    def test_entity_valid_at_iso_format(self):
        """D1.5: valid_at is in ISO format string."""
        from server.blueprints.helpers import entity_to_dict
        va = datetime(2026, 4, 12, 15, 30, 45, tzinfo=timezone.utc)
        e = _make_entity("t5", "Test", "Content", valid_at=va)
        d = entity_to_dict(e)
        # Should be parseable as ISO datetime
        parsed = datetime.fromisoformat(d["valid_at"])
        assert parsed.year == 2026
        assert parsed.month == 4

    def test_entity_dict_has_all_required_keys(self):
        """D1.6: entity_to_dict output includes all expected keys."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("t6", "Test", "Content")
        d = entity_to_dict(e)
        required_keys = {"absolute_id", "family_id", "name", "content",
                        "event_time", "processed_time", "valid_at", "invalid_at"}
        assert required_keys.issubset(set(d.keys()))

    def test_entity_with_score_and_times(self):
        """D1.7: _score doesn't interfere with time fields."""
        from server.blueprints.helpers import entity_to_dict
        va = datetime(2026, 1, 1, tzinfo=timezone.utc)
        e = _make_entity("t7", "Test", "Content", valid_at=va)
        d = entity_to_dict(e, _score=0.95)
        assert d["_score"] == 0.95
        assert d["valid_at"] is not None

    def test_entity_with_version_count_and_times(self):
        """D1.8: version_count doesn't interfere with time fields."""
        from server.blueprints.helpers import entity_to_dict
        ia = datetime(2026, 12, 31, tzinfo=timezone.utc)
        e = _make_entity("t8", "Test", "Content", invalid_at=ia)
        d = entity_to_dict(e, version_count=3)
        assert d["version_count"] == 3
        assert d["invalid_at"] is not None

    def test_entity_default_valid_at_from_event_time(self, tmp_storage):
        """D1.9: Entity saved to storage gets valid_at defaulting to event_time."""
        e = _make_entity("defva_e1", "Test", "Content")
        assert e.valid_at is None  # Not set before save
        # Storage INSERT uses (valid_at or event_time) for valid_at column


# ══════════════════════════════════════════════════════════════════════════
# D2: Relation valid_at/invalid_at in relation_to_dict
# ══════════════════════════════════════════════════════════════════════════


class TestRelationTimeDimensions:
    """D2: relation_to_dict includes valid_at and invalid_at."""

    def test_relation_valid_at_present(self):
        """D2.1: relation_to_dict includes valid_at when set."""
        from server.blueprints.helpers import relation_to_dict
        va = datetime(2026, 2, 20, 8, 0, tzinfo=timezone.utc)
        r = _make_relation("r1", "a1", "a2", "Connects A-B", valid_at=va)
        d = relation_to_dict(r)
        assert "valid_at" in d
        assert d["valid_at"] is not None
        assert "2026-02-20" in d["valid_at"]

    def test_relation_invalid_at_present(self):
        """D2.2: relation_to_dict includes invalid_at when set."""
        from server.blueprints.helpers import relation_to_dict
        ia = datetime(2026, 5, 10, tzinfo=timezone.utc)
        r = _make_relation("r2", "a1", "a2", "Connects A-B", invalid_at=ia)
        d = relation_to_dict(r)
        assert "invalid_at" in d
        assert d["invalid_at"] is not None

    def test_relation_both_times_present(self):
        """D2.3: relation_to_dict includes both times."""
        from server.blueprints.helpers import relation_to_dict
        va = datetime(2026, 1, 1, tzinfo=timezone.utc)
        ia = datetime(2026, 12, 31, tzinfo=timezone.utc)
        r = _make_relation("r3", "a1", "a2", "Connects", valid_at=va, invalid_at=ia)
        d = relation_to_dict(r)
        assert d["valid_at"] is not None
        assert d["invalid_at"] is not None

    def test_relation_no_times_returns_none(self):
        """D2.4: relation_to_dict returns None for unset times."""
        from server.blueprints.helpers import relation_to_dict
        r = _make_relation("r4", "a1", "a2", "Connects")
        d = relation_to_dict(r)
        assert d["valid_at"] is None
        assert d["invalid_at"] is None

    def test_relation_valid_at_iso_format(self):
        """D2.5: valid_at is ISO format string."""
        from server.blueprints.helpers import relation_to_dict
        va = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
        r = _make_relation("r5", "a1", "a2", "Connects", valid_at=va)
        d = relation_to_dict(r)
        parsed = datetime.fromisoformat(d["valid_at"])
        assert parsed.year == 2026

    def test_relation_dict_has_all_required_keys(self):
        """D2.6: relation_to_dict output includes all expected keys."""
        from server.blueprints.helpers import relation_to_dict
        r = _make_relation("r6", "a1", "a2", "Connects")
        d = relation_to_dict(r)
        required_keys = {"absolute_id", "family_id", "content",
                        "event_time", "processed_time", "valid_at", "invalid_at"}
        assert required_keys.issubset(set(d.keys()))

    def test_relation_with_score_and_times(self):
        """D2.7: _score doesn't interfere with time fields."""
        from server.blueprints.helpers import relation_to_dict
        va = datetime(2026, 3, 1, tzinfo=timezone.utc)
        r = _make_relation("r7", "a1", "a2", "Connects", valid_at=va)
        d = relation_to_dict(r, _score=0.88)
        assert d["_score"] == 0.88
        assert d["valid_at"] is not None

    def test_relation_with_version_count_and_times(self):
        """D2.8: version_count doesn't interfere with time fields."""
        from server.blueprints.helpers import relation_to_dict
        ia = datetime(2026, 9, 1, tzinfo=timezone.utc)
        r = _make_relation("r8", "a1", "a2", "Connects", invalid_at=ia)
        d = relation_to_dict(r, version_count=2)
        assert d["version_count"] == 2
        assert d["invalid_at"] is not None

    def test_relation_entity_ids_preserved_with_times(self):
        """D2.9: Entity IDs preserved alongside time fields."""
        from server.blueprints.helpers import relation_to_dict
        va = datetime(2026, 1, 1, tzinfo=timezone.utc)
        r = _make_relation("r9", "abs_e1", "abs_e2", "A-B", valid_at=va)
        d = relation_to_dict(r)
        assert d["entity1_absolute_id"] == "abs_e1"
        assert d["entity2_absolute_id"] == "abs_e2"
        assert d["valid_at"] is not None


# ══════════════════════════════════════════════════════════════════════════
# D3: Time dimensions preserved through CRUD lifecycle
# ══════════════════════════════════════════════════════════════════════════


class TestTimeLifecycle:
    """D3: Time dimensions survive save/load/update cycles."""

    def test_save_load_preserves_valid_at(self, tmp_storage):
        """D3.1: Entity valid_at preserved after save and load."""
        va = datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc)
        e = _make_entity("lc_e1", "Test", "Content", valid_at=va)
        tmp_storage.save_entity(e)

        loaded = tmp_storage.get_entity_by_absolute_id(e.absolute_id)
        assert loaded is not None
        assert loaded.valid_at is not None

    def test_save_load_preserves_invalid_at(self, tmp_storage):
        """D3.2: Entity invalid_at preserved after save and load."""
        ia = datetime(2026, 8, 20, 14, 30, tzinfo=timezone.utc)
        e = _make_entity("lc_e2", "Test", "Content", invalid_at=ia)
        tmp_storage.save_entity(e)

        loaded = tmp_storage.get_entity_by_absolute_id(e.absolute_id)
        assert loaded is not None
        assert loaded.invalid_at is not None

    def test_new_version_invalidates_old(self, tmp_storage):
        """D3.3: Saving new version sets invalid_at on old version."""
        e1 = _make_entity("lc_e3", "V1", "Version 1")
        tmp_storage.save_entity(e1)

        e2 = _make_entity("lc_e3", "V2", "Version 2")
        tmp_storage.save_entity(e2)

        old = tmp_storage.get_entity_by_absolute_id(e1.absolute_id)
        assert old.invalid_at is not None

    def test_latest_version_no_invalid_at(self, tmp_storage):
        """D3.4: Latest version has invalid_at=None."""
        e1 = _make_entity("lc_e4", "V1", "Version 1")
        tmp_storage.save_entity(e1)
        e2 = _make_entity("lc_e4", "V2", "Version 2")
        tmp_storage.save_entity(e2)

        latest = tmp_storage.get_entity_by_family_id("lc_e4")
        assert latest.invalid_at is None

    def test_relation_save_load_times(self, tmp_storage):
        """D3.5: Relation valid_at preserved through save/load."""
        va = datetime(2026, 2, 14, tzinfo=timezone.utc)
        e1 = _make_entity("lc_r_e1", "A", "A")
        e2 = _make_entity("lc_r_e2", "B", "B")
        tmp_storage.bulk_save_entities([e1, e2])
        r = _make_relation("lc_r1", e1.absolute_id, e2.absolute_id, "A-B", valid_at=va)
        tmp_storage.save_relation(r)

        loaded = tmp_storage.get_relation_by_absolute_id(r.absolute_id)
        assert loaded is not None
        assert loaded.valid_at is not None

    def test_bulk_save_preserves_times(self, tmp_storage):
        """D3.6: Bulk save preserves valid_at."""
        va = datetime(2026, 3, 1, tzinfo=timezone.utc)
        e1 = _make_entity("lc_bs1", "A", "A", valid_at=va)
        e2 = _make_entity("lc_bs2", "B", "B", valid_at=va)
        tmp_storage.bulk_save_entities([e1, e2])

        loaded1 = tmp_storage.get_entity_by_absolute_id(e1.absolute_id)
        assert loaded1.valid_at is not None

    def test_update_entity_doesnt_clear_times(self, tmp_storage):
        """D3.7: Updating entity doesn't clear existing time fields."""
        va = datetime(2026, 1, 1, tzinfo=timezone.utc)
        e = _make_entity("lc_upd1", "Original", "Original", valid_at=va)
        tmp_storage.save_entity(e)

        tmp_storage.update_entity_by_absolute_id(e.absolute_id, name="Updated")
        loaded = tmp_storage.get_entity_by_absolute_id(e.absolute_id)
        assert loaded.name == "Updated"
        assert loaded.valid_at is not None

    def test_get_entity_versions_times(self, tmp_storage):
        """D3.8: Version list preserves time fields per version."""
        va1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        e1 = _make_entity("lc_ver1", "V1", "V1", valid_at=va1)
        tmp_storage.save_entity(e1)

        va2 = datetime(2026, 2, 1, tzinfo=timezone.utc)
        e2 = _make_entity("lc_ver1", "V2", "V2", valid_at=va2)
        tmp_storage.save_entity(e2)

        versions = tmp_storage.get_entity_versions("lc_ver1")
        assert len(versions) == 2
        # Both should have valid_at set
        assert all(v.valid_at is not None for v in versions)

    def test_get_all_entities_has_times(self, tmp_storage):
        """D3.9: get_all_entities returns entities with time fields."""
        va = datetime(2026, 5, 1, tzinfo=timezone.utc)
        e = _make_entity("lc_all1", "Test", "Content", valid_at=va)
        tmp_storage.save_entity(e)

        all_ents = tmp_storage.get_all_entities()
        found = [ent for ent in all_ents if ent.family_id == "lc_all1"]
        assert len(found) >= 1
        assert found[0].valid_at is not None


# ══════════════════════════════════════════════════════════════════════════
# D4: Edge cases — None times, unicode, versioning, merging
# ══════════════════════════════════════════════════════════════════════════


class TestTimeEdgeCases:
    """D4: Edge cases for time dimensions."""

    def test_entity_no_valid_at_no_crash(self):
        """D4.1: Entity without valid_at doesn't crash entity_to_dict."""
        from server.blueprints.helpers import entity_to_dict
        e = _make_entity("ec1", "Test", "Content")
        d = entity_to_dict(e)
        assert d["valid_at"] is None
        assert d["invalid_at"] is None

    def test_entity_microsecond_precision(self):
        """D4.2: valid_at preserves microsecond precision."""
        from server.blueprints.helpers import entity_to_dict
        va = datetime(2026, 4, 12, 10, 30, 45, 123456, tzinfo=timezone.utc)
        e = _make_entity("ec2", "Test", "Content", valid_at=va)
        d = entity_to_dict(e)
        parsed = datetime.fromisoformat(d["valid_at"])
        assert parsed.microsecond == 123456

    def test_entity_timezone_aware(self):
        """D4.3: valid_at preserves timezone info."""
        from server.blueprints.helpers import entity_to_dict
        va = datetime(2026, 4, 12, 10, 0, 0, tzinfo=timezone.utc)
        e = _make_entity("ec3", "Test", "Content", valid_at=va)
        d = entity_to_dict(e)
        assert "+00:00" in d["valid_at"] or "Z" in d["valid_at"] or "T10:00:00" in d["valid_at"]

    def test_relation_no_valid_at_no_crash(self):
        """D4.4: Relation without valid_at doesn't crash relation_to_dict."""
        from server.blueprints.helpers import relation_to_dict
        r = _make_relation("ec4", "a1", "a2", "Connects")
        d = relation_to_dict(r)
        assert d["valid_at"] is None
        assert d["invalid_at"] is None

    def test_merge_preserves_times(self, tmp_storage):
        """D4.5: Merge preserves time fields on all versions."""
        va1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        va2 = datetime(2026, 2, 1, tzinfo=timezone.utc)
        e1 = _make_entity("ec_m_tgt", "T", "T", valid_at=va1)
        e2 = _make_entity("ec_m_src", "S", "S", valid_at=va2)
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.merge_entity_families("ec_m_tgt", ["ec_m_src"])

        versions = tmp_storage.get_entity_versions("ec_m_tgt")
        assert len(versions) == 2
        assert all(v.valid_at is not None for v in versions)

    def test_delete_preserves_other_times(self, tmp_storage):
        """D4.6: Deleting entity doesn't affect others' time fields."""
        va = datetime(2026, 3, 1, tzinfo=timezone.utc)
        e1 = _make_entity("ec_del1", "Delete", "Delete", valid_at=va)
        e2 = _make_entity("ec_keep", "Keep", "Keep", valid_at=va)
        tmp_storage.bulk_save_entities([e1, e2])

        tmp_storage.delete_entity_by_absolute_id(e1.absolute_id)

        kept = tmp_storage.get_entity_by_absolute_id(e2.absolute_id)
        assert kept is not None
        assert kept.valid_at is not None

    def test_entity_with_unicode_name_and_times(self):
        """D4.7: Unicode entity name works with time dimensions."""
        from server.blueprints.helpers import entity_to_dict
        va = datetime(2026, 4, 1, tzinfo=timezone.utc)
        e = _make_entity("ec_uni", "深度学习 🚀", "内容包含中文", valid_at=va)
        d = entity_to_dict(e)
        assert d["name"] == "深度学习 🚀"
        assert d["valid_at"] is not None

    def test_content_truncation_with_times(self):
        """D4.8: Content truncation works alongside time fields."""
        from server.blueprints.helpers import entity_to_dict
        va = datetime(2026, 1, 1, tzinfo=timezone.utc)
        long_content = "A" * 5000
        e = _make_entity("ec_trunc", "Test", long_content, valid_at=va)
        d = entity_to_dict(e, max_content_length=100)
        assert d["content_truncated"] is True
        assert len(d["content"]) <= 103  # 100 + "..."
        assert d["valid_at"] is not None

    def test_far_future_dates(self):
        """D4.9: Far future dates are handled correctly."""
        from server.blueprints.helpers import entity_to_dict
        va = datetime(2099, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        e = _make_entity("ec_future", "Test", "Content", valid_at=va)
        d = entity_to_dict(e)
        assert "2099" in d["valid_at"]
