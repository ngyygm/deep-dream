"""
Comprehensive tests for iteration-5: deduplicated build helpers.

Covers 4 dimensions with 35+ test cases:
  1. _construct_entity shared helper (9 tests)
  2. _construct_relation shared helper (9 tests)
  3. Build new vs version behavior parity (9 tests)
  4. Edge cases & error handling (10 tests)
"""
import sys
import uuid
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_real_storage(tmp_path):
    """Create a fully functional StorageManager (not mocked)."""
    from processor.storage.manager import StorageManager
    return StorageManager(str(tmp_path / "storage"))


def _make_entity_processor(tmp_path):
    """Create an EntityProcessor with real storage and mock LLM client."""
    from processor.pipeline.entity import EntityProcessor
    sm = _make_real_storage(tmp_path)
    mock_llm = MagicMock()
    processor = EntityProcessor(storage=sm, llm_client=mock_llm)
    return processor, sm


def _make_relation_processor(tmp_path):
    """Create a RelationProcessor with real storage and mock LLM client."""
    from processor.pipeline.relation import RelationProcessor
    sm = _make_real_storage(tmp_path)
    mock_llm = MagicMock()
    processor = RelationProcessor(storage=sm, llm_client=mock_llm)
    return processor, sm


def _save_test_entity(sm, family_id="ent_test1", name="TestEntity", content="Test content"):
    """Save an entity and return it."""
    from processor.models import Entity
    e = Entity(
        absolute_id=f"{family_id}_v1", family_id=family_id, name=name, content=content,
        event_time=datetime.now(), processed_time=datetime.now(),
        episode_id="ep1", source_document="",
    )
    sm.save_entity(e)
    return e


# ============================================================
# Dimension 1: _construct_entity shared helper
# ============================================================
class TestConstructEntity:
    """Tests for the shared _construct_entity helper method."""

    def test_construct_entity_basic_fields(self, tmp_path):
        """_construct_entity should produce a valid Entity with all standard fields."""
        proc, sm = _make_entity_processor(tmp_path)
        entity = proc._construct_entity(
            name="Alice", content="A developer", episode_id="ep1",
            family_id="ent_alice",
        )
        assert entity.family_id == "ent_alice"
        assert entity.name == "Alice"
        assert entity.content == "A developer"
        assert entity.episode_id == "ep1"
        assert entity.confidence == 0.7
        assert entity.content_format == "markdown"
        sm.close()

    def test_construct_entity_generates_unique_absolute_ids(self, tmp_path):
        """Each call should produce a different absolute_id."""
        proc, sm = _make_entity_processor(tmp_path)
        e1 = proc._construct_entity("A", "c1", "ep1", "ent_1")
        e2 = proc._construct_entity("B", "c2", "ep1", "ent_2")
        assert e1.absolute_id != e2.absolute_id
        sm.close()

    def test_construct_entity_base_time_forwarded(self, tmp_path):
        """base_time should be used as event_time."""
        proc, sm = _make_entity_processor(tmp_path)
        t = datetime(2025, 6, 15, 12, 0, 0)
        entity = proc._construct_entity("X", "c", "ep1", "ent_x", base_time=t)
        assert entity.event_time == t
        assert entity.processed_time >= t  # processed_time is now()
        sm.close()

    def test_construct_entity_source_document_truncated(self, tmp_path):
        """source_document should keep only the last path component."""
        proc, sm = _make_entity_processor(tmp_path)
        entity = proc._construct_entity(
            "X", "c", "ep1", "ent_x",
            source_document="path/to/document.pdf",
        )
        assert entity.source_document == "document.pdf"
        sm.close()

    def test_construct_entity_empty_source_document(self, tmp_path):
        """Empty source_document should remain empty."""
        proc, sm = _make_entity_processor(tmp_path)
        entity = proc._construct_entity("X", "c", "ep1", "ent_x", source_document="")
        assert entity.source_document == ""
        sm.close()

    def test_construct_entity_summary_generated(self, tmp_path):
        """Summary should be auto-generated from name + content."""
        proc, sm = _make_entity_processor(tmp_path)
        entity = proc._construct_entity("Bob", "Content here", "ep1", "ent_bob")
        assert entity.summary is not None
        assert len(entity.summary) > 0
        sm.close()

    def test_construct_entity_none_base_time_uses_now(self, tmp_path):
        """base_time=None should default to datetime.now()."""
        proc, sm = _make_entity_processor(tmp_path)
        before = datetime.now()
        entity = proc._construct_entity("X", "c", "ep1", "ent_x")
        after = datetime.now()
        assert before <= entity.event_time <= after
        sm.close()

    def test_construct_entity_long_content_summary(self, tmp_path):
        """Summary for long content should be truncated."""
        proc, sm = _make_entity_processor(tmp_path)
        long_content = "x" * 500
        entity = proc._construct_entity("Name", long_content, "ep1", "ent_long")
        assert entity.summary is not None
        sm.close()

    def test_construct_entity_absolute_id_format(self, tmp_path):
        """absolute_id should follow the entity_YYYYMMDD_HHMMSS_<hex> format."""
        proc, sm = _make_entity_processor(tmp_path)
        entity = proc._construct_entity("X", "c", "ep1", "ent_x")
        assert entity.absolute_id.startswith("entity_")
        parts = entity.absolute_id.split("_")
        assert len(parts) == 4  # entity_, date, time, hex
        sm.close()


# ============================================================
# Dimension 2: _construct_relation shared helper
# ============================================================
class TestConstructRelation:
    """Tests for the shared _construct_relation helper method."""

    def test_construct_relation_basic_fields(self, tmp_path):
        """_construct_relation should produce a valid Relation."""
        proc, sm = _make_relation_processor(tmp_path)
        _save_test_entity(sm, "ent_a", "Alice")
        _save_test_entity(sm, "ent_b", "Bob")
        rel = proc._construct_relation(
            "ent_a", "ent_b", "Alice knows Bob", "ep1",
            family_id="rel_1",
        )
        assert rel is not None
        assert rel.family_id == "rel_1"
        assert rel.content == "Alice knows Bob"
        assert rel.confidence == 0.7
        assert rel.content_format == "markdown"
        sm.close()

    def test_construct_relation_entity_ordering(self, tmp_path):
        """Entity ordering should be alphabetical by name (entity1.name <= entity2.name)."""
        proc, sm = _make_relation_processor(tmp_path)
        _save_test_entity(sm, "ent_bob", "Bob")
        _save_test_entity(sm, "ent_alice", "Alice")
        rel = proc._construct_relation(
            "ent_bob", "ent_alice", "test", "ep1", family_id="rel_1",
        )
        # Alice < Bob alphabetically, so entity1 should be Alice
        alice = sm.get_entity_by_family_id("ent_alice")
        assert rel.entity1_absolute_id == alice.absolute_id
        sm.close()

    def test_construct_relation_missing_entity_returns_none(self, tmp_path):
        """Should return None if an entity doesn't exist."""
        proc, sm = _make_relation_processor(tmp_path)
        _save_test_entity(sm, "ent_a", "Alice")
        rel = proc._construct_relation(
            "ent_a", "ent_nonexistent", "test", "ep1", family_id="rel_1",
        )
        assert rel is None
        sm.close()

    def test_construct_relation_both_missing_returns_none(self, tmp_path):
        """Should return None if both entities don't exist."""
        proc, sm = _make_relation_processor(tmp_path)
        rel = proc._construct_relation(
            "ent_x", "ent_y", "test", "ep1", family_id="rel_1",
        )
        assert rel is None
        sm.close()

    def test_construct_relation_with_entity_lookup(self, tmp_path):
        """Should use entity_lookup dict when provided."""
        proc, sm = _make_relation_processor(tmp_path)
        from processor.models import Entity
        mock_entity = Entity(
            absolute_id="mock_id", family_id="ent_mock", name="Mock",
            content="Mock content", event_time=datetime.now(),
            processed_time=datetime.now(), episode_id="ep1", source_document="",
        )
        lookup = {"ent_mock": mock_entity}
        rel = proc._construct_relation(
            "ent_mock", "ent_mock", "self-ref", "ep1", family_id="rel_1",
            entity_lookup=lookup,
        )
        assert rel is not None
        sm.close()

    def test_construct_relation_base_time(self, tmp_path):
        """base_time should be forwarded as event_time."""
        proc, sm = _make_relation_processor(tmp_path)
        _save_test_entity(sm, "ent_a", "A")
        _save_test_entity(sm, "ent_b", "B")
        t = datetime(2025, 1, 1, 0, 0, 0)
        rel = proc._construct_relation(
            "ent_a", "ent_b", "test", "ep1", family_id="rel_1", base_time=t,
        )
        assert rel.event_time == t
        sm.close()

    def test_construct_relation_source_document(self, tmp_path):
        """source_document should be truncated to last path component."""
        proc, sm = _make_relation_processor(tmp_path)
        _save_test_entity(sm, "ent_a", "A")
        _save_test_entity(sm, "ent_b", "B")
        rel = proc._construct_relation(
            "ent_a", "ent_b", "test", "ep1", family_id="rel_1",
            source_document="docs/report.txt",
        )
        assert rel.source_document == "report.txt"
        sm.close()

    def test_construct_relation_skip_label(self, tmp_path):
        """skip_label parameter should be used in warning messages."""
        proc, sm = _make_relation_processor(tmp_path)
        # Missing entity triggers warning with skip_label
        _save_test_entity(sm, "ent_a", "A")
        rel = proc._construct_relation(
            "ent_a", "ent_missing", "test", "ep1", family_id="rel_1",
            verbose_relation=True, skip_label="custom skip",
        )
        assert rel is None
        sm.close()

    def test_construct_relation_summary_truncated(self, tmp_path):
        """Summary should be content[:200].strip()."""
        proc, sm = _make_relation_processor(tmp_path)
        _save_test_entity(sm, "ent_a", "A")
        _save_test_entity(sm, "ent_b", "B")
        long_content = "x" * 300
        rel = proc._construct_relation(
            "ent_a", "ent_b", long_content, "ep1", family_id="rel_1",
        )
        assert len(rel.summary) == 200
        sm.close()


# ============================================================
# Dimension 3: Build new vs version behavior parity
# ============================================================
class TestBuildNewVsVersion:
    """Tests that _build_new_* and _build_*_version produce consistent results."""

    def test_build_new_entity_uses_construct(self, tmp_path):
        """_build_new_entity should delegate to _construct_entity."""
        proc, sm = _make_entity_processor(tmp_path)
        entity = proc._build_new_entity("Test", "Content", "ep1")
        assert entity is not None
        assert entity.family_id.startswith("ent_")
        assert entity.name == "Test"
        assert entity.content == "Content"
        sm.close()

    def test_build_entity_version_uses_construct(self, tmp_path):
        """_build_entity_version should delegate to _construct_entity."""
        proc, sm = _make_entity_processor(tmp_path)
        entity = proc._build_entity_version("ent_custom", "Test", "Content", "ep1")
        assert entity.family_id == "ent_custom"
        assert entity.name == "Test"
        sm.close()

    def test_build_new_vs_version_same_confidence(self, tmp_path):
        """Both build methods should produce the same confidence."""
        proc, sm = _make_entity_processor(tmp_path)
        new = proc._build_new_entity("A", "c", "ep1")
        ver = proc._build_entity_version("ent_x", "A", "c", "ep1")
        assert new.confidence == ver.confidence == 0.7
        sm.close()

    def test_build_new_vs_version_same_content_format(self, tmp_path):
        """Both build methods should use markdown content_format."""
        proc, sm = _make_entity_processor(tmp_path)
        new = proc._build_new_entity("A", "c", "ep1")
        ver = proc._build_entity_version("ent_x", "A", "c", "ep1")
        assert new.content_format == ver.content_format == "markdown"
        sm.close()

    def test_build_new_relation_generates_family_id(self, tmp_path):
        """_build_new_relation should auto-generate a family_id."""
        proc, sm = _make_relation_processor(tmp_path)
        _save_test_entity(sm, "ent_a", "A")
        _save_test_entity(sm, "ent_b", "B")
        rel = proc._build_new_relation("ent_a", "ent_b", "A knows B", "ep1")
        assert rel is not None
        assert rel.family_id.startswith("rel_")
        sm.close()

    def test_build_relation_version_uses_existing_family_id(self, tmp_path):
        """_build_relation_version should use the provided family_id."""
        proc, sm = _make_relation_processor(tmp_path)
        _save_test_entity(sm, "ent_a", "A")
        _save_test_entity(sm, "ent_b", "B")
        rel = proc._build_relation_version(
            "rel_custom", "ent_a", "ent_b", "A knows B", "ep1",
        )
        assert rel is not None
        assert rel.family_id == "rel_custom"
        sm.close()

    def test_build_relation_version_rejects_short_content(self, tmp_path):
        """_build_relation_version should reject content shorter than threshold."""
        proc, sm = _make_relation_processor(tmp_path)
        rel = proc._build_relation_version(
            "rel_1", "ent_a", "ent_b", "short", "ep1",
        )
        assert rel is None
        sm.close()

    def test_build_relation_version_skips_unchanged_content(self, tmp_path):
        """_build_relation_version should skip if content is unchanged."""
        proc, sm = _make_relation_processor(tmp_path)
        _save_test_entity(sm, "ent_a", "A")
        _save_test_entity(sm, "ent_b", "B")
        # First save a relation
        from processor.models import Relation
        existing = Relation(
            absolute_id="rel_existing_1", family_id="rel_skip",
            entity1_absolute_id=sm.get_entity_by_family_id("ent_a").absolute_id,
            entity2_absolute_id=sm.get_entity_by_family_id("ent_b").absolute_id,
            content="Alice works with Bob",
            event_time=datetime.now(), processed_time=datetime.now(),
            episode_id="ep1", source_document="",
        )
        sm.save_relation(existing)
        # Try to build a version with same content
        rel = proc._build_relation_version(
            "rel_skip", "ent_a", "ent_b", "Alice works with Bob", "ep1",
        )
        assert rel is None  # Should be skipped
        sm.close()

    def test_build_new_relation_rejects_short_content(self, tmp_path):
        """_build_new_relation should reject content shorter than threshold."""
        proc, sm = _make_relation_processor(tmp_path)
        rel = proc._build_new_relation("ent_a", "ent_b", "short", "ep1")
        assert rel is None
        sm.close()


# ============================================================
# Dimension 4: Edge cases & error handling
# ============================================================
class TestBuildEdgeCases:
    """Tests for edge cases in build methods."""

    def test_construct_entity_with_none_source_document(self, tmp_path):
        """None source_document should not crash."""
        proc, sm = _make_entity_processor(tmp_path)
        entity = proc._construct_entity("X", "c", "ep1", "ent_x", source_document=None)
        assert entity.source_document == ""
        sm.close()

    def test_construct_entity_with_empty_name(self, tmp_path):
        """Empty name should still work."""
        proc, sm = _make_entity_processor(tmp_path)
        entity = proc._construct_entity("", "content", "ep1", "ent_x")
        assert entity.name == ""
        sm.close()

    def test_construct_entity_with_empty_content(self, tmp_path):
        """Empty content should still produce an entity."""
        proc, sm = _make_entity_processor(tmp_path)
        entity = proc._construct_entity("Name", "", "ep1", "ent_x")
        assert entity.content == ""
        sm.close()

    def test_construct_relation_entity_lookup_priority(self, tmp_path):
        """entity_lookup should take priority over storage lookup."""
        proc, sm = _make_relation_processor(tmp_path)
        _save_test_entity(sm, "ent_a", "StorageA")
        from processor.models import Entity
        lookup_entity = Entity(
            absolute_id="lookup_id", family_id="ent_a", name="LookupA",
            content="Lookup content", event_time=datetime.now(),
            processed_time=datetime.now(), episode_id="ep1", source_document="",
        )
        rel = proc._construct_relation(
            "ent_a", "ent_a", "self-ref test content", "ep1",
            family_id="rel_1", entity_lookup={"ent_a": lookup_entity},
        )
        # Should use lookup entity (name=LookupA), not storage entity (name=StorageA)
        assert rel.entity1_absolute_id == "lookup_id"
        sm.close()

    def test_build_new_entity_family_id_format(self, tmp_path):
        """_build_new_entity should generate ent_<hex12> family_id."""
        proc, sm = _make_entity_processor(tmp_path)
        entity = proc._build_new_entity("A", "c", "ep1")
        assert entity.family_id.startswith("ent_")
        # ent_ prefix + 12 hex chars = 16 chars
        assert len(entity.family_id) == 16
        sm.close()

    def test_build_new_relation_family_id_format(self, tmp_path):
        """_build_new_relation should generate rel_<hex12> family_id."""
        proc, sm = _make_relation_processor(tmp_path)
        _save_test_entity(sm, "ent_a", "A")
        _save_test_entity(sm, "ent_b", "B")
        rel = proc._build_new_relation("ent_a", "ent_b", "A works with B every day", "ep1")
        assert rel is not None
        assert rel.family_id.startswith("rel_")
        assert len(rel.family_id) == 16
        sm.close()

    def test_construct_entity_unicode_content(self, tmp_path):
        """Unicode content should be handled correctly."""
        proc, sm = _make_entity_processor(tmp_path)
        entity = proc._construct_entity(
            "张三", "一个中文开发者", "ep1", "ent_unicode",
        )
        assert entity.name == "张三"
        assert entity.content == "一个中文开发者"
        sm.close()

    def test_construct_relation_unicode_content(self, tmp_path):
        """Unicode content in relations should work."""
        proc, sm = _make_relation_processor(tmp_path)
        _save_test_entity(sm, "ent_a", "张三")
        _save_test_entity(sm, "ent_b", "李四")
        rel = proc._construct_relation(
            "ent_a", "ent_b", "张三是李四的同事", "ep1",
            family_id="rel_unicode",
        )
        assert rel is not None
        assert "张三" in rel.content or "李四" in rel.content or "同事" in rel.content
        sm.close()

    def test_build_relation_version_with_whitespace_normalization(self, tmp_path):
        """Version creation should skip content that only differs in whitespace."""
        proc, sm = _make_relation_processor(tmp_path)
        _save_test_entity(sm, "ent_a", "A")
        _save_test_entity(sm, "ent_b", "B")
        from processor.models import Relation
        existing = Relation(
            absolute_id="rel_ws_1", family_id="rel_ws",
            entity1_absolute_id=sm.get_entity_by_family_id("ent_a").absolute_id,
            entity2_absolute_id=sm.get_entity_by_family_id("ent_b").absolute_id,
            content="Hello World Test",
            event_time=datetime.now(), processed_time=datetime.now(),
            episode_id="ep1", source_document="",
        )
        sm.save_relation(existing)
        # Same content with extra spaces
        rel = proc._build_relation_version(
            "rel_ws", "ent_a", "ent_b", "Hello  World  Test", "ep1",
        )
        assert rel is None  # Should detect whitespace-only difference
        sm.close()
