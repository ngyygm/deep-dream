"""
Comprehensive tests for StorageManager.

Covers: basic CRUD, entity/relation versioning, search, merge operations,
doc hash / extraction results, edge cases, and concurrent access.
"""
import sys
import os
import hashlib
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

# Ensure project root is on sys.path so ``processor.*`` imports resolve.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processor.storage.manager import StorageManager
from processor.models import Entity, Relation, Episode
from processor.storage.embedding import EmbeddingClient


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _now() -> datetime:
    """Return a naive UTC datetime for deterministic test data."""
    return datetime.now()


def _make_entity(
    absolute_id: str = "abs_1",
    family_id: str = "eid_1",
    name: str = "TestEntity",
    content: str = "A test entity description.",
    event_time: datetime | None = None,
    processed_time: datetime | None = None,
    episode_id: str = "mc_1",
    source_document: str = "test_doc.md",
) -> Entity:
    return Entity(
        absolute_id=absolute_id,
        family_id=family_id,
        name=name,
        content=content,
        event_time=event_time or _now(),
        processed_time=processed_time or _now(),
        episode_id=episode_id,
        source_document=source_document,
    )


def _make_relation(
    absolute_id: str = "rabs_1",
    family_id: str = "rid_1",
    entity1_absolute_id: str = "abs_1",
    entity2_absolute_id: str = "abs_2",
    content: str = "A test relation.",
    event_time: datetime | None = None,
    processed_time: datetime | None = None,
    episode_id: str = "mc_1",
    source_document: str = "test_doc.md",
) -> Relation:
    return Relation(
        absolute_id=absolute_id,
        family_id=family_id,
        entity1_absolute_id=entity1_absolute_id,
        entity2_absolute_id=entity2_absolute_id,
        content=content,
        event_time=event_time or _now(),
        processed_time=processed_time or _now(),
        episode_id=episode_id,
        source_document=source_document,
    )


def _make_memory_cache(
    absolute_id: str = "mc_1",
    content: str = "# Memory cache content\nSome details here.",
    event_time: datetime | None = None,
    doc_name: str = "test_doc.md",
    activity_type: str = "test",
    source_document: str = "test_doc.md",
) -> Episode:
    return Episode(
        absolute_id=absolute_id,
        content=content,
        event_time=event_time or _now(),
        source_document=source_document,
        activity_type=activity_type,
    )


@pytest.fixture
def embedding_client():
    """EmbeddingClient with use_local=False; falls back gracefully."""
    return EmbeddingClient(use_local=False)


@pytest.fixture
def storage(tmp_path, embedding_client):
    """Create a StorageManager backed by a temporary directory."""
    mgr = StorageManager(str(tmp_path), embedding_client=embedding_client)
    yield mgr
    mgr.close()


# ===================================================================
# 1. Basic CRUD
# ===================================================================

class TestEntityCRUD:
    """save_entity / get_entity_by_family_id / get_entity_by_absolute_id"""

    def test_save_and_get_by_family_id(self, storage):
        e = _make_entity(absolute_id="abs_A", family_id="eid_A", name="Alice")
        storage.save_entity(e)

        result = storage.get_entity_by_family_id("eid_A")
        assert result is not None
        assert result.absolute_id == "abs_A"
        assert result.family_id == "eid_A"
        assert result.name == "Alice"

    def test_save_and_get_by_absolute_id(self, storage):
        e = _make_entity(absolute_id="abs_B", family_id="eid_B", name="Bob")
        storage.save_entity(e)

        result = storage.get_entity_by_absolute_id("abs_B")
        assert result is not None
        assert result.name == "Bob"

    def test_get_nonexistent_entity_returns_none(self, storage):
        assert storage.get_entity_by_family_id("no_such_id") is None
        assert storage.get_entity_by_absolute_id("no_such_abs") is None

    def test_save_entity_preserves_all_fields(self, storage):
        t_event = datetime(2025, 3, 15, 10, 30, 0)
        t_proc = datetime(2025, 3, 15, 10, 31, 0)
        e = _make_entity(
            absolute_id="abs_full",
            family_id="eid_full",
            name="FullEntity",
            content="Detailed content here.",
            event_time=t_event,
            processed_time=t_proc,
            episode_id="mc_special",
            source_document="special.md",
        )
        storage.save_entity(e)

        result = storage.get_entity_by_absolute_id("abs_full")
        assert result is not None
        assert result.name == "FullEntity"
        assert result.content == "Detailed content here."
        assert result.episode_id == "mc_special"
        assert result.source_document == "special.md"
        # Datetime fields should round-trip correctly (ISO format)
        assert result.event_time.year == 2025
        assert result.event_time.month == 3
        assert result.event_time.day == 15


class TestRelationCRUD:
    """save_relation / get_relation_by_family_id / get_relations_by_entities"""

    def test_save_and_get_by_family_id(self, storage):
        # Need entities that the relation references
        e1 = _make_entity(absolute_id="abs_R1", family_id="eid_R1", name="Entity1")
        e2 = _make_entity(absolute_id="abs_R2", family_id="eid_R2", name="Entity2")
        storage.save_entity(e1)
        storage.save_entity(e2)

        r = _make_relation(
            absolute_id="rabs_1",
            family_id="rid_1",
            entity1_absolute_id="abs_R1",
            entity2_absolute_id="abs_R2",
            content="Entity1 knows Entity2",
        )
        storage.save_relation(r)

        result = storage.get_relation_by_family_id("rid_1")
        assert result is not None
        assert result.family_id == "rid_1"
        assert result.content == "Entity1 knows Entity2"

    def test_get_relations_by_entities(self, storage):
        e1 = _make_entity(absolute_id="abs_S1", family_id="eid_S1", name="Source1")
        e2 = _make_entity(absolute_id="abs_S2", family_id="eid_S2", name="Source2")
        storage.save_entity(e1)
        storage.save_entity(e2)

        r = _make_relation(
            absolute_id="rabs_S1",
            family_id="rid_S1",
            entity1_absolute_id="abs_S1",
            entity2_absolute_id="abs_S2",
            content="S1 is related to S2",
        )
        storage.save_relation(r)

        relations = storage.get_relations_by_entities("eid_S1", "eid_S2")
        assert len(relations) >= 1
        assert any(rel.content == "S1 is related to S2" for rel in relations)

    def test_get_nonexistent_relation_returns_none(self, storage):
        assert storage.get_relation_by_family_id("no_such_rid") is None


class TestEpisodeCRUD:
    """save_memory_cache / load_memory_cache"""

    def test_save_and_load_memory_cache(self, storage):
        mc = _make_memory_cache(
            absolute_id="mc_test1",
            content="Cached memory content",
            activity_type="testing",
        )
        text = "Original document text for hashing."
        doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        storage.save_episode(mc, text=text, doc_hash=doc_hash)

        loaded = storage.load_episode("mc_test1")
        assert loaded is not None
        assert loaded.absolute_id == "mc_test1"
        assert "Cached memory content" in loaded.content
        assert loaded.activity_type == "testing"

    def test_load_nonexistent_cache_returns_none(self, storage):
        assert storage.load_episode("nonexistent_mc") is None


# ===================================================================
# 2. Entity versioning
# ===================================================================

class TestEntityVersioning:
    """Multiple entities with same family_id but different absolute_id."""

    def test_version_count(self, storage):
        eid = "eid_versioned"
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        for i in range(3):
            e = _make_entity(
                absolute_id=f"abs_v{i}",
                family_id=eid,
                name=f"VersionedEntity_v{i}",
                content=f"Content v{i}",
                processed_time=t0 + timedelta(seconds=i),
            )
            storage.save_entity(e)

        assert storage.get_entity_version_count(eid) == 3

    def test_get_entity_by_family_id_returns_latest(self, storage):
        eid = "eid_latest"
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        for i in range(3):
            e = _make_entity(
                absolute_id=f"abs_l{i}",
                family_id=eid,
                name=f"LatestEntity_v{i}",
                content=f"Content v{i}",
                processed_time=t0 + timedelta(seconds=i),
            )
            storage.save_entity(e)

        result = storage.get_entity_by_family_id(eid)
        assert result is not None
        # Should return the version with the latest processed_time
        assert result.name == "LatestEntity_v2"

    def test_get_all_entities_returns_latest_per_family_id(self, storage):
        t0 = datetime(2025, 6, 1, 8, 0, 0)
        # Entity A with 2 versions
        for i in range(2):
            storage.save_entity(_make_entity(
                absolute_id=f"abs_ga_A{i}",
                family_id="eid_ga_A",
                name=f"GA_EntityA_v{i}",
                processed_time=t0 + timedelta(seconds=i),
            ))
        # Entity B with 1 version
        storage.save_entity(_make_entity(
            absolute_id="abs_ga_B0",
            family_id="eid_ga_B",
            name="GA_EntityB_v0",
            processed_time=t0 + timedelta(seconds=5),
        ))

        all_entities = storage.get_all_entities()
        eid_set = {e.family_id for e in all_entities}
        assert "eid_ga_A" in eid_set
        assert "eid_ga_B" in eid_set
        assert len(all_entities) == 2

        # The A entity should be the latest version
        a_entity = next(e for e in all_entities if e.family_id == "eid_ga_A")
        assert a_entity.name == "GA_EntityA_v1"

    def test_get_entity_versions_returns_all(self, storage):
        eid = "eid_versions"
        t0 = datetime(2025, 2, 1, 10, 0, 0)
        for i in range(4):
            storage.save_entity(_make_entity(
                absolute_id=f"abs_versions_{i}",
                family_id=eid,
                name=f"VEntity_v{i}",
                processed_time=t0 + timedelta(seconds=i),
            ))

        versions = storage.get_entity_versions(eid)
        assert len(versions) == 4


# ===================================================================
# 3. Relation versioning
# ===================================================================

class TestRelationVersioning:
    """Same pattern with family_id."""

    def test_multiple_relation_versions(self, storage):
        # Create entities
        storage.save_entity(_make_entity(absolute_id="abs_rv_e1", family_id="eid_rv_e1"))
        storage.save_entity(_make_entity(absolute_id="abs_rv_e2", family_id="eid_rv_e2"))

        rid = "rid_versioned"
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        for i in range(3):
            r = _make_relation(
                absolute_id=f"rabs_rv_{i}",
                family_id=rid,
                entity1_absolute_id="abs_rv_e1",
                entity2_absolute_id="abs_rv_e2",
                content=f"Relation content v{i}",
                processed_time=t0 + timedelta(seconds=i),
            )
            storage.save_relation(r)

        # get_relation_by_family_id should return latest
        latest = storage.get_relation_by_family_id(rid)
        assert latest is not None
        assert latest.content == "Relation content v2"

    def test_get_relation_versions_returns_all(self, storage):
        storage.save_entity(_make_entity(absolute_id="abs_rv2_e1", family_id="eid_rv2_e1"))
        storage.save_entity(_make_entity(absolute_id="abs_rv2_e2", family_id="eid_rv2_e2"))

        rid = "rid_versions2"
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        for i in range(3):
            storage.save_relation(_make_relation(
                absolute_id=f"rabs_rv2_{i}",
                family_id=rid,
                entity1_absolute_id="abs_rv2_e1",
                entity2_absolute_id="abs_rv2_e2",
                content=f"Content v{i}",
                processed_time=t0 + timedelta(seconds=i),
            ))

        versions = storage.get_relation_versions(rid)
        assert len(versions) == 3


# ===================================================================
# 4. Search
# ===================================================================

class TestSearch:
    """search_entities_by_similarity with jaccard method, get_all_entities/relations."""

    def test_search_entities_jaccard(self, storage):
        storage.save_entity(_make_entity(
            absolute_id="abs_search1",
            family_id="eid_search1",
            name="Python Programming",
            content="A programming language.",
        ))
        storage.save_entity(_make_entity(
            absolute_id="abs_search2",
            family_id="eid_search2",
            name="Java Programming",
            content="Another programming language.",
        ))
        storage.save_entity(_make_entity(
            absolute_id="abs_search3",
            family_id="eid_search3",
            name="Cooking Recipes",
            content="How to cook Italian food.",
        ))

        results = storage.search_entities_by_similarity(
            query_name="Python Programming",
            similarity_method="jaccard",
            threshold=0.3,
            max_results=5,
        )
        # "Python Programming" should match itself with high jaccard similarity
        assert len(results) >= 1
        names = [e.name for e in results]
        assert "Python Programming" in names

    def test_search_entities_jaccard_name_only_mode(self, storage):
        storage.save_entity(_make_entity(
            absolute_id="abs_nm1",
            family_id="eid_nm1",
            name="Alice Wonderland",
            content="A curious girl.",
        ))
        storage.save_entity(_make_entity(
            absolute_id="abs_nm2",
            family_id="eid_nm2",
            name="Bob Builder",
            content="Can he fix it.",
        ))

        results = storage.search_entities_by_similarity(
            query_name="Alice",
            similarity_method="jaccard",
            threshold=0.1,
            max_results=5,
            text_mode="name_only",
        )
        assert len(results) >= 1
        assert any(e.name == "Alice Wonderland" for e in results)

    def test_get_all_entities_returns_results(self, storage):
        for i in range(5):
            storage.save_entity(_make_entity(
                absolute_id=f"abs_all_e{i}",
                family_id=f"eid_all_e{i}",
                name=f"Entity_{i}",
            ))
        all_entities = storage.get_all_entities()
        assert len(all_entities) == 5

    def test_get_all_relations_returns_results(self, storage):
        storage.save_entity(_make_entity(absolute_id="abs_are1", family_id="eid_are1"))
        storage.save_entity(_make_entity(absolute_id="abs_are2", family_id="eid_are2"))
        storage.save_entity(_make_entity(absolute_id="abs_are3", family_id="eid_are3"))

        for i in range(3):
            storage.save_relation(_make_relation(
                absolute_id=f"rabs_are_{i}",
                family_id=f"rid_are_{i}",
                entity1_absolute_id="abs_are1",
                entity2_absolute_id="abs_are2" if i < 2 else "abs_are3",
                content=f"Relation {i}",
            ))

        all_relations = storage.get_all_relations()
        assert len(all_relations) == 3


# ===================================================================
# 5. Merge operations
# ===================================================================

class TestMergeOperations:
    """merge_family_ids and self-referential relations detection."""

    def test_merge_family_ids(self, storage):
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        # Target entity
        storage.save_entity(_make_entity(
            absolute_id="abs_target",
            family_id="eid_target",
            name="Target Entity",
            processed_time=t0,
        ))
        # Source entities (2)
        storage.save_entity(_make_entity(
            absolute_id="abs_src1",
            family_id="eid_src1",
            name="Source Entity 1",
            processed_time=t0 + timedelta(seconds=1),
        ))
        storage.save_entity(_make_entity(
            absolute_id="abs_src2",
            family_id="eid_src2",
            name="Source Entity 2",
            processed_time=t0 + timedelta(seconds=2),
        ))

        result = storage.merge_entity_families("eid_target", ["eid_src1", "eid_src2"])
        assert result["entities_updated"] == 2

        # After merge, get_entity_by_family_id("eid_target") should still return target entity
        target = storage.get_entity_by_family_id("eid_target")
        assert target is not None
        assert target.family_id == "eid_target"

        # Source family_ids should no longer exist as separate entities
        # All their versions now have family_id = "eid_target"
        assert storage.get_entity_version_count("eid_target") == 3  # 1 target + 2 sources
        assert storage.resolve_family_id("eid_src1") == "eid_target"
        assert storage.resolve_family_id("eid_src2") == "eid_target"
        assert storage.get_entity_by_family_id("eid_src1") is not None
        assert storage.get_entity_by_family_id("eid_src1").family_id == "eid_target"

    def test_entity_redirect_chain_resolves_to_latest(self, storage):
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        storage.save_entity(_make_entity(
            absolute_id="abs_chain_c",
            family_id="eid_chain_c",
            name="Chain Target",
            processed_time=t0,
        ))
        storage.register_entity_redirect("eid_chain_a", "eid_chain_b")
        storage.register_entity_redirect("eid_chain_b", "eid_chain_c")

        assert storage.resolve_family_id("eid_chain_a") == "eid_chain_c"
        resolved = storage.get_entity_by_family_id("eid_chain_a")
        assert resolved is not None
        assert resolved.family_id == "eid_chain_c"

    def test_get_relations_by_entities_accepts_redirected_ids(self, storage):
        t0 = datetime(2025, 1, 1, 12, 0, 0)
        e1 = _make_entity(absolute_id="abs_old_a", family_id="eid_new_a", name="A", processed_time=t0)
        e2 = _make_entity(absolute_id="abs_old_b", family_id="eid_new_b", name="B", processed_time=t0 + timedelta(seconds=1))
        storage.save_entity(e1)
        storage.save_entity(e2)
        storage.register_entity_redirect("eid_old_a", "eid_new_a")
        storage.register_entity_redirect("eid_old_b", "eid_new_b")
        storage.save_relation(_make_relation(
            absolute_id="rid_abs_redirect",
            family_id="rid_redirect",
            entity1_absolute_id=e1.absolute_id,
            entity2_absolute_id=e2.absolute_id,
            content="redirect relation",
            processed_time=t0 + timedelta(seconds=2),
        ))

        relations = storage.get_relations_by_entities("eid_old_a", "eid_old_b")
        assert any(rel.content == "redirect relation" for rel in relations)

    def test_merge_empty_source_list(self, storage):
        storage.save_entity(_make_entity(absolute_id="abs_mt", family_id="eid_mt"))
        result = storage.merge_entity_families("eid_mt", [])
        assert result["entities_updated"] == 0


# ===================================================================
# 6. Doc hash and extraction results
# ===================================================================

class TestDocHashExtraction:
    """save_extraction_result / load_extraction_result / find_cache_by_doc_hash"""

    def _setup_cache_with_doc_hash(self, storage, doc_text="Sample document text."):
        mc = _make_memory_cache(
            absolute_id="mc_extract",
            content="Extraction test cache",
            activity_type="extraction",
        )
        doc_hash = hashlib.md5(doc_text.encode("utf-8")).hexdigest()[:12]
        storage.save_episode(mc, text=doc_text, doc_hash=doc_hash)
        return doc_hash

    def test_save_and_load_extraction_result(self, storage):
        doc_hash = self._setup_cache_with_doc_hash(storage)

        entities = [{"name": "TestEntity", "content": "desc"}]
        relations = [{"content": "rel desc"}]

        ok = storage.save_extraction_result(doc_hash, entities, relations)
        assert ok is True

        result = storage.load_extraction_result(doc_hash)
        assert result is not None
        loaded_entities, loaded_relations = result
        assert len(loaded_entities) == 1
        assert loaded_entities[0]["name"] == "TestEntity"
        assert len(loaded_relations) == 1

    def test_load_extraction_result_not_found(self, storage):
        result = storage.load_extraction_result("nonexistent_hash")
        assert result is None

    def test_find_cache_by_doc_hash(self, storage):
        doc_text = "Another unique document for hash test."
        doc_hash = hashlib.md5(doc_text.encode("utf-8")).hexdigest()[:12]
        mc = _make_memory_cache(
            absolute_id="mc_find_test",
            content="Find cache test content",
            activity_type="finding",
        )
        storage.save_episode(mc, text=doc_text, doc_hash=doc_hash)

        found = storage.find_cache_by_doc_hash(doc_hash)
        assert found is not None
        assert found.absolute_id == "mc_find_test"

    def test_find_cache_by_doc_hash_not_found(self, storage):
        assert storage.find_cache_by_doc_hash("deadbeef0000") is None


# ===================================================================
# 7. Edge cases
# ===================================================================

class TestEdgeCases:
    """Empty content, very long content, Unicode/emoji, SQL injection."""

    def test_empty_content_string(self, storage):
        e = _make_entity(
            absolute_id="abs_empty",
            family_id="eid_empty",
            name="EmptyContentEntity",
            content="",
        )
        storage.save_entity(e)
        result = storage.get_entity_by_absolute_id("abs_empty")
        assert result is not None
        assert result.content == ""

    def test_very_long_content(self, storage):
        long_content = "X" * 15000  # 15000 chars
        e = _make_entity(
            absolute_id="abs_long",
            family_id="eid_long",
            name="LongContentEntity",
            content=long_content,
        )
        storage.save_entity(e)
        result = storage.get_entity_by_absolute_id("abs_long")
        assert result is not None
        assert len(result.content) == 15000

    def test_unicode_emoji_content(self, storage):
        # Use actual Unicode characters (not surrogate pairs) to avoid
        # UnicodeEncodeError when writing through SQLite.
        emoji_content = "Unicode: \u4e2d\u6587\u30c6\u30b9\u30c8 \U0001f600\U0001f30d emoji \u00e9\u00e8\u00ea"
        e = _make_entity(
            absolute_id="abs_unicode",
            family_id="eid_unicode",
            name="Unicode Entity",
            content=emoji_content,
        )
        storage.save_entity(e)
        result = storage.get_entity_by_absolute_id("abs_unicode")
        assert result is not None
        assert "\u4e2d\u6587\u30c6\u30b9\u30c8" in result.content
        assert "\U0001f600" in result.content

    def test_sql_injection_in_name(self, storage):
        malicious_name = "Robert'); DROP TABLE entities; --"
        e = _make_entity(
            absolute_id="abs_sqli",
            family_id="eid_sqli",
            name=malicious_name,
            content="Trying SQL injection",
        )
        storage.save_entity(e)

        # Verify entity was stored correctly and table still exists
        result = storage.get_entity_by_absolute_id("abs_sqli")
        assert result is not None
        assert result.name == malicious_name

        # Verify the table is intact by counting entities
        all_entities = storage.get_all_entities()
        assert len(all_entities) >= 1

    def test_sql_injection_in_content(self, storage):
        malicious_content = "'); INSERT INTO entities VALUES ('hack', 'hack', 'hack', 'hack', 'hack', 'hack', 'hack', 'hack', NULL); --"
        e = _make_entity(
            absolute_id="abs_sqli2",
            family_id="eid_sqli2",
            name="SQLInjectionContent",
            content=malicious_content,
        )
        storage.save_entity(e)
        result = storage.get_entity_by_absolute_id("abs_sqli2")
        assert result is not None
        assert result.content == malicious_content

    def test_empty_content_relation(self, storage):
        storage.save_entity(_make_entity(absolute_id="abs_ecr1", family_id="eid_ecr1"))
        storage.save_entity(_make_entity(absolute_id="abs_ecr2", family_id="eid_ecr2"))

        r = _make_relation(
            absolute_id="rabs_ecr",
            family_id="rid_ecr",
            entity1_absolute_id="abs_ecr1",
            entity2_absolute_id="abs_ecr2",
            content="",
        )
        storage.save_relation(r)
        result = storage.get_relation_by_family_id("rid_ecr")
        assert result is not None
        assert result.content == ""


# ===================================================================
# 8. Concurrent access
# ===================================================================

class TestConcurrentAccess:
    """Spawn 5 threads, each writing 20 entities simultaneously."""

    def test_concurrent_entity_writes(self, storage):
        num_threads = 5
        entities_per_thread = 20
        errors: list = []
        barrier = threading.Barrier(num_threads)

        def writer(thread_idx: int):
            try:
                barrier.wait(timeout=10)
                t0 = datetime(2025, 1, 1, 0, 0, 0)
                for j in range(entities_per_thread):
                    e = _make_entity(
                        absolute_id=f"abs_t{thread_idx}_e{j}",
                        family_id=f"eid_t{thread_idx}_e{j}",
                        name=f"Thread{thread_idx}_Entity{j}",
                        content=f"Content from thread {thread_idx}, entity {j}",
                        processed_time=t0 + timedelta(
                            seconds=thread_idx * entities_per_thread + j
                        ),
                    )
                    storage.save_entity(e)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"

        # Verify all entities are retrievable
        total_expected = num_threads * entities_per_thread
        all_entities = storage.get_all_entities()
        assert len(all_entities) == total_expected

        # Spot-check a few
        for ti in range(num_threads):
            for ej in [0, entities_per_thread - 1]:
                eid = f"eid_t{ti}_e{ej}"
                result = storage.get_entity_by_family_id(eid)
                assert result is not None, f"Missing entity {eid}"
                assert result.name == f"Thread{ti}_Entity{ej}"
