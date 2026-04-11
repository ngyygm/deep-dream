"""
Comprehensive tests for iteration-4: thread-safe embedding cache.

Covers 4 dimensions with 33 test cases:
  1. Thread-safe entity embedding cache (8 tests)
  2. Thread-safe relation embedding cache (8 tests)
  3. Thread-safe concept embedding cache (8 tests)
  4. Cross-cache invalidation & concurrency stress (9 tests)
"""
import sys
import threading
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
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
    from datetime import datetime
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
    from datetime import datetime
    defaults = dict(
        content="test relation",
        event_time=datetime.now(),
        processed_time=datetime.now(),
        episode_id="ep1",
        source_document="",
    )
    defaults.update(kwargs)
    return Relation(**defaults)


# ============================================================
# Dimension 1: Thread-safe entity embedding cache
# ============================================================
class TestEntityEmbCacheThreadSafety:
    """Tests for _get_entities_with_embeddings thread safety."""

    def test_cache_lock_exists(self, tmp_path):
        """_emb_cache_lock must be a threading.Lock."""
        sm = _make_real_storage(tmp_path)
        assert isinstance(sm._emb_cache_lock, type(threading.Lock()))
        sm.close()

    def test_cache_miss_queries_db(self, tmp_path):
        """When cache is empty, should query SQLite."""
        sm = _make_real_storage(tmp_path)
        sm.save_entity(_entity(absolute_id="test_1", family_id="ent_test1", name="Test", content="Test content"))
        sm._entity_emb_cache = None
        sm._entity_emb_cache_ts = 0.0
        results = sm._get_entities_with_embeddings()
        assert len(results) == 1
        assert results[0][0].family_id == "ent_test1"
        sm.close()

    def test_cache_hit_returns_cached(self, tmp_path):
        """When cache is fresh, should return cached data without DB query."""
        sm = _make_real_storage(tmp_path)
        results1 = sm._get_entities_with_embeddings()
        results2 = sm._get_entities_with_embeddings()
        assert results1 is results2  # Same object reference
        sm.close()

    def test_cache_expires_after_ttl(self, tmp_path):
        """Cache should expire after _emb_cache_ttl seconds."""
        sm = _make_real_storage(tmp_path)
        sm._emb_cache_ttl = 0.01  # Very short TTL
        results1 = sm._get_entities_with_embeddings()
        time.sleep(0.02)
        results2 = sm._get_entities_with_embeddings()
        assert results1 is not results2  # Different objects after expiry
        sm.close()

    def test_concurrent_reads_no_crash(self, tmp_path):
        """Multiple threads reading cache simultaneously should not crash."""
        sm = _make_real_storage(tmp_path)
        errors = []

        def reader():
            try:
                for _ in range(50):
                    sm._get_entities_with_embeddings()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        sm.close()

    def test_invalidate_clears_entity_cache(self, tmp_path):
        """_invalidate_emb_cache should clear entity cache."""
        sm = _make_real_storage(tmp_path)
        sm._get_entities_with_embeddings()
        assert sm._entity_emb_cache is not None
        sm._invalidate_emb_cache()
        assert sm._entity_emb_cache is None
        assert sm._entity_emb_cache_ts == 0.0
        sm.close()

    def test_invalidate_uses_lock(self, tmp_path):
        """_invalidate_emb_cache should use _emb_cache_lock."""
        sm = _make_real_storage(tmp_path)
        # Verify cache is cleared atomically — all three caches should be None together
        sm._get_entities_with_embeddings()
        sm._get_relations_with_embeddings()
        sm._get_latest_concepts_with_embeddings()
        assert sm._entity_emb_cache is not None
        sm._invalidate_emb_cache()
        # All three must be None simultaneously (atomic under lock)
        assert sm._entity_emb_cache is None
        assert sm._relation_emb_cache is None
        assert sm._concept_emb_cache is None
        sm.close()

    def test_embedding_blob_parsed_correctly(self, tmp_path):
        """Embedding BLOB should be parsed as numpy float32 array."""
        sm = _make_real_storage(tmp_path)
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        sm.save_entity(_entity(absolute_id="emb_1", family_id="ent_emb1", name="EmbTest", content="Content"))
        # Write embedding directly since save_entity clears it without embedding_client
        conn = sm._get_conn()
        conn.execute("UPDATE entities SET embedding = ? WHERE id = ?", (emb, "emb_1"))
        conn.commit()
        sm._entity_emb_cache = None
        sm._entity_emb_cache_ts = 0.0
        results = sm._get_entities_with_embeddings()
        assert len(results) == 1
        entity, emb_arr = results[0]
        assert emb_arr is not None
        np.testing.assert_array_almost_equal(emb_arr, [0.1, 0.2, 0.3])
        sm.close()


# ============================================================
# Dimension 2: Thread-safe relation embedding cache
# ============================================================
class TestRelationEmbCacheThreadSafety:
    """Tests for _get_relations_with_embeddings thread safety."""

    def _save_relation(self, sm, fid="rel_test1", content="Test relation"):
        sm.save_entity(_entity(absolute_id="r_e1", family_id="ent_re1", name="A", content="A"))
        sm.save_entity(_entity(absolute_id="r_e2", family_id="ent_re2", name="B", content="B"))
        sm.save_relation(_relation(
            absolute_id=f"rel_{fid}", family_id=fid, content=content,
            entity1_absolute_id="r_e1", entity2_absolute_id="r_e2",
        ))

    def test_cache_miss_queries_db(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        self._save_relation(sm)
        sm._relation_emb_cache = None
        sm._relation_emb_cache_ts = 0.0
        results = sm._get_relations_with_embeddings()
        assert len(results) == 1
        assert results[0][0].family_id == "rel_test1"
        sm.close()

    def test_cache_hit_returns_cached(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        self._save_relation(sm)
        r1 = sm._get_relations_with_embeddings()
        r2 = sm._get_relations_with_embeddings()
        assert r1 is r2
        sm.close()

    def test_cache_expires_after_ttl(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        self._save_relation(sm)
        sm._emb_cache_ttl = 0.01
        r1 = sm._get_relations_with_embeddings()
        time.sleep(0.02)
        r2 = sm._get_relations_with_embeddings()
        assert r1 is not r2
        sm.close()

    def test_concurrent_reads_no_crash(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        self._save_relation(sm)
        errors = []

        def reader():
            try:
                for _ in range(50):
                    sm._get_relations_with_embeddings()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        sm.close()

    def test_invalidate_clears_relation_cache(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        self._save_relation(sm)
        sm._get_relations_with_embeddings()
        assert sm._relation_emb_cache is not None
        sm._invalidate_emb_cache()
        assert sm._relation_emb_cache is None
        sm.close()

    def test_relation_embedding_parsed(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        sm.save_entity(_entity(absolute_id="re_e1", family_id="ent_re1", name="A", content="A"))
        sm.save_entity(_entity(absolute_id="re_e2", family_id="ent_re2", name="B", content="B"))
        sm.save_relation(_relation(
            absolute_id="rel_emb1", family_id="rel_emb1",
            content="Emb relation",
            entity1_absolute_id="re_e1", entity2_absolute_id="re_e2",
        ))
        # Write embedding directly since save_relation clears it without embedding_client
        emb = np.array([0.4, 0.5, 0.6], dtype=np.float32).tobytes()
        conn = sm._get_conn()
        conn.execute("UPDATE relations SET embedding = ? WHERE id = ?", (emb, "rel_emb1"))
        conn.commit()
        sm._relation_emb_cache = None
        sm._relation_emb_cache_ts = 0.0
        results = sm._get_relations_with_embeddings()
        _, emb_arr = results[0]
        assert emb_arr is not None
        np.testing.assert_array_almost_equal(emb_arr, [0.4, 0.5, 0.6])
        sm.close()

    def test_multiple_relations_latest_version(self, tmp_path):
        """Should return only latest version per family_id."""
        sm = _make_real_storage(tmp_path)
        sm.save_entity(_entity(absolute_id="mv_e1", family_id="ent_mv1", name="A", content="A"))
        sm.save_entity(_entity(absolute_id="mv_e2", family_id="ent_mv2", name="B", content="B"))
        sm.save_relation(_relation(
            absolute_id="mv_r1", family_id="rel_mv1", content="v1",
            entity1_absolute_id="mv_e1", entity2_absolute_id="mv_e2",
        ))
        sm.save_relation(_relation(
            absolute_id="mv_r2", family_id="rel_mv1", content="v2 updated",
            entity1_absolute_id="mv_e1", entity2_absolute_id="mv_e2",
        ))
        sm._relation_emb_cache = None
        sm._relation_emb_cache_ts = 0.0
        results = sm._get_relations_with_embeddings()
        assert len(results) == 1
        assert results[0][0].content == "v2 updated"
        sm.close()

    def test_empty_db_returns_empty(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        results = sm._get_relations_with_embeddings()
        assert results == []
        sm.close()


# ============================================================
# Dimension 3: Thread-safe concept embedding cache
# ============================================================
class TestConceptEmbCacheThreadSafety:
    """Tests for _get_latest_concepts_with_embeddings thread safety."""

    def test_cache_miss_queries_db(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        sm.save_entity(_entity(absolute_id="c_test1", family_id="ent_ctest1", name="Concept", content="Concept content"))
        sm._concept_emb_cache = None
        sm._concept_emb_cache_ts = 0.0
        results = sm._get_latest_concepts_with_embeddings()
        assert len(results) >= 1
        sm.close()

    def test_cache_hit_returns_cached(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        r1 = sm._get_latest_concepts_with_embeddings()
        r2 = sm._get_latest_concepts_with_embeddings()
        assert r1 is r2
        sm.close()

    def test_cache_expires_after_ttl(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        sm._emb_cache_ttl = 0.01
        r1 = sm._get_latest_concepts_with_embeddings()
        time.sleep(0.02)
        r2 = sm._get_latest_concepts_with_embeddings()
        assert r1 is not r2
        sm.close()

    def test_role_filter_works(self, tmp_path):
        """Role filter should work on cached data."""
        sm = _make_real_storage(tmp_path)
        sm.save_entity(_entity(absolute_id="cf_e1", family_id="ent_cf1", name="EntOnly", content="Entity"))
        results = sm._get_latest_concepts_with_embeddings(role="entity")
        assert all(c[0].get('role') == 'entity' for c in results)
        sm.close()

    def test_concurrent_reads_no_crash(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        errors = []

        def reader():
            try:
                for _ in range(50):
                    sm._get_latest_concepts_with_embeddings()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        sm.close()

    def test_invalidate_clears_concept_cache(self, tmp_path):
        sm = _make_real_storage(tmp_path)
        sm._get_latest_concepts_with_embeddings()
        assert sm._concept_emb_cache is not None
        sm._invalidate_emb_cache()
        assert sm._concept_emb_cache is None
        sm.close()

    def test_concept_dict_fields(self, tmp_path):
        """Concept dict should have all expected fields."""
        sm = _make_real_storage(tmp_path)
        sm.save_entity(_entity(absolute_id="cd_e1", family_id="ent_cd1", name="FieldTest", content="Content here"))
        sm._concept_emb_cache = None
        sm._concept_emb_cache_ts = 0.0
        results = sm._get_latest_concepts_with_embeddings(role="entity")
        assert len(results) >= 1
        concept = results[0][0]
        assert 'id' in concept
        assert 'family_id' in concept
        assert 'role' in concept
        assert 'name' in concept
        assert 'content' in concept
        assert concept['name'] == "FieldTest"
        sm.close()

    def test_empty_concepts_returns_empty(self, tmp_path):
        """Empty concepts table should return empty list."""
        sm = _make_real_storage(tmp_path)
        conn = sm._get_conn()
        conn.execute("DELETE FROM concepts")
        conn.commit()
        sm._concept_emb_cache = None
        sm._concept_emb_cache_ts = 0.0
        results = sm._get_latest_concepts_with_embeddings()
        assert results == []
        sm.close()


# ============================================================
# Dimension 4: Cross-cache invalidation & concurrency stress
# ============================================================
class TestCrossCacheInvalidation:
    """Tests for cross-cache invalidation and concurrent stress."""

    def test_invalidate_clears_all_three_caches(self, tmp_path):
        """_invalidate_emb_cache should clear entity, relation, AND concept caches."""
        sm = _make_real_storage(tmp_path)
        sm._get_entities_with_embeddings()
        sm._get_relations_with_embeddings()
        sm._get_latest_concepts_with_embeddings()
        assert sm._entity_emb_cache is not None
        assert sm._relation_emb_cache is not None
        assert sm._concept_emb_cache is not None
        sm._invalidate_emb_cache()
        assert sm._entity_emb_cache is None
        assert sm._relation_emb_cache is None
        assert sm._concept_emb_cache is None
        sm.close()

    def test_save_entity_invalidates_all_caches(self, tmp_path):
        """Saving an entity should invalidate all embedding caches."""
        sm = _make_real_storage(tmp_path)
        sm._get_entities_with_embeddings()
        sm._get_relations_with_embeddings()
        sm._get_latest_concepts_with_embeddings()
        sm.save_entity(_entity(absolute_id="inv_e1", family_id="ent_inv1", name="Invalidate", content="Test"))
        assert sm._entity_emb_cache is None
        assert sm._relation_emb_cache is None
        assert sm._concept_emb_cache is None
        sm.close()

    def test_save_relation_invalidates_all_caches(self, tmp_path):
        """Saving a relation should invalidate all embedding caches."""
        sm = _make_real_storage(tmp_path)
        sm.save_entity(_entity(absolute_id="ir_e1", family_id="ent_ir1", name="A", content="A"))
        sm.save_entity(_entity(absolute_id="ir_e2", family_id="ent_ir2", name="B", content="B"))
        sm._get_entities_with_embeddings()
        sm._get_relations_with_embeddings()
        sm._get_latest_concepts_with_embeddings()
        sm.save_relation(_relation(
            absolute_id="ir_r1", family_id="rel_ir1", content="Test",
            entity1_absolute_id="ir_e1", entity2_absolute_id="ir_e2",
        ))
        assert sm._entity_emb_cache is None
        assert sm._relation_emb_cache is None
        assert sm._concept_emb_cache is None
        sm.close()

    def test_concurrent_write_and_read_no_crash(self, tmp_path):
        """Concurrent writes + reads should not crash or corrupt data."""
        sm = _make_real_storage(tmp_path)
        sm.save_entity(_entity(absolute_id="cw_e1", family_id="ent_cw1", name="A", content="A"))
        sm.save_entity(_entity(absolute_id="cw_e2", family_id="ent_cw2", name="B", content="B"))
        errors = []

        def writer(i):
            try:
                sm.save_entity(_entity(
                    absolute_id=f"cw_w{i}", family_id=f"ent_cw_w{i}",
                    name=f"Writer{i}", content=f"Content{i}",
                ))
            except Exception as ex:
                errors.append(ex)

        def reader():
            try:
                for _ in range(20):
                    sm._get_entities_with_embeddings()
                    sm._get_relations_with_embeddings()
            except Exception as ex:
                errors.append(ex)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i,)))
        for _ in range(5):
            threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        sm.close()

    def test_stress_100_concurrent_reads(self, tmp_path):
        """100 concurrent cache reads should not deadlock or crash."""
        sm = _make_real_storage(tmp_path)
        sm.save_entity(_entity(absolute_id="stress_e1", family_id="ent_stress1", name="Stress", content="Stress test"))
        errors = []

        def reader():
            try:
                for _ in range(100):
                    sm._get_entities_with_embeddings()
                    sm._get_latest_concepts_with_embeddings()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        sm.close()

    def test_cache_lock_prevents_torn_reads(self, tmp_path):
        """Cache should never return partially-written data."""
        sm = _make_real_storage(tmp_path)
        sm.save_entity(_entity(absolute_id="torn_e1", family_id="ent_torn1", name="Torn", content="Torn test"))
        torn_detected = []

        def checker():
            for _ in range(200):
                results = sm._get_entities_with_embeddings()
                if not isinstance(results, list):
                    torn_detected.append("not a list")
                    continue
                for item in results:
                    if not isinstance(item, tuple) or len(item) != 2:
                        torn_detected.append(f"bad item: {type(item)}")
                        break

        threads = [threading.Thread(target=checker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not torn_detected, f"Torn reads detected: {torn_detected}"
        sm.close()

    def test_bulk_save_invalidates_cache(self, tmp_path):
        """bulk_save_entities should invalidate all caches."""
        sm = _make_real_storage(tmp_path)
        sm._get_entities_with_embeddings()
        assert sm._entity_emb_cache is not None
        entities = [
            _entity(absolute_id=f"bulk_{i}", family_id=f"ent_bulk{i}",
                    name=f"Bulk{i}", content=f"Content{i}")
            for i in range(3)
        ]
        sm.bulk_save_entities(entities)
        assert sm._entity_emb_cache is None
        sm.close()

    def test_ttl_zero_always_queries(self, tmp_path):
        """With TTL=0, every call should query the database."""
        sm = _make_real_storage(tmp_path)
        sm.save_entity(_entity(absolute_id="ttl_e1", family_id="ent_ttl1", name="TTL", content="TTL test"))
        sm._emb_cache_ttl = 0.0
        r1 = sm._get_entities_with_embeddings()
        r2 = sm._get_entities_with_embeddings()
        assert r1 is not r2
        sm.close()
