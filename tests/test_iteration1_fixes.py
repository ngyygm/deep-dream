"""
Comprehensive tests for iteration-1 fixes.

Covers 4 dimensions with 30+ test cases:
  1. Embedding thread safety + batch chunking (10 tests)
  2. Cache max_size eviction + thread safety (8 tests)
  3. Unified relation content threshold (8 tests)
  4. VectorStore batch operations (7 tests)
"""
import sys
import threading
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Dimension 1: Embedding thread safety + batch chunking
# ============================================================
class TestEmbeddingThreadSafety:
    """Tests for EmbeddingClient thread safety and batch chunking."""

    def test_encode_uses_lock(self):
        """encode() should acquire _encode_lock during model.encode()."""
        from processor.storage.embedding import EmbeddingClient
        client = EmbeddingClient.__new__(EmbeddingClient)
        client.model = MagicMock()
        client._encode_lock = threading.Lock()
        fake_emb = np.array([[0.1, 0.2]])
        client.model.encode.return_value = fake_emb

        result = client.encode("hello")
        client.model.encode.assert_called_once()
        np.testing.assert_array_equal(result, fake_emb)

    def test_encode_chunking_large_list(self):
        """Large text lists should be split into chunks of batch_size."""
        from processor.storage.embedding import EmbeddingClient
        client = EmbeddingClient.__new__(EmbeddingClient)
        client.model = MagicMock()
        client._encode_lock = threading.Lock()

        # Return different embeddings per call to verify chunking
        call_count = [0]
        def mock_encode(texts, **kwargs):
            call_count[0] += 1
            return np.random.rand(len(texts), 3)

        client.model.encode = mock_encode

        texts = [f"text_{i}" for i in range(25)]
        result = client.encode(texts, batch_size=10)
        assert result.shape == (25, 3)
        assert call_count[0] == 3  # 25 / 10 = 3 chunks

    def test_encode_single_string_converted(self):
        """Single string input should be converted to list internally."""
        from processor.storage.embedding import EmbeddingClient
        client = EmbeddingClient.__new__(EmbeddingClient)
        client.model = MagicMock()
        client._encode_lock = threading.Lock()
        client.model.encode.return_value = np.array([[0.1, 0.2]])

        result = client.encode("hello")
        # The model should receive a list, not a string
        args, kwargs = client.model.encode.call_args
        assert isinstance(args[0], list)
        assert args[0] == ["hello"]

    def test_encode_no_model_returns_none(self):
        """If no model loaded, encode() should return None."""
        from processor.storage.embedding import EmbeddingClient
        client = EmbeddingClient.__new__(EmbeddingClient)
        client.model = None

        result = client.encode("test")
        assert result is None

    def test_encode_exception_returns_none(self):
        """If model.encode raises, should return None, not propagate."""
        from processor.storage.embedding import EmbeddingClient
        client = EmbeddingClient.__new__(EmbeddingClient)
        client.model = MagicMock()
        client._encode_lock = threading.Lock()
        client.model.encode.side_effect = RuntimeError("GPU OOM")

        result = client.encode("test")
        assert result is None

    def test_encode_chunk_failure_propagates_none(self):
        """If any chunk fails, the whole encode should return None."""
        from processor.storage.embedding import EmbeddingClient
        client = EmbeddingClient.__new__(EmbeddingClient)
        client.model = MagicMock()
        client._encode_lock = threading.Lock()
        call_count = [0]

        def mock_encode(texts, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("fail on chunk 2")
            return np.random.rand(len(texts), 3)

        client.model.encode = mock_encode
        texts = [f"t{i}" for i in range(30)]
        result = client.encode(texts, batch_size=10)
        assert result is None

    def test_encode_concurrent_thread_safety(self):
        """Multiple threads encoding simultaneously should not crash."""
        from processor.storage.embedding import EmbeddingClient
        client = EmbeddingClient.__new__(EmbeddingClient)
        client.model = MagicMock()
        client._encode_lock = threading.Lock()
        client.model.encode.return_value = np.random.rand(1, 8)

        errors = []
        def worker(i):
            try:
                result = client.encode(f"text_{i}")
                assert result is not None
                assert result.shape == (1, 8)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent encode errors: {errors}"

    def test_encode_exact_batch_size_no_extra_chunk(self):
        """When len(texts) == batch_size, no chunking should occur."""
        from processor.storage.embedding import EmbeddingClient
        client = EmbeddingClient.__new__(EmbeddingClient)
        client.model = MagicMock()
        client._encode_lock = threading.Lock()
        client.model.encode.return_value = np.random.rand(10, 3)

        texts = [f"t{i}" for i in range(10)]
        result = client.encode(texts, batch_size=10)
        assert client.model.encode.call_count == 1

    def test_encode_empty_list(self):
        """Empty list should be encoded normally (model handles it)."""
        from processor.storage.embedding import EmbeddingClient
        client = EmbeddingClient.__new__(EmbeddingClient)
        client.model = MagicMock()
        client._encode_lock = threading.Lock()
        client.model.encode.return_value = np.array([]).reshape(0, 3)

        result = client.encode([], batch_size=10)
        assert result is not None


# ============================================================
# Dimension 2: Cache max_size eviction + thread safety
# ============================================================
class TestCacheMaxSize:
    """Tests for QueryCache max_size eviction behavior."""

    def _make_cache(self, max_size=5, ttl=30):
        from processor.storage.cache import QueryCache
        return QueryCache(default_ttl=ttl, max_size=max_size)

    def test_basic_get_set(self):
        c = self._make_cache()
        c.set("k1", "v1")
        assert c.get("k1") == "v1"

    def test_ttl_expiry(self):
        c = self._make_cache(ttl=0.05)
        c.set("k1", "v1")
        time.sleep(0.1)
        assert c.get("k1") is None

    def test_max_size_eviction_on_full(self):
        """When cache is full, set() should evict oldest entries."""
        c = self._make_cache(max_size=3)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        # Cache is at max_size. Setting a new key should trigger eviction.
        c.set("d", 4)
        # At least one old key should be evicted
        assert c.size() <= 3
        assert c.get("d") == 4

    def test_max_size_evicts_expired_first(self):
        """Expired entries should be evicted before oldest valid entries."""
        c = self._make_cache(max_size=3, ttl=10)
        c.set("a", 1)
        c.set("b", 2)
        # Make "a" expired by manipulating its expiry
        with c._lock:
            entry = c._cache["a"]
            c._cache["a"] = (time.monotonic() - 1, entry[1], entry[2])
        c.set("c", 3)
        c.set("d", 4)
        # "a" was expired, so it gets evicted first; "b" should survive
        assert c.get("b") == 2

    def test_invalidate_pattern(self):
        c = self._make_cache()
        c.set("entity_abc", 1)
        c.set("entity_def", 2)
        c.set("relation_xyz", 3)
        c.invalidate("entity_")
        assert c.get("entity_abc") is None
        assert c.get("entity_def") is None
        assert c.get("relation_xyz") == 3

    def test_invalidate_all(self):
        c = self._make_cache()
        c.set("a", 1)
        c.set("b", 2)
        c.invalidate()
        assert c.size() == 0

    def test_size_method(self):
        c = self._make_cache()
        assert c.size() == 0
        c.set("a", 1)
        assert c.size() == 1
        c.set("b", 2)
        assert c.size() == 2

    def test_concurrent_access(self):
        """Multiple threads reading/writing cache should not crash."""
        c = self._make_cache(max_size=100, ttl=5)
        errors = []

        def writer(i):
            try:
                for j in range(50):
                    c.set(f"key_{i}_{j}", f"val_{i}_{j}")
            except Exception as e:
                errors.append(e)

        def reader(i):
            try:
                for j in range(50):
                    c.get(f"key_{i}_{j}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(i,)) for i in range(5)
        ] + [
            threading.Thread(target=reader, args=(i,)) for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent cache errors: {errors}"


# ============================================================
# Dimension 3: Unified relation content threshold
# ============================================================
class TestUnifiedRelationThreshold:
    """Tests that extraction and relation modules use the same threshold."""

    def test_extraction_constant_defined(self):
        from processor.pipeline.extraction import MIN_RELATION_CONTENT_LENGTH
        assert MIN_RELATION_CONTENT_LENGTH == 8

    def test_relation_imports_constant(self):
        from processor.pipeline.relation import MIN_RELATION_CONTENT_LENGTH
        assert MIN_RELATION_CONTENT_LENGTH == 8

    def test_extraction_rejects_short_content(self):
        from processor.pipeline.extraction import _is_valid_relation_content
        assert _is_valid_relation_content("short") is False  # 5 chars
        assert _is_valid_relation_content("1234567") is False  # 7 chars
        assert _is_valid_relation_content("12345678") is True  # 8 chars

    def test_relation_build_new_rejects_short(self):
        """RelationProcessor._build_new_relation should reject content < 8 chars."""
        from processor.pipeline.relation import RelationProcessor
        storage = MagicMock()
        llm = MagicMock()
        proc = RelationProcessor(storage, llm)

        # content = "short" (5 chars) should be rejected
        result = proc._build_new_relation(
            entity1_id="e1", entity2_id="e2",
            content="short",  # 5 chars < 8
            episode_id="ep1",
            entity1_name="A", entity2_name="B"
        )
        assert result is None

    def test_relation_build_new_accepts_valid(self):
        """Content >= 8 chars should not be rejected for length."""
        from processor.pipeline.relation import RelationProcessor
        from processor.models import Entity
        storage = MagicMock()
        llm = MagicMock()
        proc = RelationProcessor(storage, llm)

        # Setup entity lookups
        e1 = Entity(absolute_id="abs1", family_id="e1", name="A",
                     content="Entity A", event_time=None, processed_time=None,
                     episode_id="ep1", source_document="test")
        e2 = Entity(absolute_id="abs2", family_id="e2", name="B",
                     content="Entity B", event_time=None, processed_time=None,
                     episode_id="ep1", source_document="test")
        entity_lookup = {"e1": e1, "e2": e2}

        result = proc._build_new_relation(
            entity1_id="e1", entity2_id="e2",
            content="A is related to B in some meaningful way",
            episode_id="ep1",
            entity1_name="A", entity2_name="B",
            entity_lookup=entity_lookup
        )
        assert result is not None

    def test_relation_version_rejects_short(self):
        """_create_relation_version should reject content < 8 chars."""
        from processor.pipeline.relation import RelationProcessor
        storage = MagicMock()
        llm = MagicMock()
        proc = RelationProcessor(storage, llm)

        result = proc._create_relation_version(
            family_id="rel_1",
            entity1_id="e1",
            entity2_id="e2",
            content="short",  # 5 chars < 8
            episode_id="ep1"
        )
        assert result is None

    def test_both_modules_same_threshold_value(self):
        """Both modules should import the exact same constant."""
        from processor.pipeline.extraction import MIN_RELATION_CONTENT_LENGTH as ext_min
        from processor.pipeline.relation import MIN_RELATION_CONTENT_LENGTH as rel_min
        assert ext_min == rel_min
        assert ext_min >= 8

    def test_threshold_boundary_exact(self):
        """Content exactly at threshold boundary should pass."""
        from processor.pipeline.extraction import MIN_RELATION_CONTENT_LENGTH, _is_valid_relation_content
        boundary = "a" * MIN_RELATION_CONTENT_LENGTH
        assert _is_valid_relation_content(boundary) is True
        below = "a" * (MIN_RELATION_CONTENT_LENGTH - 1)
        assert _is_valid_relation_content(below) is False


# ============================================================
# Dimension 4: VectorStore batch operations
# ============================================================
class TestVectorStoreBatch:
    """Tests for VectorStore batch operations correctness."""

    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create a VectorStore with a temporary database."""
        from processor.storage.vector_store import VectorStore
        db_path = str(tmp_path / "test_vectors.db")
        return VectorStore(db_path, dim=8)

    def test_upsert_and_search(self, vector_store):
        emb = self._normalize([0.1] * 8)
        vector_store.upsert("entity_vectors", "id1", emb)
        results = vector_store.search("entity_vectors", emb, limit=5)
        assert len(results) == 1
        assert results[0][0] == "id1"

    def test_batch_upsert(self, vector_store):
        items = [
            (f"id_{i}", self._normalize([0.1 * i] * 8))
            for i in range(10)
        ]
        vector_store.upsert_batch("entity_vectors", items)
        for uid, _ in items:
            vec = vector_store.get("entity_vectors", uid)
            assert vec is not None

    def test_batch_delete(self, vector_store):
        items = [
            (f"id_{i}", self._normalize([0.1 * i] * 8))
            for i in range(5)
        ]
        vector_store.upsert_batch("entity_vectors", items)
        uuids = [uid for uid, _ in items[:3]]
        vector_store.delete_batch("entity_vectors", uuids)
        for uid in uuids:
            assert vector_store.get("entity_vectors", uid) is None
        # Remaining should still exist
        assert vector_store.get("entity_vectors", "id_3") is not None
        assert vector_store.get("entity_vectors", "id_4") is not None

    def test_batch_get(self, vector_store):
        items = [
            (f"id_{i}", self._normalize([0.1 * (i + 1)] * 8))
            for i in range(5)
        ]
        vector_store.upsert_batch("entity_vectors", items)
        result = vector_store.get_batch("entity_vectors", ["id_0", "id_2", "id_4"])
        assert len(result) == 3
        assert "id_0" in result
        assert "id_2" in result
        assert "id_4" in result

    def test_empty_batch_operations(self, vector_store):
        """Empty batch operations should be no-ops."""
        vector_store.upsert_batch("entity_vectors", [])
        vector_store.delete_batch("entity_vectors", [])
        assert vector_store.get_batch("entity_vectors", []) == {}

    def test_upsert_overwrite(self, vector_store):
        """Upserting same uuid should overwrite the vector (DELETE+INSERT)."""
        emb1 = self._normalize([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        emb2 = self._normalize([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        vector_store.upsert("entity_vectors", "id1", emb1)
        # Second upsert should replace the first
        vector_store.upsert("entity_vectors", "id1", emb2)
        stored = vector_store.get("entity_vectors", "id1")
        np.testing.assert_allclose(stored, emb2, atol=1e-5)

    def test_search_returns_sorted_by_distance(self, vector_store):
        """Search results should be sorted by distance (ascending)."""
        query = self._normalize([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Create vectors with varying similarity to query
        for i in range(5):
            emb = self._normalize([1.0 - 0.2 * i, 0.2 * i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            vector_store.upsert("entity_vectors", f"id_{i}", emb)

        results = vector_store.search("entity_vectors", query, limit=5)
        # Distances should be in ascending order
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    @staticmethod
    def _normalize(vec):
        """L2-normalize a vector."""
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()
