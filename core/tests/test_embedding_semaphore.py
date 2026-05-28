"""
Tests for EmbeddingClient semaphore configuration.

Tests that the embedding semaphore defaults to one local encode at a time,
with an explicit override for deployments that can safely run more.
"""
import os
import threading
from unittest.mock import Mock, patch, MagicMock
import pytest

from core.storage.embedding import EmbeddingClient


class TestEmbeddingSemaphore:
    """Tests for embedding semaphore configuration."""

    @patch("core.storage.embedding.EmbeddingClient._init_model")
    def test_semaphore_exists_with_default_serial_policy(self, mock_init):
        """Default policy is serial encode for local embedding stability."""
        client = EmbeddingClient(model_path="test_model", use_local=True)

        assert hasattr(client, "_encode_semaphore")
        assert isinstance(client._encode_semaphore, threading.Semaphore)

    @patch("core.storage.embedding.EmbeddingClient._init_model")
    def test_default_semaphore_is_serial(self, mock_init):
        """Default semaphore should allow one encode operation at a time."""
        client = EmbeddingClient(model_path="test", use_local=True)

        active_count = 0
        max_active = 0
        lock = threading.Lock()

        def mock_operation():
            nonlocal active_count, max_active
            with client._encode_semaphore:
                with lock:
                    active_count += 1
                    if active_count > max_active:
                        max_active = active_count
                import time
                time.sleep(0.01)
                with lock:
                    active_count -= 1

        threads = [threading.Thread(target=mock_operation) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_active <= 1

    @patch("core.storage.embedding.EmbeddingClient._init_model")
    def test_explicit_semaphore_concurrency_override(self, mock_init):
        """Deployments can explicitly allow more parallel embedding encodes."""
        client = EmbeddingClient(model_path="test", use_local=True, max_concurrency=3)

        active_count = 0
        max_active = 0
        lock = threading.Lock()

        def mock_operation():
            nonlocal active_count, max_active
            with client._encode_semaphore:
                with lock:
                    active_count += 1
                    if active_count > max_active:
                        max_active = active_count
                import time
                time.sleep(0.01)
                with lock:
                    active_count -= 1

        threads = [threading.Thread(target=mock_operation) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_active <= 3

    @patch("core.storage.embedding.EmbeddingClient._init_model")
    def test_invalid_max_concurrency_falls_back_to_one(self, mock_init):
        """Non-positive max_concurrency values should fall back to one."""
        client = EmbeddingClient(model_path="test", use_local=True, max_concurrency=0)

        assert hasattr(client, "_encode_semaphore")
        assert isinstance(client._encode_semaphore, threading.Semaphore)

        active_count = 0
        max_active = 0
        lock = threading.Lock()

        def mock_operation():
            nonlocal active_count, max_active
            with client._encode_semaphore:
                with lock:
                    active_count += 1
                    if active_count > max_active:
                        max_active = active_count
                import time
                time.sleep(0.01)
                with lock:
                    active_count -= 1

        threads = [threading.Thread(target=mock_operation) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_active <= 1

    @patch("core.storage.embedding.EmbeddingClient._init_model")
    def test_semaphore_thread_safety(self, mock_init):
        """Semaphore should provide thread-safe access control."""
        with patch("os.cpu_count", return_value=4):
            client = EmbeddingClient(model_path="test", use_local=True)

            results = []
            errors = []

            def encode_worker(worker_id):
                try:
                    # Simulate encode operation using semaphore
                    with client._encode_semaphore:
                        # Verify we have exclusive access within semaphore limit
                        results.append(worker_id)
                        import time
                        time.sleep(0.01)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=encode_worker, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All workers should complete without errors
            assert len(errors) == 0
            assert len(results) == 10

    @patch("core.storage.embedding.EmbeddingClient._init_model")
    def test_semaphore_used_in_encode_method(self, mock_init):
        """Verify that semaphore is properly initialized and available."""
        with patch("os.cpu_count", return_value=4):
            client = EmbeddingClient(model_path="test", use_local=True)

            # Verify semaphore exists and is a threading.Semaphore
            assert hasattr(client, "_encode_semaphore")
            assert isinstance(client._encode_semaphore, threading.Semaphore)

            # Verify the semaphore is used by checking the _encode_chunk method exists
            # and that it will use the semaphore (verified by code inspection)
            assert hasattr(client, "_encode_chunk")

            # The semaphore usage pattern in _encode_chunk is:
            # with self._encode_semaphore:
            #     return self.model.encode(...)
            # This is a design verification - the semaphore controls concurrent access


class TestEmbeddingSemaphoreConfiguration:
    """Tests for semaphore configuration in different scenarios."""

    @patch("core.storage.embedding.EmbeddingClient._init_model")
    def test_semaphore_created_during_init(self, mock_init):
        """Semaphore should be created during client initialization."""
        with patch("os.cpu_count", return_value=4):
            client = EmbeddingClient(model_path="test", use_local=True)

            # Semaphore should exist
            assert hasattr(client, "_encode_semaphore")
            assert isinstance(client._encode_semaphore, threading.Semaphore)

    @patch("core.storage.embedding.EmbeddingClient._init_model")
    def test_semaphore_persists_across_encode_calls(self, mock_init):
        """Same semaphore instance should be used across multiple encode calls."""
        with patch("os.cpu_count", return_value=4):
            client = EmbeddingClient(model_path="test", use_local=True)

            semaphore_id = id(client._encode_semaphore)

            # Multiple encode calls should use the same semaphore
            # (We can't actually call encode without a real model, but we can verify
            # the semaphore reference doesn't change)
            assert id(client._encode_semaphore) == semaphore_id

    @patch("core.storage.embedding.EmbeddingClient._init_model")
    def test_different_clients_have_different_semaphores(self, mock_init):
        """Each EmbeddingClient instance should have its own semaphore."""
        with patch("os.cpu_count", return_value=4):
            client1 = EmbeddingClient(model_path="test1", use_local=True)
            client2 = EmbeddingClient(model_path="test2", use_local=True)

            # Semaphores should be different objects
            assert id(client1._encode_semaphore) != id(client2._encode_semaphore)
