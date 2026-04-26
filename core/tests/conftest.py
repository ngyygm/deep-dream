"""
Test configuration and fixtures for Deep-Dream tests.

This module provides common fixtures and utilities for testing the Deep-Dream
knowledge graph system, including Flask test client, processor, and storage
fixtures.
"""
import os
import sys
import pytest
from pathlib import Path
from typing import Generator

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

from flask import Flask
from core.server.api import create_app
from core.server.registry import GraphRegistry
from core.server.config import load_config
from core.server.monitor import SystemMonitor, LOG_MODE_DETAIL
from core.storage.neo4j import Neo4jStorageManager


# Test configuration
TEST_GRAPH_ID = "test_graph"
TEST_CONFIG_PATH = _project_root / "service_config.json"


@pytest.fixture(scope="session")
def test_config():
    """Load test configuration."""
    if TEST_CONFIG_PATH.exists():
        return load_config(str(TEST_CONFIG_PATH))
    else:
        # Minimal fallback config for testing
        return {
            "storage_path": "./graph/test",
            "host": "127.0.0.1",
            "port": 16200,
            "llm": {
                "api_key": "test",
                "model": "test-model",
                "base_url": "http://localhost:11434/v1",
            },
            "embedding": {
                "model_path": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu",
            },
        }


@pytest.fixture(scope="session")
def system_monitor(test_config):
    """Create a SystemMonitor for testing."""
    monitor = SystemMonitor(config=test_config, mode=LOG_MODE_DETAIL)
    yield monitor
    # Cleanup happens automatically


@pytest.fixture(scope="function")
def registry(test_config, system_monitor):
    """Create a GraphRegistry for testing."""
    storage_path = test_config.get("storage_path", "./graph/test")
    registry = GraphRegistry(
        storage_path,
        test_config,
        system_monitor=system_monitor,
    )
    yield registry
    # Cleanup test graph
    try:
        if TEST_GRAPH_ID in registry.list_graphs():
            processor = registry.get_processor(TEST_GRAPH_ID)
            if hasattr(processor.storage, 'close'):
                processor.storage.close()
    except Exception as e:
        print(f"Cleanup error: {e}")


@pytest.fixture(scope="function")
def test_app(registry):
    """Create a Flask test app."""
    config = {
        "host": "127.0.0.1",
        "port": 16200,
        "rate_limit_per_minute": 0,  # Disable rate limiting for tests
    }
    app = create_app(registry, config=config, system_monitor=registry._system_monitor)
    app.config['TESTING'] = True
    yield app


@pytest.fixture(scope="function")
def client(test_app):
    """Create a Flask test client."""
    return test_app.test_client()


@pytest.fixture(scope="function")
def processor(registry):
    """Get a test processor with Neo4j storage."""
    try:
        proc = registry.get_processor(TEST_GRAPH_ID)
        yield proc
    except Exception as e:
        pytest.skip(f"Could not create processor (Neo4j unavailable?): {e}")


@pytest.fixture(scope="function")
def storage(processor):
    """Get test storage (Neo4j)."""
    yield processor.storage


class TestHelpers:
    """Helper methods for tests."""

    @staticmethod
    def create_test_entity(storage, name: str, content: str = None, family_id: str = None):
        """Create a test entity in storage."""
        from core.models import Entity
        from datetime import datetime, timezone
        import uuid

        if content is None:
            content = f"Test entity for {name}"

        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")
        absolute_id = f"entity_{ts}_{uuid.uuid4().hex[:8]}"
        if family_id is None:
            family_id = f"ent_{uuid.uuid4().hex[:12]}"

        entity = Entity(
            absolute_id=absolute_id,
            family_id=family_id,
            name=name,
            content=content,
            event_time=now,
            processed_time=now,
            episode_id="test_episode",
            source_document="test_source.txt",
            content_format="plain",
        )
        storage.save_entity(entity)
        return entity

    @staticmethod
    def create_test_relation(storage, entity1_id: str, entity2_id: str, content: str = None, family_id: str = None):
        """Create a test relation in storage."""
        from core.models import Relation
        from datetime import datetime, timezone
        import uuid

        if content is None:
            content = f"Relation between {entity1_id} and {entity2_id}"

        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")
        absolute_id = f"relation_{ts}_{uuid.uuid4().hex[:8]}"
        if family_id is None:
            family_id = f"rel_{uuid.uuid4().hex[:12]}"

        # Sort entity IDs to ensure undirected relation consistency
        sorted_ids = sorted([entity1_id, entity2_id])

        relation = Relation(
            absolute_id=absolute_id,
            family_id=family_id,
            entity1_absolute_id=sorted_ids[0],
            entity2_absolute_id=sorted_ids[1],
            content=content,
            event_time=now,
            processed_time=now,
            episode_id="test_episode",
            source_document="test_source.txt",
            content_format="plain",
        )
        storage.save_relation(relation)
        return relation

    @staticmethod
    def create_test_episode(storage, content: str, source_document: str = None):
        """Create a test episode in storage."""
        from core.models import Episode
        from datetime import datetime, timezone
        import uuid

        now = datetime.now(timezone.utc)
        absolute_id = f"episode_{uuid.uuid4().hex}"

        if source_document is None:
            source_document = "test_source.txt"

        episode = Episode(
            absolute_id=absolute_id,
            content=content,
            event_time=now,
            source_document=source_document,
            processed_time=now,
            episode_type="fact",
        )
        storage.save_episode(episode)
        return episode


@pytest.fixture(scope="function")
def test_helpers():
    """Provide test helper methods."""
    return TestHelpers
