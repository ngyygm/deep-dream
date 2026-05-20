"""
End-to-end tests for SQLite graph storage through the Flask API.

Tests the storage compatibility adapters plus the v1 concept/document API using
the actual Flask test client with SQLite backend (no real LLM calls).
Run with: pytest core/tests/test_e2e_sqlite.py -v -s --tb=short
"""
import json
import time
import uuid

import pytest

from core.models import Entity, Relation, Episode
from core.storage.sqlite.manager import SQLiteGraphStorageManager

TEST_GRAPH_ID = "test_graph"


# ── Storage-Level E2E Tests ──────────────────────────────────────────────


class TestStorageE2E:
    """Test full CRUD lifecycle through the storage manager directly."""

    def test_entity_lifecycle(self, sqlite_storage):
        """Create → read → update → delete entity."""
        mgr = sqlite_storage
        now_iso = "2025-01-15T10:00:00+00:00"

        # Create
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        e1 = Entity(
            absolute_id=f"e2e_ent_{uuid.uuid4().hex[:8]}",
            family_id="e2e_family_1",
            name="Alice",
            content="A software engineer who works on Python",
            event_time=now,
            processed_time=now,
            episode_id="e2e_ep_1",
            source_document="e2e_test.txt",
        )
        mgr.save_entity(e1)

        # Read
        fetched = mgr.get_entity_by_family_id("e2e_family_1")
        assert fetched is not None
        assert fetched.name == "Alice"

        # Update (new version)
        e2 = Entity(
            absolute_id=f"e2e_ent_{uuid.uuid4().hex[:8]}",
            family_id="e2e_family_1",
            name="Alice Smith",
            content="A senior software engineer who works on Python",
            event_time=now,
            processed_time=now,
            episode_id="e2e_ep_2",
            source_document="e2e_test_v2.txt",
        )
        mgr.save_entity(e2)

        # Verify latest version
        fetched_v2 = mgr.get_entity_by_family_id("e2e_family_1")
        assert fetched_v2.name == "Alice Smith"

        # Verify old version is invalidated
        entities = mgr.get_entity_versions("e2e_family_1")
        assert len(entities) >= 2

    def test_relation_lifecycle(self, sqlite_storage):
        """Create → read → traverse relation."""
        mgr = sqlite_storage
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        # Create two entities
        e1 = Entity(
            absolute_id=f"e2e_e1_{uuid.uuid4().hex[:8]}",
            family_id="e2e_r_family_1",
            name="Bob",
            content="A data scientist",
            event_time=now, processed_time=now,
            episode_id="e2e_ep", source_document="e2e.txt",
        )
        e2 = Entity(
            absolute_id=f"e2e_e2_{uuid.uuid4().hex[:8]}",
            family_id="e2e_r_family_2",
            name="Carol",
            content="A product manager",
            event_time=now, processed_time=now,
            episode_id="e2e_ep", source_document="e2e.txt",
        )
        mgr.bulk_save_entities([e1, e2])

        # Create relation
        sorted_ids = sorted([e1.absolute_id, e2.absolute_id])
        rel = Relation(
            absolute_id=f"e2e_rel_{uuid.uuid4().hex[:8]}",
            family_id="e2e_rel_fam_1",
            entity1_absolute_id=sorted_ids[0],
            entity2_absolute_id=sorted_ids[1],
            content="Bob works with Carol on ML projects",
            event_time=now, processed_time=now,
            episode_id="e2e_ep", source_document="e2e.txt",
        )
        mgr.save_relation(rel)

        # Read relations for entity
        rels = mgr.get_entity_relations(e1.absolute_id)
        assert len(rels) >= 1
        assert any("Carol" in r.content or "Bob" in r.content for r in rels)

    def test_episode_lifecycle(self, sqlite_storage):
        """Create and retrieve episode."""
        mgr = sqlite_storage
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        ep = Episode(
            absolute_id=f"e2e_ep_{uuid.uuid4().hex[:12]}",
            content="Alice met Bob at the conference. They discussed machine learning.",
            event_time=now,
            source_document="e2e_conversation.txt",
            processed_time=now,
            episode_type="conversation",
        )
        mgr.save_episode(ep)

        # Retrieve by ID (returns dict)
        fetched = mgr.get_episode(ep.absolute_id)
        assert fetched is not None
        assert fetched["uuid"] == ep.absolute_id

    def test_bm25_search(self, sqlite_storage):
        """Verify BM25 full-text search works."""
        mgr = sqlite_storage
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        entities = [
            Entity(
                absolute_id=f"bm25_e_{i}_{uuid.uuid4().hex[:6]}",
                family_id=f"bm25_fam_{i}",
                name=f"Machine Learning Algorithm {i}",
                content=f"This algorithm uses neural networks for classification task {i}",
                event_time=now, processed_time=now,
                episode_id="bm25_ep", source_document="ml.txt",
            )
            for i in range(20)
        ]
        mgr.bulk_save_entities(entities)

        # Search for "neural networks"
        results = mgr.search_entities_by_bm25("neural networks", limit=10)
        assert len(results) > 0, "BM25 search returned no results"
        assert any("neural" in (e.content or "").lower() for e in results)

    def test_multi_graph_isolation(self, tmp_path):
        """Verify data isolation between different graph_ids."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        mgr_a = SQLiteGraphStorageManager(
            storage_path=str(tmp_path / "graph_a"),
            vector_dim=1024, graph_id="graph_a",
        )
        mgr_b = SQLiteGraphStorageManager(
            storage_path=str(tmp_path / "graph_b"),
            vector_dim=1024, graph_id="graph_b",
        )

        try:
            # Write to graph A
            e = Entity(
                absolute_id=f"iso_e_{uuid.uuid4().hex[:8]}",
                family_id="iso_family",
                name="Graph A Entity",
                content="Only visible in graph A",
                event_time=now, processed_time=now,
                episode_id="iso_ep", source_document="iso.txt",
            )
            mgr_a.save_entity(e)

            # Verify visible in A
            fetched_a = mgr_a.get_entity_by_family_id("iso_family")
            assert fetched_a is not None
            assert fetched_a.name == "Graph A Entity"

            # Verify NOT visible in B
            fetched_b = mgr_b.get_entity_by_family_id("iso_family")
            assert fetched_b is None

            # Count isolation
            assert mgr_a.count_unique_entities() >= 1
            assert mgr_b.count_unique_entities() == 0
        finally:
            mgr_a.close()
            mgr_b.close()

    def test_stats_accuracy(self, sqlite_storage):
        """Verify get_stats returns accurate counts."""
        mgr = sqlite_storage
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        # Create entities
        entities = [
            Entity(
                absolute_id=f"stats_e_{i}_{uuid.uuid4().hex[:6]}",
                family_id=f"stats_fam_{i}",
                name=f"Stats Entity {i}",
                content=f"Content {i}",
                event_time=now, processed_time=now,
                episode_id="stats_ep", source_document="stats.txt",
            )
            for i in range(10)
        ]
        mgr.bulk_save_entities(entities)

        stats = mgr.get_stats()
        assert stats["entities"] >= 10

    def test_clear_graph_data(self, sqlite_storage):
        """Verify clear_graph_data removes all data but keeps schema."""
        mgr = sqlite_storage
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        e = Entity(
            absolute_id=f"clear_e_{uuid.uuid4().hex[:8]}",
            family_id="clear_family",
            name="To Be Cleared",
            content="Will be deleted",
            event_time=now, processed_time=now,
            episode_id="clear_ep", source_document="clear.txt",
        )
        mgr.save_entity(e)
        assert mgr.count_unique_entities() >= 1

        mgr.clear_graph_data()
        assert mgr.count_unique_entities() == 0

        # Verify schema still works (can create new entities)
        e2 = Entity(
            absolute_id=f"clear_e2_{uuid.uuid4().hex[:8]}",
            family_id="clear_family_2",
            name="After Clear",
            content="Created after clear",
            event_time=now, processed_time=now,
            episode_id="clear_ep2", source_document="clear2.txt",
        )
        mgr.save_entity(e2)
        assert mgr.get_entity_by_family_id("clear_family_2") is not None


# ── API-Level E2E Tests ──────────────────────────────────────────────────


class TestAPIE2E:
    """Test full API flows through the Flask test client."""

    def test_health_endpoint(self, client):
        """Verify health endpoint detects SQLite backend."""
        response = client.get(f"/api/v1/health?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 200
        data = response.get_json()["data"]
        assert data["storage_backend"] == "sqlite"

    def test_concept_stats_via_api(self, client):
        """Read concept graph stats through API."""
        response = client.get(f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 200

    def test_find_endpoint(self, client):
        """Verify find endpoint accepts queries."""
        response = client.post(
            "/api/v1/find",
            json={"graph_id": TEST_GRAPH_ID, "query": "test query"},
        )
        # May return empty results, but should not error
        assert response.status_code == 200

    def test_graph_isolation_via_api(self, client):
        """Verify graph_id isolation through API."""
        # Query non-existent graph
        response = client.get("/api/v1/health?graph_id=nonexistent_graph123")
        assert response.status_code == 200
        data = response.get_json()["data"]
        assert data["graph_id"] == "nonexistent_graph123"

    def test_system_endpoints(self, client):
        """Verify system monitoring endpoints work."""
        response = client.get("/api/v1/system/overview")
        assert response.status_code in (200, 503)  # 503 if monitor not enabled

        response = client.get(f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 200

    def test_routes_index(self, client):
        """Verify route index endpoint."""
        response = client.get("/api/v1/routes")
        assert response.status_code == 200
        data = response.get_json()["data"]
        assert data["count"] > 0
        assert any("/api/v1/health" in r["path"] for r in data["routes"])
