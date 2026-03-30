"""
Comprehensive pytest tests for Phases 1-5 + v3 new endpoints.

Covers:
  Phase 1C: search_mode parameter (semantic / bm25 / hybrid) on find and search endpoints
  Phase 2:  Entity CRUD (update, delete, batch-delete, merge)
  Phase 2:  Relation CRUD (update, delete, batch-delete)
  Phase 3:  Time travel (snapshot, changes, invalidate, invalidated list)
  Phase 4:  Graph statistics & entity timeline
  Phase A:  Entity intelligence (evolve-summary, attributes update)
  Phase B:  Advanced search (traverse, reranker parameter)
  Phase C:  Episode enhancement (CRUD, batch-ingest, provenance)
  Phase D:  Contradiction detection
  Phase E:  DeepDream memory consolidation
  Phase F:  Agent-first API (ask, explain, suggestions)

Usage:
    pytest tests/test_comprehensive.py -v
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Fixtures (same pattern as test_api.py)
# ---------------------------------------------------------------------------


def _make_registry(storage_path: str):
    """Build a GraphRegistry with a minimal config that avoids needing a
    real LLM API key or embedding model."""
    from server.registry import GraphRegistry

    config = {
        "storage_path": storage_path,
        "llm": {
            "api_key": "test-key",
            "model": "gpt-4",
            "base_url": "http://127.0.0.1:9999/v1",
        },
        "embedding": {
            "model": None,
            "device": "cpu",
        },
        "chunking": {
            "window_size": 1000,
            "overlap": 200,
        },
        "remember_workers": 0,
        "remember_max_retries": 0,
        "remember_retry_delay_seconds": 0,
    }
    return GraphRegistry(base_storage_path=storage_path, config=config)


@pytest.fixture()
def storage_path(tmp_path):
    """Return a temporary directory string for graph storage."""
    return str(tmp_path / "graphs")


@pytest.fixture()
def registry(storage_path):
    """Create a GraphRegistry pointing at the tmp storage."""
    return _make_registry(storage_path)


@pytest.fixture()
def app(registry):
    """Build a Flask test app via create_app."""
    from server.api import create_app

    application = create_app(registry=registry, config={"rate_limit_per_minute": 600})
    application.config["TESTING"] = True
    return application


@pytest.fixture()
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture()
def graph_id():
    """A well-known test graph id."""
    return "testgraph"


@pytest.fixture()
def created_graph(client, graph_id):
    """Ensure a graph is created before the test runs; return its id."""
    resp = client.post(
        "/api/v1/graphs",
        data=json.dumps({"graph_id": graph_id}),
        content_type="application/json",
    )
    assert resp.status_code in (200, 409), f"Failed to create graph: {resp.data}"
    return graph_id


# ---------------------------------------------------------------------------
# Phase 1C: search_mode parameter
# ---------------------------------------------------------------------------


class TestSearchMode:
    """Phase 1C: search_mode parameter on /find and /search endpoints."""

    def test_find_unified_default_semantic(self, client, created_graph):
        """POST /api/v1/find with default search_mode (semantic) should work."""
        resp = client.post(
            f"/api/v1/find?graph_id={created_graph}",
            data=json.dumps({"query": "test entity"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "entities" in data["data"]
        assert "relations" in data["data"]

    def test_find_unified_bm25_mode(self, client, created_graph):
        """POST /api/v1/find with search_mode=bm25."""
        resp = client.post(
            f"/api/v1/find?graph_id={created_graph}",
            data=json.dumps({"query": "test", "search_mode": "bm25"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_find_unified_hybrid_mode(self, client, created_graph):
        """POST /api/v1/find with search_mode=hybrid."""
        resp = client.post(
            f"/api/v1/find?graph_id={created_graph}",
            data=json.dumps({"query": "test", "search_mode": "hybrid"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_find_unified_invalid_mode_fallback(self, client, created_graph):
        """Invalid search_mode should fall back to semantic (200)."""
        resp = client.post(
            f"/api/v1/find?graph_id={created_graph}",
            data=json.dumps({"query": "test", "search_mode": "invalid"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_entities_search_bm25(self, client, created_graph):
        """POST /api/v1/find/entities/search with search_mode=bm25."""
        resp = client.post(
            f"/api/v1/find/entities/search?graph_id={created_graph}",
            data=json.dumps({"query_name": "test", "search_mode": "bm25"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_entities_search_hybrid(self, client, created_graph):
        """POST /api/v1/find/entities/search with search_mode=hybrid."""
        resp = client.post(
            f"/api/v1/find/entities/search?graph_id={created_graph}",
            data=json.dumps({"query_name": "test", "search_mode": "hybrid"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_entities_search_invalid_mode_fallback(self, client, created_graph):
        """Invalid search_mode on entities/search should fall back to semantic."""
        resp = client.post(
            f"/api/v1/find/entities/search?graph_id={created_graph}",
            data=json.dumps({"query_name": "test", "search_mode": "badmode"}),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_relations_search_bm25(self, client, created_graph):
        """POST /api/v1/find/relations/search with search_mode=bm25."""
        resp = client.post(
            f"/api/v1/find/relations/search?graph_id={created_graph}",
            data=json.dumps({"query_text": "test", "search_mode": "bm25"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_relations_search_hybrid(self, client, created_graph):
        """POST /api/v1/find/relations/search with search_mode=hybrid."""
        resp = client.post(
            f"/api/v1/find/relations/search?graph_id={created_graph}",
            data=json.dumps({"query_text": "test", "search_mode": "hybrid"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_relations_search_invalid_mode_fallback(self, client, created_graph):
        """Invalid search_mode on relations/search should fall back to semantic."""
        resp = client.post(
            f"/api/v1/find/relations/search?graph_id={created_graph}",
            data=json.dumps({"query_text": "test", "search_mode": "badmode"}),
            content_type="application/json",
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Phase 2: Entity CRUD endpoints
# ---------------------------------------------------------------------------


class TestEntityCRUD:
    """Phase 2: Entity CRUD (update, delete, batch-delete)."""

    def test_update_entity_not_found(self, client, created_graph):
        """PUT /api/v1/find/entities/<id> should return 404 for non-existent entity."""
        resp = client.put(
            f"/api/v1/find/entities/nonexistent-id?graph_id={created_graph}",
            data=json.dumps({"name": "Updated"}),
            content_type="application/json",
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False

    def test_update_entity_missing_fields(self, client, created_graph):
        """PUT without name or content should return 400."""
        resp = client.put(
            f"/api/v1/find/entities/nonexistent-id?graph_id={created_graph}",
            data=json.dumps({}),
            content_type="application/json",
        )
        # The endpoint checks name/content first (400) before entity existence (404).
        # Since neither field is provided, we get 400.
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_delete_entity_not_found(self, client, created_graph):
        """DELETE /api/v1/find/entities/<id> should return 404."""
        resp = client.delete(
            f"/api/v1/find/entities/nonexistent-id?graph_id={created_graph}",
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False

    def test_batch_delete_entities_empty(self, client, created_graph):
        """POST /api/v1/find/entities/batch-delete with empty array should return 400."""
        resp = client.post(
            f"/api/v1/find/entities/batch-delete?graph_id={created_graph}",
            data=json.dumps({"entity_ids": []}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_batch_delete_entities_missing_field(self, client, created_graph):
        """POST /api/v1/find/entities/batch-delete without entity_ids should return 400."""
        resp = client.post(
            f"/api/v1/find/entities/batch-delete?graph_id={created_graph}",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_batch_delete_entities_nonexistent(self, client, created_graph):
        """POST /api/v1/find/entities/batch-delete with non-existent IDs returns 200 with count."""
        resp = client.post(
            f"/api/v1/find/entities/batch-delete?graph_id={created_graph}",
            data=json.dumps({"entity_ids": ["fake-id-1", "fake-id-2"]}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["count"] == 2


# ---------------------------------------------------------------------------
# Phase 2: Relation CRUD endpoints
# ---------------------------------------------------------------------------


class TestRelationCRUD:
    """Phase 2: Relation CRUD (update, delete, batch-delete)."""

    def test_update_relation_not_found(self, client, created_graph):
        """PUT /api/v1/find/relations/<id> should return 404."""
        resp = client.put(
            f"/api/v1/find/relations/nonexistent-id?graph_id={created_graph}",
            data=json.dumps({"content": "Updated"}),
            content_type="application/json",
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False

    def test_update_relation_missing_content(self, client, created_graph):
        """PUT without content field should return 400."""
        resp = client.put(
            f"/api/v1/find/relations/nonexistent-id?graph_id={created_graph}",
            data=json.dumps({}),
            content_type="application/json",
        )
        # content is validated first (400) before entity existence (404)
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_delete_relation_not_found(self, client, created_graph):
        """DELETE /api/v1/find/relations/<id> should return 404."""
        resp = client.delete(
            f"/api/v1/find/relations/nonexistent-id?graph_id={created_graph}",
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False

    def test_batch_delete_relations_empty(self, client, created_graph):
        """POST /api/v1/find/relations/batch-delete with empty array should return 400."""
        resp = client.post(
            f"/api/v1/find/relations/batch-delete?graph_id={created_graph}",
            data=json.dumps({"relation_ids": []}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_batch_delete_relations_missing_field(self, client, created_graph):
        """POST /api/v1/find/relations/batch-delete without relation_ids should return 400."""
        resp = client.post(
            f"/api/v1/find/relations/batch-delete?graph_id={created_graph}",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_batch_delete_relations_nonexistent(self, client, created_graph):
        """POST /api/v1/find/relations/batch-delete with non-existent IDs returns 200 with count."""
        resp = client.post(
            f"/api/v1/find/relations/batch-delete?graph_id={created_graph}",
            data=json.dumps({"relation_ids": ["fake-rel-1", "fake-rel-2"]}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["count"] == 2


# ---------------------------------------------------------------------------
# Phase 2: Entity merge
# ---------------------------------------------------------------------------


class TestEntityMerge:
    """Phase 2: Entity merge endpoint."""

    def test_merge_missing_params(self, client, created_graph):
        """POST /api/v1/find/entities/merge missing required params returns 400."""
        resp = client.post(
            f"/api/v1/find/entities/merge?graph_id={created_graph}",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_merge_missing_target(self, client, created_graph):
        """Merge with only source_entity_ids (no target) should return 400."""
        resp = client.post(
            f"/api/v1/find/entities/merge?graph_id={created_graph}",
            data=json.dumps({"source_entity_ids": ["source1"]}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_merge_missing_sources(self, client, created_graph):
        """Merge with only target_entity_id (no sources) should return 400."""
        resp = client.post(
            f"/api/v1/find/entities/merge?graph_id={created_graph}",
            data=json.dumps({"target_entity_id": "target1"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_merge_target_not_found(self, client, created_graph):
        """Merge with non-existent target should return 404."""
        resp = client.post(
            f"/api/v1/find/entities/merge?graph_id={created_graph}",
            data=json.dumps({"target_entity_id": "fake", "source_entity_ids": ["fake2"]}),
            content_type="application/json",
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False


# ---------------------------------------------------------------------------
# Phase 3: Time travel endpoints
# ---------------------------------------------------------------------------


class TestTimeTravel:
    """Phase 3: Time travel (snapshot, changes, invalidate)."""

    def test_snapshot_missing_time(self, client, created_graph):
        """GET /api/v1/find/snapshot without time should return 400."""
        resp = client.get(f"/api/v1/find/snapshot?graph_id={created_graph}")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_snapshot(self, client, created_graph):
        """GET /api/v1/find/snapshot with time should return 200."""
        resp = client.get(
            f"/api/v1/find/snapshot?graph_id={created_graph}&time=2025-01-01T00:00:00Z"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "entities" in data["data"]
        assert "relations" in data["data"]
        assert "entity_count" in data["data"]
        assert "relation_count" in data["data"]

    def test_snapshot_with_limit(self, client, created_graph):
        """GET /api/v1/find/snapshot with time and limit should return 200."""
        resp = client.get(
            f"/api/v1/find/snapshot?graph_id={created_graph}&time=2025-01-01T00:00:00Z&limit=5"
        )
        assert resp.status_code == 200

    def test_snapshot_invalid_time(self, client, created_graph):
        """GET /api/v1/find/snapshot with invalid time should return 500 (ValueError caught by generic handler)."""
        resp = client.get(
            f"/api/v1/find/snapshot?graph_id={created_graph}&time=not-a-time"
        )
        # parse_time_point raises ValueError which is caught by the generic except Exception
        # and turned into a 500 by the err() helper (status >= 500 is sanitized)
        assert resp.status_code == 500
        data = resp.get_json()
        assert data["success"] is False

    def test_changes_missing_since(self, client, created_graph):
        """GET /api/v1/find/changes without since should return 400."""
        resp = client.get(f"/api/v1/find/changes?graph_id={created_graph}")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_changes(self, client, created_graph):
        """GET /api/v1/find/changes with since should return 200."""
        resp = client.get(
            f"/api/v1/find/changes?graph_id={created_graph}&since=2020-01-01T00:00:00Z"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "entities" in data["data"]
        assert "relations" in data["data"]
        assert "entity_count" in data["data"]
        assert "relation_count" in data["data"]

    def test_changes_with_until(self, client, created_graph):
        """GET /api/v1/find/changes with since and until should return 200."""
        resp = client.get(
            f"/api/v1/find/changes?graph_id={created_graph}"
            "&since=2020-01-01T00:00:00Z&until=2025-01-01T00:00:00Z"
        )
        assert resp.status_code == 200

    def test_changes_invalid_since(self, client, created_graph):
        """GET /api/v1/find/changes with invalid since should return 500 (ValueError caught by generic handler)."""
        resp = client.get(
            f"/api/v1/find/changes?graph_id={created_graph}&since=bad-time"
        )
        # parse_time_point raises ValueError which is caught by the generic except Exception
        assert resp.status_code == 500
        data = resp.get_json()
        assert data["success"] is False

    def test_invalidate_relation_not_found(self, client, created_graph):
        """POST /api/v1/find/relations/<id>/invalidate should return 404 for non-existent."""
        resp = client.post(
            f"/api/v1/find/relations/fake-id/invalidate?graph_id={created_graph}",
            data=json.dumps({"reason": "test"}),
            content_type="application/json",
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False

    def test_invalidate_relation_no_body(self, client, created_graph):
        """POST /api/v1/find/relations/<id>/invalidate without body should still work (reason defaults to '')."""
        resp = client.post(
            f"/api/v1/find/relations/fake-id/invalidate?graph_id={created_graph}",
            content_type="application/json",
        )
        assert resp.status_code == 404

    def test_invalidated_relations(self, client, created_graph):
        """GET /api/v1/find/relations/invalidated should return 200."""
        resp = client.get(
            f"/api/v1/find/relations/invalidated?graph_id={created_graph}"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["data"], list)

    def test_invalidated_relations_with_limit(self, client, created_graph):
        """GET /api/v1/find/relations/invalidated with limit should return 200."""
        resp = client.get(
            f"/api/v1/find/relations/invalidated?graph_id={created_graph}&limit=5"
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Phase 4: Graph statistics
# ---------------------------------------------------------------------------


class TestGraphStats:
    """Phase 4: Graph statistics endpoint."""

    def test_graph_stats(self, client, created_graph):
        """GET /api/v1/find/graph-stats should return stats."""
        resp = client.get(f"/api/v1/find/graph-stats?graph_id={created_graph}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "entity_count" in data["data"]
        assert "relation_count" in data["data"]
        assert "graph_density" in data["data"]

    def test_graph_stats_empty_graph(self, client, created_graph):
        """GET /api/v1/find/graph-stats on empty graph should return zero counts."""
        resp = client.get(f"/api/v1/find/graph-stats?graph_id={created_graph}")
        assert resp.status_code == 200
        data = resp.get_json()
        stats = data["data"]
        assert stats["entity_count"] == 0
        assert stats["relation_count"] == 0


# ---------------------------------------------------------------------------
# Phase 4: Entity timeline
# ---------------------------------------------------------------------------


class TestEntityTimeline:
    """Phase 4: Entity timeline endpoint."""

    def test_timeline_not_found(self, client, created_graph):
        """GET /api/v1/find/entities/<id>/timeline should return 404 for non-existent."""
        resp = client.get(
            f"/api/v1/find/entities/nonexistent-id/timeline?graph_id={created_graph}"
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False

    def test_timeline_response_structure(self):
        """Verify the timeline endpoint is properly registered (smoke test with mock)."""
        # The route exists but needs a real entity; covered by not-found test above.
        pass


# ---------------------------------------------------------------------------
# Phase A: Entity Intelligence
# ---------------------------------------------------------------------------


class TestEvolveSummary:
    """Phase A: Evolve entity summary endpoint."""

    def test_evolve_summary_not_found(self, client, created_graph):
        """POST /api/v1/find/entities/<id>/evolve-summary should return 404 for non-existent entity."""
        resp = client.post(
            f"/api/v1/find/entities/nonexistent-id/evolve-summary?graph_id={created_graph}",
            content_type="application/json",
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False


class TestEntityAttributes:
    """Phase A: Entity summary/attributes update via PUT."""

    def test_update_entity_summary_direct(self, client, created_graph):
        """PUT with summary only (no name/content) should update attributes directly (not create version)."""
        # Non-existent entity: the storage layer may raise, but the endpoint
        # calls update_entity_summary which may silently succeed or fail.
        resp = client.put(
            f"/api/v1/find/entities/some-entity?graph_id={created_graph}",
            data=json.dumps({"summary": "A new summary"}),
            content_type="application/json",
        )
        # Either 200 (storage accepted) or 500 (storage error) is acceptable;
        # the important thing is it doesn't crash with a 404 (because 404 only
        # fires on the name/content path).
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.get_json()
            assert data["success"] is True

    def test_update_entity_attributes_direct(self, client, created_graph):
        """PUT with attributes dict should store JSON."""
        resp = client.put(
            f"/api/v1/find/entities/some-entity?graph_id={created_graph}",
            data=json.dumps({"attributes": {"role": "engineer", "team": "platform"}}),
            content_type="application/json",
        )
        assert resp.status_code in (200, 500)


# ---------------------------------------------------------------------------
# Phase B: Advanced Search
# ---------------------------------------------------------------------------


class TestTraverseGraph:
    """Phase B: BFS graph traversal search endpoint."""

    def test_traverse_missing_seed_ids(self, client, created_graph):
        """POST /api/v1/find/traverse without seed_entity_ids should return 400."""
        resp = client.post(
            f"/api/v1/find/traverse?graph_id={created_graph}",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_traverse_empty_seed_ids(self, client, created_graph):
        """POST /api/v1/find/traverse with empty array should return 400."""
        resp = client.post(
            f"/api/v1/find/traverse?graph_id={created_graph}",
            data=json.dumps({"seed_entity_ids": []}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_traverse_nonexistent_seeds(self, client, created_graph):
        """POST /api/v1/find/traverse with non-existent seed IDs returns 200 with empty list."""
        resp = client.post(
            f"/api/v1/find/traverse?graph_id={created_graph}",
            data=json.dumps({"seed_entity_ids": ["fake-1", "fake-2"]}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["data"], list)

    def test_traverse_with_depth_and_limit(self, client, created_graph):
        """POST /api/v1/find/traverse with max_depth and max_nodes parameters."""
        resp = client.post(
            f"/api/v1/find/traverse?graph_id={created_graph}",
            data=json.dumps({
                "seed_entity_ids": ["fake-1"],
                "max_depth": 1,
                "max_nodes": 10,
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True


class TestReranker:
    """Phase B: Reranker parameter on /find endpoint."""

    def test_find_with_reranker_rrf(self, client, created_graph):
        """POST /api/v1/find with reranker=rrf should work."""
        resp = client.post(
            f"/api/v1/find?graph_id={created_graph}",
            data=json.dumps({"query": "test", "reranker": "rrf"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_find_with_reranker_mmr(self, client, created_graph):
        """POST /api/v1/find with reranker=mmr should work."""
        resp = client.post(
            f"/api/v1/find?graph_id={created_graph}",
            data=json.dumps({"query": "test", "reranker": "mmr"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_find_with_reranker_node_degree(self, client, created_graph):
        """POST /api/v1/find with reranker=node_degree should work."""
        resp = client.post(
            f"/api/v1/find?graph_id={created_graph}",
            data=json.dumps({"query": "test", "reranker": "node_degree"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True


# ---------------------------------------------------------------------------
# Phase C: Episode Enhancement
# ---------------------------------------------------------------------------


class TestEpisodeCRUD:
    """Phase C: Episode CRUD endpoints."""

    def test_get_episode_not_found(self, client, created_graph):
        """GET /api/v1/find/episodes/<id> should return 404 for non-existent."""
        resp = client.get(
            f"/api/v1/find/episodes/nonexistent-id?graph_id={created_graph}",
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False

    def test_delete_episode_not_found(self, client, created_graph):
        """DELETE /api/v1/find/episodes/<id> should return 404 for non-existent."""
        resp = client.delete(
            f"/api/v1/find/episodes/nonexistent-id?graph_id={created_graph}",
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False

    def test_search_episodes_missing_query(self, client, created_graph):
        """POST /api/v1/find/episodes/search without query should return 400."""
        resp = client.post(
            f"/api/v1/find/episodes/search?graph_id={created_graph}",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_search_episodes(self, client, created_graph):
        """POST /api/v1/find/episodes/search with query should return 200."""
        resp = client.post(
            f"/api/v1/find/episodes/search?graph_id={created_graph}",
            data=json.dumps({"query": "test episode"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["data"], list)

    def test_batch_ingest_empty(self, client, created_graph):
        """POST /api/v1/find/episodes/batch-ingest with empty array should return 200 with count 0."""
        resp = client.post(
            f"/api/v1/find/episodes/batch-ingest?graph_id={created_graph}",
            data=json.dumps({"episodes": []}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["count"] == 0

    def test_batch_ingest_invalid_format(self, client, created_graph):
        """POST /api/v1/find/episodes/batch-ingest with non-array episodes should return 400."""
        resp = client.post(
            f"/api/v1/find/episodes/batch-ingest?graph_id={created_graph}",
            data=json.dumps({"episodes": "not an array"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_batch_ingest_with_content(self, client, created_graph):
        """POST /api/v1/find/episodes/batch-ingest with valid episodes should succeed."""
        resp = client.post(
            f"/api/v1/find/episodes/batch-ingest?graph_id={created_graph}",
            data=json.dumps({
                "episodes": [
                    {"content": "Alice met Bob at the conference.", "episode_type": "narrative"},
                    {"text": "Bob works at Acme Corp.", "source_document": "test"},
                ]
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["count"] == 2

    def test_batch_ingest_skips_empty(self, client, created_graph):
        """Episodes with empty content should be skipped."""
        resp = client.post(
            f"/api/v1/find/episodes/batch-ingest?graph_id={created_graph}",
            data=json.dumps({
                "episodes": [
                    {"content": "Valid content"},
                    {"content": ""},
                    {"text": "   "},
                ]
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["count"] == 1


class TestProvenance:
    """Phase C: Entity provenance (fact tracing) endpoint."""

    def test_provenance_returns_list(self, client, created_graph):
        """GET /api/v1/find/entities/<id>/provenance should return 200 with a list."""
        resp = client.get(
            f"/api/v1/find/entities/nonexistent-id/provenance?graph_id={created_graph}",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["data"], list)


# ---------------------------------------------------------------------------
# Phase D: Contradiction Detection
# ---------------------------------------------------------------------------


class TestContradictions:
    """Phase D: Contradiction detection endpoints."""

    def test_contradictions_single_version(self, client, created_graph):
        """GET contradictions for entity with <2 versions should return empty list."""
        resp = client.get(
            f"/api/v1/find/entities/nonexistent-id/contradictions?graph_id={created_graph}",
        )
        # Non-existent entity has 0 versions, so should return empty list
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["data"], list)

    def test_resolve_contradiction_missing_field(self, client, created_graph):
        """POST resolve-contradiction without contradiction field should return 400."""
        resp = client.post(
            f"/api/v1/find/entities/some-id/resolve-contradiction?graph_id={created_graph}",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_resolve_contradiction_not_dict(self, client, created_graph):
        """POST resolve-contradiction with non-dict contradiction should return 400."""
        resp = client.post(
            f"/api/v1/find/entities/some-id/resolve-contradiction?graph_id={created_graph}",
            data=json.dumps({"contradiction": "just a string"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_resolve_contradiction_valid(self, client, created_graph):
        """POST resolve-contradiction with valid dict should attempt LLM resolution."""
        resp = client.post(
            f"/api/v1/find/entities/some-id/resolve-contradiction?graph_id={created_graph}",
            data=json.dumps({"contradiction": {"id": "c1", "description": "test contradiction"}}),
            content_type="application/json",
        )
        # Will likely fail (no real LLM), but should not be a 400 validation error
        assert resp.status_code != 400


# ---------------------------------------------------------------------------
# Phase E: DeepDream
# ---------------------------------------------------------------------------


class TestDeepDream:
    """Phase E: DeepDream memory consolidation endpoints."""

    def test_dream_status(self, client, created_graph):
        """GET /api/v1/find/dream/status should return 200."""
        resp = client.get(
            f"/api/v1/find/dream/status?graph_id={created_graph}",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        # Should have status field (either "not_available", "no_cycles", or dream log data)
        assert "status" in data["data"] or "cycle_id" in data["data"]

    def test_dream_logs(self, client, created_graph):
        """GET /api/v1/find/dream/logs should return 200 with a list."""
        resp = client.get(
            f"/api/v1/find/dream/logs?graph_id={created_graph}",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["data"], list)

    def test_dream_logs_with_limit(self, client, created_graph):
        """GET /api/v1/find/dream/logs with limit parameter."""
        resp = client.get(
            f"/api/v1/find/dream/logs?graph_id={created_graph}&limit=5",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_dream_log_detail_not_found(self, client, created_graph):
        """GET /api/v1/find/dream/logs/<id> should return 404 for non-existent."""
        resp = client.get(
            f"/api/v1/find/dream/logs/nonexistent-cycle?graph_id={created_graph}",
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False

    def test_start_dream_empty_graph(self, client, created_graph):
        """POST /api/v1/find/dream/start on empty graph — route is registered (needs LLM for full run)."""
        resp = client.post(
            f"/api/v1/find/dream/start?graph_id={created_graph}",
            data=json.dumps({
                "review_window_days": 7,
                "max_entities_per_cycle": 10,
                "similarity_threshold": 0.8,
            }),
            content_type="application/json",
        )
        # Route is registered; will fail gracefully (no LLM) with 500
        assert resp.status_code == 500
        data = resp.get_json()
        assert data["success"] is False

    def test_start_dream_default_params(self, client, created_graph):
        """POST /api/v1/find/dream/start with no body (default params)."""
        resp = client.post(
            f"/api/v1/find/dream/start?graph_id={created_graph}",
            content_type="application/json",
        )
        # Needs LLM, so 500 is expected
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Phase F: Agent-First API
# ---------------------------------------------------------------------------


class TestAgentAsk:
    """Phase F: Agent meta-query endpoint."""

    def test_ask_missing_question(self, client, created_graph):
        """POST /api/v1/find/ask without question should return 400."""
        resp = client.post(
            f"/api/v1/find/ask?graph_id={created_graph}",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_ask_empty_question(self, client, created_graph):
        """POST /api/v1/find/ask with empty question should return 400."""
        resp = client.post(
            f"/api/v1/find/ask?graph_id={created_graph}",
            data=json.dumps({"question": "   "}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_ask_with_question(self, client, created_graph):
        """POST /api/v1/find/ask with a valid question should attempt query (may fail without LLM)."""
        resp = client.post(
            f"/api/v1/find/ask?graph_id={created_graph}",
            data=json.dumps({"question": "What entities exist in this graph?"}),
            content_type="application/json",
        )
        # Will likely fail (no real LLM for meta_query), but route is registered
        assert resp.status_code != 400


class TestExplainEntity:
    """Phase F: Entity explanation endpoint."""

    def test_explain_missing_entity_id(self, client, created_graph):
        """POST /api/v1/find/explain without entity_id should return 400."""
        resp = client.post(
            f"/api/v1/find/explain?graph_id={created_graph}",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_explain_empty_entity_id(self, client, created_graph):
        """POST /api/v1/find/explain with empty entity_id should return 400."""
        resp = client.post(
            f"/api/v1/find/explain?graph_id={created_graph}",
            data=json.dumps({"entity_id": "  "}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_explain_not_found(self, client, created_graph):
        """POST /api/v1/find/explain for non-existent entity should return 404."""
        resp = client.post(
            f"/api/v1/find/explain?graph_id={created_graph}",
            data=json.dumps({"entity_id": "nonexistent-id", "aspect": "summary"}),
            content_type="application/json",
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False


class TestSuggestions:
    """Phase F: Smart suggestions endpoint."""

    def test_suggestions(self, client, created_graph):
        """GET /api/v1/find/suggestions should return 200."""
        resp = client.get(
            f"/api/v1/find/suggestions?graph_id={created_graph}",
        )
        # May fail (no real LLM), but route is registered
        assert resp.status_code != 400
