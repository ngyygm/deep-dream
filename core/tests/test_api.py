"""
Integration tests for the v1 Document-first Concept API.

The public model is now Document -> Episode -> Concept. Legacy entity,
relation, episode CRUD routes are intentionally absent; entity/relation still
exist as concept roles inside the storage layer and Step9/Step10 pipeline.

Run with: pytest core/tests/test_api.py -v
"""
from __future__ import annotations

import uuid

from core.tests.conftest import TEST_GRAPH_ID


class TestSystemEndpoints:
    """System and route index endpoints."""

    def test_health_check(self, client):
        response = client.get(f"/api/v1/health?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "library_id" in data["data"]

    def test_route_index(self, client):
        response = client.get("/api/v1/routes")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "routes" in data["data"]

    def test_find_stats(self, client):
        response = client.get(f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "total_concepts" in data["data"]
        assert "total_documents" in data["data"]


class TestRememberEndpoints:
    """Document-first remember endpoints."""

    def test_remember_requires_text_or_file(self, client):
        response = client.post("/api/v1/remember", json={"graph_id": TEST_GRAPH_ID})
        assert response.status_code == 400
        assert response.get_json()["success"] is False

    def test_remember_text_queues_task(self, client):
        response = client.post(
            "/api/v1/remember",
            json={
                "graph_id": TEST_GRAPH_ID,
                "text": "# API Test\n\nDeep-Dream stores Markdown as documents.",
                "source_name": "api-test.md",
            },
        )
        assert response.status_code == 202
        data = response.get_json()
        assert data["success"] is True
        assert data["data"]["status"] in {"queued", "processing", "completed"}
        assert "task_id" in data["data"]

    def test_remember_task_list(self, client):
        response = client.get(f"/api/v1/remember/tasks?graph_id={TEST_GRAPH_ID}&limit=10")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert isinstance(data["data"]["tasks"], list)


class TestConceptEndpoints:
    """Concept search, traversal, provenance, and documents."""

    def test_concept_search_requires_query(self, client):
        response = client.post("/api/v1/concepts/search", json={"graph_id": TEST_GRAPH_ID})
        assert response.status_code == 400
        assert response.get_json()["success"] is False

    def test_concept_search(self, client):
        response = client.post(
            "/api/v1/concepts/search",
            json={"graph_id": TEST_GRAPH_ID, "query": "Markdown document", "limit": 5},
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "concepts" in data["data"]
        assert "total" in data["data"]

    def test_concepts_list(self, client):
        response = client.get(f"/api/v1/concepts?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "concepts" in data["data"]

    def test_documents_list(self, client):
        response = client.get(f"/api/v1/documents?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "documents" in data["data"]

    def test_documents_graph_requires_document_selection(self, client):
        response = client.post("/api/v1/documents/graph", json={"graph_id": TEST_GRAPH_ID})
        assert response.status_code == 400
        assert response.get_json()["success"] is False

    def test_traverse_requires_start_id(self, client):
        response = client.post("/api/v1/traverse", json={"graph_id": TEST_GRAPH_ID})
        assert response.status_code == 400
        assert response.get_json()["success"] is False

    def test_agent_sql_endpoint(self, client):
        response = client.post(
            "/api/v1/agent/sql",
            json={
                "graph_id": TEST_GRAPH_ID,
                "sql": "SELECT name FROM sqlite_master WHERE type = 'view' AND name = 'v_latest_concept'",
                "limit": 5,
            },
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["data"]["columns"] == ["name"]

    def test_agent_sql_rejects_write(self, client):
        response = client.post(
            "/api/v1/agent/sql",
            json={"graph_id": TEST_GRAPH_ID, "sql": "DELETE FROM concept_family"},
        )
        assert response.status_code == 400
        assert response.get_json()["success"] is False

    def test_agent_semantic_search_endpoint(self, client):
        response = client.post(
            "/api/v1/agent/semantic-search",
            json={"graph_id": TEST_GRAPH_ID, "query": "Markdown", "role": "entity", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "results" in data["data"]

    def test_concept_not_found(self, client):
        response = client.get(f"/api/v1/concepts/missing_family?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 404

    def test_concept_versions_not_found(self, client):
        response = client.get(f"/api/v1/concepts/missing_family/versions?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 404

    def test_concept_provenance_not_found(self, client):
        response = client.get(f"/api/v1/concepts/missing_family/provenance?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 404



class TestRemovedLegacyRoutes:
    """Old public model routes should not reappear."""

    def test_old_entity_routes_removed(self, client):
        response = client.get(f"/api/v1/find/entities?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 404

    def test_old_relation_routes_removed(self, client):
        response = client.get(f"/api/v1/find/relations?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 404

    def test_old_episode_routes_removed(self, client):
        response = client.get(f"/api/v1/episodes?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 404

    def test_old_dream_routes_removed(self, client):
        response = client.get(f"/api/v1/find/dream/status?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 404


class TestErrorHandling:
    """API error handling."""

    def test_404_route(self, client):
        response = client.get("/api/v1/nonexistent-route")
        assert response.status_code == 404

    def test_invalid_json_body(self, client):
        response = client.post(
            "/api/v1/concepts/search",
            data="invalid json",
            content_type="application/json",
            headers={"X-Graph-Id": TEST_GRAPH_ID},
        )
        assert response.status_code == 400

    def test_invalid_graph_id_format(self, client):
        response = client.get("/api/v1/health?graph_id=invalid_id!")
        assert response.status_code == 400
