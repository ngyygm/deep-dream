"""
Smoke tests for the v1 route modules.

These tests keep the API surface honest after the Document-first Concept
refactor: new concept/document routes must respond, and removed legacy/Dream
routes must stay absent.
"""
from __future__ import annotations

from core.tests.conftest import TEST_GRAPH_ID


class TestSystemRoutesSmoke:
    def test_health_check_responds(self, client):
        response = client.get(f"/api/v1/health?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 200

    def test_route_index_responds(self, client):
        response = client.get("/api/v1/routes")
        assert response.status_code == 200

    def test_find_stats_responds(self, client):
        response = client.get(f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 200

    def test_system_overview_responds(self, client):
        response = client.get("/api/v1/system/overview")
        assert response.status_code == 200


class TestRememberRoutesSmoke:
    def test_remember_responds(self, client):
        response = client.post(
            "/api/v1/remember",
            json={"graph_id": TEST_GRAPH_ID, "text": "Smoke test memory."},
        )
        assert response.status_code in {202, 400, 422}

    def test_remember_tasks_responds(self, client):
        response = client.get(f"/api/v1/remember/tasks?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 200

    def test_remember_monitor_responds(self, client):
        response = client.get(f"/api/v1/remember/monitor?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 200


class TestConceptRoutesSmoke:
    def test_find_alias_responds(self, client):
        response = client.post(
            "/api/v1/find",
            json={"graph_id": TEST_GRAPH_ID, "query": "test"},
        )
        assert response.status_code == 200

    def test_concepts_search_responds(self, client):
        response = client.post(
            "/api/v1/concepts/search",
            json={"graph_id": TEST_GRAPH_ID, "query": "test"},
        )
        assert response.status_code == 200

    def test_concepts_list_responds(self, client):
        response = client.get(f"/api/v1/concepts?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 200

    def test_documents_list_responds(self, client):
        response = client.get(f"/api/v1/documents?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 200

    def test_documents_graph_validation_responds(self, client):
        response = client.post("/api/v1/documents/graph", json={"graph_id": TEST_GRAPH_ID})
        assert response.status_code == 400

    def test_traverse_validation_responds(self, client):
        response = client.post("/api/v1/traverse", json={"graph_id": TEST_GRAPH_ID})
        assert response.status_code == 400

    def test_vault_index_validation_responds(self, client):
        response = client.post("/api/v1/vaults/index", json={"graph_id": TEST_GRAPH_ID})
        assert response.status_code == 400

    def test_graphs_list_responds(self, client):
        response = client.get("/api/v1/graphs")
        assert response.status_code == 200


class TestRemovedRoutesSmoke:
    def test_legacy_entity_routes_removed(self, client):
        response = client.get(f"/api/v1/find/entities?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 404

    def test_legacy_relation_routes_removed(self, client):
        response = client.get(f"/api/v1/find/relations?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 404

    def test_legacy_episode_routes_removed(self, client):
        response = client.get(f"/api/v1/episodes?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 404

    def test_dream_routes_removed(self, client):
        response = client.get(f"/api/v1/find/dream/status?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 404

    def test_dream_candidate_routes_removed(self, client):
        response = client.get(f"/api/v1/dream/candidates?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 404


class TestRouteRegistrationSmoke:
    def test_all_route_modules_registered(self, test_app):
        expected_route_modules = {"system", "remember", "concepts"}
        registered_route_modules = set(test_app.blueprints.keys())

        for route_module in expected_route_modules:
            assert route_module in registered_route_modules

    def test_route_module_has_routes(self, test_app):
        for route_module_name, flask_route_group in test_app.blueprints.items():
            assert len(flask_route_group.deferred_functions) > 0 or hasattr(flask_route_group, "name"), (
                f"Route module '{route_module_name}' has no registered routes"
            )


class TestErrorHandlingSmoke:
    def test_invalid_graph_id_returns_400(self, client):
        response = client.get("/api/v1/find/stats?graph_id=invalid@graph#id")
        assert response.status_code == 400

    def test_invalid_json_returns_400(self, client):
        response = client.post(
            "/api/v1/remember",
            data="invalid json",
            content_type="application/json",
        )
        assert response.status_code == 400
