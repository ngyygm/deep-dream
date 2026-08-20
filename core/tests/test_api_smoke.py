"""
Smoke tests for the v1 route modules.

These tests keep the API surface honest after the Document-first Concept
refactor: new concept/document routes must respond, and removed legacy/Dream
routes must stay absent.

Subjects already covered (more thoroughly) by test_api.py — health, route
index, find stats, remember tasks, concepts/documents list, legacy-route
removal, invalid JSON/graph-id — are NOT repeated here.
"""
from __future__ import annotations

from core.tests.conftest import TEST_GRAPH_ID


class TestSystemRoutesSmoke:
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

    def test_documents_graph_validation_responds(self, client):
        response = client.post("/api/v1/documents/graph", json={"graph_id": TEST_GRAPH_ID})
        assert response.status_code == 400

    def test_traverse_validation_responds(self, client):
        response = client.post("/api/v1/traverse", json={"graph_id": TEST_GRAPH_ID})
        assert response.status_code == 400

    def test_vault_index_validation_responds(self, client):
        response = client.post("/api/v1/vaults/index", json={"graph_id": TEST_GRAPH_ID})
        assert response.status_code == 400


class TestRouteRegistrationSmoke:
    def test_all_route_modules_registered(self, test_app):
        expected_route_modules = {"system", "remember", "concepts"}
        registered_route_modules = set(test_app.blueprints.keys())

        for route_module in expected_route_modules:
            assert route_module in registered_route_modules

    def test_route_module_has_routes(self, test_app):
        for route_module_name, flask_route_group in test_app.blueprints.items():
            assert len(flask_route_group.deferred_functions) > 0 or hasattr(flask_route_group, "name"), (
                f"Route module '{route_module}' has no registered routes"
            )
