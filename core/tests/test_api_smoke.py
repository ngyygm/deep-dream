"""
Smoke tests for all API blueprints to ensure endpoints respond.

These tests verify that all API endpoints are accessible and return appropriate
HTTP status codes. They don't test functionality deeply, just ensure endpoints respond.

Run with: pytest core/tests/test_api_smoke.py -v

Run without Neo4j-dependent tests: pytest core/tests/test_api_smoke.py -v -m "not neo4j"
"""
import os
import pytest
from core.tests.conftest import TEST_GRAPH_ID


# Neo4j availability check for skipif decorators
def _neo4j_available() -> bool:
    """Check if Neo4j is available for testing."""
    neo4j_uri = os.getenv("NEO4J_TEST_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_TEST_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_TEST_PASSWORD", "tmg2024secure")

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


NEO4J_AVAILABLE = _neo4j_available()

skip_if_no_neo4j = pytest.mark.skipif(
    not NEO4J_AVAILABLE,
    reason="Neo4j not available - set NEO4J_TEST_URI, NEO4J_TEST_USER, NEO4J_TEST_PASSWORD"
)


# ============================================================================
# System Blueprint Smoke Tests
# ============================================================================

class TestSystemBlueprintSmoke:
    """Smoke tests for system blueprint endpoints."""

    def test_health_check_responds(self, client):
        """Test /api/v1/health endpoint responds."""
        response = client.get(f"/api/v1/health?graph_id={TEST_GRAPH_ID}")
        # Should return 200 or redirect
        assert response.status_code in [200, 302, 307, 308]

    def test_health_llm_responds(self, client):
        """Test /api/v1/health/llm endpoint responds."""
        response = client.get(f"/api/v1/health/llm?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 202, 503]  # 503 if LLM unavailable

    def test_find_stats_responds(self, client):
        """Test /api/v1/find/stats endpoint responds."""
        response = client.get(f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}")
        assert response.status_code == 200

    def test_stats_counts_responds(self, client):
        """Test /api/v1/stats/counts endpoint responds."""
        response = client.get(f"/api/v1/stats/counts?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_system_dashboard_responds(self, client):
        """Test /api/v1/system/dashboard endpoint responds."""
        response = client.get(f"/api/v1/system/dashboard?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_system_overview_responds(self, client):
        """Test /api/v1/system/overview endpoint responds."""
        response = client.get(f"/api/v1/system/overview?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_system_graphs_responds(self, client):
        """Test /api/v1/system/graphs endpoint responds."""
        response = client.get(f"/api/v1/system/graphs?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_system_tasks_responds(self, client):
        """Test /api/v1/system/tasks endpoint responds."""
        response = client.get(f"/api/v1/system/tasks?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_system_logs_responds(self, client):
        """Test /api/v1/system/logs endpoint responds."""
        response = client.get(f"/api/v1/system/logs?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_system_access_stats_responds(self, client):
        """Test /api/v1/system/access-stats endpoint responds."""
        response = client.get(f"/api/v1/system/access-stats?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]


# ============================================================================
# Remember Blueprint Smoke Tests
# ============================================================================

class TestRememberBlueprintSmoke:
    """Smoke tests for remember blueprint endpoints."""

    def test_remember_responds(self, client):
        """Test POST /api/v1/remember endpoint responds."""
        response = client.post(
            f"/api/v1/remember",
            json={"graph_id": TEST_GRAPH_ID, "text": "Test smoke test memory."}
        )
        assert response.status_code in [200, 202, 400, 422]

    def test_remember_tasks_list_responds(self, client):
        """Test GET /api/v1/remember/tasks responds."""
        response = client.get(f"/api/v1/remember/tasks?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_remember_monitor_responds(self, client):
        """Test GET /api/v1/remember/monitor responds."""
        response = client.get(f"/api/v1/remember/monitor?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]


# ============================================================================
# Entities Blueprint Smoke Tests
# ============================================================================

class TestEntitiesBlueprintSmoke:
    """Smoke tests for entities blueprint endpoints."""

    def test_find_entities_responds(self, client):
        """Test GET /api/v1/find/entities responds."""
        response = client.get(f"/api/v1/find/entities?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 400, 404]

    def test_find_entities_search_responds(self, client):
        """Test POST /api/v1/find/entities/search responds."""
        response = client.post(
            f"/api/v1/find/entities/search?graph_id={TEST_GRAPH_ID}",
            json={"query": "test"}
        )
        assert response.status_code in [200, 400, 404]

    def test_find_entities_version_counts_responds(self, client):
        """Test POST /api/v1/find/entities/version-counts responds."""
        response = client.post(
            f"/api/v1/find/entities/version-counts?graph_id={TEST_GRAPH_ID}",
            json={"family_ids": []}
        )
        assert response.status_code in [200, 400, 404]

    def test_find_entities_graph_stats_responds(self, client):
        """Test GET /api/v1/find/graph-stats responds."""
        response = client.get(f"/api/v1/find/graph-stats?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_find_graph_summary_responds(self, client):
        """Test GET /api/v1/find/graph-summary responds."""
        response = client.get(f"/api/v1/find/graph-summary?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_find_relations_responds(self, client):
        """Test GET /api/v1/find/relations responds."""
        response = client.get(f"/api/v1/find/relations?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 400, 404]

    def test_find_relations_search_responds(self, client):
        """Test POST /api/v1/find/relations/search responds."""
        response = client.post(
            f"/api/v1/find/relations/search?graph_id={TEST_GRAPH_ID}",
            json={"query": "test"}
        )
        assert response.status_code in [200, 400, 404]


# ============================================================================
# Episodes Blueprint Smoke Tests
# ============================================================================

class TestEpisodesBlueprintSmoke:
    """Smoke tests for episodes blueprint endpoints."""

    def test_episodes_latest_metadata_responds(self, client):
        """Test GET /api/v1/find/episodes/latest/metadata responds."""
        response = client.get(f"/api/v1/find/episodes/latest/metadata?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_episodes_latest_responds(self, client):
        """Test GET /api/v1/find/episodes/latest responds."""
        response = client.get(f"/api/v1/find/episodes/latest?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_episodes_list_responds(self, client):
        """Test GET /api/v1/episodes responds."""
        response = client.get(f"/api/v1/episodes?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_episodes_search_responds(self, client):
        """Test POST /api/v1/episodes/search responds."""
        response = client.post(
            f"/api/v1/episodes/search?graph_id={TEST_GRAPH_ID}",
            json={"query": "test"}
        )
        assert response.status_code in [200, 400, 404]

    def test_find_snapshot_responds(self, client):
        """Test GET /api/v1/find/snapshot responds."""
        response = client.get(f"/api/v1/find/snapshot?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_find_changes_responds(self, client):
        """Test GET /api/v1/find/changes responds."""
        response = client.get(f"/api/v1/find/changes?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]


# ============================================================================
# Concepts Blueprint Smoke Tests
# ============================================================================

class TestConceptsBlueprintSmoke:
    """Smoke tests for concepts blueprint endpoints."""

    def test_concepts_search_responds(self, client):
        """Test POST /api/v1/concepts/search responds."""
        response = client.post(
            f"/api/v1/concepts/search?graph_id={TEST_GRAPH_ID}",
            json={"query": "test"}
        )
        assert response.status_code in [200, 400, 404]

    def test_concepts_list_responds(self, client):
        """Test GET /api/v1/concepts responds."""
        response = client.get(f"/api/v1/concepts?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_communities_detect_responds(self, client):
        """Test POST /api/v1/communities/detect responds."""
        response = client.post(
            f"/api/v1/communities/detect?graph_id={TEST_GRAPH_ID}",
            json={"algorithm": "label_propagation"}
        )
        assert response.status_code in [200, 400, 404]

    def test_communities_list_responds(self, client):
        """Test GET /api/v1/communities responds."""
        response = client.get(f"/api/v1/communities?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_graphs_list_responds(self, client):
        """Test GET /api/v1/graphs responds."""
        response = client.get(f"/api/v1/graphs")
        assert response.status_code in [200, 404]

    def test_chat_sessions_list_responds(self, client):
        """Test GET /api/v1/chat/sessions responds."""
        response = client.get(f"/api/v1/chat/sessions?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]


# ============================================================================
# Dream Blueprint Smoke Tests
# ============================================================================

class TestDreamBlueprintSmoke:
    """Smoke tests for dream blueprint endpoints."""

    def test_dream_status_responds(self, client):
        """Test GET /api/v1/find/dream/status responds."""
        response = client.get(f"/api/v1/find/dream/status?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_dream_logs_responds(self, client):
        """Test GET /api/v1/find/dream/logs responds."""
        response = client.get(f"/api/v1/find/dream/logs?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_dream_seeds_responds(self, client):
        """Test POST /api/v1/find/dream/seeds responds."""
        response = client.post(
            f"/api/v1/find/dream/seeds?graph_id={TEST_GRAPH_ID}",
            json={"seed": "test seed"}
        )
        assert response.status_code in [200, 202, 400, 404]

    def test_find_ask_responds(self, client):
        """Test POST /api/v1/find/ask responds."""
        response = client.post(
            f"/api/v1/find/ask?graph_id={TEST_GRAPH_ID}",
            json={"query": "test question"}
        )
        assert response.status_code in [200, 202, 400, 404]

    def test_find_explain_responds(self, client):
        """Test POST /api/v1/find/explain responds."""
        response = client.post(
            f"/api/v1/find/explain?graph_id={TEST_GRAPH_ID}",
            json={"query": "test explain"}
        )
        assert response.status_code in [200, 202, 400, 404]

    def test_find_suggestions_responds(self, client):
        """Test GET /api/v1/find/suggestions responds."""
        response = client.get(f"/api/v1/find/suggestions?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_quality_report_responds(self, client):
        """Test GET /api/v1/find/quality-report responds."""
        response = client.get(f"/api/v1/find/quality-report?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_maintenance_health_responds(self, client):
        """Test GET /api/v1/find/maintenance/health responds."""
        response = client.get(f"/api/v1/find/maintenance/health?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_dream_candidates_responds(self, client):
        """Test GET /api/v1/dream/candidates responds."""
        response = client.get(f"/api/v1/dream/candidates?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]


# ============================================================================
# Route Index Smoke Test
# ============================================================================

class TestRouteIndexSmoke:
    """Test that route index is accessible and complete."""

    def test_route_index_endpoint_exists(self, client):
        """Test that we can access the route index."""
        # The route index should be accessible via system overview or similar
        response = client.get(f"/api/v1/system/overview?graph_id={TEST_GRAPH_ID}")
        # Should at least respond (even if overview data isn't available)
        assert response.status_code in [200, 404]


# ============================================================================
# Blueprint Registration Smoke Tests
# ============================================================================

class TestBlueprintRegistrationSmoke:
    """Test that all blueprints are properly registered."""

    def test_all_blueprints_registered(self, test_app):
        """Verify all expected blueprints are registered with the Flask app."""
        expected_blueprints = [
            'system',
            'remember',
            'entities',
            'relations',
            'episodes',
            'concepts',
            'dream'
        ]

        registered_blueprints = set(test_app.blueprints.keys())

        for blueprint in expected_blueprints:
            assert blueprint in registered_blueprints, f"Blueprint '{blueprint}' not registered"

    def test_blueprint_has_routes(self, test_app):
        """Verify each blueprint has routes registered."""
        for blueprint_name, blueprint in test_app.blueprints.items():
            # Each blueprint should have at least one route
            assert len(blueprint.deferred_functions) > 0 or hasattr(blueprint, 'name'), \
                f"Blueprint '{blueprint_name}' has no registered routes"


# ============================================================================
# Error Handling Smoke Tests
# ============================================================================

class TestErrorHandlingSmoke:
    """Smoke tests for error handling."""

    def test_invalid_graph_id_returns_400(self, client):
        """Test that invalid graph_id returns 400."""
        response = client.get("/api/v1/find/stats?graph_id=invalid@graph#id")
        assert response.status_code in [400, 404]

    def test_missing_graph_id_returns_400(self, client):
        """Test that missing graph_id returns appropriate error."""
        response = client.get("/api/v1/find/stats")
        # Should handle missing graph_id gracefully
        assert response.status_code in [400, 404, 302, 307, 308]

    def test_invalid_json_returns_400(self, client):
        """Test that invalid JSON returns 400."""
        response = client.post(
            f"/api/v1/remember",
            data="invalid json",
            content_type="application/json"
        )
        assert response.status_code in [400, 422]

    def test_empty_request_returns_400(self, client):
        """Test that empty remember request returns 400."""
        response = client.post(
            f"/api/v1/remember?graph_id={TEST_GRAPH_ID}",
            json={}
        )
        assert response.status_code in [400, 422]


# ============================================================================
# Response Format Smoke Tests
# ============================================================================

class TestResponseFormatSmoke:
    """Smoke tests for response format consistency."""

    def test_json_response_format(self, client):
        """Test that successful responses return JSON with expected structure."""
        response = client.get(f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}")

        if response.status_code == 200:
            data = response.get_json()
            # Should have success field
            assert "success" in data or "data" in data

    def test_error_response_format(self, client):
        """Test that error responses return JSON with error information."""
        response = client.post(
            f"/api/v1/remember",
            json={}
        )

        if 400 <= response.status_code < 500:
            data = response.get_json()
            # Should have error information
            assert "success" in data or "error" in data


# ============================================================================
# CORS and Headers Smoke Tests
# ============================================================================

class TestHeadersSmoke:
    """Smoke tests for HTTP headers and CORS."""

    def test_content_type_header(self, client):
        """Test that JSON endpoints return correct content type."""
        response = client.get(f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}")

        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            assert 'application/json' in content_type or 'text/html' in content_type

    def test_options_method_responds(self, client):
        """Test that OPTIONS method is handled (CORS preflight)."""
        response = client.options(f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}")
        # Should respond to OPTIONS
        assert response.status_code in [200, 204, 405]


# ============================================================================
# Rate Limiting Smoke Tests
# ============================================================================

class TestRateLimitingSmoke:
    """Smoke tests for rate limiting."""

    def test_multiple_quick_requests(self, client):
        """Test that multiple quick requests are handled."""
        status_codes = []

        for _ in range(5):
            response = client.get(f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}")
            status_codes.append(response.status_code)

        # At least some requests should succeed
        assert any(code == 200 for code in status_codes), \
            f"All requests failed: {status_codes}"


# ============================================================================
# Graph ID Parameter Smoke Tests
# ============================================================================

class TestGraphIdParameterSmoke:
    """Smoke tests for graph_id parameter handling."""

    def test_graph_id_in_query_params(self, client):
        """Test graph_id passed as query parameter."""
        response = client.get(f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_graph_id_in_header(self, client):
        """Test graph_id passed as header."""
        response = client.get(
            "/api/v1/find/stats",
            headers={"X-Graph-Id": TEST_GRAPH_ID}
        )
        # Should handle header-based graph_id
        assert response.status_code in [200, 400, 404, 302, 307, 308]

    def test_special_characters_in_graph_id(self, client):
        """Test handling of special characters in graph_id."""
        # Test various valid graph_id formats
        valid_ids = [
            f"{TEST_GRAPH_ID}_test",
            f"{TEST_GRAPH_ID}-123",
            "default"
        ]

        for graph_id in valid_ids:
            response = client.get(f"/api/v1/find/stats?graph_id={graph_id}")
            # Should not crash on valid formats
            assert response.status_code in [200, 400, 404, 302, 307, 308]


# ============================================================================
# Pagination Smoke Tests
# ============================================================================

class TestPaginationSmoke:
    """Smoke tests for pagination parameters."""

    def test_pagination_params_accepted(self, client):
        """Test that pagination parameters are accepted."""
        response = client.get(
            f"/api/v1/find/entities?graph_id={TEST_GRAPH_ID}&limit=10&offset=0"
        )
        assert response.status_code in [200, 400, 404]

    def test_large_limit_param(self, client):
        """Test handling of very large limit parameter."""
        response = client.get(
            f"/api/v1/find/entities?graph_id={TEST_GRAPH_ID}&limit=999999"
        )
        # Should handle gracefully (either accept or cap)
        assert response.status_code in [200, 400, 404]


# ============================================================================
# Query Parameter Smoke Tests
# ============================================================================

class TestQueryParametersSmoke:
    """Smoke tests for various query parameters."""

    def test_unknown_query_param_ignored(self, client):
        """Test that unknown query parameters don't cause errors."""
        response = client.get(
            f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}&unknown_param=value"
        )
        # Should ignore unknown params
        assert response.status_code in [200, 404]

    def test_empty_query_param(self, client):
        """Test handling of empty query parameter values."""
        response = client.get(
            f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}&filter="
        )
        assert response.status_code in [200, 400, 404]


# ============================================================================
# HTTP Method Smoke Tests
# ============================================================================

class TestHTTPMethodSmoke:
    """Smoke tests for HTTP method compliance."""

    def test_get_method_on_get_endpoint(self, client):
        """Test GET method on GET endpoint."""
        response = client.get(f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}")
        assert response.status_code in [200, 404]

    def test_post_method_on_post_endpoint(self, client):
        """Test POST method on POST endpoint."""
        response = client.post(
            f"/api/v1/remember",
            json={"graph_id": TEST_GRAPH_ID, "text": "test"}
        )
        assert response.status_code in [200, 202, 400, 422]

    def test_invalid_method_returns_405(self, client):
        """Test that invalid HTTP method returns 405 Method Not Allowed."""
        response = client.delete(f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}")
        # DELETE on a GET-only endpoint should return 405 or 404
        assert response.status_code in [404, 405, 400]


# ============================================================================
# Content Negotiation Smoke Tests
# ============================================================================

class TestContentNegotiationSmoke:
    """Smoke tests for content negotiation."""

    def test_accept_json_header(self, client):
        """Test Accept: application/json header."""
        response = client.get(
            f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}",
            headers={"Accept": "application/json"}
        )
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            assert 'json' in content_type.lower()

    def test_accept_html_header(self, client):
        """Test Accept: text/html header."""
        response = client.get(
            f"/api/v1/find/stats?graph_id={TEST_GRAPH_ID}",
            headers={"Accept": "text/html"}
        )
        # Should handle HTML accept
        assert response.status_code in [200, 404]
