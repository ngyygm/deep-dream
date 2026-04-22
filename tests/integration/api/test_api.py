"""
Comprehensive pytest tests for the Flask API server (server/api.py).

Covers:
  1. App creation
  2. Graph management (create, list, get info)
  3. Entity endpoints
  4. Relation endpoints
  5. Memory cache endpoints
  6. Remember endpoint (async task submission)
  7. Error handling (missing params, invalid graph_id, bad time format)
  8. Statistics endpoint
  9. CORS headers
  10. Rate limiting
  11. API routes index
  12. Health check
  13. System endpoints
  14. Docs listing
  15. API redirect (/api/... -> /api/v1/...)

Usage:
    pytest tests/test_api.py -v
"""
from __future__ import annotations

import json
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from server.task_queue import RememberTask

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_registry(storage_path: str):
    """Build a GraphRegistry with a minimal config that avoids needing a
    real LLM API key or embedding model."""
    from server.registry import GraphRegistry

    config: Dict[str, Any] = {
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
        "remember_workers": 0,       # no worker threads; tasks stay queued
        "remember_max_retries": 0,
        "remember_retry_delay_seconds": 0,
    }
    return GraphRegistry(base_storage_path=storage_path, config=config)


@pytest.fixture()
def storage_path(tmp_path: Path) -> str:
    """Return a temporary directory string for graph storage."""
    return str(tmp_path / "graphs")


@pytest.fixture()
def registry(storage_path: str):
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
def graph_id() -> str:
    """A well-known test graph id."""
    return "testgraph"


@pytest.fixture()
def created_graph(client, graph_id) -> str:
    """Ensure a graph is created before the test runs; return its id."""
    resp = client.post(
        "/api/v1/graphs",
        data=json.dumps({"graph_id": graph_id}),
        content_type="application/json",
    )
    assert resp.status_code in (200, 409), f"Failed to create graph: {resp.data}"
    return graph_id


# ---------------------------------------------------------------------------
# 1. App creation
# ---------------------------------------------------------------------------


class TestAppCreation:
    def test_create_app_returns_flask_app(self, registry):
        from server.api import create_app

        from flask import Flask

        application = create_app(registry=registry)
        assert isinstance(application, Flask)

    def test_app_has_testing_mode(self, app):
        assert app.config["TESTING"] is True


# ---------------------------------------------------------------------------
# 2. Graph management
# ---------------------------------------------------------------------------


class TestGraphManagement:
    def test_create_graph(self, client):
        resp = client.post(
            "/api/v1/graphs",
            data=json.dumps({"graph_id": "mygraph"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["graph_id"] == "mygraph"

    def test_create_graph_duplicate(self, client):
        client.post(
            "/api/v1/graphs",
            data=json.dumps({"graph_id": "dup"}),
            content_type="application/json",
        )
        resp = client.post(
            "/api/v1/graphs",
            data=json.dumps({"graph_id": "dup"}),
            content_type="application/json",
        )
        assert resp.status_code == 409
        data = resp.get_json()
        assert data["success"] is False

    def test_create_graph_empty_id(self, client):
        resp = client.post(
            "/api/v1/graphs",
            data=json.dumps({"graph_id": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_create_graph_invalid_id(self, client):
        resp = client.post(
            "/api/v1/graphs",
            data=json.dumps({"graph_id": "../bad"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_list_graphs_empty(self, client):
        resp = client.get("/api/v1/graphs")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["data"]["graphs"], list)

    def test_list_graphs_after_create(self, client):
        client.post(
            "/api/v1/graphs",
            data=json.dumps({"graph_id": "g1"}),
            content_type="application/json",
        )
        resp = client.get("/api/v1/graphs")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "g1" in data["data"]["graphs"]


# ---------------------------------------------------------------------------
# 3. Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_health(self, client, created_graph):
        resp = client.get(f"/api/v1/health?graph_id={created_graph}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["graph_id"] == created_graph

    def test_health_default_graph(self, client):
        """Without graph_id the default graph is lazily created."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["data"]["graph_id"] == "default"


# ---------------------------------------------------------------------------
# 4. Entity endpoints
# ---------------------------------------------------------------------------


class TestEntityEndpoints:
    def test_get_entities_empty(self, client, created_graph):
        resp = client.get(f"/api/v1/find/entities?graph_id={created_graph}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["data"]["entities"], list)

    def test_get_entity_by_family_id_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/nonexistent?graph_id={created_graph}"
        )
        assert resp.status_code == 404

    def test_get_entity_by_absolute_id_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/absolute/abs-nonexistent?graph_id={created_graph}"
        )
        assert resp.status_code == 404

    def test_search_entities_missing_query(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/search?graph_id={created_graph}"
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_search_entities_post(self, client, created_graph):
        resp = client.post(
            f"/api/v1/find/entities/search?graph_id={created_graph}",
            data=json.dumps({"query_name": "test"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_get_entity_versions_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/noid/versions?graph_id={created_graph}"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["data"], list)

    def test_get_entity_version_count_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/noid/version-count?graph_id={created_graph}"
        )
        assert resp.status_code == 404

    def test_entities_as_of_time_missing_param(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/as-of-time?graph_id={created_graph}"
        )
        assert resp.status_code == 400

    def test_entities_as_of_time_invalid_time(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/as-of-time?graph_id={created_graph}&time_point=not-a-time"
        )
        assert resp.status_code == 400

    def test_entity_at_time_missing_param(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/someid/as-of-time?graph_id={created_graph}"
        )
        assert resp.status_code == 400

    def test_entity_nearest_to_time_missing_param(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/someid/nearest-to-time?graph_id={created_graph}"
        )
        assert resp.status_code == 400

    def test_entity_around_time_missing_params(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/someid/around-time?graph_id={created_graph}"
        )
        assert resp.status_code == 400

    def test_entity_around_time_missing_within_seconds(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/someid/around-time?graph_id={created_graph}"
            "&time_point=2025-01-01T00:00:00"
        )
        assert resp.status_code == 400

    def test_entity_embedding_preview_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/absolute/abs-no/embedding-preview?graph_id={created_graph}"
        )
        assert resp.status_code == 404

    def test_entity_version_counts_invalid_body(self, client, created_graph):
        """空 body 返回空结果而非 400，服务器对缺失 family_ids 宽容处理。"""
        resp = client.post(
            f"/api/v1/find/entities/version-counts?graph_id={created_graph}",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"] == {}


# ---------------------------------------------------------------------------
# 5. Relation endpoints
# ---------------------------------------------------------------------------


class TestRelationEndpoints:
    def test_get_relations_empty(self, client, created_graph):
        resp = client.get(f"/api/v1/find/relations?graph_id={created_graph}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["data"]["relations"], list)

    def test_get_relation_by_absolute_id_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/relations/absolute/abs-nope?graph_id={created_graph}"
        )
        assert resp.status_code == 404

    def test_get_relations_by_entity_absolute_id(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/absolute/abs-no/relations?graph_id={created_graph}"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_get_relations_by_family_id(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/noid/relations?graph_id={created_graph}"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_search_relations_missing_query(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/relations/search?graph_id={created_graph}"
        )
        assert resp.status_code == 400

    def test_search_relations_post(self, client, created_graph):
        resp = client.post(
            f"/api/v1/find/relations/search?graph_id={created_graph}",
            data=json.dumps({"query_text": "test"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_relations_between_missing_params(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/relations/between?graph_id={created_graph}"
        )
        assert resp.status_code == 400

    def test_relations_between_post(self, client, created_graph):
        resp = client.post(
            f"/api/v1/find/relations/between?graph_id={created_graph}",
            data=json.dumps({"family_id_a": "a", "family_id_b": "b"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_relation_versions_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/relations/noid/versions?graph_id={created_graph}"
        )
        assert resp.status_code == 200

    def test_relation_embedding_preview_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/relations/absolute/abs-no/embedding-preview?graph_id={created_graph}"
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 6. Memory cache endpoints
# ---------------------------------------------------------------------------


class TestEpisodeEndpoints:
    def test_get_latest_memory_cache_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/episodes/latest?graph_id={created_graph}"
        )
        assert resp.status_code == 404

    def test_get_latest_memory_cache_metadata_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/episodes/latest/metadata?graph_id={created_graph}"
        )
        assert resp.status_code == 404

    def test_get_memory_cache_by_id_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/episodes/no-such-cache?graph_id={created_graph}"
        )
        assert resp.status_code == 404

    def test_get_episode_text_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/episodes/no-such-cache/text?graph_id={created_graph}"
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 7. Remember endpoint
# ---------------------------------------------------------------------------


class TestRememberEndpoint:
    def test_remember_missing_text(self, client, created_graph):
        resp = client.post(
            f"/api/v1/remember?graph_id={created_graph}",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_remember_queues_task(self, client, created_graph):
        resp = client.post(
            f"/api/v1/remember?graph_id={created_graph}",
            data=json.dumps({"text": "Hello world"}),
            content_type="application/json",
        )
        assert resp.status_code == 202
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["task_id"]
        assert data["data"]["status"] == "queued"

    def test_remember_task_status_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/remember/tasks/nonexistent-task?graph_id={created_graph}"
        )
        assert resp.status_code == 404

    def test_remember_task_status_exists(self, client, created_graph):
        # Submit a task
        resp = client.post(
            f"/api/v1/remember?graph_id={created_graph}",
            data=json.dumps({"text": "Some text to remember"}),
            content_type="application/json",
        )
        task_id = resp.get_json()["data"]["task_id"]

        # Query its status
        resp = client.get(
            f"/api/v1/remember/tasks/{task_id}?graph_id={created_graph}"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["task_id"] == task_id
        assert data["data"]["status"] in ("queued", "running", "completed", "failed")

    def test_remember_omitted_load_cache_uses_config_default(self, client, created_graph):
        resp = client.post(
            f"/api/v1/remember?graph_id={created_graph}",
            data=json.dumps({"text": "Hello world"}),
            content_type="application/json",
        )
        assert resp.status_code == 202
        task_id = resp.get_json()["data"]["task_id"]

        status = client.get(f"/api/v1/remember/tasks/{task_id}?graph_id={created_graph}")
        assert status.status_code == 200
        data = status.get_json()
        assert data["success"] is True
        assert data["data"]["load_cache_memory"] is False

    def test_remember_explicit_load_cache_is_persisted(self, client, created_graph):
        resp = client.post(
            f"/api/v1/remember?graph_id={created_graph}",
            data=json.dumps({"text": "Hello world", "load_cache_memory": True}),
            content_type="application/json",
        )
        assert resp.status_code == 202
        task_id = resp.get_json()["data"]["task_id"]

        status = client.get(f"/api/v1/remember/tasks/{task_id}?graph_id={created_graph}")
        assert status.status_code == 200
        data = status.get_json()
        assert data["success"] is True
        assert data["data"]["load_cache_memory"] is True

    def test_remember_file_upload_parses_load_cache_false(self, client, created_graph):
        resp = client.post(
            f"/api/v1/remember?graph_id={created_graph}",
            data={
                "file": (BytesIO(b"file upload content"), "upload.txt"),
                "load_cache_memory": "false",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 202
        task_id = resp.get_json()["data"]["task_id"]

        status = client.get(f"/api/v1/remember/tasks/{task_id}?graph_id={created_graph}")
        assert status.status_code == 200
        data = status.get_json()
        assert data["success"] is True
        assert data["data"]["load_cache_memory"] is False

    def test_remember_delete_queued_task(self, client, created_graph):
        resp = client.post(
            f"/api/v1/remember?graph_id={created_graph}",
            data=json.dumps({"text": "Delete me"}),
            content_type="application/json",
        )
        assert resp.status_code == 202
        task_id = resp.get_json()["data"]["task_id"]

        deleted = client.delete(f"/api/v1/remember/tasks/{task_id}?graph_id={created_graph}")
        assert deleted.status_code == 200
        payload = deleted.get_json()
        assert payload["success"] is True
        assert payload["data"]["status"] == "deleted"

        status = client.get(f"/api/v1/remember/tasks/{task_id}?graph_id={created_graph}")
        assert status.status_code == 404

    def test_remember_delete_missing_task(self, client, created_graph):
        resp = client.delete(
            f"/api/v1/remember/tasks/nonexistent-task?graph_id={created_graph}"
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False

    def test_remember_pause_running_task(self, client, registry, created_graph):
        resp = client.post(
            f"/api/v1/remember?graph_id={created_graph}",
            data=json.dumps({"text": "Pause me"}),
            content_type="application/json",
        )
        assert resp.status_code == 202
        task_id = resp.get_json()["data"]["task_id"]
        queue = registry.get_queue(created_graph)
        task = queue.get_status(task_id)
        assert task is not None
        task.status = "running"

        paused = client.post(f"/api/v1/remember/tasks/{task_id}/pause?graph_id={created_graph}")
        assert paused.status_code == 200
        data = paused.get_json()
        assert data["success"] is True
        assert data["data"]["status"] == "pausing"

    def test_remember_resume_paused_task(self, client, registry, created_graph):
        queue = registry.get_queue(created_graph)
        task = RememberTask(
            task_id="paused-task",
            text="Resume me",
            source_name="resume.txt",
            load_cache=False,
            control_action=None,
            event_time=None,
            original_path="",
            status="paused",
            phase="paused",
            phase_label="已暂停",
            message="任务已暂停，可继续",
        )
        queue._tasks[task.task_id] = task

        resumed = client.post(f"/api/v1/remember/tasks/{task.task_id}/resume?graph_id={created_graph}")
        assert resumed.status_code == 200
        data = resumed.get_json()
        assert data["success"] is True
        assert data["data"]["status"] == "queued"

    def test_remember_task_list(self, client, created_graph):
        resp = client.get(
            f"/api/v1/remember/tasks?graph_id={created_graph}"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["data"]["tasks"], list)

    def test_remember_monitor(self, client, created_graph):
        resp = client.get(
            f"/api/v1/remember/monitor?graph_id={created_graph}"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_remember_with_event_time(self, client, created_graph):
        resp = client.post(
            f"/api/v1/remember?graph_id={created_graph}",
            data=json.dumps({
                "text": "Something happened",
                "event_time": "2025-06-15T10:30:00",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 202
        data = resp.get_json()
        assert data["success"] is True

    def test_remember_invalid_event_time(self, client, created_graph):
        resp = client.post(
            f"/api/v1/remember?graph_id={created_graph}",
            data=json.dumps({
                "text": "Bad time",
                "event_time": "not-a-date",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False


# ---------------------------------------------------------------------------
# 8. Statistics endpoint
# ---------------------------------------------------------------------------


class TestStatisticsEndpoint:
    def test_stats_empty_graph(self, client, created_graph):
        resp = client.get(f"/api/v1/find/stats?graph_id={created_graph}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["total_entities"] == 0
        assert data["data"]["total_relations"] == 0

    def test_stats_includes_memory_cache_count(self, client, created_graph):
        resp = client.get(f"/api/v1/find/stats?graph_id={created_graph}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_episodes" in data["data"]


# ---------------------------------------------------------------------------
# 9. Find unified search
# ---------------------------------------------------------------------------


class TestFindUnified:
    def test_find_missing_query(self, client, created_graph):
        resp = client.post(
            f"/api/v1/find?graph_id={created_graph}",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_find_with_query(self, client, created_graph):
        resp = client.post(
            f"/api/v1/find?graph_id={created_graph}",
            data=json.dumps({"query": "test query"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "entities" in data["data"]
        assert "relations" in data["data"]

    def test_find_with_time_filters(self, client, created_graph):
        resp = client.post(
            f"/api/v1/find?graph_id={created_graph}",
            data=json.dumps({
                "query": "test",
                "time_before": "2026-01-01T00:00:00",
                "time_after": "2024-01-01T00:00:00",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_find_invalid_time_format(self, client, created_graph):
        resp = client.post(
            f"/api/v1/find?graph_id={created_graph}",
            data=json.dumps({
                "query": "test",
                "time_before": "not-a-date",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_find_candidates(self, client, created_graph):
        resp = client.post(
            f"/api/v1/find/candidates?graph_id={created_graph}",
            data=json.dumps({"query_text": "test"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "entities" in data["data"]
        assert "relations" in data["data"]

    def test_find_candidates_invalid_time(self, client, created_graph):
        resp = client.post(
            f"/api/v1/find/candidates?graph_id={created_graph}",
            data=json.dumps({
                "query_text": "test",
                "time_before": "garbage",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 10. CORS headers
# ---------------------------------------------------------------------------


class TestCORSHeaders:
    def test_cors_headers_with_localhost_origin(self, client, created_graph):
        resp = client.get(
            f"/api/v1/health?graph_id={created_graph}",
            headers={"Origin": "http://localhost"},
        )
        assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost"
        assert "GET" in resp.headers.get("Access-Control-Allow-Methods", "")

    def test_cors_headers_with_127001_origin(self, client, created_graph):
        resp = client.get(
            f"/api/v1/health?graph_id={created_graph}",
            headers={"Origin": "http://127.0.0.1"},
        )
        assert resp.headers.get("Access-Control-Allow-Origin") == "http://127.0.0.1"

    def test_cors_headers_with_disallowed_origin(self, client, created_graph):
        resp = client.get(
            f"/api/v1/health?graph_id={created_graph}",
            headers={"Origin": "http://evil.example.com"},
        )
        # Non-allowed origins should get an empty ACAO header
        assert resp.headers.get("Access-Control-Allow-Origin") == ""

    def test_cors_preflight(self, client):
        resp = client.options(
            "/api/v1/graphs",
            headers={"Origin": "http://localhost"},
        )
        assert resp.status_code == 204


# ---------------------------------------------------------------------------
# 11. Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_many_rapid_requests_succeed(self, client):
        """Under the default limit (600/min), a burst of 20 should all pass."""
        for _ in range(20):
            resp = client.get("/api/v1/graphs")
            assert resp.status_code == 200

    def test_rate_limit_triggered(self, app, registry, storage_path):
        """If rate limit is set very low, 429 should appear after the limit."""
        from server.api import create_app

        low_limit_app = create_app(
            registry=registry,
            config={"rate_limit_per_minute": 5},
        )
        low_limit_app.config["TESTING"] = True
        cl = low_limit_app.test_client()

        statuses = []
        for _ in range(10):
            resp = cl.get("/api/v1/graphs")
            statuses.append(resp.status_code)

        assert 429 in statuses, f"Expected at least one 429, got statuses: {statuses}"

    def test_rate_limit_zero_disables(self, app, registry, storage_path):
        """Setting rate_limit_per_minute to 0 disables rate limiting."""
        from server.api import create_app

        no_limit_app = create_app(
            registry=registry,
            config={"rate_limit_per_minute": 0},
        )
        no_limit_app.config["TESTING"] = True
        cl = no_limit_app.test_client()

        for _ in range(20):
            resp = cl.get("/api/v1/graphs")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 12. API routes index
# ---------------------------------------------------------------------------


class TestAPIRoutesIndex:
    def test_routes_endpoint(self, client):
        resp = client.get("/api/v1/routes")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        # Verify the major sections are present
        for section in ("health", "remember", "find", "entity", "relation", "episode", "system"):
            assert section in data["data"], f"Missing section: {section}"


# ---------------------------------------------------------------------------
# 13. Error handling - invalid graph_id
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_invalid_graph_id_special_chars(self, client):
        """graph_id with path traversal should be rejected."""
        resp = client.get("/api/v1/health?graph_id=../etc/passwd")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_invalid_graph_id_spaces(self, client):
        """graph_id with only spaces falls back to 'default'."""
        resp = client.get("/api/v1/health?graph_id=%20%20")
        # Spaces-only graph_id is stripped to empty, which triggers the
        # default fallback rather than a validation error.
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["data"]["graph_id"] == "default"

    def test_missing_graph_id_defaults_to_default(self, client):
        """When graph_id is not supplied, the server uses 'default'."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["data"]["graph_id"] == "default"


# ---------------------------------------------------------------------------
# 14. System endpoints
# ---------------------------------------------------------------------------


class TestSystemEndpoints:
    def test_system_overview_no_monitor(self, client):
        """Without a SystemMonitor, should return 503."""
        resp = client.get("/api/v1/system/overview")
        assert resp.status_code == 503

    def test_system_graphs_no_monitor(self, client):
        resp = client.get("/api/v1/system/graphs")
        assert resp.status_code == 503

    def test_system_tasks_no_monitor(self, client):
        resp = client.get("/api/v1/system/tasks")
        assert resp.status_code == 503

    def test_system_logs_no_monitor(self, client):
        resp = client.get("/api/v1/system/logs")
        assert resp.status_code == 503

    def test_system_access_stats_no_monitor(self, client):
        resp = client.get("/api/v1/system/access-stats")
        assert resp.status_code == 503

    def test_system_dashboard_no_monitor(self, client):
        resp = client.get("/api/v1/system/dashboard")
        assert resp.status_code == 503

    def test_system_graph_detail_no_monitor(self, client):
        resp = client.get("/api/v1/system/graphs/somegraph")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# 15. Docs endpoint
# ---------------------------------------------------------------------------


class TestDocsEndpoint:
    def test_list_docs_empty(self, client, created_graph):
        resp = client.get(f"/api/v1/docs?graph_id={created_graph}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["data"]["docs"], list)


# ---------------------------------------------------------------------------
# 16. API redirect (/api/... -> /api/v1/...)
# ---------------------------------------------------------------------------


class TestAPIRedirect:
    def test_api_root_redirect(self, client):
        resp = client.get("/api")
        # 308 permanent redirect
        assert resp.status_code == 308

    def test_api_subpath_redirect(self, client):
        resp = client.get("/api/graphs")
        assert resp.status_code == 308


# ---------------------------------------------------------------------------
# 17. graph_id resolution from different sources
# ---------------------------------------------------------------------------


class TestGraphIdResolution:
    def test_graph_id_from_query_string(self, client):
        resp = client.get("/api/v1/health?graph_id=idfromquery")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["data"]["graph_id"] == "idfromquery"

    def test_graph_id_from_post_json_body(self, client):
        resp = client.post(
            "/api/v1/remember",
            data=json.dumps({"text": "test", "graph_id": "idfrombody"}),
            content_type="application/json",
        )
        # Should succeed in queueing (202), not fail on graph_id
        assert resp.status_code == 202

    def test_graph_id_from_form_data(self, client):
        resp = client.post(
            "/api/v1/remember",
            data={"text": "test", "graph_id": "idfromform"},
        )
        assert resp.status_code == 202

    def test_post_json_body_overrides_query(self, client):
        """When graph_id is in both JSON body and query param, JSON body wins."""
        resp = client.post(
            "/api/v1/remember?graph_id=fromquery",
            data=json.dumps({"text": "test", "graph_id": "frombody"}),
            content_type="application/json",
        )
        assert resp.status_code == 202
        data = resp.get_json()
        assert data["success"] is True


# ---------------------------------------------------------------------------
# 18. Response format consistency
# ---------------------------------------------------------------------------


class TestResponseFormat:
    def test_success_response_has_success_true(self, client, created_graph):
        resp = client.get(f"/api/v1/health?graph_id={created_graph}")
        data = resp.get_json()
        assert "success" in data
        assert data["success"] is True
        assert "data" in data

    def test_error_response_has_success_false(self, client):
        resp = client.get("/api/v1/health?graph_id=../bad")
        data = resp.get_json()
        assert "success" in data
        assert data["success"] is False
        assert "error" in data

    def test_success_response_includes_elapsed_ms(self, client, created_graph):
        resp = client.get(f"/api/v1/health?graph_id={created_graph}")
        data = resp.get_json()
        assert "elapsed_ms" in data
        assert isinstance(data["elapsed_ms"], (int, float))


# ---------------------------------------------------------------------------
# 19. Entity temporal query edge cases
# ---------------------------------------------------------------------------


class TestEntityTemporalEdgeCases:
    def test_entity_as_of_time_valid_format_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/someid/as-of-time?graph_id={created_graph}"
            "&time_point=2025-06-01T12:00:00"
        )
        assert resp.status_code == 404

    def test_entity_nearest_to_time_valid_format_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/someid/nearest-to-time?graph_id={created_graph}"
            "&time_point=2025-06-01T12:00:00"
        )
        assert resp.status_code == 404

    def test_entity_around_time_valid_format_not_found(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/someid/around-time?graph_id={created_graph}"
            "&time_point=2025-06-01T12:00:00&within_seconds=3600"
        )
        assert resp.status_code == 404

    def test_entity_nearest_to_time_with_max_delta(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/someid/nearest-to-time?graph_id={created_graph}"
            "&time_point=2025-06-01T12:00:00&max_delta_seconds=10"
        )
        assert resp.status_code == 404

    def test_entity_nearest_to_time_invalid_max_delta(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/someid/nearest-to-time?graph_id={created_graph}"
            "&time_point=2025-06-01T12:00:00&max_delta_seconds=notanumber"
        )
        assert resp.status_code == 400

    def test_entity_as_of_time_invalid_format(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/someid/as-of-time?graph_id={created_graph}"
            "&time_point=garbage"
        )
        assert resp.status_code == 400

    def test_entity_relations_with_time_point(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/someid/relations?graph_id={created_graph}"
            "&time_point=2025-01-01T00:00:00"
        )
        assert resp.status_code == 200

    def test_entity_relations_invalid_time_point(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/entities/someid/relations?graph_id={created_graph}"
            "&time_point=bad-time"
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 20. Relations between with various param names
# ---------------------------------------------------------------------------


class TestRelationsBetweenParams:
    def test_relations_between_get(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/relations/between?graph_id={created_graph}"
            "&family_id_a=a&family_id_b=b"
        )
        assert resp.status_code == 200

    def test_relations_between_only_one_param(self, client, created_graph):
        resp = client.get(
            f"/api/v1/find/relations/between?graph_id={created_graph}"
            "&family_id_a=a"
        )
        assert resp.status_code == 400

    def test_relations_between_post_with_from_to(self, client, created_graph):
        """Test the alternate param names from_family_id / to_family_id."""
        resp = client.post(
            f"/api/v1/find/relations/between?graph_id={created_graph}",
            data=json.dumps({"from_family_id": "a", "to_family_id": "b"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
