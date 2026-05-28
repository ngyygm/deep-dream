from core.server.api import create_app
from core.server.config import DEFAULTS
from core.server.registry import GraphRegistry


def _app(tmp_path):
    config = {
        **DEFAULTS,
        "storage_path": str(tmp_path / "library"),
        "storage": {"backend": "sqlite", "vector_dim": 8},
        "llm": {"api_key": "test", "base_url": "http://localhost:1/v1", "model": "test"},
        "embedding": {"model": None, "device": "cpu"},
    }
    return create_app(GraphRegistry(config["storage_path"], config), config)


def test_removed_legacy_route_groups_are_not_registered(tmp_path):
    app = _app(tmp_path)
    routes = {rule.rule for rule in app.url_map.iter_rules()}

    assert "/api/v1/graphs" not in routes
    assert "/api/v1/graphs/<graph_id>" not in routes
    assert "/api/v1/graphs/<graph_id>/clear" not in routes
    assert "/api/v1/communities" not in routes
    assert "/api/v1/communities/detect" not in routes
    assert "/api/v1/chat/sessions" not in routes
    assert "/api/v1/documents/<document_version_id>/file" not in routes
    assert "/api/v1/graph/stats" not in routes


def test_document_content_route_is_registered_once(tmp_path):
    app = _app(tmp_path)
    matches = [
        rule for rule in app.url_map.iter_rules()
        if rule.rule == "/api/v1/documents/<document_version_id>/content"
    ]
    assert len(matches) == 1
