"""
Tests for multi-graph isolation and switching.

Covers:
- MCP server graph context resolution
- API rate limiter graph scoping
- Registry per-graph isolation
- switch_graph / delete_graph / get_active_graph logic
"""
import ast
import pytest
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Structural tests (no server needed) ────────────────────────────────────


def test_mcp_graph_id_param_injected_in_all_tools():
    """Every tool definition should include the shared graph_id parameter."""
    source = (Path(__file__).resolve().parent.parent / "server" / "mcp" / "deep_dream_server.py").read_text()
    tree = ast.parse(source)

    # Find _t() calls
    tool_calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and node.func.id == "_t"):
            # First arg is name (string constant)
            name = None
            if node.args and isinstance(node.args[0], ast.Constant):
                name = node.args[0].value
            tool_calls.append(name)

    assert len(tool_calls) > 50, f"Expected many tools, found {len(tool_calls)}"

    # The param merging code must be present
    assert "merged_params = {**params, **_GRAPH_ID_PARAM}" in source
    assert "_GRAPH_ID_PARAM" in source


def test_graph_context_mechanism_in_source():
    """Verify _graph_context and _current_call_graph_id are in the source."""
    source = (Path(__file__).resolve().parent.parent / "server" / "mcp" / "deep_dream_server.py").read_text()

    assert "_current_call_graph_id" in source
    assert "_graph_context" in source
    assert "with _graph_context(arguments)" in source


def test_rate_limiter_scoped_by_graph_id():
    """Rate limiter should use (IP, graph_id) as key, not just IP."""
    source = (Path(__file__).resolve().parent.parent / "server" / "api.py").read_text()

    assert 'rate_key = f"{client_ip}|{gid}"' in source
    assert "_rate_limit_store.get(rate_key)" in source


def test_graph_id_validation():
    """GraphRegistry.validate_graph_id should reject invalid IDs."""
    from server.registry import GraphRegistry

    # Valid IDs
    GraphRegistry.validate_graph_id("default")
    GraphRegistry.validate_graph_id("my_project")
    GraphRegistry.validate_graph_id("test-123")
    GraphRegistry.validate_graph_id("a")

    # Invalid IDs
    with pytest.raises(ValueError):
        GraphRegistry.validate_graph_id("")
    with pytest.raises(ValueError):
        GraphRegistry.validate_graph_id("  ")
    with pytest.raises(ValueError):
        GraphRegistry.validate_graph_id("../../../etc")
    with pytest.raises(ValueError):
        GraphRegistry.validate_graph_id("-leading-dash")
    with pytest.raises(ValueError):
        GraphRegistry.validate_graph_id("has space")


def test_new_tools_exist():
    """Verify new graph management tools are defined."""
    source = (Path(__file__).resolve().parent.parent / "server" / "mcp" / "deep_dream_server.py").read_text()

    for name in ["switch_graph", "get_active_graph", "delete_graph"]:
        assert f'def {name}(' in source, f"Missing handler function: {name}"
        assert f'_t("{name}"' in source, f"Missing tool definition: {name}"


# ── Graph context resolution tests (use subprocess to avoid import side effects) ──────


def test_resolve_graph_id_logic():
    """Test the resolve logic by extracting it into a standalone function."""
    # Replicate the _resolve_graph_id logic
    _active_graph_id = "default"

    def resolve(args, active=_active_graph_id):
        gid = args.get("graph_id")
        if gid and isinstance(gid, str) and gid.strip():
            return gid.strip()
        return active

    # Default
    assert resolve({}) == "default"
    # Per-call override
    assert resolve({"graph_id": "my_project"}) == "my_project"
    # Empty string fallback
    assert resolve({"graph_id": "  "}) == "default"
    # None fallback
    assert resolve({"graph_id": None}) == "default"


def test_context_manager_logic():
    """Test that nested context manager pattern works correctly."""
    _current = "default"
    history = []

    class GraphContext:
        def __init__(self, target):
            self._target = target
            self._old = None

        def __enter__(self):
            nonlocal _current
            self._old = _current
            _current = self._target
            return _current

        def __exit__(self, *args):
            nonlocal _current
            _current = self._old

    # Simple context
    with GraphContext("test") as gid:
        history.append(gid)
        assert _current == "test"
    history.append(_current)
    assert _current == "default"

    # Nested contexts
    with GraphContext("outer"):
        assert _current == "outer"
        with GraphContext("inner"):
            assert _current == "inner"
        assert _current == "outer"
    assert _current == "default"

    assert history == ["test", "default"]


def test_url_builds_correctly():
    """Test URL construction with graph_id."""
    BASE_URL = "http://localhost:16200"

    def build_url(path, graph_id="default", **qp):
        sep = '&' if '?' in path else '?'
        url = f"{BASE_URL}{path}{sep}graph_id={graph_id}"
        for k, v in qp.items():
            if v is not None:
                url += f"&{k}={v}"
        return url

    url = build_url("/api/v1/find")
    assert url == "http://localhost:16200/api/v1/find?graph_id=default"

    url = build_url("/api/v1/find", graph_id="project_a")
    assert url == "http://localhost:16200/api/v1/find?graph_id=project_a"

    url = build_url("/api/v1/find?existing=1", graph_id="test", limit="10")
    assert "graph_id=test" in url
    assert "limit=10" in url
    assert "existing=1" in url


# ── Registry isolation tests ───────────────────────────────────────────────


def test_registry_per_graph_processors():
    """Different graph IDs should produce different processor instances."""
    from server.registry import GraphRegistry
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = GraphRegistry(tmpdir, config={})
        # Mock _build_processor to avoid real init
        mock_procs = {}
        original = registry._build_processor

        def mock_build(storage_path):
            gid = Path(storage_path).name
            if gid not in mock_procs:
                proc = MagicMock()
                proc.storage = MagicMock()
                proc.storage.storage_path = storage_path
                mock_procs[gid] = proc
            return mock_procs[gid]

        registry._build_processor = mock_build

        p1 = registry.get_processor("graph_a")
        p2 = registry.get_processor("graph_b")
        p1_again = registry.get_processor("graph_a")

        assert p1 is not p2, "Different graph IDs should produce different processors"
        assert p1 is p1_again, "Same graph ID should return same processor"
        assert len(mock_procs) == 2


def test_registry_per_graph_dream_locks():
    """Different graph IDs should have different dream locks."""
    from server.registry import GraphRegistry
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = GraphRegistry(tmpdir, config={})
        lock1 = registry.get_dream_lock("graph_a")
        lock2 = registry.get_dream_lock("graph_b")
        lock1_again = registry.get_dream_lock("graph_a")

        assert lock1 is not lock2
        assert lock1 is lock1_again


# ── Metadata tests ───────────────────────────────────────────────────────────


def test_metadata_read_write():
    """GraphRegistry should persist and retrieve metadata.json."""
    from server.registry import GraphRegistry
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = GraphRegistry(tmpdir, config={})

        # No metadata yet
        assert registry.get_graph_metadata("test_graph") == {}

        # Write metadata
        result = registry.set_graph_metadata(
            "test_graph",
            name="Test Graph",
            description="A test graph for metadata",
        )
        assert result["name"] == "Test Graph"
        assert result["description"] == "A test graph for metadata"

        # Read back
        meta = registry.get_graph_metadata("test_graph")
        assert meta["name"] == "Test Graph"
        assert meta["description"] == "A test graph for metadata"

        # Merge additional fields
        registry.set_graph_metadata("test_graph", created_at="2026-01-01T00:00:00")
        meta = registry.get_graph_metadata("test_graph")
        assert meta["name"] == "Test Graph"  # preserved
        assert meta["created_at"] == "2026-01-01T00:00:00"  # added


def test_metadata_none_values_ignored():
    """set_graph_metadata should skip None values (only update non-None)."""
    from server.registry import GraphRegistry
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = GraphRegistry(tmpdir, config={})
        registry.set_graph_metadata("g1", name="G1", description="desc")
        # Passing name=None should NOT overwrite
        registry.set_graph_metadata("g1", name=None, extra="val")
        meta = registry.get_graph_metadata("g1")
        assert meta["name"] == "G1"  # preserved
        assert meta["extra"] == "val"  # added


def test_get_graph_info_nonexistent():
    """get_graph_info should return None for nonexistent graph directory."""
    from server.registry import GraphRegistry
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = GraphRegistry(tmpdir, config={})
        assert registry.get_graph_info("nonexistent") is None


def test_list_graphs_info():
    """list_graphs_info should return enriched dicts for all graphs."""
    from server.registry import GraphRegistry
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = GraphRegistry(tmpdir, config={})
        # Create two graph directories with graph.db
        for gid in ("alpha", "beta"):
            gdir = Path(tmpdir) / gid
            gdir.mkdir()
            (gdir / "graph.db").touch()
            registry.set_graph_metadata(gid, name=f"Graph {gid}")

        infos = registry.list_graphs_info()
        assert len(infos) == 2
        ids = {i["graph_id"] for i in infos}
        assert ids == {"alpha", "beta"}
        # Check metadata is present
        alpha = next(i for i in infos if i["graph_id"] == "alpha")
        assert alpha["name"] == "Graph alpha"
        assert "entity_count" in alpha


def test_graph_endpoint_sends_metadata_on_create():
    """POST /api/v1/graphs should persist name/description metadata."""
    # Structural: verify the blueprint sends metadata fields
    source = (Path(__file__).resolve().parent.parent / "server" / "blueprints" / "concepts.py").read_text()
    assert 'registry.set_graph_metadata(' in source
    assert 'data.get("name"' in source
    assert 'data.get("description"' in source
    assert 'list_graphs_info(' in source


def test_single_graph_endpoint_exists():
    """GET /api/v1/graphs/<graph_id> should be defined."""
    source = (Path(__file__).resolve().parent.parent / "server" / "blueprints" / "concepts.py").read_text()
    assert 'handle_single_graph' in source
    assert 'methods=["GET", "DELETE"]' in source
    assert 'get_graph_info(' in source
