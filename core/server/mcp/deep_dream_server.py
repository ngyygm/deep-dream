#!/usr/bin/env python3
"""
Deep Dream MCP Server — exposes all Deep Dream API endpoints as MCP tools.

Protocol: stdio with Content-Length / NDJSON auto-detection.
Upstream: Deep Dream REST API on localhost:16200.

This is the main entry point that assembles transport, tool schemas,
response formatting, and handler modules into a working MCP server.

Run as:
    python core/server/mcp/deep_dream_server.py
    python -m core.server.mcp.deep_dream_server
"""

import sys
import os

# When run directly as a script (not as a package), set up import paths
# so that `from core.server.mcp.X import ...` works correctly.
if __name__ == "__main__" and __package__ is None:
    # Walk up 4 levels from this file to reach the project root
    # core/server/mcp/deep_dream_server.py -> project root
    _here = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_here, '..', '..', '..'))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    __package__ = "core.server.mcp"

# Transport must be imported first — it reconfigures stdin/stdout to unbuffered binary
from .transport import debug_log, send_response, read_message, _graph_context, BASE_URL, _DEFAULT_GRAPH_ID
from .tool_schemas import TOOLS
from .response_format import _result

# Handler modules — each provides a register_handlers(tool_map) function
from .handlers_remember import register_handlers as _register_remember
from .handlers_entities import register_handlers as _register_entities
from .handlers_relations import register_handlers as _register_relations
from .handlers_episodes import register_handlers as _register_episodes
from .handlers_dream import register_handlers as _register_dream
from .handlers_graph import register_handlers as _register_graph
from .handlers_concepts import register_handlers as _register_concepts
from .handlers_misc import register_handlers as _register_misc


# ── Tool dispatch ─────────────────────────────────────────────────────────

_TOOL_MAP = {}


def _build_tool_map():
    """Populate _TOOL_MAP by calling each handler module's register_handlers."""
    _register_remember(_TOOL_MAP)
    _register_entities(_TOOL_MAP)
    _register_relations(_TOOL_MAP)
    _register_episodes(_TOOL_MAP)
    _register_dream(_TOOL_MAP)
    _register_graph(_TOOL_MAP)
    _register_concepts(_TOOL_MAP)
    _register_misc(_TOOL_MAP)


# Build the dispatch table at import time
_build_tool_map()


# ── Request handler ───────────────────────────────────────────────────────

def handle_request(request):
    method = request.get("method", "")
    params = request.get("params", {})
    rid = request.get("id")

    if rid is None:
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": rid,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "deep-dream", "version": "1.0.0"},
                "instructions": "Deep Dream is a knowledge graph memory layer. Key workflows:\n"
                    "FIND: quick_search(query) for most queries; find_entity_by_name(name) for exact lookup; "
                    "entity_profile(family_id) for full details; explore_topic(topic) for deep exploration; "
                    "ask(question) for AI-powered Q&A.\n"
                    "REMEMBER: remember(content) to store text (async, returns task_id); "
                    "remember_and_explore(content) for store+search in one call.\n"
                    "DREAM: dream_quick_start() to start consolidation; get_dream_seeds(strategy) for manual exploration.\n"
                    "MAINTAIN: graph_overview() at session start; butler_report() for AI maintenance.\n"
                    "Tips: Start with graph_overview(). Prefer quick_search over semantic_search. "
                    "Use entity_profile over get_entity. Use batch_profiles for multiple entities.",
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        handler = _TOOL_MAP.get(tool_name)
        if handler is None:
            return {
                "jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }

        try:
            # Set per-call graph context so _url() picks up the right graph_id
            with _graph_context(arguments):
                result = handler(arguments)
            return {"jsonrpc": "2.0", "id": rid, "result": result}
        except Exception as e:
            debug_log(f"Tool error: {tool_name}: {e}")
            return {
                "jsonrpc": "2.0", "id": rid,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0", "id": rid,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def main():
    debug_log(f"Deep Dream MCP Server starting, upstream={BASE_URL}, default_graph_id={_DEFAULT_GRAPH_ID}")
    while True:
        try:
            request = read_message()
            if request is None:
                break
            response = handle_request(request)
            if response:
                send_response(response)
        except Exception as e:
            debug_log(f"Main loop error: {e}")


if __name__ == "__main__":
    main()
