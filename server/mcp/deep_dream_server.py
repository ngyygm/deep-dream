#!/usr/bin/env python3
"""
Deep Dream MCP Server — exposes all Deep Dream API endpoints as MCP tools.

Protocol: stdio with Content-Length / NDJSON auto-detection.
Upstream: Deep Dream REST API on localhost:16200.
"""

import json
import os
import sys
import httpx

sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', buffering=0)
sys.stdin = os.fdopen(sys.stdin.fileno(), 'rb', buffering=0)

DEBUG_LOG = "/tmp/deep-dream-mcp-debug.log"
BASE_URL = os.environ.get("DEEP_DREAM_BASE_URL", "http://localhost:16200")
_DEFAULT_GRAPH_ID = os.environ.get("DEEP_DREAM_GRAPH_ID", "default")
_active_graph_id = _DEFAULT_GRAPH_ID  # runtime switchable

_use_ndjson = False


def debug_log(msg):
    try:
        with open(DEBUG_LOG, "a") as f:
            from datetime import datetime
            f.write(f"{datetime.now().isoformat()} {msg}\n")
    except Exception:
        pass


def send_response(resp):
    global _use_ndjson
    data = json.dumps(resp, ensure_ascii=False, separators=(',', ':')).encode()
    if _use_ndjson:
        sys.stdout.write(data + b'\n')
    else:
        sys.stdout.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.flush()


def read_message():
    global _use_ndjson
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.decode().rstrip('\r\n')
    if line.lower().startswith("content-length:"):
        n = int(line.split(':', 1)[1].strip())
        while True:
            h = sys.stdin.readline()
            if not h:
                return None
            if h.decode().rstrip('\r\n') == '':
                break
        body = sys.stdin.read(n)
        return json.loads(body.decode())
    elif line.startswith('{') or line.startswith('['):
        _use_ndjson = True
        return json.loads(line)
    return None


# ── HTTP helpers ──────────────────────────────────────────────────────────

_client = httpx.Client(timeout=60.0)


def _resolve_graph_id(args):
    """Resolve graph_id for a tool call.

    Priority: per-call args['graph_id'] > _active_graph_id (set by switch_graph) > env default.
    """
    gid = args.get("graph_id")
    if gid and isinstance(gid, str) and gid.strip():
        return gid.strip()
    return _active_graph_id


# ── Per-call graph context ─────────────────────────────────────────────────
# When a tool is dispatched, we set _current_call_graph_id so that _url() can
# pick it up without every handler needing to pass graph_id explicitly.

_current_call_graph_id = _DEFAULT_GRAPH_ID

import contextlib


@contextlib.contextmanager
def _graph_context(args):
    """Context manager: set per-call graph_id for the duration of a tool dispatch."""
    global _current_call_graph_id
    old = _current_call_graph_id
    _current_call_graph_id = _resolve_graph_id(args)
    try:
        yield _current_call_graph_id
    finally:
        _current_call_graph_id = old


def _url(path, graph_id=None, **qp):
    """Build URL with graph_id query param.

    Priority: explicit graph_id > per-call context > active graph > env default.
    """
    gid = graph_id or _current_call_graph_id or _active_graph_id
    sep = '&' if '?' in path else '?'
    url = f"{BASE_URL}{path}{sep}graph_id={gid}"
    for k, v in qp.items():
        if v is not None:
            url += f"&{k}={v}"
    return url


def _get(path, **qp):
    r = _client.get(_url(path, **qp))
    return r.json(), r.status_code


def _post(path, body=None, **qp):
    r = _client.post(_url(path, **qp), json=body or {})
    return r.json(), r.status_code


def _put(path, body=None, **qp):
    r = _client.put(_url(path, **qp), json=body or {})
    return r.json(), r.status_code


def _delete(path, body=None, **qp):
    kw = {}
    if body:
        kw["json"] = body
    r = _client.delete(_url(path, **qp), **kw)
    return r.json(), r.status_code


_MAX_RESPONSE_CHARS = 80000  # ~20k tokens safety cap

_TRIM_FIELDS = {"embedding", "embeddings", "content_hash", "raw_content", "vector"}

# Fields to truncate content to save agent tokens (content can be multi-KB markdown)
_CONTENT_TRUNCATE_LEN = 200
_SUMMARY_FIELDS = {
    # entity-like items: keep essential identifiers, truncate content
    "entity": {"keep": {"family_id", "name", "summary", "attributes", "event_time", "processed_time"}},
    # relation-like items
    "relation": {"keep": {"family_id", "content", "entity1_id", "entity2_id", "entity1_name", "entity2_name", "event_time", "relation_type"}},
    # version items
    "version": {"keep": {"absolute_id", "name", "event_time"}},
    # episode items
    "episode": {"keep": {"cache_id", "source", "event_time"}},
}


def _truncate_text(text, max_len=_CONTENT_TRUNCATE_LEN):
    """Truncate text at sentence boundary when possible."""
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    # Try breaking at last sentence boundary in second half of snippet
    for sep in ('. ', '。', '\n', '！', '？', '; ', '；'):
        idx = cut.rfind(sep)
        if idx > max_len // 2:
            return text[:idx + len(sep)].rstrip() + "..."
    # No good break point — truncate at last space if possible
    last_space = cut.rfind(' ')
    if last_space > max_len // 2:
        return text[:last_space] + "..."
    return cut + "..."


def _compact_entity(item):
    """Compact an entity dict: truncate content at sentence boundaries, keep essential fields including absolute_id for create_relation."""
    if not isinstance(item, dict):
        return item
    out = {}
    for k in ("family_id", "name", "summary", "absolute_id", "confidence"):
        if k in item:
            out[k] = item[k]
    if "_score" in item:
        out["_score"] = item["_score"]
    # Truncate content at sentence boundary
    for ck in ("content", "markdown_content"):
        if ck in item and isinstance(item[ck], str):
            out[ck] = _truncate_text(item[ck])
    if "event_time" in item:
        out["event_time"] = item["event_time"]
    if "version_count" in item:
        out["version_count"] = item["version_count"]
    # Keep small attributes; for large ones, keep key list as hint
    if "attributes" in item and isinstance(item["attributes"], dict) and item["attributes"]:
        attrs = item["attributes"]
        attr_len = sum(len(str(v)) for v in attrs.values())
        if attr_len <= 200:
            out["attributes"] = attrs
        else:
            out["_attr_keys"] = list(attrs.keys())[:8]
    return out


def _compact_relation(item):
    """Compact a relation dict: truncate content at sentence boundary, keep endpoints + confidence."""
    if not isinstance(item, dict):
        return item
    out = {}
    for k in ("family_id", "entity1_id", "entity2_id", "entity1_name", "entity2_name", "relation_type", "event_time", "confidence"):
        if k in item:
            out[k] = item[k]
    if "_score" in item:
        out["_score"] = item["_score"]
    if "content" in item and isinstance(item["content"], str):
        out["content"] = _truncate_text(item["content"])
    return out


def _compact_version(item):
    """Compact a version dict: truncate content at sentence boundary."""
    if not isinstance(item, dict):
        return item
    out = {}
    for k in ("absolute_id", "name", "event_time", "processed_time"):
        if k in item:
            out[k] = item[k]
    for ck in ("content", "markdown_content", "summary"):
        if ck in item and isinstance(item[ck], str):
            out[ck] = _truncate_text(item[ck])
    return out


def _compact_list(data, compact_fn, list_key=None):
    """Apply compact_fn to a list within a response dict.
    Works with patterns like {data: {entities: [...]}} or just {entities: [...]}."""
    if not isinstance(data, dict):
        return data

    # Check nested data.data pattern
    inner = data.get("data", data)
    keys_to_try = [list_key] if list_key else ["entities", "relations", "versions", "episodes", "items"]

    for key in keys_to_try:
        if key in inner and isinstance(inner[key], list):
            inner[key] = [compact_fn(item) for item in inner[key]]
            break

    return data


_NOISE_KEYS = {"success", "elapsed_ms", "timestamp"}


def _trim_response(data, max_chars=_MAX_RESPONSE_CHARS):
    """Remove bulky fields and unwrap boilerplate to save agent tokens."""
    if not isinstance(data, dict):
        text = json.dumps(data, ensure_ascii=False)
        if len(text) > max_chars:
            return text[:max_chars] + "\n... [truncated]"
        return text

    # Unwrap {success: true, data: {...}, elapsed_ms: ...} → just {...}
    inner = data.get("data")
    if isinstance(inner, dict) and "success" in data:
        # Keep only the inner data, but merge back any non-boilerplate top-level keys
        data = inner

    # Strip expensive fields from nested structures
    data = _strip_bulky(data)

    text = json.dumps(data, ensure_ascii=False)
    if len(text) <= max_chars:
        return text

    # Try progressively trimming large lists
    for key in ("entities", "relations", "versions", "episodes"):
        items = data.get(key)
        if not isinstance(items, list) or len(items) <= 3:
            continue
        # Binary search for max items that fit
        lo, hi = 1, min(len(items), 20)
        best_kept = None
        while lo <= hi:
            mid = (lo + hi) // 2
            kept = items[:mid]
            candidate = {**data, key: kept, f"{key}_total": len(items), f"{key}_shown": mid}
            ct = len(json.dumps(candidate, ensure_ascii=False))
            if ct <= max_chars:
                best_kept = (mid, candidate)
                lo = mid + 1
            else:
                hi = mid - 1
        if best_kept:
            mid, candidate = best_kept
            omitted = len(items) - mid
            text = json.dumps(candidate, ensure_ascii=False)
            if omitted > 0:
                text += f"\n→ {omitted} more {key} omitted. Use offset/limit to fetch more."
            return text

    # Last resort: hard truncate
    return text[:max_chars] + "\n... [response truncated to fit context]"


def _strip_bulky(obj):
    """Recursively remove embedding/vector/hash fields from response objects."""
    if isinstance(obj, dict):
        return {k: _strip_bulky(v) for k, v in obj.items() if k not in _TRIM_FIELDS}
    if isinstance(obj, list):
        return [_strip_bulky(item) for item in obj]
    return obj


def _error_hint(data):
    """Generate actionable hints from API error responses."""
    if not isinstance(data, dict):
        return ""
    msg = ""
    if isinstance(data.get("data"), dict):
        msg = data["data"].get("error", data["data"].get("message", ""))
    elif isinstance(data.get("error"), str):
        msg = data["error"]
    elif "detail" in data:
        msg = data["detail"]

    hints = []
    lower = msg.lower()
    if "not found" in lower and "entity" in lower:
        hints.append("Hint: use search_entities or find_entity_by_name to find the correct family_id.")
    if "not found" in lower and "relation" in lower:
        hints.append("Hint: use search_relations or get_entity_relations to find the correct relation.")
    if "not found" in lower and "episode" in lower:
        hints.append("Hint: use search_episodes or get_latest_episode to find valid cache_ids.")
    if "not found" in lower and "community" in lower:
        hints.append("Hint: use list_communities to see valid community IDs. Run detect_communities first if empty.")
    if "not found" in lower and "task" in lower:
        hints.append("Hint: use remember_tasks to list all tasks with valid IDs.")
    if "neo4j" in lower and ("not available" in lower or "sqlite" in lower):
        hints.append("Hint: this feature requires Neo4j backend. SQLite does not support it.")
    if "context budget" in lower or "token" in lower:
        hints.append("Hint: reduce max_entities/max_relations or shorten your query.")
    if "already exists" in lower:
        hints.append("Hint: use find_entity_by_name to check if the entity already exists.")
    if "timeout" in lower or "timed out" in lower:
        hints.append("Hint: the operation took too long. Try with smaller input or fewer items.")
    if "rate limit" in lower or "429" in lower:
        hints.append("Hint: too many requests. Wait a moment and retry.")
    if "invalid" in lower and ("id" in lower or "identifier" in lower):
        hints.append("Hint: check that the ID format is correct. family_ids start with 'ent_' or 'rel_', absolute_ids are UUIDs.")
    if "merge" in lower and ("same" in lower or "cannot" in lower or "error" in lower):
        hints.append("Hint: merge_entities requires at least 2 different entity family_ids.")
    if "cannot" in lower and ("delete" in lower or "remove" in lower):
        hints.append("Hint: the resource may be in use. Check system_tasks for running operations.")
    if "permission" in lower or "forbidden" in lower or "unauthorized" in lower:
        hints.append("Hint: check your API key configuration and graph_id access permissions.")
    if "validation" in lower or ("required" in lower and "field" in lower):
        hints.append("Hint: check the tool's required parameters. Missing or empty fields cause validation errors.")
    if "conflict" in lower or "409" in str(data.get("status_code", "")):
        hints.append("Hint: resource state conflict. The data may have changed since last read — refresh with get_entity or entity_profile.")

    return " ".join(hints) if hints else ""


def _empty_search_hint(data, query_param="query"):
    """Append hint when search returns zero results."""
    if not isinstance(data, dict):
        return data
    inner = _inner(data)
    entities = inner.get("entities", [])
    relations = inner.get("relations", [])
    if not entities and not relations:
        hint = f"\n→ No results found. Try: lower similarity_threshold, use search_mode='hybrid', or rephrase the {query_param}."
        _hint(data, hint)
    return data


def _pagination_hint(data, list_key, limit, offset=0):
    """Append hint when a list result equals the limit (suggesting more results exist).
    Only adds a hint if the result count matches the limit — a strong signal of truncation."""
    if not isinstance(data, dict) or not limit:
        return data
    inner = _inner(data)
    items = inner.get(list_key, [])
    if isinstance(items, list) and len(items) >= limit:
        next_offset = offset + limit
        hint = f"\n→ Result count matches limit ({limit}) — more results likely exist. Use offset={next_offset} to fetch the next page."
        _hint(data, hint)
    return data


def _hint(data, text):
    """Set a workflow hint on a response dict. Handles both wrapped and unwrapped responses."""
    if isinstance(data, dict):
        if isinstance(data.get("data"), dict):
            data["data"]["_hint"] = text
        else:
            data["_hint"] = text


def _inner(data):
    """Unwrap {data: {...}} boilerplate to get the inner dict."""
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        return data["data"]
    return data


_MAX_ERROR_CHARS = 2000  # Errors should be concise — agents need hints, not stack traces


def _compact_error(data):
    """Extract the essential error message from an API error response, discarding bulky fields."""
    if not isinstance(data, dict):
        return str(data)[:500]
    inner = data.get("data", data)
    # Extract core error fields
    msg = ""
    for key in ("error", "message", "detail", "error_message"):
        if key in inner and isinstance(inner[key], str):
            msg = inner[key]
            break
    if not msg and isinstance(inner.get("error"), dict):
        msg = inner["error"].get("message", "")
    if not msg:
        msg = str(data)[:500]
    # Truncate error message
    if len(msg) > 500:
        msg = msg[:500] + "..."
    # Include entity/relation ID context if present
    context_parts = []
    for k in ("family_id", "entity_id", "relation_id", "task_id", "cache_id"):
        if k in inner and isinstance(inner[k], str):
            context_parts.append(f"{k}={inner[k]}")
    result = {"error": msg}
    if context_parts:
        result["context"] = ", ".join(context_parts)
    return json.dumps(result, ensure_ascii=False)


def _result(data, code):
    # Extract workflow hints before trimming
    workflow_hint = ""
    if isinstance(data, dict):
        inner = _inner(data)
        workflow_hint = inner.pop("_hint", "") or data.pop("_hint", "")

    # For errors: use compact error format to save agent tokens
    if code >= 400:
        text = _compact_error(data)
        if workflow_hint:
            text = text.rstrip() + workflow_hint
        hint = _error_hint(data)
        if hint:
            text = text.rstrip() + "\n" + hint
        return {"content": [{"type": "text", "text": text}], "isError": True}

    text = _trim_response(data)
    if workflow_hint:
        text = text.rstrip() + workflow_hint
    return {"content": [{"type": "text", "text": text}]}


# ── Tool definitions ──────────────────────────────────────────────────────

TOOLS = []

# Shared graph_id parameter — injected into every tool for per-call graph routing.
# If omitted, uses the active graph (set by switch_graph, defaults to env var).
_GRAPH_ID_PARAM = {
    "graph_id": {
        "type": "string",
        "description": "Target graph ID. If omitted, operates on the active graph (set via switch_graph, defaults to env DEEP_DREAM_GRAPH_ID='default'). Use list_graphs to see available graphs.",
    },
}


def _t(name, desc, params, required=None):
    # Inject graph_id into every tool's parameter schema
    merged_params = {**params, **_GRAPH_ID_PARAM}
    TOOLS.append({
        "name": name,
        "description": desc,
        "inputSchema": {
            "type": "object",
            "properties": merged_params,
            **({"required": required} if required else {}),
        },
    })


# ── Remember (7) ──────────────────────────────────────────────────────────

_t("remember", "Submit text for async entity/relation extraction. Returns immediately with a task_id — poll remember_task_status(task_id='...') to check progress. The pipeline extracts entities and relations from the text and adds them to the knowledge graph. Typical workflow: remember(content='...') → remember_task_status(task_id='...') → repeat until completed.", {
    "content": {"type": "string", "description": "Text content to remember"},
    "source": {"type": "string", "description": "Source label (e.g. 'user', 'document:file.txt')"},
    "metadata": {"type": "object", "description": "Optional metadata dict"},
}, ["content"])

_t("remember_tasks", "List remember task queue. Use status filter to see pending/processing/completed/failed tasks. For a specific task, use remember_task_status(task_id=...).", {
    "status": {"type": "string", "description": "Filter by status (pending/processing/completed/failed)"},
})

_t("remember_task_status", "Get status of a specific remember task. Poll this after calling remember until status='completed'. The task_id comes from the remember response.", {
    "task_id": {"type": "string", "description": "Task ID (from remember() response)"},
}, ["task_id"])

_t("delete_remember_task", "Delete a remember task. Use this to clean up completed or failed tasks from the queue.", {
    "task_id": {"type": "string", "description": "Task ID (from remember() response)"},
}, ["task_id"])

_t("pause_remember_task", "Pause a running remember task. The task can be resumed later with resume_remember_task.", {
    "task_id": {"type": "string", "description": "Task ID (from remember() response)"},
}, ["task_id"])

_t("resume_remember_task", "Resume a paused remember task. Only paused tasks can be resumed.", {
    "task_id": {"type": "string", "description": "Task ID (from remember() response)"},
}, ["task_id"])

_t("remember_monitor", "Get remember pipeline monitor snapshot. Shows pending/processing counts. For task-level details, use remember_tasks.", {})


# ── Health (2) ────────────────────────────────────────────────────────────

_t("health_check", "Check if the Deep Dream API server is running and responsive", {})
_t("health_check_llm", "Check if the LLM backend (used for extraction/Q&A) is reachable. Use this when remember or ask calls fail — may indicate LLM provider issues.", {})


# ── Stats (3) ─────────────────────────────────────────────────────────────

_t("search_stats", "Get search engine usage statistics (query counts, cache hit rates). For graph content stats, use graph_stats or graph_summary.", {})
_t("graph_stats", "Get graph statistics: entity count and relation count. For a richer overview including backend type and embedding status, use graph_summary.", {})


# ── Find/Search (5) ──────────────────────────────────────────────────────

_t("semantic_search", "Semantic search across entities and relations. Use mode='entities' for entities only, mode='relations' for relations only. For most searches, quick_search is simpler and faster — only use semantic_search when you need: (1) specific mode control, (2) expanded graph context, or (3) custom top_k per category.", {
    "query": {"type": "string", "description": "Search query text"},
    "top_k": {"type": "integer", "description": "Max results per category (default 10)"},
    "mode": {"type": "string", "description": "Search mode: entities, relations, or all (default)"},
    "expand": {"type": "boolean", "description": "Whether to expand graph context (default false)"},
}, ["query"])

_t("search_candidates", "Find candidate entities matching a description using hybrid search. Use this before create_entity to avoid duplicates, or during entity resolution to check if a similar entity already exists. For simple name lookups, find_entity_by_name is faster.", {
    "description": {"type": "string", "description": "Entity description to match"},
    "top_k": {"type": "integer", "description": "Max candidates (default 10)"},
}, ["description"])

_t("search_entities", "Search entities by text query. Returns matching entities with names and content. Use this for broad text-based search; for semantic similarity use semantic_search, or for name lookup use find_entity_by_name.", {
    "query": {"type": "string", "description": "Search query"},
    "limit": {"type": "integer", "description": "Max results (default 20)"},
    "offset": {"type": "integer", "description": "Offset for pagination (0-based)"},
}, ["query"])

_t("search_relations", "Search relations by text query. Returns matching relations with content and connected entities. For comprehensive search returning both entities and relations, prefer quick_search.", {
    "query": {"type": "string", "description": "Search query"},
    "limit": {"type": "integer", "description": "Max results (default 20)"},
    "offset": {"type": "integer", "description": "Offset for pagination (0-based)"},
}, ["query"])

_t("traverse_graph", "BFS traverse from seed entity(s) to discover connected subgraph. Returns entities and relations within max_depth hops. Use depth=2 for immediate neighborhood, depth=3 for broader context. Good for understanding how entities are interconnected. Use this instead of entity_profile when you want to explore beyond a single entity's direct relations. Supports time_point for temporal traversal.", {
    "start_entity_id": {"type": "string", "description": "Starting entity family_id (or JSON array of family_ids for multiple seeds)"},
    "max_depth": {"type": "integer", "description": "Max traversal depth (default 2)"},
    "max_nodes": {"type": "integer", "description": "Max nodes to return (default 50)"},
    "time_point": {"type": "string", "description": "ISO 8601 timestamp — only return entities/relations valid at this time"},
}, ["start_entity_id"])


# ── Entity Query (10) ────────────────────────────────────────────────────

_t("list_entities", "List all entities with pagination. Use this only to browse the full entity catalog. For finding specific entities, prefer quick_search, find_entity_by_name, or search_entities instead.", {
    "limit": {"type": "integer", "description": "Max results (default 50)"},
    "offset": {"type": "integer", "description": "Offset for pagination (0-based)"},
})

_t("get_entity", "Get entity current version by family_id. Returns the entity's content, summary, attributes. NOTE: if you also need the entity's relations (most common case), use entity_profile instead — it returns entity + relations + version count in one call. Use get_entity only when you need just the raw entity data or the absolute_id for create_relation.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123') — use find_entity_by_name if unknown"},
}, ["family_id"])

_t("get_entity_versions", "Get all versions of an entity. Each version represents a state change. Use this to audit how an entity evolved. For a chronological view with relation events, prefer get_entity_timeline. For comparing two specific versions, use get_entity_version_diff.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "limit": {"type": "integer", "description": "Max versions to return"},
    "offset": {"type": "integer", "description": "Offset for pagination (0-based)"},
}, ["family_id"])

_t("get_entity_at_time", "Get entity state at an exact point in time (time travel). Returns the entity as it was at that timestamp. For approximate matches, use get_entity_nearest_to_time. For a time range, use get_entity_around_time.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "timestamp": {"type": "string", "description": "ISO 8601 timestamp (e.g. '2024-06-01T12:00:00')"},
}, ["family_id", "timestamp"])

_t("get_entity_nearest_to_time", "Get entity version closest to a given time. Tolerates slight time mismatches.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "timestamp": {"type": "string", "description": "ISO 8601 timestamp"},
}, ["family_id", "timestamp"])

_t("get_entity_around_time", "Get entity versions within a time window around a point. Useful for seeing what changed near a specific time.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "timestamp": {"type": "string", "description": "ISO 8601 timestamp (center of window)"},
    "within_seconds": {"type": "number", "description": "Time window radius in seconds (e.g. 3600 for ±1 hour)"},
}, ["family_id", "timestamp"])

_t("get_entity_relations", "Get relations connected to an entity. Use relation_scope to control time range. For a complete view with entity details + relations + version count in one call, prefer entity_profile.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "limit": {"type": "integer", "description": "Max results (default 50)"},
    "time_point": {"type": "string", "description": "Filter relations by time (ISO 8601)"},
    "relation_scope": {"type": "string", "description": "accumulated (default, all active), version_only (current version), all_versions (including future)"},
}, ["family_id"])

_t("get_entity_timeline", "Get entity timeline: all version changes and relation events in chronological order. Combines both version history and relation changes in one view. For version-only history use get_entity_versions, for relation-only use get_entity_relations.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "limit": {"type": "integer", "description": "Max events"},
}, ["family_id"])

_t("get_entity_by_absolute_id", "Get a specific entity version by its absolute (version) ID. Use family_id for the current version instead.", {
    "absolute_id": {"type": "string", "description": "Entity absolute/version ID (UUID format, from get_entity_versions)"},
}, ["absolute_id"])

_t("get_entity_version_counts", "Get version counts for multiple entities in one call. Useful for identifying entities with many versions that may need cleanup or consolidation.", {
    "family_ids": {"type": "array", "items": {"type": "string"}, "description": "List of entity family IDs"},
}, ["family_ids"])


# ── Entity CRUD (8) ──────────────────────────────────────────────────────

_t("create_entity", "Create a new entity manually. For bulk creation from text, use remember instead.", {
    "name": {"type": "string", "description": "Entity name (required)"},
    "content": {"type": "string", "description": "Entity content/description"},
    "episode_id": {"type": "string", "description": "Episode ID to link (optional)"},
    "source_document": {"type": "string", "description": "Source document label (optional)"},
}, ["name"])

_t("update_entity", "Update entity metadata (name, summary, attributes) by family_id. Does NOT create a new version — modifies the current version in place. Use evolve_entity_summary for AI-driven summary regeneration.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "name": {"type": "string", "description": "New name"},
    "summary": {"type": "string", "description": "New summary"},
    "attributes": {"type": "object", "description": "Updated attributes (merged with existing)"},
    "source": {"type": "string", "description": "Source label for this update"},
}, ["family_id"])

_t("update_entity_by_absolute_id", "Update a specific entity version by absolute_id. Does NOT create a new version.", {
    "absolute_id": {"type": "string", "description": "Entity absolute/version ID (UUID format, from get_entity_versions)"},
    "name": {"type": "string", "description": "New name"},
    "summary": {"type": "string", "description": "New summary"},
    "attributes": {"type": "object", "description": "Updated attributes"},
}, ["absolute_id"])

_t("delete_entity", "Delete entity and all its versions by family_id. This is permanent and cannot be undone. For a softer approach, consider whether the entity can be left as-is or merged with merge_entities.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
}, ["family_id"])

_t("delete_entity_by_absolute_id", "Delete a specific entity version by absolute_id. Does NOT delete the entire entity — only the specified version snapshot.", {
    "absolute_id": {"type": "string", "description": "Entity absolute/version ID (UUID format, from get_entity_versions)"},
}, ["absolute_id"])

_t("batch_delete_entities", "Delete multiple entities at once by their family IDs. For single entity deletion, use delete_entity. For targeted version removal, use batch_delete_entity_versions.", {
    "family_ids": {"type": "array", "items": {"type": "string"}, "description": "List of entity family IDs to delete"},
}, ["family_ids"])

_t("merge_entities", "Merge multiple entities into one. All relations and versions are consolidated into the target entity. Workflow: (1) search_similar_entities or find_entity_by_name to identify duplicates, (2) batch_profiles to compare content, (3) merge_entities to consolidate. Irreversible — verify before merging.", {
    "family_ids": {"type": "array", "items": {"type": "string"}, "description": "All entity family IDs to merge (target + sources)"},
    "target_family_id": {"type": "string", "description": "Which entity to keep as the target (optional, defaults to first)"},
    "target_name": {"type": "string", "description": "New name for merged entity (optional)"},
    "target_summary": {"type": "string", "description": "New summary for merged entity (optional)"},
}, ["family_ids"])

_t("split_entity_version", "Separate a specific version into its own new entity. Useful when an entity has accumulated mixed topics across versions. Workflow: get_entity_versions → identify the version to split → split_entity_version. Get the version_id (absolute_id) from get_entity_versions.", {
    "family_id": {"type": "string", "description": "Source entity family ID"},
    "version_id": {"type": "string", "description": "Version absolute ID to split out"},
    "new_name": {"type": "string", "description": "Name for the new entity"},
}, ["family_id", "version_id"])


# ── Entity Intelligence (6) ──────────────────────────────────────────────

_t("evolve_entity_summary", "Use LLM to regenerate entity summary by analyzing all version history. Call this when an entity has accumulated significant new information across multiple versions and the summary is outdated or incomplete. Uses one LLM call per entity.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "context": {"type": "string", "description": "Additional context to incorporate (optional)"},
}, ["family_id"])

_t("get_entity_contradictions", "Detect contradictions between entity versions. Returns list of conflicting data points with severity. Call this after remember adds new data to check for inconsistencies. Follow with resolve_entity_contradiction to fix.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
}, ["family_id"])

_t("resolve_entity_contradiction", "Resolve a detected contradiction by choosing a strategy (keep_new, keep_old, merge, flag_for_review). Call get_entity_contradictions first to get the contradiction_id. Use 'flag_for_review' when unsure.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "contradiction_id": {"type": "string", "description": "Contradiction ID from get_entity_contradictions"},
    "resolution": {"type": "string", "description": "Resolution strategy: keep_new, keep_old, merge, or flag_for_review"},
}, ["family_id", "contradiction_id", "resolution"])

_t("get_relation_contradictions", "Detect contradictions between relation versions. Returns list of conflicting data points with severity. Call this after remember adds new data to check for inconsistencies in relations. Follow with resolve_relation_contradiction to fix.", {
    "family_id": {"type": "string", "description": "Relation family ID (e.g. 'rel_abc123')"},
}, ["family_id"])

_t("resolve_relation_contradiction", "Resolve a detected relation contradiction by choosing a strategy (keep_new, keep_old, merge, flag_for_review). Call get_relation_contradictions first to get the contradiction_id. Use 'flag_for_review' when unsure.", {
    "family_id": {"type": "string", "description": "Relation family ID (e.g. 'rel_abc123')"},
    "contradiction_id": {"type": "string", "description": "Contradiction ID from get_relation_contradictions"},
    "resolution": {"type": "string", "description": "Resolution strategy: keep_new, keep_old, merge, or flag_for_review"},
}, ["family_id", "contradiction_id", "resolution"])

_t("get_entity_provenance", "Trace where entity data came from: source documents, extraction timestamps, confidence scores. Use this to verify data reliability or debug incorrect extractions. For a broader audit, combine with get_entity_versions.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
}, ["family_id"])

_t("get_entity_version_diff", "Compare two entity versions to see what changed. Provide both from_version and to_version for meaningful results.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "from_version": {"type": "string", "description": "Source version absolute ID (get from get_entity_versions)"},
    "to_version": {"type": "string", "description": "Target version absolute ID"},
}, ["family_id"])

_t("get_entity_patches", "Get incremental patches (diffs) applied to an entity over time. Shows what was added/removed at each version transition. For comparing two specific versions, use get_entity_version_diff instead.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "limit": {"type": "integer", "description": "Max patches to return"},
}, ["family_id"])


# ── Relation Query (6) ───────────────────────────────────────────────────

_t("list_relations", "List all relations with pagination and optional type filter. Use this only to browse the full catalog. For finding specific relations, prefer search_relations, quick_search, or get_relations_between.", {
    "limit": {"type": "integer", "description": "Max results (default 50)"},
    "offset": {"type": "integer", "description": "Offset for pagination (0-based)"},
    "relation_type": {"type": "string", "description": "Filter by relation type label (e.g. 'related_to', 'part_of')"},
})

_t("get_relation_by_absolute_id", "Get a specific relation version by its absolute (version) ID. Use family_id for the current version instead.", {
    "absolute_id": {"type": "string", "description": "Relation absolute/version ID (UUID format, from get_relation_versions)"},
}, ["absolute_id"])

_t("get_relation_versions", "Get all versions of a relation. Each version represents a content or linkage change over time. Use absolute_ids from this for batch_delete_relation_versions.", {
    "family_id": {"type": "string", "description": "Relation family ID (e.g. 'rel_abc123')"},
    "limit": {"type": "integer", "description": "Max versions to return"},
}, ["family_id"])

_t("get_relations_between", "Find all relations connecting two entities. Returns both directions. Use this to verify if/how two entities are linked. For discovering indirect connections, use search_shortest_path or traverse_graph.", {
    "entity_a": {"type": "string", "description": "First entity family_id"},
    "entity_b": {"type": "string", "description": "Second entity family_id"},
}, ["entity_a", "entity_b"])

_t("search_shortest_path", "Find the shortest path between two entities in the graph. Returns intermediate entities and relations along the path. Use this to discover how two seemingly unrelated entities are connected. Requires Neo4j backend. Start with max_depth=5, increase for sparse graphs.", {
    "from_entity": {"type": "string", "description": "Start entity family_id"},
    "to_entity": {"type": "string", "description": "End entity family_id"},
    "max_depth": {"type": "integer", "description": "Max search depth (default 5). Increase for sparse graphs."},
}, ["from_entity", "to_entity"])

_t("search_shortest_path_cypher", "Find shortest path using native Cypher query. Same as search_shortest_path but uses Neo4j Cypher directly. Prefer search_shortest_path unless you need Cypher-specific behavior.", {
    "from_entity": {"type": "string", "description": "Start entity family_id"},
    "to_entity": {"type": "string", "description": "End entity family_id"},
    "max_depth": {"type": "integer", "description": "Max search depth (default 5)"},
}, ["from_entity", "to_entity"])


# ── Relation CRUD (7) ────────────────────────────────────────────────────

_t("create_relation", "Create a new relation between two entities. IMPORTANT: requires absolute_ids (version-specific IDs like UUIDs), NOT family_ids (like 'ent_abc123'). Workflow: (1) get_entity(family_id='ent_X') → note the absolute_id from response, (2) get_entity(family_id='ent_Y') → note absolute_id, (3) create_relation with those absolute_ids. For dream-discovered relations, prefer create_dream_relation which uses family_ids directly.", {
    "entity1_absolute_id": {"type": "string", "description": "Absolute ID (version ID) of the first entity — use get_entity(family_id=...) to find this"},
    "entity2_absolute_id": {"type": "string", "description": "Absolute ID (version ID) of the second entity — use get_entity(family_id=...) to find this"},
    "content": {"type": "string", "description": "Relation content/description"},
    "episode_id": {"type": "string", "description": "Episode ID to link (optional)"},
    "source_document": {"type": "string", "description": "Source document label (optional)"},
}, ["entity1_absolute_id", "entity2_absolute_id", "content"])

_t("update_relation", "Update relation metadata by family_id. Changes the current version without creating a new version. For fixing incorrect entity linkages, use redirect_relation instead.", {
    "family_id": {"type": "string", "description": "Relation family ID (e.g. 'rel_abc123')"},
    "content": {"type": "string", "description": "New content/description"},
    "summary": {"type": "string", "description": "New summary"},
    "attributes": {"type": "object", "description": "Updated attributes (merged with existing)"},
}, ["family_id"])

_t("update_relation_by_absolute_id", "Update a specific relation version by absolute_id. Use family_id for the current version instead.", {
    "absolute_id": {"type": "string", "description": "Relation absolute/version ID (UUID format, from get_relation_versions)"},
    "content": {"type": "string", "description": "New content/description"},
    "relation_type": {"type": "string", "description": "New relation type"},
    "summary": {"type": "string", "description": "New summary"},
}, ["absolute_id"])

_t("delete_relation", "Delete relation and all its versions by family_id. This is permanent. For a reversible approach, use invalidate_relation instead (soft-delete that can be cleaned up later).", {
    "family_id": {"type": "string", "description": "Relation family ID (e.g. 'rel_abc123')"},
}, ["family_id"])

_t("delete_relation_by_absolute_id", "Delete a specific relation version by absolute_id. Does NOT delete the entire relation — only the specified version snapshot.", {
    "absolute_id": {"type": "string", "description": "Relation absolute/version ID (UUID format, from get_relation_versions)"},
}, ["absolute_id"])

_t("batch_delete_relations", "Delete multiple relations at once by their family IDs. For single relation deletion, use delete_relation. For targeted version removal, use batch_delete_relation_versions.", {
    "family_ids": {"type": "array", "items": {"type": "string"}, "description": "List of relation family IDs to delete"},
}, ["family_ids"])

_t("redirect_relation", "Re-point one end of a relation to a different entity. Useful for fixing incorrect linkages. Use side='entity1' or 'entity2' to choose which end to redirect.", {
    "relation_family_id": {"type": "string", "description": "Relation family_id to redirect"},
    "new_target_id": {"type": "string", "description": "New target entity family_id"},
    "side": {"type": "string", "description": "Which end to redirect: 'entity1' or 'entity2'"},
}, ["relation_family_id", "new_target_id"])


# ── Episode (4) ─────────────────────────────────────────────────────

_t("get_latest_episode", "Get the most recent Episode (snapshot of all entities/relations). Use this to see the current state of the graph. For just metadata without the heavy content, use get_latest_episode_metadata.", {})
_t("get_latest_episode_metadata", "Get metadata of the latest Episode without the full content. Faster than get_latest_episode when you only need timestamps and counts.", {})
_t("get_episode_by_id", "Get a specific Episode by its cache_id. Returns the full snapshot with entity and relation data. Use search_episodes to find relevant episodes first.", {
    "cache_id": {"type": "string", "description": "Episode cache ID (from get_latest_episode or search_episodes)"},
}, ["cache_id"])

_t("get_episode_doc", "Get the source document content associated with an Episode. Use get_episode_text for raw text instead.", {
    "cache_id": {"type": "string", "description": "Episode cache ID (from get_latest_episode or search_episodes)"},
}, ["cache_id"])


# ── Snapshot/Changes (2) ─────────────────────────────────────────────────

_t("get_snapshot", "Get a full graph snapshot at a point in time. Omit timestamp for the latest snapshot. Use get_changes(since=...) instead if you only need what changed since a specific time — much lighter weight.", {
    "timestamp": {"type": "string", "description": "ISO 8601 timestamp for point-in-time snapshot (omit for latest)"},
})

_t("get_changes", "Get all entity/relation changes in a time range. Useful for incremental sync or audit logging.", {
    "since": {"type": "string", "description": "ISO 8601 timestamp — return changes after this time"},
    "until": {"type": "string", "description": "ISO 8601 timestamp — return changes before this time (optional, defaults to now)"},
    "limit": {"type": "integer", "description": "Max changes to return"},
}, ["since"])


# ── Episodes (3) ─────────────────────────────────────────────────────────

_t("search_episodes", "Search episodes by content using semantic similarity. Returns matching episodes with metadata. Use this to find which remember operations produced specific information.", {
    "query": {"type": "string", "description": "Search query text"},
    "limit": {"type": "integer", "description": "Max results to return"},
}, ["query"])

_t("delete_episode", "Delete an episode by its cache_id. This removes the episode and its associated data.", {
    "cache_id": {"type": "string", "description": "Episode cache ID to delete"},
}, ["cache_id"])

_t("batch_ingest_episodes", "Bulk import multiple episodes at once. Each episode triggers async extraction like remember. For single texts, use remember instead. Each episode object needs at least 'content'.", {
    "episodes": {"type": "array", "items": {"type": "object"}, "description": "List of episode objects: [{\"content\": \"...\", \"source\": \"optional\", \"episode_type\": \"optional\"}]"},
}, ["episodes"])


# ── Dream (6) ────────────────────────────────────────────────────────────

_t("get_dream_status", "Get current dream consolidation status. Shows whether a dream cycle is running and its progress. Start with this before initiating dream exploration.", {})
_t("get_dream_logs", "Get dream cycle history logs. Each log entry summarizes a completed dream cycle. Use get_dream_log_detail(cycle_id=...) for full details of a specific cycle.", {
    "limit": {"type": "integer", "description": "Max logs to return (default 20)"},
})

_t("get_dream_log_detail", "Get detailed information about a specific dream cycle, including entities explored and relations discovered.", {
    "cycle_id": {"type": "string", "description": "Dream cycle ID (from get_dream_logs)"},
}, ["cycle_id"])

_t("get_dream_seeds", "Get seed entities for dream exploration. Seeds are starting points for discovering hidden connections. Workflow: get_dream_seeds → entity_profile for each seed → traverse_graph(depth=2) → create_dream_relation for discoveries → save_dream_episode. Strategies: 'hub' (highly connected), 'orphan' (isolated, good for connecting loose ends), 'recent' (newly added), 'random'.", {
    "strategy": {"type": "string", "description": "Seed selection strategy: hub, orphan, recent, or random (default)"},
    "count": {"type": "integer", "description": "Number of seeds to return (default 5)"},
})

_t("create_dream_relation", "Create a relation discovered during dream exploration. Unlike create_relation, this uses family_ids (not absolute_ids) and records confidence/reasoning metadata. Always verify with get_relations_between first to avoid duplicates.", {
    "entity1_id": {"type": "string", "description": "First entity family_id"},
    "entity2_id": {"type": "string", "description": "Second entity family_id"},
    "content": {"type": "string", "description": "Relation description / content"},
    "confidence": {"type": "number", "description": "Confidence score 0-1 (default 0.7)"},
    "reasoning": {"type": "string", "description": "Why this relation was discovered"},
    "dream_type": {"type": "string", "description": "Dream type: free_association, cross_domain, etc."},
}, ["entity1_id", "entity2_id"])

_t("save_dream_episode", "Save a dream exploration episode record. Call this AFTER completing a dream cycle to persist the summary and insights. Part of the dream workflow: get_dream_seeds → explore → create_dream_relation → save_dream_episode.", {
    "dream_type": {"type": "string", "description": "Type of dream: free_association, cross_domain, consolidation, etc."},
    "entities_explored": {"type": "array", "items": {"type": "string"}, "description": "Entity family_ids explored during this cycle"},
    "relations_found": {"type": "integer", "description": "Number of new relations found"},
    "summary": {"type": "string", "description": "Episode summary text"},
    "insights": {"type": "string", "description": "Key insights from this dream"},
}, ["dream_type", "summary"])


# ── Agent / Ask (3) ──────────────────────────────────────────────────────

_t("ask", "Ask a natural language question about the knowledge graph. The AI will reason over entities and relations to produce an answer. For complex questions, provide context for better results.", {
    "question": {"type": "string", "description": "Question to ask (e.g. 'How are X and Y related?')"},
    "context": {"type": "string", "description": "Additional context to guide the answer (optional)"},
}, ["question"])

_t("explain_entity", "Get an AI-generated explanation of an entity. Optionally focus on a specific aspect. Uses LLM reasoning over entity data. For raw data without AI interpretation, use entity_profile.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "question": {"type": "string", "description": "Specific aspect to explain (e.g. 'Why is this important?' or 'What changed recently?')"},
}, ["family_id"])

_t("get_suggestions", "Get AI-curated suggestions for entities worth exploring. Good starting point when you're not sure what to look at. Optionally seed from a specific entity to discover related topics. For specific searches, use quick_search instead.", {
    "entity_id": {"type": "string", "description": "Starting entity family_id (optional — omit for global suggestions)"},
    "limit": {"type": "integer", "description": "Max suggestions to return"},
})


# ── Communities (5) ──────────────────────────────────────────────────────

_t("detect_communities", "Run community detection on the graph. Returns community assignments for entities. Requires Neo4j backend. After detection, use list_communities to see results and get_community to inspect members. Higher resolution = more, smaller communities.", {
    "algorithm": {"type": "string", "description": "Algorithm: louvain (default) or label_propagation"},
    "resolution": {"type": "number", "description": "Resolution parameter for louvain (default 1.0). Higher values produce more, smaller communities."},
})

_t("list_communities", "List all detected communities with member counts and statistics. Run detect_communities first.", {})
_t("get_community", "Get details of a specific community: member entities, internal relations, and summary.", {
    "cid": {"type": "string", "description": "Community ID (from list_communities or detect_communities)"},
}, ["cid"])

_t("get_community_graph", "Get the subgraph for a specific community. Returns all entities and relations within the community for visualization. For just the member list and summary, use get_community (lighter weight).", {
    "cid": {"type": "string", "description": "Community ID (from list_communities or detect_communities)"},
}, ["cid"])

_t("clear_communities", "Remove all community detection labels from entities. Does not delete entities or relations.", {})


# ── Graphs (2) ───────────────────────────────────────────────────────────

# Note: list_graphs and create_graph don't use _GRAPH_ID_PARAM for graph routing
# (they operate across graphs), but it's still injected — just ignored.

_t("list_graphs", "List all knowledge graphs registered in the system. Each graph is an isolated namespace.", {})
_t("create_graph", "Create a new knowledge graph. Each graph is an isolated namespace for entities and relations.", {
    "graph_id": {"type": "string", "description": "Unique graph identifier (e.g. 'my_project', 'research_notes')"},
    "name": {"type": "string", "description": "Human-readable name (optional)"},
    "description": {"type": "string", "description": "Graph description (optional)"},
}, ["graph_id"])

_t("delete_graph", "Delete a knowledge graph and all its data permanently. This cannot be undone.", {
    "graph_id": {"type": "string", "description": "Graph ID to delete"},
}, ["graph_id"])

_t("switch_graph", "Switch the active graph for subsequent tool calls. All future operations will target this graph unless overridden with a per-call graph_id parameter. Returns the new active graph info.", {
    "graph_id": {"type": "string", "description": "Graph ID to switch to. Must be an existing graph — use list_graphs to see available IDs."},
}, ["graph_id"])

_t("get_active_graph", "Get the currently active graph ID and its summary. All tool calls without an explicit graph_id parameter will target this graph.", {})


# ── Docs (2) ─────────────────────────────────────────────────────────────

_t("list_docs", "List documentation files stored in the system. These are source documents that were processed via remember. Use get_doc_content(filename=...) to read a specific document.", {})
_t("get_doc_content", "Get the full content of a stored document by filename. Use list_docs first to see available documents.", {
    "filename": {"type": "string", "description": "Document filename (from list_docs)"},
}, ["filename"])


# ── Neo4j (3) ────────────────────────────────────────────────────────────

_t("get_entity_neighbors", "Get immediate graph neighbors of an entity from Neo4j. Requires the entity's UUID (Neo4j internal ID, not family_id). For most use cases, entity_profile or traverse_graph are easier as they accept family_id. Only use this for low-level Neo4j access.", {
    "uuid": {"type": "string", "description": "Entity UUID (Neo4j internal ID, not family_id)"},
    "direction": {"type": "string", "description": "Edge direction: 'outgoing', 'incoming', or 'both' (default)"},
    "limit": {"type": "integer", "description": "Max neighbors to return"},
}, ["uuid"])

_t("list_episodes", "List all episodes stored in Neo4j with pagination. For searching episode content, use search_episodes. Requires Neo4j backend.", {
    "limit": {"type": "integer", "description": "Max results to return"},
    "offset": {"type": "integer", "description": "Offset for pagination (0-based)"},
})

_t("get_neo4j_episode", "Get a specific episode by its Neo4j UUID. Returns the full episode record. For cache_id based access, use get_episode_by_id instead.", {
    "uuid": {"type": "string", "description": "Episode UUID (Neo4j internal ID)"},
}, ["uuid"])


# ── Data Quality & Maintenance (4) ──────────────────────────────────────────

_t("delete_isolated_entities", "Delete all isolated entities (entities with zero relations). Always use dry_run=true first to preview what will be deleted.", {
    "dry_run": {"type": "boolean", "description": "Preview only without deleting (default: false). Strongly recommended to run true first."},
})

_t("get_data_quality_report", "Get a comprehensive data quality report: counts of valid, invalidated, and isolated entities/relations. For a combined report with graph stats, use maintenance_health instead. For AI-powered recommendations, use butler_report.", {})

_t("cleanup_old_versions", "Remove invalidated (soft-deleted) entity/relation versions to reclaim storage. Safe — only removes already-invalidated data.", {
    "before_date": {"type": "string", "description": "ISO date string — only remove versions invalidated before this date (optional)"},
    "dry_run": {"type": "boolean", "description": "Preview only without deleting (default: false)"},
})

_t("search_similar_entities", "Find potentially duplicate entities by name similarity. Returns entities with similar names for merge review.", {
    "name": {"type": "string", "description": "Entity name to search for duplicates"},
    "similarity_threshold": {"type": "number", "description": "Minimum similarity score 0-1 (default: 0.7). Lower = more results."},
})


# ── System (6) ───────────────────────────────────────────────────────────

_t("system_dashboard", "Get the system dashboard: uptime, entity/relation counts, API stats, thread info. For graph-specific stats, use graph_summary instead.", {})
_t("system_overview", "Get a high-level system overview: version, backend status, configuration. Use health_check for a simple connectivity test.", {})
_t("system_graphs", "Get information about all graphs in the system. Includes storage backend and entity/relation counts per graph.", {})
_t("system_tasks", "Get running and queued system tasks. Shows active remember tasks, dream cycles, and maintenance operations.", {})
_t("system_logs", "Get system log entries. Filter by level for targeted debugging.", {
    "level": {"type": "string", "description": "Log level filter: 'info', 'warn', or 'error'"},
    "limit": {"type": "integer", "description": "Max log entries to return"},
})

_t("system_access_stats", "Get API access statistics: request counts, latencies, endpoint usage.", {})


# ── Relation Invalidation (2) ────────────────────────────────────────────

_t("invalidate_relation", "Soft-delete a relation by family_id. The relation is marked as invalidated but not permanently removed. Can be cleaned up later with cleanup_old_versions.", {
    "family_id": {"type": "string", "description": "Relation family ID to invalidate"},
    "reason": {"type": "string", "description": "Reason for invalidation (optional, stored for audit)"},
}, ["family_id"])

_t("list_invalidated_relations", "List all soft-deleted (invalidated) relations. These can be permanently removed with cleanup_old_versions.", {
    "limit": {"type": "integer", "description": "Max results to return (default 100)"},
})


# ── Version Management (2) ────────────────────────────────────────────────

_t("batch_delete_entity_versions", "Delete specific entity version snapshots by their absolute IDs. Unlike delete_entity (which removes all versions), this surgically removes individual versions. Get absolute_ids from get_entity_versions.", {
    "absolute_ids": {"type": "array", "items": {"type": "string"}, "description": "List of entity absolute (version) IDs to permanently delete"},
}, ["absolute_ids"])

_t("batch_delete_relation_versions", "Delete specific relation version snapshots by their absolute IDs. Unlike delete_relation (which removes all versions), this surgically removes individual versions. Get absolute_ids from get_relation_versions.", {
    "absolute_ids": {"type": "array", "items": {"type": "string"}, "description": "List of relation absolute (version) IDs to permanently delete"},
}, ["absolute_ids"])


# ── Section History (1) ───────────────────────────────────────────────────

_t("get_section_history", "Track how a specific Markdown section of an entity evolved over time. Section keys correspond to headings in entity content (e.g. '## Summary', '## Details'). Useful for auditing changes to a particular aspect of an entity.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "section": {"type": "string", "description": "Markdown heading key (e.g. '## Summary', '## Details', '## Key Facts')"},
}, ["family_id", "section"])


# ── Episode Text (1) ──────────────────────────────────────────────────────

_t("get_episode_text", "Get the original raw text that was submitted when creating an episode. Use get_episode_doc for the processed document instead.", {
    "cache_id": {"type": "string", "description": "Episode cache ID (from list_episodes or get_latest_episode)"},
}, ["cache_id"])


# ── Isolated Entities (1) ─────────────────────────────────────────────────

_t("list_isolated_entities", "List all isolated entities (entities with zero relations). These may be extraction artifacts. Review before bulk-deleting with delete_isolated_entities.", {
    "limit": {"type": "integer", "description": "Max results (default 100)"},
    "offset": {"type": "integer", "description": "Offset for pagination (0-based)"},
})

# ── Aggregation Tools (1 call replaces 3-5) ─────────────────────────────────
_t("entity_profile", "Get a complete entity profile in one call: current entity data + all connected relations + version count. Use this INSTEAD of get_entity when you also need relations (which is almost always). Only use get_entity if you need just the raw entity data without any relation context.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
}, ["family_id"])

_t("graph_summary", "Get graph overview in one call: total entity/relation counts, storage backend type, embedding model status. Use this as your FIRST call to understand the graph before performing operations. For just entity/relation counts, graph_stats is lighter.", {})

_t("maintenance_health", "Combined health check: graph statistics + data quality report (valid/invalidated/isolated counts) + isolated entity list. Use this before running cleanup operations.", {})

_t("maintenance_cleanup", "One-click cleanup that combines two operations: (1) remove all invalidated (soft-deleted) versions, (2) delete isolated entities with zero relations. Always use dry_run=true first to preview.", {
    "dry_run": {"type": "boolean", "description": "Preview only without deleting (default false). Strongly recommended to run true first."},
})

_t("butler_report", "Comprehensive AI-powered health report combining graph stats + data quality + dream status. Returns actionable recommendations (e.g. 'cleanup_isolated', 'evolve_summaries'). Workflow: butler_report → review recommendations → butler_execute(actions=[...]). Use dry_run=true on butler_execute to preview before applying.", {})

_t("butler_execute", "Execute butler optimization actions on the memory graph. Get action names from butler_report. Available actions: cleanup_isolated (remove isolated entities), cleanup_invalidated (remove soft-deleted versions), detect_communities (run community detection), evolve_summaries (regenerate entity summaries with LLM).", {
    "actions": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Actions to execute: cleanup_isolated, cleanup_invalidated, detect_communities, evolve_summaries",
    },
    "dry_run": {"type": "boolean", "description": "Preview only without executing (default false). Strongly recommended for evolve_summaries since it uses LLM calls."},
}, ["actions"])

_t("quick_search", "All-in-one search: auto-selects the best search mode and returns both entities and relations in a single call. This is the RECOMMENDED starting point for most searches. Use this instead of semantic_search unless you need specific mode control or expand context. Typical use: 'What does the graph know about X?' → quick_search(query='X').", {
    "query": {"type": "string", "description": "Search query text"},
    "max_entities": {"type": "integer", "description": "Max entities to return (default 10, max 50)"},
    "max_relations": {"type": "integer", "description": "Max relations to return (default 20, max 100)"},
    "similarity_threshold": {"type": "number", "description": "Min similarity score (default 0.4). Lower = more results, higher = more precise."},
}, ["query"])

_t("find_entity_by_name", "Fast entity lookup by name using fuzzy matching. Returns the entity and its connected relations. Prefer this over search_entities when you know the entity name. Threshold guide: 1.0=exact, 0.7+=strict (recommended default), 0.5=broad, 0.3=very broad.", {
    "name": {"type": "string", "description": "Entity name to search for (supports partial/fuzzy matching)"},
    "threshold": {"type": "number", "description": "Min similarity threshold (default 0.5). Use 0.7+ for strict matching, 0.3 for broad."},
    "limit": {"type": "integer", "description": "Max candidate entities to return (default 5)"},
}, ["name"])

_t("batch_profiles", "Get profiles for up to 20 entities in one call. Each profile includes entity details + relations + version count. More efficient than calling entity_profile in a loop.", {
    "family_ids": {"type": "array", "items": {"type": "string"}, "description": "List of entity family IDs (max 20)"},
}, ["family_ids"])

_t("recent_activity", "Get a snapshot of recent graph activity: newest entities, newest relations, and current statistics. Useful as a dashboard summary or to see what changed recently. For specific entity details, follow up with entity_profile or batch_profiles.", {
    "limit": {"type": "integer", "description": "Max items per category (default 10, max 50)"},
})


# ── Concepts — 统一概念查询 (7) ──────────────────────────────────────────

_t("search_concepts", "Unified concept search across all roles (entity, relation, observation). Searches the unified concept space using BM25 text matching. Optionally filter by role. Returns concepts with their role, family_id, and content.", {
    "query": {"type": "string", "description": "Search query text"},
    "role": {"type": "string", "description": "Optional role filter: 'entity', 'relation', or 'observation'"},
    "limit": {"type": "integer", "description": "Max results (default 20, max 100)"},
}, ["query"])

_t("list_concepts", "List concepts with pagination and optional role filter. Returns concepts from the unified concept table across all roles.", {
    "role": {"type": "string", "description": "Optional role filter: 'entity', 'relation', or 'observation'"},
    "limit": {"type": "integer", "description": "Max results per page (default 50, max 100)"},
    "offset": {"type": "integer", "description": "Pagination offset (default 0)"},
})

_t("get_concept", "Get a concept by family_id. Works for any role — entity, relation, or observation. Returns the latest version of the concept.", {
    "family_id": {"type": "string", "description": "Concept family ID (e.g. 'ent_abc123', 'rel_abc123', or episode ID)"},
}, ["family_id"])

_t("get_concept_neighbors", "Get neighbors of a concept, regardless of its role. For entities: returns connected relations. For relations: returns connected entities. For observations: returns mentioned concepts.", {
    "family_id": {"type": "string", "description": "Concept family ID"},
    "max_depth": {"type": "integer", "description": "Neighbor depth (default 1, max 3)"},
}, ["family_id"])

_t("get_concept_provenance", "Trace a concept back to its source observations. Returns all episodes (observations) that mention this concept, enabling full provenance tracking.", {
    "family_id": {"type": "string", "description": "Concept family ID"},
}, ["family_id"])

_t("traverse_concepts", "BFS traverse the concept graph starting from one or more seed concepts. Discovers connected concepts across all roles in a unified graph traversal.", {
    "start_family_ids": {"type": "array", "items": {"type": "string"}, "description": "List of starting concept family IDs"},
    "max_depth": {"type": "integer", "description": "Max traversal depth (default 2, max 5)"},
}, ["start_family_ids"])

_t("get_concept_mentions", "Get all episodes that mention a given concept. Alias for get_concept_provenance but with a clearer name for the 'which episodes mention this concept' use case.", {
    "family_id": {"type": "string", "description": "Concept family ID"},
}, ["family_id"])


# ── Tool dispatch ─────────────────────────────────────────────────────────

_TOOL_MAP = {}


def _register(fn):
    _TOOL_MAP[fn.__name__] = fn
    return fn


def _arg(args, key, default=None):
    return args.get(key, default)


def _req(args, key):
    """Get required argument with clear error if missing."""
    val = args.get(key)
    if val is None:
        raise ValueError(f"Missing required parameter: {key}")
    if isinstance(val, str) and not val.strip():
        raise ValueError(f"Parameter '{key}' must not be empty")
    return val


def _gid(args):
    """Extract per-call graph_id for HTTP routing. Removes it from args to avoid passing it to API body."""
    return _resolve_graph_id(args)


def _validate_absolute_id(value, param_name="absolute_id"):
    """Check that a value looks like an absolute_id (version ID), not a family_id."""
    if not value or not isinstance(value, str):
        raise ValueError(f"{param_name} is required (use get_entity to find the current absolute_id for an entity)")
    # family_ids start with "ent_" or "rel_" while absolute_ids are UUIDs or longer
    for prefix in ("ent_", "rel_"):
        if value.startswith(prefix) and "-" not in value:
            which = "entity" if prefix == "ent_" else "relation"
            raise ValueError(
                f"'{value}' looks like a family_id, but {param_name} requires an absolute_id (version ID). "
                f"Use get_entity(family_id='{value}') to find the current absolute_id." if prefix == "ent_" else
                f"'{value}' looks like a relation family_id, but {param_name} requires an absolute_id (version ID). "
                f"Use get_entity(family_id=...) for entity absolute_ids."
            )


def _validate_family_id(value, param_name="family_id"):
    """Check that a value looks like a family_id, not an absolute_id (UUID)."""
    if not value or not isinstance(value, str):
        raise ValueError(f"{param_name} is required")
    # UUIDs contain hyphens in 8-4-4-4-12 pattern — family_ids don't
    if len(value) == 36 and value.count("-") == 4:
        raise ValueError(
            f"'{value[:8]}...' looks like an absolute_id (UUID), but {param_name} requires a family_id (e.g. 'ent_abc123' or 'rel_abc123'). "
            f"Use get_entity_by_absolute_id(absolute_id='{value}') if you need to access by version ID."
        )


# ── Remember ──────────────────────────────────────────────────────────────

@_register
def remember(args):
    text = _req(args, "content")
    if len(text.strip()) < 5:
        raise ValueError("content is too short to extract meaningful entities (minimum 5 characters). Combine with surrounding context and retry.")
    body = {"text": text}
    if _arg(args, "source"):
        body["source_name"] = args["source"]
    if _arg(args, "metadata"):
        body["metadata"] = args["metadata"]
    data, code = _post("/api/v1/remember", body)
    if code < 400 and isinstance(data, dict):
        task_id = _inner(data).get("task_id", "")
        if task_id:
            _hint(data, f"\n→ Poll with remember_task_status(task_id='{task_id}') to check extraction progress.")
    return _result(data, code)


@_register
def remember_tasks(args):
    qp = {}
    if _arg(args, "status"):
        qp["status"] = args["status"]
    data, code = _get("/api/v1/remember/tasks", **qp)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        tasks = inner.get("tasks", [])
        if tasks and isinstance(tasks, list) and len(tasks) > 0:
            # Compact tasks to save tokens
            for i, t in enumerate(tasks):
                if isinstance(t, dict):
                    tasks[i] = {k: v for k, v in t.items() if k in ("task_id", "status", "source_name", "phase", "progress", "created_at")}
            first = tasks[0] if isinstance(tasks[0], dict) else {}
            tid = first.get("task_id", "")
            status = first.get("status", "")
            if tid:
                _hint(data, f"\n→ {len(tasks)} task(s). Latest: {status}. Check with remember_task_status(task_id='{tid}').")
        elif isinstance(tasks, list) and len(tasks) == 0:
            _hint(data, "\n→ No tasks in queue. Use remember(content='...') to submit text for extraction.")
    return _result(data, code)


@_register
def remember_task_status(args):
    data, code = _get(f"/api/v1/remember/tasks/{args['task_id']}")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        status = inner.get("status", "")
        if status == "completed":
            parts = ["Extraction complete."]
            # Extract entity names from result if available
            result = inner.get("result", {})
            if isinstance(result, dict):
                entities = result.get("entities", result.get("new_entities", []))
                if isinstance(entities, int):
                    # remember_text() returns integer counts
                    relations = result.get("relations", 0)
                    chunks = result.get("chunks_processed", "?")
                    parts.append(f"Extracted {entities} entities, {relations} relations from {chunks} chunk(s).")
                elif isinstance(entities, list) and entities:
                    names = [e.get("name", "") for e in entities if isinstance(e, dict) and e.get("name")]
                    if names:
                        sample = ", ".join(names[:5])
                        suffix = f" (+{len(names)-5} more)" if len(names) > 5 else ""
                        parts.append(f"Entities: {sample}{suffix}.")
            parts.append("Use quick_search or graph_summary to explore.")
            _hint(data, "\n→ " + " ".join(parts))
        elif status in ("pending", "processing"):
            phase = inner.get("phase", "")
            progress = inner.get("progress", "")
            phase_info = f" (phase: {phase})" if phase else ""
            progress_info = f" ({progress})" if progress else ""
            _hint(data, f"\n→ Still {status}{phase_info}{progress_info}. Poll again with remember_task_status(task_id='{args['task_id']}').")
    return _result(data, code)


@_register
def delete_remember_task(args):
    return _result(*_delete(f"/api/v1/remember/tasks/{args['task_id']}"))


@_register
def pause_remember_task(args):
    return _result(*_post(f"/api/v1/remember/tasks/{args['task_id']}/pause"))


@_register
def resume_remember_task(args):
    return _result(*_post(f"/api/v1/remember/tasks/{args['task_id']}/resume"))


@_register
def remember_monitor(args):
    data, code = _get("/api/v1/remember/monitor")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        pending = inner.get("pending", 0)
        processing = inner.get("processing", 0)
        if pending or processing:
            _hint(data, f"\n→ {pending} pending, {processing} processing. Use remember_tasks(status='pending') to view queue.")
    return _result(data, code)


# ── Health ────────────────────────────────────────────────────────────────

@_register
def health_check(args):
    return _result(*_get("/api/v1/health"))


@_register
def health_check_llm(args):
    return _result(*_get("/api/v1/health/llm"))


# ── Stats ─────────────────────────────────────────────────────────────────

@_register
def search_stats(args):
    return _result(*_get("/api/v1/find/stats"))


@_register
def graph_stats(args):
    return _result(*_get("/api/v1/find/graph-stats"))


# ── Search ────────────────────────────────────────────────────────────────

@_register
def semantic_search(args):
    body = {"query": args["query"]}
    if _arg(args, "top_k"):
        body["max_entities"] = args["top_k"]
        body["max_relations"] = args["top_k"] * 2
    else:
        body["max_entities"] = 10
        body["max_relations"] = 20
    if _arg(args, "mode"):
        mode = args["mode"]
        if mode == "entities":
            body["max_relations"] = 0
        elif mode == "relations":
            body["max_entities"] = 0
    body["expand"] = _arg(args, "expand", False)
    data, code = _post("/api/v1/find", body)
    if code < 400:
        data = _compact_list(data, _compact_entity, "entities")
        data = _compact_list(data, _compact_relation, "relations")
        data = _empty_search_hint(data)
    return _result(data, code)


@_register
def search_candidates(args):
    body = {"query": args["description"], "search_mode": "hybrid"}
    if _arg(args, "top_k"):
        body["max_entities"] = args["top_k"]
    data, code = _post("/api/v1/find", body)
    if code < 400:
        data = _compact_list(data, _compact_entity, "entities")
        data = _empty_search_hint(data, "description")
    return _result(data, code)


@_register
def search_entities(args):
    qp = {"query_name": args["query"]}
    limit = int(args.get("limit", 20))
    offset = int(args.get("offset", 0))
    if _arg(args, "limit"):
        qp["limit"] = str(limit)
    if _arg(args, "offset"):
        qp["offset"] = str(offset)
    data, code = _get("/api/v1/find/entities/search", **qp)
    if code < 400:
        data = _compact_list(data, _compact_entity, "entities")
        data = _pagination_hint(data, "entities", limit, offset)
        data = _empty_search_hint(data)
    return _result(data, code)


@_register
def search_relations(args):
    qp = {"query_text": args["query"]}
    limit = int(args.get("limit", 20))
    if _arg(args, "limit"):
        qp["limit"] = str(limit)
    data, code = _get("/api/v1/find/relations/search", **qp)
    if code < 400:
        data = _compact_list(data, _compact_relation, "relations")
        data = _pagination_hint(data, "relations", limit)
        data = _empty_search_hint(data)
    return _result(data, code)


@_register
def traverse_graph(args):
    seed = args["start_entity_id"]
    seed_ids = seed if isinstance(seed, list) else [seed]
    body = {"seed_family_ids": seed_ids}
    if _arg(args, "max_depth"):
        body["max_depth"] = args["max_depth"]
    if _arg(args, "max_nodes"):
        body["max_nodes"] = args["max_nodes"]
    if _arg(args, "time_point"):
        body["time_point"] = args["time_point"]
    data, code = _post("/api/v1/find/traverse", body)
    if code < 400:
        data = _compact_list(data, _compact_entity, "entities")
        data = _compact_list(data, _compact_relation, "relations")
        inner = _inner(data)
        entities = inner.get("entities", [])
        relations = inner.get("relations", [])
        if isinstance(entities, list) and isinstance(relations, list):
            hint = f"\n→ Traversed {len(entities)} entities, {len(relations)} relations. Use entity_profile to dive deeper into any entity."
            _hint(data, hint)
    return _result(data, code)


# ── Entity Query ──────────────────────────────────────────────────────────

@_register
def list_entities(args):
    qp = {}
    limit = int(args.get("limit", 50))
    offset = int(args.get("offset", 0))
    if _arg(args, "limit"):
        qp["limit"] = str(limit)
    if _arg(args, "offset"):
        qp["offset"] = str(offset)
    data, code = _get("/api/v1/find/entities", **qp)
    if code < 400:
        data = _compact_list(data, _compact_entity, "entities")
        data = _pagination_hint(data, "entities", limit, offset)
        inner = _inner(data)
        entities = inner.get("entities", [])
        if isinstance(entities, list) and len(entities) > 0:
            sample = entities[0]
            sample_name = sample.get("name", "") if isinstance(sample, dict) else ""
            sample_fid = sample.get("family_id", "") if isinstance(sample, dict) else ""
            hint = f"\n→ {len(entities)} entities listed. Use entity_profile(family_id='{sample_fid}') for '{sample_name}' details, or search_entities to filter by content."
            _hint(data, hint)
        elif isinstance(entities, list) and len(entities) == 0:
            _hint(data, "\n→ No entities found. Use remember(content='...') to add text and create entities.")
    return _result(data, code)


@_register
def get_entity(args):
    fid = args['family_id']
    _validate_family_id(fid)
    data, code = _get(f"/api/v1/find/entities/{fid}")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        abs_id = inner.get("absolute_id", "")
        parts = []
        if abs_id:
            parts.append(f"create_relation needs absolute_id='{abs_id[:8]}...'")
        if "relations" not in inner:
            parts.append("entity_profile(family_id='{}') for entity + relations in one call".format(fid))
        if parts:
            _hint(data, "\n→ " + "; ".join(parts))
    return _result(data, code)


@_register
def get_entity_versions(args):
    qp = {}
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/versions", **qp)
    if code < 400:
        data = _compact_list(data, _compact_version, "versions")
        inner = _inner(data)
        versions = inner.get("versions", [])
        if isinstance(versions, list) and len(versions) >= 2:
            hint = f"\n→ {len(versions)} versions found. Use get_entity_version_diff to compare specific versions, or get_entity_timeline for a chronological view with relation events."
            _hint(data, hint)
    return _result(data, code)


@_register
def get_entity_at_time(args):
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/as-of-time", time_point=args["timestamp"])
    if code < 400:
        hint = f"\n→ Compare with current: entity_profile(family_id='{args['family_id']}'). Or see full timeline with get_entity_timeline."
        _hint(data, hint)
    return _result(data, code)


@_register
def get_entity_nearest_to_time(args):
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/nearest-to-time", time_point=args["timestamp"])
    if code < 400:
        _hint(data, f"\n→ For exact time match, use get_entity_at_time. For a range, use get_entity_around_time.")
    return _result(data, code)


@_register
def get_entity_around_time(args):
    qp = {"time_point": args["timestamp"]}
    if _arg(args, "within_seconds"):
        qp["within_seconds"] = str(args["within_seconds"])
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/around-time", **qp)
    if code < 400 and isinstance(data, dict):
        _hint(data, f"\n→ Versions within time window. Use get_entity_version_diff to compare specific versions, or get_entity_timeline for full history.")
    return _result(data, code)


@_register
def get_entity_relations(args):
    qp = {}
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    if _arg(args, "time_point"):
        qp["time_point"] = args["time_point"]
    if _arg(args, "relation_scope"):
        qp["relation_scope"] = args["relation_scope"]
    fid = args['family_id']
    data, code = _get(f"/api/v1/find/entities/{fid}/relations", **qp)
    if code < 400:
        data = _compact_list(data, _compact_relation, "relations")
        inner = _inner(data)
        relations = inner.get("relations", [])
        if isinstance(relations, list) and len(relations) > 0:
            hint = f"\n→ {len(relations)} relations found. For entity details too, use entity_profile(family_id='{fid}') instead."
            _hint(data, hint)
        elif isinstance(relations, list) and len(relations) == 0:
            hint = f"\n→ No relations. Use traverse_graph(start_entity_id='{fid}') to explore nearby entities."
            _hint(data, hint)
    return _result(data, code)


@_register
def get_entity_timeline(args):
    qp = {}
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/timeline", **qp)
    if code < 400:
        data = _compact_list(data, _compact_version, "events")
        inner = _inner(data)
        events = inner.get("events", [])
        if isinstance(events, list) and len(events) > 0:
            hint = f"\n→ {len(events)} timeline events. Use get_entity_version_diff to compare specific versions."
            _hint(data, hint)
    return _result(data, code)


@_register
def get_entity_by_absolute_id(args):
    data, code = _get(f"/api/v1/find/entities/absolute/{args['absolute_id']}")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        fid = inner.get("family_id", "")
        if fid:
            hint = f"\n→ This is a specific version. Use entity_profile(family_id='{fid}') for the current version with all relations."
            _hint(data, hint)
    return _result(data, code)


@_register
def get_entity_version_counts(args):
    data, code = _post("/api/v1/find/entities/version-counts", {"family_ids": args["family_ids"]})
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        counts = inner.get("counts", inner.get("version_counts", {}))
        if isinstance(counts, dict):
            bloated = [k for k, v in counts.items() if isinstance(v, int) and v > 10]
            if bloated:
                hint = f"\n→ {len(bloated)} entities with >10 versions: {', '.join(bloated[:3])}. Consider cleanup_old_versions or merge_entities for consolidation."
                _hint(data, hint)
    return _result(data, code)


# ── Entity CRUD ───────────────────────────────────────────────────────────

@_register
def create_entity(args):
    body = {"name": args["name"]}
    if _arg(args, "content"):
        body["content"] = args["content"]
    for k in ("episode_id", "source_document"):
        if _arg(args, k):
            body[k] = args[k]
    data, code = _post("/api/v1/find/entities/create", body)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        fid = inner.get("family_id", "")
        abs_id = inner.get("absolute_id", "")
        parts = []
        if fid:
            parts.append(f"entity_profile(family_id='{fid}') to view details")
        if abs_id:
            parts.append(f"create_relation(entity1_absolute_id='{abs_id}', ...) to link it")
        if parts:
            hint = "\n→ Next: " + " or ".join(parts) + "."
            _hint(data, hint)
    return _result(data, code)


@_register
def update_entity(args):
    fid = args['family_id']
    _validate_family_id(fid)
    body = {}
    for k in ("name", "summary", "attributes", "source"):
        if _arg(args, k):
            body[k] = args[k]
    data, code = _put(f"/api/v1/find/entities/{fid}", body)
    if code < 400:
        hint = f"\n→ Entity updated. Use entity_profile(family_id='{fid}') to verify."
        _hint(data, hint)
    return _result(data, code)


@_register
def update_entity_by_absolute_id(args):
    _validate_absolute_id(args["absolute_id"])
    body = {}
    for k in ("name", "summary", "attributes"):
        if _arg(args, k):
            body[k] = args[k]
    data, code = _put(f"/api/v1/find/entities/absolute/{args['absolute_id']}", body)
    if code < 400:
        _hint(data, "\n→ Specific version updated. Use entity_profile to see the current state.")
    return _result(data, code)


@_register
def delete_entity(args):
    fid = args['family_id']
    _validate_family_id(fid)
    data, code = _delete(f"/api/v1/find/entities/{fid}")
    if code < 400:
        hint = "\n→ Entity deleted permanently. Related relations are now orphaned — use delete_isolated_entities(dry_run=true) to check."
        _hint(data, hint)
    return _result(data, code)


@_register
def delete_entity_by_absolute_id(args):
    _validate_absolute_id(args["absolute_id"])
    data, code = _delete(f"/api/v1/find/entities/absolute/{args['absolute_id']}")
    if code < 400:
        _hint(data, "\n→ Version deleted (entity may still have other versions). Use get_entity_versions to check remaining versions.")
    return _result(data, code)


@_register
def batch_delete_entities(args):
    ids = args.get("family_ids", [])
    if not ids:
        raise ValueError("family_ids must be a non-empty list of entity family IDs to delete. Use list_entities or find_entity_by_name to find IDs.")
    data, code = _post("/api/v1/find/entities/batch-delete", {"family_ids": ids})
    if code < 400:
        hint = f"\n→ {len(ids)} entities deleted. Use graph_summary to verify counts."
        if isinstance(data, dict):
            _hint(data, hint)
    return _result(data, code)


@_register
def merge_entities(args):
    family_ids = args.get("family_ids", [])
    if len(family_ids) < 2:
        raise ValueError("family_ids must contain at least 2 entity family IDs to merge. Use search_similar_entities to find duplicates.")
    target_id = _arg(args, "target_family_id") or (family_ids[0] if family_ids else "")
    source_ids = [fid for fid in family_ids if fid != target_id]
    body = {"target_family_id": target_id, "source_family_ids": source_ids}
    data, code = _post("/api/v1/find/entities/merge", body)
    if code < 400:
        hint = f"\n→ Merged into {target_id}. Use entity_profile(family_id='{target_id}') to verify the merged result."
        if isinstance(data, dict):
            _hint(data, hint)
    return _result(data, code)


@_register
def split_entity_version(args):
    vid = args.get("version_id", "").strip()
    if not vid:
        raise ValueError("version_id (absolute ID of the version to split) is required. Get it from get_entity_versions.")
    body = {"absolute_id": vid}
    if _arg(args, "new_name"):
        body["new_family_id"] = args["new_name"]
    data, code = _post("/api/v1/find/entities/split-version", body)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        new_fid = inner.get("family_id", "")
        if new_fid:
            hint = f"\n→ Version split into new entity. Use entity_profile(family_id='{new_fid}') to view."
            _hint(data, hint)
    return _result(data, code)


# ── Entity Intelligence ──────────────────────────────────────────────────

@_register
def evolve_entity_summary(args):
    body = {}
    if _arg(args, "context"):
        body["context"] = args["context"]
    data, code = _post(f"/api/v1/find/entities/{args['family_id']}/evolve-summary", body)
    if code < 400:
        fid = args['family_id']
        hint = f"\n→ Summary evolved. Use entity_profile(family_id='{fid}') to see the updated result."
        if isinstance(data, dict):
            _hint(data, hint)
    return _result(data, code)


@_register
def get_entity_contradictions(args):
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/contradictions")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        contradictions = inner.get("contradictions", [])
        if contradictions and isinstance(contradictions, list):
            hint = f"\n→ {len(contradictions)} contradiction(s) found. Use resolve_entity_contradiction to fix."
            _hint(data, hint)
    return _result(data, code)


@_register
def resolve_entity_contradiction(args):
    body = {
        "contradiction": {
            "contradiction_id": args["contradiction_id"],
            "resolution_strategy": args["resolution"],
        }
    }
    data, code = _post(f"/api/v1/find/entities/{args['family_id']}/resolve-contradiction", body)
    if code < 400:
        hint = "\n→ Contradiction resolved. Use entity_profile to verify."
        _hint(data, hint)
    return _result(data, code)


@_register
def get_relation_contradictions(args):
    data, code = _get(f"/api/v1/find/relations/{args['family_id']}/contradictions")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        contradictions = inner if isinstance(inner, list) else inner.get("contradictions", [])
        if contradictions and isinstance(contradictions, list):
            hint = f"\n→ {len(contradictions)} contradiction(s) found. Use resolve_relation_contradiction to fix."
            _hint(data, hint)
    return _result(data, code)


@_register
def resolve_relation_contradiction(args):
    body = {
        "contradiction": {
            "contradiction_id": args["contradiction_id"],
            "resolution_strategy": args["resolution"],
        }
    }
    data, code = _post(f"/api/v1/find/relations/{args['family_id']}/resolve-contradiction", body)
    if code < 400:
        hint = "\n→ Contradiction resolved. Use get_relation_versions to verify."
        _hint(data, hint)
    return _result(data, code)


@_register
def get_entity_provenance(args):
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/provenance")
    if code < 400 and isinstance(data, dict):
        _hint(data, f"\n→ Provenance shows where data came from. Use get_entity_versions(family_id='{args['family_id']}') for the full history.")
    return _result(data, code)


@_register
def get_entity_version_diff(args):
    qp = {}
    if _arg(args, "from_version"):
        qp["v1"] = args["from_version"]
    if _arg(args, "to_version"):
        qp["v2"] = args["to_version"]
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/version-diff", **qp)
    if code < 400:
        _hint(data, f"\n→ Diff shown. Use get_entity_timeline(family_id='{args['family_id']}') for the full change history.")
    return _result(data, code)


@_register
def get_entity_patches(args):
    qp = {}
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/patches", **qp)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        patches = inner.get("patches", [])
        if isinstance(patches, list) and len(patches) > 0:
            hint = f"\n→ {len(patches)} patches. Use get_entity_version_diff to compare two specific versions in detail."
            _hint(data, hint)
    return _result(data, code)


# ── Relation Query ────────────────────────────────────────────────────────

@_register
def list_relations(args):
    qp = {}
    limit = int(args.get("limit", 50))
    offset = int(args.get("offset", 0))
    if _arg(args, "limit"):
        qp["limit"] = str(limit)
    if _arg(args, "offset"):
        qp["offset"] = str(offset)
    if _arg(args, "relation_type"):
        qp["relation_type"] = args["relation_type"]
    data, code = _get("/api/v1/find/relations", **qp)
    if code < 400:
        data = _compact_list(data, _compact_relation, "relations")
        data = _pagination_hint(data, "relations", limit, offset)
        inner = _inner(data)
        relations = inner.get("relations", [])
        if isinstance(relations, list) and len(relations) > 0:
            hint = f"\n→ {len(relations)} relations listed. Use search_relations to filter by content, or get_relations_between to check specific entity pairs."
            _hint(data, hint)
    return _result(data, code)


@_register
def get_relation_by_absolute_id(args):
    data, code = _get(f"/api/v1/find/relations/absolute/{args['absolute_id']}")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        fid = inner.get("family_id", "")
        if fid:
            hint = f"\n→ This is a specific version. Use get_relation_versions(family_id='{fid}') for all versions."
            _hint(data, hint)
    return _result(data, code)


@_register
def get_relation_versions(args):
    qp = {}
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    data, code = _get(f"/api/v1/find/relations/{args['family_id']}/versions", **qp)
    if code < 400:
        data = _compact_list(data, _compact_version, "versions")
        inner = _inner(data)
        versions = inner.get("versions", [])
        if isinstance(versions, list) and len(versions) > 1:
            hint = f"\n→ {len(versions)} versions. Use update_relation(family_id='{args['family_id']}') to modify the current version."
            _hint(data, hint)
    return _result(data, code)


@_register
def get_relations_between(args):
    body = {"family_id_a": args["entity_a"], "family_id_b": args["entity_b"]}
    data, code = _post("/api/v1/find/relations/between", body)
    if code < 400:
        data = _compact_list(data, _compact_relation, "relations")
        inner = _inner(data)
        relations = inner.get("relations", [])
        if isinstance(relations, list) and len(relations) == 0:
            hint = "\n→ No direct relations. Use search_shortest_path to find indirect connections, or traverse_graph to explore neighborhoods."
            _hint(data, hint)
        elif isinstance(relations, list) and len(relations) > 0:
            hint = f"\n→ {len(relations)} direct relation(s) found between these entities."
            _hint(data, hint)
    return _result(data, code)


@_register
def search_shortest_path(args):
    body = {"family_id_a": args["from_entity"], "family_id_b": args["to_entity"]}
    if _arg(args, "max_depth"):
        body["max_depth"] = args["max_depth"]
    data, code = _post("/api/v1/find/paths/shortest", body)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        path = inner.get("path", inner.get("nodes", []))
        if path and isinstance(path, list) and len(path) > 0:
            hint = f"\n→ Path found with {len(path)} nodes. Use traverse_graph or entity_profile to explore intermediate entities."
            _hint(data, hint)
        elif not path:
            hint = "\n→ No path found. Try increasing max_depth or verify both entities exist with find_entity_by_name."
            _hint(data, hint)
    return _result(data, code)


@_register
def search_shortest_path_cypher(args):
    body = {"family_id_a": args["from_entity"], "family_id_b": args["to_entity"]}
    if _arg(args, "max_depth"):
        body["max_depth"] = args["max_depth"]
    data, code = _post("/api/v1/find/paths/shortest-cypher", body)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        path = inner.get("path", inner.get("nodes", []))
        if not path:
            _hint(data, "\n→ No path found. Try increasing max_depth or verify entities with find_entity_by_name.")
    return _result(data, code)


# ── Relation CRUD ─────────────────────────────────────────────────────────

@_register
def create_relation(args):
    e1 = args["entity1_absolute_id"]
    e2 = args["entity2_absolute_id"]
    _validate_absolute_id(e1, "entity1_absolute_id")
    _validate_absolute_id(e2, "entity2_absolute_id")
    content = args.get("content", "").strip()
    if not content:
        raise ValueError("content is required for create_relation (describes how the two entities are related)")
    body = {
        "entity1_absolute_id": e1,
        "entity2_absolute_id": e2,
        "content": content,
    }
    for k in ("episode_id", "source_document"):
        if _arg(args, k):
            body[k] = args[k]
    data, code = _post("/api/v1/find/relations/create", body)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        e1_name = inner.get("entity1_name", "")
        e2_name = inner.get("entity2_name", "")
        if e1_name and e2_name:
            hint = f"\n→ Relation created between '{e1_name}' and '{e2_name}'. Verify with get_relations_between(entity_a=..., entity_b=...)."
        else:
            hint = "\n→ Relation created. Verify with get_relations_between(entity_a=..., entity_b=...)."
        _hint(data, hint)
    return _result(data, code)


@_register
def update_relation(args):
    fid = args['family_id']
    _validate_family_id(fid, "family_id (for relation)")
    body = {}
    if _arg(args, "content"):
        body["content"] = args["content"]
    for k in ("summary", "attributes"):
        if _arg(args, k):
            body[k] = args[k]
    data, code = _put(f"/api/v1/find/relations/{fid}", body)
    if code < 400:
        hint = "\n→ Relation updated. Verify with get_relations_between or get_entity_relations."
        _hint(data, hint)
    return _result(data, code)


@_register
def update_relation_by_absolute_id(args):
    _validate_absolute_id(args["absolute_id"])
    body = {}
    for k in ("content", "relation_type", "summary"):
        if _arg(args, k):
            body[k] = args[k]
    data, code = _put(f"/api/v1/find/relations/absolute/{args['absolute_id']}", body)
    if code < 400:
        _hint(data, "\n→ Relation version updated. Use get_relations_between to verify the connection.")
    return _result(data, code)


@_register
def delete_relation(args):
    fid = args['family_id']
    _validate_family_id(fid, "family_id (for relation)")
    data, code = _delete(f"/api/v1/find/relations/{fid}")
    if code < 400:
        hint = "\n→ Relation deleted permanently."
        _hint(data, hint)
    return _result(data, code)


@_register
def delete_relation_by_absolute_id(args):
    _validate_absolute_id(args["absolute_id"])
    data, code = _delete(f"/api/v1/find/relations/absolute/{args['absolute_id']}")
    if code < 400:
        _hint(data, "\n→ Relation version deleted. Use get_relation_versions to check remaining versions.")
    return _result(data, code)


@_register
def batch_delete_relations(args):
    ids = args.get("family_ids", [])
    if not ids:
        raise ValueError("family_ids must be a non-empty list of relation family IDs to delete. Use search_relations or list_relations to find IDs.")
    data, code = _post("/api/v1/find/relations/batch-delete", {"family_ids": ids})
    if code < 400:
        hint = f"\n→ {len(ids)} relations deleted. Use graph_summary to verify counts."
        if isinstance(data, dict):
            _hint(data, hint)
    return _result(data, code)


@_register
def redirect_relation(args):
    rel_id = args.get("relation_family_id", "").strip()
    new_id = args.get("new_target_id", "").strip()
    if not rel_id or not new_id:
        raise ValueError("Both relation_family_id and new_target_id are required")
    body = {
        "family_id": rel_id,
        "new_family_id": new_id,
    }
    side = _arg(args, "side")
    if side:
        if side in ("source", "entity1"):
            body["side"] = "entity1"
        elif side in ("target", "entity2"):
            body["side"] = "entity2"
        else:
            body["side"] = side
    data, code = _post("/api/v1/find/relations/redirect", body)
    if code < 400:
        hint = f"\n→ Relation redirected to {new_id}. Verify with get_entity_relations."
        _hint(data, hint)
    return _result(data, code)


# ── Episode ──────────────────────────────────────────────────────────

@_register
def get_latest_episode(args):
    data, code = _get("/api/v1/find/episodes/latest")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        cache_id = inner.get("cache_id", "")
        if cache_id:
            hint = f"\n→ Latest episode: {cache_id}. Use get_episode_text(cache_id='{cache_id}') for raw text or search_episodes to find specific content."
            _hint(data, hint)
    return _result(data, code)


@_register
def get_latest_episode_metadata(args):
    data, code = _get("/api/v1/find/episodes/latest/metadata")
    if code < 400 and isinstance(data, dict):
        _hint(data, "\n→ Metadata only. Use get_latest_episode for full content, or search_episodes to find specific episodes.")
    return _result(data, code)


@_register
def get_episode_by_id(args):
    data, code = _get(f"/api/v1/find/episodes/{args['cache_id']}")
    if code < 400 and isinstance(data, dict):
        _hint(data, f"\n→ Episode loaded. Use get_episode_text(cache_id='{args['cache_id']}') for raw text or get_episode_doc for the processed document.")
    return _result(data, code)


@_register
def get_episode_doc(args):
    data, code = _get(f"/api/v1/find/episodes/{args['cache_id']}/doc")
    if code < 400:
        _hint(data, f"\n→ Document content loaded. Use get_episode_text(cache_id='{args['cache_id']}') for raw text.")
    return _result(data, code)


# ── Snapshot/Changes ─────────────────────────────────────────────────────

@_register
def get_snapshot(args):
    qp = {}
    if _arg(args, "timestamp"):
        qp["time"] = args["timestamp"]
    data, code = _get("/api/v1/find/snapshot", **qp)
    if code < 400:
        data = _compact_list(data, _compact_entity, "entities")
        data = _compact_list(data, _compact_relation, "relations")
        inner = _inner(data)
        entities = inner.get("entities", [])
        relations = inner.get("relations", [])
        if isinstance(entities, list) and isinstance(relations, list):
            hint = f"\n→ Snapshot: {len(entities)} entities, {len(relations)} relations. Use quick_search to find specific items."
            _hint(data, hint)
    return _result(data, code)


@_register
def get_changes(args):
    qp = {"since": args["since"]}
    if _arg(args, "until"):
        qp["until"] = str(args["until"])
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    data, code = _get("/api/v1/find/changes", **qp)
    if code < 400:
        data = _compact_list(data, _compact_entity, "entities")
        data = _compact_list(data, _compact_relation, "relations")
        inner = _inner(data)
        entities = inner.get("entities", [])
        relations = inner.get("relations", [])
        if isinstance(entities, list) and isinstance(relations, list):
            total = len(entities) + len(relations)
            if total > 0:
                until_str = args.get("until", "now")
                hint = f"\n→ {len(entities)} entity changes, {len(relations)} relation changes between {args['since']} and {until_str}."
                _hint(data, hint)
    return _result(data, code)


# ── Episodes ──────────────────────────────────────────────────────────────

@_register
def search_episodes(args):
    body = {"query": args["query"]}
    if _arg(args, "limit"):
        body["limit"] = args["limit"]
    data, code = _post("/api/v1/find/episodes/search", body)
    if code < 400 and isinstance(data, dict):
        data = _compact_list(data, _compact_version, "episodes")
        data = _empty_search_hint(data, "query")
    return _result(data, code)


@_register
def delete_episode(args):
    data, code = _delete(f"/api/v1/find/episodes/{args['cache_id']}")
    if code < 400:
        _hint(data, "\n→ Episode deleted. Entities/relations extracted from it are NOT affected — only the episode record is removed.")
    return _result(data, code)


@_register
def batch_ingest_episodes(args):
    episodes = args.get("episodes", [])
    if not episodes:
        raise ValueError("episodes must be a non-empty list of episode objects, each with at least a 'content' field")
    data, code = _post("/api/v1/find/episodes/batch-ingest", {"episodes": episodes})
    if code < 400 and isinstance(data, dict):
        hint = f"\n→ {len(episodes)} episodes submitted. Use remember_tasks to track extraction progress."
        _hint(data, hint)
    return _result(data, code)


# ── Dream ─────────────────────────────────────────────────────────────────

@_register
def get_dream_status(args):
    data, code = _get("/api/v1/find/dream/status")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        status = inner.get("status", "")
        if status == "idle":
            hint = "\n→ Dream engine idle. Use get_dream_seeds to get starting entities, then explore."
            _hint(data, hint)
    return _result(data, code)


@_register
def get_dream_logs(args):
    qp = {}
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    data, code = _get("/api/v1/find/dream/logs", **qp)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        logs = inner.get("logs", inner.get("cycles", []))
        if logs and isinstance(logs, list):
            # Compact logs to essential fields
            for i, log in enumerate(logs):
                if isinstance(log, dict):
                    logs[i] = {k: v for k, v in log.items() if k in ("cycle_id", "id", "strategy", "started_at", "finished_at", "entities_examined_count", "relations_created_count", "summary")}
            if len(logs) > 0:
                first_id = ""
                if isinstance(logs[0], dict):
                    first_id = logs[0].get("cycle_id", logs[0].get("id", ""))
                if first_id:
                    hint = f"\n→ {len(logs)} dream cycles recorded. Use get_dream_log_detail(cycle_id='{first_id}') for details."
                    _hint(data, hint)
    return _result(data, code)


@_register
def get_dream_log_detail(args):
    data, code = _get(f"/api/v1/find/dream/logs/{args['cycle_id']}")
    if code < 400 and isinstance(data, dict):
        _hint(data, "\n→ Dream cycle details. Use get_dream_logs to see all cycles or get_dream_seeds to start a new exploration.")
    return _result(data, code)


@_register
def get_dream_seeds(args):
    body = {}
    if _arg(args, "strategy"):
        body["strategy"] = args["strategy"]
    if _arg(args, "count"):
        body["count"] = args["count"]
    data, code = _post("/api/v1/find/dream/seeds", body)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        seeds = inner.get("seeds", inner.get("entities", []))
        if seeds and isinstance(seeds, list):
            # Compact seed entities
            for i, s in enumerate(seeds):
                if isinstance(s, dict):
                    seeds[i] = {k: v for k, v in s.items() if k in ("family_id", "name", "summary")}
            ids = [s.get("family_id", "") for s in seeds if isinstance(s, dict) and s.get("family_id")]
            if ids:
                hint = f"\n→ {len(ids)} seeds ready. Explore with entity_profile or traverse_graph(start_entity_id='{ids[0]}')."
                _hint(data, hint)
    return _result(data, code)


@_register
def create_dream_relation(args):
    e1 = args.get("entity1_id", "").strip()
    e2 = args.get("entity2_id", "").strip()
    if not e1 or not e2:
        raise ValueError("Both entity1_id and entity2_id (entity family_ids) are required")
    if e1 == e2:
        raise ValueError("entity1_id and entity2_id must be different entities")
    body = {
        "entity1_id": e1,
        "entity2_id": e2,
        "content": args.get("content", ""),
        "reasoning": args.get("reasoning", ""),
        "confidence": args.get("confidence", 0.7),
    }
    if _arg(args, "dream_type"):
        body["dream_type"] = args["dream_type"]
    data, code = _post("/api/v1/find/dream/relation", body)
    if code < 400:
        hint = f"\n→ Dream relation created. Verify with get_relations_between(entity_a='{e1}', entity_b='{e2}')."
        if isinstance(data, dict):
            _hint(data, hint)
    return _result(data, code)


@_register
def save_dream_episode(args):
    content = args["summary"]
    if _arg(args, "insights"):
        content += f"\n\nInsights: {args['insights']}"
    body = {
        "content": content,
        "strategy_used": args["dream_type"],
    }
    if _arg(args, "entities_explored"):
        body["entities_examined"] = args["entities_explored"]
    if _arg(args, "relations_found"):
        # REST API expects a list, but MCP tool sends an int count
        val = args["relations_found"]
        if isinstance(val, int):
            body["relations_created_count"] = val
        else:
            body["relations_created"] = val
    data, code = _post("/api/v1/find/dream/episode", body)
    if code < 400:
        hint = "\n→ Dream episode saved. Use get_dream_logs to review all dream history."
        if isinstance(data, dict):
            _hint(data, hint)
    return _result(data, code)


# ── Agent / Ask ───────────────────────────────────────────────────────────

@_register
def ask(args):
    body = {"question": args["question"]}
    if _arg(args, "context"):
        body["context"] = args["context"]
    data, code = _post("/api/v1/find/ask", body)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        answer = inner.get("answer", inner.get("content", ""))
        if answer and isinstance(answer, str) and len(answer) > 100:
            hint = "\n→ For follow-up questions, use ask again with context from this answer."
            _hint(data, hint)
    return _result(data, code)


@_register
def explain_entity(args):
    body = {"family_id": args["family_id"]}
    if _arg(args, "question"):
        body["aspect"] = args["question"]
    data, code = _post("/api/v1/find/explain", body)
    if code < 400:
        hint = f"\n→ For deeper analysis, try get_entity_timeline(family_id='{args['family_id']}') or get_entity_contradictions."
        _hint(data, hint)
    return _result(data, code)


@_register
def get_suggestions(args):
    qp = {}
    if _arg(args, "entity_id"):
        qp["entity_id"] = args["entity_id"]
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    data, code = _get("/api/v1/find/suggestions", **qp)
    if code < 400 and isinstance(data, dict):
        data = _compact_list(data, _compact_entity, "suggestions")
        inner = _inner(data)
        suggestions = inner.get("suggestions", inner.get("entities", []))
        if suggestions and isinstance(suggestions, list):
            ids = [s.get("family_id", "") for s in suggestions if isinstance(s, dict) and s.get("family_id")]
            if ids:
                sample = ids[0]
                hint = f"\n→ {len(ids)} suggestions. Explore with entity_profile(family_id='{sample}') or traverse_graph."
                _hint(data, hint)
    return _result(data, code)


# ── Communities ───────────────────────────────────────────────────────────

@_register
def detect_communities(args):
    body = {}
    if _arg(args, "algorithm"):
        body["algorithm"] = args["algorithm"]
    if _arg(args, "resolution"):
        body["resolution"] = float(args["resolution"])
    data, code = _post("/api/v1/communities/detect", body)
    if code < 400:
        hint = "\n→ Detection complete. Use list_communities to see all communities, then get_community(cid='...') to inspect members."
        if isinstance(data, dict):
            _hint(data, hint)
    return _result(data, code)


@_register
def list_communities(args):
    data, code = _get("/api/v1/communities")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        comms = inner.get("communities", [])
        if comms and isinstance(comms, list):
            # Compact communities: keep essential fields only
            for i, c in enumerate(comms):
                if isinstance(c, dict):
                    comms[i] = {k: v for k, v in c.items() if k in ("community_id", "cid", "name", "member_count", "internal_relation_count", "summary")}
            if len(comms) > 0:
                hint = f"\n→ {len(comms)} communities detected. Use get_community(cid='...') to inspect members."
                _hint(data, hint)
    return _result(data, code)


@_register
def get_community(args):
    cid = args["cid"]
    data, code = _get(f"/api/v1/communities/{cid}")
    if code < 400 and isinstance(data, dict):
        hint = f"\n→ Use get_community_graph(cid='{cid}') for subgraph visualization."
        _hint(data, hint)
    return _result(data, code)


@_register
def get_community_graph(args):
    data, code = _get(f"/api/v1/communities/{args['cid']}/graph")
    if code < 400:
        data = _compact_list(data, _compact_entity, "entities")
        data = _compact_list(data, _compact_relation, "relations")
        inner = _inner(data)
        entities = inner.get("entities", [])
        relations = inner.get("relations", [])
        if isinstance(entities, list) and isinstance(relations, list):
            _hint(data, f"\n→ Community subgraph: {len(entities)} entities, {len(relations)} relations.")
    return _result(data, code)


@_register
def clear_communities(args):
    data, code = _delete("/api/v1/communities")
    if code < 400:
        _hint(data, "\n→ Community labels cleared. Entities and relations are NOT deleted. Run detect_communities to rebuild.")
    return _result(data, code)


# ── Graphs ────────────────────────────────────────────────────────────────

@_register
def list_graphs(args):
    data, code = _get("/api/v1/graphs")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        graphs_info = inner.get("graphs_info", [])
        graphs = inner.get("graphs", [])
        if isinstance(graphs_info, list) and len(graphs_info) > 0:
            active = _active_graph_id
            parts = []
            for g in graphs_info:
                if not isinstance(g, dict):
                    continue
                gid = g.get("graph_id", "?")
                marker = " (active)" if gid == active else ""
                name = g.get("name", "")
                ec = g.get("entity_count", "?")
                rc = g.get("relation_count", "?")
                label = f"'{gid}'{marker}"
                if name:
                    label += f" [{name}]"
                label += f" ({ec}E/{rc}R)"
                parts.append(label)
            hint = f"\n→ {len(parts)} graph(s): {', '.join(parts)}. Use switch_graph(graph_id='...') to change active graph."
            _hint(data, hint)
        elif isinstance(graphs, list) and len(graphs) > 0:
            active = _active_graph_id
            graph_list = ", ".join(f"'{g}'{' (active)' if g == active else ''}" for g in graphs)
            _hint(data, f"\n→ {len(graphs)} graph(s): {graph_list}. Use switch_graph(graph_id='...') to change active graph.")
    return _result(data, code)


@_register
def create_graph(args):
    body = {"graph_id": args["graph_id"]}
    if _arg(args, "name"):
        body["name"] = args["name"]
    if _arg(args, "description"):
        body["description"] = args["description"]
    data, code = _post("/api/v1/graphs", body)
    if code < 400:
        gid = args["graph_id"]
        hint = f"\n→ Graph '{gid}' created. Use switch_graph(graph_id='{gid}') to start using it."
        if isinstance(data, dict):
            _hint(data, hint)
    return _result(data, code)


@_register
def delete_graph(args):
    gid = _req(args, "graph_id")
    data, code = _delete(f"/api/v1/graphs/{gid}")
    if code < 400:
        # If deleted graph was active, switch back to default
        global _active_graph_id
        was_active = _active_graph_id == gid
        if was_active:
            _active_graph_id = _DEFAULT_GRAPH_ID
        hint = f"\n→ Graph '{gid}' deleted permanently."
        if was_active:
            hint += f" Active graph reset to '{_active_graph_id}'."
        else:
            hint += f" Active graph remains '{_active_graph_id}'."
        if isinstance(data, dict):
            _hint(data, hint)
    return _result(data, code)


@_register
def switch_graph(args):
    """Switch the active graph for all subsequent tool calls."""
    global _active_graph_id
    gid = _req(args, "graph_id")
    # Verify graph exists by listing
    data, code = _get("/api/v1/graphs")
    if code >= 400:
        return _result(data, code)
    inner = _inner(data) if isinstance(data, dict) else {}
    graphs = inner.get("graphs", [])
    if gid not in graphs:
        available = ", ".join(f"'{g}'" for g in graphs) if graphs else "none"
        return _result({"error": f"Graph '{gid}' does not exist. Available: {available}. Use create_graph first."}, 404)
    old = _active_graph_id
    _active_graph_id = gid
    # Get summary of the new graph
    summary_data, _ = _get("/api/v1/find/graph-summary")
    summary_inner = _inner(summary_data) if isinstance(summary_data, dict) else {}
    entity_count = summary_inner.get("entity_count", "?")
    relation_count = summary_inner.get("relation_count", "?")
    result = {
        "active_graph": gid,
        "previous_graph": old,
        "entity_count": entity_count,
        "relation_count": relation_count,
    }
    return {"content": [{"type": "text", "text": f"Switched active graph: '{old}' → '{gid}'\nGraph '{gid}': {entity_count} entities, {relation_count} relations."}]}


@_register
def get_active_graph(args):
    """Get the currently active graph ID and its summary."""
    gid = _active_graph_id
    summary_data, _ = _get("/api/v1/find/graph-summary")
    summary_inner = _inner(summary_data) if isinstance(summary_data, dict) else {}
    entity_count = summary_inner.get("entity_count", "?")
    relation_count = summary_inner.get("relation_count", "?")
    return {"content": [{"type": "text", "text": f"Active graph: '{gid}' (env default: '{_DEFAULT_GRAPH_ID}')\nEntities: {entity_count}, Relations: {relation_count}"}]}


# ── Docs ──────────────────────────────────────────────────────────────────

@_register
def list_docs(args):
    data, code = _get("/api/v1/docs")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        docs = inner.get("docs", inner.get("documents", []))
        if isinstance(docs, list) and len(docs) > 0:
            _hint(data, f"\n→ {len(docs)} documents. Use get_doc_content(filename='...') to read a specific document.")
        elif isinstance(docs, list) and len(docs) == 0:
            _hint(data, "\n→ No documents stored. Documents are created automatically when using remember with a source_document parameter.")
    return _result(data, code)


@_register
def get_doc_content(args):
    return _result(*_get(f"/api/v1/docs/{args['filename']}"))


# ── Neo4j ─────────────────────────────────────────────────────────────────

@_register
def get_entity_neighbors(args):
    qp = {}
    if _arg(args, "direction"):
        qp["direction"] = args["direction"]
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    data, code = _get(f"/api/v1/find/entities/{args['uuid']}/neighbors", **qp)
    if code < 400 and isinstance(data, dict):
        _hint(data, "\n→ Neo4j neighbors. For family_id-based access, use traverse_graph or entity_profile instead.")
    return _result(data, code)


@_register
def list_episodes(args):
    qp = {}
    limit = int(args.get("limit", 20))
    offset = int(args.get("offset", 0))
    if _arg(args, "limit"):
        qp["limit"] = str(limit)
    if _arg(args, "offset"):
        qp["offset"] = str(offset)
    data, code = _get("/api/v1/episodes", **qp)
    if code < 400:
        data = _compact_list(data, _compact_version, "episodes")
        data = _pagination_hint(data, "episodes", limit, offset)
        inner = _inner(data)
        episodes = inner.get("episodes", [])
        if isinstance(episodes, list) and len(episodes) > 0:
            hint = f"\n→ {len(episodes)} episodes. Use search_episodes to filter by content."
            _hint(data, hint)
    return _result(data, code)


@_register
def get_neo4j_episode(args):
    data, code = _get(f"/api/v1/episodes/{args['uuid']}")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        cache_id = inner.get("cache_id", "")
        if cache_id:
            _hint(data, f"\n→ Episode loaded. For cache_id-based access, use get_episode_by_id(cache_id='{cache_id}').")
    return _result(data, code)


# ── Data Quality & Maintenance ────────────────────────────────────────────────

@_register
def delete_isolated_entities(args):
    # Safety: default to dry_run if agent doesn't explicitly set it
    if "dry_run" not in args:
        args["dry_run"] = True
    body = {}
    if _arg(args, "dry_run"):
        body["dry_run"] = args["dry_run"]
    data, code = _post("/api/v1/find/entities/delete-isolated", body)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        deleted = inner.get("deleted_count", 0)
        previewed = inner.get("preview_count", deleted)
        if body.get("dry_run"):
            hint = f"\n→ Preview: {previewed} isolated entities would be deleted. Re-run with dry_run=false to actually delete."
            _hint(data, hint)
        elif deleted:
            hint = f"\n→ Deleted {deleted} isolated entities. Use graph_summary to verify."
            _hint(data, hint)
    return _result(data, code)


@_register
def get_data_quality_report(args):
    data, code = _get("/api/v1/find/quality-report")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        isolated = inner.get("isolated_entities", 0)
        inv_rels = inner.get("invalidated_relations", 0)
        inv_ents = inner.get("invalidated_entities", 0)
        if isolated > 0 or inv_rels > 0 or inv_ents > 0:
            actions = []
            if isolated > 0:
                actions.append("delete_isolated_entities")
            if inv_rels > 0 or inv_ents > 0:
                actions.append("cleanup_old_versions")
            hint = f"\n→ Issues found: {isolated} isolated entities, {inv_ents} invalidated entities, {inv_rels} invalidated relations. Fix with: {', '.join(actions)}."
            _hint(data, hint)
    return _result(data, code)


@_register
def cleanup_old_versions(args):
    # Safety: default to dry_run if agent doesn't explicitly set it
    if "dry_run" not in args:
        args["dry_run"] = True
    body = {}
    if _arg(args, "dry_run"):
        body["dry_run"] = args["dry_run"]
    if _arg(args, "before_date"):
        body["before_date"] = args["before_date"]
    data, code = _post("/api/v1/find/cleanup/invalidated-versions", body)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        removed = inner.get("removed_count", 0)
        if body.get("dry_run"):
            hint = f"\n→ Preview: {removed} invalidated versions would be removed. Re-run with dry_run=false to execute."
            _hint(data, hint)
        elif removed:
            hint = f"\n→ Removed {removed} invalidated versions. Use get_data_quality_report to verify."
            _hint(data, hint)
    return _result(data, code)


@_register
def search_similar_entities(args):
    qp = {"query_name": args.get("name", "")}
    if _arg(args, "similarity_threshold"):
        qp["similarity_threshold"] = str(args["similarity_threshold"])
    qp["search_mode"] = "hybrid"
    qp["max_results"] = "20"
    data, code = _post("/api/v1/find/entities/search", qp)
    if code < 400:
        data = _compact_list(data, _compact_entity, "entities")
        inner = _inner(data)
        entities = inner.get("entities", [])
        if entities and isinstance(entities, list) and len(entities) >= 2:
            hint = f"\n→ {len(entities)} similar entities found. Consider merging duplicates with merge_entities."
            _hint(data, hint)
    return _result(data, code)


# ── System ────────────────────────────────────────────────────────────────

@_register
def system_dashboard(args):
    return _result(*_get("/api/v1/system/dashboard"))


@_register
def system_overview(args):
    return _result(*_get("/api/v1/system/overview"))


@_register
def system_graphs(args):
    return _result(*_get("/api/v1/system/graphs"))


@_register
def system_tasks(args):
    return _result(*_get("/api/v1/system/tasks"))


@_register
def system_logs(args):
    qp = {}
    if _arg(args, "level"):
        qp["level"] = args["level"]
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    return _result(*_get("/api/v1/system/logs", **qp))


@_register
def system_access_stats(args):
    return _result(*_get("/api/v1/system/access-stats"))


# ── Relation Invalidation ──────────────────────────────────────────────────

@_register
def invalidate_relation(args):
    fid = args['family_id']
    _validate_family_id(fid, "family_id (for relation)")
    body = {}
    if _arg(args, "reason"):
        body["reason"] = args["reason"]
    data, code = _post(f"/api/v1/find/relations/{fid}/invalidate", body)
    if code < 400:
        hint = "\n→ Relation invalidated (soft-deleted). Permanently remove with cleanup_old_versions."
        _hint(data, hint)
    return _result(data, code)


@_register
def list_invalidated_relations(args):
    qp = {}
    limit = int(args.get("limit", 100))
    if _arg(args, "limit"):
        qp["limit"] = str(limit)
    data, code = _get("/api/v1/find/relations/invalidated", **qp)
    if code < 400:
        data = _compact_list(data, _compact_relation, "relations")
        data = _pagination_hint(data, "relations", limit)
        inner = _inner(data)
        rels = inner.get("relations", [])
        if isinstance(rels, list) and len(rels) > 0:
            hint = f"\n→ {len(rels)} invalidated relations. Permanently remove with cleanup_old_versions(dry_run=true)."
            _hint(data, hint)
    return _result(data, code)


# ── Version Management ────────────────────────────────────────────────────

@_register
def batch_delete_entity_versions(args):
    ids = args.get("absolute_ids", [])
    if not ids:
        raise ValueError("absolute_ids must be a non-empty list of entity absolute (version) IDs to delete. Get them from get_entity_versions.")
    data, code = _post("/api/v1/find/entities/batch-delete-versions", {"absolute_ids": ids})
    if code < 400:
        hint = f"\n→ {len(ids)} entity versions deleted. Use get_entity_versions to verify remaining versions."
        _hint(data, hint)
    return _result(data, code)


@_register
def batch_delete_relation_versions(args):
    ids = args.get("absolute_ids", [])
    if not ids:
        raise ValueError("absolute_ids must be a non-empty list of relation absolute (version) IDs to delete. Get them from get_relation_versions.")
    data, code = _post("/api/v1/find/relations/batch-delete-versions", {"absolute_ids": ids})
    if code < 400:
        hint = f"\n→ {len(ids)} relation versions deleted. Use get_relation_versions to verify remaining versions."
        _hint(data, hint)
    return _result(data, code)


# ── Section History ──────────────────────────────────────────────────────

@_register
def get_section_history(args):
    qp = {"section": args["section"]}
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/section-history", **qp)
    if code < 400:
        _hint(data, f"\n→ Section history for '{args['section']}'. Use get_entity_versions(family_id='{args['family_id']}') for complete version history.")
    return _result(data, code)


# ── Episode Text ──────────────────────────────────────────────────────────

@_register
def get_episode_text(args):
    data, code = _get(f"/api/v1/find/episodes/{args['cache_id']}/text")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        text = inner.get("text", inner.get("content", ""))
        if isinstance(text, str) and len(text) > 200:
            _hint(data, f"\n→ Raw text: {len(text)} chars. Use get_episode_doc(cache_id='{args['cache_id']}') for the processed document.")
    return _result(data, code)


# ── Isolated Entities ────────────────────────────────────────────────────

@_register
def list_isolated_entities(args):
    qp = {}
    limit = int(args.get("limit", 100))
    offset = int(args.get("offset", 0))
    if _arg(args, "limit"):
        qp["limit"] = str(limit)
    if _arg(args, "offset"):
        qp["offset"] = str(offset)
    data, code = _get("/api/v1/find/entities/isolated", **qp)
    if code < 400:
        data = _compact_list(data, _compact_entity, "entities")
        data = _pagination_hint(data, "entities", limit, offset)
        inner = _inner(data)
        entities = inner.get("entities", [])
        if isinstance(entities, list) and len(entities) > 0:
            hint = f"\n→ {len(entities)} isolated entities. Use delete_isolated_entities(dry_run=true) to preview cleanup, or create_relation to link them."
            _hint(data, hint)
    return _result(data, code)


# ── Aggregation dispatchers ──────────────────────────────────────────────────

@_register
def entity_profile(args):
    fid = args['family_id']
    _validate_family_id(fid)
    data, code = _get(f"/api/v1/find/entities/{fid}/profile")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        abs_id = inner.get("absolute_id", "")
        relations = inner.get("relations", [])
        parts = []
        if abs_id:
            parts.append(f"absolute_id='{abs_id[:8]}...' for create_relation")
        if relations and isinstance(relations, list):
            # Extract unique neighbor IDs for easy reference
            neighbors = set()
            for r in relations:
                if isinstance(r, dict):
                    e1 = r.get("entity1_id", "")
                    e2 = r.get("entity2_id", "")
                    if e1 and e1 != fid:
                        neighbors.add(e1)
                    if e2 and e2 != fid:
                        neighbors.add(e2)
            parts.append(f"{len(relations)} relations, {len(neighbors)} neighbors")
            if neighbors:
                sample = next(iter(neighbors))
                parts.append(f"explore: entity_profile(family_id='{sample}')")
        if parts:
            _hint(data, "\n→ " + "; ".join(parts))
    return _result(data, code)


@_register
def graph_summary(args):
    data, code = _get("/api/v1/find/graph-summary")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        entity_count = inner.get("entity_count", 0)
        relation_count = inner.get("relation_count", 0)
        if entity_count == 0 and relation_count == 0:
            hint = "\n→ Empty graph. Use remember(content='...') to start building your knowledge graph from text."
            _hint(data, hint)
        elif entity_count > 0 and relation_count > 0:
            ratio = entity_count / relation_count if relation_count else 0
            if ratio > 3:
                hint = "\n→ High entity-to-relation ratio. Consider running butler_report to find optimization opportunities."
                _hint(data, hint)
            elif ratio < 0.5:
                hint = "\n→ Dense graph. Use quick_search to find specific information, or detect_communities to discover clusters."
                _hint(data, hint)
        elif entity_count > 0 and relation_count == 0:
            hint = "\n→ Entities exist but no relations. Run butler_report or use remember to add more text and build connections."
            _hint(data, hint)
    return _result(data, code)


@_register
def maintenance_health(args):
    data, code = _get("/api/v1/find/maintenance/health")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        quality = inner.get("quality", inner)
        isolated = quality.get("isolated_entities", 0) if isinstance(quality, dict) else 0
        invalidated = quality.get("invalidated_relations", 0) if isinstance(quality, dict) else 0
        if isolated > 10 or invalidated > 10:
            hint = f"\n→ {isolated} isolated entities, {invalidated} invalidated relations. Run maintenance_cleanup(dry_run=true) to preview cleanup."
            _hint(data, hint)
    return _result(data, code)


@_register
def maintenance_cleanup(args):
    # Safety: default to dry_run if agent doesn't explicitly set it
    if "dry_run" not in args:
        args["dry_run"] = True
    body = {}
    if _arg(args, "dry_run"):
        body["dry_run"] = args["dry_run"]
    data, code = _post("/api/v1/find/maintenance/cleanup", body)
    if code < 400 and isinstance(data, dict):
        if body.get("dry_run"):
            inner = _inner(data)
            preview = inner.get("preview", {})
            ent_del = preview.get("isolated_entities_to_delete", 0)
            ver_del = preview.get("invalidated_versions_to_remove", 0)
            if ent_del or ver_del:
                hint = f"\n→ Preview: {ent_del} isolated entities, {ver_del} invalidated versions would be removed. Re-run with dry_run=false to execute."
            else:
                hint = "\n→ Preview: nothing to clean up. Graph is tidy."
            _hint(data, hint)
        else:
            hint = "\n→ Cleanup complete. Use graph_summary or maintenance_health to verify."
            _hint(data, hint)
    return _result(data, code)


# ── Butler Management ──────────────────────────────────────────────────────

@_register
def butler_report(args):
    """Get a comprehensive health report with AI-generated recommendations for memory graph optimization."""
    data, code = _get("/api/v1/butler/report")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        recommendations = inner.get("recommendations", [])
        if recommendations:
            action_names = [r.get("action", "") for r in recommendations if r.get("action")]
            if action_names:
                names_str = ", ".join(f"'{a}'" for a in action_names)
                hint = f"\n→ Execute with butler_execute(actions=[{names_str}]). Use dry_run=true to preview first."
            else:
                hint = "\n→ Review recommendations above, then execute with butler_execute(actions=[...])."
            _hint(data, hint)
        else:
            hint = "\n→ Graph is healthy. No actions recommended."
            _hint(data, hint)
    return _result(data, code)


@_register
def butler_execute(args):
    """Execute recommended butler actions to optimize the memory graph."""
    actions = args.get("actions", [])
    if isinstance(actions, str):
        actions = [a.strip() for a in actions.split(",") if a.strip()]
    body = {"actions": actions}
    if _arg(args, "dry_run"):
        body["dry_run"] = True
    data, code = _post("/api/v1/butler/execute", body)
    if code < 400 and not args.get("dry_run"):
        hint = "\n→ Execution complete. Use graph_summary or butler_report to verify results."
        if isinstance(data, dict):
            _hint(data, hint)
    return _result(data, code)


# ── Convenience Tools ─────────────────────────────────────────────────────

@_register
def quick_search(args):
    body = {"query": args["query"]}
    for k in ("max_entities", "max_relations", "similarity_threshold"):
        if _arg(args, k):
            body[k] = args[k]
    data, code = _post("/api/v1/find/quick-search", body)
    if code < 400:
        data = _compact_list(data, _compact_entity, "entities")
        data = _compact_list(data, _compact_relation, "relations")
        data = _empty_search_hint(data)
    return _result(data, code)


@_register
def find_entity_by_name(args):
    qp = {}
    if _arg(args, "threshold"):
        qp["threshold"] = str(args["threshold"])
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    data, code = _get(f"/api/v1/find/entities/by-name/{args['name']}", **qp)
    if code < 400 and isinstance(data, dict):
        data = _compact_list(data, _compact_entity, "entities")
        inner = _inner(data)
        # Check if result includes relations (newer API)
        entities = inner.get("entities", [inner] if inner.get("family_id") else [])
        if isinstance(entities, list) and len(entities) > 0:
            best = entities[0] if isinstance(entities[0], dict) else {}
            fid = best.get("family_id", "")
            if fid:
                hint = f"\n→ Found: {best.get('name', args['name'])}. Use entity_profile(family_id='{fid}') for complete details with relations."
                _hint(data, hint)
        elif isinstance(entities, list) and len(entities) == 0:
            _hint(data, f"\n→ No match for '{args['name']}'. Try lowering threshold or use search_entities for broader search.")
    return _result(data, code)


@_register
def batch_profiles(args):
    ids = args.get("family_ids", [])
    if not ids:
        raise ValueError("family_ids must be a non-empty list of entity family IDs (max 20)")
    if len(ids) > 20:
        raise ValueError(f"Too many family_ids ({len(ids)}). Maximum is 20 per call.")
    data, code = _post("/api/v1/find/batch-profiles", {"family_ids": ids})
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        profiles = inner.get("profiles", [])
        if isinstance(profiles, list) and len(profiles) > 0:
            hint = f"\n→ {len(profiles)} profiles loaded. Use get_relations_between to check connections between any pair."
            _hint(data, hint)
    return _result(data, code)


@_register
def recent_activity(args):
    qp = {}
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    data, code = _get("/api/v1/find/recent-activity", **qp)
    if code < 400:
        data = _compact_list(data, _compact_entity, "entities")
        data = _compact_list(data, _compact_relation, "relations")
        inner = _inner(data)
        entities = inner.get("entities", [])
        relations = inner.get("relations", [])
        e_count = len(entities) if isinstance(entities, list) else 0
        r_count = len(relations) if isinstance(relations, list) else 0
        if e_count or r_count:
            hint = f"\n→ Recent: {e_count} new entities, {r_count} new relations. Use entity_profile to explore any item."
            _hint(data, hint)
    return _result(data, code)


# ── Concepts — 统一概念查询处理函数 ──────────────────────────────────────

@_register
def search_concepts(args):
    query = _req(args, "query")
    body = {"query": query}
    if _arg(args, "role"):
        body["role"] = args["role"]
    if _arg(args, "limit"):
        body["limit"] = int(args["limit"])
    data, code = _post("/api/v1/concepts/search", body)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        concepts = inner.get("concepts", [])
        if isinstance(concepts, list) and len(concepts) > 0:
            roles = {}
            for c in concepts:
                r = c.get("role", "unknown")
                roles[r] = roles.get(r, 0) + 1
            parts = [f"{v} {k}(s)" for k, v in sorted(roles.items())]
            hint = f"\n→ Found {len(concepts)} concepts: {', '.join(parts)}. Use get_concept to explore any item."
            _hint(data, hint)
    return _result(data, code)


@_register
def list_concepts(args):
    qp = {}
    if _arg(args, "role"):
        qp["role"] = args["role"]
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    if _arg(args, "offset"):
        qp["offset"] = str(args["offset"])
    data, code = _get("/api/v1/concepts", **qp)
    return _result(data, code)


@_register
def get_concept(args):
    family_id = _req(args, "family_id")
    data, code = _get(f"/api/v1/concepts/{family_id}")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        role = inner.get("role", "unknown")
        name = inner.get("name", "")
        hint = f"\n→ Concept (role={role}"
        if name:
            hint += f", name={name}"
        hint += f"). Use get_concept_neighbors(family_id='{family_id}') to explore connections."
        _hint(data, hint)
    return _result(data, code)


@_register
def get_concept_neighbors(args):
    family_id = _req(args, "family_id")
    max_depth = args.get("max_depth", 1)
    data, code = _get(f"/api/v1/concepts/{family_id}/neighbors", max_depth=str(max_depth))
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        neighbors = inner.get("neighbors", [])
        if isinstance(neighbors, list) and len(neighbors) > 0:
            hint = f"\n→ {len(neighbors)} neighbors found. Use get_concept to explore any neighbor."
            _hint(data, hint)
    return _result(data, code)


@_register
def get_concept_provenance(args):
    family_id = _req(args, "family_id")
    data, code = _get(f"/api/v1/concepts/{family_id}/provenance")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        prov = inner.get("provenance", [])
        if isinstance(prov, list) and len(prov) > 0:
            hint = f"\n→ {len(prov)} source observations found."
            _hint(data, hint)
    return _result(data, code)


@_register
def traverse_concepts(args):
    start_ids = _req(args, "start_family_ids")
    if not isinstance(start_ids, list) or not start_ids:
        raise ValueError("start_family_ids must be a non-empty list of concept family IDs")
    body = {"start_family_ids": start_ids}
    if _arg(args, "max_depth"):
        body["max_depth"] = int(args["max_depth"])
    data, code = _post("/api/v1/concepts/traverse", body)
    return _result(data, code)


@_register
def get_concept_mentions(args):
    family_id = _req(args, "family_id")
    data, code = _get(f"/api/v1/concepts/{family_id}/mentions")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        mentions = inner.get("mentions", [])
        if isinstance(mentions, list) and len(mentions) > 0:
            hint = f"\n→ Mentioned in {len(mentions)} episodes."
            _hint(data, hint)
    return _result(data, code)


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
