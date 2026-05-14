#!/usr/bin/env python3
"""
Miscellaneous handlers for Deep Dream MCP Server.

Handles: butler, quick search, explore, query, composite workflow tools.
Includes: butler_report, butler_execute, quick_search, find_entity_by_name,
          batch_profiles, recent_activity, ask, explain_entity, get_suggestions,
          remember_and_explore, explore_topic, graph_overview, dream_quick_start.
"""

from .transport import _get, _post, _current_call_graph_id
from .response_format import (
    _result, _hint, _inner,
    _compact_entity, _compact_relation, _compact_list,
    _empty_search_hint,
)
from .dispatch_helpers import _arg, _req


# ── Butler Management ──────────────────────────────────────────────────────

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
        if isinstance(entities, list) and entities:
            best = entities[0] if isinstance(entities[0], dict) else {}
            fid = best.get("family_id", "")
            if fid:
                hint = f"\n→ Found: {best.get('name', args['name'])}. Use entity_profile(family_id='{fid}') for complete details with relations."
                _hint(data, hint)
        elif isinstance(entities, list) and not entities:
            _hint(data, f"\n→ No match for '{args['name']}'. Try lowering threshold or use search_entities for broader search.")
    return _result(data, code)


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
        if isinstance(profiles, list) and profiles:
            hint = f"\n→ {len(profiles)} profiles loaded. Use get_relations_between to check connections between any pair."
            _hint(data, hint)
    return _result(data, code)


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


# ── Agent / Ask ───────────────────────────────────────────────────────────

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


def explain_entity(args):
    body = {"family_id": args["family_id"]}
    if _arg(args, "question"):
        body["aspect"] = args["question"]
    data, code = _post("/api/v1/find/explain", body)
    if code < 400:
        hint = f"\n→ For deeper analysis, try get_entity_timeline(family_id='{args['family_id']}') or get_entity_contradictions."
        _hint(data, hint)
    return _result(data, code)


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


# ── Composite workflow handlers ───────────────────────────────────────────

def remember_and_explore(args):
    text = _req(args, "content")
    if len(text.strip()) < 5:
        raise ValueError("content too short (min 5 chars)")
    body = {"text": text, "wait": True, "timeout": 300}
    if _arg(args, "source"):
        body["source_name"] = args["source"]
    data, code = _post("/api/v1/remember", body)
    if code >= 400:
        return _result(data, code)
    # Now search for what was extracted
    search_data, search_code = _post("/api/v1/find", {"query": text[:200], "search_mode": "hybrid", "max_entities": 10, "max_relations": 10, "format": "compact"})
    result = _inner(data) if isinstance(data, dict) else {}
    search_inner = _inner(search_data) if isinstance(search_data, dict) else {}
    combined = {
        "remember_status": result.get("status", "unknown"),
        "remember_result": result.get("result", {}),
        "extracted_entities": [_compact_entity(e) for e in search_inner.get("entities", [])[:10]],
        "extracted_relations": [_compact_relation(r) for r in search_inner.get("relations", [])[:10]],
    }
    _hint(combined, "\n→ Use entity_profile(family_id=...) to explore any entity in detail.")
    return _result({"success": True, "data": combined}, 200)


def explore_topic(args):
    topic = _req(args, "topic")
    depth = _arg(args, "depth", 2)
    # Step 1: Search
    search_data, search_code = _post("/api/v1/find", {"query": topic, "search_mode": "hybrid", "max_entities": 5, "max_relations": 5, "format": "compact"})
    if search_code >= 400:
        return _result(search_data, search_code)
    search_inner = _inner(search_data) if isinstance(search_data, dict) else {}
    entities = search_inner.get("entities", [])
    relations = search_inner.get("relations", [])

    # Step 2: Traverse from top entities
    traversal_entities = []
    traversal_relations = []
    seed_fids = [e.get("family_id") for e in entities[:3] if e.get("family_id")]
    if seed_fids:
        trav_data, trav_code = _post("/api/v1/find/traverse", {
            "seed_family_ids": seed_fids, "max_depth": min(depth, 4), "max_nodes": 30
        })
        if trav_code < 400:
            trav_inner = _inner(trav_data) if isinstance(trav_data, dict) else {}
            traversal_entities = trav_inner.get("entities", [])
            traversal_relations = trav_inner.get("relations", [])

    combined = {
        "search_results": {
            "entities": [_compact_entity(e) for e in entities[:5]],
            "relations": [_compact_relation(r) for r in relations[:5]],
        },
        "graph_context": {
            "entities": [_compact_entity(e) for e in traversal_entities[:20]],
            "relations": [_compact_relation(r) for r in traversal_relations[:20]],
            "depth": depth,
        },
    }
    _hint(combined, "\n→ Use entity_profile(family_id=...) for details on any entity.")
    return _result({"success": True, "data": combined}, 200)


def graph_overview(args):
    # Get graph summary
    summary_data, _ = _get("/api/v1/find/graph-summary")
    # Get recent activity
    recent_data, _ = _get("/api/v1/find/entities", limit=5, sort="recent")

    summary_inner = _inner(summary_data) if isinstance(summary_data, dict) else {}
    recent_inner = _inner(recent_data) if isinstance(recent_data, dict) else {}

    combined = {
        "graph_id": _current_call_graph_id,
        "stats": {k: v for k, v in summary_inner.items() if k in ("entity_count", "relation_count", "episode_count", "storage_backend", "embedding_available")},
        "recent_entities": [_compact_entity(e) for e in recent_inner.get("entities", [])[:5]],
    }
    return _result({"success": True, "data": combined}, 200)


def dream_quick_start(args):
    """Start dream with smart defaults - checks status first."""
    # Check if dream is already running
    status_data, status_code = _get("/api/v1/find/dream/status")
    if status_code < 400:
        status_inner = _inner(status_data) if isinstance(status_data, dict) else {}
        if status_inner.get("running") or status_inner.get("is_running"):
            _hint(status_data, "\n→ Dream is already running. Use get_dream_status to monitor progress.")
            return _result(status_data, status_code)

    # Start dream with defaults
    body = {
        "max_cycles": _arg(args, "max_cycles", 5),
        "strategies": _arg(args, "strategies", ["free_association", "cross_domain", "leap"]),
        "strategy_mode": "round_robin",
        "confidence_threshold": 0.6,
        "max_tool_calls_per_cycle": 15,
    }
    data, code = _post("/api/v1/find/dream/run", body)
    if code < 400:
        _hint(data, "\n→ Dream started. Use get_dream_status to monitor. Use get_dream_logs to review discoveries.")
    return _result(data, code)


def register_handlers(tool_map):
    tool_map["butler_report"] = butler_report
    tool_map["butler_execute"] = butler_execute
    tool_map["quick_search"] = quick_search
    tool_map["find_entity_by_name"] = find_entity_by_name
    tool_map["batch_profiles"] = batch_profiles
    tool_map["recent_activity"] = recent_activity
    tool_map["ask"] = ask
    tool_map["explain_entity"] = explain_entity
    tool_map["get_suggestions"] = get_suggestions
    tool_map["remember_and_explore"] = remember_and_explore
    tool_map["explore_topic"] = explore_topic
    tool_map["graph_overview"] = graph_overview
    tool_map["dream_quick_start"] = dream_quick_start
