#!/usr/bin/env python3
"""
Dream handlers for Deep Dream MCP Server.

Handles: get_dream_status, get_dream_logs, get_dream_log_detail,
         get_dream_seeds, create_dream_relation, save_dream_episode.
"""

from .transport import _get, _post
from .response_format import _result, _hint, _inner
from .dispatch_helpers import _arg


def get_dream_status(args):
    data, code = _get("/api/v1/find/dream/status")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        status = inner.get("status", "")
        if status == "idle":
            hint = "\n→ Dream engine idle. Use get_dream_seeds to get starting entities, then explore."
            _hint(data, hint)
    return _result(data, code)


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
            if logs:
                first_id = ""
                if isinstance(logs[0], dict):
                    first_id = logs[0].get("cycle_id", logs[0].get("id", ""))
                if first_id:
                    hint = f"\n→ {len(logs)} dream cycles recorded. Use get_dream_log_detail(cycle_id='{first_id}') for details."
                    _hint(data, hint)
    return _result(data, code)


def get_dream_log_detail(args):
    data, code = _get(f"/api/v1/find/dream/logs/{args['cycle_id']}")
    if code < 400 and isinstance(data, dict):
        _hint(data, "\n→ Dream cycle details. Use get_dream_logs to see all cycles or get_dream_seeds to start a new exploration.")
    return _result(data, code)


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


def register_handlers(tool_map):
    tool_map["get_dream_status"] = get_dream_status
    tool_map["get_dream_logs"] = get_dream_logs
    tool_map["get_dream_log_detail"] = get_dream_log_detail
    tool_map["get_dream_seeds"] = get_dream_seeds
    tool_map["create_dream_relation"] = create_dream_relation
    tool_map["save_dream_episode"] = save_dream_episode
