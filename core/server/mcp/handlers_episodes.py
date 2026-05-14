#!/usr/bin/env python3
"""
Episode handlers for Deep Dream MCP Server.

Handles: get_latest_episode, get_latest_episode_metadata, get_episode_by_id,
         get_episode_doc, search_episodes, delete_episode, batch_ingest_episodes,
         get_snapshot, get_changes, get_episode_text,
         list_episodes (Neo4j), get_neo4j_episode.
"""

from .transport import _get, _post, _delete
from .response_format import (
    _result, _hint, _inner,
    _compact_entity, _compact_relation, _compact_version, _compact_list,
    _pagination_hint, _empty_search_hint,
)
from .dispatch_helpers import _arg


def get_latest_episode(args):
    data, code = _get("/api/v1/find/episodes/latest")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        cache_id = inner.get("cache_id", "")
        if cache_id:
            hint = f"\n→ Latest episode: {cache_id}. Use get_episode_text(cache_id='{cache_id}') for raw text or search_episodes to find specific content."
            _hint(data, hint)
    return _result(data, code)


def get_latest_episode_metadata(args):
    data, code = _get("/api/v1/find/episodes/latest/metadata")
    if code < 400 and isinstance(data, dict):
        _hint(data, "\n→ Metadata only. Use get_latest_episode for full content, or search_episodes to find specific episodes.")
    return _result(data, code)


def get_episode_by_id(args):
    data, code = _get(f"/api/v1/find/episodes/{args['cache_id']}")
    if code < 400 and isinstance(data, dict):
        _hint(data, f"\n→ Episode loaded. Use get_episode_text(cache_id='{args['cache_id']}') for raw text or get_episode_doc for the processed document.")
    return _result(data, code)


def get_episode_doc(args):
    data, code = _get(f"/api/v1/find/episodes/{args['cache_id']}/doc")
    if code < 400:
        _hint(data, f"\n→ Document content loaded. Use get_episode_text(cache_id='{args['cache_id']}') for raw text.")
    return _result(data, code)


def search_episodes(args):
    body = {"query": args["query"]}
    if _arg(args, "limit"):
        body["limit"] = args["limit"]
    data, code = _post("/api/v1/find/episodes/search", body)
    if code < 400 and isinstance(data, dict):
        data = _compact_list(data, _compact_version, "episodes")
        data = _empty_search_hint(data, "query")
    return _result(data, code)


def delete_episode(args):
    data, code = _delete(f"/api/v1/find/episodes/{args['cache_id']}")
    if code < 400:
        _hint(data, "\n→ Episode deleted. Entities/relations extracted from it are NOT affected — only the episode record is removed.")
    return _result(data, code)


def batch_ingest_episodes(args):
    episodes = args.get("episodes", [])
    if not episodes:
        raise ValueError("episodes must be a non-empty list of episode objects, each with at least a 'content' field")
    data, code = _post("/api/v1/find/episodes/batch-ingest", {"episodes": episodes})
    if code < 400 and isinstance(data, dict):
        hint = f"\n→ {len(episodes)} episodes submitted. Use remember_tasks to track extraction progress."
        _hint(data, hint)
    return _result(data, code)


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


def get_episode_text(args):
    data, code = _get(f"/api/v1/find/episodes/{args['cache_id']}/text")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        text = inner.get("text", inner.get("content", ""))
        if isinstance(text, str) and len(text) > 200:
            _hint(data, f"\n→ Raw text: {len(text)} chars. Use get_episode_doc(cache_id='{args['cache_id']}') for the processed document.")
    return _result(data, code)


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
        if isinstance(episodes, list) and episodes:
            hint = f"\n→ {len(episodes)} episodes. Use search_episodes to filter by content."
            _hint(data, hint)
    return _result(data, code)


def get_neo4j_episode(args):
    data, code = _get(f"/api/v1/episodes/{args['uuid']}")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        cache_id = inner.get("cache_id", "")
        if cache_id:
            _hint(data, f"\n→ Episode loaded. For cache_id-based access, use get_episode_by_id(cache_id='{cache_id}').")
    return _result(data, code)


def register_handlers(tool_map):
    tool_map["get_latest_episode"] = get_latest_episode
    tool_map["get_latest_episode_metadata"] = get_latest_episode_metadata
    tool_map["get_episode_by_id"] = get_episode_by_id
    tool_map["get_episode_doc"] = get_episode_doc
    tool_map["search_episodes"] = search_episodes
    tool_map["delete_episode"] = delete_episode
    tool_map["batch_ingest_episodes"] = batch_ingest_episodes
    tool_map["get_snapshot"] = get_snapshot
    tool_map["get_changes"] = get_changes
    tool_map["get_episode_text"] = get_episode_text
    tool_map["list_episodes"] = list_episodes
    tool_map["get_neo4j_episode"] = get_neo4j_episode
