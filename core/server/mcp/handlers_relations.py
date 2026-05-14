#!/usr/bin/env python3
"""
Relation CRUD and query handlers for Deep Dream MCP Server.

Handles: relation listing, get, create, update, delete, redirect,
         path search, invalidation, version management.
"""

from .transport import _get, _post, _put, _delete
from .response_format import (
    _result, _hint, _inner,
    _compact_relation, _compact_version, _compact_list,
    _pagination_hint,
)
from .dispatch_helpers import _arg, _validate_family_id, _validate_absolute_id


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
        if isinstance(relations, list) and relations:
            hint = f"\n→ {len(relations)} relations listed. Use search_relations to filter by content, or get_relations_between to check specific entity pairs."
            _hint(data, hint)
    return _result(data, code)


def get_relation_by_absolute_id(args):
    data, code = _get(f"/api/v1/find/relations/absolute/{args['absolute_id']}")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        fid = inner.get("family_id", "")
        if fid:
            hint = f"\n→ This is a specific version. Use get_relation_versions(family_id='{fid}') for all versions."
            _hint(data, hint)
    return _result(data, code)


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


def get_relations_between(args):
    body = {"family_id_a": args["entity_a"], "family_id_b": args["entity_b"]}
    data, code = _post("/api/v1/find/relations/between", body)
    if code < 400:
        data = _compact_list(data, _compact_relation, "relations")
        inner = _inner(data)
        relations = inner.get("relations", [])
        if isinstance(relations, list) and not relations:
            hint = "\n→ No direct relations. Use search_shortest_path to find indirect connections, or traverse_graph to explore neighborhoods."
            _hint(data, hint)
        elif isinstance(relations, list) and relations:
            hint = f"\n→ {len(relations)} direct relation(s) found between these entities."
            _hint(data, hint)
    return _result(data, code)


def search_shortest_path(args):
    body = {"family_id_a": args["from_entity"], "family_id_b": args["to_entity"]}
    if _arg(args, "max_depth"):
        body["max_depth"] = args["max_depth"]
    data, code = _post("/api/v1/find/paths/shortest", body)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        path = inner.get("path", inner.get("nodes", []))
        if isinstance(path, list) and path:
            hint = f"\n→ Path found with {len(path)} nodes. Use traverse_graph or entity_profile to explore intermediate entities."
            _hint(data, hint)
        elif not path:
            hint = "\n→ No path found. Try increasing max_depth or verify both entities exist with find_entity_by_name."
            _hint(data, hint)
    return _result(data, code)


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


def delete_relation(args):
    fid = args['family_id']
    _validate_family_id(fid, "family_id (for relation)")
    data, code = _delete(f"/api/v1/find/relations/{fid}")
    if code < 400:
        hint = "\n→ Relation deleted permanently."
        _hint(data, hint)
    return _result(data, code)


def delete_relation_by_absolute_id(args):
    _validate_absolute_id(args["absolute_id"])
    data, code = _delete(f"/api/v1/find/relations/absolute/{args['absolute_id']}")
    if code < 400:
        _hint(data, "\n→ Relation version deleted. Use get_relation_versions to check remaining versions.")
    return _result(data, code)


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
        if isinstance(rels, list) and rels:
            hint = f"\n→ {len(rels)} invalidated relations. Permanently remove with cleanup_old_versions(dry_run=true)."
            _hint(data, hint)
    return _result(data, code)


def batch_delete_relation_versions(args):
    ids = args.get("absolute_ids", [])
    if not ids:
        raise ValueError("absolute_ids must be a non-empty list of relation absolute (version) IDs to delete. Get them from get_relation_versions.")
    data, code = _post("/api/v1/find/relations/batch-delete-versions", {"absolute_ids": ids})
    if code < 400:
        hint = f"\n→ {len(ids)} relation versions deleted. Use get_relation_versions to verify remaining versions."
        _hint(data, hint)
    return _result(data, code)


def register_handlers(tool_map):
    tool_map["list_relations"] = list_relations
    tool_map["get_relation_by_absolute_id"] = get_relation_by_absolute_id
    tool_map["get_relation_versions"] = get_relation_versions
    tool_map["get_relations_between"] = get_relations_between
    tool_map["search_shortest_path"] = search_shortest_path
    tool_map["search_shortest_path_cypher"] = search_shortest_path_cypher
    tool_map["create_relation"] = create_relation
    tool_map["update_relation"] = update_relation
    tool_map["update_relation_by_absolute_id"] = update_relation_by_absolute_id
    tool_map["delete_relation"] = delete_relation
    tool_map["delete_relation_by_absolute_id"] = delete_relation_by_absolute_id
    tool_map["batch_delete_relations"] = batch_delete_relations
    tool_map["redirect_relation"] = redirect_relation
    tool_map["invalidate_relation"] = invalidate_relation
    tool_map["list_invalidated_relations"] = list_invalidated_relations
    tool_map["batch_delete_relation_versions"] = batch_delete_relation_versions
