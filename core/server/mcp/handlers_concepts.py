#!/usr/bin/env python3
"""
Concept handlers for Deep Dream MCP Server.

Handles: search_concepts, list_concepts, get_concept, get_concept_neighbors,
         get_concept_provenance, traverse_concepts, get_concept_mentions.
"""

from .transport import _get, _post
from .response_format import _result, _hint, _inner
from .dispatch_helpers import _arg, _req


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
        if isinstance(concepts, list) and concepts:
            roles = {}
            for c in concepts:
                r = c.get("role", "unknown")
                roles[r] = roles.get(r, 0) + 1
            parts = [f"{v} {k}(s)" for k, v in sorted(roles.items())]
            hint = f"\n→ Found {len(concepts)} concepts: {', '.join(parts)}. Use get_concept to explore any item."
            _hint(data, hint)
    return _result(data, code)


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


def get_concept_neighbors(args):
    family_id = _req(args, "family_id")
    max_depth = args.get("max_depth", 1)
    data, code = _get(f"/api/v1/concepts/{family_id}/neighbors", max_depth=str(max_depth))
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        neighbors = inner.get("neighbors", [])
        if isinstance(neighbors, list) and neighbors:
            hint = f"\n→ {len(neighbors)} neighbors found. Use get_concept to explore any neighbor."
            _hint(data, hint)
    return _result(data, code)


def get_concept_provenance(args):
    family_id = _req(args, "family_id")
    data, code = _get(f"/api/v1/concepts/{family_id}/provenance")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        prov = inner.get("provenance", [])
        if isinstance(prov, list) and prov:
            hint = f"\n→ {len(prov)} source observations found."
            _hint(data, hint)
    return _result(data, code)


def traverse_concepts(args):
    start_ids = _req(args, "start_family_ids")
    if not isinstance(start_ids, list) or not start_ids:
        raise ValueError("start_family_ids must be a non-empty list of concept family IDs")
    body = {"start_family_ids": start_ids}
    if _arg(args, "max_depth"):
        body["max_depth"] = int(args["max_depth"])
    data, code = _post("/api/v1/concepts/traverse", body)
    return _result(data, code)


def get_concept_mentions(args):
    family_id = _req(args, "family_id")
    data, code = _get(f"/api/v1/concepts/{family_id}/mentions")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        mentions = inner.get("mentions", [])
        if isinstance(mentions, list) and mentions:
            hint = f"\n→ Mentioned in {len(mentions)} episodes."
            _hint(data, hint)
    return _result(data, code)


def register_handlers(tool_map):
    tool_map["search_concepts"] = search_concepts
    tool_map["list_concepts"] = list_concepts
    tool_map["get_concept"] = get_concept
    tool_map["get_concept_neighbors"] = get_concept_neighbors
    tool_map["get_concept_provenance"] = get_concept_provenance
    tool_map["traverse_concepts"] = traverse_concepts
    tool_map["get_concept_mentions"] = get_concept_mentions
