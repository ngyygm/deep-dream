#!/usr/bin/env python3
"""
Entity CRUD and query handlers for Deep Dream MCP Server.

Handles: entity listing, get, create, update, delete, merge, split,
         intelligence (contradictions, provenance, patches, evolve),
         version management, time-travel, search, profile, batch operations.
"""

from .transport import _get, _post, _put, _delete
from .response_format import (
    _result, _hint, _inner,
    _compact_entity, _compact_version, _compact_relation, _compact_list,
    _pagination_hint, _empty_search_hint,
)
from .dispatch_helpers import _arg, _req, _validate_family_id, _validate_absolute_id


# ── Health / Stats (used in entity context) ───────────────────────────────

def health_check(args):
    return _result(*_get("/api/v1/health"))


def health_check_llm(args):
    return _result(*_get("/api/v1/health/llm"))


def search_stats(args):
    return _result(*_get("/api/v1/find/stats"))


def graph_stats(args):
    return _result(*_get("/api/v1/find/graph-stats"))


# ── Search ────────────────────────────────────────────────────────────────

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
    body["format"] = "compact"
    data, code = _post("/api/v1/find", body)
    if code < 400:
        data = _compact_list(data, _compact_entity, "entities")
        data = _compact_list(data, _compact_relation, "relations")
        data = _empty_search_hint(data)
    return _result(data, code)


def search_candidates(args):
    body = {"query": args["description"], "search_mode": "hybrid"}
    if _arg(args, "top_k"):
        body["max_entities"] = args["top_k"]
    data, code = _post("/api/v1/find", body)
    if code < 400:
        data = _compact_list(data, _compact_entity, "entities")
        data = _empty_search_hint(data, "description")
    return _result(data, code)


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
        if isinstance(entities, list) and entities:
            sample = entities[0]
            _is_dict = isinstance(sample, dict)
            sample_name = sample.get("name", "") if _is_dict else ""
            sample_fid = sample.get("family_id", "") if _is_dict else ""
            hint = f"\n→ {len(entities)} entities listed. Use entity_profile(family_id='{sample_fid}') for '{sample_name}' details, or search_entities to filter by content."
            _hint(data, hint)
        elif isinstance(entities, list) and not entities:
            _hint(data, "\n→ No entities found. Use remember(content='...') to add text and create entities.")
    return _result(data, code)


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


def get_entity_at_time(args):
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/as-of-time", time_point=args["timestamp"])
    if code < 400:
        hint = f"\n→ Compare with current: entity_profile(family_id='{args['family_id']}'). Or see full timeline with get_entity_timeline."
        _hint(data, hint)
    return _result(data, code)


def get_entity_nearest_to_time(args):
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/nearest-to-time", time_point=args["timestamp"])
    if code < 400:
        _hint(data, "\n→ For exact time match, use get_entity_at_time. For a range, use get_entity_around_time.")
    return _result(data, code)


def get_entity_around_time(args):
    qp = {"time_point": args["timestamp"]}
    if _arg(args, "within_seconds"):
        qp["within_seconds"] = str(args["within_seconds"])
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/around-time", **qp)
    if code < 400 and isinstance(data, dict):
        _hint(data, f"\n→ Versions within time window. Use get_entity_version_diff to compare specific versions, or get_entity_timeline for full history.")
    return _result(data, code)


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
        if isinstance(relations, list) and relations:
            hint = f"\n→ {len(relations)} relations found. For entity details too, use entity_profile(family_id='{fid}') instead."
            _hint(data, hint)
        elif isinstance(relations, list) and not relations:
            hint = f"\n→ No relations. Use traverse_graph(start_entity_id='{fid}') to explore nearby entities."
            _hint(data, hint)
    return _result(data, code)


def get_entity_timeline(args):
    qp = {}
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/timeline", **qp)
    if code < 400:
        data = _compact_list(data, _compact_version, "events")
        inner = _inner(data)
        events = inner.get("events", [])
        if isinstance(events, list) and events:
            hint = f"\n→ {len(events)} timeline events. Use get_entity_version_diff to compare specific versions."
            _hint(data, hint)
    return _result(data, code)


def get_entity_by_absolute_id(args):
    data, code = _get(f"/api/v1/find/entities/absolute/{args['absolute_id']}")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        fid = inner.get("family_id", "")
        if fid:
            hint = f"\n→ This is a specific version. Use entity_profile(family_id='{fid}') for the current version with all relations."
            _hint(data, hint)
    return _result(data, code)


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


def delete_entity(args):
    fid = args['family_id']
    _validate_family_id(fid)
    data, code = _delete(f"/api/v1/find/entities/{fid}")
    if code < 400:
        hint = "\n→ Entity deleted permanently. Related relations are now orphaned — use delete_isolated_entities(dry_run=true) to check."
        _hint(data, hint)
    return _result(data, code)


def delete_entity_by_absolute_id(args):
    _validate_absolute_id(args["absolute_id"])
    data, code = _delete(f"/api/v1/find/entities/absolute/{args['absolute_id']}")
    if code < 400:
        _hint(data, "\n→ Version deleted (entity may still have other versions). Use get_entity_versions to check remaining versions.")
    return _result(data, code)


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


def merge_entities(args):
    family_ids = args.get("family_ids", [])
    if len(family_ids) < 2:
        raise ValueError("family_ids must contain at least 2 entity family IDs to merge. Use search_similar_entities to find duplicates.")
    target_id = _arg(args, "target_family_id") or (family_ids[0] if family_ids else "")
    source_ids = [fid for fid in family_ids if fid != target_id]
    body = {"target_family_id": target_id, "source_family_ids": source_ids}
    if args.get("skip_name_check"):
        body["skip_name_check"] = True
    data, code = _post("/api/v1/find/entities/merge", body)
    if code < 400:
        hint = f"\n→ Merged into {target_id}. Use entity_profile(family_id='{target_id}') to verify the merged result."
        if isinstance(data, dict):
            _hint(data, hint)
    return _result(data, code)


def refresh_graph_edges(args):
    data, code = _post("/api/v1/find/entities/refresh-edges", {})
    return _result(data, code)


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


def get_entity_contradictions(args):
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/contradictions")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        contradictions = inner.get("contradictions", [])
        if contradictions and isinstance(contradictions, list):
            hint = f"\n→ {len(contradictions)} contradiction(s) found. Use resolve_entity_contradiction to fix."
            _hint(data, hint)
    return _result(data, code)


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


def get_relation_contradictions(args):
    data, code = _get(f"/api/v1/find/relations/{args['family_id']}/contradictions")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        contradictions = inner if isinstance(inner, list) else inner.get("contradictions", [])
        if contradictions and isinstance(contradictions, list):
            hint = f"\n→ {len(contradictions)} contradiction(s) found. Use resolve_relation_contradiction to fix."
            _hint(data, hint)
    return _result(data, code)


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


def get_entity_provenance(args):
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/provenance")
    if code < 400 and isinstance(data, dict):
        _hint(data, f"\n→ Provenance shows where data came from. Use get_entity_versions(family_id='{args['family_id']}') for the full history.")
    return _result(data, code)


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


def get_entity_patches(args):
    qp = {}
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/patches", **qp)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        patches = inner.get("patches", [])
        if isinstance(patches, list) and patches:
            hint = f"\n→ {len(patches)} patches. Use get_entity_version_diff to compare two specific versions in detail."
            _hint(data, hint)
    return _result(data, code)


def get_section_history(args):
    qp = {"section": args["section"]}
    data, code = _get(f"/api/v1/find/entities/{args['family_id']}/section-history", **qp)
    if code < 400:
        _hint(data, f"\n→ Section history for '{args['section']}'. Use get_entity_versions(family_id='{args['family_id']}') for complete version history.")
    return _result(data, code)


# ── Data Quality & Maintenance ────────────────────────────────────────────

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
        if isinstance(entities, list) and entities:
            hint = f"\n→ {len(entities)} isolated entities. Use delete_isolated_entities(dry_run=true) to preview cleanup, or create_relation to link them."
            _hint(data, hint)
    return _result(data, code)


def batch_delete_entity_versions(args):
    ids = args.get("absolute_ids", [])
    if not ids:
        raise ValueError("absolute_ids must be a non-empty list of entity absolute (version) IDs to delete. Get them from get_entity_versions.")
    data, code = _post("/api/v1/find/entities/batch-delete-versions", {"absolute_ids": ids})
    if code < 400:
        hint = f"\n→ {len(ids)} entity versions deleted. Use get_entity_versions to verify remaining versions."
        _hint(data, hint)
    return _result(data, code)


# ── Aggregation / Profile ────────────────────────────────────────────────

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


def maintenance_health(args):
    data, code = _get("/api/v1/find/maintenance/health")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        quality = inner.get("quality", inner)
        _q_dict = isinstance(quality, dict)
        isolated = quality.get("isolated_entities", 0) if _q_dict else 0
        invalidated = quality.get("invalidated_relations", 0) if _q_dict else 0
        if isolated > 10 or invalidated > 10:
            hint = f"\n→ {isolated} isolated entities, {invalidated} invalidated relations. Run maintenance_cleanup(dry_run=true) to preview cleanup."
            _hint(data, hint)
    return _result(data, code)


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


def register_handlers(tool_map):
    tool_map["health_check"] = health_check
    tool_map["health_check_llm"] = health_check_llm
    tool_map["search_stats"] = search_stats
    tool_map["graph_stats"] = graph_stats
    tool_map["semantic_search"] = semantic_search
    tool_map["search_candidates"] = search_candidates
    tool_map["search_entities"] = search_entities
    tool_map["search_relations"] = search_relations
    tool_map["traverse_graph"] = traverse_graph
    tool_map["list_entities"] = list_entities
    tool_map["get_entity"] = get_entity
    tool_map["get_entity_versions"] = get_entity_versions
    tool_map["get_entity_at_time"] = get_entity_at_time
    tool_map["get_entity_nearest_to_time"] = get_entity_nearest_to_time
    tool_map["get_entity_around_time"] = get_entity_around_time
    tool_map["get_entity_relations"] = get_entity_relations
    tool_map["get_entity_timeline"] = get_entity_timeline
    tool_map["get_entity_by_absolute_id"] = get_entity_by_absolute_id
    tool_map["get_entity_version_counts"] = get_entity_version_counts
    tool_map["create_entity"] = create_entity
    tool_map["update_entity"] = update_entity
    tool_map["update_entity_by_absolute_id"] = update_entity_by_absolute_id
    tool_map["delete_entity"] = delete_entity
    tool_map["delete_entity_by_absolute_id"] = delete_entity_by_absolute_id
    tool_map["batch_delete_entities"] = batch_delete_entities
    tool_map["merge_entities"] = merge_entities
    tool_map["refresh_graph_edges"] = refresh_graph_edges
    tool_map["split_entity_version"] = split_entity_version
    tool_map["evolve_entity_summary"] = evolve_entity_summary
    tool_map["get_entity_contradictions"] = get_entity_contradictions
    tool_map["resolve_entity_contradiction"] = resolve_entity_contradiction
    tool_map["get_relation_contradictions"] = get_relation_contradictions
    tool_map["resolve_relation_contradiction"] = resolve_relation_contradiction
    tool_map["get_entity_provenance"] = get_entity_provenance
    tool_map["get_entity_version_diff"] = get_entity_version_diff
    tool_map["get_entity_patches"] = get_entity_patches
    tool_map["get_section_history"] = get_section_history
    tool_map["delete_isolated_entities"] = delete_isolated_entities
    tool_map["get_data_quality_report"] = get_data_quality_report
    tool_map["cleanup_old_versions"] = cleanup_old_versions
    tool_map["search_similar_entities"] = search_similar_entities
    tool_map["list_isolated_entities"] = list_isolated_entities
    tool_map["batch_delete_entity_versions"] = batch_delete_entity_versions
    tool_map["entity_profile"] = entity_profile
    tool_map["graph_summary"] = graph_summary
    tool_map["maintenance_health"] = maintenance_health
    tool_map["maintenance_cleanup"] = maintenance_cleanup
