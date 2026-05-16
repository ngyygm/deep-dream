"""
Relation domain operations — redirect, confidence, contradiction, invalidation,
graph stats/summary, and traversal.
"""
from __future__ import annotations

import logging

from flask import current_app, request

from core.find.graph_traversal import GraphTraversalSearcher
from core.server.blueprints import helpers as _h
from core.server.blueprints.relations import relations_bp

ok, err = _h.ok, _h.err
run_async = _h.run_async
_get_processor = _h._get_processor
_get_searcher = _h._get_searcher
_get_graph_id = _h._get_graph_id
relation_to_dict = _h.relation_to_dict
enrich_relations = _h.enrich_relations
entity_to_dict = _h.entity_to_dict
enrich_entity_version_counts = _h.enrich_entity_version_counts
enrich_relation_version_counts = _h.enrich_relation_version_counts
get_json_body = _h.get_json_body
safe_endpoint = _h.safe_endpoint

logger = logging.getLogger(__name__)

_VALID_SIDES = frozenset(("entity1", "entity2"))


# -- Redirect ----------------------------------------------------------------

@relations_bp.route("/api/v1/find/relations/redirect", methods=["POST"])
def redirect_relation():
    """Redirect relation entity endpoint."""
    try:
        processor = _get_processor()
        body = get_json_body()
        family_id = (body.get("family_id") or "").strip()
        side = (body.get("side") or "").strip()
        new_family_id = (body.get("new_family_id") or "").strip()
        if not family_id or not side or not new_family_id:
            return err("family_id, side, new_family_id 为必填", 400)
        if side not in _VALID_SIDES:
            return err("side 必须为 entity1 或 entity2", 400)
        count = processor.storage.redirect_relation(family_id, side, new_family_id)
        return ok({
            "family_id": family_id,
            "side": side,
            "new_family_id": new_family_id,
            "relations_updated": count,
        })
    except Exception as e:
        return err(str(e), 500)


# -- Confidence --------------------------------------------------------------

@relations_bp.route("/api/v1/find/relations/<family_id>/confidence", methods=["PUT"])
def update_relation_confidence(family_id: str):
    """Manually set relation confidence (overrides automatic evolution)."""
    try:
        processor = _get_processor()
        body = get_json_body()
        confidence = body.get("confidence")
        if confidence is None:
            return err("confidence 为必填字段", 400)
        confidence = float(confidence)
        if not (0.0 <= confidence <= 1.0):
            return err("confidence 必须在 0.0 ~ 1.0 之间", 400)
        relation = processor.storage.get_relation_by_family_id(family_id)
        if not relation:
            return err(f"关系不存在: {family_id}", 404)
        processor.storage.update_relation_confidence(family_id, confidence)
        # Patch in-memory instead of re-reading from DB
        relation.confidence = confidence
        return ok(relation_to_dict(relation))
    except Exception as e:
        return err(str(e), 500)


# -- Invalidation ------------------------------------------------------------

@relations_bp.route("/api/v1/find/relations/<family_id>/invalidate", methods=["POST"])
def invalidate_relation(family_id: str):
    """Mark relation as invalidated (not deleted, preserved for history)."""
    try:
        processor = _get_processor()
        body = get_json_body()
        reason = body.get("reason", "")
        count = processor.storage.invalidate_relation(family_id, reason)
        if count == 0:
            return err(f"未找到可失效的关系: {family_id}", 404)
        return ok({"message": f"已标记 {count} 个关系版本为失效", "family_id": family_id})
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/relations/invalidated", methods=["GET"])
def find_invalidated_relations():
    """List all invalidated relations."""
    try:
        processor = _get_processor()
        limit = request.args.get("limit", type=int, default=100)
        relations = processor.storage.get_invalidated_relations(limit)
        dicts = [relation_to_dict(r) for r in relations]
        enrich_relations(dicts, processor)
        return ok(dicts)
    except Exception as e:
        return err(str(e), 500)


# -- Contradictions ----------------------------------------------------------

@relations_bp.route("/api/v1/find/relations/<family_id>/contradictions", methods=["GET"])
def get_relation_contradictions(family_id: str):
    """Detect contradictions between relation versions."""
    try:
        processor = _get_processor()
        versions = processor.storage.get_relation_versions(family_id)
        if len(versions) < 2:
            return ok([])

        contradictions = run_async(
            processor.llm_client.detect_contradictions(family_id, versions, concept_type="relation")
        )

        return ok(contradictions)
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/relations/<family_id>/resolve-contradiction", methods=["POST"])
def resolve_relation_contradiction(family_id: str):
    """Resolve contradictions between relation versions."""
    try:
        body = get_json_body()
        contradiction = body.get("contradiction")
        if not contradiction or not isinstance(contradiction, dict):
            return err("contradiction 为必填字段", 400)

        processor = _get_processor()
        resolution = run_async(
            processor.llm_client.resolve_contradiction(contradiction)
        )

        return ok(resolution)
    except Exception as e:
        return err(str(e), 500)


# -- Graph stats / summary ---------------------------------------------------

@relations_bp.route("/api/v1/find/graph-stats", methods=["GET"])
def find_graph_stats():
    """Graph structure statistics."""
    try:
        processor = _get_processor()
        stats = processor.storage.get_graph_statistics()
        return ok(stats)
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/graph-summary", methods=["GET"])
def graph_summary():
    """Aggregated response: graph statistics + health status."""
    try:
        graph_id = _get_graph_id()
        registry = current_app.config["registry"]
        if graph_id not in registry.list_graphs():
            return err(f"图不存在: {graph_id}", 404)
        processor = _get_processor()
        stats = processor.storage.get_graph_statistics()
        embedding_available = (
            processor.embedding_client is not None
            and processor.embedding_client.is_available()
        )
        storage_backend = "neo4j"
        return ok({
            "graph_id": _get_graph_id(),
            "storage_backend": storage_backend,
            "embedding_available": embedding_available,
            "statistics": stats,
        })
    except Exception as e:
        return err(str(e), 500)


# -- Traversal (BFS + MMR) ---------------------------------------------------

@relations_bp.route("/api/v1/find/traverse", methods=["POST"])
@safe_endpoint
def traverse_graph():
    """BFS graph traversal search."""
    try:
        body = get_json_body()
        seed_ids = body.get("seed_family_ids") or body.get("start_entity_ids", [])
        if not isinstance(seed_ids, list) or not seed_ids:
            return err("seed_family_ids 需为非空数组", 400)
        max_depth = min(max(int(body.get("max_depth", 2)), 1), 5)
        max_nodes = min(max(int(body.get("max_nodes", 50)), 1), 200)
        time_point = body.get("time_point")

        processor = _get_processor()
        searcher = GraphTraversalSearcher(processor.storage)
        entities, relations, visited = searcher.bfs_expand_with_relations(
            seed_ids, max_depth=max_depth, max_nodes=max_nodes,
            time_point=time_point)
        ent_dicts = [entity_to_dict(e) for e in entities]
        rel_dicts = [relation_to_dict(r) for r in relations]
        enrich_entity_version_counts(ent_dicts, processor.storage)
        enrich_relation_version_counts(rel_dicts, processor.storage)
        enrich_relations(rel_dicts, processor)
        return ok({
            "entities": ent_dicts,
            "relations": rel_dicts,
            "visited_count": len(visited),
        })
    except Exception as e:
        return err(str(e), 500)
