"""
Relation search, unified find, candidate lookup, and path-finding routes.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from flask import request

from core.models import Relation
from core.perf import _perf_timer
from core.server.blueprints import helpers as _h
from core.server.blueprints.relations import relations_bp, _shared_pool, _PAREN_ANNOTATION_RE

ok, err = _h.ok, _h.err
_get_processor = _h._get_processor
_get_searcher = _h._get_searcher
_get_graph_id = _h._get_graph_id
entity_to_dict = _h.entity_to_dict
relation_to_dict = _h.relation_to_dict
enrich_relations = _h.enrich_relations
enrich_entity_version_counts = _h.enrich_entity_version_counts
enrich_relation_version_counts = _h.enrich_relation_version_counts
parse_time_point = _h.parse_time_point
_normalize_time_for_compare = _h._normalize_time_for_compare
_extract_candidate_ids = _h._extract_candidate_ids
get_json_body = _h.get_json_body

# Import extracted helper bodies from the companion module
from core.server.blueprints.relations._search_helpers import (
    find_unified_body,
    find_relations_search_body,
    find_relations_by_entity_body,
    create_relation_body,
    quick_search_body,
)

logger = logging.getLogger(__name__)


# =========================================================
# Find: unified semantic retrieval entry point
# =========================================================
@relations_bp.route("/api/v1/find", methods=["POST"])
def find_unified():
    """Unified semantic retrieval: recall relevant sub-graphs from natural language.

    Request body:
        query (str, required): natural language query
        similarity_threshold (float): semantic similarity threshold, default 0.5
        max_entities (int): max entities returned, default 20
        max_relations (int): max relations returned, default 50
        expand (bool): expand neighbourhood from hit entities, default true
        time_before (str, ISO): only return memories before this time
        time_after (str, ISO): only return memories after this time

    Returns:
        entities: matched concept entities
        relations: matched concept relations
    """
    try:
        return find_unified_body()
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/candidates", methods=["POST"])
def find_query_one():
    """Return candidate entities and relations matching request body conditions."""
    try:
        processor = _get_processor()
        body = get_json_body()
        include_entities = body.get("include_entities", True)
        include_relations = body.get("include_relations", True)
        try:
            family_ids, relation_family_ids = _extract_candidate_ids(
                processor.storage, body,
            )
        except ValueError as ve:
            return err(str(ve), 400)
        storage = processor.storage
        entities_data: List[Dict[str, Any]] = []
        relations_data: List[Dict[str, Any]] = []
        if include_entities:
            batch = storage.get_entities_by_absolute_ids(list(family_ids))
            entities_data = [entity_to_dict(e, skip_sections=True) for e in batch if e]
        if include_relations:
            batch_rels = storage.get_relations_by_entity_absolute_ids(list(relation_family_ids))
            for r in batch_rels:
                if r.absolute_id in relation_family_ids:
                    relations_data.append(relation_to_dict(r))
        return ok({"entities": entities_data, "relations": relations_data})
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Relation listing & search
# =========================================================
@relations_bp.route("/api/v1/find/relations", methods=["GET"])
def find_relations_all():
    try:
        processor = _get_processor()
        limit = request.args.get("limit", type=int)
        if limit is not None and limit < 1:
            return err("limit must be a positive integer", 400)
        offset = request.args.get("offset", type=int, default=0) or 0
        total = processor.storage.count_unique_relations()
        relations = processor.storage.get_all_relations(
            limit=limit, offset=offset if offset > 0 else None,
            exclude_embedding=True,
        )
        dicts = [relation_to_dict(r) for r in relations]
        enrich_relations(dicts, processor)
        return ok({
            "relations": dicts,
            "total": total,
            "offset": offset,
            "limit": limit,
        })
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/relations/search", methods=["GET", "POST"])
def find_relations_search():
    try:
        return find_relations_search_body()
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/relations/between", methods=["GET", "POST"])
def find_relations_between():
    try:
        processor = _get_processor()
        body = get_json_body() if request.method == "POST" else None
        body = body if isinstance(body, dict) else {}
        family_id_a = (body.get("family_id_a") or body.get("from_family_id") or body.get("entity1_family_id") or request.args.get("family_id_a") or request.args.get("from_family_id") or request.args.get("entity1_family_id") or "").strip()
        family_id_b = (body.get("family_id_b") or body.get("to_family_id") or body.get("entity2_family_id") or request.args.get("family_id_b") or request.args.get("to_family_id") or request.args.get("entity2_family_id") or "").strip()
        if not family_id_a or not family_id_b:
            return err("family_id_a 与 family_id_b 为必填参数", 400)
        with _perf_timer("find_relations_between"):
            relations = processor.storage.get_relations_by_entities(family_id_a, family_id_b)
        dicts = [relation_to_dict(r) for r in relations]
        enrich_relations(dicts, processor)
        return ok(dicts)
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Path finding
# =========================================================
@relations_bp.route("/api/v1/find/paths/shortest", methods=["GET", "POST"])
def find_shortest_paths():
    """Find shortest paths between two entities."""
    try:
        processor = _get_processor()
        body = get_json_body() if request.method == "POST" else None
        body = body if isinstance(body, dict) else {}
        family_id_a = (body.get("family_id_a") or body.get("from_family_id")
                         or request.args.get("family_id_a")
                         or request.args.get("from_family_id") or "").strip()
        family_id_b = (body.get("family_id_b") or body.get("to_family_id")
                         or request.args.get("family_id_b")
                         or request.args.get("to_family_id") or "").strip()
        if not family_id_a or not family_id_b:
            return err("family_id_a 与 family_id_b 为必填参数", 400)

        max_depth = body.get("max_depth") if body else None
        if max_depth is None:
            max_depth = request.args.get("max_depth", type=int)
        max_depth = max_depth or 6

        max_paths = body.get("max_paths") if body else None
        if max_paths is None:
            max_paths = request.args.get("max_paths", type=int)
        max_paths = max_paths or 10

        # Validate both entities exist
        source_ent = processor.storage.get_entity_by_family_id(family_id_a)
        target_ent = processor.storage.get_entity_by_family_id(family_id_b)
        if not source_ent:
            return err(f"未找到实体: {family_id_a}", 404)
        if not target_ent:
            return err(f"未找到实体: {family_id_b}", 404)

        result = processor.storage.find_shortest_paths(
            source_family_id=family_id_a,
            target_family_id=family_id_b,
            max_depth=max_depth,
            max_paths=max_paths,
        )

        serialized_paths = []
        all_rel_dicts = []
        for p in result.get("paths", []):
            ent_dicts = [entity_to_dict(e) for e in p.get("entities", [])]
            rel_dicts = [relation_to_dict(r) for r in p.get("relations", [])]
            serialized_paths.append({
                "entities": ent_dicts,
                "relations": rel_dicts,
                "length": p.get("length", 0),
            })
            all_rel_dicts.extend(rel_dicts)
        # Batch enrich all relations in a single pass (avoids N DB round-trips)
        if all_rel_dicts:
            enrich_relations(all_rel_dicts, processor)

        return ok({
            "source_entity": entity_to_dict(result["source_entity"]) if result.get("source_entity") else None,
            "target_entity": entity_to_dict(result["target_entity"]) if result.get("target_entity") else None,
            "path_length": result.get("path_length", -1),
            "total_shortest_paths": result.get("total_shortest_paths", 0),
            "paths": serialized_paths,
        })
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/paths/shortest-cypher", methods=["POST"])
def find_shortest_path_cypher():
    """使用 Cypher shortestPath 查找路径。"""
    try:
        processor = _get_processor()
        if not hasattr(processor.storage, 'find_shortest_path_cypher'):
            return err("当前存储后端不支持 Cypher 路径查询", 400)
        body = get_json_body()
        entity_a = (body.get("family_id_a") or body.get("entity_a") or "").strip()
        entity_b = (body.get("family_id_b") or body.get("entity_b") or "").strip()
        if not entity_a or not entity_b:
            return err("family_id_a 和 family_id_b 不能为空", 400)
        max_depth = min(max(int(body.get("max_depth", 6)), 1), 10)
        paths = processor.storage.find_shortest_path_cypher(entity_a, entity_b, max_depth=max_depth)
        return ok({
            "paths": paths,
            "source_family_id": entity_a,
            "target_family_id": entity_b,
        })
    except Exception as e:
        return err(str(e), 500)


# -- Embedding preview -------------------------------------------------------

@relations_bp.route("/api/v1/find/relations/absolute/<absolute_id>/embedding-preview", methods=["GET"])
def find_relation_embedding_preview(absolute_id: str):
    try:
        processor = _get_processor()
        num_values = request.args.get("num_values", type=int, default=5)
        preview = processor.storage.get_relation_embedding_preview(absolute_id, num_values=num_values)
        if preview is None:
            return err(f"未找到关系 embedding 或关系不存在: {absolute_id}", 404)
        return ok({"absolute_id": absolute_id, "values": preview})
    except Exception as e:
        return err(str(e), 500)


# -- Relations by entity absolute_id ----------------------------------------

@relations_bp.route("/api/v1/find/entities/absolute/<entity_absolute_id>/relations", methods=["GET"])
def find_relations_by_entity_absolute_id(entity_absolute_id: str):
    try:
        processor = _get_processor()
        limit = request.args.get("limit", type=int)
        time_point_str = request.args.get("time_point")
        try:
            time_point = parse_time_point(time_point_str)
        except ValueError as ve:
            return err(str(ve), 400)
        relations = processor.storage.get_entity_relations(
            entity_absolute_id=entity_absolute_id,
            limit=limit,
            time_point=time_point,
        )
        dicts = [relation_to_dict(r) for r in relations]
        enrich_relations(dicts, processor)
        return ok(dicts)
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Convenience endpoints for Agent workflows
# =========================================================
@relations_bp.route("/api/v1/find/quick-search", methods=["POST"])
def quick_search():
    """One-shot search: hybrid BM25+embedding RRF fusion with name boosting."""
    try:
        return quick_search_body()
    except Exception as e:
        return err(str(e), 500)
