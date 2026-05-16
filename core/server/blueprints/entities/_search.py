"""
Entity search, SSE streaming, name lookup, profiles, and recent-activity routes.
"""
from __future__ import annotations

import logging
from typing import Any

from flask import request

from core.perf import _perf_timer
from core.server.blueprints import helpers as _h
from core.server.blueprints._constants import _VALID_SEARCH_MODES, _VALID_TEXT_MODES, _VALID_SIM_METHODS
from core.server.blueprints.entities import entities_bp, _shared_pool, _CORE_NAME_RE

ok, err, run_async = _h.ok, _h.err, _h.run_async
safe_endpoint = _h.safe_endpoint
_get_processor = _h._get_processor
_get_searcher = _h._get_searcher

# Import validation helpers
_validate_text_input = _h._validate_text_input
_validate_positive_int = _h._validate_positive_int
get_json_body = _h.get_json_body

logger = logging.getLogger(__name__)


def _sse_stream(total, items, event_name, serialize_fn):
    """Generic SSE stream generator: meta -> N x event -> done."""
    from core.server.sse import sse_event
    try:
        yield sse_event("meta", {"total": total})
        for item in items:
            yield sse_event(event_name, serialize_fn(item))
        yield sse_event("done", {"status": "completed"})
    except GeneratorExit:
        pass
    except Exception as e:
        logger.warning("%s stream error: %s", event_name, e)
        try:
            yield sse_event("error", {"message": str(e)})
        except (GeneratorExit, StopIteration):
            pass


# -- Entity listing ---------------------------------------------------------

@entities_bp.route("/api/v1/find/entities", methods=["GET"])
@safe_endpoint
def find_entities_all():
    try:
        processor = _get_processor()
        limit = min(request.args.get("limit", type=int) or 500, 5000)
        offset = request.args.get("offset", type=int, default=0) or 0
        total = processor.storage.count_unique_entities()
        entities = processor.storage.get_all_entities(limit=limit, offset=offset if offset > 0 else None, exclude_embedding=True)
        h = _h
        # Batch version counts
        family_ids = [e.family_id for e in entities if e.family_id]
        vc_map = processor.storage.get_entity_version_counts(family_ids) if family_ids else {}
        return ok({
            "entities": [h.entity_to_dict(e, version_count=vc_map.get(e.family_id), skip_sections=True) for e in entities],
            "total": total,
            "offset": offset,
            "limit": limit,
        })
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/graph/stream/entities", methods=["GET"])
def stream_graph_entities():
    """SSE streaming -- yields entities one by one.  Events: meta -> N x entity -> done.

    Query params:
      since -- ISO timestamp; if given, only stream entities modified after this time.
    """
    from core.server.sse import sse_event, sse_response

    processor = _get_processor()
    store = processor.storage
    h = _h
    since = request.args.get("since")

    def generate():
        total = store.count_entities_since(since) if since else store.count_unique_entities()
        yield from _sse_stream(
            total,
            store.stream_all_entities(exclude_embedding=True, since=since),
            "entity",
            lambda item: h.entity_to_dict(item[0], skip_sections=True, version_count=item[1]),
        )

    return sse_response(generate())


@entities_bp.route("/api/v1/find/graph/stream/relations", methods=["GET"])
def stream_graph_relations():
    """SSE streaming -- yields relations one by one.  Events: meta -> N x relation -> done.

    Query params:
      since -- ISO timestamp; if given, only stream relations modified after this time.
    """
    from core.server.sse import sse_event, sse_response

    processor = _get_processor()
    store = processor.storage
    h = _h
    since = request.args.get("since")

    def generate():
        name_map = store.get_all_entity_names_map()

        def _serialize(rel):
            d = h.relation_to_dict(rel)
            d['entity1_name'] = name_map.get(rel.entity1_absolute_id, '')
            d['entity2_name'] = name_map.get(rel.entity2_absolute_id, '')
            return d

        total = store.count_relations_since(since) if since else store.count_unique_relations()
        yield from _sse_stream(
            total,
            store.stream_all_relations(exclude_embedding=True, since=since),
            "relation",
            _serialize,
        )

    return sse_response(generate())


@entities_bp.route("/api/v1/find/graph/version", methods=["GET"])
@safe_endpoint
def graph_version():
    """Lightweight version endpoint -- returns counts and last_modified timestamp."""
    processor = _get_processor()
    store = processor.storage
    counts = store.get_graph_version()
    return ok(counts)


@entities_bp.route("/api/v1/find/entities/as-of-time", methods=["GET"])
@safe_endpoint
def find_entities_all_before_time():
    try:
        processor = _get_processor()
        h = _h
        time_point_str = request.args.get("time_point")
        if not time_point_str:
            return err("time_point 为必填参数（ISO 格式）", 400)
        try:
            time_point = h.parse_time_point(time_point_str)
        except ValueError as ve:
            return err(str(ve), 400)
        limit = request.args.get("limit", type=int)
        entities = processor.storage.get_all_entities_before_time(time_point, limit=limit, exclude_embedding=True)
        return ok([h.entity_to_dict(e, skip_sections=True) for e in entities])
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/version-counts", methods=["POST"])
@safe_endpoint
def find_entity_version_counts():
    try:
        processor = _get_processor()
        body = get_json_body()
        if not isinstance(body, dict):
            body = {}
        family_ids = body.get("family_ids")
        if not family_ids or not isinstance(family_ids, list):
            return ok({})
        family_ids = [x for x in family_ids if isinstance(x, str)]
        if not family_ids:
            return ok({})
        counts = processor.storage.get_entity_version_counts(family_ids)
        return ok(counts)
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/absolute/<absolute_id>/embedding-preview", methods=["GET"])
@safe_endpoint
def find_entity_embedding_preview(absolute_id: str):
    try:
        processor = _get_processor()
        num_values = request.args.get("num_values", type=int, default=5)
        preview = processor.storage.get_entity_embedding_preview(absolute_id, num_values=num_values)
        if preview is None:
            return err(f"未找到实体 embedding 或实体不存在: {absolute_id}", 404)
        return ok({"absolute_id": absolute_id, "values": preview})
    except Exception as e:
        return err(str(e), 500)


# -- Entity search -----------------------------------------------------------

@entities_bp.route("/api/v1/find/entities/search", methods=["GET", "POST"])
@safe_endpoint
def find_entities_search():
    try:
        processor = _get_processor()
        h = _h
        body = get_json_body() if request.method == "POST" else None
        body = body if isinstance(body, dict) else {}

        def _get_value(name: str, default: Any = None) -> Any:
            if name in body and body[name] is not None:
                return body[name]
            return request.args.get(name, default)

        query_name = str(_get_value("query_name", "") or "").strip()
        if not query_name:
            return err("query_name 为必填参数", 400)

        # Validate query_name input
        _validate_text_input(query_name, "query_name", min_len=1, max_len=1000)

        query_content = _get_value("query_content") or None
        if query_content:
            # Validate query_content if provided
            query_content = _validate_text_input(query_content, "query_content", min_len=0, max_len=10000)

        threshold = float(_get_value("similarity_threshold") or _get_value("threshold", 0.5))
        max_results_raw = _get_value("max_results", 10)
        max_results = _validate_positive_int(max_results_raw, "max_results")
        max_results = min(max_results, 500)  # Cap at 500

        text_mode = str(_get_value("text_mode", "name_and_content") or "name_and_content")
        if text_mode not in _VALID_TEXT_MODES:
            text_mode = "name_and_content"
        similarity_method = str(_get_value("similarity_method", "embedding") or "embedding")
        if similarity_method not in _VALID_SIM_METHODS:
            similarity_method = "embedding"
        content_snippet_length = int(_get_value("content_snippet_length", 50))

        search_mode = str(_get_value("search_mode", "hybrid") or "hybrid").strip().lower()
        if search_mode not in _VALID_SEARCH_MODES:
            search_mode = "hybrid"

        searcher = _get_searcher(processor.storage)
        if search_mode == "hybrid":
            ranked = searcher.search_entities(
                query_text=query_name,
                top_k=max_results,
                semantic_threshold=threshold,
            )
        else:
            if search_mode == "bm25":
                entities = processor.storage.search_entities_by_bm25(
                    query_name, limit=max_results
                )
            else:
                entities = processor.storage.search_entities_by_similarity(
                    query_name=query_name,
                    query_content=query_content,
                    threshold=threshold,
                    max_results=max_results,
                    content_snippet_length=content_snippet_length,
                    text_mode=text_mode,
                    similarity_method=similarity_method,
                )
            ranked = [(e, 1.0 - i * 0.01) for i, e in enumerate(entities)]
        ranked = searcher.confidence_rerank(ranked, alpha=0.2, time_decay_half_life_days=90.0)

        # Fallback: if no results and query looks like an entity name, try prefix match
        if not ranked and 1 <= len(query_name) <= 50 and hasattr(processor.storage, 'find_entity_by_name_prefix'):
            prefix_matches = processor.storage.find_entity_by_name_prefix(query_name, limit=max_results)
            ranked = [(e, 0.6) for e in prefix_matches]
        dicts = [h.entity_to_dict(e, _score=score, skip_sections=True) for e, score in ranked]
        h.enrich_entity_version_counts(dicts, processor.storage)
        return ok(dicts)
    except ValueError as ve:
        return err(str(ve), 400)
    except Exception as e:
        return err(str(e), 500)


# -- Entity lookup by name ---------------------------------------------------

@entities_bp.route("/api/v1/find/entities/by-name/<name>", methods=["GET"])
@safe_endpoint
def find_entity_by_name(name: str):
    try:
        processor = _get_processor()
        h = _h
        threshold = float(request.args.get("threshold", "0.5"))
        limit = int(request.args.get("limit", "5"))
        best = None
        match_method = "none"

        # Step 1+2: Combined exact + core-name match in single lookup
        core = _CORE_NAME_RE.sub('', name).strip()
        lookup_names = [name]
        if core and core != name:
            lookup_names.append(core)
        name_map = processor.storage.get_family_ids_by_names(lookup_names)
        if name_map:
            fid = list(name_map.values())[0]
            best = processor.storage.get_entity_by_family_id(fid)
            best._score = 1.0
            match_method = "exact"

        # Step 2B: Prefix match -- find entities whose name starts with query + "("
        # Handles searching for short name that has parenthetical annotation
        # Also handles title prefix differences (Dr., Prof., etc.)
        if not best and hasattr(processor.storage, 'find_entity_by_name_prefix'):
            try:
                prefix_matches = processor.storage.find_entity_by_name_prefix(name, limit=5)
                for candidate in prefix_matches:
                    cname = getattr(candidate, 'name', '')
                    ccore = _CORE_NAME_RE.sub('', cname).strip()
                    # Exact core-name match or parenthetical pattern
                    if ccore == name or cname.startswith(name + '（') or cname.startswith(name + '('):
                        best = candidate
                        best._score = 0.95
                        match_method = 'prefix_exact'
                        break
                    # Substring containment: query is contained in entity name
                    if name.lower() in cname.lower():
                        best = candidate
                        best._score = 0.9
                        match_method = 'prefix'
                        break
            except Exception as e:
                logger.debug("by-name prefix match failed for '%s': %s", name, e)

        # Step 3: BM25 text search
        if not best:
            try:
                entities = processor.storage.search_entities_by_bm25(name, limit=1)
                if entities:
                    candidate = entities[0]
                    score = getattr(candidate, '_score', 0) or 0
                    if score >= 0.7:
                        best = candidate
                        match_method = "bm25"
            except Exception as e:
                logger.debug("by-name BM25 failed for '%s': %s", name, e)

        # Step 4: Embedding fallback (semantic, slower)
        if not best:
            entities = processor.storage.search_entities_by_similarity(
                query_name=name,
                query_content=name,
                threshold=threshold,
                max_results=limit,
                text_mode="name_only",
                similarity_method="embedding",
            )
            if entities:
                candidate = entities[0]
                score = getattr(candidate, '_score', 0) or 0
                if score >= threshold:
                    best = candidate

                    match_method = "embedding"
        if not best:
            return err(f"No entity found matching '{name}'", 404)
        rels = processor.storage.get_entity_relations_by_family_id(best.family_id)
        rel_dicts = [h.relation_to_dict(r) for r in rels]
        h.enrich_relations(rel_dicts, processor)
        vc_map = processor.storage.get_entity_version_counts([best.family_id])
        best_score = getattr(best, '_score', 0) or 0
        result = {
            "entity": h.entity_to_dict(best),
            "relations": rel_dicts,
            "relation_count": len(rel_dicts),
            "version_count": vc_map.get(best.family_id, 1),
            "match_score": best_score,
            "match_method": match_method,
        }
        if match_method in ("bm25", "embedding"):
            result["hint"] = "Fuzzy match — verify this is the intended entity"
        return ok(result)
    except Exception as e:
        return err(str(e), 500)


# -- Batch profiles ----------------------------------------------------------

@entities_bp.route("/api/v1/find/batch-profiles", methods=["POST"])
@safe_endpoint
def batch_profiles():
    try:
        processor = _get_processor()
        h = _h
        body = get_json_body()
        family_ids = body.get("family_ids", [])
        if not family_ids:
            return err("family_ids is required", 400)
        if len(family_ids) > 20:
            return err("Maximum 20 entities per batch", 400)

        batch_results = processor.storage.batch_get_entity_profiles(family_ids)
        profiles = []
        for item in batch_results:
            entity = item.get("entity")
            relations = item.get("relations", [])
            rel_dicts = [h.relation_to_dict(r) for r in relations]
            h.enrich_relations(rel_dicts, processor)
            profiles.append({
                "family_id": item["family_id"],
                "entity": h.entity_to_dict(entity) if entity else None,
                "relations": rel_dicts,
                "relation_count": len(rel_dicts),
                "version_count": item.get("version_count", 0),
            })
        return ok({"profiles": profiles, "count": len(profiles)})
    except Exception as e:
        return err(str(e), 500)


# -- Recent activity ---------------------------------------------------------

@entities_bp.route("/api/v1/find/recent-activity", methods=["GET"])
@safe_endpoint
def recent_activity():
    try:
        processor = _get_processor()
        h = _h
        limit = min(int(request.args.get("limit", "10")), 50)

        # Run 3 independent queries in parallel (using shared pool)
        ent_fut = _shared_pool.submit(processor.storage.get_all_entities, limit, True)
        rel_fut = _shared_pool.submit(processor.storage.get_all_relations, limit, True)
        stat_fut = _shared_pool.submit(processor.storage.get_graph_statistics)
        latest_entities = ent_fut.result()
        latest_relations = rel_fut.result()
        stats = stat_fut.result()

        entity_dicts = [h.entity_to_dict(e) for e in reversed(latest_entities)]
        rel_dicts = [h.relation_to_dict(r) for r in reversed(latest_relations)]
        h.enrich_relations(rel_dicts, processor)

        return ok({
            "statistics": stats,
            "latest_entities": entity_dicts,
            "latest_relations": rel_dicts,
        })
    except Exception as e:
        return err(str(e), 500)
