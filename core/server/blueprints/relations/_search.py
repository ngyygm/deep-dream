"""
Relation search, unified find, candidate lookup, and path-finding routes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from flask import request

from core.models import Entity, Relation
from core.find.hybrid import HybridSearcher
from core.perf import _perf_timer
from core.server.blueprints import helpers as _h
from core.server.blueprints._constants import _VALID_SEARCH_MODES
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
        body = request.get_json(silent=True) or {}
        query = (body.get("query") or "").strip()
        if not query:
            return err("query 为必填字段", 400)

        similarity_threshold = float(body.get("similarity_threshold", 0.5))
        max_entities = int(body.get("max_entities", 20))
        max_relations = int(body.get("max_relations", 50))
        expand = body.get("expand", True)
        time_before = body.get("time_before")
        time_after = body.get("time_after")
        reranker = (body.get("reranker", "rrf") or "rrf").strip().lower()

        search_mode = (body.get("search_mode", "hybrid") or "hybrid").strip().lower()
        if search_mode not in _VALID_SEARCH_MODES:
            search_mode = "hybrid"

        try:
            time_before_dt = parse_time_point(time_before) if time_before else None
            time_after_dt = parse_time_point(time_after) if time_after else None
        except ValueError as ve:
            return err(str(ve), 400)

        processor = _get_processor()
        storage = processor.storage

        # Create HybridSearcher once, shared for entity and relation search
        _hybrid_searcher = _get_searcher(storage) if search_mode == "hybrid" else None

        # --- Step 1+2: Entity and relation recall run in parallel ---
        entity_score_map: Dict[str, float] = {}
        relation_score_map: Dict[str, float] = {}

        def _recall_entities():
            """Step 1: Entity recall by search_mode."""
            with _perf_timer("find_unified | step1_entity_recall"):
                if search_mode == "bm25":
                    entities = storage.search_entities_by_bm25(query, limit=max_entities)
                elif search_mode == "hybrid":
                    hybrid_entities = _hybrid_searcher.search_entities(
                        query_text=query, top_k=max_entities,
                        semantic_threshold=similarity_threshold,
                    )
                    entities = []
                    for e, score in hybrid_entities:
                        entities.append(e)
                        entity_score_map[e.absolute_id] = score
                else:
                    entities = storage.search_entities_by_similarity(
                        query_name=query, query_content=query,
                        threshold=similarity_threshold,
                        max_results=max_entities,
                        text_mode="name_and_content",
                        similarity_method="embedding",
                    )

                # Core name prefix match supplement
                if len(query) >= 2 and len(query) <= 20 and entities:
                    seen_fids = {getattr(e, 'family_id', '') for e in entities}
                    _has_core = any(
                        _PAREN_ANNOTATION_RE.sub('', getattr(e, 'name', '')).strip() == query
                        or getattr(e, 'name', '').startswith(query + '（')
                        or getattr(e, 'name', '').startswith(query + '(')
                        for e in entities
                    )
                    if not _has_core:
                        prefix_matches = storage.find_entity_by_name_prefix(query, limit=3)
                        for e in prefix_matches:
                            fid = getattr(e, 'family_id', '')
                            if fid and fid not in seen_fids:
                                seen_fids.add(fid)
                                entities.insert(0, e)
            return entities

        def _recall_relations():
            """Step 2: Relation recall by search_mode."""
            with _perf_timer("find_unified | step2_relation_recall"):
                if search_mode == "bm25":
                    rels = storage.search_relations_by_bm25(query, limit=max_relations)
                elif search_mode == "hybrid":
                    hybrid_relations = _hybrid_searcher.search_relations(
                        query_text=query, top_k=max_relations,
                        semantic_threshold=similarity_threshold,
                    )
                    rels = []
                    for r, score in hybrid_relations:
                        rels.append(r)
                        relation_score_map[r.absolute_id] = score
                else:
                    rels = storage.search_relations_by_similarity(
                        query_text=query,
                        threshold=similarity_threshold,
                        max_results=max_relations,
                    )
            return rels

        _ent_fut = _shared_pool.submit(_recall_entities)
        _rel_fut = _shared_pool.submit(_recall_relations)
        matched_entities = _ent_fut.result()
        matched_relations = _rel_fut.result()

        entity_abs_ids: Set[str] = {e.absolute_id for e in matched_entities}
        relation_abs_ids: Set[str] = {r.absolute_id for r in matched_relations}
        entities_by_abs: Dict[str, Entity] = {e.absolute_id: e for e in matched_entities}

        # --- Step 3: Supplement associated entities from semantically matched relations (batch) ---
        with _perf_timer("find_unified | step3_entity_completion"):
            missing_abs_ids = set()
            missing_source_scores: Dict[str, float] = {}  # abs_id -> best relation score
            for r in list(matched_relations):
                r_score = relation_score_map.get(r.absolute_id, 0.0)
                for abs_id in (r.entity1_absolute_id, r.entity2_absolute_id):
                    if abs_id not in entity_abs_ids:
                        missing_abs_ids.add(abs_id)
                        if abs_id not in missing_source_scores or r_score > missing_source_scores[abs_id]:
                            missing_source_scores[abs_id] = r_score
            if missing_abs_ids:
                batch_entities = storage.get_entities_by_absolute_ids(list(missing_abs_ids))
                for e in batch_entities:
                    if e:
                        entities_by_abs[e.absolute_id] = e
                        entity_abs_ids.add(e.absolute_id)
                        if e.absolute_id not in entity_score_map:
                            entity_score_map[e.absolute_id] = missing_source_scores.get(e.absolute_id, 0.0) * 0.5

        # --- Step 4: Graph neighbourhood expansion ---
        with _perf_timer("find_unified | step4_graph_expansion"):
            if expand and entity_abs_ids:
                expanded_rels = storage.get_relations_by_entity_absolute_ids(
                    list(entity_abs_ids), limit=max_relations
                )
                expand_missing = set()
                for r in expanded_rels:
                    if r.absolute_id not in relation_abs_ids:
                        relation_abs_ids.add(r.absolute_id)
                        matched_relations.append(r)
                        if r.absolute_id not in relation_score_map:
                            e1_score = entity_score_map.get(r.entity1_absolute_id, 0.0)
                            e2_score = entity_score_map.get(r.entity2_absolute_id, 0.0)
                            relation_score_map[r.absolute_id] = max(e1_score, e2_score) * 0.3
                    for abs_id in (r.entity1_absolute_id, r.entity2_absolute_id):
                        if abs_id not in entity_abs_ids:
                            expand_missing.add(abs_id)
                if expand_missing:
                    batch_entities = storage.get_entities_by_absolute_ids(list(expand_missing))
                    for e in batch_entities:
                        if e:
                            entities_by_abs[e.absolute_id] = e
                            entity_abs_ids.add(e.absolute_id)
                            if e.absolute_id not in entity_score_map:
                                entity_score_map[e.absolute_id] = 0.0

        # --- Step 5: Time filtering ---
        final_entities: List[Entity] = []
        for e in entities_by_abs.values():
            if time_before_dt and e.event_time and e.event_time > time_before_dt:
                continue
            if time_after_dt and e.event_time and e.event_time < time_after_dt:
                continue
            final_entities.append(e)

        final_relations: List[Relation] = []
        seen_rel_ids: Set[str] = set()
        for r in matched_relations:
            if r.absolute_id in seen_rel_ids:
                continue
            if time_before_dt and r.event_time and r.event_time > time_before_dt:
                continue
            if time_after_dt and r.event_time and r.event_time < time_after_dt:
                continue
            seen_rel_ids.add(r.absolute_id)
            final_relations.append(r)

        # --- Step 6A: Optional reranking ---
        _reranker_searcher = _hybrid_searcher or _get_searcher(storage)
        if reranker == "node_degree":
            degree_map = storage.batch_get_entity_degrees(
                [e.family_id for e in final_entities]
            )
            scored = [(e, entity_score_map.get(e.absolute_id, 0.0)) for e in final_entities]
            reranked = _reranker_searcher.node_degree_rerank(scored, degree_map)
            final_entities = [e for e, _ in reranked[:max_entities]]

        # --- Step 6B: Confidence weighting (all search modes) ---
        if reranker != "node_degree":
            ent_scored = [(e, entity_score_map.get(e.absolute_id, 0.0)) for e in final_entities]
            reranked_ents = _reranker_searcher.confidence_rerank(ent_scored, alpha=0.2, time_decay_half_life_days=90.0)
            final_entities = [e for e, _ in reranked_ents[:max_entities]]
            for e, score in reranked_ents[:max_entities]:
                entity_score_map[e.absolute_id] = score

            rel_scored = [(r, relation_score_map.get(r.absolute_id, 0.0)) for r in final_relations]
            reranked_rels = _reranker_searcher.confidence_rerank(rel_scored, alpha=0.2, time_decay_half_life_days=90.0)
            final_relations = [r for r, _ in reranked_rels[:max_relations]]
            for r, score in reranked_rels[:max_relations]:
                relation_score_map[r.absolute_id] = score

        # --- Step 6C: Core name boosting ---
        if len(query) >= 2 and len(query) <= 20 and final_entities:
            _boosted = []
            _rest = []
            for e in final_entities:
                name = getattr(e, 'name', '')
                core = _PAREN_ANNOTATION_RE.sub('', name).strip()
                if (core == query or name == query
                    or name.startswith(query + '，') or name.startswith(query + '(')):
                    _boosted.append(e)
                else:
                    _rest.append(e)
            if _boosted:
                final_entities = _boosted + _rest

        # --- Step 6D: Family-ID deduplication ---
        _seen_fids: Set[str] = set()
        _deduped_entities: List[Entity] = []
        for e in final_entities:
            fid = getattr(e, 'family_id', None) or e.absolute_id
            if fid not in _seen_fids:
                _seen_fids.add(fid)
                _deduped_entities.append(e)
        final_entities = _deduped_entities

        _seen_rel_fids: Set[str] = set()
        _seen_content_keys: Set[str] = set()
        _deduped_relations: List[Relation] = []
        for r in final_relations:
            fid = getattr(r, 'family_id', None) or r.absolute_id
            if fid in _seen_rel_fids:
                continue
            _seen_rel_fids.add(fid)
            rc = (getattr(r, 'content', '') or '')[:60].strip()
            if rc and rc in _seen_content_keys:
                continue
            if rc:
                _seen_content_keys.add(rc)
            _deduped_relations.append(r)
        final_relations = _deduped_relations

        # --- Step 7: Format output ---
        output_format = (body.get("format", "full") or "full").strip().lower()

        if output_format == "compact":
            compact_entities = []
            for e in final_entities:
                ce = {
                    "fid": e.family_id,
                    "name": e.name,
                    "score": round(entity_score_map.get(e.absolute_id, 0.0), 3),
                }
                if getattr(e, "summary", None):
                    s = e.summary
                    ce["summary"] = s[:150] + ("..." if len(s) > 150 else "")
                elif e.content:
                    s = e.content
                    ce["summary"] = s[:150] + ("..." if len(s) > 150 else "")
                if getattr(e, "confidence", None) is not None:
                    ce["conf"] = round(e.confidence, 2)
                compact_entities.append(ce)

            compact_relations = []
            all_abs_ids = set()
            for r in final_relations:
                all_abs_ids.add(r.entity1_absolute_id)
                all_abs_ids.add(r.entity2_absolute_id)
            name_map = storage.get_entity_names_by_absolute_ids(list(all_abs_ids))
            for r in final_relations:
                cr = {
                    "fid": r.family_id,
                    "from": name_map.get(r.entity1_absolute_id, "?"),
                    "to": name_map.get(r.entity2_absolute_id, "?"),
                    "rel": r.content[:100] if r.content else "",
                    "score": round(relation_score_map.get(r.absolute_id, 0.0), 3),
                }
                if getattr(r, "relation_type", None):
                    cr["type"] = r.relation_type
                compact_relations.append(cr)

            return ok({
                "query": query,
                "entities": compact_entities,
                "relations": compact_relations,
                "counts": {"entities": len(compact_entities), "relations": len(compact_relations)},
            })

        # Full format (default): backward compatible
        result: Dict[str, Any] = {
            "query": query,
            "entities": [entity_to_dict(e, _score=entity_score_map.get(e.absolute_id), skip_sections=True) for e in final_entities],
            "relations": [relation_to_dict(r, _score=relation_score_map.get(r.absolute_id)) for r in final_relations],
            "entity_count": len(final_entities),
            "relation_count": len(final_relations),
        }
        # Parallelize 3 independent enrichment DB calls (reuse shared pool)
        _evc_fut = _shared_pool.submit(enrich_entity_version_counts, result["entities"], processor.storage)
        _rvc_fut = _shared_pool.submit(enrich_relation_version_counts, result["relations"], processor.storage)
        _er_fut = _shared_pool.submit(enrich_relations, result["relations"], processor)
        _evc_fut.result()
        _rvc_fut.result()
        _er_fut.result()
        return ok(result)
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/candidates", methods=["POST"])
def find_query_one():
    """Return candidate entities and relations matching request body conditions."""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
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
        processor = _get_processor()
        body = request.get_json(silent=True) if request.method == "POST" else None
        body = body if isinstance(body, dict) else {}

        def _get_value(name: str, default: Any = None) -> Any:
            if name in body and body[name] is not None:
                return body[name]
            return request.args.get(name, default)

        query_text = str(_get_value("query_text") or _get_value("query") or "").strip()
        if not query_text:
            return err("query_text 为必填参数", 400)
        threshold = float(_get_value("similarity_threshold") or _get_value("threshold", 0.5))
        max_results = int(_get_value("max_results") or _get_value("limit", 10))

        search_mode = str(_get_value("search_mode", "semantic") or "semantic").strip().lower()
        if search_mode not in _VALID_SEARCH_MODES:
            search_mode = "semantic"

        searcher = _get_searcher(processor.storage)
        if search_mode == "hybrid":
            ranked = searcher.search_relations(
                query_text=query_text,
                top_k=max_results,
                semantic_threshold=threshold,
            )
        else:
            if search_mode == "bm25":
                relations = processor.storage.search_relations_by_bm25(
                    query_text, limit=max_results
                )
            else:
                relations = processor.storage.search_relations_by_similarity(
                    query_text=query_text,
                    threshold=threshold,
                    max_results=max_results,
                )
            ranked = [(r, 1.0 - i * 0.01) for i, r in enumerate(relations)]
        ranked = searcher.confidence_rerank(ranked, alpha=0.2, time_decay_half_life_days=90.0)

        # Fallback: if no results and query is short, try entity name prefix -> their relations
        if not ranked and 1 <= len(query_text) <= 50 and hasattr(processor.storage, 'find_entity_by_name_prefix'):
            try:
                prefix_matches = processor.storage.find_entity_by_name_prefix(query_text, limit=3)
                if prefix_matches:
                    fids = [e.family_id for e in prefix_matches[:3]]
                    all_rels = []
                    seen_ids = set()
                    for fid in fids:
                        rels = processor.storage.get_entity_relations_by_family_id(fid)
                        for r in rels:
                            if r.absolute_id not in seen_ids:
                                seen_ids.add(r.absolute_id)
                                all_rels.append(r)
                    ranked = [(r, 0.5) for r in all_rels[:max_results]]
            except Exception:
                pass
        dicts = [relation_to_dict(r, _score=score) for r, score in ranked]
        enrich_relation_version_counts(dicts, processor.storage)
        enrich_relations(dicts, processor)
        return ok(dicts)
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/relations/between", methods=["GET", "POST"])
def find_relations_between():
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) if request.method == "POST" else None
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
        body = request.get_json(silent=True) if request.method == "POST" else None
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
    """Use Cypher shortestPath to find paths (Neo4j only)."""
    try:
        processor = _get_processor()
        if not hasattr(processor.storage, 'find_shortest_path_cypher'):
            return err("此功能需要 Neo4j 后端", 400)
        body = request.get_json(silent=True) or {}
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


# -- Quick search (Agent workflows) -----------------------------------------

# =========================================================
# Convenience endpoints for Agent workflows
# =========================================================
@relations_bp.route("/api/v1/find/quick-search", methods=["POST"])
def quick_search():
    """One-shot search: hybrid BM25+embedding RRF fusion with name boosting."""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        query = body.get("query", "").strip()
        if not query:
            return err("query is required", 400)
        max_entities = min(int(body.get("max_entities", 10)), 50)
        max_relations = min(int(body.get("max_relations", 20)), 100)
        threshold = max(0.0, min(1.0, float(body.get("similarity_threshold", 0.4))))

        # Phase 1: Exact name match (instant, highest confidence)
        exact_entities = []
        seen_fids = set()
        exact_map = processor.storage.get_family_ids_by_names([query])
        if exact_map:
            fid = list(exact_map.values())[0]
            ent = processor.storage.get_entity_by_family_id(fid)
            if ent:
                exact_entities.append(ent)
                seen_fids.add(ent.family_id)

        # Phase 2: BM25 + embedding RRF fusion via HybridSearcher
        searcher = _get_searcher(processor.storage)

        fused_entities = searcher.search_entities(
            query_text=query,
            top_k=max_entities,
            semantic_threshold=threshold,
        )
        fused_entities = searcher.confidence_rerank(fused_entities, alpha=0.2, time_decay_half_life_days=90.0)
        entity_score_map: Dict[str, float] = {}
        rrf_entities = []
        for ent, score in fused_entities:
            if ent.family_id not in seen_fids:
                rrf_entities.append(ent)
                entity_score_map[ent.absolute_id] = score
                seen_fids.add(ent.family_id)

        entities = exact_entities + rrf_entities
        entities = entities[:max_entities]

        # Phase 3: Relation search via HybridSearcher RRF fusion
        fused_relations = searcher.search_relations(
            query_text=query,
            top_k=max_relations,
            semantic_threshold=max(0.2, threshold - 0.1),
        )
        fused_relations = searcher.confidence_rerank(fused_relations, alpha=0.2, time_decay_half_life_days=90.0)
        relation_score_map: Dict[str, float] = {r.absolute_id: score for r, score in fused_relations}
        relations = [r for r, _ in fused_relations]

        entity_dicts = [entity_to_dict(e, _score=entity_score_map.get(e.absolute_id), skip_sections=True) for e in entities]
        rel_dicts = [relation_to_dict(r, _score=relation_score_map.get(r.absolute_id)) for r in relations]
        enrich_entity_version_counts(entity_dicts, processor.storage)
        enrich_relation_version_counts(rel_dicts, processor.storage)
        enrich_relations(rel_dicts, processor)

        return ok({
            "query": query,
            "entities": entity_dicts,
            "entity_count": len(entity_dicts),
            "relations": rel_dicts,
            "relation_count": len(rel_dicts),
        })
    except Exception as e:
        return err(str(e), 500)
