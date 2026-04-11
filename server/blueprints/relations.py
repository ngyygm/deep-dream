"""
Relation-related routes extracted from server/api.py.

All relation CRUD, search, path finding, and generic find/search endpoints.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from flask import Blueprint, request

from processor.models import Entity, Relation
from processor.search.hybrid import HybridSearcher
from processor.search.graph_traversal import GraphTraversalSearcher
from processor.perf import _perf_timer

from server.blueprints.helpers import (
    ok, err, _get_processor, _get_graph_id,
    entity_to_dict, relation_to_dict, enrich_relations,
    parse_time_point, _normalize_time_for_compare, _extract_candidate_ids,
)

logger = logging.getLogger(__name__)

relations_bp = Blueprint('relations', __name__)


# =========================================================
# Find: 统一语义检索入口（推荐）
# =========================================================
@relations_bp.route("/api/v1/find", methods=["POST"])
def find_unified():
    """统一语义检索：用自然语言从总记忆图中唤醒相关的局部区域。

    请求体:
        query (str, 必填): 自然语言查询
        similarity_threshold (float): 语义相似度阈值，默认 0.5
        max_entities (int): 最大返回实体数，默认 20
        max_relations (int): 最大返回关系数，默认 50
        expand (bool): 是否从命中实体向外扩展邻域，默认 true
        time_before (str, ISO): 只返回此时间之前的记忆
        time_after (str, ISO): 只返回此时间之后的记忆

    返回:
        entities: 命中的概念实体列表
        relations: 命中的概念关系列表
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
        reranker = str(body.get("reranker", "rrf") or "rrf").strip().lower()

        search_mode = str(body.get("search_mode", "semantic") or "semantic").strip().lower()
        if search_mode not in ("semantic", "bm25", "hybrid"):
            search_mode = "semantic"

        try:
            time_before_dt = parse_time_point(time_before) if time_before else None
            time_after_dt = parse_time_point(time_after) if time_after else None
        except ValueError as ve:
            return err(str(ve), 400)

        processor = _get_processor()
        storage = processor.storage

        # --- 第一步：按 search_mode 召回实体 ---
        with _perf_timer("find_unified | step1_entity_recall"):
            if search_mode == "bm25":
                matched_entities = storage.search_entities_by_bm25(
                    query, limit=max_entities
                )
            elif search_mode == "hybrid":
                searcher = HybridSearcher(storage)
                hybrid_entities = searcher.search_entities(
                    query_text=query,
                    top_k=max_entities,
                    semantic_threshold=similarity_threshold,
                )
                # HybridSearcher returns List[Tuple[Entity, float]], unpack
                matched_entities = [e for e, _ in hybrid_entities]
            else:
                matched_entities = storage.search_entities_by_similarity(
                    query_name=query,
                    query_content=query,
                    threshold=similarity_threshold,
                    max_results=max_entities,
                    text_mode="name_and_content",
                    similarity_method="embedding",
                )

        # --- 第二步：按 search_mode 召回关系 ---
        with _perf_timer("find_unified | step2_relation_recall"):
            if search_mode == "bm25":
                matched_relations = storage.search_relations_by_bm25(
                    query, limit=max_relations
                )
            elif search_mode == "hybrid":
                searcher = HybridSearcher(storage)
                hybrid_relations = searcher.search_relations(
                    query_text=query,
                    top_k=max_relations,
                    semantic_threshold=similarity_threshold,
                )
                # HybridSearcher returns List[Tuple[Relation, float]], unpack
                matched_relations = [r for r, _ in hybrid_relations]
            else:
                matched_relations = storage.search_relations_by_similarity(
                    query_text=query,
                    threshold=similarity_threshold,
                    max_results=max_relations,
                )

        entity_abs_ids: Set[str] = {e.absolute_id for e in matched_entities}
        relation_abs_ids: Set[str] = {r.absolute_id for r in matched_relations}
        entities_by_abs: Dict[str, Entity] = {e.absolute_id: e for e in matched_entities}

        # --- 第三步：从语义命中的关系中补充关联实体（批量获取） ---
        with _perf_timer("find_unified | step3_entity_completion"):
            missing_abs_ids = set()
            for r in list(matched_relations):
                for abs_id in (r.entity1_absolute_id, r.entity2_absolute_id):
                    if abs_id not in entity_abs_ids:
                        missing_abs_ids.add(abs_id)
            if missing_abs_ids:
                batch_entities = storage.get_entities_by_absolute_ids(list(missing_abs_ids))
                for e in batch_entities:
                    if e:
                        entities_by_abs[e.absolute_id] = e
                        entity_abs_ids.add(e.absolute_id)

        # --- 第四步：图谱邻域扩展 ---
        with _perf_timer("find_unified | step4_graph_expansion"):
            if expand and entity_abs_ids:
                expanded_rels = storage.get_relations_by_entity_absolute_ids(
                    list(entity_abs_ids), limit=max_relations
                )
                # 批量获取扩展关系中的新实体
                expand_missing = set()
                for r in expanded_rels:
                    if r.absolute_id not in relation_abs_ids:
                        relation_abs_ids.add(r.absolute_id)
                        matched_relations.append(r)
                    for abs_id in (r.entity1_absolute_id, r.entity2_absolute_id):
                        if abs_id not in entity_abs_ids:
                            expand_missing.add(abs_id)
                if expand_missing:
                    batch_entities = storage.get_entities_by_absolute_ids(list(expand_missing))
                    for e in batch_entities:
                        if e:
                            entities_by_abs[e.absolute_id] = e
                            entity_abs_ids.add(e.absolute_id)

        # --- 第五步：时间过滤 ---
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

        # --- 第六步：可选重排序 ---
        if reranker == "node_degree":
            degree_map = storage.batch_get_entity_degrees(
                [e.family_id for e in final_entities]
            )
            searcher_hybrid = HybridSearcher(storage)
            scored = [(e, 1.0) for e in final_entities]
            reranked = searcher_hybrid.node_degree_rerank(scored, degree_map)
            final_entities = [e for e, _ in reranked[:max_entities]]

        result: Dict[str, Any] = {
            "query": query,
            "entities": [entity_to_dict(e) for e in final_entities],
            "relations": [relation_to_dict(r) for r in final_relations],
            "entity_count": len(final_entities),
            "relation_count": len(final_relations),
        }
        enrich_relations(result["relations"], processor)
        return ok(result)
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/candidates", methods=["POST"])
def find_query_one():
    """按请求体条件一次性返回候选实体与关系；推荐路径 /api/v1/find/candidates。"""
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
            entities_data = [entity_to_dict(e) for e in batch if e]
        if include_relations:
            batch_rels = storage.get_relations_by_entity_absolute_ids(list(relation_family_ids))
            for r in batch_rels:
                if r.absolute_id in relation_family_ids:
                    relations_data.append(relation_to_dict(r))
        return ok({"entities": entities_data, "relations": relations_data})
    except Exception as e:
        return err(str(e), 500)



# =========================================================
# Find: 关系原子接口
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

        query_text = str(_get_value("query_text", "") or "").strip()
        if not query_text:
            return err("query_text 为必填参数", 400)
        threshold = float(_get_value("similarity_threshold") or _get_value("threshold", 0.5))
        max_results = int(_get_value("max_results", 10))

        search_mode = str(_get_value("search_mode", "semantic") or "semantic").strip().lower()
        if search_mode not in ("semantic", "bm25", "hybrid"):
            search_mode = "semantic"

        if search_mode == "bm25":
            relations = processor.storage.search_relations_by_bm25(
                query_text, limit=max_results
            )
        elif search_mode == "hybrid":
            searcher = HybridSearcher(processor.storage)
            hybrid_rels = searcher.search_relations(
                query_text=query_text,
                top_k=max_results,
                semantic_threshold=threshold,
            )
            relations = [r for r, _ in hybrid_rels]
        else:
            relations = processor.storage.search_relations_by_similarity(
                query_text=query_text,
                threshold=threshold,
                max_results=max_results,
            )
        dicts = [relation_to_dict(r) for r in relations]
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
        family_id_a = str(body.get("family_id_a") or body.get("from_family_id") or request.args.get("family_id_a") or request.args.get("from_family_id") or "").strip()
        family_id_b = str(body.get("family_id_b") or body.get("to_family_id") or request.args.get("family_id_b") or request.args.get("to_family_id") or "").strip()
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
    """查找两个实体之间的最短路径"""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) if request.method == "POST" else None
        body = body if isinstance(body, dict) else {}
        family_id_a = str(body.get("family_id_a") or body.get("from_family_id")
                         or request.args.get("family_id_a")
                         or request.args.get("from_family_id") or "").strip()
        family_id_b = str(body.get("family_id_b") or body.get("to_family_id")
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
        for p in result.get("paths", []):
            serialized_paths.append({
                "entities": [entity_to_dict(e) for e in p.get("entities", [])],
                "relations": [relation_to_dict(r) for r in p.get("relations", [])],
                "length": p.get("length", 0),
            })

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
    """使用 Cypher shortestPath 查找路径（Neo4j 专属）。"""
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


@relations_bp.route("/api/v1/find/relations/absolute/<absolute_id>", methods=["GET"])
def find_relation_by_absolute_id(absolute_id: str):
    try:
        processor = _get_processor()
        relation = processor.storage.get_relation_by_absolute_id(absolute_id)
        if relation is None:
            return err(f"未找到关系版本: {absolute_id}", 404)
        d = relation_to_dict(relation)
        enrich_relations([d], processor)
        return ok(d)
    except Exception as e:
        return err(str(e), 500)


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


@relations_bp.route("/api/v1/find/relations/<family_id>/versions", methods=["GET"])
def find_relation_versions(family_id: str):
    try:
        processor = _get_processor()
        versions = processor.storage.get_relation_versions(family_id)
        dicts = [relation_to_dict(r) for r in versions]
        enrich_relations(dicts, processor)
        return ok(dicts)
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/relations/<family_id>", methods=["PUT"])
def update_relation_by_family(family_id: str):
    """编辑关系：创建新版本。"""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        content = body.get("content")
        if not content:
            return err("content 为必填字段", 400)

        current_versions = processor.storage.get_relation_versions(family_id)
        if not current_versions:
            return err(f"未找到关系: {family_id}", 404)
        current = current_versions[0]  # 最新版本

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        from processor.models import Relation as RelationModel
        updated = RelationModel(
            absolute_id=str(uuid.uuid4()),
            family_id=family_id,
            entity1_absolute_id=current.entity1_absolute_id,
            entity2_absolute_id=current.entity2_absolute_id,
            content=content,
            event_time=now,
            processed_time=now,
            episode_id=current.episode_id,
            source_document=current.source_document,
            valid_at=now,
        )
        processor.storage.save_relation(updated)
        return ok({"message": "关系已更新", "absolute_id": updated.absolute_id, "family_id": family_id})
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/relations/<family_id>", methods=["DELETE"])
def delete_relation_family(family_id: str):
    """删除关系所有版本。"""
    try:
        processor = _get_processor()
        count = processor.storage.delete_relation_all_versions(family_id)
        if count == 0:
            return err(f"未找到关系: {family_id}", 404)
        return ok({"message": f"已删除 {count} 个关系版本", "family_id": family_id})
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/relations/batch-delete", methods=["POST"])
def batch_delete_relations():
    """批量删除关系。"""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        family_ids = body.get("family_ids") or body.get("relation_ids", [])
        if not isinstance(family_ids, list) or not family_ids:
            return err("family_ids 需为非空数组", 400)
        if len(family_ids) > 100:
            return err("单次批量删除上限 100 个", 400)
        total = processor.storage.batch_delete_relations(family_ids)
        return ok({"message": f"已删除 {total} 个关系版本", "count": len(family_ids)})
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/entities/<family_id>/relations", methods=["GET"])
def find_relations_by_entity(family_id: str):
    try:
        processor = _get_processor()
        limit = request.args.get("limit", type=int)
        time_point_str = request.args.get("time_point")
        try:
            time_point = parse_time_point(time_point_str)
        except ValueError as ve:
            return err(str(ve), 400)
        max_version_absolute_id = (request.args.get("max_version_absolute_id") or "").strip() or None
        relation_scope = (request.args.get("relation_scope") or "accumulated").strip()

        if relation_scope not in ("accumulated", "version_only", "all_versions"):
            relation_scope = "accumulated"

        # When no max_version_absolute_id, all modes degenerate to returning all relations
        if not max_version_absolute_id:
            relations = processor.storage.get_entity_relations_by_family_id(
                family_id=family_id,
                limit=limit,
                time_point=time_point,
                max_version_absolute_id=None,
            )
            dicts = [relation_to_dict(r) for r in relations]
            enrich_relations(dicts, processor)
            return ok(dicts)

        # ---- Shared queries ----
        # current_rels: relations directly linked to the focused version only
        current_rels = processor.storage.get_entity_relations(
            max_version_absolute_id,
            limit=limit,
            time_point=time_point,
        )
        # accum_rels: accumulated relations from v1 through focused version
        accum_rels = processor.storage.get_entity_relations_by_family_id(
            family_id=family_id,
            limit=limit,
            time_point=time_point,
            max_version_absolute_id=max_version_absolute_id,
        )

        # Dedup by family_id
        accum_by_rid = {r.family_id: r for r in accum_rels}
        current_by_rid = {r.family_id: r for r in current_rels}
        accum_rids = set(accum_by_rid)
        current_rids = set(current_by_rid)

        # ---- version_only: only relations directly linked to this version ----
        if relation_scope == "version_only":
            dicts = [relation_to_dict(r) for r in current_rels]
            enrich_relations(dicts, processor)
            return ok(dicts)

        # ---- accumulated: v1..vN union + future from latest ----
        if relation_scope == "accumulated":
            latest_rels = processor.storage.get_entity_relations_by_family_id(
                family_id=family_id,
                limit=limit,
                time_point=time_point,
                max_version_absolute_id=None,
            )
            latest_by_rid = {r.family_id: r for r in latest_rels}
            latest_rids = set(latest_by_rid)

            all_rels = []
            for rid in accum_rids | latest_rids:
                if rid in current_rids:
                    all_rels.append(current_by_rid[rid])
                elif rid in accum_rids:
                    all_rels.append(accum_by_rid[rid])
                else:
                    all_rels.append(latest_by_rid[rid])

            dicts = [relation_to_dict(r) for r in all_rels]
            enrich_relations(dicts, processor)

            for d in dicts:
                rid = d["family_id"]
                if rid not in current_rids:
                    if rid not in accum_rids:
                        d["_future"] = True
                    else:
                        d["_inherited"] = True

            return ok(dicts)

        # ---- all_versions: (v1..vN) ∪ latest, classify as current/inherited/future ----
        latest_rels = processor.storage.get_entity_relations_by_family_id(
            family_id=family_id,
            limit=limit,
            time_point=time_point,
            max_version_absolute_id=None,
        )
        latest_by_rid = {r.family_id: r for r in latest_rels}
        latest_rids = set(latest_by_rid)

        all_rels = []
        for rid in accum_rids | latest_rids:
            if rid in latest_rids:
                all_rels.append(latest_by_rid[rid])
            else:
                all_rels.append(accum_by_rid[rid])

        dicts = [relation_to_dict(r) for r in all_rels]
        enrich_relations(dicts, processor)

        for d in dicts:
            rid = d["family_id"]
            if rid in current_rids:
                d["_version_scope"] = "current"
            elif rid in accum_rids:
                d["_version_scope"] = "inherited"
            else:
                d["_version_scope"] = "future"

        return ok(dicts)
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/relations/create", methods=["POST"])
def create_relation():
    """手动创建关系（生成新 family_id + absolute_id）。

    实体 ID 参数支持两种形式（二选一，优先使用 absolute_id）：
      - entity1_absolute_id / entity2_absolute_id（版本快照 ID）
      - entity1_family_id / entity2_family_id（逻辑 ID，自动解析为最新 absolute_id）
    """
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}

        # Resolve entity IDs: prefer absolute_id, fall back to family_id
        e1 = (body.get("entity1_absolute_id") or "").strip()
        e2 = (body.get("entity2_absolute_id") or "").strip()

        if not e1:
            e1_fid = (body.get("entity1_family_id") or "").strip()
            if e1_fid:
                entity1 = processor.storage.get_entity_by_family_id(e1_fid)
                if entity1:
                    e1 = entity1.absolute_id
                else:
                    return err(f"entity1_family_id '{e1_fid}' 未找到对应实体", 404)

        if not e2:
            e2_fid = (body.get("entity2_family_id") or "").strip()
            if e2_fid:
                entity2 = processor.storage.get_entity_by_family_id(e2_fid)
                if entity2:
                    e2 = entity2.absolute_id
                else:
                    return err(f"entity2_family_id '{e2_fid}' 未找到对应实体", 404)

        if not e1 or not e2:
            return err("需要 entity1_absolute_id 或 entity1_family_id（entity2 同理）", 400)

        content = (body.get("content") or "").strip()
        if not content:
            return err("content 为必填", 400)

        import uuid as _uuid

        now = datetime.now()
        ts = now.strftime("%Y%m%d_%H%M%S")
        # 确保 entity1 < entity2（无向关系）
        if e1 > e2:
            e1, e2 = e2, e1
        family_id = f"rel_{_uuid.uuid4().hex[:12]}"
        absolute_id = f"relation_{ts}_{_uuid.uuid4().hex[:8]}"
        for _ in range(10):
            existing = processor.storage.get_relation_by_absolute_id(absolute_id)
            if not existing:
                break
            absolute_id = f"relation_{ts}_{_uuid.uuid4().hex[:8]}"
        for _ in range(10):
            existing = processor.storage.get_relation_by_family_id(family_id)
            if not existing:
                break
            family_id = f"rel_{_uuid.uuid4().hex[:12]}"

        relation = Relation(
            absolute_id=absolute_id,
            family_id=family_id,
            entity1_absolute_id=e1,
            entity2_absolute_id=e2,
            content=content,
            event_time=now,
            processed_time=now,
            episode_id=body.get("episode_id", ""),
            source_document=body.get("source_document", ""),
        )
        processor.storage.save_relation(relation)
        return ok(relation_to_dict(relation))
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/relations/absolute/<absolute_id>", methods=["PUT"])
def update_relation_absolute(absolute_id: str):
    """更新指定版本关系。"""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        fields = {}
        for key in ("content", "summary", "attributes", "confidence"):
            if key in body:
                fields[key] = body[key]
        if not fields:
            return err("至少提供一个可更新字段", 400)
        updated = processor.storage.update_relation_by_absolute_id(absolute_id, **fields)
        if not updated:
            return err(f"未找到关系版本: {absolute_id}", 404)
        return ok(relation_to_dict(updated))
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/relations/absolute/<absolute_id>", methods=["DELETE"])
def delete_relation_absolute(absolute_id: str):
    """删除指定版本关系。"""
    try:
        processor = _get_processor()
        success = processor.storage.delete_relation_by_absolute_id(absolute_id)
        if not success:
            return err(f"未找到关系版本: {absolute_id}", 404)
        return ok({"absolute_id": absolute_id, "deleted": True})
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/relations/batch-delete-versions", methods=["POST"])
def batch_delete_relation_versions():
    """批量删除关系版本。"""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        absolute_ids = body.get("absolute_ids", [])
        if not isinstance(absolute_ids, list) or not absolute_ids:
            return err("absolute_ids 需为非空数组", 400)
        deleted_count = processor.storage.batch_delete_relation_versions_by_absolute_ids(absolute_ids)
        return ok({
            "deleted": absolute_ids[:deleted_count],
            "failed": absolute_ids[deleted_count:] if deleted_count < len(absolute_ids) else [],
            "summary": {"deleted_count": deleted_count, "failed_count": len(absolute_ids) - deleted_count},
        })
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/relations/redirect", methods=["POST"])
def redirect_relation():
    """重定向关系的实体端点。"""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        family_id = (body.get("family_id") or "").strip()
        side = (body.get("side") or "").strip()
        new_family_id = (body.get("new_family_id") or "").strip()
        if not family_id or not side or not new_family_id:
            return err("family_id, side, new_family_id 为必填", 400)
        if side not in ("entity1", "entity2"):
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


@relations_bp.route("/api/v1/find/relations/<family_id>/confidence", methods=["PUT"])
def update_relation_confidence(family_id: str):
    """手动设置关系置信度（覆盖自动演化值）。"""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
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
        updated = processor.storage.get_relation_by_family_id(family_id)
        from server.blueprints.helpers import relation_to_dict
        return ok(relation_to_dict(updated))
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/relations/<family_id>/invalidate", methods=["POST"])
def invalidate_relation(family_id: str):
    """标记关系为失效（不删除，保留历史）"""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        reason = body.get("reason", "")
        count = processor.storage.invalidate_relation(family_id, reason)
        if count == 0:
            return err(f"未找到可失效的关系: {family_id}", 404)
        return ok({"message": f"已标记 {count} 个关系版本为失效", "family_id": family_id})
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/relations/invalidated", methods=["GET"])
def find_invalidated_relations():
    """列出所有已失效的关系"""
    try:
        processor = _get_processor()
        limit = request.args.get("limit", type=int, default=100)
        relations = processor.storage.get_invalidated_relations(limit)
        dicts = [relation_to_dict(r) for r in relations]
        enrich_relations(dicts, processor)
        return ok(dicts)
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# 图谱结构统计
# =========================================================
@relations_bp.route("/api/v1/find/graph-stats", methods=["GET"])
def find_graph_stats():
    """图谱结构统计"""
    try:
        processor = _get_processor()
        stats = processor.storage.get_graph_statistics()
        return ok(stats)
    except Exception as e:
        return err(str(e), 500)


@relations_bp.route("/api/v1/find/graph-summary", methods=["GET"])
def graph_summary():
    """聚合返回：图谱统计 + 健康状态。"""
    try:
        processor = _get_processor()
        stats = processor.storage.get_graph_statistics()
        embedding_available = (
            processor.embedding_client is not None
            and processor.embedding_client.is_available()
        )
        storage_backend = "neo4j" if hasattr(processor.storage, 'is_neo4j') else "sqlite"
        return ok({
            "graph_id": _get_graph_id(),
            "storage_backend": storage_backend,
            "embedding_available": embedding_available,
            "statistics": stats,
        })
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Phase B: Advanced Search — BFS 遍历 + MMR 重排序
# =========================================================
@relations_bp.route("/api/v1/find/traverse", methods=["POST"])
def traverse_graph():
    """BFS 图遍历搜索。"""
    try:
        body = request.get_json(silent=True) or {}
        seed_ids = body.get("seed_family_ids") or body.get("start_entity_ids", [])
        if not isinstance(seed_ids, list) or not seed_ids:
            return err("seed_family_ids 需为非空数组", 400)
        max_depth = int(body.get("max_depth", 2))
        max_nodes = int(body.get("max_nodes", 50))

        processor = _get_processor()
        searcher = GraphTraversalSearcher(processor.storage)
        entities = searcher.bfs_expand(seed_ids, max_depth=max_depth, max_nodes=max_nodes)
        return ok([entity_to_dict(e) for e in entities])
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Convenience endpoints for Agent workflows
# =========================================================
@relations_bp.route("/api/v1/find/quick-search", methods=["POST"])
def quick_search():
    """One-shot search: hybrid BM25+embedding RRF fusion with name boosting.

    Phase 1: exact name match (highest confidence)
    Phase 2: BM25 + embedding via HybridSearcher RRF fusion
    Phase 3: relation search via HybridSearcher RRF fusion
    """
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
        searcher = HybridSearcher(processor.storage)

        fused_entities = searcher.search_entities(
            query_text=query,
            top_k=max_entities,
            semantic_threshold=threshold,
        )
        # Dedup: skip entities already found by exact match
        rrf_entities = []
        for ent, score in fused_entities:
            if ent.family_id not in seen_fids:
                rrf_entities.append(ent)
                seen_fids.add(ent.family_id)

        entities = exact_entities + rrf_entities
        entities = entities[:max_entities]

        # Phase 3: Relation search via HybridSearcher RRF fusion
        fused_relations = searcher.search_relations(
            query_text=query,
            top_k=max_relations,
            semantic_threshold=max(0.2, threshold - 0.1),
        )
        relations = [r for r, _ in fused_relations]

        entity_dicts = [entity_to_dict(e) for e in entities]
        rel_dicts = [relation_to_dict(r) for r in relations]
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
