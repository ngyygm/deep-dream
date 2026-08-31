"""
Concept routes — Concept CRUD/search/traverse.

P4.3 拆分说明：
  - 检索执行逻辑（RRF 融合/role boost/CJK 阈值/BM25+语义两路）在
    core/find/concept_search.py，与 CLI ``concept search`` 共用同一实现（P4.2）。
  - /documents*、/episodes*、/vaults* 路由在 routes/documents.py。
本模块只保留概念/实体/关系路由。
"""
from __future__ import annotations

import logging
import math as _math
import re as _re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from flask import Blueprint, request

from core.server.routes.helpers import (
    ok,
    err,
    _get_processor,
    _get_graph_id,
    _get_searcher,
    get_json_body,
)
from core.server.routes._constants import _VALID_SEARCH_MODES, _VALID_RERANKERS
# P4.2：检索实现单点化——server 与 CLI 共用（别名保持本模块内的既有调用名）
from core.find.concept_search import (
    normalize_results as _normalize_results,
    bm25_concept_search as _bm25_concept_search,
    semantic_concept_search as _semantic_concept_search,
    hybrid_concept_search as _hybrid_concept_search,
)

logger = logging.getLogger(__name__)

concepts_bp = Blueprint("concepts", __name__)

_shared_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="concept")

# Pre-compiled regex for duplicate entity name normalization
_BOOK_MARKS_RE = _re.compile(r'[《》]')
_PAREN_ANNOTATION_RE = _re.compile(r'\s*[（(][^）)]+[）)]\s*')
_VALID_CONCEPT_ROLES = ("document", "episode", "entity", "relation")


def _strict_body_bool(value, field_name: str, default: bool = False) -> bool:
    """Parse a JSON boolean without Python's truthiness surprises.

    In particular, ``bool("false")`` is True, which previously made a
    seemingly harmless request enable expensive expand/group work.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
    raise ValueError(f"{field_name} 必须为布尔值")


def _body_text(body: dict, field_name: str, *, default: str = "") -> str:
    value = body.get(field_name, default)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须为字符串")
    return value.strip()


# =========================================================
# Concepts — 统一概念查询接口（Phase 4）
# =========================================================

@concepts_bp.route("/api/v1/concepts/search", methods=["POST"])
@concepts_bp.route("/api/v1/find", methods=["POST"])
def search_concepts():
    """统一概念搜索（可选 role 过滤，支持 semantic/bm25/hybrid 模式）。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'search_concepts_by_bm25'):
            return err("此功能暂不可用", 400)
        body = get_json_body()
        _body_compact = str(body.get("compact", "")).lower() in ("true", "1", "yes")
        if _body_compact:
            from flask import g as _g
            _g.compact = True
        query = _body_text(body, "query")
        if not query:
            return err("query 不能为空", 400)
        if len(query) > 4096:
            return err("query 过长（最多 4096 个字符）", 400)
        role = body.get("role") or None
        if role is not None:
            if str(role).strip().lower() not in _VALID_CONCEPT_ROLES:
                return err(f"role '{role}' 无效，可选: {', '.join(_VALID_CONCEPT_ROLES)}", 400)
            role = str(role).strip().lower()
        # Validate limit: must be a non-negative integer
        raw_limit = body.get("limit", 20)
        if isinstance(raw_limit, float) and not raw_limit.is_integer():
            return err("limit 必须为整数", 400)
        try:
            limit = int(raw_limit)
        except (ValueError, TypeError):
            return err("limit 必须为整数", 400)
        if limit < 0:
            return err("limit 不能为负数", 400)
        if limit == 0:
            # Return empty results immediately
            return ok({"concepts": [], "total": 0})
        limit = min(limit, 1000)
        try:
            threshold = float(body.get("threshold", 0.5))
        except (ValueError, TypeError):
            return err("threshold 必须为数字", 400)
        if not _math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            return err("threshold 必须是 0 到 1 之间的有限数字", 400)
        search_mode = str(body.get("search_mode", "bm25") or "bm25").strip().lower()
        if search_mode not in _VALID_SEARCH_MODES:
            return err(f"search_mode '{search_mode}' 无效，可选: {', '.join(_VALID_SEARCH_MODES)}", 400)
        time_point = _body_text(body, "time_point") or None
        # Web UI 双界过滤：time_after/time_before 作为闭区间双界分别下推存储层
        # （P2.8）。旧实现把两界折叠成单个 time_point，静默丢弃另一界；
        # time_point 在搜索路径的存储层不消费，仅保留透传以兼容旧调用方。
        time_after = _body_text(body, "time_after") or None
        time_before = _body_text(body, "time_before") or None
        source_document = _body_text(body, "source_document") or None
        # max_name_length: opt-in filter to exclude long dialogue-fragment entity names.
        # Default 0 = disabled. Recommended: 15 to filter novel dialogue fragments.
        try:
            max_name_length = max(int(body.get("max_name_length", 0)), 0)
        except (ValueError, TypeError):
            max_name_length = 0
        reranker = (body.get("reranker") or "").strip().lower() or "rrf"
        if reranker not in _VALID_RERANKERS:
            return err(f"reranker '{reranker}' 无效，可选: {', '.join(_VALID_RERANKERS)}", 400)
        try:
            expand = _strict_body_bool(body.get("expand"), "expand")
            group = _strict_body_bool(body.get("group"), "group")
        except ValueError as exc:
            return err(str(exc), 400)

        # P4.2：三种模式的执行体收敛在 core/find/concept_search.py
        # （与 CLI concept search 同一实现；语义腿统一走
        # storage.agent_semantic_search 单入口）。
        def _search(role_filter, result_limit):
            if search_mode == "bm25":
                return _bm25_concept_search(
                    storage, query, role_filter, result_limit, threshold,
                    time_point=time_point, source_document=source_document,
                    time_after=time_after, time_before=time_before)
            if search_mode == "semantic":
                return _semantic_concept_search(
                    storage, query, role_filter, result_limit, threshold,
                    reranker=reranker, time_point=time_point,
                    source_document=source_document,
                    time_after=time_after, time_before=time_before)
            return _hybrid_concept_search(
                storage, query, role_filter, result_limit, threshold,
                time_point=time_point, source_document=source_document,
                reranker=reranker, time_after=time_after, time_before=time_before)

        if request.path == "/api/v1/find":
            raw_me = body.get("max_entities", body.get("maxEntities", 20))
            raw_mr = body.get("max_relations", body.get("maxRelations", 50))
            for label, raw_val in [("max_entities", raw_me), ("max_relations", raw_mr)]:
                if isinstance(raw_val, float) and not raw_val.is_integer():
                    return err(f"{label} 必须为整数", 400)
                try:
                    int(raw_val)
                except (ValueError, TypeError):
                    return err(f"{label} 必须为整数", 400)
            max_entities = min(max(int(raw_me), 1), 1000)
            max_relations = min(max(int(raw_mr), 1), 1000)
            # Run entity and relation searches in parallel to avoid blocking
            ent_fut = _shared_pool.submit(_search, "entity", max_entities)
            rel_fut = _shared_pool.submit(_search, "relation", max_relations)
            entities, ent_meta = ent_fut.result()
            relations, rel_meta = rel_fut.result()
            entities = _normalize_results(entities)
            relations = _normalize_results(relations)
            # Apply max_name_length filter to both entity and relation results
            if max_name_length > 0:
                entities = [item for item in entities if len((item.get("name") or "")) <= max_name_length]
                relations = [item for item in relations if len((item.get("name") or "")) <= max_name_length]
            # Enrich with _degree for entities AND relations
            if (entities or relations) and hasattr(storage, 'batch_get_entity_degrees'):
                try:
                    all_fids = (
                        [item.get("family_id", "") or item.get("id", "") for item in entities]
                        + [item.get("family_id", "") or item.get("id", "") for item in relations]
                    )
                    degree_map = storage.batch_get_entity_degrees(all_fids)
                    for item in entities:
                        fid = item.get("family_id", "") or item.get("id", "")
                        item["_degree"] = degree_map.get(fid, 0)
                    for item in relations:
                        fid = item.get("family_id", "") or item.get("id", "")
                        item["_degree"] = degree_map.get(fid, 0)
                except Exception:
                    pass
            # Merge search_meta from both calls
            merged_meta = {
                "entity_search": ent_meta,
                "relation_search": rel_meta,
                "total_entities": len(entities),
                "total_relations": len(relations),
            }
            # Apply fields filtering to /find results
            fields_raw = (body.get("fields") or "").strip()
            if fields_raw:
                allowed = set(f.strip() for f in fields_raw.split(",") if f.strip())
                allowed.add("family_id")
                entities = [{"family_id": i.get("family_id", "")} | {k: v for k, v in i.items() if k in allowed} for i in entities]
                relations = [{"family_id": i.get("family_id", "")} | {k: v for k, v in i.items() if k in allowed} for i in relations]
            return ok({
                "entities": entities,
                "relations": relations,
                "concepts": entities + relations,
                "total": len(entities) + len(relations),
                "search_meta": merged_meta,
            })

        results, search_meta = _search(role, limit)
        results = _normalize_results(results)
        # Apply max_name_length filter (opt-in): exclude long dialogue-fragment names
        if max_name_length > 0:
            results = [item for item in results if len((item.get("name") or "")) <= max_name_length]
        # Enrich with _degree (graph connectivity) for each result concept
        if results and hasattr(storage, 'batch_get_entity_degrees'):
            try:
                fids = [item.get("family_id", "") or item.get("id", "") for item in results]
                degree_map = storage.batch_get_entity_degrees(fids)
                for item in results:
                    fid = item.get("family_id", "") or item.get("id", "")
                    item["_degree"] = degree_map.get(fid, 0)
            except Exception:
                pass
        # Apply expand: fetch neighbors for each result concept
        if expand and results and hasattr(storage, 'get_concept_neighbors'):
            for item in results:
                fid = item.get("family_id", "") or item.get("id", "")
                if fid:
                    try:
                        neighbors = storage.get_concept_neighbors(fid, max_depth=1)
                        item["expanded_neighbors"] = neighbors
                    except Exception:
                        item["expanded_neighbors"] = []
        resp_data = {"concepts": results, "total": len(results)}
        if search_meta is not None:
            resp_data["search_meta"] = search_meta
        if expand:
            resp_data["expanded"] = True
        # Field filtering (opt-in): return only requested fields + family_id
        fields_raw = (body.get("fields") or "").strip()
        if fields_raw and results:
            allowed = set(f.strip() for f in fields_raw.split(",") if f.strip())
            # Always keep family_id for identification
            allowed.add("family_id")
            filtered = []
            for item in results:
                filtered.append({"family_id": item.get("family_id", "")} | {k: v for k, v in item.items() if k in allowed})
            results = filtered
            resp_data["concepts"] = results
        # Clustering (opt-in): group results by semantic similarity
        if group and len(results) >= 3:
            try:
                searcher = _get_searcher(storage)
                if searcher is not None:
                    # Pre-load embeddings for clustering from vector cache
                    if hasattr(storage, '_vector_cache_for_role'):
                        try:
                            role_for_cache = role or "entity"
                            cache = storage._vector_cache_for_role(role_for_cache)
                            matrix = cache.get("matrix")
                            cache_rows = cache.get("rows") or []
                            if matrix is not None and cache_rows:
                                # Build family_id -> matrix row index mapping
                                fid_to_idx = {}
                                for i, r in enumerate(cache_rows):
                                    fid = r.get("family_id", "")
                                    if fid:
                                        fid_to_idx[fid] = i
                                # Inject embeddings into results
                                for item in results:
                                    fid = item.get("family_id", "") or item.get("id", "")
                                    if fid in fid_to_idx and "_embedding" not in item:
                                        item["_embedding"] = matrix[fid_to_idx[fid]].tolist()
                        except Exception:
                            pass
                    num_clusters = min(5, max(2, len(results) // 3))
                    # Use moderate threshold that allows merging while preventing
                    # overly aggressive grouping.  Lower than before (R8) to avoid
                    # excessive singletons when search results are thematically
                    # diverse (common for CJK queries).
                    sim_threshold = 0.45 + min(0.1, _math.log2(max(len(results), 2)) * 0.02)
                    clusters = searcher.cluster_results(results, num_clusters=num_clusters, sim_threshold=sim_threshold)
                    if clusters:
                        # Filter out singleton clusters to reduce noise.
                        # Report their count so callers know some items were
                        # not grouped.  Keep at least one cluster even if all are
                        # singletons (preserve the top-scoring result).
                        multi = [c for c in clusters if len(c.get("items", [])) > 1]
                        singletons = [c for c in clusters if len(c.get("items", [])) <= 1]
                        if multi:
                            resp_data["clusters"] = multi
                            resp_data["_singleton_count"] = len(singletons)
                        else:
                            # All singletons — return top 3 by cluster score
                            # (first item _score as proxy) so the response is not empty.
                            sorted_singletons = sorted(singletons, key=lambda c: c.get("items", [{}])[0].get("_score", 0), reverse=True)
                            resp_data["clusters"] = sorted_singletons[:3]
                            resp_data["_singleton_count"] = len(sorted_singletons) - min(3, len(sorted_singletons))
                            resp_data["_all_singletons"] = True
                        resp_data["grouped"] = True
                        # Strip injected embeddings from results to keep response compact
                        for item in results:
                            item.pop("_embedding", None)
                        for cluster in clusters:
                            for item in cluster.get("items", []):
                                item.pop("_embedding", None)
            except Exception as exc:
                logger.debug("Clustering failed: %s", exc)
        return ok(resp_data)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/suggest", methods=["GET", "POST"])
def suggest_concepts():
    """概念建议：根据自然语言查询，返回图谱中最接近的实体名称（自动补全/消歧）。"""
    if request.method == "POST":
        return err("suggest 只支持 GET 方法，请使用 GET /api/v1/concepts/suggest?query=X", 405)
    try:
        processor = _get_processor()
        storage = processor.storage
        query = (request.args.get("query") or "").strip()
        if not query:
            return err("query 不能为空", 400)
        if len(query) < 2:
            return err("query 至少需要 2 个字符", 400)
        if len(query) > 200:
            return err("query 过长", 400)
        role = (request.args.get("role") or "entity").strip().lower()
        if role not in _VALID_CONCEPT_ROLES:
            return err(f"role '{role}' 无效，可选: {', '.join(_VALID_CONCEPT_ROLES)}", 400)
        try:
            limit = min(max(int(request.args.get("limit", 10)), 1), 50)
        except (ValueError, TypeError):
            return err("limit 必须为整数", 400)
        source_document = (request.args.get("source_document") or "").strip() or None
        try:
            max_name_length = max(int(request.args.get("max_name_length", 0)), 0)
        except (ValueError, TypeError):
            max_name_length = 0
        if not hasattr(storage, 'suggest_concepts'):
            return err("此功能暂不可用", 400)
        suggestions = storage.suggest_concepts(query, role=role, limit=limit, source_document=source_document)
        # Apply max_name_length filter on response side (opt-in)
        if max_name_length > 0:
            suggestions = [s for s in suggestions if len((s.get("name") or "")) <= max_name_length]
        return ok({"query": query, "suggestions": suggestions, "total": len(suggestions)})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts", methods=["GET"])
def list_concepts():
    """列出概念（分页 + 可选 role 过滤）。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'list_concepts'):
            return err("此功能暂不可用", 400)
        role = request.args.get("role") or None
        if role is not None:
            if str(role).strip().lower() not in _VALID_CONCEPT_ROLES:
                return err(f"role '{role}' 无效，可选: {', '.join(_VALID_CONCEPT_ROLES)}", 400)
            role = str(role).strip().lower()
        try:
            limit = min(max(int(request.args.get('limit', 50)), 1), 1000)
        except (ValueError, TypeError):
            return err("limit 必须为整数", 400)
        try:
            offset = max(int(request.args.get('offset', 0)), 0)
        except (ValueError, TypeError):
            return err("offset 必须为整数", 400)
        time_point = (request.args.get("time_point") or "").strip() or None
        name = (request.args.get("name") or "").strip() or None
        concepts = storage.list_concepts(role=role, limit=limit, offset=offset, time_point=time_point, name=name)
        total = storage.count_concepts(role=role, time_point=time_point, name=name) if hasattr(storage, 'count_concepts') else len(concepts)
        return ok({"concepts": concepts, "total": total, "limit": limit, "offset": offset})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/<family_id>", methods=["GET", "PATCH"])
def get_concept(family_id: str):
    """获取概念（任意 role）。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if request.method == "PATCH":
            if not hasattr(storage, "update_concept_manual"):
                return err("此功能暂不可用", 400)
            body = get_json_body()
            allowed = {k: body[k] for k in ("name", "content", "confidence", "metadata") if k in body}
            if not allowed:
                return err("至少提供 name/content/confidence/metadata 之一", 400)
            updated = storage.update_concept_manual(family_id, allowed)
            if isinstance(updated, dict) and not updated.get("updated", True):
                return err(f"概念不存在: {family_id}", 404)
            return ok({"family_id": family_id, "version": updated, "message": "概念已保存为新版本"})
        if not hasattr(storage, 'get_concept_by_family_id'):
            return err("此功能暂不可用", 400)
        time_point = (request.args.get("time_point") or "").strip() or None
        concept = storage.get_concept_by_family_id(family_id, time_point=time_point)
        if concept is None:
            return err(f"概念不存在: {family_id} (graph={_get_graph_id()})", 404)
        return ok(concept)
    except KeyError as e:
        return err(str(e.args[0]) if e.args else str(e), 404)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/<family_id>/versions", methods=["GET"])
def get_concept_versions(family_id: str):
    """List all versions for a concept family."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "get_concept_versions"):
            return err("此功能暂不可用", 400)
        versions = storage.get_concept_versions(family_id)
        if not versions:
            return err("概念不存在", 404)
        return ok({"family_id": family_id, "versions": versions, "total": len(versions)})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/<family_id>/neighbors", methods=["GET"])
def get_concept_neighbors(family_id: str):
    """获取概念邻居（无论 role）。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'get_concept_neighbors'):
            return err("此功能暂不可用", 400)
        try:
            max_depth = min(max(int(request.args.get('max_depth', 1)), 1), 3)
        except (ValueError, TypeError):
            return err("max_depth 必须为整数", 400)
        try:
            max_results = min(max(int(request.args.get('max_results', 200)), 1), 1000)
        except (ValueError, TypeError):
            return err("max_results 必须为整数", 400)
        time_point = (request.args.get("time_point") or "").strip() or None
        neighbors = storage.get_concept_neighbors(family_id, max_depth=max_depth, time_point=time_point, max_results=max_results)
        # Field filtering (opt-in): return only requested fields + family_id
        fields_raw = (request.args.get("fields") or "").strip()
        if fields_raw:
            allowed = set(f.strip() for f in fields_raw.split(",") if f.strip())
            filtered = []
            for n in neighbors:
                fid = n.get("family_id", "")
                filtered.append({"family_id": fid} | {k: v for k, v in n.items() if k in allowed})
            neighbors = filtered
        return ok({"family_id": family_id, "neighbors": neighbors, "max_depth": max_depth})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/batch-neighbors", methods=["POST"])
def batch_concept_neighbors():
    """批量获取多个概念的邻居。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'get_concept_neighbors'):
            return err("此功能暂不可用", 400)
        body = get_json_body()
        family_ids = body.get("family_ids") or []
        if not family_ids:
            return err("family_ids 不能为空", 400)
        if not isinstance(family_ids, list) or len(family_ids) > 50:
            return err("family_ids 必须为列表，最多 50 个", 400)
        try:
            max_depth = min(max(int(body.get('max_depth', 1)), 1), 3)
        except (ValueError, TypeError):
            return err("max_depth 必须为整数", 400)
        try:
            max_results = min(max(int(body.get('max_results', 200)), 1), 1000)
        except (ValueError, TypeError):
            return err("max_results 必须为整数", 400)
        time_point = _body_text(body, "time_point") or None
        fields_raw = (body.get("fields") or "").strip()
        allowed = set(f.strip() for f in fields_raw.split(",") if f.strip()) if fields_raw else None

        results = {}
        for fid in family_ids:
            if not fid or not isinstance(fid, str):
                continue
            try:
                neighbors = storage.get_concept_neighbors(
                    fid, max_depth=max_depth, time_point=time_point, max_results=max_results)
                if allowed:
                    neighbors = [
                        {"family_id": n.get("family_id", "")} | {k: v for k, v in n.items() if k in allowed}
                        for n in neighbors
                    ]
                results[fid] = neighbors
            except Exception:
                results[fid] = []

        return ok({"results": results, "total": len(results), "max_depth": max_depth})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/<family_id>/provenance", methods=["GET"])
def get_concept_provenance(family_id: str):
    """溯源：返回所有提及此概念的 observation。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'get_concept_provenance'):
            return err("此功能暂不可用", 400)
        time_point = (request.args.get("time_point") or "").strip() or None
        if hasattr(storage, "get_concept_by_family_id") and storage.get_concept_by_family_id(family_id, time_point=time_point) is None:
            return err("概念不存在", 404)
        provenance = storage.get_concept_provenance(family_id, time_point=time_point)
        return ok({"family_id": family_id, "provenance": provenance})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/traverse", methods=["POST"])
@concepts_bp.route("/api/v1/traverse", methods=["POST"])
def traverse_concepts():
    """BFS 遍历概念图。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'traverse_concepts'):
            return err("此功能暂不可用", 400)
        body = get_json_body()
        start_ids = body.get("start_family_ids") or []
        if not start_ids:
            return err("start_family_ids 不能为空", 400)
        if not isinstance(start_ids, list) or len(start_ids) > 100:
            return err("start_family_ids 必须为列表，最多 100 个", 400)
        if any(not isinstance(fid, str) or not fid.strip() or len(fid) > 512 for fid in start_ids):
            return err("start_family_ids 包含无效 ID", 400)
        try:
            max_depth = min(max(int(body.get('max_depth', 2)), 1), 3)
        except (ValueError, TypeError):
            return err("max_depth 必须为整数", 400)
        # Accept both max_results (SKILL.md) and max_nodes (Web UI) parameters
        raw_max = body.get('max_results') or body.get('max_nodes') or 500
        try:
            max_results = min(max(int(raw_max), 1), 2000)
        except (ValueError, TypeError):
            return err("max_results 必须为整数", 400)
        time_point = _body_text(body, "time_point") or None
        edge_types = body.get("edge_types") or body.get("edge_type") or None
        if isinstance(edge_types, str):
            edge_types = [edge_types]
        elif edge_types is not None:
            if not isinstance(edge_types, list) or len(edge_types) > 32 or any(
                not isinstance(item, str) or len(item) > 64 for item in edge_types
            ):
                return err("edge_types 必须为最多 32 个字符串的列表", 400)
        # Scale per-level timeout with depth: ~15s per level, min 30s total
        _traverse_timeout = max(30.0, 15.0 * max_depth)
        result = storage.traverse_concepts(start_ids, max_depth=max_depth, time_point=time_point, edge_types=edge_types, max_results=max_results, _timeout_seconds=_traverse_timeout)
        return ok(result)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/<family_id>/mentions", methods=["GET"])
def get_concept_mentions(family_id: str):
    """获取提及此概念的所有 Episode。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'get_concept_mentions'):
            return err("此功能暂不可用", 400)
        time_point = (request.args.get("time_point") or "").strip() or None
        mentions = storage.get_concept_mentions(family_id, time_point=time_point)
        return ok({"family_id": family_id, "mentions": mentions})
    except Exception as e:
        return err(str(e), 500)


# ── Duplicate detection & merge ────────────────────────────────────────────

@concepts_bp.route("/api/v1/concepts/duplicates", methods=["GET"])
def find_duplicate_entities():
    """Detect potential duplicate entities by core-name matching.

    Groups entities whose names normalize to the same core (stripping
    parenthetical annotations and book marks) but have different family_ids.
    Returns groups with relation counts to help decide merge direction.
    """
    try:
        processor = _get_processor()
        try:
            limit = min(max(int(request.args.get("limit", 500)), 1), 2000)
        except (ValueError, TypeError):
            return err("limit 必须为整数", 400)

        # Use fast SQL-based method when available
        if hasattr(processor.storage, 'find_duplicate_entities_fast'):
            duplicates = processor.storage.find_duplicate_entities_fast(limit=limit)
            return ok({"duplicates": duplicates, "count": len(duplicates)})

        # Fallback: legacy Python-based method
        entities = processor.storage.get_all_entities(limit=limit, exclude_embedding=True)

        def _normalize(name: str) -> str:
            n = _BOOK_MARKS_RE.sub('', name)
            n = _PAREN_ANNOTATION_RE.sub('', n)
            return n.strip()

        groups: Dict[str, list] = {}
        for e in entities:
            core = _normalize(getattr(e, 'name', ''))
            if not core or len(core) < 2:
                continue
            groups.setdefault(core, []).append(e)

        duplicates = []
        for core, items in sorted(groups.items()):
            fids = {getattr(e, 'family_id', '') for e in items}
            if len(fids) < 2:
                continue
            all_fids = list(fids)
            if hasattr(processor.storage, 'count_entity_relations_by_family_ids'):
                rel_counts = processor.storage.count_entity_relations_by_family_ids(all_fids)
            else:
                rel_counts = {fid: len(processor.storage.get_entity_relations_by_family_id(fid)) for fid in all_fids}
            if hasattr(processor.storage, 'get_entity_version_counts'):
                ver_counts = processor.storage.get_entity_version_counts(all_fids)
            else:
                ver_counts = {fid: processor.storage.get_entity_version_count(fid) for fid in all_fids}
            group = {"core_name": core, "entities": []}
            for e in items:
                group["entities"].append({
                    "family_id": e.family_id,
                    "name": getattr(e, 'name', ''),
                    "relation_count": rel_counts.get(e.family_id, 0),
                    "version_count": ver_counts.get(e.family_id, 0),
                })
            group["entities"].sort(key=lambda x: x["relation_count"], reverse=True)
            duplicates.append(group)

        return ok({"duplicates": duplicates, "count": len(duplicates)})
    except Exception as e:
        return err(str(e), 500)
