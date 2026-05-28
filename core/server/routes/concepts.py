"""
Concept routes — Concept CRUD/search/traverse and document graph helpers.
"""
from __future__ import annotations

import logging
import math as _math
import re as _re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from flask import Blueprint, current_app, request

from core.server.routes.helpers import (
    ok,
    err,
    _get_processor,
    _get_graph_id,
    entity_to_dict,
    relation_to_dict,
    enrich_relations,
    episode_to_dict,
    parse_time_point,
    _get_searcher,
    get_json_body,
)
from core.server.routes._constants import _VALID_SEARCH_MODES, _VALID_RERANKERS

logger = logging.getLogger(__name__)

concepts_bp = Blueprint("concepts", __name__)

# ── CJK detection for BM25 fallback ──────────────────────────────────────────
_CJK_RE = _re.compile(r'[一-鿿㐀-䶿]')


def _has_cjk(query: str) -> bool:
    """Return True if the query contains any CJK character."""
    if not query:
        return False
    return bool(_CJK_RE.search(query))


def _is_cjk_dominant(query: str) -> bool:
    """Return True if CJK characters make up >50% of the query.

    When this returns True we skip BM25 entirely and fall back to
    semantic search, because FTS5 unicode61 tokenizer produces noise
    for CJK text.
    """
    if not query:
        return False
    cjk_chars = len(_CJK_RE.findall(query))
    return cjk_chars > len(query) * 0.5

_shared_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="concept")


def _entity_to_search_dict(e):
    """Convert an Entity object to a search result dict."""
    return {
        "family_id": e.family_id,
        "id": e.absolute_id,
        "name": e.name,
        "content": e.content,
        "role": "entity",
        "_score": getattr(e, "_score", 0.0),
    }


def _relation_to_search_dict(r):
    """Convert a Relation object to a search result dict."""
    return {
        "family_id": r.family_id,
        "id": r.absolute_id,
        "name": "",
        "content": r.content,
        "role": "relation",
        "entity1_name": "",
        "entity2_name": "",
        "_score": getattr(r, "_score", 0.0),
    }


def _normalize_results(results: list) -> list:
    """Add ``_rank`` (1-based) and ``relevance`` (0-100) to search results.

    Normalisation strategy within *one* result list:
      - Highest score -> relevance 100
      - Lowest non-zero score -> relevance 10
      - Scores of 0 -> relevance 0
      - Everything else is linearly interpolated between 10 and 100.
    """
    if not results:
        return results

    # Shallow-copy each item to avoid mutating shared objects
    results = [{**item} for item in results]
    scores = [item.get("_score") or 0.0 for item in results]
    max_score = max(scores) if scores else 0.0

    if max_score < 1e-8:
        # All scores are zero – give everything relevance 0
        for idx, item in enumerate(results):
            item["_rank"] = idx + 1
            item["relevance"] = 0
        return results

    # Find the lowest *non-zero* score
    non_zero_scores = [s for s in scores if s > 0]
    if not non_zero_scores:
        min_nonzero = 0.0
    else:
        min_nonzero = min(non_zero_scores)

    RELEVANCE_FLOOR = 10
    RELEVANCE_CEIL = 100

    for idx, item in enumerate(results):
        item["_rank"] = idx + 1
        score = scores[idx]
        if score <= 0.0:
            item["relevance"] = 0
        elif abs(max_score - min_nonzero) < 1e-8:
            # All non-zero scores are identical — differentiate by name length
            # (shorter names are more likely to be real concept names)
            name_len = len((item.get("name") or ""))
            # Normalize name_len: 2 chars = best (100), 20+ chars = worst (10)
            name_factor = max(0.0, 1.0 - max(0, name_len - 2) / 18.0)
            item["relevance"] = round(RELEVANCE_FLOOR + name_factor * (RELEVANCE_CEIL - RELEVANCE_FLOOR), 1)
        else:
            # Linear interpolation: map [min_nonzero, max_score] -> [10, 100]
            ratio = (score - min_nonzero) / (max_score - min_nonzero)
            item["relevance"] = round(RELEVANCE_FLOOR + ratio * (RELEVANCE_CEIL - RELEVANCE_FLOOR), 1)

    return results


# ── Standalone reranker functions (dict-based, no HybridSearcher dependency) ────

def _node_degree_rerank_standalone(items, degree_map, alpha=0.3):
    """Node degree reranker for dict results.

    Boosts items with more graph connections (higher degree).
    """
    if not items:
        return items
    max_degree = max(degree_map.values()) if degree_map else 1
    if max_degree == 0:
        max_degree = 1
    inv_alpha = 1 - alpha
    results = []
    for item in items:
        fid = item.get("family_id", "") or item.get("id", "")
        score = item.get("_score", 0.0) or 0.0
        degree = degree_map.get(fid, 0)
        adjusted = score * inv_alpha + (degree / max_degree) * alpha
        item = dict(item)
        item["_score"] = round(adjusted, 6)
        results.append(item)
    results.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    return results


def _char_bigrams(text: str) -> set:
    """Extract character bigrams from text for CJK similarity comparison.

    For CJK text (which lacks whitespace word boundaries), character
    bigrams provide a meaningful overlap metric. For Latin text, falls
    back to whitespace-split tokens (existing behaviour).
    """
    if not text:
        return set()
    if _has_cjk(text):
        # Use character bigrams for CJK text — captures meaningful
        # sub-string overlap without requiring word segmentation.
        bigrams = set()
        chars = [c for c in text if not c.isspace()]
        for i in range(len(chars) - 1):
            bigrams.add(chars[i] + chars[i + 1])
        return bigrams if bigrams else {text}
    return set(text.split()) if text.strip() else set()


def _mmr_rerank_standalone(items, query_text="", lambda_=0.5, top_k=20):
    """MMR diversity reranker for dict results.

    MMR = (1 - lambda) * relevance - lambda * max_sim_to_selected
    Uses Jaccard word/bigram overlap as similarity (no embedding dependency).
    For CJK text, character bigrams replace whitespace tokenization.
    """
    if not items or len(items) <= 1:
        return items[:]
    top_k = min(top_k, len(items))

    def _get_tokens(item):
        name = (item.get("name") or "").strip()
        content = (item.get("content") or "")[:200]
        text = (name + " " + content).strip()
        return _char_bigrams(text)

    def _jaccard(sa, sb):
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    item_tokens = [_get_tokens(item) for item in items]

    selected = []
    remaining = list(range(len(items)))

    # Sort by score descending, pick first
    remaining.sort(key=lambda i: items[i].get("_score", 0.0), reverse=True)
    first = remaining.pop(0)
    selected.append(first)

    while remaining and len(selected) < top_k:
        best_mmr = -float("inf")
        best_idx_pos = 0
        for pos, idx in enumerate(remaining):
            relevance = items[idx].get("_score", 0.0) or 0.0
            max_sim = 0.0
            for s_idx in selected:
                sim = _jaccard(item_tokens[idx], item_tokens[s_idx])
                if sim > max_sim:
                    max_sim = sim
            mmr = (1 - lambda_) * relevance - lambda_ * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx_pos = pos
        selected.append(remaining.pop(best_idx_pos))

    return [items[i] for i in selected]


# Pre-compiled regex for duplicate entity name normalization
_BOOK_MARKS_RE = _re.compile(r'[《》]')
_PAREN_ANNOTATION_RE = _re.compile(r'\s*[（(][^）)]+[）)]\s*')
_VALID_CONCEPT_ROLES = ("document", "episode", "entity", "relation")


# =========================================================
# Concepts — 统一概念查询接口（Phase 4）
# =========================================================

@concepts_bp.route("/api/v1/agent/sql", methods=["POST"])
def agent_read_sql():
    """Agent-facing graph-local read-only SQL workbench."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "read_sql"):
            return err("当前存储后端不支持 Agent SQL 查询", 400)
        body = get_json_body()
        sql = (body.get("sql") or "").strip()
        params = body.get("params")
        limit = body.get("limit", 200)
        timeout_seconds = body.get("timeout_seconds", 5.0)
        explain = bool(body.get("explain") or body.get("include_query_plan"))
        result = storage.read_sql(
            sql,
            params=params,
            limit=limit,
            timeout_seconds=timeout_seconds,
            include_query_plan=explain,
        )
        result["graph_id"] = _get_graph_id()
        return ok(result)
    except (ValueError, TypeError, sqlite3.Error, TimeoutError) as exc:
        return err(str(exc), 400)
    except Exception as exc:
        logger.exception("Agent SQL query failed: %s", exc)
        return err("Agent SQL 查询失败", 500)


@concepts_bp.route("/api/v1/agent/semantic-search", methods=["POST"])
def agent_semantic_search():
    """Agent-facing semantic candidate recall helper."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "agent_semantic_search"):
            return err("当前存储后端不支持 Agent 语义检索", 400)
        body = get_json_body()
        role = body.get("role") or None
        result = storage.agent_semantic_search(
            body.get("query") or "",
            role=role,
            top_k=body.get("top_k", body.get("limit", 20)),
            threshold=body.get("threshold", 0.3),
            source_document=(body.get("source_document") or "").strip() or None,
        )
        result["graph_id"] = _get_graph_id()
        return ok(result)
    except (ValueError, TypeError) as exc:
        return err(str(exc), 400)
    except Exception as exc:
        logger.exception("Agent semantic search failed: %s", exc)
        return err("Agent 语义检索失败", 500)


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
        query = (body.get("query") or "").strip()
        if not query:
            return err("query 不能为空", 400)
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
        search_mode = str(body.get("search_mode", "bm25") or "bm25").strip().lower()
        if search_mode not in _VALID_SEARCH_MODES:
            return err(f"search_mode '{search_mode}' 无效，可选: {', '.join(_VALID_SEARCH_MODES)}", 400)
        time_point = (body.get("time_point") or "").strip() or None
        # Also accept time_after/time_before (used by the Web UI)
        time_after = (body.get("time_after") or "").strip() or None
        time_before = (body.get("time_before") or "").strip() or None
        # time_after/time_before take precedence over time_point
        if time_after or time_before:
            if time_after:
                time_point = time_after
            elif time_before:
                time_point = time_before
        source_document = (body.get("source_document") or "").strip() or None
        # max_name_length: opt-in filter to exclude long dialogue-fragment entity names.
        # Default 0 = disabled. Recommended: 15 to filter novel dialogue fragments.
        try:
            max_name_length = max(int(body.get("max_name_length", 0)), 0)
        except (ValueError, TypeError):
            max_name_length = 0
        reranker = (body.get("reranker") or "").strip().lower() or "rrf"
        if reranker not in _VALID_RERANKERS:
            return err(f"reranker '{reranker}' 无效，可选: {', '.join(_VALID_RERANKERS)}", 400)
        expand = bool(body.get("expand", False))
        group = bool(body.get("group", False))

        def _search(role_filter, result_limit):
            if search_mode == "bm25":
                # Fetch extra candidates so threshold filtering doesn't empty results
                candidate_limit = max(result_limit * 5, 50)
                if role_filter == "entity":
                    results = storage.search_entities_by_bm25(query, limit=candidate_limit, time_point=time_point)
                    results = [_entity_to_search_dict(e) for e in results]
                elif role_filter == "relation":
                    results = storage.search_relations_by_bm25(query, limit=candidate_limit, time_point=time_point)
                    results = [_relation_to_search_dict(r) for r in results]
                else:
                    results = storage.search_concepts_by_bm25(query, role=role_filter, limit=candidate_limit, time_point=time_point, source_document=source_document)
                # Apply threshold to BM25 results (BM25 _score is normalized 0-1)
                # For CJK queries, lower threshold to compensate for LIKE-based scoring
                bm25_thresh = min(threshold, 0.15) if _has_cjk(query) else threshold
                if bm25_thresh > 0:
                    results = [item for item in results if (item.get("_score") or 0.0) >= bm25_thresh]
                # Truncate to requested limit after threshold filtering
                results = results[:result_limit]
                meta = {"bm25_results": len(results), "semantic_results": 0, "effective_mode": "bm25_only"}
                return results, meta
            if search_mode == "semantic":
                sem_threshold = min(threshold, 0.3) if _has_cjk(query) else threshold
                # For non-CJK single-word queries (often cross-language), lower
                # threshold slightly to avoid losing borderline semantic matches.
                if not _has_cjk(query) and len(query.split()) <= 3 and sem_threshold > 0.45:
                    sem_threshold = 0.45
                results = storage.search_concepts_by_similarity(
                    query_text=query, role=role_filter, threshold=sem_threshold, max_results=result_limit, time_point=time_point, source_document=source_document
                )
                meta = {"bm25_results": 0, "semantic_results": len(results), "effective_mode": "semantic_only"}
                if _has_cjk(query):
                    meta["effective_mode"] = "semantic_cjk"
                # Apply role boost when no role filter is specified, matching
                # hybrid mode's entity > relation > episode > document priority.
                # Semantic cosine scores are in 0-1 range (typically 0.45-0.7),
                # so boost values need to be larger than RRF's ~0.01-scale to
                # have meaningful impact on ranking.
                if role_filter is None and results:
                    _role_rank = {"entity": 0.05, "relation": 0.02, "episode": 0.005, "document": 0.0}
                    for item in results:
                        item_role = item.get("role", "")
                        boost = _role_rank.get(item_role, 0.0)
                        if boost > 0:
                            item["_score"] = (item.get("_score") or 0.0) + boost
                    # Re-sort by boosted scores (storage returns pre-sorted, but
                    # boost changes the ordering). Skip if a reranker will handle it.
                    if reranker == "rrf":
                        results.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
                # Apply reranker to standalone semantic results too
                if reranker == "node_degree" and results and hasattr(storage, 'batch_get_entity_degrees'):
                    try:
                        fids = [item.get("family_id", "") or item.get("id", "") for item in results]
                        degree_map = storage.batch_get_entity_degrees(fids)
                        results = _node_degree_rerank_standalone(results, degree_map)
                        meta["reranker"] = "node_degree"
                    except Exception as exc:
                        logger.debug("node_degree reranker failed: %s", exc)
                elif reranker == "mmr" and results:
                    try:
                        results = _mmr_rerank_standalone(results, query_text=query, top_k=result_limit)
                        meta["reranker"] = "mmr"
                    except Exception as exc:
                        logger.debug("mmr reranker failed: %s", exc)
                return results, meta
            return _hybrid_concept_search(storage, query, role_filter, result_limit, threshold, time_point=time_point, source_document=source_document, reranker=reranker)

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
                            import numpy as _np
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


def _hybrid_concept_search(storage, query: str, role, limit: int,
                           threshold: float, time_point: str = None, source_document: str = None, reranker: str = "rrf"):
    """Hybrid concept search: BM25 + semantic embedding, fused via RRF.

    Returns (results, meta) where meta indicates which search modes contributed.

    For CJK queries, BM25 uses LIKE-based n-gram matching (not FTS5) and
    semantic threshold is lowered to 0.3 for better recall on short queries.
    """

    has_cjk = _has_cjk(query)

    # Fetch a larger BM25 candidate pool so that threshold filtering
    # doesn't accidentally empty the results for small limits.
    bm25_candidate_limit = max(limit * 5, 50)

    def _bm25():
        try:
            return storage.search_concepts_by_bm25(query, role=role, limit=bm25_candidate_limit, time_point=time_point, source_document=source_document)
        except Exception as exc:
            logger.warning("BM25 search failed for query=%r: %s", query, exc)
            return []

    # For CJK queries, lower the semantic threshold so short keyword
    # queries (e.g. "爱情") can match entity embeddings.
    # For short non-CJK queries (often cross-language), lower threshold to
    # 0.45 to avoid losing borderline semantic matches.
    semantic_threshold = threshold
    if has_cjk:
        semantic_threshold = min(threshold, 0.3)
    elif not has_cjk and len(query.split()) <= 3 and semantic_threshold > 0.45:
        semantic_threshold = 0.45

    def _semantic():
        try:
            return storage.search_concepts_by_similarity(
                query_text=query, role=role, threshold=semantic_threshold, max_results=limit * 2, time_point=time_point, source_document=source_document
            )
        except Exception as exc:
            logger.warning("Semantic search failed for query=%r: %s", query, exc)
            return []

    bm25_fut = _shared_pool.submit(_bm25)
    sem_fut = _shared_pool.submit(_semantic)
    bm25_results = bm25_fut.result()
    semantic_results = sem_fut.result()

    # Apply threshold to BM25 results in hybrid mode.
    # BM25 scores are normalized 0-1; filter out results below threshold.
    # For CJK queries, use a lower BM25 threshold because LIKE-based n-gram
    # matching produces lower scores for multi-word queries (each entity may
    # match only a subset of the space-separated terms).
    bm25_threshold = threshold
    if has_cjk:
        bm25_threshold = min(threshold, 0.15)
    if bm25_threshold > 0 and bm25_results:
        bm25_results = [item for item in bm25_results
                        if (item.get("_score") or 0.0) >= bm25_threshold]

    meta = {
        "bm25_results": len(bm25_results),
        "semantic_results": len(semantic_results),
        "effective_mode": "hybrid",
    }
    if has_cjk:
        meta["effective_mode"] = "hybrid_cjk"
        meta["reason"] = "CJK query — BM25 uses LIKE n-gram fallback, semantic threshold lowered"
    if not bm25_results and not semantic_results:
        return [], meta
    _base_mode = ""
    if not bm25_results:
        _base_mode = "semantic_only"
    elif not semantic_results:
        _base_mode = "bm25_only"
    if _base_mode:
        meta["effective_mode"] = (_base_mode + "_cjk") if has_cjk else _base_mode

    if not bm25_results and not semantic_results:
        return [], meta

    # RRF fusion on dict results (keyed by family_id)
    k = 60
    scores: Dict[str, float] = {}
    items: Dict[str, dict] = {}
    best_contrib: Dict[str, float] = {}
    bm25_weight = 0.3
    sem_weight = 0.7

    for rank, item in enumerate(bm25_results):
        fid = item.get("family_id", "") or item.get("id", "")
        rrf = bm25_weight / (k + rank + 1)
        scores[fid] = scores.get(fid, 0.0) + rrf
        if fid not in items or rrf > best_contrib.get(fid, 0.0):
            items[fid] = item
            best_contrib[fid] = rrf

    for rank, item in enumerate(semantic_results):
        fid = item.get("family_id", "") or item.get("id", "")
        rrf = sem_weight / (k + rank + 1)
        scores[fid] = scores.get(fid, 0.0) + rrf
        if fid not in items or rrf > best_contrib.get(fid, 0.0):
            items[fid] = item
            best_contrib[fid] = rrf

    # When no role filter is applied, boost entity results above relation
    # results, and relation above episode/document, to match the documented
    # ranking priority: entity > relation > episode > document.
    role_boost = {}
    if role is None:
        _role_rank = {"entity": 0.03, "relation": 0.015, "episode": 0.005, "document": 0.0}
        for fid in scores:
            item_role = items.get(fid, {}).get("role", "")
            role_boost[fid] = _role_rank.get(item_role, 0.0)

    sorted_items = sorted(scores.items(), key=lambda x: x[1] + role_boost.get(x[0], 0.0), reverse=True)
    fused = []
    for fid, rrf_score in sorted_items[:limit]:
        item = items[fid]
        final_score = rrf_score + role_boost.get(fid, 0.0)
        item["_score"] = round(final_score, 6)
        fused.append(item)

    # Apply reranker (rrf = no-op, already RRF fused above)
    if reranker == "node_degree" and fused and hasattr(storage, 'batch_get_entity_degrees'):
        try:
            fids = [item.get("family_id", "") or item.get("id", "") for item in fused]
            degree_map = storage.batch_get_entity_degrees(fids)
            fused = _node_degree_rerank_standalone(fused, degree_map)
            meta["reranker"] = "node_degree"
        except Exception as exc:
            logger.debug("node_degree reranker failed: %s", exc)
    elif reranker == "mmr" and fused:
        try:
            fused = _mmr_rerank_standalone(fused, query_text=query, top_k=limit)
            meta["reranker"] = "mmr"
        except Exception as exc:
            logger.debug("mmr reranker failed: %s", exc)

    return fused, meta


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
        time_point = (body.get("time_point") or "").strip() or None
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
        time_point = (body.get("time_point") or "").strip() or None
        edge_types = body.get("edge_types") or body.get("edge_type") or None
        if isinstance(edge_types, str):
            edge_types = [edge_types]
        # Scale per-level timeout with depth: ~15s per level, min 30s total
        _traverse_timeout = max(30.0, 15.0 * max_depth)
        result = storage.traverse_concepts(start_ids, max_depth=max_depth, time_point=time_point, edge_types=edge_types, max_results=max_results, _timeout_seconds=_traverse_timeout)
        return ok(result)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/documents", methods=["GET"])
def list_documents():
    """List indexed Markdown documents for the current graph."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "list_documents"):
            return err("此功能暂不可用", 400)
        try:
            limit = min(max(int(request.args.get("limit", 50)), 1), 200)
        except (ValueError, TypeError):
            return err("limit 必须为整数", 400)
        try:
            offset = max(int(request.args.get("offset", 0)), 0)
        except (ValueError, TypeError):
            return err("offset 必须为整数", 400)
        source_document = (request.args.get("source_document") or "").strip() or None
        documents = storage.list_documents(limit=limit, offset=offset, source_document=source_document)
        runtime = (current_app.config.get("config") or {}).get("runtime") or {}
        integrity_cfg = runtime.get("integrity") or {}
        if bool(integrity_cfg.get("auto_check_documents", True)) and documents:
            try:
                from core.server.routes.remember import _get_queue as _get_remember_queue
                remember_queue = _get_remember_queue()
                for doc in documents:
                    doc_id = doc.get("document_version_id")
                    if not doc_id:
                        continue
                    try:
                        integrity = remember_queue.assess_document_integrity(doc_id)
                        doc["integrity"] = {
                            "complete": bool(integrity.get("complete")),
                            "total_windows": integrity.get("total_windows", 0),
                            "complete_windows": integrity.get("complete_windows", 0),
                            "missing_windows": integrity.get("missing_windows", 0),
                            "missing_window_indices": integrity.get("missing_window_indices", [])[:20],
                        }
                        if hasattr(storage, "update_document_integrity_metadata"):
                            storage.update_document_integrity_metadata(doc_id, integrity)
                    except Exception as exc:
                        doc["integrity"] = {"complete": None, "error": str(exc)}
            except Exception as exc:
                logger.debug("document integrity auto-check skipped: %s", exc)
        # Get actual total count (independent of pagination)
        total = storage.count_documents(source_document=source_document) if hasattr(storage, "count_documents") else len(documents)
        return ok({"documents": documents, "total": total, "limit": limit, "offset": offset})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/documents/graph", methods=["POST"])
def get_documents_graph():
    """Return a Document -> Episode -> Concept subgraph for selected documents."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "get_document_graph"):
            return err("此功能暂不可用", 400)
        body = get_json_body()
        document_version_ids = body.get("document_version_ids") or []
        document_family_ids = body.get("document_family_ids") or []
        if isinstance(document_version_ids, str):
            document_version_ids = [document_version_ids]
        if isinstance(document_family_ids, str):
            document_family_ids = [document_family_ids]
        if not document_version_ids and not document_family_ids:
            return err("document_version_ids 或 document_family_ids 至少提供一个", 400)
        include_relations = bool(body.get("include_relations", True))
        include_versions = bool(body.get("include_versions", True))
        try:
            max_episodes = min(max(int(body.get("max_episodes", 5000)), 1), 10000)
        except (ValueError, TypeError):
            return err("max_episodes 必须为整数", 400)
        try:
            max_concepts = min(max(int(body.get("max_concepts", 20000)), 1), 50000)
        except (ValueError, TypeError):
            return err("max_concepts 必须为整数", 400)
        result = storage.get_document_graph(
            document_version_ids=document_version_ids,
            document_family_ids=document_family_ids,
            include_relations=include_relations,
            include_versions=include_versions,
            max_episodes=max_episodes,
            max_concepts=max_concepts,
        )
        return ok(result)
    except ValueError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/documents/graph/outline", methods=["POST"])
def get_documents_graph_outline():
    """Return the fast Document -> Episode skeleton for progressive graph rendering."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "get_document_graph_outline"):
            return err("此功能暂不可用", 400)
        body = get_json_body()
        document_version_ids = body.get("document_version_ids") or []
        document_family_ids = body.get("document_family_ids") or []
        if isinstance(document_version_ids, str):
            document_version_ids = [document_version_ids]
        if isinstance(document_family_ids, str):
            document_family_ids = [document_family_ids]
        if not document_version_ids and not document_family_ids:
            return err("document_version_ids 或 document_family_ids 至少提供一个", 400)
        try:
            max_episodes = min(max(int(body.get("max_episodes", 10000)), 1), 10000)
        except (ValueError, TypeError):
            return err("max_episodes 必须为整数", 400)
        result = storage.get_document_graph_outline(
            document_version_ids=document_version_ids,
            document_family_ids=document_family_ids,
            max_episodes=max_episodes,
        )
        return ok(result)
    except ValueError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/documents/graph/chunk", methods=["POST"])
def get_documents_graph_chunk():
    """Return one episode-ordered concept batch for progressive graph rendering."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "get_document_graph_chunk"):
            return err("此功能暂不可用", 400)
        body = get_json_body()
        document_version_ids = body.get("document_version_ids") or []
        document_family_ids = body.get("document_family_ids") or []
        if isinstance(document_version_ids, str):
            document_version_ids = [document_version_ids]
        if isinstance(document_family_ids, str):
            document_family_ids = [document_family_ids]
        if not document_version_ids and not document_family_ids:
            return err("document_version_ids 或 document_family_ids 至少提供一个", 400)
        try:
            cursor = max(int(body.get("cursor", 0)), 0)
        except (ValueError, TypeError):
            return err("cursor 必须为整数", 400)
        try:
            limit = min(max(int(body.get("limit", 12)), 1), 100)
        except (ValueError, TypeError):
            return err("limit 必须为整数", 400)
        include_relations = bool(body.get("include_relations", True))
        include_versions = bool(body.get("include_versions", True))
        try:
            max_concepts = min(max(int(body.get("max_concepts", 8000)), 1), 50000)
        except (ValueError, TypeError):
            return err("max_concepts 必须为整数", 400)
        result = storage.get_document_graph_chunk(
            document_version_ids=document_version_ids,
            document_family_ids=document_family_ids,
            cursor=cursor,
            limit=limit,
            include_relations=include_relations,
            include_versions=include_versions,
            max_concepts=max_concepts,
        )
        return ok(result)
    except ValueError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/episodes/<episode_version_id>/content", methods=["GET"])
def get_episode_content(episode_version_id: str):
    """Return source content for an episode version."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if hasattr(storage, "get_episode_content_detail"):
            detail = storage.get_episode_content_detail(episode_version_id)
            if detail is None:
                return err(f"episode_version_id 不存在: {episode_version_id}", 404)
            return ok(detail)
        if not hasattr(storage, "load_episode"):
            return err("此功能暂不可用", 400)
        episode = storage.load_episode(episode_version_id)
        if episode is None:
            return err(f"episode_version_id 不存在: {episode_version_id}", 404)
        return ok({
            "episode_id": episode.absolute_id,
            "content": episode.content,
            "source_document": episode.source_document or "",
            "event_time": episode.event_time.isoformat() if episode.event_time else None,
            "processed_time": episode.processed_time.isoformat() if episode.processed_time else None,
            "activity_type": getattr(episode, "activity_type", None),
            "episode_type": getattr(episode, "episode_type", None),
        })
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/documents/batch", methods=["DELETE"])
def batch_delete_documents():
    """批量删除文档版本。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "delete_document_version"):
            return err("此功能暂不可用", 400)
        body = get_json_body()
        ids = body.get("document_version_ids") or []
        if not ids:
            return err("document_version_ids 不能为空", 400)
        if not isinstance(ids, list) or len(ids) > 100:
            return err("document_version_ids 必须为列表，最多 100 个", 400)
        results = []
        for doc_id in ids:
            try:
                storage.delete_document_version(doc_id)
                results.append({"id": doc_id, "success": True})
            except Exception as e:
                results.append({"id": doc_id, "success": False, "error": str(e)})
        return ok({"results": results, "deleted": sum(1 for r in results if r["success"])})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/documents/<document_version_id>", methods=["DELETE"])
def delete_document_version(document_version_id: str):
    """删除文档版本，以及该文档下的 episode/concept version/edge。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "delete_document_version"):
            return err("此功能暂不可用", 400)
        result = storage.delete_document_version(document_version_id)
        return ok(result)
    except KeyError as e:
        return err(str(e.args[0]) if e.args else str(e), 404)
    except ValueError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/vaults/index", methods=["POST"])
def index_vault():
    """Index a read-only Markdown/Obsidian vault into the current graph."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "index_vault"):
            return err("此功能暂不可用", 400)
        body = get_json_body()
        path = (body.get("path") or body.get("vault_path") or "").strip()
        if not path:
            return err("path 不能为空", 400)
        force = bool(body.get("force", False))
        result = storage.index_vault(path, force=force)
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
            limit = min(int(request.args.get("limit", 500)), 2000)
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

