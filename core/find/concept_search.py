"""统一概念搜索执行（P4.2/P4.3 自 core/server/routes/concepts.py 抽出）。

server 路由（POST /api/v1/concepts/search 与 /api/v1/find）与 CLI
``concept search --mode bm25|semantic|hybrid`` 共用本模块：RRF 融合、
role boost、CJK 短查询阈值、BM25 阈值过滤只有这一份实现，
两端行为以本模块为准（此前 CLI 自带一套去重拼接，与 server 不一致）。

本模块不依赖 Flask；storage 协议 = LibraryManager 暴露的检索面：
  - search_concepts_by_bm25 / search_entities_by_bm25 / search_relations_by_bm25
  - agent_semantic_search（语义检索单入口，P4.2）
  - batch_get_entity_degrees / get_concept_neighbors（可选增强）
"""
from __future__ import annotations

import logging
import re as _re
from typing import Dict

logger = logging.getLogger(__name__)

# ── CJK detection for BM25 fallback ──────────────────────────────────────────
_CJK_RE = _re.compile(r'[一-鿿㐀-䶿]')


def has_cjk(query: str) -> bool:
    """Return True if the query contains any CJK character."""
    if not query:
        return False
    return bool(_CJK_RE.search(query))


def entity_to_search_dict(e):
    """Convert an Entity object to a search result dict."""
    return {
        "family_id": e.family_id,
        "id": e.absolute_id,
        "name": e.name,
        "content": e.content,
        "role": "entity",
        "_score": getattr(e, "_score", 0.0),
    }


def relation_to_search_dict(r):
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


def normalize_results(results: list) -> list:
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

def node_degree_rerank(items, degree_map, alpha=0.3):
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
    if has_cjk(text):
        # Use character bigrams for CJK text — captures meaningful
        # sub-string overlap without requiring word segmentation.
        bigrams = set()
        chars = [c for c in text if not c.isspace()]
        for i in range(len(chars) - 1):
            bigrams.add(chars[i] + chars[i + 1])
        return bigrams if bigrams else {text}
    return set(text.split()) if text.strip() else set()


def mmr_rerank(items, query_text="", lambda_=0.5, top_k=20):
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


# ── 三种检索模式的统一执行（server /find、/concepts/search 与 CLI 共用）───────

def bm25_concept_search(storage, query: str, role_filter, result_limit: int,
                        threshold: float, time_point: str = None,
                        source_document: str = None,
                        time_after: str = None, time_before: str = None):
    """BM25 模式：按 role 分腿检索 + 阈值过滤 + 截断。

    返回 (results, meta)，meta 为 {"bm25_results", "semantic_results",
    "effective_mode"}。BM25 _score 归一化 0-1；CJK 查询阈值降至 0.15
    （LIKE n-gram 打分偏低）。
    """
    # Fetch extra candidates so threshold filtering doesn't empty results
    candidate_limit = max(result_limit * 5, 50)
    if role_filter == "entity":
        results = storage.search_entities_by_bm25(query, limit=candidate_limit, time_point=time_point, source_document=source_document, time_after=time_after, time_before=time_before)
        results = [entity_to_search_dict(e) for e in results]
    elif role_filter == "relation":
        results = storage.search_relations_by_bm25(query, limit=candidate_limit, time_point=time_point, source_document=source_document, time_after=time_after, time_before=time_before)
        results = [relation_to_search_dict(r) for r in results]
    else:
        results = storage.search_concepts_by_bm25(query, role=role_filter, limit=candidate_limit, time_point=time_point, source_document=source_document, time_after=time_after, time_before=time_before)
    # Apply threshold to BM25 results (BM25 _score is normalized 0-1)
    # For CJK queries, lower threshold to compensate for LIKE-based scoring
    bm25_thresh = min(threshold, 0.15) if has_cjk(query) else threshold
    if bm25_thresh > 0:
        results = [item for item in results if (item.get("_score") or 0.0) >= bm25_thresh]
    # Truncate to requested limit after threshold filtering
    results = results[:result_limit]
    meta = {"bm25_results": len(results), "semantic_results": 0, "effective_mode": "bm25_only"}
    return results, meta


def semantic_concept_search(storage, query: str, role_filter, result_limit: int,
                            threshold: float, reranker: str = "rrf",
                            time_point: str = None, source_document: str = None,
                            time_after: str = None, time_before: str = None):
    """语义模式：统一走 storage.agent_semantic_search 单入口（P4.2）。

    返回 (results, meta)。CJK/短查询阈值放宽与 role boost 与 hybrid 模式
    同一套规则（见模块 docstring）。
    """
    sem_threshold = min(threshold, 0.3) if has_cjk(query) else threshold
    # For non-CJK single-word queries (often cross-language), lower
    # threshold slightly to avoid losing borderline semantic matches.
    if not has_cjk(query) and len(query.split()) <= 3 and sem_threshold > 0.45:
        sem_threshold = 0.45
    result = storage.agent_semantic_search(
        query, role=role_filter, top_k=result_limit, threshold=sem_threshold,
        source_document=source_document, time_point=time_point,
        time_after=time_after, time_before=time_before,
    )
    results = result.get("results", [])
    meta = {"bm25_results": 0, "semantic_results": len(results), "effective_mode": "semantic_only"}
    if has_cjk(query):
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
            results = node_degree_rerank(results, degree_map)
            meta["reranker"] = "node_degree"
        except Exception as exc:
            logger.debug("node_degree reranker failed: %s", exc)
    elif reranker == "mmr" and results:
        try:
            results = mmr_rerank(results, query_text=query, top_k=result_limit)
            meta["reranker"] = "mmr"
        except Exception as exc:
            logger.debug("mmr reranker failed: %s", exc)
    return results, meta


def hybrid_concept_search(storage, query: str, role, limit: int,
                          threshold: float, time_point: str = None, source_document: str = None, reranker: str = "rrf",
                          time_after: str = None, time_before: str = None):
    """Hybrid concept search: BM25 + semantic embedding, fused via RRF.

    Returns (results, meta) where meta indicates which search modes contributed.

    For CJK queries, BM25 uses LIKE-based n-gram matching (not FTS5) and
    semantic threshold is lowered to 0.3 for better recall on short queries.

    P2.6：本函数不向调用方的共享线程池嵌套 submit，BM25/语义两路在调用线程
    同步执行。外层 /find 已把 entity/relation 检索提交到同一个共享池并阻塞
    等待结果；若此处再向同一池 submit，并发请求下全部 worker 都在等待排在
    自己后面的内层任务，互等即自死锁。且 storage 为 per-thread sqlite 连接
    （不变式 c），阻塞任务不得嵌套 submit 到同一池。
    """

    cjk = has_cjk(query)

    # Fetch a larger BM25 candidate pool so that threshold filtering
    # doesn't accidentally empty the results for small limits.
    bm25_candidate_limit = max(limit * 5, 50)

    def _bm25():
        try:
            return storage.search_concepts_by_bm25(query, role=role, limit=bm25_candidate_limit, time_point=time_point, source_document=source_document, time_after=time_after, time_before=time_before)
        except Exception as exc:
            logger.warning("BM25 search failed for query=%r: %s", query, exc)
            return []

    # For CJK queries, lower the semantic threshold so short keyword
    # queries (e.g. "爱情") can match entity embeddings.
    # For short non-CJK queries (often cross-language), lower threshold to
    # 0.45 to avoid losing borderline semantic matches.
    semantic_threshold = threshold
    if cjk:
        semantic_threshold = min(threshold, 0.3)
    elif not cjk and len(query.split()) <= 3 and semantic_threshold > 0.45:
        semantic_threshold = 0.45

    def _semantic():
        try:
            result = storage.agent_semantic_search(
                query, role=role, top_k=limit * 2, threshold=semantic_threshold,
                source_document=source_document, time_point=time_point,
                time_after=time_after, time_before=time_before,
            )
            return result.get("results", [])
        except Exception as exc:
            logger.warning("Semantic search failed for query=%r: %s", query, exc)
            return []

    # 同步执行（见函数 docstring 的 P2.6 说明）：不得向共享池嵌套 submit
    bm25_results = _bm25()
    semantic_results = _semantic()

    # Apply threshold to BM25 results in hybrid mode.
    # BM25 scores are normalized 0-1; filter out results below threshold.
    # For CJK queries, use a lower BM25 threshold because LIKE-based n-gram
    # matching produces lower scores for multi-word queries (each entity may
    # match only a subset of the space-separated terms).
    bm25_threshold = threshold
    if cjk:
        bm25_threshold = min(threshold, 0.15)
    if bm25_threshold > 0 and bm25_results:
        bm25_results = [item for item in bm25_results
                        if (item.get("_score") or 0.0) >= bm25_threshold]

    meta = {
        "bm25_results": len(bm25_results),
        "semantic_results": len(semantic_results),
        "effective_mode": "hybrid",
    }
    if cjk:
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
        meta["effective_mode"] = (_base_mode + "_cjk") if cjk else _base_mode

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
            fused = node_degree_rerank(fused, degree_map)
            meta["reranker"] = "node_degree"
        except Exception as exc:
            logger.debug("node_degree reranker failed: %s", exc)
    elif reranker == "mmr" and fused:
        try:
            fused = mmr_rerank(fused, query_text=query, top_k=limit)
            meta["reranker"] = "mmr"
        except Exception as exc:
            logger.debug("mmr reranker failed: %s", exc)

    return fused, meta
