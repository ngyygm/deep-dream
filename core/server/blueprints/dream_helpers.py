"""
Helper functions for the Dream blueprint.

Contains background task logic, data formatting helpers, and response builders
used by the route handlers in dream.py.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.server.blueprints.helpers import (
    entity_to_dict,
    relation_to_dict,
    enrich_relations,
    enrich_entity_version_counts,
    enrich_relation_version_counts,
    run_async,
)
from core.server.llm_utils import call_llm_with_backoff
from core.find.graph_traversal import GraphTraversalSearcher
from core.find.hybrid import HybridSearcher

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
VALID_STRATEGIES = frozenset(("random", "orphan", "hub", "time_gap", "cross_community", "low_confidence"))


# ── Shared search execution for ask endpoints ─────────────────────────────────

def execute_ask_search(processor, query_type: str, query_text: str, intent: dict):
    """Execute search based on parsed query plan.

    Returns (entities, relations, entity_score_map, relation_score_map).
    """
    entities = []
    relations = []
    entity_score_map: Dict[str, float] = {}
    relation_score_map: Dict[str, float] = {}

    if query_type == "traverse":
        entity_name = intent.get("entity_name", "")
        if entity_name:
            seed_entities = processor.storage.search_entities_by_bm25(entity_name, limit=3)
            seed_ids = [e.family_id for e in seed_entities]
            if seed_ids:
                searcher = GraphTraversalSearcher(processor.storage)
                entities = searcher.bfs_expand(seed_ids, max_depth=2, max_nodes=20)
                # Fetch relations for the expanded entities
                seen_rids = set()
                for e in entities:
                    for r in processor.storage.get_entity_relations_by_family_id(e.family_id, limit=10):
                        if r.family_id not in seen_rids:
                            seen_rids.add(r.family_id)
                            relations.append(r)
    else:
        # Compute query embedding for hybrid search (vector + BM25)
        query_embedding = None
        try:
            ec = getattr(processor.storage, 'embedding_client', None)
            if ec and getattr(ec, 'is_available', lambda: False)():
                query_embedding = ec.encode([query_text])[0]
        except Exception as _emb_err:
            logger.warning("ask search embedding 失败: %s", _emb_err)
        # Reuse cached searcher on storage object
        _searcher = getattr(processor.storage, '_hybrid_searcher', None)
        if _searcher is None:
            _searcher = HybridSearcher(processor.storage)
            processor.storage._hybrid_searcher = _searcher
        searcher = _searcher
        entity_hits = searcher.search_entities(query_text=query_text, query_embedding=query_embedding, top_k=20)
        relation_hits = searcher.search_relations(query_text=query_text, query_embedding=query_embedding, top_k=10)
        # Apply confidence-weighted reranking with time decay
        entity_hits = searcher.confidence_rerank(entity_hits, alpha=0.2, time_decay_half_life_days=90.0)
        relation_hits = searcher.confidence_rerank(relation_hits, alpha=0.2, time_decay_half_life_days=90.0)
        # Preserve scores from hybrid search (after reranking)
        entity_score_map = {e.absolute_id: score for e, score in entity_hits}
        relation_score_map = {r.absolute_id: score for r, score in relation_hits}
        entities = [e for e, _ in entity_hits]
        relations = [r for r, _ in relation_hits]

    return entities, relations, entity_score_map, relation_score_map


def serialize_ask_results(entities, relations, entity_score_map, relation_score_map, processor):
    """Serialize search results to dicts with scores and version counts."""
    entity_dicts = [entity_to_dict(e, _score=entity_score_map.get(e.absolute_id)) for e in entities]
    relation_dicts = [relation_to_dict(r, _score=relation_score_map.get(r.absolute_id)) for r in relations]
    enrich_entity_version_counts(entity_dicts, processor.storage)
    enrich_relation_version_counts(relation_dicts, processor.storage)
    enrich_relations(relation_dicts, processor)
    return entity_dicts, relation_dicts


# ── LLM backoff helper ────────────────────────────────────────────────────────

def call_llm(processor, prompt, timeout=60, max_waits=5, backoff_base_seconds=2):
    """调用 LLM（指数退避重试）—— 代理到共享模块。"""
    return call_llm_with_backoff(processor, prompt, timeout=timeout, max_waits=max_waits, backoff_base_seconds=backoff_base_seconds)


# ── Butler recommendation builder ─────────────────────────────────────────────

def build_butler_recommendations(
    health: dict,
    quality: dict,
    dream_status_data: dict,
) -> List[dict]:
    """Build actionable recommendations from health / quality / dream data.

    Returns a list of recommendation dicts sorted by priority.
    """
    recommendations = []

    iso_count = quality.get("isolated_entities", 0)
    inv_count = quality.get("invalidated_entities", 0) + quality.get("invalidated_relations", 0)

    if iso_count > 0:
        recommendations.append({
            "action": "cleanup_isolated",
            "priority": "high" if iso_count > 20 else "medium",
            "description": f"发现 {iso_count} 个孤立实体（无关联关系），建议清理",
            "estimated_impact": f"释放约 {iso_count} 个实体的存储空间",
            "dry_run_available": True,
        })

    if inv_count > 0:
        recommendations.append({
            "action": "cleanup_invalidated",
            "priority": "medium",
            "description": f"发现 {inv_count} 个已失效版本，建议清理",
            "estimated_impact": f"释放约 {inv_count} 个节点的存储空间",
            "dry_run_available": True,
        })

    # Dangling entity refs in relations
    quality_rels = quality.get("relations", {})
    dangling_count = quality_rels.get("dangling_entity_refs", 0) if isinstance(quality_rels, dict) else 0
    if dangling_count > 0:
        recommendations.append({
            "action": "fix_dangling_refs",
            "priority": "high" if dangling_count > 20 else "medium",
            "description": f"发现 {dangling_count} 个关系中的悬空实体引用，建议修复",
            "estimated_impact": f"修复约 {dangling_count} 个数据完整性问题",
            "dry_run_available": True,
        })

    total_ent = health["total_entities"]
    total_rel = health["total_relations"]
    if total_ent > 0 and total_rel < total_ent * 0.3:
        recommendations.append({
            "action": "run_dream",
            "priority": "high",
            "description": f"关系密度低（{total_rel}/{total_ent}），建议运行梦境发现隐含关联",
            "estimated_impact": "发现并创建新的跨域关系，提升图谱连通性",
            "dream_type_suggestion": "free_association",
        })

    if dream_status_data["status"] == "no_cycles":
        recommendations.append({
            "action": "run_dream",
            "priority": "medium",
            "description": "尚未运行过梦境周期，建议开始首次探索",
            "estimated_impact": "发现图谱中隐含的概念关联",
            "dream_type_suggestion": "random",
        })

    # 社区检测建议
    if health.get("total_communities", 0) == 0 and total_ent > 20:
        recommendations.append({
            "action": "detect_communities",
            "priority": "low",
            "description": f"图谱有 {total_ent} 个实体但未做社区检测，建议运行以发现主题聚类",
            "estimated_impact": "识别知识领域边界，辅助梦境探索策略",
        })

    # 实体摘要进化建议 — sample-based check (storage-agnostic)
    recommendations = _check_summary_recommendations(recommendations, health)

    recommendations.sort(key=lambda r: PRIORITY_ORDER.get(r["priority"], 3))
    return recommendations


def _check_summary_recommendations(recommendations: List[dict], health: dict) -> List[dict]:
    """Append a recommendation if many entities are missing summaries.

    This is separated out because it needs access to *storage* which the
    caller provides indirectly via the health dict's _sample attributes.
    For now we rely on the caller having pre-computed the sample stats.
    """
    # The sample stats are attached by the caller as private keys on health.
    no_summary = health.get("_sample_no_summary", 0)
    sample_total = health.get("_sample_total", 0)
    if sample_total > 0 and no_summary > sample_total * 0.5:
        recommendations.append({
            "action": "evolve_summaries",
            "priority": "low",
            "description": f"抽样显示 {no_summary}/{sample_total} 个实体缺少摘要",
            "estimated_impact": "提升语义检索质量",
        })
    return recommendations


# ── Butler action runner ──────────────────────────────────────────────────────

def run_butler_action(storage, action: str, dry_run: bool) -> dict:
    """Run a single independent butler action and return the result dict."""
    if action == "cleanup_isolated":
        isolated = storage.get_isolated_entities(limit=10000)
        family_ids = list({e.family_id for e in isolated if e.family_id})
        if dry_run:
            return {"status": "preview", "count": len(family_ids), "family_ids": family_ids[:20]}
        else:
            deleted = storage.batch_delete_entities(family_ids)
            return {"status": "done", "deleted_families": len(family_ids), "deleted_versions": deleted}

    elif action == "cleanup_invalidated":
        if hasattr(storage, 'cleanup_invalidated_versions'):
            return storage.cleanup_invalidated_versions(dry_run=dry_run)
        return {"status": "skipped", "reason": "当前存储后端不支持"}

    elif action == "detect_communities":
        if hasattr(storage, 'detect_communities'):
            return storage.detect_communities()
        return {"status": "skipped", "reason": "当前存储后端不支持社区检测"}

    elif action == "fix_dangling_refs":
        if hasattr(storage, "fix_dangling_relation_refs"):
            return storage.fix_dangling_relation_refs(dry_run=dry_run)
        return {"status": "skipped", "reason": "当前存储后端不支持"}

    elif action == "cleanup_stale_redirects":
        if dry_run:
            return {"status": "preview", "message": "Will delete stale EntityRedirect nodes whose target entities no longer exist"}
        deleted = storage.cleanup_stale_redirects()
        return {"status": "done", "deleted": deleted}

    return {"status": "unknown", "reason": f"未知操作: {action}"}


def evolve_summaries(processor, storage, dry_run: bool, dream_pool) -> dict:
    """Evolve entity summaries for entities that lack them.

    Returns a result dict with evolved/failed counts.
    """
    evolved = 0
    failed = 0
    sample = storage.get_all_entities(limit=20, exclude_embedding=True)
    to_evolve = [e for e in sample if not getattr(e, 'summary', None)]

    if to_evolve:
        def _evolve_one(entity):
            try:
                summary = run_async(processor.llm_client.evolve_entity_summary(entity))
                return (entity.family_id, summary, None)
            except Exception as ex:
                return (entity.family_id, None, ex)

        # Parallelize LLM calls
        summaries = {}
        if len(to_evolve) > 1:
            for fid, summary, _evo_err in dream_pool.map(_evolve_one, to_evolve):
                if _evo_err:
                    logger.warning("evolve_entity_summary %s 失败: %s", fid, _evo_err)
                    failed += 1
                else:
                    summaries[fid] = summary
                    evolved += 1
        else:
            for e in to_evolve:
                fid, summary, _evo_err = _evolve_one(e)
                if _evo_err:
                    logger.warning("evolve_entity_summary %s 失败: %s", fid, _evo_err)
                    failed += 1
                else:
                    summaries[fid] = summary
                    evolved += 1

        # Batch write summaries
        if summaries:
            batch_fn = getattr(storage, 'batch_update_entity_summaries', None)
            if batch_fn:
                try:
                    batch_fn(summaries)
                except Exception:
                    for fid, summary in summaries.items():
                        try:
                            storage.update_entity_summary(fid, summary)
                        except Exception:
                            pass
            else:
                for fid, summary in summaries.items():
                    try:
                        storage.update_entity_summary(fid, summary)
                    except Exception:
                        pass

    return {"status": "done", "evolved": evolved, "failed": failed, "dry_run": dry_run}


# ── Dream seed formatting helper ──────────────────────────────────────────────

def format_seeds(seeds: list) -> list:
    """Format dream seed dicts for JSON response."""
    for s in seeds:
        if s.get("event_time"):
            s["event_time"] = str(s["event_time"])
        if s.get("confidence") is not None:
            s["confidence"] = round(float(s["confidence"]), 4)
        if s.get("degree") is not None:
            s["degree"] = int(s["degree"])
        if s.get("community_id") is not None:
            s["community_id"] = int(s["community_id"])
    return seeds


# ── Butler report data builders ───────────────────────────────────────────────

def build_health_data(graph_stats: dict, graph_id, embedding_available: bool) -> dict:
    """Build the health section of the butler report."""
    return {
        "graph_id": graph_id,
        "total_entities": graph_stats.get("entity_count", 0),
        "total_relations": graph_stats.get("relation_count", 0),
        "total_episodes": graph_stats.get("episode_count", 0),
        "total_communities": graph_stats.get("community_count", 0),
        "storage_backend": "sqlite",
        "embedding_available": embedding_available,
    }


def build_dream_status_from_logs(logs: list) -> dict:
    """Build dream status dict from the latest dream log entry."""
    dream_status_data = {"status": "not_available", "last_cycle_id": None, "last_cycle_time": None}
    if logs:
        last = logs[0]
        dream_status_data = {
            "status": last.get("status", "completed"),
            "last_cycle_id": last.get("cycle_id"),
            "last_cycle_time": last.get("end_time") or last.get("start_time") or last.get("started_at") or last.get("created_at"),
            "entities_explored": last.get("entities_explored", 0),
            "relations_created": last.get("relations_created", 0),
        }
    else:
        dream_status_data["status"] = "no_cycles"
    return dream_status_data


def compute_sample_summary_stats(storage) -> tuple:
    """Check a sample of entities for missing summaries.

    Returns (no_summary_count, sample_total).
    """
    no_summary = 0
    sample_total = 0
    try:
        sample = storage.get_all_entities(limit=50, exclude_embedding=True)
        sample_total = len(sample)
        no_summary = sum(1 for e in sample if not getattr(e, 'summary', None))
    except Exception:
        sample_total = 0
    return no_summary, sample_total


# ── SSE fallback answer builder ───────────────────────────────────────────────

def build_fallback_answer(query_text: str, entities: list, relations: list) -> str:
    """Build a simple fallback answer when LLM synthesis fails in SSE stream."""
    parts = [f"基于「{query_text}」的检索结果："]
    if entities:
        parts.append(f"找到 {len(entities)} 个相关实体")
        for e in entities[:5]:
            name = getattr(e, 'name', '') or ''
            content = (getattr(e, 'content', '') or '')[:80]
            parts.append(f"  - {name}: {content}")
    if relations:
        parts.append(f"找到 {len(relations)} 条相关关系")
        for r in relations[:5]:
            content = (getattr(r, 'content', '') or '')[:80]
            parts.append(f"  - {content}")
    return "\n".join(parts)
