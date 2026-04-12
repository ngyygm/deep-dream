"""
Dream blueprint — Dream exploration, ask/explain/suggestions, quality report,
maintenance, and butler routes.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import Blueprint, request

from server.blueprints.helpers import (
    ok,
    err,
    _get_processor,
    _get_graph_id,
    entity_to_dict,
    relation_to_dict,
    enrich_relations,
    enrich_entity_version_counts,
    enrich_relation_version_counts,
    parse_time_point,
)
from server.sse import sse_response, queue_to_generator

logger = logging.getLogger(__name__)

dream_bp = Blueprint("dream", __name__)


# ── LLM backoff helper (used by ask / explain / dream) ────────────────────

def _call_llm_with_backoff(processor, prompt, timeout=60, max_waits=5, backoff_base_seconds=2):
    """调用 LLM（指数退避重试）—— 代理到共享模块。"""
    from server.llm_utils import call_llm_with_backoff
    return call_llm_with_backoff(processor, prompt, timeout=timeout, max_waits=max_waits, backoff_base_seconds=backoff_base_seconds)


# =========================================================
# Phase E: DeepDream 记忆巩固（积木端点 — 编排由 Agent Skill 驱动）
# =========================================================

@dream_bp.route("/api/v1/find/dream/status", methods=["GET"])
def dream_status():
    """查询梦境状态（最近一次）。"""
    try:
        processor = _get_processor()
        logs = processor.storage.list_dream_logs(request.graph_id or "default", limit=1)
        if logs:
            return ok(logs[0])
        return ok({"status": "no_cycles"})
    except Exception as e:
        return err(str(e), 500)


@dream_bp.route("/api/v1/find/dream/logs", methods=["GET"])
def dream_logs():
    """历史梦境日志列表。"""
    try:
        processor = _get_processor()
        limit = request.args.get("limit", type=int, default=20)
        logs = processor.storage.list_dream_logs(request.graph_id or "default", limit=limit)
        return ok(logs)
    except Exception as e:
        return err(str(e), 500)


@dream_bp.route("/api/v1/find/dream/logs/<cycle_id>", methods=["GET"])
def dream_log_detail(cycle_id: str):
    """单条梦境日志详情。"""
    try:
        processor = _get_processor()
        log = processor.storage.get_dream_log(cycle_id)
        if log is None:
            return err(f"未找到梦境日志: {cycle_id}", 404)
        return ok(log)
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Phase E.2: DeepDream 梦境积木端点 — 种子 / 关系 / 记录
# =========================================================

@dream_bp.route("/api/v1/find/dream/seeds", methods=["POST"])
def dream_seeds():
    """获取梦境种子实体，支持多种策略。"""
    try:
        body = request.get_json(silent=True) or {}
        strategy = str(body.get("strategy", "random")).strip()
        count = min(int(body.get("count", 10)), 100)
        exclude_ids = body.get("exclude_family_ids") or []
        community_id = body.get("community_id")
        if community_id is not None:
            try:
                community_id = int(community_id)
            except (ValueError, TypeError):
                return err("community_id 必须是整数", 400)

        valid_strategies = ["random", "orphan", "hub", "time_gap", "cross_community", "low_confidence"]
        if strategy not in valid_strategies:
            return err(f"无效策略: {strategy}，可选: {', '.join(valid_strategies)}", 400)

        processor = _get_processor()

        seeds = processor.storage.get_dream_seeds(
            strategy=strategy,
            count=count,
            exclude_ids=exclude_ids,
            community_id=int(community_id) if community_id is not None else None,
        )

        # 格式化返回
        for s in seeds:
            if s.get("event_time"):
                s["event_time"] = str(s["event_time"])
            if s.get("confidence") is not None:
                s["confidence"] = round(float(s["confidence"]), 4)
            if s.get("degree") is not None:
                s["degree"] = int(s["degree"])
            if s.get("community_id") is not None:
                s["community_id"] = int(s["community_id"])

        return ok({"seeds": seeds, "strategy": strategy, "count": len(seeds)})
    except Exception as e:
        return err(str(e), 500)


@dream_bp.route("/api/v1/find/dream/relation", methods=["POST"])
def dream_create_relation():
    """创建梦境发现的关系。"""
    try:
        body = request.get_json(silent=True) or {}
        entity1_id = str(body.get("entity1_id") or "").strip()
        entity2_id = str(body.get("entity2_id") or "").strip()
        content = str(body.get("content") or "").strip()
        confidence = body.get("confidence")
        reasoning = str(body.get("reasoning") or "").strip()
        dream_cycle_id = str(body.get("dream_cycle_id") or "").strip() or None
        episode_id = str(body.get("episode_id") or "").strip() or None

        # 参数校验
        if not entity1_id or not entity2_id:
            return err("entity1_id 与 entity2_id 为必填参数", 400)
        if not content:
            return err("content 为必填参数", 400)
        if not reasoning:
            return err("reasoning 为必填参数，必须说明为什么这两个实体有关联", 400)
        if confidence is None:
            return err("confidence 为必填参数", 400)
        confidence = float(confidence)
        if not (0.0 <= confidence <= 1.0):
            return err("confidence 必须在 0.0-1.0 之间", 400)
        if entity1_id == entity2_id:
            return err("不能创建自环关系", 400)

        processor = _get_processor()
        if not hasattr(processor.storage, 'save_dream_relation'):
            return err("DeepDream 不可用", 404)

        result = processor.storage.save_dream_relation(
            entity1_id=entity1_id,
            entity2_id=entity2_id,
            content=content,
            confidence=confidence,
            reasoning=reasoning,
            dream_cycle_id=dream_cycle_id,
            episode_id=episode_id,
        )
        return ok(result)
    except ValueError as e:
        return err(str(e), 409)
    except Exception as e:
        return err(str(e), 500)


@dream_bp.route("/api/v1/find/dream/episode", methods=["POST"])
def dream_save_episode():
    """保存梦境 episode。"""
    try:
        body = request.get_json(silent=True) or {}
        content = str(body.get("content") or "").strip()
        entities_examined = body.get("entities_examined") or []
        relations_created = body.get("relations_created") or []
        # Accept int count or separate count key (from MCP tools)
        if isinstance(relations_created, int):
            relations_created_count = relations_created
            relations_created = []
        elif body.get("relations_created_count") is not None:
            relations_created_count = int(body["relations_created_count"])
        else:
            relations_created_count = len(relations_created)
        strategy_used = str(body.get("strategy_used") or "").strip()
        dream_cycle_id = str(body.get("dream_cycle_id") or "").strip() or None

        if not content:
            return err("content 为必填参数", 400)

        processor = _get_processor()

        result = processor.storage.save_dream_episode(
            content=content,
            entities_examined=entities_examined,
            relations_created=relations_created,
            strategy_used=strategy_used,
            dream_cycle_id=dream_cycle_id,
            relations_created_count=relations_created_count,
        )
        return ok(result)
    except Exception as e:
        return err(str(e), 500)


@dream_bp.route("/api/v1/find/dream/run", methods=["POST"])
def dream_run():
    """一键梦境巩固：获取种子 → 探索邻居 → 发现隐藏关系 → 返回结果。

    这是 Agent 友好的一次调用接口，替代手动 15-25 次 API 调用。

    参数：
      - strategy（可选）：种子策略，默认 "random"。可选: random, orphan, hub, time_gap, cross_community, low_confidence
      - seed_count（可选）：种子数量，默认 3
      - max_depth（可选）：BFS 遍历深度，默认 2
      - max_relations（可选）：本轮最多创建关系数，默认 5
      - min_confidence（可选）：最低置信度阈值，默认 0.5
      - exclude_ids（可选）：排除的 family_id 列表
      - llm_concurrency（可选）：LLM 并发数，默认 3
    """
    try:
        body = request.get_json(silent=True) or {}

        processor = _get_processor()

        from processor.dream import DreamOrchestrator, DreamConfig, VALID_STRATEGIES

        strategy = str(body.get("strategy", "random")).strip()
        if strategy not in VALID_STRATEGIES:
            return err(f"无效策略: {strategy}，可选: {', '.join(VALID_STRATEGIES)}", 400)

        config = DreamConfig(
            strategy=strategy,
            seed_count=int(body.get("seed_count", 3)),
            max_depth=int(body.get("max_depth", 2)),
            max_relations=int(body.get("max_relations", 5)),
            min_confidence=float(body.get("min_confidence", 0.5)),
            exclude_ids=body.get("exclude_ids") or body.get("exclude_family_ids") or [],
            llm_concurrency=int(body.get("llm_concurrency", 3)),
        )

        orchestrator = DreamOrchestrator(processor.storage, processor.llm_client, config)
        result = orchestrator.run()

        return ok({
            "cycle_id": result.cycle_id,
            "strategy": result.strategy,
            "seeds": result.seeds,
            "explored": result.explored,
            "relations_created": result.relations_created,
            "stats": result.stats,
            "cycle_summary": result.cycle_summary,
        })
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Phase F: Agent-First API — 元查询 / 解释 / 建议
# =========================================================

@dream_bp.route("/api/v1/find/ask", methods=["POST"])
def agent_ask():
    """Agent 元查询：自然语言问题 → 结构化查询 + 回答。"""
    try:
        body = request.get_json(silent=True) or {}
        question = (body.get("question") or "").strip()
        if not question:
            return err("question 为必填", 400)

        processor = _get_processor()
        from server.blueprints.helpers import run_async
        result = run_async(
            processor.llm_client.agent_meta_query(question, request.graph_id or "default")
        )

        # 根据 query_plan 执行实际搜索
        intent = result.get("query_plan", {})
        query_type = intent.get("query_type", "hybrid")
        query_text = intent.get("query_text", question)

        entities = []
        relations = []
        entity_score_map: Dict[str, float] = {}
        relation_score_map: Dict[str, float] = {}

        if query_type == "traverse":
            entity_name = intent.get("entity_name", "")
            if entity_name:
                from processor.search.graph_traversal import GraphTraversalSearcher
                seed_entities = processor.storage.search_entities_by_bm25(entity_name, limit=3)
                seed_ids = [e.family_id for e in seed_entities]
                if seed_ids:
                    searcher = GraphTraversalSearcher(processor.storage)
                    entities = searcher.bfs_expand(seed_ids, max_depth=2, max_nodes=20)
        else:
            # Compute query embedding for hybrid search (vector + BM25)
            from processor.search.hybrid import HybridSearcher
            query_embedding = None
            try:
                ec = getattr(processor.storage, 'embedding_client', None)
                if ec and getattr(ec, 'is_available', lambda: False)():
                    query_embedding = ec.encode([query_text])[0]
            except Exception as _emb_err:
                logger.warning("search_graph embedding 失败: %s", _emb_err)
            searcher = HybridSearcher(processor.storage)
            entity_hits = searcher.search_entities(query_text=query_text, query_embedding=query_embedding, top_k=20)
            relation_hits = searcher.search_relations(query_text=query_text, query_embedding=query_embedding, top_k=10)
            # Preserve scores from hybrid search
            entity_score_map = {e.absolute_id: score for e, score in entity_hits}
            relation_score_map = {r.absolute_id: score for r, score in relation_hits}
            entities = [e for e, _ in entity_hits]
            relations = [r for r, _ in relation_hits]

        entity_dicts = [entity_to_dict(e, _score=entity_score_map.get(e.absolute_id)) for e in entities]
        relation_dicts = [relation_to_dict(r, _score=relation_score_map.get(r.absolute_id)) for r in relations]
        enrich_entity_version_counts(entity_dicts, processor.storage)
        enrich_relation_version_counts(relation_dicts, processor.storage)
        result["results"] = {
            "entities": entity_dicts,
            "relations": relation_dicts,
        }

        # 用 LLM 综合搜索结果生成自然语言回答
        try:
            answer = processor.llm_client.synthesize_answer(question, entity_dicts, relation_dicts)
            result["answer"] = answer
        except Exception as _synth_err:
            logger.warning("synthesize_answer 失败: %s", _synth_err)

        return ok(result)
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# SSE streaming endpoints
# =========================================================

@dream_bp.route("/api/v1/find/ask/stream", methods=["POST"])
def agent_ask_stream():
    """SSE streaming endpoint for Ask Agent."""
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return err("question 为必填", 400)

    q: queue.Queue = queue.Queue()
    _STREAM_SENTINEL = object()

    try:
        processor = _get_processor()
        _graph_id = request.graph_id or "default"

        def _run():
            from server.sse import sse_event
            from server.blueprints.helpers import run_async
            try:
                result = run_async(
                    processor.llm_client.agent_meta_query(question, _graph_id)
                )

                intent = result.get("query_plan", {})
                query_type = intent.get("query_type", "hybrid")
                query_text = intent.get("query_text", question)

                q.put(sse_event("thought", {
                    "text": result.get("thought", ""),
                    "query_plan": intent,
                }))

                # Execute search
                q.put(sse_event("tool_call", {
                    "tool": "search",
                    "arguments": {"query_text": query_text, "type": query_type},
                }))

                entities = []
                relations = []
                entity_score_map: Dict[str, float] = {}
                relation_score_map: Dict[str, float] = {}

                if query_type == "traverse":
                    entity_name = intent.get("entity_name", "")
                    if entity_name:
                        from processor.search.graph_traversal import GraphTraversalSearcher
                        seed_entities = processor.storage.search_entities_by_bm25(entity_name, limit=3)
                        seed_ids = [e.family_id for e in seed_entities]
                        if seed_ids:
                            searcher = GraphTraversalSearcher(processor.storage)
                            entities = searcher.bfs_expand(seed_ids, max_depth=2, max_nodes=20)
                else:
                    # Compute query embedding for hybrid search (vector + BM25)
                    from processor.search.hybrid import HybridSearcher
                    query_embedding = None
                    try:
                        ec = getattr(processor.storage, 'embedding_client', None)
                        if ec and getattr(ec, 'is_available', lambda: False)():
                            query_embedding = ec.encode([query_text])[0]
                    except Exception as _emb_err:
                        logger.warning("stream search embedding 失败: %s", _emb_err)
                    searcher = HybridSearcher(processor.storage)
                    entity_hits = searcher.search_entities(query_text=query_text, query_embedding=query_embedding, top_k=20)
                    relation_hits = searcher.search_relations(query_text=query_text, query_embedding=query_embedding, top_k=10)
                    # Preserve scores from hybrid search
                    entity_score_map = {e.absolute_id: score for e, score in entity_hits}
                    relation_score_map = {r.absolute_id: score for r, score in relation_hits}
                    entities = [e for e, _ in entity_hits]
                    relations = [r for r, _ in relation_hits]

                q.put(sse_event("tool_result", {
                    "tool": "search",
                    "success": True,
                    "data": {
                        "entity_count": len(entities),
                        "relation_count": len(relations),
                    },
                }))

                # Generate summary answer using LLM synthesis
                entity_dicts = [entity_to_dict(e, _score=entity_score_map.get(e.absolute_id)) for e in entities]
                relation_dicts = [relation_to_dict(r, _score=relation_score_map.get(r.absolute_id)) for r in relations]
                enrich_entity_version_counts(entity_dicts, processor.storage)
                enrich_relation_version_counts(relation_dicts, processor.storage)
                result["results"] = {
                    "entities": entity_dicts,
                    "relations": relation_dicts,
                }

                answer = result.get("answer", "")
                if not answer:
                    try:
                        answer = processor.llm_client.synthesize_answer(question, entity_dicts, relation_dicts)
                    except Exception as _synth_err:
                        logger.warning("stream synthesize_answer 失败: %s", _synth_err)
                        # 回退：简单拼接
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
                        answer = "\n".join(parts)

                q.put(sse_event("summary", {
                    "answer": answer,
                    "query_plan": intent,
                    "results": {
                        "entity_count": len(entities),
                        "relation_count": len(relations),
                    },
                }))

            except Exception as e:
                logger.error("stream search error: %s", e, exc_info=True)
                q.put(sse_event("error", {"message": str(e)}))
            finally:
                q.put(sse_event("done", {"status": "completed"}))
                q.put(_STREAM_SENTINEL)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    except Exception as e:
        return err(str(e), 500)

    return sse_response(queue_to_generator(q, sentinel=_STREAM_SENTINEL))


@dream_bp.route("/api/v1/find/explain", methods=["POST"])
def explain_entity():
    """自然语言解释实体。"""
    try:
        body = request.get_json(silent=True) or {}
        family_id = (body.get("family_id") or "").strip()
        aspect = (body.get("aspect") or "summary").strip()
        if not family_id:
            return err("family_id 为必填", 400)

        processor = _get_processor()
        entity = processor.storage.get_entity_by_family_id(family_id)
        if entity is None:
            return err(f"未找到实体: {family_id}", 404)

        from server.blueprints.helpers import run_async
        explanation = run_async(
            processor.llm_client.explain_entity(entity, aspect)
        )

        return ok({"family_id": family_id, "aspect": aspect, "explanation": explanation})
    except Exception as e:
        return err(str(e), 500)


@dream_bp.route("/api/v1/find/suggestions", methods=["GET"])
def get_suggestions():
    """智能建议。"""
    try:
        processor = _get_processor()
        entities = processor.storage.get_all_entities(limit=30, exclude_embedding=True)
        entity_count = processor.storage.count_unique_entities()
        relation_count = processor.storage.count_unique_relations()

        from server.blueprints.helpers import run_async
        suggestions = run_async(
            processor.llm_client.generate_suggestions(entities, entity_count, relation_count)
        )

        return ok(suggestions)
    except Exception as e:
        return err(str(e), 500)


@dream_bp.route("/api/v1/find/quality-report", methods=["GET"])
def quality_report():
    """数据质量报告。"""
    try:
        processor = _get_processor()
        stats = processor.storage.get_data_quality_report()
        return ok(stats)
    except Exception as e:
        return err(str(e), 500)


@dream_bp.route("/api/v1/find/cleanup/invalidated-versions", methods=["POST"])
def cleanup_invalidated_versions():
    """清理已失效的旧版本节点。"""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        before_date = body.get("before_date")
        dry_run = body.get("dry_run", False)
        result = processor.storage.cleanup_invalidated_versions(
            before_date=before_date, dry_run=dry_run,
        )
        return ok(result)
    except Exception as e:
        return err(str(e), 500)


@dream_bp.route("/api/v1/find/maintenance/health", methods=["GET"])
def maintenance_health():
    """数据健康度报告：孤立实体数/失效版本数/质量统计。"""
    try:
        processor = _get_processor()
        stats = processor.storage.get_graph_statistics()
        quality = processor.storage.get_data_quality_report()
        isolated_count = 0
        if hasattr(processor.storage, 'count_isolated_entities'):
            isolated_count = processor.storage.count_isolated_entities()
        return ok({
            "graph_id": request.graph_id,
            "statistics": stats,
            "quality": quality,
            "isolated_entity_count": isolated_count,
        })
    except Exception as e:
        return err(str(e), 500)


@dream_bp.route("/api/v1/find/maintenance/cleanup", methods=["POST"])
def maintenance_cleanup():
    """一键清理：失效版本 + 孤立实体。"""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        dry_run = body.get("dry_run", False)
        results = {}
        # 清理失效版本
        if hasattr(processor.storage, 'cleanup_invalidated_versions'):
            results["invalidated_versions"] = processor.storage.cleanup_invalidated_versions(
                dry_run=dry_run,
            )
        # 清理孤立实体
        isolated = processor.storage.get_isolated_entities(limit=10000)
        if isolated:
            family_ids = list({e.family_id for e in isolated if e.family_id})
            if dry_run:
                results["isolated_entities"] = {
                    "message": f"预览：将删除 {len(family_ids)} 个孤立实体",
                    "family_ids": family_ids,
                    "dry_run": True,
                }
            else:
                deleted = processor.storage.batch_delete_entities(family_ids)
                results["isolated_entities"] = {
                    "message": f"已删除 {len(family_ids)} 个孤立实体（{deleted} 个版本）",
                    "deleted_families": len(family_ids),
                    "deleted_versions": deleted,
                }
        else:
            results["isolated_entities"] = {"message": "没有孤立实体", "deleted": 0}
        return ok(results)
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Butler: 管家式管理 — 一键健康分析 + 维护操作
# =========================================================

@dream_bp.route("/api/v1/butler/report", methods=["GET"])
def butler_report():
    """管家报告：一次调用获取完整图谱健康状况、推荐操作。

    返回:
      health: { graph_id, total_entities, total_relations, total_episodes,
                storage_backend, embedding_available }
      quality: { valid_entities, invalidated_entities, isolated_entities,
                 valid_relations, invalidated_relations }
      dream: { status, last_cycle_id, last_cycle_time }
      recommendations: [ { action, priority, description, estimated_impact } ]
    """
    try:
        processor = _get_processor()
        storage = processor.storage

        # 1. 基本统计
        graph_stats = storage.get_graph_statistics()
        embedding_available = (
            processor.embedding_client is not None
            and processor.embedding_client.is_available()
        )
        storage_backend = "neo4j" if hasattr(storage, 'is_neo4j') else "sqlite"

        health = {
            "graph_id": request.graph_id,
            "total_entities": graph_stats.get("entity_count", 0),
            "total_relations": graph_stats.get("relation_count", 0),
            "total_episodes": graph_stats.get("episode_count", 0),
            "total_communities": graph_stats.get("community_count", 0),
            "storage_backend": storage_backend,
            "embedding_available": embedding_available,
        }

        # 2. 数据质量
        quality = {"valid_entities": 0, "invalidated_entities": 0, "isolated_entities": 0,
                    "valid_relations": 0, "invalidated_relations": 0}
        quality = storage.get_data_quality_report()
        if hasattr(storage, 'count_isolated_entities'):
            quality["isolated_entities"] = storage.count_isolated_entities()

        # 3. 梦境状态
        dream_status_data = {"status": "not_available", "last_cycle_id": None, "last_cycle_time": None}
        logs = storage.list_dream_logs(request.graph_id or "default", limit=1)
        if logs:
            last = logs[0]
            dream_status_data = {
                "status": last.get("status", "completed"),
                "last_cycle_id": last.get("cycle_id"),
                "last_cycle_time": last.get("started_at") or last.get("created_at"),
                "entities_explored": last.get("entities_explored", 0),
                "relations_created": last.get("relations_created", 0),
            }
        else:
            dream_status_data["status"] = "no_cycles"

        # 4. 推荐操作
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
        if health["total_communities"] == 0 and total_ent > 20:
            recommendations.append({
                "action": "detect_communities",
                "priority": "low",
                "description": f"图谱有 {total_ent} 个实体但未做社区检测，建议运行以发现主题聚类",
                "estimated_impact": "识别知识领域边界，辅助梦境探索策略",
            })

        # 实体摘要进化建议
        no_summary = 0
        sample = storage.get_all_entities(limit=50, exclude_embedding=True)
        for e in sample:
            if not getattr(e, 'summary', None):
                no_summary += 1
        if no_summary > len(sample) * 0.5:
            recommendations.append({
                "action": "evolve_summaries",
                "priority": "low",
                "description": f"抽样显示 {no_summary}/{len(sample)} 个实体缺少摘要",
                "estimated_impact": "提升语义检索质量",
            })

        recommendations.sort(key=lambda r: {"high": 0, "medium": 1, "low": 2}.get(r["priority"], 3))

        return ok({
            "health": health,
            "quality": quality,
            "dream": dream_status_data,
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
        })
    except Exception as e:
        return err(str(e), 500)


@dream_bp.route("/api/v1/butler/execute", methods=["POST"])
def butler_execute():
    """管家执行：一键执行推荐操作。

    请求体:
      actions: list[str] — 要执行的操作列表
        可选值: "cleanup_isolated", "cleanup_invalidated", "detect_communities",
                "evolve_summaries"
      dry_run: bool — 仅预览不实际执行（默认 false）
    """
    try:
        body = request.get_json(silent=True) or {}
        actions = body.get("actions", [])
        dry_run = body.get("dry_run", False)
        if not isinstance(actions, list) or not actions:
            return err("actions 需为非空数组", 400)

        processor = _get_processor()
        storage = processor.storage
        results = {}

        for action in actions:
            if action == "cleanup_isolated":
                isolated = storage.get_isolated_entities(limit=10000)
                family_ids = list({e.family_id for e in isolated if e.family_id})
                if dry_run:
                    results[action] = {"status": "preview", "count": len(family_ids), "family_ids": family_ids[:20]}
                else:
                    deleted = storage.batch_delete_entities(family_ids)
                    results[action] = {"status": "done", "deleted_families": len(family_ids), "deleted_versions": deleted}

            elif action == "cleanup_invalidated":
                if hasattr(storage, 'cleanup_invalidated_versions'):
                    results[action] = storage.cleanup_invalidated_versions(dry_run=dry_run)
                else:
                    results[action] = {"status": "skipped", "reason": "当前存储后端不支持"}

            elif action == "detect_communities":
                if hasattr(storage, 'detect_communities'):
                    results[action] = storage.detect_communities()
                else:
                    results[action] = {"status": "skipped", "reason": "需要 Neo4j 后端"}

            elif action == "evolve_summaries":
                # 仅对缺少 summary 的实体执行进化
                evolved = 0
                failed = 0
                sample = storage.get_all_entities(limit=20, exclude_embedding=True)
                for e in sample:
                    if not getattr(e, 'summary', None):
                        try:
                            from server.blueprints.helpers import run_async
                            summary = run_async(
                                processor.llm_client.evolve_entity_summary(e)
                            )
                            storage.update_entity_summary(e.family_id, summary)
                            evolved += 1
                        except Exception as ex:
                            logger.warning("evolve_entity_summary %s 失败: %s", e.family_id, ex)
                            failed += 1
                results[action] = {"status": "done", "evolved": evolved, "failed": failed, "dry_run": dry_run}

            else:
                results[action] = {"status": "unknown", "reason": f"未知操作: {action}"}

        return ok({"actions": results, "dry_run": dry_run})
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Dream Candidate Layer — review, promote, demote candidates
# =========================================================
@dream_bp.route("/api/v1/dream/candidates", methods=["GET"])
def list_dream_candidates():
    """列出 Dream 候选层关系。"""
    try:
        processor = _get_processor()
        limit = request.args.get("limit", type=int, default=50)
        offset = request.args.get("offset", type=int, default=0)
        status = request.args.get("status")  # hypothesized | verified | rejected
        relations = processor.storage.get_candidate_relations(
            limit=limit, offset=offset, status=status)
        total = processor.storage.count_candidate_relations(status=status)
        dicts = [relation_to_dict(r) for r in relations]
        enrich_relations(dicts, processor)
        enrich_relation_version_counts(dicts, processor.storage)
        return ok({
            "relations": dicts,
            "total": total,
            "offset": offset,
            "limit": limit,
        })
    except Exception as e:
        return err(str(e), 500)


@dream_bp.route("/api/v1/dream/candidates/<family_id>/promote", methods=["POST"])
def promote_dream_candidate(family_id: str):
    """将候选关系提升为已验证状态。"""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        evidence_source = body.get("evidence_source", "manual")
        new_confidence = body.get("confidence")
        if new_confidence is not None:
            new_confidence = float(new_confidence)
        result = processor.storage.promote_candidate_relation(
            family_id, evidence_source=evidence_source, new_confidence=new_confidence)
        return ok(result)
    except ValueError as ve:
        return err(str(ve), 404)
    except Exception as e:
        return err(str(e), 500)


@dream_bp.route("/api/v1/dream/candidates/<family_id>/demote", methods=["POST"])
def demote_dream_candidate(family_id: str):
    """将候选关系降级为已拒绝状态。"""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        reason = body.get("reason", "")
        result = processor.storage.demote_candidate_relation(family_id, reason=reason)
        return ok(result)
    except ValueError as ve:
        return err(str(ve), 404)
    except Exception as e:
        return err(str(e), 500)


@dream_bp.route("/api/v1/dream/candidates/corroborate", methods=["POST"])
def corroborate_dream_candidate():
    """对 Dream 候选关系进行佐证检查。"""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        entity1_family_id = body.get("entity1_family_id", "")
        entity2_family_id = body.get("entity2_family_id", "")
        if not entity1_family_id or not entity2_family_id:
            return err("entity1_family_id 和 entity2_family_id 为必填", 400)
        result = processor.storage.corroborate_dream_relation(
            entity1_family_id, entity2_family_id)
        return ok(result)
    except Exception as e:
        return err(str(e), 500)
