"""
Dream blueprint — Dream exploration, ask/explain/suggestions, quality report,
maintenance, and butler routes.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import Blueprint, current_app, request

from core.server.blueprints.helpers import (
    ok,
    safe_endpoint,
    err,
    _get_processor,
    _get_graph_id,
    entity_to_dict,
    relation_to_dict,
    enrich_relations,
    enrich_entity_version_counts,
    enrich_relation_version_counts,
    parse_time_point,
    run_async,
    get_json_body,
)
from core.server.sse import sse_response, sse_event, queue_to_generator
from core.server.llm_utils import call_llm_with_backoff
from core.find.graph_traversal import GraphTraversalSearcher
from core.find.hybrid import HybridSearcher
from core.dream import DreamOrchestrator, DreamConfig, VALID_STRATEGIES as _DREAM_STRATS

# Imported helpers from the extracted module
from core.server.blueprints.dream_helpers import (
    PRIORITY_ORDER as _PRIORITY_ORDER,
    VALID_STRATEGIES as _VALID_STRATEGIES,
    execute_ask_search as _execute_ask_search,
    serialize_ask_results as _serialize_ask_results,
    call_llm as _call_llm_with_backoff,
    build_butler_recommendations,
    build_health_data,
    build_dream_status_from_logs,
    compute_sample_summary_stats,
    build_fallback_answer,
    run_butler_action,
    evolve_summaries as _evolve_summaries,
    format_seeds,
)

logger = logging.getLogger(__name__)

dream_bp = Blueprint("dream", __name__)

# Shared pool for dream endpoint parallel queries (avoids per-request thread creation)
_dream_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dream")


# =========================================================
# Phase E: DeepDream 记忆巩固（积木端点 — 编排由 Agent Skill 驱动）
# =========================================================

@dream_bp.route("/api/v1/find/dream/status", methods=["GET"])
def dream_status():
    """查询梦境状态（最近一次）+ 当前是否正在执行。"""
    try:
        graph_id = request.graph_id or "default"
        processor = _get_processor()

        # Check in-memory lock state
        currently_running = False
        registry = current_app.config.get("registry")
        if registry is not None:
            lock = registry.get_dream_lock(graph_id)
            if lock is not None and lock.locked():
                currently_running = True

        logs = processor.storage.list_dream_logs(graph_id, limit=1)
        if logs:
            logs[0]["currently_running"] = currently_running
            return ok(logs[0])
        return ok({"status": "no_cycles", "currently_running": currently_running})
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
        body = get_json_body()
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
        if strategy not in _VALID_STRATEGIES:
            return err(f"无效策略: {strategy}，可选: {', '.join(valid_strategies)}", 400)

        processor = _get_processor()

        seeds = processor.storage.get_dream_seeds(
            strategy=strategy,
            count=count,
            exclude_ids=exclude_ids,
            community_id=int(community_id) if community_id is not None else None,
        )

        format_seeds(seeds)

        return ok({"seeds": seeds, "strategy": strategy, "count": len(seeds)})
    except Exception as e:
        return err(str(e), 500)


@dream_bp.route("/api/v1/find/dream/relation", methods=["POST"])
def dream_create_relation():
    """创建梦境发现的关系。"""
    try:
        body = get_json_body()
        entity1_id = (body.get("entity1_id") or "").strip()
        entity2_id = (body.get("entity2_id") or "").strip()
        content = (body.get("content") or "").strip()
        confidence = body.get("confidence")
        reasoning = (body.get("reasoning") or "").strip()
        dream_cycle_id = (body.get("dream_cycle_id") or "").strip() or None
        episode_id = (body.get("episode_id") or "").strip() or None

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
        body = get_json_body()
        content = (body.get("content") or "").strip()
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
        strategy_used = (body.get("strategy_used") or "").strip()
        dream_cycle_id = (body.get("dream_cycle_id") or "").strip() or None

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
      - min_pair_similarity（可选）：配对语义相似度阈值，默认 0.0（不过滤）。
        当 > 0 时，在 LLM 判断前用 embedding 余弦相似度预过滤低相关配对，减少 LLM 调用
      - auto_rotate（可选）：是否自动轮换策略，默认 False。启用后忽略 strategy 参数，
        由 DreamHistory 根据跨周期效果自动选择下一个策略
    """
    try:
        body = get_json_body()

        processor = _get_processor()
        graph_id = request.graph_id or "default"

        auto_rotate = bool(body.get("auto_rotate", False))

        strategy = str(body.get("strategy", "random")).strip()
        if strategy not in _DREAM_STRATS:
            return err(f"无效策略: {strategy}，可选: {', '.join(_DREAM_STRATS)}", 400)

        config = DreamConfig(
            strategy=strategy,
            seed_count=int(body.get("seed_count", 3)),
            max_depth=int(body.get("max_depth", 2)),
            max_relations=int(body.get("max_relations", 5)),
            min_confidence=float(body.get("min_confidence", 0.5)),
            exclude_ids=body.get("exclude_ids") or body.get("exclude_family_ids") or [],
            llm_concurrency=int(body.get("llm_concurrency", 3)),
            min_pair_similarity=float(body.get("min_pair_similarity", 0.0)),
            discovery_mode=bool(body.get("discovery_mode", False)),
        )

        # Use persistent orchestrator from registry (preserves cross-cycle LRU history)
        registry = current_app.config.get("registry")
        if registry is not None:
            orchestrator = registry.get_dream_orchestrator(graph_id, config)
            dream_lock = registry.get_dream_lock(graph_id)
        else:
            # Fallback: no registry (e.g. testing)
            orchestrator = DreamOrchestrator(processor.storage, processor.llm_client, config)
            dream_lock = None

        def _run_dream():
            return orchestrator.run(auto_rotate=auto_rotate)

        if dream_lock is not None:
            if not dream_lock.acquire(timeout=5):
                return err("梦境周期正在执行中，请稍后再试", 429,
                           hint="A dream cycle is currently running. Check GET /find/dream/status for progress.")
            try:
                result = _run_dream()
            finally:
                dream_lock.release()
        else:
            result = _run_dream()

        resp = {
            "cycle_id": result.cycle_id,
            "strategy": result.strategy,
            "seeds": result.seeds,
            "explored": result.explored,
            "relations_created": result.relations_created,
            "stats": result.stats,
            "strategy_stats": orchestrator._history.get_strategy_stats(),
            "cycle_summary": result.cycle_summary,
        }
        if result.warnings:
            resp["warnings"] = result.warnings
        return ok(resp)
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Phase F: Agent-First API — 元查询 / 解释 / 建议
# =========================================================

@dream_bp.route("/api/v1/find/ask", methods=["POST"])
def agent_ask():
    """Agent 元查询：自然语言问题 → 结构化查询 + 回答。"""
    try:
        body = get_json_body()
        question = (body.get("question") or "").strip()
        if not question:
            return err("question 为必填", 400)

        processor = _get_processor()
        result = run_async(
            processor.llm_client.agent_meta_query(question, request.graph_id or "default")
        )

        # 根据 query_plan 执行实际搜索
        intent = result.get("query_plan", {})
        query_type = intent.get("query_type", "hybrid")
        query_text = intent.get("query_text", question)

        entities, relations, entity_score_map, relation_score_map = _execute_ask_search(
            processor, query_type, query_text, intent,
        )
        entity_dicts, relation_dicts = _serialize_ask_results(
            entities, relations, entity_score_map, relation_score_map, processor,
        )
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
    body = get_json_body()
    question = (body.get("question") or "").strip()
    if not question:
        return err("question 为必填", 400)

    q: queue.Queue = queue.Queue()
    _STREAM_SENTINEL = object()

    try:
        processor = _get_processor()
        _graph_id = request.graph_id or "default"

        def _run():
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

                # Execute search using shared helper
                entities, relations, entity_score_map, relation_score_map = _execute_ask_search(
                    processor, query_type, query_text, intent,
                )

                q.put(sse_event("tool_result", {
                    "tool": "search",
                    "success": True,
                    "data": {
                        "entity_count": len(entities),
                        "relation_count": len(relations),
                    },
                }))

                # Serialize results using shared helper
                entity_dicts, relation_dicts = _serialize_ask_results(
                    entities, relations, entity_score_map, relation_score_map, processor,
                )
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
                        answer = build_fallback_answer(query_text, entities, relations)

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
        body = get_json_body()
        family_id = (body.get("family_id") or "").strip()
        aspect = (body.get("aspect") or "summary").strip()
        if not family_id:
            return err("family_id 为必填", 400)

        processor = _get_processor()
        entity = processor.storage.get_entity_by_family_id(family_id)
        if entity is None:
            return err(f"未找到实体: {family_id}", 404)

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
        body = get_json_body()
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
        body = get_json_body()
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
@safe_endpoint
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

        # Fetch all independent data in parallel (using shared pool)
        _has_isolated = hasattr(storage, 'count_isolated_entities')
        _f_stats = _dream_pool.submit(storage.get_graph_statistics)
        _f_quality = _dream_pool.submit(storage.get_data_quality_report)
        _f_isolated = _dream_pool.submit(storage.count_isolated_entities) if _has_isolated else None
        _f_logs = _dream_pool.submit(storage.list_dream_logs, request.graph_id or "default", 1)
        graph_stats = _f_stats.result()
        quality = _f_quality.result()
        logs = _f_logs.result()
        if _f_isolated is not None:
            quality["isolated_entities"] = _f_isolated.result()

        embedding_available = (
            processor.embedding_client is not None
            and processor.embedding_client.is_available()
        )

        health = build_health_data(graph_stats, request.graph_id, embedding_available)

        # 3. 梦境状态
        dream_status_data = build_dream_status_from_logs(logs)

        # 4. Sample summary check (for recommendation engine)
        no_summary, sample_total = compute_sample_summary_stats(storage)
        health["_sample_no_summary"] = no_summary
        health["_sample_total"] = sample_total

        # Build recommendations using extracted helper
        recommendations = build_butler_recommendations(health, quality, dream_status_data)

        # Remove internal keys before sending response
        health.pop("_sample_no_summary", None)
        health.pop("_sample_total", None)

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
        body = get_json_body()
        actions = body.get("actions", [])
        dry_run = body.get("dry_run", False)
        if not isinstance(actions, list) or not actions:
            return err("actions 需为非空数组", 400)

        processor = _get_processor()
        storage = processor.storage
        results = {}

        # Partition actions: independent ones can run in parallel
        _independent_actions = {"cleanup_isolated", "cleanup_invalidated", "detect_communities", "fix_dangling_refs", "cleanup_stale_redirects"}
        independent = [a for a in actions if a in _independent_actions]
        sequential = [a for a in actions if a not in _independent_actions]

        # Run independent actions in parallel
        if len(independent) > 1:
            futures = {_dream_pool.submit(run_butler_action, storage, a, dry_run): a for a in independent}
            for f in futures:
                action = futures[f]
                try:
                    results[action] = f.result()
                except Exception as ex:
                    results[action] = {"status": "error", "reason": str(ex)}
        else:
            for action in independent:
                results[action] = run_butler_action(storage, action, dry_run)

        # Run sequential actions (evolve_summaries, unknown, etc.)
        for action in sequential:
            if action == "evolve_summaries":
                results[action] = _evolve_summaries(processor, storage, dry_run, _dream_pool)
            else:
                results[action] = {"status": "unknown", "reason": f"未知操作: {action}"}

        _errors = [a for a, r in results.items() if isinstance(r, dict) and r.get("status") == "error"]
        resp = {"actions": results, "dry_run": dry_run}
        if _errors and not dry_run:
            resp["warnings"] = f"{len(_errors)} action(s) had errors: {', '.join(_errors)}"
        return ok(resp)
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
        body = get_json_body()
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
        body = get_json_body()
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
        body = get_json_body()
        entity1_family_id = body.get("entity1_family_id", "")
        entity2_family_id = body.get("entity2_family_id", "")
        if not entity1_family_id or not entity2_family_id:
            return err("entity1_family_id 和 entity2_family_id 为必填", 400)
        result = processor.storage.corroborate_dream_relation(
            entity1_family_id, entity2_family_id)
        return ok(result)
    except Exception as e:
        return err(str(e), 500)
