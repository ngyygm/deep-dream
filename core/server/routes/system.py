"""
System routes — Health checks, system monitoring, stats, and route index.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from flask import Blueprint, current_app, request

from core.llm.client import LLM_PRIORITY_STEP6
from core.server.llm_utils import call_llm_with_backoff, check_llm_available
from core.server.routes.helpers import ok, err, _get_processor, _get_system_monitor

logger = logging.getLogger(__name__)

system_bp = Blueprint("system", __name__)

# Rate limit for LLM health check (prevent credit burn)
_last_llm_health_time = 0.0
_LLM_HEALTH_MIN_INTERVAL = 30.0  # seconds


# LLM helpers - delegate to shared modules
_call_llm_with_backoff = call_llm_with_backoff
_check_llm_available = lambda processor: check_llm_available(processor, priority_steps=[6])


# ── Route Index ─────────────────────────────────────────────────────────

_API_ROUTE_INDEX = {
    "health": [
        {"path": "/api/v1/health", "methods": ["GET"], "summary": "服务健康检查"},
        {"path": "/api/v1/health/llm", "methods": ["GET"], "summary": "LLM 连通性检查"},
    ],
    "remember": [
        {"path": "/api/v1/remember", "methods": ["POST"], "summary": "写入 Markdown/text 记忆"},
        {"path": "/api/v1/remember/tasks", "methods": ["GET"], "summary": "查看 remember 任务队列"},
    ],
    "documents": [
        {"path": "/api/v1/vaults/index", "methods": ["POST"], "summary": "索引只读 Markdown/Obsidian vault"},
        {"path": "/api/v1/documents", "methods": ["GET"], "summary": "列出文档版本"},
        {"path": "/api/v1/documents/graph", "methods": ["POST"], "summary": "读取文档到 Episode 和 Concept 的可视化子图"},
    ],
    "concepts": [
        {"path": "/api/v1/concepts", "methods": ["GET"], "summary": "列出概念"},
        {"path": "/api/v1/concepts/search", "methods": ["POST"], "summary": "搜索概念"},
        {"path": "/api/v1/concepts/<family_id>", "methods": ["GET"], "summary": "读取概念"},
        {"path": "/api/v1/concepts/<family_id>/versions", "methods": ["GET"], "summary": "读取概念版本"},
        {"path": "/api/v1/concepts/<family_id>/provenance", "methods": ["GET"], "summary": "读取概念溯源"},
        {"path": "/api/v1/traverse", "methods": ["POST"], "summary": "遍历概念图"},
    ],
    "graphs": [
        {"path": "/api/v1/graphs", "methods": ["GET", "POST"], "summary": "列出或创建物理隔离图谱"},
        {"path": "/api/v1/graphs/<graph_id>", "methods": ["GET", "DELETE"], "summary": "读取或删除图谱"},
        {"path": "/api/v1/graphs/<graph_id>/clear", "methods": ["POST"], "summary": "清空图谱"},
    ],
    "system": [
        {"path": "/api/v1/routes", "methods": ["GET"], "summary": "动态路由索引"},
        {"path": "/api/v1/stats/counts", "methods": ["GET"], "summary": "概念计数"},
        {"path": "/api/v1/system/overview", "methods": ["GET"], "summary": "系统总览"},
    ],
}


@system_bp.route("/api/v1/routes", methods=["GET"])
def route_index():
    """返回所有已注册的 API 路由。"""
    routes = []
    for rule in current_app.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        routes.append({
            "path": rule.rule,
            "methods": sorted(rule.methods - {"HEAD", "OPTIONS"}),
        })
    routes.sort(key=lambda r: r["path"])
    return ok({"routes": routes, "count": len(routes)})


@system_bp.route("/api/v1/health", methods=["GET"])
def health():
    """健康检查；推荐使用 /api/v1/health。"""
    try:
        gid = getattr(request, 'graph_id', None) or request.args.get('graph_id', 'default')
        try:
            from core.server.registry import GraphRegistry
            GraphRegistry.validate_graph_id(gid)
        except ValueError as e:
            return err(str(e), 400)
        processor = current_app.config["registry"].get_processor(gid)
        embedding_available = (
            processor.embedding_client is not None
            and processor.embedding_client.is_available()
        )
        storage_backend = "sqlite"
        return ok({
            "graph_id": gid,
            "storage_backend": storage_backend,
            "embedding_available": embedding_available,
        })
    except Exception as e:
        return err(str(e), 500)


@system_bp.route("/api/v1/health/llm", methods=["GET"])
def health_llm():
    """检查大模型是否可访问。"""
    global _last_llm_health_time
    now = time.time()
    gid = getattr(request, 'graph_id', None) or request.args.get('graph_id', 'default')
    if now - _last_llm_health_time < _LLM_HEALTH_MIN_INTERVAL:
        return ok({
            "graph_id": gid,
            "llm_available": True,
            "message": "LLM 健康检查冷却中，请稍后重试",
            "cooldown_remaining": round(_LLM_HEALTH_MIN_INTERVAL - (now - _last_llm_health_time), 1),
        })
    _last_llm_health_time = now
    try:
        cfg = current_app.config.get("config") or {}
        llm_cfg = cfg.get("llm") or {}
        if not llm_cfg.get("api_key") and not llm_cfg.get("base_url"):
            return err("大模型未配置", 503)
        processor = current_app.config["registry"].get_processor(gid)
        response = _call_llm_with_backoff(
            processor,
            "请只回复一个词：OK",
            timeout=60,
        )
        return ok({"graph_id": gid, "llm_available": True, "message": "大模型访问正常", "response_preview": response.strip()[:80]})
    except Exception as e:
        return err(f"大模型不可用: {e}", 503)


# ── Stats ───────────────────────────────────────────────────────────────

@system_bp.route("/api/v1/find/stats", methods=["GET"])
@system_bp.route("/api/v1/graph/stats", methods=["GET"])
def find_stats():
    try:
        processor = _get_processor()
        total_entities = processor.storage.count_unique_entities()
        total_relations = processor.storage.count_unique_relations()

        cache_json_dir = processor.storage.cache_json_dir
        total_episodes = 0
        if hasattr(processor.storage, 'count_episodes'):
            total_episodes = processor.storage.count_episodes()
        else:
            cache_dir = processor.storage.cache_dir
            json_files = list(cache_json_dir.glob("*.json"))
            docs_meta_files = list(processor.storage.docs_dir.glob("*/meta.json")) if processor.storage.docs_dir.is_dir() else []
            if docs_meta_files:
                total_episodes = len(docs_meta_files)
            elif json_files:
                total_episodes = len(json_files)
            else:
                total_episodes = len(list(cache_dir.glob("*.json")))

        total_communities = 0
        if hasattr(processor.storage, 'count_communities'):
            total_communities = processor.storage.count_communities()

        return ok({
            "total_concepts": processor.storage.count_concepts() if hasattr(processor.storage, "count_concepts") else total_entities + total_relations + total_episodes,
            "total_documents": processor.storage.count_concepts("document") if hasattr(processor.storage, "count_concepts") else 0,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "total_episodes": total_episodes,
            "total_communities": total_communities,
        })
    except Exception as e:
        return err(str(e), 500)


@system_bp.route("/api/v1/stats/counts", methods=["GET"])
def stats_counts():
    """快速计数端点（兼容旧路径）。"""
    return find_stats()


# ── System Monitor ──────────────────────────────────────────────────────

@system_bp.route("/api/v1/system/dashboard", methods=["GET"])
def system_dashboard():
    """仪表盘合并端点：一次返回 overview、graphs、tasks、logs、access-stats。"""
    try:
        system_monitor = _get_system_monitor()
        if system_monitor is None:
            return err("SystemMonitor 未启用", 503)
        task_limit = request.args.get("task_limit", 50, type=int)
        log_limit = request.args.get("log_limit", 100, type=int)
        log_level = request.args.get("log_level")
        log_source = request.args.get("log_source")
        access_since = request.args.get("access_since", 300, type=float)
        return ok(system_monitor.dashboard_snapshot(
            task_limit=task_limit, log_limit=log_limit,
            log_level=log_level, log_source=log_source,
            access_since=access_since,
        ))
    except Exception as e:
        return err(str(e), 500)


@system_bp.route("/api/v1/system/overview", methods=["GET"])
def system_overview():
    """系统总览：图谱数量、运行时间、线程数。"""
    try:
        system_monitor = _get_system_monitor()
        if system_monitor is None:
            return err("SystemMonitor 未启用", 503)
        return ok(system_monitor.overview())
    except Exception as e:
        return err(str(e), 500)


@system_bp.route("/api/v1/system/graphs", methods=["GET"])
def system_graphs():
    """所有图谱摘要列表。"""
    try:
        system_monitor = _get_system_monitor()
        if system_monitor is None:
            return err("SystemMonitor 未启用", 503)
        return ok(system_monitor.all_graphs())
    except Exception as e:
        return err(str(e), 500)


@system_bp.route("/api/v1/system/graphs/<graph_id>", methods=["GET"])
def system_graph_detail(graph_id: str):
    """单图谱详细状态（存储+队列+线程）。"""
    try:
        system_monitor = _get_system_monitor()
        if system_monitor is None:
            return err("SystemMonitor 未启用", 503)
        detail = system_monitor.graph_detail(graph_id)
        if detail is None:
            return err(f"图谱不存在: {graph_id}", 404)
        return ok(detail)
    except Exception as e:
        return err(str(e), 500)


@system_bp.route("/api/v1/system/tasks", methods=["GET"])
def system_tasks():
    """所有图谱的任务列表。"""
    try:
        system_monitor = _get_system_monitor()
        if system_monitor is None:
            return err("SystemMonitor 未启用", 503)
        limit = request.args.get("limit", 50, type=int)
        return ok(system_monitor.all_tasks(limit=limit))
    except Exception as e:
        return err(str(e), 500)


@system_bp.route("/api/v1/system/logs", methods=["GET"])
def system_logs():
    """最近系统日志。支持 ?limit=&level=&source= 筛选。"""
    try:
        system_monitor = _get_system_monitor()
        if system_monitor is None:
            return err("SystemMonitor 未启用", 503)
        limit = request.args.get("limit", 50, type=int)
        level = request.args.get("level")
        source = request.args.get("source")
        return ok(system_monitor.recent_logs(limit=limit, level=level, source=source))
    except Exception as e:
        return err(str(e), 500)


@system_bp.route("/api/v1/system/access-stats", methods=["GET"])
def system_access_stats():
    """API 访问统计。支持 ?since_seconds= 指定统计周期（默认 300 秒）。"""
    try:
        system_monitor = _get_system_monitor()
        if system_monitor is None:
            return err("SystemMonitor 未启用", 503)
        since = request.args.get("since_seconds", 300, type=float)
        return ok(system_monitor.access_stats(since_seconds=since))
    except Exception as e:
        return err(str(e), 500)


