"""
System blueprint — Health checks, system monitoring, stats, and route index.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from flask import Blueprint, current_app, request

from processor.llm.client import LLM_PRIORITY_STEP6

logger = logging.getLogger(__name__)

system_bp = Blueprint("system", __name__)


def _get_system_monitor():
    """Get the SystemMonitor from app config."""
    return current_app.config.get("system_monitor")


def _get_processor():
    """Get processor for current graph_id."""
    return current_app.config["registry"].get_processor(request.graph_id)


def _call_llm_with_backoff(
    processor,
    prompt: str,
    timeout: int = 60,
    max_waits: int = 5,
    backoff_base_seconds: int = 3,
) -> str:
    """
    调用 LLM（指数退避重试）。
    等待序列：3, 9, 27, 81, 243 秒（最多等待 max_waits 次）。
    """
    last_error: Optional[str] = None
    max_attempts = max_waits + 1
    for attempt in range(1, max_attempts + 1):
        try:
            response = processor.llm_client._call_llm(
                prompt,
                max_retries=0,
                timeout=timeout,
                allow_mock_fallback=False,
            )
            if response is not None and isinstance(response, str) and len(response.strip()) > 0:
                return response
            last_error = "大模型未返回有效结果"
        except Exception as e:
            last_error = str(e)

        if attempt <= max_waits:
            wait_seconds = backoff_base_seconds ** attempt
            print(f"[LLM] 访问失败，第 {attempt} 次重试前等待 {wait_seconds}s；错误: {last_error}")
            time.sleep(wait_seconds)

    raise RuntimeError(f"重试 {max_attempts} 次仍失败: {last_error or '未知错误'}")


def _check_llm_available(processor) -> tuple[bool, str | None]:
    """启动前握手：检查上游 LLM；若启用 alignment 专用通道，再按步骤 6/7 优先级检查对齐端点。"""
    try:
        _ = _call_llm_with_backoff(
            processor,
            "请只回复一个词：OK",
            timeout=60,
        )
        lc = processor.llm_client
        if getattr(lc, "alignment_enabled", False):
            _old_pri = getattr(lc._priority_local, "priority", None)
            lc._priority_local.priority = LLM_PRIORITY_STEP6
            try:
                _ = _call_llm_with_backoff(
                    processor,
                    "请只回复一个词：OK",
                    timeout=60,
                )
            finally:
                if _old_pri is not None:
                    lc._priority_local.priority = _old_pri
                else:
                    try:
                        del lc._priority_local.priority
                    except AttributeError:
                        pass
        return True, None
    except Exception as e:
        return False, f"大模型不可用: {e}"


# ── Shared response helpers (imported from helpers) ─────────────────────

def ok(data: Any) -> tuple:
    from server.blueprints.helpers import ok as _ok
    return _ok(data)


def err(message: str, status: int = 400) -> tuple:
    from server.blueprints.helpers import err as _err
    return _err(message, status)


# ── Route Index ─────────────────────────────────────────────────────────

@system_bp.route("/api/v1/routes", methods=["GET"])
def api_routes():
    """返回推荐接口索引，帮助客户端快速理解推荐路径、方法与参数。"""
    return ok({
        "health": [
            {
                "path": "/api/v1/health",
                "methods": ["GET"],
                "summary": "服务健康检查",
                "query": {"graph_id": "string，必填"},
                "aliases": ["/health"],
            },
            {
                "path": "/api/v1/health/llm",
                "methods": ["GET"],
                "summary": "LLM 连通性检查",
                "query": {"graph_id": "string，必填"},
            },
        ],
        "remember": [
            {
                "path": "/api/v1/remember",
                "methods": ["POST"],
                "summary": "提交异步记忆写入任务",
                "body": {
                    "graph_id": "string，必填，目标图谱 ID",
                    "text": "string，或通过 file 上传（二选一必填）",
                    "file": "multipart 文件上传",
                    "file_path": "string，服务端本地文件路径（可选）",
                    "source_name": "string，可选",
                    "doc_name": "string，可选，兼容旧字段（内部映射为 source_document）",
                    "source_document": "string，可选，新字段（优先于 doc_name）",
                    "load_cache_memory": "bool，可选",
                    "event_time": "ISO 8601 string，可选",
                },
            },
            {
                "path": "/api/v1/remember/tasks/<task_id>",
                "methods": ["GET", "DELETE"],
                "summary": "查询或删除 remember 任务",
            },
            {
                "path": "/api/v1/remember/tasks/<task_id>/pause",
                "methods": ["POST"],
                "summary": "暂停运行中的 remember 任务",
            },
            {
                "path": "/api/v1/remember/tasks/<task_id>/resume",
                "methods": ["POST"],
                "summary": "继续已暂停的 remember 任务",
            },
            {
                "path": "/api/v1/remember/tasks",
                "methods": ["GET"],
                "summary": "查看 remember 任务队列",
                "query": {"limit": "int，可选，默认 50"},
            },
            {
                "path": "/api/v1/remember/monitor",
                "methods": ["GET"],
                "summary": "获取 remember 实时监控快照",
                "query": {"limit": "int，可选，默认 6"},
            },
        ],
        "find": [
            {
                "path": "/api/v1/find",
                "methods": ["POST"],
                "summary": "统一语义检索入口",
                "body": {
                    "graph_id": "string，必填，目标图谱 ID",
                    "query": "string，必填",
                    "similarity_threshold": "float，可选，默认 0.5",
                    "max_entities": "int，可选，默认 20",
                    "max_relations": "int，可选，默认 50",
                    "expand": "bool，可选，默认 true",
                    "time_before": "ISO 8601 string，可选",
                    "time_after": "ISO 8601 string，可选",
                },
            },
            {
                "path": "/api/v1/find/candidates",
                "methods": ["POST"],
                "summary": "一次性按条件返回候选实体与关系",
            },
            {
                "path": "/api/v1/find/entities/search",
                "methods": ["GET", "POST"],
                "summary": "按文本搜索实体",
                "body_or_query": {
                    "query_name": "string，必填",
                    "query_content": "string，可选",
                    "similarity_threshold": "float，可选",
                    "max_results": "int，可选",
                    "text_mode": "name_only | content_only | name_and_content",
                    "similarity_method": "embedding | text | jaccard | bleu",
                },
            },
            {
                "path": "/api/v1/find/relations/search",
                "methods": ["GET", "POST"],
                "summary": "按文本搜索关系",
                "body_or_query": {
                    "query_text": "string，必填",
                    "similarity_threshold": "float，可选",
                    "max_results": "int，可选",
                },
            },
        ],
        "entity": [
            {
                "path": "/api/v1/find/entities",
                "methods": ["GET"],
                "summary": "列出实体",
                "query": {"limit": "int，可选"},
            },
            {
                "path": "/api/v1/find/entities/as-of-time",
                "methods": ["GET"],
                "summary": "列出每个实体在指定时间点的最新版本",
                "query": {
                    "time_point": "ISO 8601 string，必填",
                    "limit": "int，可选",
                },
            },
            {
                "path": "/api/v1/find/entities/absolute/<absolute_id>",
                "methods": ["GET"],
                "summary": "按实体 absolute_id 读取单个实体版本",
            },
            {
                "path": "/api/v1/find/entities/<family_id>/as-of-time",
                "methods": ["GET"],
                "summary": "返回该实体在指定时间点的最近过去版本",
                "query": {"time_point": "ISO 8601 string，必填"},
            },
            {
                "path": "/api/v1/find/entities/<family_id>/nearest-to-time",
                "methods": ["GET"],
                "summary": "返回该实体距离指定时间点最近的版本",
                "query": {
                    "time_point": "ISO 8601 string，必填",
                    "max_delta_seconds": "float，可选",
                },
            },
            {
                "path": "/api/v1/find/entities/<family_id>/around-time",
                "methods": ["GET"],
                "summary": "返回该实体在指定时间点附近窗口内的所有版本",
                "query": {
                    "time_point": "ISO 8601 string，必填",
                    "within_seconds": "float，必填",
                },
            },
            {
                "path": "/api/v1/find/entities/<family_id>/relations",
                "methods": ["GET"],
                "summary": "按实体业务 ID 查询相关关系",
            },
        ],
        "relation": [
            {
                "path": "/api/v1/find/relations",
                "methods": ["GET"],
                "summary": "列出关系",
                "query": {
                    "limit": "int，可选",
                    "offset": "int，可选，默认 0",
                },
            },
            {
                "path": "/api/v1/find/relations/absolute/<absolute_id>",
                "methods": ["GET"],
                "summary": "按关系 absolute_id 读取单条关系版本",
            },
            {
                "path": "/api/v1/find/relations/between",
                "methods": ["GET", "POST"],
                "summary": "查询两个实体之间的关系",
                "body_or_query": {
                    "family_id_a": "string，必填",
                    "family_id_b": "string，必填",
                },
            },
            {
                "path": "/api/v1/find/paths/shortest",
                "methods": ["GET", "POST"],
                "summary": "查找两个实体之间的最短路径",
                "body_or_query": {
                    "family_id_a": "string，必填",
                    "family_id_b": "string，必填",
                    "max_depth": "int，可选，默认6",
                    "max_paths": "int，可选，默认10",
                },
            },
        ],
        "episode": [
            {
                "path": "/api/v1/find/episodes/latest",
                "methods": ["GET"],
                "summary": "读取最新 Episode",
            },
            {
                "path": "/api/v1/find/episodes/latest/metadata",
                "methods": ["GET"],
                "summary": "读取最新 Episode 元数据",
            },
            {
                "path": "/api/v1/find/episodes/<cache_id>",
                "methods": ["GET"],
                "summary": "按 cache_id 读取 Episode",
            },
        ],
        "system": [
            {
                "path": "/api/v1/system/overview",
                "methods": ["GET"],
                "summary": "系统总览：图谱数量、运行时间、线程数",
            },
            {
                "path": "/api/v1/system/graphs",
                "methods": ["GET"],
                "summary": "所有图谱摘要列表",
            },
            {
                "path": "/api/v1/system/graphs/<graph_id>",
                "methods": ["GET"],
                "summary": "单图谱详细状态",
            },
            {
                "path": "/api/v1/system/tasks",
                "methods": ["GET"],
                "summary": "所有图谱的任务列表",
            },
            {
                "path": "/api/v1/system/logs",
                "methods": ["GET"],
                "summary": "最近系统日志",
            },
            {
                "path": "/api/v1/system/access-stats",
                "methods": ["GET"],
                "summary": "API 访问统计",
            },
        ],
    })


# ── Health ──────────────────────────────────────────────────────────────

@system_bp.route("/api/v1/health", methods=["GET"])
def health():
    """健康检查；推荐使用 /api/v1/health。"""
    try:
        processor = _get_processor()
        embedding_available = (
            processor.embedding_client is not None
            and processor.embedding_client.is_available()
        )
        storage_backend = "neo4j" if hasattr(processor.storage, 'is_neo4j') else "sqlite"
        return ok({
            "graph_id": request.graph_id,
            "storage_backend": storage_backend,
            "storage_path": str(processor.storage.storage_path),
            "embedding_available": embedding_available,
        })
    except Exception as e:
        return err(str(e), 500)


@system_bp.route("/api/v1/health/llm", methods=["GET"])
def health_llm():
    """检查大模型是否可访问。"""
    try:
        processor = _get_processor()
        response = _call_llm_with_backoff(
            processor,
            "请只回复一个词：OK",
            timeout=60,
        )
        return ok({"graph_id": request.graph_id, "llm_available": True, "message": "大模型访问正常", "response_preview": response.strip()[:80]})
    except Exception as e:
        return err(f"大模型不可用: {e}", 503)


# ── Stats ───────────────────────────────────────────────────────────────

@system_bp.route("/api/v1/find/stats", methods=["GET"])
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
