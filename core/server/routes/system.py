"""
System routes — Health checks, system monitoring, stats, and route index.
"""
from __future__ import annotations

import logging
import json
import os
import threading
import time
from pathlib import Path

from flask import Blueprint, current_app, request

from core.server.llm_utils import call_llm_with_backoff, check_llm_available
from core.server.routes.helpers import ok, err, _get_processor, _get_system_monitor

logger = logging.getLogger(__name__)

system_bp = Blueprint("system", __name__)

# Rate limit for LLM health check (prevent credit burn)
_last_llm_health_time = 0.0
_last_llm_health_lock = threading.Lock()
_LLM_HEALTH_MIN_INTERVAL = 30.0  # seconds


# LLM helpers - delegate to shared modules
_call_llm_with_backoff = call_llm_with_backoff
_check_llm_available = check_llm_available


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
        gid = getattr(request, 'graph_id', None) or request.args.get('graph_id', 'library')
        try:
            from core.server.registry import GraphRegistry
            gid = GraphRegistry.normalize_graph_id(gid)
        except ValueError as e:
            return err(str(e), 400)
        processor = current_app.config["registry"].get_processor(gid)
        embedding_available = (
            processor.embedding_client is not None
            and processor.embedding_client.is_available()
        )
        storage_backend = "sqlite"
        return ok({
            "library_id": gid,
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
    gid = getattr(request, 'graph_id', None) or request.args.get('graph_id', 'library')
    from core.server.registry import GraphRegistry
    gid = GraphRegistry.normalize_graph_id(gid)
    with _last_llm_health_lock:
        if now - _last_llm_health_time < _LLM_HEALTH_MIN_INTERVAL:
            return ok({
                "library_id": gid,
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
        return ok({"library_id": gid, "llm_available": True, "message": "大模型访问正常", "response_preview": response.strip()[:80]})
    except Exception as e:
        return err(f"大模型不可用: {e}", 503)


# ── Stats ───────────────────────────────────────────────────────────────

@system_bp.route("/api/v1/find/stats", methods=["GET"])
def find_stats():
    try:
        processor = _get_processor()
        storage = processor.storage
        return ok({
            "total_concepts": storage.count_concepts(),
            "total_documents": storage.count_documents(),
            "total_entities": storage.count_unique_entities(),
            "total_relations": storage.count_unique_relations(),
            "total_episodes": storage.count_episodes(),
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


# ── Graph CRUD (Frontend-facing /api/v1/graphs) ─────────────────────────

@system_bp.route("/api/v1/graphs", methods=["GET"])
def list_graphs():
    """Frontend-facing: list graphs."""
    try:
        registry = current_app.config.get("registry")
        graphs = registry.list_graphs_info() if registry else []
        return ok({"graphs": graphs})
    except Exception as e:
        return err(str(e), 500)


@system_bp.route("/api/v1/graphs", methods=["POST"])
def create_graph():
    """Frontend-facing: create graph (single-library mode: no-op, returns existing)."""
    try:
        body = request.get_json(force=True) or {}
        graph_id = body.get("graph_id", "library")
        registry = current_app.config.get("registry")
        if registry is None:
            return err("Registry 未初始化", 503)
        from core.server.registry import GraphRegistry
        graph_id = GraphRegistry.normalize_graph_id(graph_id)
        _processor = registry.get_processor(graph_id)
        info = registry.get_graph_info(graph_id)
        return ok(info or {"graph_id": graph_id, "status": "exists"})
    except ValueError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(str(e), 500)


@system_bp.route("/api/v1/graphs/<graph_id>", methods=["DELETE"])
def delete_graph(graph_id: str):
    """Frontend-facing: delete graph (single-library mode: returns error advising clear)."""
    try:
        registry = current_app.config.get("registry")
        if registry is None:
            return err("Registry 未初始化", 503)
        registry.delete_graph(graph_id)
        return ok({"deleted": graph_id})
    except ValueError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(str(e), 500)


@system_bp.route("/api/v1/graphs/<graph_id>/clear", methods=["POST"])
def clear_graph(graph_id: str):
    """Frontend-facing: clear graph data."""
    try:
        registry = current_app.config.get("registry")
        if registry is None:
            return err("Registry 未初始化", 503)
        registry.clear_graph(graph_id)
        return ok({"cleared": graph_id})
    except ValueError as e:
        return err(str(e), 400)
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


@system_bp.route("/api/v1/system/config", methods=["GET", "PATCH"])
def system_config():
    """读取/更新服务配置文件。部分运行时配置需重启后完全生效。"""
    try:
        cfg = current_app.config.get("config") or {}
        config_path = cfg.get("_config_path") or "service_config.json"
        path = Path(config_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if request.method == "GET":
            # Redact sensitive fields before returning to client
            _SENSITIVE_KEYS = frozenset({"api_key", "secret_key", "password", "token"})
            def _redact(d):
                if not isinstance(d, dict):
                    return d
                out = {}
                for k, v in d.items():
                    if k in _SENSITIVE_KEYS and isinstance(v, str) and v:
                        out[k] = v[:4] + "****" if len(v) > 4 else "****"
                    elif isinstance(v, dict):
                        out[k] = _redact(v)
                    else:
                        out[k] = v
                return out
            return ok({
                "config": _redact(cfg),
                "config_path": str(path),
                "notes": [
                    "llm.max_concurrency 控制全局 LLM 并发上限",
                    "runtime.concurrency.queue_workers 控制同时运行的 remember 任务数",
                    "embedding.max_concurrency 本地 embedding 通常建议为 1",
                    "已存在的图谱处理器可能需要重启服务后应用模型/embedding 改动",
                ],
            })

        body = request.get_json(force=True) or {}
        patch = body.get("config") if isinstance(body.get("config"), dict) else body
        if not isinstance(patch, dict):
            return err("config patch 必须是对象", 400)
        allowed_top = {
            "llm", "embedding", "runtime", "pipeline", "chunking",
            "port", "host", "flask_threaded",
        }
        rejected = sorted(k for k in patch if k not in allowed_top)
        if rejected:
            return err("不允许修改配置项: " + ", ".join(rejected), 400)
        next_cfg = dict(cfg)

        def deep_merge(a, b):
            out = dict(a or {})
            for k, v in (b or {}).items():
                if isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = deep_merge(out[k], v)
                else:
                    out[k] = v
            return out

        next_cfg = deep_merge(next_cfg, patch)
        next_cfg.pop("_config_path", None)
        config_json = json.dumps(next_cfg, ensure_ascii=False, indent=2)
        # Atomic write: write to temp file then rename to prevent corruption
        import tempfile
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(config_json)
            # os.replace is atomic on both Windows (if same volume) and POSIX
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        next_cfg["_config_path"] = str(path)
        current_app.config["config"] = next_cfg
        return ok({
            "config": _redact(next_cfg),
            "config_path": str(path),
            "message": "配置已保存；模型、embedding、worker 数等对已创建实例可能需要重启服务后完全生效",
        })
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