"""
System routes — Health checks, system monitoring, stats, and route index.
"""
from __future__ import annotations

import logging
import json
import os
import threading
import time
from urllib.parse import urlparse
from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, request

from core.server.llm_utils import call_llm_with_backoff, check_llm_available
from core.server.routes.helpers import ok, err, _get_processor, _get_system_monitor

logger = logging.getLogger(__name__)

system_bp = Blueprint("system", __name__)

# Rate limit for LLM health check (prevent credit burn)
_last_llm_health_time = 0.0
_last_llm_health_lock = threading.Lock()
_LLM_HEALTH_MIN_INTERVAL = 30.0  # seconds
_last_llm_health_result: tuple[bool, str, int] | None = None

_SENSITIVE_CONFIG_KEYS = frozenset({
    "api_key", "secret_key", "password", "token", "authorization",
})
_REDACTED_VALUE = "••••••••"
_config_write_lock = threading.Lock()


def _is_sensitive_key(key: str) -> bool:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(key))
    compact = normalized.replace("_", "")
    if normalized in _SENSITIVE_CONFIG_KEYS:
        return True
    return any(token in compact for token in ("apikey", "secret", "password", "token", "credential", "privatekey", "authorization"))


def _redact_config(value: Any, key: str | None = None) -> Any:
    """Recursively redact credentials without leaking a useful prefix."""
    if key is not None and _is_sensitive_key(key):
        return _REDACTED_VALUE if value else ""
    if isinstance(value, dict):
        return {k: _redact_config(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_config(v) for v in value]
    return value


def _is_redacted_value(value: Any) -> bool:
    """Recognize values returned by the redactor in a client PATCH."""
    if not isinstance(value, str):
        return False
    return value == _REDACTED_VALUE or (value.endswith("****") and "*" in value)


def _deep_merge_config(base: dict, patch: dict) -> dict:
    """Merge a config patch while preserving credentials left redacted."""
    out = dict(base or {})
    for key, value in (patch or {}).items():
        if _is_sensitive_key(str(key)) and _is_redacted_value(value):
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_config(out[key], value)
        else:
            out[key] = value
    return out


def _validate_config_patch(patch: dict) -> str | None:
    """Validate the small set of values exposed by the Web settings editor."""
    def _endpoint_at(path: tuple[str, ...]) -> str | None:
        cur: Any = patch
        for part in path:
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        if cur in (None, ""):
            return None
        if not isinstance(cur, str) or len(cur) > 2048:
            return f"{'.'.join(path)} 必须是有效的 HTTP(S) URL"
        parsed = urlparse(cur.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            return f"{'.'.join(path)} 必须是有效的 HTTP(S) URL，且不能包含用户凭据"
        return None

    def _int_at(path: tuple[str, ...], minimum: int, maximum: int) -> str | None:
        cur: Any = patch
        for part in path:
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        if cur is None:
            return None
        if isinstance(cur, bool) or not isinstance(cur, int) or not minimum <= cur <= maximum:
            return f"{'.'.join(path)} 必须是 {minimum} 到 {maximum} 之间的整数"
        return None

    checks = [
        (("llm", "max_concurrency"), 1, 128),
        (("embedding", "max_concurrency"), 1, 64),
        (("runtime", "concurrency", "queue_workers"), 1, 64),
        (("chunking", "window_size"), 1, 1_000_000),
        (("chunking", "overlap"), 0, 999_999),
        (("port",), 1, 65_535),
        (("max_request_bytes",), 1_048_576, 256 * 1024 * 1024),
        (("rate_limit_per_minute",), 0, 100_000),
    ]
    for path, minimum, maximum in checks:
        error = _int_at(path, minimum, maximum)
        if error:
            return error

    for endpoint_path in (
        ("llm", "base_url"),
        ("llm", "alignment", "base_url"),
        ("embedding", "api_base"),
    ):
        endpoint_error = _endpoint_at(endpoint_path)
        if endpoint_error:
            return endpoint_error

    if "host" in patch:
        host = patch["host"]
        if not isinstance(host, str) or host.strip().lower() not in {"127.0.0.1", "localhost", "::1"}:
            return "host 只能通过配置文件/启动参数显式设置为非本机地址"

    debug = patch.get("pipeline", {}).get("debug", {}) if isinstance(patch.get("pipeline"), dict) else {}
    if isinstance(debug, dict) and "distill_data_dir" in debug:
        raw_dir = debug["distill_data_dir"]
        if not isinstance(raw_dir, str) or not raw_dir.strip() or any(part == ".." for part in Path(raw_dir).parts):
            return "pipeline.debug.distill_data_dir 不允许为空或跳出 storage_path"

    window = patch.get("chunking", {}).get("window_size") if isinstance(patch.get("chunking"), dict) else None
    overlap = patch.get("chunking", {}).get("overlap") if isinstance(patch.get("chunking"), dict) else None
    if window is not None and overlap is not None and overlap >= window:
        return "chunking.overlap 必须小于 chunking.window_size"
    return None


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
@system_bp.route("/health", methods=["GET"])
def health():
    """Lightweight liveness check that never initializes models/processors."""
    try:
        gid = getattr(request, 'graph_id', None) or request.args.get('graph_id', 'library')
        try:
            from core.server.registry import GraphRegistry
            gid = GraphRegistry.normalize_graph_id(gid)
        except ValueError as e:
            return err(str(e), 400)
        registry = current_app.config["registry"]
        processor = getattr(registry, "_processor", None)
        embedding_available = None
        if processor is not None:
            embedding_available = bool(
                processor.embedding_client is not None
                and processor.embedding_client.is_available()
            )
        storage_backend = "sqlite"
        return ok({
            "library_id": gid,
            "storage_backend": storage_backend,
            "embedding_available": embedding_available,
            "processor_initialized": processor is not None,
        })
    except Exception as e:
        return err(str(e), 500)


@system_bp.route("/api/v1/health/llm", methods=["GET"])
def health_llm():
    """检查大模型是否可访问。"""
    global _last_llm_health_time, _last_llm_health_result
    now = time.time()
    gid = getattr(request, 'graph_id', None) or request.args.get('graph_id', 'library')
    from core.server.registry import GraphRegistry
    gid = GraphRegistry.normalize_graph_id(gid)
    with _last_llm_health_lock:
        if now - _last_llm_health_time < _LLM_HEALTH_MIN_INTERVAL:
            remaining = round(_LLM_HEALTH_MIN_INTERVAL - (now - _last_llm_health_time), 1)
            # Do not report a failed check as healthy merely because it is in
            # cooldown.  If another request is still running, expose a
            # truthful 503 instead of starting a second expensive probe.
            if _last_llm_health_result is None:
                return err("大模型健康检查正在进行，请稍后重试", 503)
            healthy, message, status = _last_llm_health_result
            payload = {
                "library_id": gid,
                "llm_available": healthy,
                "message": message,
                "cooldown_remaining": remaining,
                "cached": True,
            }
            return ok(payload) if status == 200 else err(message, status)
        _last_llm_health_time = now
        _last_llm_health_result = None
    try:
        cfg = current_app.config.get("config") or {}
        llm_cfg = cfg.get("llm") or {}
        if not llm_cfg.get("api_key") and not llm_cfg.get("base_url"):
            message = "大模型未配置"
            with _last_llm_health_lock:
                _last_llm_health_result = (False, message, 503)
            return err(message, 503)
        processor = current_app.config["registry"].get_processor(gid)
        response = _call_llm_with_backoff(
            processor,
            "请只回复一个词：OK",
            timeout=60,
        )
        message = "大模型访问正常"
        with _last_llm_health_lock:
            _last_llm_health_result = (True, message, 200)
        return ok({"library_id": gid, "llm_available": True, "message": message, "response_preview": response.strip()[:80]})
    except Exception as e:
        message = f"大模型不可用: {e}"
        with _last_llm_health_lock:
            _last_llm_health_result = (False, message, 503)
        return err(message, 503)


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
        task_limit = min(max(request.args.get("task_limit", 50, type=int) or 50, 1), 200)
        log_limit = min(max(request.args.get("log_limit", 100, type=int) or 100, 1), 500)
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
        try:
            body = request.get_json(force=True)
        except Exception:
            return err("请求 JSON 无效", 400)
        if body is None:
            body = {}
        if not isinstance(body, dict):
            return err("请求 JSON 必须是对象", 400)
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
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or body.get("confirm_graph_id") != graph_id:
            return err("清空图谱需要 confirm_graph_id 与目标图谱一致", 400)
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
        limit = min(max(request.args.get("limit", 50, type=int) or 50, 1), 200)
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
            return ok({
                "config": _redact_config(cfg),
                "config_path": str(path),
                "notes": [
                    "llm.max_concurrency 控制全局 LLM 并发上限",
                    "runtime.concurrency.queue_workers 控制同时运行的 remember 任务数",
                    "embedding.max_concurrency 本地 embedding 通常建议为 1",
                    "已存在的图谱处理器可能需要重启服务后应用模型/embedding 改动",
                ],
            })

        body = request.get_json(force=True)
        if not isinstance(body, dict):
            return err("请求 JSON 必须是对象", 400)
        patch = body.get("config") if isinstance(body.get("config"), dict) else body
        if not isinstance(patch, dict):
            return err("config patch 必须是对象", 400)

        # Accept the pre-v0.2 UI spelling once, then normalize it to the
        # canonical nested setting.  This keeps older clients usable while
        # preventing the obsolete key from being persisted.
        if "remember_workers" in patch:
            patch = dict(patch)
            legacy_workers = patch.pop("remember_workers")
            runtime = dict(patch.get("runtime") or {})
            concurrency = dict(runtime.get("concurrency") or {})
            concurrency.setdefault("queue_workers", legacy_workers)
            runtime["concurrency"] = concurrency
            patch["runtime"] = runtime

        allowed_top = {
            "llm", "embedding", "runtime", "pipeline", "chunking",
            "port", "host", "flask_threaded", "max_request_bytes", "rate_limit_per_minute",
        }
        rejected = sorted(k for k in patch if k not in allowed_top)
        if rejected:
            return err("不允许修改配置项: " + ", ".join(rejected), 400)
        validation_error = _validate_config_patch(patch)
        if validation_error:
            return err(validation_error, 400)

        # Serialize read/merge/write/update so two settings tabs cannot lose
        # each other's changes or return a response for a half-written file.
        with _config_write_lock:
            next_cfg = _deep_merge_config(dict(cfg), patch)
            next_cfg.pop("_config_path", None)
            try:
                from core.server.config import _validate_config
                _validate_config(next_cfg)
            except Exception as exc:
                # ConfigError is a user-facing 400; unexpected validator
                # failures are still kept out of the response body.
                if exc.__class__.__name__ == "ConfigError":
                    return err(str(exc), 400)
                logger.exception("配置 patch 校验失败")
                return err("配置校验失败", 400)
            config_json = json.dumps(next_cfg, ensure_ascii=False, indent=2)
            # Atomic write: write to temp file then rename to prevent corruption
            import tempfile
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(config_json)
                    f.flush()
                    os.fsync(f.fileno())
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
            "config": _redact_config(next_cfg),
            "config_path": str(path),
            "message": "配置已保存；模型、embedding、worker 数等对已创建实例可能需要重启服务后完全生效",
        })
    except Exception:
        logger.exception("system config request failed")
        return err("配置读写失败", 500)


@system_bp.route("/api/v1/system/logs", methods=["GET"])
def system_logs():
    """最近系统日志。支持 ?limit=&level=&source= 筛选。"""
    try:
        system_monitor = _get_system_monitor()
        if system_monitor is None:
            return err("SystemMonitor 未启用", 503)
        limit = min(max(request.args.get("limit", 50, type=int) or 50, 1), 500)
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
