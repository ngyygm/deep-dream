"""
DeepDream 自然语言记忆图 API（多图谱模式）

一个以自然语言为核心的统一记忆图服务。系统只有两个核心职责：
  - Remember：接收自然语言文本或文档，自动构建概念实体/关系图。
  - Find：通过语义检索从总图中唤醒相关的局部记忆区域。

多图谱模式：支持按 graph_id 隔离不同知识图谱，所有 API 请求需带 graph_id 参数。
系统不负责 select，外部智能体根据 find 结果自行决策。

路由已拆分至 server/blueprints/ 下的各 Blueprint 模块，此文件仅作为应用工厂。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import atexit
import errno
import logging
import os
import re as _re

logger = logging.getLogger(__name__)
import signal
import socket
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, abort, jsonify, make_response, redirect, request

_MUTATING_METHODS = frozenset(("POST", "PUT", "DELETE", "PATCH"))
_MUTATING_JSON_METHODS = frozenset(("POST", "PUT", "PATCH"))
from werkzeug.exceptions import NotFound

from core.server.config import load_config, merge_llm_alignment, resolve_embedding_model
from core.server.monitor import LOG_MODE_DETAIL, LOG_MODE_MONITOR, SystemMonitor
from core.server.task_queue import RememberTask, RememberTaskQueue
from core.server.registry import GraphRegistry
from core.server.sse import sse_response, queue_to_generator
from core.server import auth as auth_module
from core import TemporalMemoryGraphProcessor
from core.llm.client import LLM_PRIORITY_STEP6
from core.server.llm_utils import check_llm_available, call_llm_with_backoff


# ----------------------------------------------------------------------
# Input validation helpers
# ----------------------------------------------------------------------

def _validate_graph_id(graph_id: Any) -> Tuple[bool, Optional[str], int]:
    """Validate graph_id parameter. Returns (is_valid, error_message, status_code)."""
    if not graph_id:
        return False, "graph_id is required", 400
    if not isinstance(graph_id, str):
        return False, "graph_id must be a string", 400
    try:
        GraphRegistry.validate_graph_id(graph_id)
        return True, None, 200
    except ValueError as e:
        return False, str(e), 400


def _validate_text_input(text: Any, field_name: str = "text", min_length: int = 1,
                          max_length: int = 10_000_000) -> Tuple[bool, Optional[str], int]:
    """Validate text input parameters. Returns (is_valid, error_message, status_code)."""
    if text is None:
        return False, f"{field_name} is required", 400
    if not isinstance(text, str):
        return False, f"{field_name} must be a string", 400
    text = text.strip()
    if len(text) < min_length:
        return False, f"{field_name} must be at least {min_length} character(s)", 400
    if len(text) > max_length:
        return False, f"{field_name} exceeds maximum length of {max_length} characters", 400
    return True, None, 200


def _validate_positive_int(value: Any, field_name: str = "value",
                           min_val: int = 1, max_val: int = 10000) -> Tuple[bool, Optional[str], int]:
    """Validate positive integer parameters. Returns (is_valid, error_message, status_code)."""
    if value is None:
        return True, None, 200  # Optional parameter
    try:
        int_val = int(value)
        if int_val < min_val:
            return False, f"{field_name} must be at least {min_val}", 400
        if int_val > max_val:
            return False, f"{field_name} must not exceed {max_val}", 400
        return True, None, 200
    except (ValueError, TypeError):
        return False, f"{field_name} must be a valid integer", 400


def _validate_float_range(value: Any, field_name: str = "value",
                          min_val: float = 0.0, max_val: float = 1.0) -> Tuple[bool, Optional[str], int]:
    """Validate float parameters in a range. Returns (is_valid, error_message, status_code)."""
    if value is None:
        return True, None, 200  # Optional parameter
    try:
        float_val = float(value)
        if float_val < min_val or float_val > max_val:
            return False, f"{field_name} must be between {min_val} and {max_val}", 400
        return True, None, 200
    except (ValueError, TypeError):
        return False, f"{field_name} must be a valid number", 400


def _make_validation_error(message: str) -> tuple:
    """Return a standardized validation error response."""
    return jsonify({"success": False, "error": message}), 400


def create_app(
    registry,
    config: Optional[Dict[str, Any]] = None,
    system_monitor: Optional[SystemMonitor] = None,
) -> Flask:
    static_dir = Path(__file__).resolve().parent / "static"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="/static")
    app.json.ensure_ascii = False
    app.config["system_monitor"] = system_monitor
    app.config["registry"] = registry

    # CORS：仅允许同源和 localhost 跨域调用
    _ALLOWED_ORIGINS = {"http://localhost", "http://127.0.0.1"}
    @app.after_request
    def _cors_headers(response):
        origin = request.environ.get("HTTP_ORIGIN")
        # Security: Use exact match instead of startswith to prevent subdomain attacks
        if origin in _ALLOWED_ORIGINS:
            response.headers["Access-Control-Allow-Origin"] = origin
        else:
            response.headers["Access-Control-Allow-Origin"] = ""
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Graph-Id, X-API-Key"
        # Security: Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # Comprehensive CSP policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # unsafe-inline needed for Chart.js and dynamic scripts
            "style-src 'self' 'unsafe-inline'",  # unsafe-inline needed for component styles
            "img-src 'self' data: https: blob:",
            "font-src 'self' data:",
            "connect-src 'self' http://localhost:* http://127.0.0.1:*",  # Allow local API calls
            "object-src 'none'",  # Prevent plugins
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
            "upgrade-insecure-requests",
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        # 静态文件禁止缓存，确保部署后浏览器立即加载最新版本
        if request.path.startswith("/static/") and request.path.endswith((".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.before_request
    def _cors_preflight():
        if request.method == "OPTIONS":
            return make_response("", 204)

    @app.before_request
    def _record_start():
        _now = time.time()
        request.start_time = _now
        request._monitor_start = _now

    @app.after_request
    def _track_access(response):
        monitor = app.config.get("system_monitor")
        if monitor is not None and hasattr(request, "_monitor_start"):
            duration_ms = (time.time() - request._monitor_start) * 1000
            monitor.access_tracker.record(
                method=request.method,
                path=request.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
                graph_id=getattr(request, "graph_id", None),
            )
            # Auto-log server errors (5xx) to event_log so they appear in system logs
            if response.status_code >= 500:
                monitor.event_log.error(
                    "API",
                    f"{request.method} {request.path} → {response.status_code}",
                )
        return response

    config = config or {}

    # Initialize authentication module
    auth_module.init_auth(config)

    # Configure authentication from config
    auth_config = config.get("auth", {})
    auth_enabled = auth_config.get("enabled", True)
    strict_mode = auth_config.get("strict_mode", False)

    # Public endpoints that don't require authentication
    _PUBLIC_ROUTES = {
        "/", "/health", "/api/", "/api/v1/",
        "/api/v1/graphs", "/api/v1/routes",
        "/static/", "/favicon.ico",
    }

    # Read-only endpoints that can be accessed with read permission
    _READ_ONLY_ROUTES = {
        "/api/v1/find", "/api/v1/find/entities",
        "/api/v1/find/relations", "/api/v1/find/episodes",
        "/api/v1/stats",
    }

    def _is_public_route(path: str) -> bool:
        """Check if a path is a public endpoint."""
        for route in _PUBLIC_ROUTES:
            if path == route or path.startswith(route):
                return True
        return False

    def _is_read_only_route(path: str) -> bool:
        """Check if a path is a read-only endpoint."""
        for route in _READ_ONLY_ROUTES:
            if path.startswith(route):
                return True
        return False

    @app.before_request
    def _authenticate_request():
        """Authenticate incoming requests using the auth module."""
        # Skip authentication if disabled
        if not auth_enabled:
            return None

        # Skip authentication for public routes
        if _is_public_route(request.path):
            return None

        # Skip OPTIONS requests for CORS preflight
        if request.method == "OPTIONS":
            return None

        # Try API key authentication
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            is_valid, permissions = auth_module._validate_api_key(api_key)
            if is_valid:
                # Store auth info in Flask's g object
                from flask import g
                g.authenticated = True
                g.auth_method = "api_key"
                g.user_id = f"api_key:{api_key[:8]}"
                g.permissions = permissions
                return None
            else:
                return jsonify({
                    "success": False,
                    "error": "Invalid API key"
                }), 401

        # Try JWT authentication
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            is_valid, permissions, payload = auth_module._validate_jwt_token(token)
            if is_valid:
                from flask import g
                g.authenticated = True
                g.auth_method = "jwt"
                g.user_id = payload.get("user_id") if payload else None
                g.permissions = permissions
                return None
            else:
                return jsonify({
                    "success": False,
                    "error": "Invalid or expired token"
                }), 401

        # No authentication provided
        if strict_mode:
            return jsonify({
                "success": False,
                "error": "Authentication required. Provide X-API-Key or Authorization: Bearer <token> header."
            }), 401

        # In non-strict mode, allow the request but log a warning
        logger.warning(f"Unauthenticated request to {request.path} (auth enabled but not enforced)")

        # Set default permissions for unauthenticated requests in dev mode
        from flask import g
        g.authenticated = False
        g.permissions = {"read"}  # Default to read-only access
        return None

    # 不需要 graph_id 的路由（白名单）
    _NO_GRAPH_ID_ROUTES = frozenset([
        "/", "/api/v1/routes", "/api/v1/health",
        "/api/v1/chat/sessions", "/api/v1/graphs",
    ])
    # 系统 API 前缀（不需要 graph_id）
    _SYSTEM_API_PREFIX = "/api/v1/system/"

    # 简单内存限流（按 IP + graph_id，滑动窗口）
    from collections import deque as _deque
    _rate_limit_store: Dict[str, _deque] = {}
    _rate_limit_lock = threading.Lock()
    _RATE_LIMIT = int(config.get("rate_limit_per_minute", 0))
    _RATE_WINDOW = 60.0  # 秒

    @app.before_request
    def _validate_content_type():
        """Validate Content-Type for JSON endpoints to prevent CSRF."""
        # Skip validation for GET/DELETE/OPTIONS and non-JSON endpoints
        if request.method not in _MUTATING_JSON_METHODS:
            return
        # Skip for file upload endpoints
        if request.path.startswith("/api/v1/remember") and request.content_type and "multipart" in request.content_type:
            return
        # For JSON endpoints, require application/json content type
        if request.content_type and not request.content_type.startswith("application/json"):
            # Allow form data for compatibility
            if request.content_type.startswith("application/x-www-form-urlencoded") or request.content_type.startswith("multipart/form-data"):
                return
            return jsonify({"success": False, "error": "Invalid Content-Type. Use application/json"}), 415

    @app.before_request
    def _resolve_graph_id():
        """在请求进入端点前解析 graph_id 并挂到 request.graph_id 上。
        不需要 graph_id 的路由跳过。未填时默认使用 'default'。"""
        path = request.path
        if path in _NO_GRAPH_ID_ROUTES or path.startswith(_SYSTEM_API_PREFIX):
            return
        gid = ""
        # 1. Header: X-Graph-Id（前端 / MCP 客户端常用）
        gid = (request.headers.get("X-Graph-Id") or "").strip()
        if gid:
            try:
                GraphRegistry.validate_graph_id(gid)
            except ValueError as e:
                return jsonify({"success": False, "error": str(e)}), 400
            request.graph_id = gid
            return
        # 2. Request body (POST/PUT/DELETE/PATCH)
        if request.method in _MUTATING_METHODS:
            body = request.get_json(silent=True)
            if isinstance(body, dict):
                gid = (body.get("graph_id") or "").strip()
        # 3. Form data
        if not gid:
            gid = (request.form.get("graph_id") or "").strip()
        # 4. Query parameter
        if not gid:
            gid = (request.args.get("graph_id") or "").strip()
        # 5. Default
        if not gid:
            gid = "default"
        try:
            GraphRegistry.validate_graph_id(gid)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        request.graph_id = gid

    @app.before_request
    def _rate_limit_check():
        if _RATE_LIMIT <= 0:
            return
        now = time.time()
        # Rate limit is scoped per (IP, graph_id) to prevent cross-graph interference
        client_ip = request.remote_addr or "unknown"
        gid = getattr(request, "graph_id", "default")
        rate_key = f"{client_ip}|{gid}"
        with _rate_limit_lock:
            timestamps = _rate_limit_store.get(rate_key)
            if timestamps is None:
                _rate_limit_store[rate_key] = _deque([now], maxlen=_RATE_LIMIT + 10)
                return
            # O(1) eviction: expired timestamps are always at the front (arrive in order)
            cutoff = now - _RATE_WINDOW
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            timestamps.append(now)
            # Periodic stale-key cleanup (~1% of requests when store grows large)
            if len(_rate_limit_store) > 1000 and (int(now * 1000) % 100 < 1):
                stale_keys = [k for k, ts in _rate_limit_store.items()
                              if not ts or ts[-1] < cutoff]
                for k in stale_keys:
                    del _rate_limit_store[k]
            if len(timestamps) > _RATE_LIMIT:
                return jsonify({"success": False, "error": "请求过于频繁，请稍后再试"}), 429

    # 向后兼容：/api/<path> → /api/v1/<path>（308 永久重定向）
    # 仅对非 v1/ 开头的路径重定向，避免 /api/v1/xxx → /api/v1/v1/xxx 双重前缀
    @app.route("/api/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE"])
    def _api_redirect(subpath):
        if subpath.startswith("v1/"):
            # 已含 v1 前缀的路径到达此 catch-all，说明路由不存在，返回 404
            abort(404)
        return redirect(f"/api/v1/{subpath}", code=308)

    @app.route("/api")
    def _api_root_redirect():
        return redirect("/api/v1/", code=308)

    # ── Register all Blueprint modules ────────────────────────────────────
    from core.server.blueprints.system import system_bp
    from core.server.blueprints.remember import remember_bp
    from core.server.blueprints.entities import entities_bp
    from core.server.blueprints.relations import relations_bp
    from core.server.blueprints.episodes import episodes_bp
    from core.server.blueprints.dream import dream_bp
    from core.server.blueprints.concepts import concepts_bp

    app.register_blueprint(system_bp)
    app.register_blueprint(remember_bp)
    app.register_blueprint(entities_bp)
    app.register_blueprint(relations_bp)
    app.register_blueprint(episodes_bp)
    app.register_blueprint(dream_bp)
    app.register_blueprint(concepts_bp)

    # ── SPA fallback routes (must be last) ─────────────────────────────────
    _API_PREFIXES = ("/api/", "/health")

    @app.route("/", methods=["GET"])
    def serve_index():
        return app.send_static_file("index.html")

    @app.route("/<path:path>", methods=["GET"])
    def serve_spa(path):
        # API 路由不拦截
        if path.startswith("/api/") or path.startswith("/health"):
            return abort(404)
        # 尝试静态文件
        try:
            return app.send_static_file(path)
        except NotFound:
            return app.send_static_file("index.html")

    return app


def build_processor(config: Dict[str, Any]) -> TemporalMemoryGraphProcessor:
    storage_path = config.get("storage_path", "./graph/tmg_storage")
    chunking = config.get("chunking") or {}
    window_size = chunking.get("window_size", 1000)
    overlap = chunking.get("overlap", 200)
    llm = config.get("llm") or {}
    embedding = config.get("embedding") or {}
    pipeline = config.get("pipeline") or {}
    runtime = config.get("runtime") or {}
    runtime_concurrency = runtime.get("concurrency") or {}
    runtime_task = runtime.get("task") or {}
    pipeline_search = pipeline.get("search") or {}
    pipeline_alignment = pipeline.get("alignment") or {}
    pipeline_extraction = pipeline.get("extraction") or {}
    pipeline_remember = pipeline.get("remember") or {}
    pipeline_debug = pipeline.get("debug") or {}
    max_concurrency = llm.get("max_concurrency")
    model_path, model_name, use_local = resolve_embedding_model(embedding)
    kwargs: Dict[str, Any] = {
        "storage_path": storage_path,
        "window_size": window_size,
        "overlap": overlap,
        "llm_api_key": llm.get("api_key"),
        "llm_model": llm.get("model", "gpt-4"),
        "llm_base_url": llm.get("base_url"),
        "alignment_llm": merge_llm_alignment(llm),
        "llm_think_mode": bool(llm.get("think", llm.get("think_mode", False))),
        "llm_max_tokens": llm.get("max_tokens") if llm.get("max_tokens") else None,
        "llm_context_window_tokens": llm.get("context_window_tokens"),
        "max_llm_concurrency": max_concurrency,
        "llm_timeout_seconds": llm.get("timeout_seconds", 300),
        "llm_connect_timeout_seconds": llm.get("connect_timeout_seconds", 30),
        "embedding_model_path": model_path,
        "embedding_model_name": model_name,
        "embedding_device": embedding.get("device", "cpu"),
        "embedding_use_local": use_local,
        "embedding_cache_max_size": embedding.get("cache_max_size"),
        "embedding_cache_ttl": embedding.get("cache_ttl"),
        "load_cache_memory": runtime_task.get("load_cache_memory", pipeline.get("load_cache_memory")),
        "max_concurrent_windows": runtime_concurrency.get("window_workers", pipeline.get("max_concurrent_windows")),
    }
    for key in (
        "similarity_threshold", "max_similar_entities", "content_snippet_length",
        "relation_content_snippet_length", "relation_endpoint_jaccard_threshold",
        "relation_endpoint_embedding_threshold",
        "jaccard_search_threshold",
        "embedding_name_search_threshold", "embedding_full_search_threshold",
    ):
        if key in pipeline_search:
            kwargs[key] = pipeline_search[key]
    if "max_alignment_candidates" in pipeline_alignment:
        kwargs["max_alignment_candidates"] = pipeline_alignment["max_alignment_candidates"]
    for key in (
        "extraction_rounds", "entity_extraction_rounds", "relation_extraction_rounds",
        "entity_post_enhancement", "prompt_episode_max_chars",
        "compress_multi_round_extraction",
    ):
        if key in pipeline_extraction:
            kwargs[key] = pipeline_extraction[key]
    if pipeline_remember:
        kwargs["remember_config"] = pipeline_remember
    if "distill_data_dir" in pipeline_debug:
        kwargs["distill_data_dir"] = pipeline_debug["distill_data_dir"]
    return TemporalMemoryGraphProcessor(**kwargs)


def _check_llm_available(processor) -> tuple[bool, str | None]:
    """启动前握手：检查上游 LLM；若启用 alignment 专用通道，再按步骤 6/7 优先级检查对齐端点。"""
    return check_llm_available(processor, priority_steps=[6])


def _call_llm_with_backoff(processor, prompt, timeout=60, max_waits=5, backoff_base_seconds=3):
    """调用 LLM（指数退避重试）—— 代理到共享模块。"""
    return call_llm_with_backoff(processor, prompt, timeout=timeout, max_waits=max_waits, backoff_base_seconds=backoff_base_seconds)


def _tcp_bind_probe(host: str, port: int) -> Tuple[bool, Optional[str]]:
    """尝试在 host:port 上独占 bind，用于启动前检测端口是否可用。"""
    bind_addr = host if host not in ("", "0.0.0.0") else "0.0.0.0"
    sock: Optional[socket.socket] = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_addr, int(port)))
        return True, None
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            return False, "端口已被占用 (EADDRINUSE)"
        return False, str(e)
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def _get_port_pids(port: int) -> List[int]:
    """获取占用指定端口的 PID 列表（排除自身）。"""
    my_pid = os.getpid()
    pids: List[int] = []

    # 优先用 ss（更快、更普遍）
    try:
        result = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=5,
        )
        for m in _re.finditer(r"pid=(\d+)", result.stdout):
            pid = int(m.group(1))
            if pid != my_pid:
                pids.append(pid)
        if pids:
            return pids
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 回退到 lsof
    try:
        result = subprocess.run(
            ["lsof", "-t", "-i", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            pid = int(line.strip())
            if pid != my_pid:
                pids.append(pid)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return pids


def _kill_port_occupants(port: int) -> bool:
    """Kill processes occupying the given port. Returns True if all killed."""
    pids = _get_port_pids(port)
    if not pids:
        return True

    all_killed = True
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            logging.info("已发送 SIGTERM 到进程 %d (占用端口 %d)", pid, port)
        except ProcessLookupError:
            pass
        except PermissionError:
            logging.warning("无权限终止进程 %d", pid)
            all_killed = False

    # 轮询等待进程退出（最多 3 秒）
    for _ in range(15):
        remaining = _get_port_pids(port)
        if not remaining:
            return True
        time.sleep(0.2)

    # SIGTERM 没杀掉，升级到 SIGKILL
    remaining = _get_port_pids(port)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
            logging.warning("SIGTERM 无效，已发送 SIGKILL 到进程 %d (占用端口 %d)", pid, port)
        except ProcessLookupError:
            pass
        except PermissionError:
            logging.warning("无权限终止进程 %d", pid)
            all_killed = False

    # 再等 1 秒确认
    time.sleep(1)
    return not _get_port_pids(port)


def _resolve_listen_port(
    host: str,
    preferred_port: int,
    auto_fallback: bool,
    max_extra: int = 10,
) -> Tuple[int, bool]:
    """
    若 preferred_port 可 bind 则用之；否则在 auto_fallback 时尝试 preferred_port+1 … +max_extra。
    返回 (实际端口, 是否发生了端口切换)。
    """
    ok, _ = _tcp_bind_probe(host, preferred_port)
    if ok:
        return preferred_port, False
    if not auto_fallback:
        return preferred_port, False
    for delta in range(1, max_extra + 1):
        p = preferred_port + delta
        ok2, _ = _tcp_bind_probe(host, p)
        if ok2:
            return p, True
    return preferred_port, False


def _check_storage_writable(storage_root: Path) -> Optional[str]:
    """在 storage_path 下尝试创建/删除测试文件，不可写则返回错误说明。"""
    probe = storage_root / ".tmg_write_probe"
    try:
        storage_root.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return None
    except OSError as e:
        return f"存储路径不可写或无法创建: {storage_root} ({e})"


def main() -> int:
    # 优先使用本地缓存的模型，避免每次启动都尝试联网检查更新
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    parser = argparse.ArgumentParser(description="DeepDream 自然语言记忆图 API（Remember + Find）")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径（如 service_config.json）")
    parser.add_argument("--host", type=str, default=None, help="覆盖配置中的 host")
    parser.add_argument("--port", type=int, default=None, help="覆盖配置中的 port")
    parser.add_argument(
        "--log-mode",
        type=str,
        choices=[LOG_MODE_DETAIL, LOG_MODE_MONITOR],
        default=None,
        help="日志模式：detail 输出细节；monitor 固定刷新监控面板",
    )
    parser.add_argument(
        "--monitor-refresh",
        type=float,
        default=None,
        help="monitor 模式下面板刷新周期（秒，默认 1.0）",
    )
    parser.add_argument(
        "--skip-llm-check",
        action="store_true",
        help="跳过启动前 LLM 握手（仅适合调试 Find；Remember 仍可能在运行时失败）",
    )
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="若配置端口被占用，自动尝试后续连续端口（最多 +10）",
    )
    parser.add_argument("--debug", action="store_true", help="开启 Flask 调试模式")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).exists():
        print(f"错误：配置文件不存在: {config_path}")
        return 1

    config = load_config(config_path)
    log_mode = args.log_mode if args.log_mode is not None else config.get("log_mode", LOG_MODE_DETAIL)
    monitor_refresh = args.monitor_refresh if args.monitor_refresh is not None else config.get("monitor_refresh_seconds", 1.0)
    config["log_mode"] = log_mode
    config["monitor_refresh_seconds"] = monitor_refresh
    config["_config_path"] = config_path

    # 禁用 Flask/werkzeug 的 HTTP access log（控制台太吵）
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    # 创建 SystemMonitor（替代 ConsoleReporter）
    system_monitor = SystemMonitor(config=config, mode=log_mode)

    host = args.host if args.host is not None else config.get("host", "0.0.0.0")
    port = args.port if args.port is not None else config.get("port", 5001)
    config["host"] = host
    config["port"] = port
    storage_path = config.get("storage_path", "./graph")
    storage_root = Path(storage_path)
    Path(storage_path).mkdir(parents=True, exist_ok=True)
    wr_err = _check_storage_writable(storage_root)
    if wr_err:
        system_monitor.event_log.error("System", f"错误：{wr_err}")
        return 1

    registry = GraphRegistry(storage_path, config, system_monitor=system_monitor)

    # 启动前对配置的 LLM 做握手，不可用则报错退出（可用 --skip-llm-check 跳过）
    if args.skip_llm_check:
        system_monitor.event_log.warn(
            "System",
            "已跳过 LLM 握手（--skip-llm-check）。Remember 与 /api/v1/health/llm 可能在运行时失败。",
        )
    else:
        system_monitor.event_log.info(
            "System",
            "正在检查配置的 LLM 是否可用（单次最多约 60s；失败将按 3/9/27… 秒退避重试，请稍候）…",
        )
        default_processor = registry.get_processor("default")
        ok_llm, err_msg = _check_llm_available(default_processor)
        if not ok_llm:
            system_monitor.event_log.error("System", f"错误：{err_msg}")
            system_monitor.event_log.error("System", "请检查 service_config 中 llm.api_key / llm.base_url / llm.model 及网络。")
            system_monitor.event_log.error("System", "若仅需先起服务再排查 LLM，可在启动命令加: --skip-llm-check")
            return 1
        system_monitor.event_log.info("System", "LLM 握手成功，模型可用。")

    app = create_app(registry, config, system_monitor=system_monitor)

    auto_fb = bool(args.auto_port or config.get("auto_port_fallback", False))
    listen_port, port_switched = _resolve_listen_port(host, port, auto_fb)
    ok_bind, bind_err = _tcp_bind_probe(host, listen_port)
    if not ok_bind:
        # 自动尝试 kill 占用端口的旧进程
        system_monitor.event_log.warn("System", f"端口 {host}:{listen_port} 已被占用，尝试自动释放...")
        killed = _kill_port_occupants(listen_port)
        ok_bind, bind_err = _tcp_bind_probe(host, listen_port)
        if not ok_bind:
            system_monitor.event_log.error("System", f"错误：无法在 {host}:{listen_port} 上绑定: {bind_err}")
            system_monitor.event_log.error("System", f"  配置的端口为 {port}。")
            if not auto_fb:
                system_monitor.event_log.error(
                    "System",
                    "  解决：结束占用该端口的进程，或改用 --port <其他端口>，"
                    "或在配置中设置 auto_port_fallback: true 并加 --auto-port。",
                )
                try:
                    system_monitor.event_log.error("System", f"  排查示例: ss -tlnp | grep ':{port} ' 或 lsof -i :{port}")
                except Exception as _e:
                    logger.debug("端口排查提示失败: %s", _e)
            else:
                system_monitor.event_log.error("System", "  已尝试自动换端口但仍失败，请检查系统权限或防火墙设置。")
            return 1
        else:
            system_monitor.event_log.info("System", f"旧进程已清理，端口 {listen_port} 已释放。")
    if port_switched:
        system_monitor.event_log.warn("System", f"注意：端口 {port} 已被占用，已自动改用 {listen_port}。")

    # 启动时触发 default 图谱的 queue 创建（会自动注册到 SystemMonitor）
    registry.get_queue("default")

    if log_mode == LOG_MODE_MONITOR:
        from core.server.dashboard import DeepDreamDashboard
        system_monitor.event_log.info("System", "监控面板已启用；任务细节日志已收敛为总览。")
        dashboard = DeepDreamDashboard(system_monitor, refresh_interval=monitor_refresh)
        dashboard.start()
    else:
        stats = system_monitor.graph_detail("default")
        entities = stats["storage"]["entities"] if stats else 0
        relations = stats["storage"]["relations"] if stats else 0
        caches = stats["storage"]["episodes"] if stats else 0
        print(f"""
╔══════════════════════════════════════════════════════════╗
║     DeepDream — 自然语言记忆图 API           ║
╚══════════════════════════════════════════════════════════╝

  当前大脑记忆库 (default):
    实体: {entities}  关系: {relations}  Episode: {caches}

  服务地址: http://{host}:{listen_port}
  健康检查: GET  http://{host}:{listen_port}/api/v1/health?graph_id=default
  LLM 健康: GET  http://{host}:{listen_port}/api/v1/health/llm?graph_id=default
  图谱列表: GET  http://{host}:{listen_port}/api/v1/graphs
  记忆写入: POST http://{host}:{listen_port}/api/v1/remember （JSON 含 graph_id / text，或 multipart file 上传）
  任务状态: GET  http://{host}:{listen_port}/api/v1/remember/tasks/<task_id>?graph_id=default
  监控快照: GET  http://{host}:{listen_port}/api/v1/remember/monitor?graph_id=default
  系统总览: GET  http://{host}:{listen_port}/api/v1/system/overview
  系统日志: GET  http://{host}:{listen_port}/api/v1/system/logs
  访问统计: GET  http://{host}:{listen_port}/api/v1/system/access-stats
  语义检索: POST http://{host}:{listen_port}/api/v1/find
  接口索引: GET  http://{host}:{listen_port}/api/v1/routes
  原子查询: GET  http://{host}:{listen_port}/api/v1/find/...

  存储基础路径: {storage_path} （各图谱在 {storage_path}/<graph_id>/ 下）
  HTTP 多线程: 处理中 Find 与 Remember 可并行（Flask threaded）
  多图谱模式: 所有 API 请求需带 graph_id 参数
  日志模式: {log_mode}

  按 Ctrl+C 停止服务
""")
    threaded = bool(config.get("flask_threaded", True))

    # 优雅关闭：捕获 SIGTERM（SIGINT 由 Python 默认 KeyboardInterrupt 处理）
    dashboard_ref = dashboard if log_mode == LOG_MODE_MONITOR else None

    def _on_signal(signum, frame):
        system_monitor.event_log.warn("System", f"收到信号 {signum}，正在优雅关闭…")
        if dashboard_ref is not None:
            try:
                dashboard_ref.stop()
            except Exception as _e:
                logger.debug("关闭仪表盘失败: %s", _e)
        # 关闭所有图谱的数据库连接
        for gid in registry.list_graphs():
            try:
                proc = registry.get_processor(gid)
                if hasattr(proc, 'storage') and hasattr(proc.storage, 'close'):
                    proc.storage.close()
            except Exception as _e:
                logger.debug("关闭 graph %s 存储失败: %s", gid, _e)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_signal)

    @atexit.register
    def _cleanup():
        if dashboard_ref is not None:
            try:
                dashboard_ref.stop()
            except Exception as _e:
                logger.debug("atexit 关闭仪表盘失败: %s", _e)
        for gid in registry.list_graphs():
            try:
                proc = registry.get_processor(gid)
                if hasattr(proc, 'storage') and hasattr(proc.storage, 'close'):
                    proc.storage.close()
            except Exception as _e:
                logger.debug("atexit 关闭 graph %s 存储失败: %s", gid, _e)

    try:
        app.run(host=host, port=listen_port, debug=args.debug, threaded=threaded)
    except OSError as e:
        system_monitor.event_log.error("System", f"错误：HTTP 服务启动失败: {e}")
        if e.errno == errno.EADDRINUSE:
            system_monitor.event_log.error("System", "  端口在探测后仍被占用（竞态），请重试或更换端口。")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
