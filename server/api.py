"""
DeepDream 自然语言记忆图 API（多图谱模式）

一个以自然语言为核心的统一记忆图服务。系统只有两个核心职责：
  - Remember：接收自然语言文本或文档，自动构建概念实体/关系图。
  - Find：通过语义检索从总图中唤醒相关的局部记忆区域。

多图谱模式：支持按 graph_id 隔离不同知识图谱，所有 API 请求需带 graph_id 参数。
系统不负责 select，外部智能体根据 find 结果自行决策。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import atexit
import errno
import json
import logging
import os

logger = logging.getLogger(__name__)
import queue
import signal
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from flask import Flask, abort, jsonify, make_response, request, send_from_directory
from werkzeug.exceptions import NotFound

from server.config import load_config, merge_llm_alignment, resolve_embedding_model
from server.monitor import LOG_MODE_DETAIL, LOG_MODE_MONITOR, SystemMonitor
from server.task_queue import RememberTask, RememberTaskQueue
from server.registry import GraphRegistry
from server.sse import sse_response, queue_to_generator
from processor import TemporalMemoryGraphProcessor
from processor.llm.client import LLM_PRIORITY_STEP6
from processor.models import Entity, Episode, Relation
from processor.search.hybrid import HybridSearcher
from processor.search.graph_traversal import GraphTraversalSearcher
from processor.perf import _perf_timer


# ---------------------------------------------------------------------------
# 文件读取工具
# ---------------------------------------------------------------------------

_TEXT_EXTENSIONS = {".txt", ".md", ".text", ".log", ".csv", ".json", ".xml",
                   ".yaml", ".yml", ".ini", ".conf", ".cfg", ".rst", ".html"}


def _read_file_content(path: str) -> str:
    """读取文件内容为纯文本。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    ext = p.suffix.lower()

    if ext in _TEXT_EXTENSIONS or ext == "":
        return p.read_text(encoding="utf-8")

    return p.read_text(encoding="utf-8")


def entity_to_dict(e: Entity, max_content_length: int = 2000) -> Dict[str, Any]:
    from processor.content_schema import parse_markdown_sections
    sections = parse_markdown_sections(e.content) if e.content else {}
    content = e.content or ""
    truncated = len(content) > max_content_length
    content_display = content[:max_content_length] + ("..." if truncated else "")
    return {
        "id": e.absolute_id,  # 向后兼容
        "absolute_id": e.absolute_id,
        "family_id": e.family_id,
        "name": e.name,
        "content": content_display,
        "content_truncated": truncated,
        "content_format": getattr(e, "content_format", "plain"),
        "content_sections": sections if sections else None,
        "event_time": e.event_time.isoformat() if e.event_time else None,
        "processed_time": e.processed_time.isoformat() if e.processed_time else None,
        "episode_id": e.episode_id,
        "source_document": getattr(e, "source_document", "") or getattr(e, "doc_name", "") or "",
        "doc_name": getattr(e, "source_document", "") or getattr(e, "doc_name", "") or "",
        "summary": getattr(e, "summary", None),
        "attributes": getattr(e, "attributes", None),
        "confidence": getattr(e, "confidence", None),
        "community_id": getattr(e, "community_id", None),
    }


def relation_to_dict(r: Relation) -> Dict[str, Any]:
    return {
        "id": r.absolute_id,  # 向后兼容
        "absolute_id": r.absolute_id,
        "family_id": r.family_id,
        "entity1_absolute_id": r.entity1_absolute_id,
        "entity2_absolute_id": r.entity2_absolute_id,
        "content": r.content,
        "event_time": r.event_time.isoformat() if r.event_time else None,
        "processed_time": r.processed_time.isoformat() if r.processed_time else None,
        "episode_id": r.episode_id,
        "source_document": getattr(r, "source_document", "") or getattr(r, "doc_name", "") or "",
        "doc_name": getattr(r, "source_document", "") or getattr(r, "doc_name", "") or "",
        "summary": getattr(r, "summary", None),
        "attributes": getattr(r, "attributes", None),
        "confidence": getattr(r, "confidence", None),
        "provenance": getattr(r, "provenance", None),
        "content_format": getattr(r, "content_format", "plain"),
    }


def enrich_relations(relations_dicts, processor):
    """为关系列表补充 entity1_name / entity2_name"""
    abs_ids = set()
    for rd in relations_dicts:
        if rd.get('entity1_absolute_id'):
            abs_ids.add(rd['entity1_absolute_id'])
        if rd.get('entity2_absolute_id'):
            abs_ids.add(rd['entity2_absolute_id'])
    if not abs_ids:
        return relations_dicts
    name_map = processor.storage.get_entity_names_by_absolute_ids(list(abs_ids))
    for rd in relations_dicts:
        rd['entity1_name'] = name_map.get(rd.get('entity1_absolute_id'), '')
        rd['entity2_name'] = name_map.get(rd.get('entity2_absolute_id'), '')
    return relations_dicts


def episode_to_dict(c: Episode) -> Dict[str, Any]:
    return {
        "id": c.absolute_id,  # 向后兼容
        "absolute_id": c.absolute_id,
        "content": c.content,
        "event_time": c.event_time.isoformat() if c.event_time else None,
        "source_document": getattr(c, "source_document", "") or getattr(c, "doc_name", "") or "",
        "doc_name": getattr(c, "source_document", "") or getattr(c, "doc_name", "") or "",
        "activity_type": getattr(c, "activity_type", None),
        "episode_type": getattr(c, "episode_type", None),
    }


def ok(data: Any) -> tuple:
    out: Dict[str, Any] = {"success": True, "data": data}
    try:
        if hasattr(request, "start_time"):
            out["elapsed_ms"] = round((time.time() - request.start_time) * 1000, 2)
    except RuntimeError:
        pass
    return jsonify(out), 200


def err(message: str, status: int = 400) -> tuple:
    if status >= 500:
        logger.error("API error (%d): %s", status, message, exc_info=True)
    out: Dict[str, Any] = {"success": False, "error": message}
    try:
        if hasattr(request, "start_time"):
            out["elapsed_ms"] = round((time.time() - request.start_time) * 1000, 2)
    except RuntimeError:
        pass
    return jsonify(out), status


def _extract_candidate_ids(
    storage: Any,
    body: Dict[str, Any],
    parse_time_point: Any,
) -> Tuple[Set[str], Set[str]]:
    """按 query_text / 时间等条件从主图抽取实体与关系的 absolute id 集合（供 query-one 等接口使用）。"""
    entity_absolute_ids: Set[str] = set()
    relation_absolute_ids: Set[str] = set()
    time_before = body.get("time_before")
    time_after = body.get("time_after")
    max_entities = body.get("max_entities")
    if max_entities is None:
        max_entities = 100
    max_relations = body.get("max_relations")
    if max_relations is None:
        max_relations = 500
    time_before_dt = parse_time_point(time_before) if time_before else None
    time_after_dt = parse_time_point(time_after) if time_after else None

    entity_name = (body.get("entity_name") or body.get("query_text") or "").strip()
    with _perf_timer("_extract_candidate_ids | entity_search"):
        if entity_name:
            entities = storage.search_entities_by_similarity(
                query_name=entity_name,
                query_content=body.get("query_text") or entity_name,
                threshold=float(body.get("similarity_threshold", 0.5)),
                max_results=int(max_entities),
                text_mode=body.get("text_mode") or "name_and_content",
                similarity_method=body.get("similarity_method") or "embedding",
            )
            for e in entities:
                entity_absolute_ids.add(e.absolute_id)
        elif time_before_dt:
            entities = storage.get_all_entities_before_time(time_before_dt, limit=max_entities, exclude_embedding=True)
            for e in entities:
                entity_absolute_ids.add(e.absolute_id)
        else:
            entities = storage.get_all_entities(limit=max_entities, exclude_embedding=True)
            for e in entities:
                entity_absolute_ids.add(e.absolute_id)

    if not entity_absolute_ids:
        return entity_absolute_ids, relation_absolute_ids

    with _perf_timer("_extract_candidate_ids | relation_search"):
        relations = storage.get_relations_by_entity_absolute_ids(
            list(entity_absolute_ids), limit=max_relations
        )
        for r in relations:
            if time_before_dt and r.event_time and r.event_time > time_before_dt:
                continue
            if time_after_dt and r.event_time and r.event_time < time_after_dt:
                continue
            relation_absolute_ids.add(r.absolute_id)

    drop_entities = set()
    with _perf_timer("_extract_candidate_ids | time_filter"):
        if (time_before_dt or time_after_dt) and hasattr(storage, 'get_entities_by_absolute_ids'):
            batch_entities = storage.get_entities_by_absolute_ids(list(entity_absolute_ids))
            for e in batch_entities:
                if e and e.event_time:
                    if time_before_dt and e.event_time > time_before_dt:
                        drop_entities.add(e.absolute_id)
                    elif time_after_dt and e.event_time < time_after_dt:
                        drop_entities.add(e.absolute_id)
        else:
            for eid in entity_absolute_ids:
                e = storage.get_entity_by_absolute_id(eid)
                if e and e.event_time:
                    if time_before_dt and e.event_time > time_before_dt:
                        drop_entities.add(eid)
                    elif time_after_dt and e.event_time < time_after_dt:
                        drop_entities.add(eid)
    entity_absolute_ids -= drop_entities
    return entity_absolute_ids, relation_absolute_ids


def create_app(
    registry,
    config: Optional[Dict[str, Any]] = None,
    system_monitor: Optional[SystemMonitor] = None,
) -> Flask:
    static_dir = Path(__file__).resolve().parent / "static"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="/static")
    app.json.ensure_ascii = False
    app.config["system_monitor"] = system_monitor

    # CORS：仅允许同源和 localhost 跨域调用
    _ALLOWED_ORIGINS = {"http://localhost", "http://127.0.0.1"}
    @app.after_request
    def _cors_headers(response):
        origin = request.environ.get("HTTP_ORIGIN")
        if origin and any(origin.startswith(allowed) for allowed in _ALLOWED_ORIGINS):
            response.headers["Access-Control-Allow-Origin"] = origin
        else:
            response.headers["Access-Control-Allow-Origin"] = ""
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
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
        request.start_time = time.time()
        request._monitor_start = time.time()

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

    # 不需要 graph_id 的路由（白名单）
    _NO_GRAPH_ID_ROUTES = {"/api/v1/graphs", "/api/v1/routes"}
    # 系统 API 前缀（不需要 graph_id）
    _SYSTEM_API_PREFIX = "/api/v1/system/"

    # 简单内存限流（按 IP，滑动窗口）
    _rate_limit_store: Dict[str, List[float]] = {}
    _rate_limit_lock = threading.Lock()
    _RATE_LIMIT = int(config.get("rate_limit_per_minute", 0))
    _RATE_WINDOW = 60.0  # 秒

    @app.before_request
    def _resolve_graph_id():
        """在请求进入端点前解析 graph_id 并挂到 request.graph_id 上。
        不需要 graph_id 的路由跳过。未填时默认使用 'default'。"""
        path = request.path
        if path in _NO_GRAPH_ID_ROUTES or path.startswith(_SYSTEM_API_PREFIX):
            return
        gid = ""
        if request.method == "POST":
            body = request.get_json(silent=True)
            if isinstance(body, dict):
                gid = (body.get("graph_id") or "").strip()
        if not gid:
            gid = (request.form.get("graph_id") or "").strip()
        if not gid:
            gid = (request.args.get("graph_id") or "").strip()
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
        client_ip = request.remote_addr or "unknown"
        with _rate_limit_lock:
            timestamps = _rate_limit_store.get(client_ip)
            if timestamps is None:
                _rate_limit_store[client_ip] = [now]
                return
            # 清理过期时间戳
            cutoff = now - _RATE_WINDOW
            timestamps = [t for t in timestamps if t > cutoff]
            timestamps.append(now)
            _rate_limit_store[client_ip] = timestamps
            # 定期清理长时间不活跃的 IP（超过窗口 2 倍时间未活跃则移除）
            if len(_rate_limit_store) > 1000:
                stale_ips = [ip for ip, ts in _rate_limit_store.items()
                             if not ts or ts[-1] < cutoff]
                for ip in stale_ips:
                    del _rate_limit_store[ip]
            if len(timestamps) > _RATE_LIMIT:
                return jsonify({"success": False, "error": "请求过于频繁，请稍后再试"}), 429

    def _get_graph_id() -> str:
        """获取当前请求的 graph_id（由 before_request 解析）。"""
        return request.graph_id

    def _get_processor():
        """获取当前请求对应的 Processor。"""
        return registry.get_processor(request.graph_id)

    def _get_queue():
        """获取当前请求对应的 RememberTaskQueue。"""
        return registry.get_queue(request.graph_id)

    def parse_time_point(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("time_point 需为 ISO 格式")

    def _normalize_time_for_compare(value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def _parse_non_negative_seconds(name: str) -> Optional[float]:
        raw = (request.args.get(name) or "").strip()
        if not raw:
            return None
        try:
            seconds = float(raw)
        except ValueError:
            raise ValueError(f"{name} 需为非负数字（秒）")
        if seconds < 0:
            raise ValueError(f"{name} 需为非负数字（秒）")
        return seconds

    def _score_entity_versions_against_time(family_id: str, time_point: datetime, proc=None) -> List[Tuple[float, int, Entity]]:
        if proc is None:
            proc = _get_processor()
        target = _normalize_time_for_compare(time_point)
        scored: List[Tuple[float, int, Entity]] = []
        for version in proc.storage.get_entity_versions(family_id):
            if not version.event_time:
                continue
            vt = _normalize_time_for_compare(version.event_time)
            delta_seconds = abs((vt - target).total_seconds())
            direction_bias = 0 if vt <= target else 1
            scored.append((delta_seconds, direction_bias, version))
        scored.sort(key=lambda item: (item[0], item[1], -_normalize_time_for_compare(item[2].processed_time).timestamp()))
        return scored

    # 向后兼容：/api/<path> → /api/v1/<path>（308 永久重定向）
    @app.route("/api/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE"])
    def _api_redirect(subpath):
        from flask import redirect as flask_redirect
        return flask_redirect(f"/api/v1/{subpath}", code=308)

    @app.route("/api")
    def _api_root_redirect():
        from flask import redirect as flask_redirect
        return flask_redirect("/api/v1/", code=308)

    @app.route("/api/v1/health", methods=["GET"])
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

    @app.route("/api/v1/health/llm", methods=["GET"])
    def health_llm():
        """检查大模型是否可访问；推荐使用 /api/v1/health/llm。"""
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

    def _parse_bool_query(name: str) -> Optional[bool]:
        v = request.args.get(name)
        if v is None or v == "":
            return None
        s = v.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        return None

    @app.route("/api/v1/remember", methods=["POST"])
    def remember():
        """记忆写入：POST 请求发起异步任务，入队后立即返回 task_id。

        输入方式（三选一）：
          - JSON body 的 text 字段（适合短文本）
          - multipart/form-data 的 file 字段（适合长文本/文件上传）
          - JSON body 的 file_path 字段（仅限服务端本机文件）

        参数：
          - graph_id（可选）：目标图谱 ID，默认 "default"
          - text（可选）：正文
          - file（可选）：上传文件（multipart）
          - source_name / doc_name / source_document（可选）：来源名称，默认 api_input
          - load_cache_memory（可选）：
            true = 接续图谱中已有缓存链（同图任务需串行）
            false = 不接续外部缓存链，但任务内部滑窗仍续写自己的 cache 链（可并行）
          - event_time（可选）：ISO 8601 事件时间
          - wait（可选）：true 时同步等待完成再返回（默认 false，异步返回 202）
          - timeout（可选）：同步等待超时秒数（默认 300，仅 wait=true 时生效）

        返回：
          - wait=false（默认）：HTTP 202 + task_id（异步轮询模式）
          - wait=true：HTTP 200 + 完整结果（同步阻塞模式，适合 Agent 单次调用）
        """
        try:
            processor = _get_processor()
            remember_queue = _get_queue()
            post_json: Dict[str, Any] = {}
            if request.method == "POST":
                pj = request.get_json(silent=True)
                if isinstance(pj, dict):
                    post_json = pj

            def _remember_get_str(name: str) -> str:
                if name in post_json and post_json[name] is not None:
                    v = post_json[name]
                    return (v if isinstance(v, str) else str(v)).strip()
                if request.method == "POST" and request.form and name in request.form:
                    return (request.form.get(name) or "").strip()
                return (request.args.get(name) or "").strip()

            def _remember_get_bool(name: str) -> Optional[bool]:
                def _parse_bool_value(v: Any) -> Optional[bool]:
                    if isinstance(v, bool):
                        return v
                    if isinstance(v, int) and v in (0, 1):
                        return bool(v)
                    if isinstance(v, str):
                        s = v.strip().lower()
                        if s in ("1", "true", "yes", "on"):
                            return True
                        if s in ("0", "false", "no", "off"):
                            return False
                    return None

                if name in post_json:
                    parsed = _parse_bool_value(post_json[name])
                    if parsed is not None:
                        return parsed
                if request.method == "POST" and request.form and name in request.form:
                    parsed = _parse_bool_value(request.form.get(name))
                    if parsed is not None:
                        return parsed
                return _parse_bool_query(name)

            text = _remember_get_str("text")

            # 如果 text 为空，尝试从 multipart 上传文件读取
            if not text and request.files:
                file = request.files.get("file")
                if file and file.filename:
                    text = file.read().decode("utf-8")

            if not text:
                return err("缺少 text 或 file（必填其一）", 400)

            sn = _remember_get_str("source_name")
            dn = _remember_get_str("doc_name")
            sd = _remember_get_str("source_document")
            # 如果从文件上传且未指定 source_name，用文件名
            if request.files and request.files.get("file") and request.files["file"].filename:
                if not sn and not dn and not sd:
                    sn = request.files["file"].filename
            source_name = (sn or sd or dn or "api_input")
            load_cache = _remember_get_bool("load_cache_memory")
            if load_cache is None:
                # 任务入队时就固化默认值，避免服务重启或配置变更后语义漂移。
                load_cache = bool(getattr(processor, "load_cache_memory", False))

            # 以“首次接收请求的时间”为基准：若未传 event_time，则使用当前接收时间并持久化到 journal。
            receive_time = datetime.now()
            event_time: Optional[datetime] = receive_time
            et_str = _remember_get_str("event_time") or None
            if et_str:
                try:
                    event_time = datetime.fromisoformat(et_str.replace("Z", "+00:00"))
                except ValueError:
                    return err("event_time 需为 ISO 8601 格式", 400)

            # 原文由处理器在 save_episode 阶段保存到 docs/{timestamp}_{hash}/original.txt
            # 此处不再额外保存扁平文件，避免重复
            original_path = ""

            preview = (text[:80] + "…") if len(text) > 80 else text
            event_time_display = event_time.isoformat() if event_time else "未指定"
            if system_monitor is not None:
                system_monitor.event_log.info(
                    "Remember",
                    f"收到({request.method}): source_name={source_name!r}, "
                    f"文本长度={len(text)} 字符, event_time={event_time_display}"
                )
                if system_monitor.mode == LOG_MODE_DETAIL:
                    system_monitor.event_log.info("Remember", f"内容预览: {preview!r}")
            else:
                print(
                    f"[Remember] 收到({request.method}): source_name={source_name!r}, "
                    f"文本长度={len(text)} 字符, event_time={event_time_display}"
                )

            task_id = uuid.uuid4().hex
            task = RememberTask(
                task_id=task_id,
                text=text,
                source_name=source_name,
                load_cache=load_cache,
                control_action=None,
                event_time=event_time,
                original_path=original_path,
            )

            remember_queue.submit(task)

            # Synchronous wait mode: block until task completes, then return full result
            wait_mode = _remember_get_bool("wait")
            if wait_mode:
                timeout = 300
                timeout_str = _remember_get_str("timeout")
                if timeout_str:
                    try:
                        timeout = max(10, min(3600, float(timeout_str)))
                    except ValueError:
                        pass
                done_task = remember_queue.wait_for_task(task_id, timeout=timeout)
                if done_task is None:
                    return err(f"任务 {task_id} 未找到", 404)
                task_dict = remember_queue._task_to_dict(done_task)
                if done_task.status == "completed":
                    return make_response(jsonify({
                        "success": True,
                        "data": {
                            "task_id": task_id,
                            "status": "completed",
                            "result": done_task.result,
                            **task_dict,
                        },
                    }), 200)
                elif done_task.status == "failed":
                    return make_response(jsonify({
                        "success": False,
                        "data": {
                            "task_id": task_id,
                            "status": "failed",
                            "error": done_task.error,
                            **task_dict,
                        },
                    }), 500)
                else:
                    # Timeout: still running, return current state with 202
                    return make_response(jsonify({
                        "success": True,
                        "data": {
                            "task_id": task_id,
                            "status": done_task.status,
                            "message": f"同步等待超时（{timeout}秒），任务仍在处理中。GET /api/v1/remember/tasks/{task_id} 继续轮询",
                            **task_dict,
                        },
                    }), 202)

            # Default async mode: return 202 immediately
            return make_response(jsonify({
                "success": True,
                "data": {
                    "task_id": task_id,
                    "status": "queued",
                    "message": "已加入队列；Find 与 Remember 可并发。崩溃重启后未完成任务会从 journal 恢复。GET /api/v1/remember/tasks/<task_id> 查询进度",
                    "original_path": original_path,
                },
            }), 202)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/remember/tasks/<task_id>", methods=["GET", "DELETE"])
    def remember_status(task_id: str):
        """查询或删除异步记忆写入任务；推荐使用 /api/v1/remember/tasks/<task_id>。"""
        try:
            remember_queue = _get_queue()
            if request.method == "DELETE":
                deleted, message, status = remember_queue.request_delete_task(task_id)
                if not deleted:
                    if message == "任务不存在":
                        return err(message, 404)
                    return err(message, 409)
                return ok({
                    "task_id": task_id,
                    "status": status,
                    "message": message,
                })
            t = remember_queue.get_status(task_id)
            if t is None:
                return err("任务不存在", 404)
            data: Dict[str, Any] = remember_queue._task_to_dict(t)
            data["original_path"] = t.original_path
            if t.status == "completed" and t.result:
                data["result"] = t.result
            if t.status == "failed" and t.error:
                data["error"] = t.error
            return ok(data)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/remember/tasks/<task_id>/pause", methods=["POST"])
    def remember_pause(task_id: str):
        try:
            remember_queue = _get_queue()
            ok_pause, message, status = remember_queue.request_pause_task(task_id)
            if not ok_pause:
                if message == "任务不存在":
                    return err(message, 404)
                return err(message, 409)
            return ok({
                "task_id": task_id,
                "status": status,
                "message": message,
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/remember/tasks/<task_id>/resume", methods=["POST"])
    def remember_resume(task_id: str):
        try:
            remember_queue = _get_queue()
            ok_resume, message, status = remember_queue.resume_task(task_id)
            if not ok_resume:
                if message == "任务不存在":
                    return err(message, 404)
                return err(message, 409)
            return ok({
                "task_id": task_id,
                "status": status,
                "message": message,
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/remember/tasks", methods=["GET"])
    def remember_queue_list():
        """查看记忆写入任务队列；推荐使用 /api/v1/remember/tasks。"""
        try:
            remember_queue = _get_queue()
            limit = min(request.args.get("limit", 50, type=int), 200)
            tasks = remember_queue.list_tasks(limit=limit)
            return ok({"tasks": tasks, "count": len(tasks)})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/remember/monitor", methods=["GET"])
    def remember_monitor():
        """返回 remember 的实时监控快照，适合 watch 或外部面板轮询。"""
        try:
            detail = system_monitor.graph_detail(request.graph_id) if system_monitor else None
            if detail is None:
                remember_queue = _get_queue()
                limit = request.args.get("limit", 6, type=int)
                return ok({
                    "graph_id": request.graph_id,
                    "queue": remember_queue.get_monitor_snapshot(limit=limit),
                })
            return ok({
                "graph_id": request.graph_id,
                "storage": detail["storage"],
                "queue": detail["queue"],
                "threads": detail["threads"],
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/routes", methods=["GET"])
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
                    "summary": "按文本搜索实体；POST 推荐 JSON body，GET 适合简单调试",
                    "body_or_query": {
                        "query_name": "string，必填",
                        "query_content": "string，可选",
                        "similarity_threshold": "float，可选（兼容旧名 threshold）",
                        "max_results": "int，可选",
                        "text_mode": "name_only | content_only | name_and_content",
                        "similarity_method": "embedding | text | jaccard | bleu",
                    },
                },
                {
                    "path": "/api/v1/find/relations/search",
                    "methods": ["GET", "POST"],
                    "summary": "按文本搜索关系；POST 推荐 JSON body，GET 适合简单调试",
                    "body_or_query": {
                        "query_text": "string，必填",
                        "similarity_threshold": "float，可选（兼容旧名 threshold）",
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
                        "max_delta_seconds": "float，可选；超出该误差则返回 404",
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
                    "path": "/api/v1/find/entities/absolute/<absolute_id>/relations",
                    "methods": ["GET"],
                    "summary": "按实体 absolute_id 查询相关关系",
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
                    "path": "/api/v1/find/relations/by-entity-absolute-id/<entity_absolute_id>",
                    "methods": ["GET"],
                    "summary": "按实体 absolute_id 查询相关关系",
                    "aliases": ["/api/v1/find/entities/absolute/<absolute_id>/relations"],
                },
                {
                    "path": "/api/v1/find/relations/by-entity-id/<family_id>",
                    "methods": ["GET"],
                    "summary": "按实体业务 ID 查询相关关系",
                    "aliases": ["/api/v1/find/entities/<family_id>/relations"],
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
                    "summary": "单图谱详细状态（存储+队列+线程）",
                },
                {
                    "path": "/api/v1/system/tasks",
                    "methods": ["GET"],
                    "summary": "所有图谱的任务列表",
                    "query": {"limit": "int，可选，默认 50"},
                },
                {
                    "path": "/api/v1/system/logs",
                    "methods": ["GET"],
                    "summary": "最近系统日志",
                    "query": {"limit": "int，可选", "level": "INFO/WARN/ERROR，可选", "source": "string，可选"},
                },
                {
                    "path": "/api/v1/system/access-stats",
                    "methods": ["GET"],
                    "summary": "API 访问统计",
                    "query": {"since_seconds": "float，可选，默认 300"},
                },
            ],
            "entity_v3": [
                {
                    "path": "/api/v1/find/entities/<family_id>/evolve-summary",
                    "methods": ["POST"],
                    "summary": "LLM 演化实体摘要（Phase A）",
                },
                {
                    "path": "/api/v1/find/entities/<family_id>/contradictions",
                    "methods": ["GET"],
                    "summary": "检测实体版本间矛盾（Phase D）",
                },
                {
                    "path": "/api/v1/find/entities/<family_id>/resolve-contradiction",
                    "methods": ["POST"],
                    "summary": "LLM 裁决实体矛盾（Phase D）",
                    "body": {"contradiction_id": "string，必填"},
                },
                {
                    "path": "/api/v1/find/entities/<family_id>/provenance",
                    "methods": ["GET"],
                    "summary": "实体事实溯源 - 返回提及该实体的 Episode（Phase C）",
                },
            ],
            "search_v3": [
                {
                    "path": "/api/v1/find/traverse",
                    "methods": ["POST"],
                    "summary": "BFS 图遍历搜索（Phase B）",
                    "body": {
                        "seed_family_ids": "list[string]，必填",
                        "max_depth": "int，可选，默认 3",
                        "max_nodes": "int，可选，默认 100",
                    },
                },
            ],
            "episode_v3": [
                {
                    "path": "/api/v1/find/episodes/<uuid>",
                    "methods": ["GET", "DELETE"],
                    "summary": "获取/删除 Episode 详情（Phase C）",
                },
                {
                    "path": "/api/v1/find/episodes/search",
                    "methods": ["POST"],
                    "summary": "搜索 Episode（Phase C）",
                    "body": {"query": "string，必填", "limit": "int，可选"},
                },
                {
                    "path": "/api/v1/find/episodes/batch-ingest",
                    "methods": ["POST"],
                    "summary": "批量导入 Episode（Phase C）",
                    "body": {"episodes": "list[{content, source_document, episode_type}]，必填"},
                },
            ],
            "butler": [
                {
                    "path": "/api/v1/butler/report",
                    "methods": ["GET"],
                    "summary": "管家报告：一次调用获取图谱健康 + 质量 + 梦境状态 + 推荐操作",
                    "query": {"graph_id": "string，必填"},
                },
                {
                    "path": "/api/v1/butler/execute",
                    "methods": ["POST"],
                    "summary": "管家执行：一键执行推荐操作（清理/社区检测/摘要进化）",
                    "body": {
                        "graph_id": "string，必填",
                        "actions": "list[string]，必填，可选: cleanup_isolated, cleanup_invalidated, detect_communities, evolve_summaries",
                        "dry_run": "bool，可选，默认 false",
                    },
                },
            ],
            "dream": [
                {
                    "path": "/api/v1/find/dream/status",
                    "methods": ["GET"],
                    "summary": "查询 DeepDream 当前状态",
                },
                {
                    "path": "/api/v1/find/dream/logs",
                    "methods": ["GET"],
                    "summary": "获取 DeepDream 历史日志（Phase E）",
                    "query": {"limit": "int，可选，默认 20"},
                },
                {
                    "path": "/api/v1/find/dream/logs/<cycle_id>",
                    "methods": ["GET"],
                    "summary": "获取单次 DeepDream 日志详情（Phase E）",
                },
            ],
            "agent": [
                {
                    "path": "/api/v1/find/ask",
                    "methods": ["POST"],
                    "summary": "Agent 元查询 - 自然语言问答（Phase F）",
                    "body": {"question": "string，必填"},
                },
                {
                    "path": "/api/v1/find/explain",
                    "methods": ["POST"],
                    "summary": "LLM 解释实体（Phase F）",
                    "body": {"family_id": "string，必填", "aspect": "summary | relations | timeline | contradictions"},
                },
                {
                    "path": "/api/v1/find/suggestions",
                    "methods": ["GET"],
                    "summary": "智能建议 - 分析图谱返回改进建议（Phase F）",
                },
            ],
        })

    # =========================================================
    # System: 系统监控 API（无需 graph_id）
    # =========================================================
    @app.route("/api/v1/system/dashboard", methods=["GET"])
    def system_dashboard():
        """仪表盘合并端点：一次返回 overview、graphs、tasks、logs、access-stats。"""
        try:
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

    @app.route("/api/v1/system/overview", methods=["GET"])
    def system_overview():
        """系统总览：图谱数量、运行时间、线程数。"""
        try:
            if system_monitor is None:
                return err("SystemMonitor 未启用", 503)
            return ok(system_monitor.overview())
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/system/graphs", methods=["GET"])
    def system_graphs():
        """所有图谱摘要列表。"""
        try:
            if system_monitor is None:
                return err("SystemMonitor 未启用", 503)
            return ok(system_monitor.all_graphs())
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/system/graphs/<graph_id>", methods=["GET"])
    def system_graph_detail(graph_id: str):
        """单图谱详细状态（存储+队列+线程）。"""
        try:
            if system_monitor is None:
                return err("SystemMonitor 未启用", 503)
            detail = system_monitor.graph_detail(graph_id)
            if detail is None:
                return err(f"图谱不存在: {graph_id}", 404)
            return ok(detail)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/system/tasks", methods=["GET"])
    def system_tasks():
        """所有图谱的任务列表。"""
        try:
            if system_monitor is None:
                return err("SystemMonitor 未启用", 503)
            limit = request.args.get("limit", 50, type=int)
            return ok(system_monitor.all_tasks(limit=limit))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/system/logs", methods=["GET"])
    def system_logs():
        """最近系统日志。支持 ?limit=&level=&source= 筛选。"""
        try:
            if system_monitor is None:
                return err("SystemMonitor 未启用", 503)
            limit = request.args.get("limit", 50, type=int)
            level = request.args.get("level")
            source = request.args.get("source")
            return ok(system_monitor.recent_logs(limit=limit, level=level, source=source))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/system/access-stats", methods=["GET"])
    def system_access_stats():
        """API 访问统计。支持 ?since_seconds= 指定统计周期（默认 300 秒）。"""
        try:
            if system_monitor is None:
                return err("SystemMonitor 未启用", 503)
            since = request.args.get("since_seconds", 300, type=float)
            return ok(system_monitor.access_stats(since_seconds=since))
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Find: 统计
    # =========================================================
    @app.route("/api/v1/find/stats", methods=["GET"])
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
                # 旧后端：从文件系统计数
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

    # =========================================================
    # Find: 统一语义检索入口（推荐）
    # =========================================================
    @app.route("/api/v1/find", methods=["POST"])
    def find_unified():
        """统一语义检索：用自然语言从总记忆图中唤醒相关的局部区域。

        请求体:
            query (str, 必填): 自然语言查询
            similarity_threshold (float): 语义相似度阈值，默认 0.5
            max_entities (int): 最大返回实体数，默认 20
            max_relations (int): 最大返回关系数，默认 50
            expand (bool): 是否从命中实体向外扩展邻域，默认 true
            time_before (str, ISO): 只返回此时间之前的记忆
            time_after (str, ISO): 只返回此时间之后的记忆

        返回:
            entities: 命中的概念实体列表
            relations: 命中的概念关系列表
        """
        try:
            body = request.get_json(silent=True) or {}
            query = (body.get("query") or "").strip()
            if not query:
                return err("query 为必填字段", 400)

            similarity_threshold = float(body.get("similarity_threshold", 0.5))
            max_entities = int(body.get("max_entities", 20))
            max_relations = int(body.get("max_relations", 50))
            expand = body.get("expand", True)
            time_before = body.get("time_before")
            time_after = body.get("time_after")
            reranker = str(body.get("reranker", "rrf") or "rrf").strip().lower()

            search_mode = str(body.get("search_mode", "semantic") or "semantic").strip().lower()
            if search_mode not in ("semantic", "bm25", "hybrid"):
                search_mode = "semantic"

            try:
                time_before_dt = parse_time_point(time_before) if time_before else None
                time_after_dt = parse_time_point(time_after) if time_after else None
            except ValueError as ve:
                return err(str(ve), 400)

            processor = _get_processor()
            storage = processor.storage

            # --- 第一步：按 search_mode 召回实体 ---
            with _perf_timer("find_unified | step1_entity_recall"):
                if search_mode == "bm25":
                    matched_entities = storage.search_entities_by_bm25(
                        query, limit=max_entities
                    )
                elif search_mode == "hybrid":
                    searcher = HybridSearcher(storage)
                    matched_entities = searcher.search_entities(
                        query_text=query,
                        top_k=max_entities,
                        semantic_threshold=similarity_threshold,
                    )
                else:
                    matched_entities = storage.search_entities_by_similarity(
                        query_name=query,
                        query_content=query,
                        threshold=similarity_threshold,
                        max_results=max_entities,
                        text_mode="name_and_content",
                        similarity_method="embedding",
                    )

            # --- 第二步：按 search_mode 召回关系 ---
            with _perf_timer("find_unified | step2_relation_recall"):
                if search_mode == "bm25":
                    matched_relations = storage.search_relations_by_bm25(
                        query, limit=max_relations
                    )
                elif search_mode == "hybrid":
                    searcher = HybridSearcher(storage)
                    matched_relations = searcher.search_relations(
                        query_text=query,
                        top_k=max_relations,
                        semantic_threshold=similarity_threshold,
                    )
                else:
                    matched_relations = storage.search_relations_by_similarity(
                        query_text=query,
                        threshold=similarity_threshold,
                        max_results=max_relations,
                    )

            entity_abs_ids: Set[str] = {e.absolute_id for e in matched_entities}
            relation_abs_ids: Set[str] = {r.absolute_id for r in matched_relations}
            entities_by_abs: Dict[str, Entity] = {e.absolute_id: e for e in matched_entities}

            # --- 第三步：从语义命中的关系中补充关联实体（批量获取） ---
            with _perf_timer("find_unified | step3_entity_completion"):
                missing_abs_ids = set()
                for r in list(matched_relations):
                    for abs_id in (r.entity1_absolute_id, r.entity2_absolute_id):
                        if abs_id not in entity_abs_ids:
                            missing_abs_ids.add(abs_id)
                if missing_abs_ids and hasattr(storage, 'get_entities_by_absolute_ids'):
                    batch_entities = storage.get_entities_by_absolute_ids(list(missing_abs_ids))
                    for e in batch_entities:
                        if e:
                            entities_by_abs[e.absolute_id] = e
                            entity_abs_ids.add(e.absolute_id)

            # --- 第四步：图谱邻域扩展 ---
            with _perf_timer("find_unified | step4_graph_expansion"):
                if expand and entity_abs_ids:
                    expanded_rels = storage.get_relations_by_entity_absolute_ids(
                        list(entity_abs_ids), limit=max_relations
                    )
                    # 批量获取扩展关系中的新实体
                    expand_missing = set()
                    for r in expanded_rels:
                        if r.absolute_id not in relation_abs_ids:
                            relation_abs_ids.add(r.absolute_id)
                            matched_relations.append(r)
                        for abs_id in (r.entity1_absolute_id, r.entity2_absolute_id):
                            if abs_id not in entity_abs_ids:
                                expand_missing.add(abs_id)
                    if expand_missing and hasattr(storage, 'get_entities_by_absolute_ids'):
                        batch_entities = storage.get_entities_by_absolute_ids(list(expand_missing))
                        for e in batch_entities:
                            if e:
                                entities_by_abs[e.absolute_id] = e
                                entity_abs_ids.add(e.absolute_id)

            # --- 第五步：时间过滤 ---
            final_entities: List[Entity] = []
            for e in entities_by_abs.values():
                if time_before_dt and e.event_time and e.event_time > time_before_dt:
                    continue
                if time_after_dt and e.event_time and e.event_time < time_after_dt:
                    continue
                final_entities.append(e)

            final_relations: List[Relation] = []
            seen_rel_ids: Set[str] = set()
            for r in matched_relations:
                if r.absolute_id in seen_rel_ids:
                    continue
                if time_before_dt and r.event_time and r.event_time > time_before_dt:
                    continue
                if time_after_dt and r.event_time and r.event_time < time_after_dt:
                    continue
                seen_rel_ids.add(r.absolute_id)
                final_relations.append(r)

            # --- 第六步：可选重排序 ---
            if reranker == "node_degree" and hasattr(storage, 'get_entity_degree'):
                degree_map = {}
                for e in final_entities:
                    degree_map[e.family_id] = storage.get_entity_degree(e.family_id)
                searcher_hybrid = HybridSearcher(storage)
                scored = [(e, 1.0) for e in final_entities]
                reranked = searcher_hybrid.node_degree_rerank(scored, degree_map)
                final_entities = [e for e, _ in reranked[:max_entities]]

            result: Dict[str, Any] = {
                "query": query,
                "entities": [entity_to_dict(e) for e in final_entities],
                "relations": [relation_to_dict(r) for r in final_relations],
                "entity_count": len(final_entities),
                "relation_count": len(final_relations),
            }
            enrich_relations(result["relations"], processor)
            return ok(result)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/candidates", methods=["POST"])
    def find_query_one():
        """按请求体条件一次性返回候选实体与关系；推荐路径 /api/v1/find/candidates。"""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            include_entities = body.get("include_entities", True)
            include_relations = body.get("include_relations", True)
            try:
                family_ids, relation_family_ids = _extract_candidate_ids(
                    processor.storage, body, parse_time_point
                )
            except ValueError as ve:
                return err(str(ve), 400)
            storage = processor.storage
            entities_data: List[Dict[str, Any]] = []
            relations_data: List[Dict[str, Any]] = []
            if include_entities:
                if hasattr(storage, 'get_entities_by_absolute_ids'):
                    batch = storage.get_entities_by_absolute_ids(list(family_ids))
                    entities_data = [entity_to_dict(e) for e in batch if e]
                else:
                    for eid in family_ids:
                        e = storage.get_entity_by_absolute_id(eid)
                        if e:
                            entities_data.append(entity_to_dict(e))
            if include_relations:
                if hasattr(storage, 'get_relations_by_entity_absolute_ids'):
                    batch_rels = storage.get_relations_by_entity_absolute_ids(list(relation_family_ids))
                    for r in batch_rels:
                        if r.absolute_id in relation_family_ids:
                            relations_data.append(relation_to_dict(r))
                else:
                    for rid in relation_family_ids:
                        r = storage.get_relation_by_absolute_id(rid)
                        if r:
                            relations_data.append(relation_to_dict(r))
            return ok({"entities": entities_data, "relations": relations_data})
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Stats: counts endpoint
    # =========================================================
    @app.route("/api/v1/stats/counts", methods=["GET"])
    def get_counts():
        try:
            processor = _get_processor()
            return ok({
                "entity_count": processor.storage.count_unique_entities(),
                "relation_count": processor.storage.count_unique_relations(),
            })
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Find: 实体原子接口
    # =========================================================
    @app.route("/api/v1/find/entities", methods=["GET"])
    def find_entities_all():
        try:
            processor = _get_processor()
            limit = min(request.args.get("limit", type=int) or 500, 500)
            offset = request.args.get("offset", type=int, default=0) or 0
            total = processor.storage.count_unique_entities()
            entities = processor.storage.get_all_entities(limit=limit, offset=offset if offset > 0 else None, exclude_embedding=True)
            return ok({
                "entities": [entity_to_dict(e) for e in entities],
                "total": total,
                "offset": offset,
                "limit": limit,
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/as-of-time", methods=["GET"])
    def find_entities_all_before_time():
        try:
            processor = _get_processor()
            time_point_str = request.args.get("time_point")
            if not time_point_str:
                return err("time_point 为必填参数（ISO 格式）", 400)
            try:
                time_point = parse_time_point(time_point_str)
            except ValueError as ve:
                return err(str(ve), 400)
            limit = request.args.get("limit", type=int)
            entities = processor.storage.get_all_entities_before_time(time_point, limit=limit, exclude_embedding=True)
            return ok([entity_to_dict(e) for e in entities])
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/version-counts", methods=["POST"])
    def find_entity_version_counts():
        try:
            processor = _get_processor()
            body = request.get_json(silent=True)
            if not isinstance(body, dict):
                body = {}
            family_ids = body.get("family_ids")
            if not family_ids or not isinstance(family_ids, list):
                return ok({})
            # Filter to only valid strings
            family_ids = [x for x in family_ids if isinstance(x, str)]
            if not family_ids:
                return ok({})
            counts = processor.storage.get_entity_version_counts(family_ids)
            return ok(counts)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/absolute/<absolute_id>/embedding-preview", methods=["GET"])
    def find_entity_embedding_preview(absolute_id: str):
        try:
            processor = _get_processor()
            num_values = request.args.get("num_values", type=int, default=5)
            preview = processor.storage.get_entity_embedding_preview(absolute_id, num_values=num_values)
            if preview is None:
                return err(f"未找到实体 embedding 或实体不存在: {absolute_id}", 404)
            return ok({"absolute_id": absolute_id, "values": preview})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/absolute/<absolute_id>", methods=["GET"])
    def find_entity_by_absolute_id(absolute_id: str):
        try:
            processor = _get_processor()
            entity = processor.storage.get_entity_by_absolute_id(absolute_id)
            if entity is None:
                return err(f"未找到实体版本: {absolute_id}", 404)
            return ok(entity_to_dict(entity))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/search", methods=["GET", "POST"])
    def find_entities_search():
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) if request.method == "POST" else None
            body = body if isinstance(body, dict) else {}

            def _get_value(name: str, default: Any = None) -> Any:
                if name in body and body[name] is not None:
                    return body[name]
                return request.args.get(name, default)

            query_name = str(_get_value("query_name", "") or "").strip()
            if not query_name:
                return err("query_name 为必填参数", 400)
            query_content = _get_value("query_content") or None
            threshold = float(_get_value("similarity_threshold") or _get_value("threshold", 0.5))
            max_results = int(_get_value("max_results", 10))
            text_mode = str(_get_value("text_mode", "name_and_content") or "name_and_content")
            if text_mode not in ("name_only", "content_only", "name_and_content"):
                text_mode = "name_and_content"
            similarity_method = str(_get_value("similarity_method", "embedding") or "embedding")
            if similarity_method not in ("embedding", "text", "jaccard", "bleu"):
                similarity_method = "embedding"
            content_snippet_length = int(_get_value("content_snippet_length", 50))

            search_mode = str(_get_value("search_mode", "semantic") or "semantic").strip().lower()
            if search_mode not in ("semantic", "bm25", "hybrid"):
                search_mode = "semantic"

            if search_mode == "bm25":
                entities = processor.storage.search_entities_by_bm25(
                    query_name, limit=max_results
                )
            elif search_mode == "hybrid":
                searcher = HybridSearcher(processor.storage)
                entities = searcher.search_entities(
                    query_text=query_name,
                    top_k=max_results,
                    semantic_threshold=threshold,
                )
            else:
                entities = processor.storage.search_entities_by_similarity(
                    query_name=query_name,
                    query_content=query_content,
                    threshold=threshold,
                    max_results=max_results,
                    content_snippet_length=content_snippet_length,
                    text_mode=text_mode,
                    similarity_method=similarity_method,
                )
            return ok([entity_to_dict(e) for e in entities])
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<family_id>/versions", methods=["GET"])
    def find_entity_versions(family_id: str):
        try:
            processor = _get_processor()
            versions = processor.storage.get_entity_versions(family_id)
            return ok([entity_to_dict(e) for e in versions])
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<family_id>/as-of-time", methods=["GET"])
    def find_entity_at_time(family_id: str):
        try:
            processor = _get_processor()
            time_point_str = request.args.get("time_point")
            if not time_point_str:
                return err("time_point 为必填参数（ISO 格式）", 400)
            try:
                time_point = parse_time_point(time_point_str)
            except ValueError as ve:
                return err(str(ve), 400)
            entity = processor.storage.get_entity_version_at_time(family_id, time_point)
            if entity is None:
                return err(f"未找到该时间点版本: {family_id}", 404)
            return ok(entity_to_dict(entity))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<family_id>/nearest-to-time", methods=["GET"])
    def find_entity_nearest_to_time(family_id: str):
        try:
            processor = _get_processor()
            time_point_str = request.args.get("time_point")
            if not time_point_str:
                return err("time_point 为必填参数（ISO 格式）", 400)
            try:
                time_point = parse_time_point(time_point_str)
                max_delta_seconds = _parse_non_negative_seconds("max_delta_seconds")
            except ValueError as ve:
                return err(str(ve), 400)

            scored = _score_entity_versions_against_time(family_id, time_point, proc=processor)
            if not scored:
                return err(f"未找到实体: {family_id}", 404)

            delta_seconds, _, entity = scored[0]
            if max_delta_seconds is not None and delta_seconds > max_delta_seconds:
                return err(f"最近版本超出允许误差: {delta_seconds:.3f}s > {max_delta_seconds:.3f}s", 404)

            return ok({
                "family_id": family_id,
                "query_time": time_point.isoformat(),
                "matched": entity_to_dict(entity),
                "delta_seconds": round(delta_seconds, 6),
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<family_id>/around-time", methods=["GET"])
    def find_entity_around_time(family_id: str):
        try:
            processor = _get_processor()
            time_point_str = request.args.get("time_point")
            if not time_point_str:
                return err("time_point 为必填参数（ISO 格式）", 400)
            try:
                time_point = parse_time_point(time_point_str)
                within_seconds = _parse_non_negative_seconds("within_seconds")
            except ValueError as ve:
                return err(str(ve), 400)
            if within_seconds is None:
                return err("within_seconds 为必填参数（秒）", 400)

            target = _normalize_time_for_compare(time_point)
            matches: List[Dict[str, Any]] = []
            for delta_seconds, _, entity in _score_entity_versions_against_time(family_id, time_point, proc=processor):
                if delta_seconds > within_seconds:
                    continue
                item = entity_to_dict(entity)
                item["delta_seconds"] = round(delta_seconds, 6)
                direction = _normalize_time_for_compare(entity.event_time) - target
                item["relative_position"] = "before_or_exact" if direction.total_seconds() <= 0 else "after"
                matches.append(item)

            if not matches:
                return err(f"未找到 {within_seconds:.3f} 秒范围内的实体版本: {family_id}", 404)

            return ok({
                "family_id": family_id,
                "query_time": time_point.isoformat(),
                "within_seconds": within_seconds,
                "count": len(matches),
                "matches": matches,
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<family_id>/version-count", methods=["GET"])
    def find_entity_version_count(family_id: str):
        try:
            processor = _get_processor()
            count = processor.storage.get_entity_version_count(family_id)
            if count <= 0:
                return err(f"未找到实体: {family_id}", 404)
            return ok({"family_id": family_id, "version_count": count})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<family_id>", methods=["GET"])
    def find_entity_by_family_id(family_id: str):
        try:
            processor = _get_processor()
            entity = processor.storage.get_entity_by_family_id(family_id)
            if entity is None:
                return err(f"未找到实体: {family_id}", 404)
            return ok(entity_to_dict(entity))
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Entity / Relation CRUD (Phase 2)
    # =========================================================

    @app.route("/api/v1/find/entities/<family_id>", methods=["DELETE"])
    def delete_entity_family(family_id: str):
        """删除实体所有版本。"""
        try:
            processor = _get_processor()
            cascade = request.args.get("cascade", "false").lower() == "true"
            count = processor.storage.delete_entity_all_versions(family_id)
            if count == 0:
                return err(f"未找到实体: {family_id}", 404)
            return ok({"message": f"已删除 {count} 个实体版本", "family_id": family_id, "cascade": cascade})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/batch-delete", methods=["POST"])
    def batch_delete_entities():
        """批量删除实体。"""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            family_ids = body.get("family_ids") or body.get("entity_ids", [])
            if not isinstance(family_ids, list) or not family_ids:
                return err("family_ids 需为非空数组", 400)
            if len(family_ids) > 100:
                return err("单次批量删除上限 100 个", 400)
            cascade = body.get("cascade", False)
            total = 0
            for eid in family_ids:
                total += processor.storage.delete_entity_all_versions(eid)
            return ok({"message": f"已删除 {total} 个实体版本", "count": len(family_ids)})
        except Exception as e:
            return err(str(e), 500)

    # ------------------------------------------------------------------
    # 管理类 API：孤立实体清理、数据质量报告
    # ------------------------------------------------------------------

    @app.route("/api/v1/find/entities/isolated", methods=["GET"])
    def find_isolated_entities():
        """列出所有孤立实体（无任何有效关系的有效实体）。仅 Neo4j 后端支持。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'get_isolated_entities'):
                return ok({"entities": [], "total": 0, "message": "当前存储后端不支持孤立实体检测"})
            limit = request.args.get("limit", type=int, default=100)
            offset = request.args.get("offset", type=int, default=0) or 0
            isolated = processor.storage.get_isolated_entities(limit=limit, offset=offset)
            total = processor.storage.count_isolated_entities()
            return ok({
                "entities": [entity_to_dict(e) for e in isolated],
                "total": total,
                "offset": offset,
                "limit": limit,
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/delete-isolated", methods=["POST"])
    def delete_isolated_entities():
        """批量删除所有孤立实体（无任何有效关系的有效实体）。仅 Neo4j 后端支持。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'get_isolated_entities'):
                return ok({"message": "当前存储后端不支持孤立实体检测", "deleted": 0})
            dry_run = request.get_json(silent=True) or {}
            dry_run = dry_run.get("dry_run", False) if isinstance(dry_run, dict) else False
            isolated = processor.storage.get_isolated_entities(limit=10000)
            if not isolated:
                return ok({"message": "没有孤立实体", "deleted": 0})
            family_ids = list({e.family_id for e in isolated if e.family_id})
            if dry_run:
                return ok({
                    "message": f"预览：将删除 {len(family_ids)} 个孤立实体",
                    "family_ids": family_ids,
                    "dry_run": True,
                })
            deleted = 0
            for fid in family_ids:
                deleted += processor.storage.delete_entity_all_versions(fid)
            return ok({
                "message": f"已删除 {len(family_ids)} 个孤立实体（{deleted} 个版本）",
                "deleted_families": len(family_ids),
                "deleted_versions": deleted,
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/quality-report", methods=["GET"])
    def quality_report():
        """数据质量报告。"""
        try:
            processor = _get_processor()
            stats = processor.storage.get_data_quality_report()
            return ok(stats)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/cleanup/invalidated-versions", methods=["POST"])
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

    @app.route("/api/v1/find/entities/merge", methods=["POST"])
    def merge_entities():
        """合并实体：将源实体合并到目标实体。"""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            target_id = (body.get("target_family_id") or "").strip()
            source_ids = body.get("source_family_ids", [])
            if not target_id or not isinstance(source_ids, list) or not source_ids:
                return err("target_family_id 和 source_family_ids 为必填", 400)
            # 验证 target 存在
            target = processor.storage.get_entity_by_family_id(target_id)
            if target is None:
                return err(f"目标实体不存在: {target_id}", 404)
            result = processor.storage.merge_entity_families(target_id, source_ids)
            return ok({"message": "实体合并完成", "target_family_id": target_id, "source_family_ids": source_ids, "merged_count": result})
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Find: 关系原子接口
    # =========================================================
    @app.route("/api/v1/find/relations", methods=["GET"])
    def find_relations_all():
        try:
            processor = _get_processor()
            limit = request.args.get("limit", type=int)
            offset = request.args.get("offset", type=int, default=0) or 0
            total = processor.storage.count_unique_relations()
            relations = processor.storage.get_all_relations(
                limit=limit, offset=offset if offset > 0 else None,
                exclude_embedding=True,
            )
            dicts = [relation_to_dict(r) for r in relations]
            enrich_relations(dicts, processor)
            return ok({
                "relations": dicts,
                "total": total,
                "offset": offset,
                "limit": limit,
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/relations/search", methods=["GET", "POST"])
    def find_relations_search():
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) if request.method == "POST" else None
            body = body if isinstance(body, dict) else {}

            def _get_value(name: str, default: Any = None) -> Any:
                if name in body and body[name] is not None:
                    return body[name]
                return request.args.get(name, default)

            query_text = str(_get_value("query_text", "") or "").strip()
            if not query_text:
                return err("query_text 为必填参数", 400)
            threshold = float(_get_value("similarity_threshold") or _get_value("threshold", 0.5))
            max_results = int(_get_value("max_results", 10))

            search_mode = str(_get_value("search_mode", "semantic") or "semantic").strip().lower()
            if search_mode not in ("semantic", "bm25", "hybrid"):
                search_mode = "semantic"

            if search_mode == "bm25":
                relations = processor.storage.search_relations_by_bm25(
                    query_text, limit=max_results
                )
            elif search_mode == "hybrid":
                searcher = HybridSearcher(processor.storage)
                relations = searcher.search_relations(
                    query_text=query_text,
                    top_k=max_results,
                    semantic_threshold=threshold,
                )
            else:
                relations = processor.storage.search_relations_by_similarity(
                    query_text=query_text,
                    threshold=threshold,
                    max_results=max_results,
                )
            dicts = [relation_to_dict(r) for r in relations]
            enrich_relations(dicts, processor)
            return ok(dicts)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/relations/between", methods=["GET", "POST"])
    def find_relations_between():
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) if request.method == "POST" else None
            body = body if isinstance(body, dict) else {}
            family_id_a = str(body.get("family_id_a") or body.get("from_family_id") or request.args.get("family_id_a") or request.args.get("from_family_id") or "").strip()
            family_id_b = str(body.get("family_id_b") or body.get("to_family_id") or request.args.get("family_id_b") or request.args.get("to_family_id") or "").strip()
            if not family_id_a or not family_id_b:
                return err("family_id_a 与 family_id_b 为必填参数", 400)
            with _perf_timer("find_relations_between"):
                relations = processor.storage.get_relations_by_entities(family_id_a, family_id_b)
            dicts = [relation_to_dict(r) for r in relations]
            enrich_relations(dicts, processor)
            return ok(dicts)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/paths/shortest", methods=["GET", "POST"])
    def find_shortest_paths():
        """查找两个实体之间的最短路径"""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) if request.method == "POST" else None
            body = body if isinstance(body, dict) else {}
            family_id_a = str(body.get("family_id_a") or body.get("from_family_id")
                             or request.args.get("family_id_a")
                             or request.args.get("from_family_id") or "").strip()
            family_id_b = str(body.get("family_id_b") or body.get("to_family_id")
                             or request.args.get("family_id_b")
                             or request.args.get("to_family_id") or "").strip()
            if not family_id_a or not family_id_b:
                return err("family_id_a 与 family_id_b 为必填参数", 400)

            max_depth = body.get("max_depth") if body else None
            if max_depth is None:
                max_depth = request.args.get("max_depth", type=int)
            max_depth = max_depth or 6

            max_paths = body.get("max_paths") if body else None
            if max_paths is None:
                max_paths = request.args.get("max_paths", type=int)
            max_paths = max_paths or 10

            result = processor.storage.find_shortest_paths(
                source_family_id=family_id_a,
                target_family_id=family_id_b,
                max_depth=max_depth,
                max_paths=max_paths,
            )

            serialized_paths = []
            for p in result.get("paths", []):
                serialized_paths.append({
                    "entities": [entity_to_dict(e) for e in p.get("entities", [])],
                    "relations": [relation_to_dict(r) for r in p.get("relations", [])],
                    "length": p.get("length", 0),
                })

            return ok({
                "source_entity": entity_to_dict(result["source_entity"]) if result.get("source_entity") else None,
                "target_entity": entity_to_dict(result["target_entity"]) if result.get("target_entity") else None,
                "path_length": result.get("path_length", -1),
                "total_shortest_paths": result.get("total_shortest_paths", 0),
                "paths": serialized_paths,
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/relations/absolute/<absolute_id>/embedding-preview", methods=["GET"])
    def find_relation_embedding_preview(absolute_id: str):
        try:
            processor = _get_processor()
            num_values = request.args.get("num_values", type=int, default=5)
            preview = processor.storage.get_relation_embedding_preview(absolute_id, num_values=num_values)
            if preview is None:
                return err(f"未找到关系 embedding 或关系不存在: {absolute_id}", 404)
            return ok({"absolute_id": absolute_id, "values": preview})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/relations/absolute/<absolute_id>", methods=["GET"])
    def find_relation_by_absolute_id(absolute_id: str):
        try:
            processor = _get_processor()
            relation = processor.storage.get_relation_by_absolute_id(absolute_id)
            if relation is None:
                return err(f"未找到关系版本: {absolute_id}", 404)
            d = relation_to_dict(relation)
            enrich_relations([d], processor)
            return ok(d)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/absolute/<entity_absolute_id>/relations", methods=["GET"])
    def find_relations_by_entity_absolute_id(entity_absolute_id: str):
        try:
            processor = _get_processor()
            limit = request.args.get("limit", type=int)
            time_point_str = request.args.get("time_point")
            try:
                time_point = parse_time_point(time_point_str)
            except ValueError as ve:
                return err(str(ve), 400)
            relations = processor.storage.get_entity_relations(
                entity_absolute_id=entity_absolute_id,
                limit=limit,
                time_point=time_point,
            )
            dicts = [relation_to_dict(r) for r in relations]
            enrich_relations(dicts, processor)
            return ok(dicts)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/relations/<family_id>/versions", methods=["GET"])
    def find_relation_versions(family_id: str):
        try:
            processor = _get_processor()
            versions = processor.storage.get_relation_versions(family_id)
            dicts = [relation_to_dict(r) for r in versions]
            enrich_relations(dicts, processor)
            return ok(dicts)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/relations/<family_id>", methods=["PUT"])
    def update_relation_by_family(family_id: str):
        """编辑关系：创建新版本。"""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            content = body.get("content")
            if not content:
                return err("content 为必填字段", 400)

            current_versions = processor.storage.get_relation_versions(family_id)
            if not current_versions:
                return err(f"未找到关系: {family_id}", 404)
            current = current_versions[0]  # 最新版本

            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            from processor.models import Relation as RelationModel
            updated = RelationModel(
                absolute_id=str(uuid.uuid4()),
                family_id=family_id,
                entity1_absolute_id=current.entity1_absolute_id,
                entity2_absolute_id=current.entity2_absolute_id,
                content=content,
                event_time=now,
                processed_time=now,
                episode_id=current.episode_id,
                source_document=current.source_document,
                valid_at=now,
            )
            processor.storage.save_relation(updated)
            return ok({"message": "关系已更新", "absolute_id": updated.absolute_id, "family_id": family_id})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/relations/<family_id>", methods=["DELETE"])
    def delete_relation_family(family_id: str):
        """删除关系所有版本。"""
        try:
            processor = _get_processor()
            count = processor.storage.delete_relation_all_versions(family_id)
            if count == 0:
                return err(f"未找到关系: {family_id}", 404)
            return ok({"message": f"已删除 {count} 个关系版本", "family_id": family_id})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/relations/batch-delete", methods=["POST"])
    def batch_delete_relations():
        """批量删除关系。"""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            family_ids = body.get("family_ids") or body.get("relation_ids", [])
            if not isinstance(family_ids, list) or not family_ids:
                return err("family_ids 需为非空数组", 400)
            if len(family_ids) > 100:
                return err("单次批量删除上限 100 个", 400)
            total = 0
            for rid in family_ids:
                total += processor.storage.delete_relation_all_versions(rid)
            return ok({"message": f"已删除 {total} 个关系版本", "count": len(family_ids)})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<family_id>/relations", methods=["GET"])
    def find_relations_by_entity(family_id: str):
        try:
            processor = _get_processor()
            limit = request.args.get("limit", type=int)
            time_point_str = request.args.get("time_point")
            try:
                time_point = parse_time_point(time_point_str)
            except ValueError as ve:
                return err(str(ve), 400)
            max_version_absolute_id = (request.args.get("max_version_absolute_id") or "").strip() or None
            relation_scope = (request.args.get("relation_scope") or "accumulated").strip()

            if relation_scope not in ("accumulated", "version_only", "all_versions"):
                relation_scope = "accumulated"

            # When no max_version_absolute_id, all modes degenerate to returning all relations
            if not max_version_absolute_id:
                relations = processor.storage.get_entity_relations_by_family_id(
                    family_id=family_id,
                    limit=limit,
                    time_point=time_point,
                    max_version_absolute_id=None,
                )
                dicts = [relation_to_dict(r) for r in relations]
                enrich_relations(dicts, processor)
                return ok(dicts)

            # ---- Shared queries ----
            # current_rels: relations directly linked to the focused version only
            current_rels = processor.storage.get_entity_relations(
                max_version_absolute_id,
                limit=limit,
                time_point=time_point,
            )
            # accum_rels: accumulated relations from v1 through focused version
            accum_rels = processor.storage.get_entity_relations_by_family_id(
                family_id=family_id,
                limit=limit,
                time_point=time_point,
                max_version_absolute_id=max_version_absolute_id,
            )

            # Dedup by family_id
            accum_by_rid = {r.family_id: r for r in accum_rels}
            current_by_rid = {r.family_id: r for r in current_rels}
            accum_rids = set(accum_by_rid)
            current_rids = set(current_by_rid)

            # ---- version_only: only relations directly linked to this version ----
            if relation_scope == "version_only":
                dicts = [relation_to_dict(r) for r in current_rels]
                enrich_relations(dicts, processor)
                return ok(dicts)

            # ---- accumulated: v1..vN union + future from latest ----
            if relation_scope == "accumulated":
                latest_rels = processor.storage.get_entity_relations_by_family_id(
                    family_id=family_id,
                    limit=limit,
                    time_point=time_point,
                    max_version_absolute_id=None,
                )
                latest_by_rid = {r.family_id: r for r in latest_rels}
                latest_rids = set(latest_by_rid)

                all_rels = []
                for rid in accum_rids | latest_rids:
                    if rid in current_rids:
                        all_rels.append(current_by_rid[rid])
                    elif rid in accum_rids:
                        all_rels.append(accum_by_rid[rid])
                    else:
                        all_rels.append(latest_by_rid[rid])

                dicts = [relation_to_dict(r) for r in all_rels]
                enrich_relations(dicts, processor)

                for d in dicts:
                    rid = d["family_id"]
                    if rid not in current_rids:
                        if rid not in accum_rids:
                            d["_future"] = True
                        else:
                            d["_inherited"] = True

                return ok(dicts)

            # ---- all_versions: (v1..vN) ∪ latest, classify as current/inherited/future ----
            latest_rels = processor.storage.get_entity_relations_by_family_id(
                family_id=family_id,
                limit=limit,
                time_point=time_point,
                max_version_absolute_id=None,
            )
            latest_by_rid = {r.family_id: r for r in latest_rels}
            latest_rids = set(latest_by_rid)

            all_rels = []
            for rid in accum_rids | latest_rids:
                if rid in latest_rids:
                    all_rels.append(latest_by_rid[rid])
                else:
                    all_rels.append(accum_by_rid[rid])

            dicts = [relation_to_dict(r) for r in all_rels]
            enrich_relations(dicts, processor)

            for d in dicts:
                rid = d["family_id"]
                if rid in current_rids:
                    d["_version_scope"] = "current"
                elif rid in accum_rids:
                    d["_version_scope"] = "inherited"
                else:
                    d["_version_scope"] = "future"

            return ok(dicts)
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Find: 版本级 CRUD（Phase 2 — absolute_id 优先操作）
    # =========================================================

    @app.route("/api/v1/find/entities/create", methods=["POST"])
    def create_entity():
        """手动创建实体（生成新 family_id + absolute_id）。"""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            name = (body.get("name") or "").strip()
            content = (body.get("content") or "").strip()
            if not name:
                return err("name 为必填", 400)
            import uuid as _uuid

            now = datetime.now()
            ts = now.strftime("%Y%m%d_%H%M%S")
            # 生成唯一 ID（生成后查库校验）
            family_id = f"ent_{_uuid.uuid4().hex[:12]}"
            absolute_id = f"entity_{ts}_{_uuid.uuid4().hex[:8]}"
            for _ in range(10):
                existing = processor.storage.get_entity_by_absolute_id(absolute_id)
                if not existing:
                    break
                absolute_id = f"entity_{ts}_{_uuid.uuid4().hex[:8]}"
            for _ in range(10):
                existing = processor.storage.get_entity_by_family_id(family_id)
                if not existing:
                    break
                family_id = f"ent_{_uuid.uuid4().hex[:12]}"

            entity = Entity(
                absolute_id=absolute_id,
                family_id=family_id,
                name=name,
                content=content,
                event_time=now,
                processed_time=now,
                episode_id=body.get("episode_id", ""),
                source_document=body.get("source_document", ""),
            )
            processor.storage.save_entity(entity)
            return ok(entity_to_dict(entity))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/absolute/<absolute_id>", methods=["PUT"])
    def update_entity_absolute(absolute_id: str):
        """更新指定版本实体（不触发版本链 invalidate）。"""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            fields = {}
            for key in ("name", "content", "summary", "attributes", "confidence"):
                if key in body:
                    fields[key] = body[key]
            if not fields:
                return err("至少提供一个可更新字段", 400)
            updated = processor.storage.update_entity_by_absolute_id(absolute_id, **fields)
            if not updated:
                return err(f"未找到实体版本: {absolute_id}", 404)
            return ok(entity_to_dict(updated))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/absolute/<absolute_id>", methods=["DELETE"])
    def delete_entity_absolute(absolute_id: str):
        """删除指定版本实体（带删除保护：有关联关系时拒绝）。"""
        try:
            processor = _get_processor()
            # 删除保护：检查是否有关系引用该 absolute_id
            blocking = processor.storage.get_relations_referencing_absolute_id(absolute_id)
            if blocking:
                blocking_dicts = [relation_to_dict(r) for r in blocking[:10]]
                return ok({
                    "absolute_id": absolute_id,
                    "deleted": False,
                    "reason": f"该版本仍有 {len(blocking)} 条关联关系",
                    "blocking_relations": blocking_dicts,
                }, 409) if False else err(
                    f"该版本仍有 {len(blocking)} 条关联关系，请先删除或重定向这些关系",
                    409,
                )
            success = processor.storage.delete_entity_by_absolute_id(absolute_id)
            if not success:
                return err(f"未找到实体版本: {absolute_id}", 404)
            return ok({"absolute_id": absolute_id, "deleted": True})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/batch-delete-versions", methods=["POST"])
    def batch_delete_entity_versions():
        """批量删除实体版本（带删除保护）。"""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            absolute_ids = body.get("absolute_ids", [])
            if not isinstance(absolute_ids, list) or not absolute_ids:
                return err("absolute_ids 需为非空数组", 400)
            deleted = []
            blocked = {}
            for aid in absolute_ids:
                blocking = processor.storage.get_relations_referencing_absolute_id(aid)
                if blocking:
                    blocked[aid] = {
                        "blocking_count": len(blocking),
                        "blocking_relations": [relation_to_dict(r) for r in blocking[:5]],
                    }
                else:
                    success = processor.storage.delete_entity_by_absolute_id(aid)
                    if success:
                        deleted.append(aid)
                    else:
                        blocked[aid] = {"blocking_count": 0, "reason": "未找到"}
            return ok({
                "deleted": deleted,
                "blocked": blocked,
                "summary": {"deleted_count": len(deleted), "blocked_count": len(blocked)},
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/relations/create", methods=["POST"])
    def create_relation():
        """手动创建关系（生成新 family_id + absolute_id）。

        实体 ID 参数支持两种形式（二选一，优先使用 absolute_id）：
          - entity1_absolute_id / entity2_absolute_id（版本快照 ID）
          - entity1_family_id / entity2_family_id（逻辑 ID，自动解析为最新 absolute_id）
        """
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}

            # Resolve entity IDs: prefer absolute_id, fall back to family_id
            e1 = (body.get("entity1_absolute_id") or "").strip()
            e2 = (body.get("entity2_absolute_id") or "").strip()

            if not e1:
                e1_fid = (body.get("entity1_family_id") or "").strip()
                if e1_fid:
                    entity1 = processor.storage.get_entity_by_family_id(e1_fid)
                    if entity1:
                        e1 = entity1.absolute_id
                    else:
                        return err(f"entity1_family_id '{e1_fid}' 未找到对应实体", 404)

            if not e2:
                e2_fid = (body.get("entity2_family_id") or "").strip()
                if e2_fid:
                    entity2 = processor.storage.get_entity_by_family_id(e2_fid)
                    if entity2:
                        e2 = entity2.absolute_id
                    else:
                        return err(f"entity2_family_id '{e2_fid}' 未找到对应实体", 404)

            if not e1 or not e2:
                return err("需要 entity1_absolute_id 或 entity1_family_id（entity2 同理）", 400)

            content = (body.get("content") or "").strip()
            if not content:
                return err("content 为必填", 400)

            import uuid as _uuid

            now = datetime.now()
            ts = now.strftime("%Y%m%d_%H%M%S")
            # 确保 entity1 < entity2（无向关系）
            if e1 > e2:
                e1, e2 = e2, e1
            family_id = f"rel_{_uuid.uuid4().hex[:12]}"
            absolute_id = f"relation_{ts}_{_uuid.uuid4().hex[:8]}"
            for _ in range(10):
                existing = processor.storage.get_relation_by_absolute_id(absolute_id)
                if not existing:
                    break
                absolute_id = f"relation_{ts}_{_uuid.uuid4().hex[:8]}"
            for _ in range(10):
                existing = processor.storage.get_relation_by_family_id(family_id)
                if not existing:
                    break
                family_id = f"rel_{_uuid.uuid4().hex[:12]}"

            relation = Relation(
                absolute_id=absolute_id,
                family_id=family_id,
                entity1_absolute_id=e1,
                entity2_absolute_id=e2,
                content=content,
                event_time=now,
                processed_time=now,
                episode_id=body.get("episode_id", ""),
                source_document=body.get("source_document", ""),
            )
            processor.storage.save_relation(relation)
            return ok(relation_to_dict(relation))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/relations/absolute/<absolute_id>", methods=["PUT"])
    def update_relation_absolute(absolute_id: str):
        """更新指定版本关系。"""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            fields = {}
            for key in ("content", "summary", "attributes", "confidence"):
                if key in body:
                    fields[key] = body[key]
            if not fields:
                return err("至少提供一个可更新字段", 400)
            updated = processor.storage.update_relation_by_absolute_id(absolute_id, **fields)
            if not updated:
                return err(f"未找到关系版本: {absolute_id}", 404)
            return ok(relation_to_dict(updated))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/relations/absolute/<absolute_id>", methods=["DELETE"])
    def delete_relation_absolute(absolute_id: str):
        """删除指定版本关系。"""
        try:
            processor = _get_processor()
            success = processor.storage.delete_relation_by_absolute_id(absolute_id)
            if not success:
                return err(f"未找到关系版本: {absolute_id}", 404)
            return ok({"absolute_id": absolute_id, "deleted": True})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/relations/batch-delete-versions", methods=["POST"])
    def batch_delete_relation_versions():
        """批量删除关系版本。"""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            absolute_ids = body.get("absolute_ids", [])
            if not isinstance(absolute_ids, list) or not absolute_ids:
                return err("absolute_ids 需为非空数组", 400)
            deleted = []
            failed = []
            for aid in absolute_ids:
                success = processor.storage.delete_relation_by_absolute_id(aid)
                if success:
                    deleted.append(aid)
                else:
                    failed.append(aid)
            return ok({
                "deleted": deleted,
                "failed": failed,
                "summary": {"deleted_count": len(deleted), "failed_count": len(failed)},
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/split-version", methods=["POST"])
    def split_entity_version():
        """将实体版本拆分到新的 family_id。"""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            absolute_id = (body.get("absolute_id") or "").strip()
            if not absolute_id:
                return err("absolute_id 为必填", 400)
            new_family_id = (body.get("new_family_id") or "").strip()
            # 获取旧的 family_id（用于返回）
            old_entity = processor.storage.get_entity_by_absolute_id(absolute_id)
            if not old_entity:
                return err(f"未找到实体版本: {absolute_id}", 404)
            old_family_id = old_entity.family_id
            updated = processor.storage.split_entity_version(absolute_id, new_family_id)
            if not updated:
                return err(f"拆分失败: {absolute_id}", 500)
            return ok({
                "absolute_id": absolute_id,
                "old_family_id": old_family_id,
                "new_family_id": updated.family_id,
                "entity": entity_to_dict(updated),
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/relations/redirect", methods=["POST"])
    def redirect_relation():
        """重定向关系的实体端点。"""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            family_id = (body.get("family_id") or "").strip()
            side = (body.get("side") or "").strip()
            new_family_id = (body.get("new_family_id") or "").strip()
            if not family_id or not side or not new_family_id:
                return err("family_id, side, new_family_id 为必填", 400)
            if side not in ("entity1", "entity2"):
                return err("side 必须为 entity1 或 entity2", 400)
            count = processor.storage.redirect_relation(family_id, side, new_family_id)
            return ok({
                "family_id": family_id,
                "side": side,
                "new_family_id": new_family_id,
                "relations_updated": count,
            })
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Find: Episode 原子接口
    # =========================================================
    @app.route("/api/v1/find/episodes/latest/metadata", methods=["GET"])
    def find_latest_episode_metadata():
        try:
            processor = _get_processor()
            activity_type = request.args.get("activity_type")
            metadata = processor.storage.get_latest_episode_metadata(activity_type=activity_type)
            if metadata is None:
                return err("未找到 Episode 元数据", 404)
            return ok(metadata)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/episodes/latest", methods=["GET"])
    def find_latest_episode():
        try:
            processor = _get_processor()
            activity_type = request.args.get("activity_type")
            cache = processor.storage.get_latest_episode(activity_type=activity_type)
            if cache is None:
                return err("未找到 Episode", 404)
            return ok(episode_to_dict(cache))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/episodes/<cache_id>/text", methods=["GET"])
    def find_episode_text(cache_id: str):
        try:
            processor = _get_processor()
            text = processor.storage.get_episode_text(cache_id)
            if text is None:
                return err(f"未找到 Episode 或原文: {cache_id}", 404)
            return ok({"cache_id": cache_id, "text": text})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/episodes/<cache_id>", methods=["GET"])
    def find_episode(cache_id: str):
        try:
            processor = _get_processor()
            cache = processor.storage.load_episode(cache_id)
            if cache is None:
                return err(f"未找到 Episode: {cache_id}", 404)
            return ok(episode_to_dict(cache))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/episodes/<cache_id>/doc", methods=["GET"])
    def find_episode_doc(cache_id: str):
        """获取 Episode 对应的完整文档内容（原文 + 缓存摘要）。"""
        try:
            processor = _get_processor()
            doc_hash = processor.storage.get_doc_hash_by_cache_id(cache_id)
            if not doc_hash:
                return err(f"未找到文档: {cache_id}", 404)
            content = processor.storage.get_doc_content(doc_hash)
            if content is None:
                return err("文档内容不存在", 404)
            # 不返回 meta 中的大文本字段
            meta = content.get("meta") or {}
            meta_for_response = {k: v for k, v in meta.items() if k not in ("text", "document_path")}
            return ok({
                "meta": meta_for_response,
                "original": content.get("original"),
                "cache": content.get("cache"),
            })
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Time Travel / Snapshot / Invalidation (Phase 3)
    # =========================================================
    @app.route("/api/v1/find/snapshot", methods=["GET"])
    def find_snapshot():
        """获取指定时间点的图谱快照"""
        try:
            time_str = request.args.get("time")
            if not time_str:
                return err("time 为必填参数（ISO 格式）", 400)
            time_point = parse_time_point(time_str)
            limit = request.args.get("limit", type=int)

            processor = _get_processor()
            snapshot = processor.storage.get_snapshot(time_point, limit=limit)
            return ok({
                "time": time_point.isoformat(),
                "entities": [entity_to_dict(e) for e in snapshot["entities"]],
                "relations": [relation_to_dict(r) for r in snapshot["relations"]],
                "entity_count": len(snapshot["entities"]),
                "relation_count": len(snapshot["relations"]),
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/changes", methods=["GET"])
    def find_changes():
        """获取时间范围内的变更记录"""
        try:
            since_str = request.args.get("since")
            until_str = request.args.get("until")
            if not since_str:
                return err("since 为必填参数（ISO 格式）", 400)
            since = parse_time_point(since_str)
            until = parse_time_point(until_str) if until_str else None
            limit = request.args.get("limit", type=int)

            processor = _get_processor()
            changes = processor.storage.get_changes(since, until=until)
            return ok({
                "since": since.isoformat(),
                "until": (until or datetime.now(timezone.utc)).isoformat(),
                "entities": [entity_to_dict(e) for e in changes["entities"]],
                "relations": [relation_to_dict(r) for r in changes["relations"]],
                "entity_count": len(changes["entities"]),
                "relation_count": len(changes["relations"]),
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/relations/<family_id>/invalidate", methods=["POST"])
    def invalidate_relation(family_id: str):
        """标记关系为失效（不删除，保留历史）"""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            reason = body.get("reason", "")
            count = processor.storage.invalidate_relation(family_id, reason)
            if count == 0:
                return err(f"未找到可失效的关系: {family_id}", 404)
            return ok({"message": f"已标记 {count} 个关系版本为失效", "family_id": family_id})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/relations/invalidated", methods=["GET"])
    def find_invalidated_relations():
        """列出所有已失效的关系"""
        try:
            processor = _get_processor()
            limit = request.args.get("limit", type=int, default=100)
            relations = processor.storage.get_invalidated_relations(limit)
            dicts = [relation_to_dict(r) for r in relations]
            enrich_relations(dicts, processor)
            return ok(dicts)
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # 图谱结构统计 & 实体时间线
    # =========================================================
    @app.route("/api/v1/find/graph-stats", methods=["GET"])
    def find_graph_stats():
        """图谱结构统计"""
        try:
            processor = _get_processor()
            stats = processor.storage.get_graph_statistics()
            return ok(stats)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<family_id>/timeline", methods=["GET"])
    def find_entity_timeline(family_id: str):
        """实体版本时间线"""
        try:
            processor = _get_processor()
            with _perf_timer("find_entity_timeline"):
                versions = processor.storage.get_entity_versions(family_id)
            if not versions:
                return err(f"未找到实体: {family_id}", 404)

            # 获取关联关系的版本（批量优化：获取所有版本的关系，按 processed_time 过滤）
            relations_timeline = []
            if hasattr(processor.storage, 'get_entity_relations_timeline'):
                timeline_data = processor.storage.get_entity_relations_timeline(
                    family_id, [v.absolute_id for v in versions]
                )
                for item in timeline_data:
                    relations_timeline.append({
                        "family_id": item.get("relation_id") or item.get("family_id"),
                        "content": item["content"],
                        "event_time": item["event_time"],
                        "absolute_id": item["absolute_id"],
                    })
            else:
                # 回退到逐版本查询
                for v in versions:
                    rels = processor.storage.get_entity_relations_by_family_id(
                        family_id=family_id, max_version_absolute_id=v.absolute_id,
                    )
                    for r in rels:
                        relations_timeline.append({
                            "family_id": r.family_id,
                            "content": r.content,
                            "event_time": r.event_time.isoformat() if r.event_time else None,
                            "absolute_id": r.absolute_id,
                        })

            # 去重
            seen = set()
            unique_rels = []
            for r in relations_timeline:
                if r["absolute_id"] not in seen:
                    seen.add(r["absolute_id"])
                    unique_rels.append(r)

            return ok({
                "family_id": family_id,
                "versions": [entity_to_dict(v) for v in versions],
                "relations_timeline": unique_rels,
            })
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Phase: Section-Level Version Control
    # =========================================================

    @app.route("/api/v1/find/entities/<family_id>/section-history", methods=["GET"])
    def entity_section_history(family_id: str):
        """获取单个 section 的全版本变更历史。"""
        try:
            processor = _get_processor()
            section_key = request.args.get("section", "")
            if not section_key:
                return err("缺少 section 参数", 400)
            patches = processor.storage.get_section_history(family_id, section_key)
            return ok({
                "family_id": family_id,
                "section_key": section_key,
                "patches": [
                    {
                        "uuid": p.uuid,
                        "target_absolute_id": p.target_absolute_id,
                        "change_type": p.change_type,
                        "old_hash": p.old_hash,
                        "new_hash": p.new_hash,
                        "diff_summary": p.diff_summary,
                        "source_document": p.source_document,
                        "event_time": p.event_time.isoformat() if p.event_time else None,
                    }
                    for p in patches
                ],
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<family_id>/version-diff", methods=["GET"])
    def entity_version_diff(family_id: str):
        """获取两个版本之间的 section 级 diff."""
        try:
            processor = _get_processor()
            v1 = request.args.get("v1", "")
            v2 = request.args.get("v2", "")
            if not v1 or not v2:
                return err("需要 v1 和 v2 参数（两个版本的 absolute_id）", 400)
            diff = processor.storage.get_version_diff(family_id, v1, v2)
            return ok({
                "family_id": family_id,
                "v1": v1,
                "v2": v2,
                "sections": {
                    key: {
                        "old": info.get("old", ""),
                        "new": info.get("new", ""),
                        "changed": info.get("changed", False),
                        "change_type": info.get("change_type", "unchanged"),
                    }
                    for key, info in diff.items()
                },
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<family_id>/patches", methods=["GET"])
    def entity_patches(family_id: str):
        """获取实体的所有 ContentPatch 记录.\
        可选参数 section: 过滤特定 section"""
        try:
            processor = _get_processor()
            section_key = request.args.get("section", None)
            patches = processor.storage.get_content_patches(family_id, section_key=section_key)
            return ok({
                "family_id": family_id,
                "patches": [
                    {
                        "uuid": p.uuid,
                        "target_type": p.target_type,
                        "target_absolute_id": p.target_absolute_id,
                        "section_key": p.section_key,
                        "change_type": p.change_type,
                        "old_hash": p.old_hash,
                        "new_hash": p.new_hash,
                        "diff_summary": p.diff_summary,
                        "source_document": p.source_document,
                        "event_time": p.event_time.isoformat() if p.event_time else None,
                    }
                    for p in patches
                ],
            })
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Phase A: Entity Intelligence — 摘要进化 / 属性更新
    # =========================================================
    @app.route("/api/v1/find/entities/<family_id>/evolve-summary", methods=["POST"])
    def evolve_entity_summary(family_id: str):
        """手动触发实体摘要进化。"""
        try:
            processor = _get_processor()
            entity = processor.storage.get_entity_by_family_id(family_id)
            if entity is None:
                return err(f"未找到实体: {family_id}", 404)

            versions = processor.storage.get_entity_versions(family_id)
            old_version = versions[1] if len(versions) > 1 else None

            import asyncio
            loop = asyncio.new_event_loop()
            try:
                summary = loop.run_until_complete(
                    processor.llm_client.evolve_entity_summary(entity, old_version)
                )
            finally:
                loop.close()

            processor.storage.update_entity_summary(family_id, summary)
            return ok({"family_id": family_id, "summary": summary})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<family_id>", methods=["PUT"])
    def update_entity_v2(family_id: str):
        """编辑实体：支持更新 summary 和 attributes（不创建新版本）。"""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            summary = body.get("summary")
            attributes = body.get("attributes")

            if summary is not None:
                processor.storage.update_entity_summary(family_id, str(summary))
            if attributes is not None:
                attr_str = json.dumps(attributes, ensure_ascii=False) if isinstance(attributes, dict) else str(attributes)
                processor.storage.update_entity_attributes(family_id, attr_str)

            if summary is None and attributes is None:
                # 回退到原版：创建新版本
                name = body.get("name")
                content = body.get("content")
                if not name and not content:
                    return err("name 或 content 至少需要提供一个", 400)
                current = processor.storage.get_entity_by_family_id(family_id)
                if current is None:
                    return err(f"未找到实体: {family_id}", 404)
                now = datetime.now(timezone.utc)
                updated = Entity(
                    absolute_id=str(uuid.uuid4()),
                    family_id=family_id,
                    name=name if name else current.name,
                    content=content if content else current.content,
                    event_time=now, processed_time=now,
                    episode_id=current.episode_id,
                    source_document=current.source_document,
                    valid_at=now,
                )
                processor.storage.save_entity(updated)
                return ok({"message": "实体已更新", "absolute_id": updated.absolute_id})

            return ok({"message": "实体属性已更新", "family_id": family_id})
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Phase B: Advanced Search — BFS 遍历 + MMR 重排序
    # =========================================================
    @app.route("/api/v1/find/traverse", methods=["POST"])
    def traverse_graph():
        """BFS 图遍历搜索。"""
        try:
            body = request.get_json(silent=True) or {}
            seed_ids = body.get("seed_family_ids") or body.get("start_entity_ids", [])
            if not isinstance(seed_ids, list) or not seed_ids:
                return err("seed_family_ids 需为非空数组", 400)
            max_depth = int(body.get("max_depth", 2))
            max_nodes = int(body.get("max_nodes", 50))

            processor = _get_processor()
            searcher = GraphTraversalSearcher(processor.storage)
            entities = searcher.bfs_expand(seed_ids, max_depth=max_depth, max_nodes=max_nodes)
            return ok([entity_to_dict(e) for e in entities])
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Phase C: Episode Enhancement — CRUD + 事实溯源
    # =========================================================

    @app.route("/api/v1/find/episodes/search", methods=["POST"])
    def search_episodes():
        """搜索 Episode。"""
        try:
            body = request.get_json(silent=True) or {}
            query = (body.get("query") or "").strip()
            if not query:
                return err("query 为必填", 400)
            limit = int(body.get("limit", 20))
            processor = _get_processor()
            results = processor.storage.search_episodes_by_bm25(query, limit=limit)
            return ok([episode_to_dict(c) for c in results])
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/episodes/<cache_id>", methods=["DELETE"])
    def find_episode_delete(cache_id: str):
        """删除 Episode。"""
        try:
            processor = _get_processor()
            if hasattr(processor.storage, 'delete_episode_mentions'):
                processor.storage.delete_episode_mentions(cache_id)
            count = processor.storage.delete_episode(cache_id)
            if count == 0:
                return err(f"未找到 Episode: {cache_id}", 404)
            return ok({"message": "已删除", "cache_id": cache_id})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/episodes/batch-ingest", methods=["POST"])
    def batch_ingest_episodes():
        """批量导入 Episode。使用 bulk_save 优化写入性能。"""
        try:
            body = request.get_json(silent=True) or {}
            episodes = body.get("episodes", [])
            if not isinstance(episodes, list):
                return err("episodes 需为数组", 400)
            processor = _get_processor()
            episode_objects = []
            for ep in episodes:
                text = (ep.get("content") or ep.get("text") or "").strip()
                if not text:
                    continue
                source = ep.get("source_document", "batch_ingest")
                ep_type = ep.get("episode_type")
                mc = Episode(
                    absolute_id=str(uuid.uuid4()),
                    content=text,
                    event_time=datetime.now(timezone.utc),
                    source_document=source,
                    episode_type=ep_type,
                )
                episode_objects.append(mc)
            if episode_objects:
                # Use bulk save for Neo4j, fallback to loop for SQLite
                if hasattr(processor.storage, 'bulk_save_episodes'):
                    processor.storage.bulk_save_episodes(episode_objects)
                else:
                    for mc in episode_objects:
                        processor.storage.save_episode(mc)
            return ok({"message": f"已导入 {len(episode_objects)} 条 Episode", "count": len(episode_objects)})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<family_id>/provenance", methods=["GET"])
    def get_entity_provenance(family_id: str):
        """事实溯源：获取提及该实体的所有 Episode。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'get_entity_provenance'):
                return ok([])
            provenance = processor.storage.get_entity_provenance(family_id)
            return ok(provenance)
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Phase D: 矛盾检测
    # =========================================================
    @app.route("/api/v1/find/entities/<family_id>/contradictions", methods=["GET"])
    def get_entity_contradictions(family_id: str):
        """检测同一实体不同版本之间的矛盾。"""
        try:
            processor = _get_processor()
            versions = processor.storage.get_entity_versions(family_id)
            if len(versions) < 2:
                return ok([])

            import asyncio
            loop = asyncio.new_event_loop()
            try:
                contradictions = loop.run_until_complete(
                    processor.llm_client.detect_contradictions(family_id, versions)
                )
            finally:
                loop.close()

            return ok(contradictions)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<family_id>/resolve-contradiction", methods=["POST"])
    def resolve_entity_contradiction(family_id: str):
        """LLM 裁决矛盾。"""
        try:
            body = request.get_json(silent=True) or {}
            contradiction = body.get("contradiction")
            if not contradiction or not isinstance(contradiction, dict):
                return err("contradiction 为必填字段", 400)

            processor = _get_processor()
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                resolution = loop.run_until_complete(
                    processor.llm_client.resolve_contradiction(contradiction)
                )
            finally:
                loop.close()

            return ok(resolution)
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Phase E: DeepDream 记忆巩固（积木端点 — 编排由 Agent Skill 驱动）
    # =========================================================

    @app.route("/api/v1/find/dream/status", methods=["GET"])
    def dream_status():
        """查询梦境状态（最近一次）。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'list_dream_logs'):
                return ok({"status": "not_available"})
            logs = processor.storage.list_dream_logs(request.graph_id or "default", limit=1)
            if logs:
                return ok(logs[0])
            return ok({"status": "no_cycles"})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/dream/logs", methods=["GET"])
    def dream_logs():
        """历史梦境日志列表。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'list_dream_logs'):
                return ok([])
            limit = request.args.get("limit", type=int, default=20)
            logs = processor.storage.list_dream_logs(request.graph_id or "default", limit=limit)
            return ok(logs)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/dream/logs/<cycle_id>", methods=["GET"])
    def dream_log_detail(cycle_id: str):
        """单条梦境日志详情。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'get_dream_log'):
                return err("DeepDream 不可用", 404)
            log = processor.storage.get_dream_log(cycle_id)
            if log is None:
                return err(f"未找到梦境日志: {cycle_id}", 404)
            return ok(log)
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Phase E.2: DeepDream 梦境积木端点 — 种子 / 关系 / 记录
    # =========================================================

    @app.route("/api/v1/find/dream/seeds", methods=["POST"])
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
            if not hasattr(processor.storage, 'get_dream_seeds'):
                return err("DeepDream 不可用", 404)

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

    @app.route("/api/v1/find/dream/relation", methods=["POST"])
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

    @app.route("/api/v1/find/dream/episode", methods=["POST"])
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
            if not hasattr(processor.storage, 'save_dream_episode'):
                return err("DeepDream 不可用", 404)

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

    @app.route("/api/v1/find/dream/run", methods=["POST"])
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
            if not hasattr(processor.storage, 'get_dream_seeds'):
                return err("DeepDream 不可用（需要 Neo4j 后端）", 404)

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
    @app.route("/api/v1/find/ask", methods=["POST"])
    def agent_ask():
        """Agent 元查询：自然语言问题 → 结构化查询 + 回答。"""
        try:
            body = request.get_json(silent=True) or {}
            question = (body.get("question") or "").strip()
            if not question:
                return err("question 为必填", 400)

            processor = _get_processor()
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    processor.llm_client.agent_meta_query(question, request.graph_id or "default")
                )
            finally:
                loop.close()

            # 根据 query_plan 执行实际搜索
            intent = result.get("query_plan", {})
            query_type = intent.get("query_type", "hybrid")
            query_text = intent.get("query_text", question)

            entities = []
            relations = []

            if query_type == "traverse":
                entity_name = intent.get("entity_name", "")
                if entity_name:
                    seed_entities = processor.storage.search_entities_by_bm25(entity_name, limit=3)
                    seed_ids = [e.family_id for e in seed_entities]
                    if seed_ids:
                        searcher = GraphTraversalSearcher(processor.storage)
                        entities = searcher.bfs_expand(seed_ids, max_depth=2, max_nodes=20)
            else:
                # Compute query embedding for hybrid search (vector + BM25)
                query_embedding = None
                try:
                    ec = getattr(processor.storage, 'embedding_client', None)
                    if ec and getattr(ec, 'is_available', lambda: False)():
                        query_embedding = ec.encode([query_text])[0]
                except Exception:
                    pass
                searcher = HybridSearcher(processor.storage)
                entity_hits = searcher.search_entities(query_text=query_text, query_embedding=query_embedding, top_k=20)
                relation_hits = searcher.search_relations(query_text=query_text, query_embedding=query_embedding, top_k=10)
                # HybridSearcher returns List[Tuple[Entity/Relation, float]] — unpack
                entities = [e for e, _ in entity_hits]
                relations = [r for r, _ in relation_hits]

            entity_dicts = [entity_to_dict(e) for e in entities]
            relation_dicts = [relation_to_dict(r) for r in relations]
            result["results"] = {
                "entities": entity_dicts,
                "relations": relation_dicts,
            }

            # 用 LLM 综合搜索结果生成自然语言回答
            try:
                answer = processor.llm_client.synthesize_answer(question, entity_dicts, relation_dicts)
                result["answer"] = answer
            except Exception:
                pass  # 回退到 agent_meta_query 的默认 answer

            return ok(result)
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # SSE streaming endpoints
    # =========================================================

    @app.route("/api/v1/find/ask/stream", methods=["POST"])
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
                import asyncio
                from server.sse import sse_event
                try:
                    loop = asyncio.new_event_loop()
                    try:
                        result = loop.run_until_complete(
                            processor.llm_client.agent_meta_query(question, _graph_id)
                        )
                    finally:
                        loop.close()

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

                    if query_type == "traverse":
                        entity_name = intent.get("entity_name", "")
                        if entity_name:
                            seed_entities = processor.storage.search_entities_by_bm25(entity_name, limit=3)
                            seed_ids = [e.family_id for e in seed_entities]
                            if seed_ids:
                                searcher = GraphTraversalSearcher(processor.storage)
                                entities = searcher.bfs_expand(seed_ids, max_depth=2, max_nodes=20)
                    else:
                        # Compute query embedding for hybrid search (vector + BM25)
                        query_embedding = None
                        try:
                            ec = getattr(processor.storage, 'embedding_client', None)
                            if ec and getattr(ec, 'is_available', lambda: False)():
                                query_embedding = ec.encode([query_text])[0]
                        except Exception:
                            pass
                        searcher = HybridSearcher(processor.storage)
                        entity_hits = searcher.search_entities(query_text=query_text, query_embedding=query_embedding, top_k=20)
                        relation_hits = searcher.search_relations(query_text=query_text, query_embedding=query_embedding, top_k=10)
                        # HybridSearcher returns List[Tuple[Entity/Relation, float]] — unpack
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
                    entity_dicts = [entity_to_dict(e) for e in entities]
                    relation_dicts = [relation_to_dict(r) for r in relations]
                    result["results"] = {
                        "entities": entity_dicts,
                        "relations": relation_dicts,
                    }

                    answer = result.get("answer", "")
                    if not answer:
                        try:
                            answer = processor.llm_client.synthesize_answer(question, entity_dicts, relation_dicts)
                        except Exception:
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
                    import traceback; traceback.print_exc()
                    q.put(sse_event("error", {"message": str(e)}))
                finally:
                    q.put(sse_event("done", {"status": "completed"}))
                    q.put(_STREAM_SENTINEL)

            t = threading.Thread(target=_run, daemon=True)
            t.start()

        except Exception as e:
            return err(str(e), 500)

        from server.sse import sse_response, queue_to_generator
        return sse_response(queue_to_generator(q, sentinel=_STREAM_SENTINEL))

    @app.route("/api/v1/find/explain", methods=["POST"])
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

            import asyncio
            loop = asyncio.new_event_loop()
            try:
                explanation = loop.run_until_complete(
                    processor.llm_client.explain_entity(entity, aspect)
                )
            finally:
                loop.close()

            return ok({"family_id": family_id, "aspect": aspect, "explanation": explanation})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/suggestions", methods=["GET"])
    def get_suggestions():
        """智能建议。"""
        try:
            processor = _get_processor()
            entities = processor.storage.get_all_entities(limit=30, exclude_embedding=True)
            entity_count = processor.storage.get_entity_count()
            relation_count = processor.storage.get_relation_count()

            import asyncio
            loop = asyncio.new_event_loop()
            try:
                suggestions = loop.run_until_complete(
                    processor.llm_client.generate_suggestions(entities, entity_count, relation_count)
                )
            finally:
                loop.close()

            return ok(suggestions)
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # 图谱管理
    # =========================================================
    @app.route("/api/v1/graphs", methods=["GET", "POST"])
    def handle_graphs():
        """GET: 列出所有图谱。POST: 创建新图谱。"""
        if request.method == "POST":
            try:
                data = request.get_json(force=True) or {}
                graph_id = (data.get("graph_id") or "").strip()
                GraphRegistry.validate_graph_id(graph_id)
                # 检查是否已存在
                existing = registry.list_graphs()
                if graph_id in existing:
                    return err(f"图谱 '{graph_id}' 已存在", 409)
                # 触发懒创建：访问 processor 即会初始化 graph.db
                registry.get_processor(graph_id)
                return ok({"graph_id": graph_id, "message": "图谱创建成功"})
            except ValueError as e:
                return err(str(e), 400)
            except Exception as e:
                return err(str(e), 500)
        try:
            graphs = registry.list_graphs()
            return ok({"graphs": graphs, "count": len(graphs)})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/graphs/<graph_id>", methods=["DELETE"])
    def delete_graph(graph_id: str):
        """删除指定图谱（含所有数据）。"""
        try:
            GraphRegistry.validate_graph_id(graph_id)
            existing = registry.list_graphs()
            if graph_id not in existing:
                return err(f"图谱 '{graph_id}' 不存在", 404)
            registry.delete_graph(graph_id)
            return ok({"graph_id": graph_id, "message": "图谱已删除"})
        except ValueError as e:
            return err(str(e), 400)
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # 文档管理
    # =========================================================
    @app.route("/api/v1/docs", methods=["GET"])
    def list_docs():
        """列出当前图谱的所有文档。"""
        try:
            processor = _get_processor()
            docs = processor.storage.list_docs()
            # 返回精简字段
            result = []
            for d in docs:
                result.append({
                    "doc_hash": d.get("doc_hash", ""),
                    "source_document": d.get("source_document") or d.get("doc_name", ""),
                    "doc_name": d.get("source_document") or d.get("doc_name", ""),
                    "source_name": d.get("source_name", ""),
                    "event_time": d.get("event_time"),
                    "processed_time": d.get("processed_time"),
                    "activity_type": d.get("activity_type", ""),
                    "text_length": d.get("text_length", 0),
                    "filename": d.get("filename", ""),
                })
            return ok({"docs": result, "count": len(result)})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/docs/<path:filename>", methods=["GET"])
    def get_doc_content(filename):
        """获取文档内容（原始文本和缓存摘要）。"""
        try:
            processor = _get_processor()
            content = processor.storage.get_doc_content(filename)
            if content is None:
                return err("Document not found", 404)
            # 不返回 meta 中的大文本字段
            meta = content.get("meta") or {}
            meta_for_response = {k: v for k, v in meta.items() if k not in ("text", "document_path")}
            return ok({
                "meta": meta_for_response,
                "original": content.get("original"),
                "cache": content.get("cache"),
            })
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Neo4j: Entity Neighbors
    # =========================================================
    @app.route("/api/v1/find/entities/<entity_uuid>/neighbors", methods=["GET"])
    def find_entity_neighbors(entity_uuid: str):
        """获取实体的邻居图（Neo4j 专属）。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'get_entity_neighbors'):
                return err("此功能需要 Neo4j 后端", 400)
            depth = min(max(int(request.args.get('depth', 1)), 1), 5)
            with _perf_timer(f"find_entity_neighbors | depth={depth}"):
                result = processor.storage.get_entity_neighbors(entity_uuid, depth=depth)
            return ok(result)
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Neo4j: Cypher Shortest Path
    # =========================================================
    @app.route("/api/v1/find/paths/shortest-cypher", methods=["POST"])
    def find_shortest_path_cypher():
        """使用 Cypher shortestPath 查找路径（Neo4j 专属）。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'find_shortest_path_cypher'):
                return err("此功能需要 Neo4j 后端", 400)
            body = request.get_json(silent=True) or {}
            entity_a = (body.get("family_id_a") or body.get("entity_a") or "").strip()
            entity_b = (body.get("family_id_b") or body.get("entity_b") or "").strip()
            if not entity_a or not entity_b:
                return err("family_id_a 和 family_id_b 不能为空", 400)
            max_depth = min(max(int(body.get("max_depth", 6)), 1), 10)
            paths = processor.storage.find_shortest_path_cypher(entity_a, entity_b, max_depth=max_depth)
            return ok({
                "paths": paths,
                "source_family_id": entity_a,
                "target_family_id": entity_b,
            })
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Episodes
    # =========================================================
    @app.route("/api/v1/episodes", methods=["GET"])
    def list_episodes():
        """分页列出 Episodes（Neo4j 专属）。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'list_episodes'):
                return err("此功能需要 Neo4j 后端", 400)
            limit = min(max(int(request.args.get('limit', 20)), 1), 100)
            offset = max(int(request.args.get('offset', 0)), 0)
            episodes = processor.storage.list_episodes(limit=limit, offset=offset)
            total = processor.storage.count_episodes()
            return ok({"episodes": episodes, "total": total, "limit": limit, "offset": offset})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/episodes/<uuid>", methods=["GET"])
    def get_episode(uuid: str):
        """获取 Episode 详情（Neo4j 专属）。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'get_episode'):
                return err("此功能需要 Neo4j 后端", 400)
            episode = processor.storage.get_episode(uuid)
            if episode is None:
                return err("Episode 不存在", 404)
            return ok(episode)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/episodes/<uuid>/entities", methods=["GET"])
    def get_episode_entities(uuid: str):
        """获取 Episode 关联实体（Neo4j 专属）。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'get_episode_entities'):
                return err("此功能需要 Neo4j 后端", 400)
            entities = processor.storage.get_episode_entities(uuid)
            return ok({"entities": entities})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/episodes/search", methods=["POST"])
    def neo4j_search_episodes():
        """搜索 Episodes（Neo4j 专属）。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'search_episodes'):
                return err("此功能需要 Neo4j 后端", 400)
            body = request.get_json(silent=True) or {}
            query = (body.get("query") or "").strip()
            if not query:
                return err("query 不能为空", 400)
            limit = min(max(int(body.get("limit", 20)), 1), 100)
            episodes = processor.storage.search_episodes(query, limit=limit)
            return ok({"episodes": episodes})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/episodes/<uuid>", methods=["DELETE"])
    def neo4j_delete_episode(uuid: str):
        """删除 Episode（Neo4j 专属）。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'delete_episode'):
                return err("此功能需要 Neo4j 后端", 400)
            success = processor.storage.delete_episode(uuid)
            if not success:
                return err("Episode 不存在或删除失败", 404)
            return ok({"deleted": True})
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Communities
    # =========================================================
    @app.route("/api/v1/communities/detect", methods=["POST"])
    def detect_communities():
        """运行社区检测（Neo4j 专属）。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'detect_communities'):
                return err("此功能需要 Neo4j 后端", 400)
            body = request.get_json(silent=True) or {}
            algorithm = (body.get("algorithm") or "louvain").strip()
            resolution = float(body.get("resolution", 1.0))
            resolution = min(max(resolution, 0.1), 10.0)
            result = processor.storage.detect_communities(algorithm=algorithm, resolution=resolution)
            return ok(result)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/communities", methods=["GET"])
    def list_communities():
        """列出社区（Neo4j 专属）。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'get_communities'):
                return err("此功能需要 Neo4j 后端", 400)
            min_size = max(int(request.args.get('min_size', 3)), 1)
            limit = min(max(int(request.args.get('limit', 50)), 1), 200)
            offset = max(int(request.args.get('offset', 0)), 0)
            communities, total = processor.storage.get_communities(limit=limit, min_size=min_size, offset=offset)
            return ok({"communities": communities, "count": len(communities), "total": total})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/communities/<int:cid>", methods=["GET"])
    def get_community(cid: int):
        """获取社区详情（Neo4j 专属）。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'get_community'):
                return err("此功能需要 Neo4j 后端", 400)
            community = processor.storage.get_community(cid)
            if community is None:
                return err("社区不存在", 404)
            return ok(community)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/communities/<int:cid>/graph", methods=["GET"])
    def get_community_graph(cid: int):
        """获取社区子图数据（Neo4j 专属）。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'get_community_graph'):
                return err("此功能需要 Neo4j 后端", 400)
            graph_data = processor.storage.get_community_graph(cid)
            return ok(graph_data)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/communities", methods=["DELETE"])
    def clear_communities():
        """清除所有 community_id（Neo4j 专属）。"""
        try:
            processor = _get_processor()
            if not hasattr(processor.storage, 'clear_communities'):
                return err("此功能需要 Neo4j 后端", 400)
            cleared = processor.storage.clear_communities()
            return ok({"cleared": cleared})
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Butler: 管家式管理 — 一键健康分析 + 维护操作
    # =========================================================

    @app.route("/api/v1/butler/report", methods=["GET"])
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
            if hasattr(storage, 'get_data_quality_report'):
                quality = storage.get_data_quality_report()
            if hasattr(storage, 'count_isolated_entities'):
                quality["isolated_entities"] = storage.count_isolated_entities()

            # 3. 梦境状态
            dream_status = {"status": "not_available", "last_cycle_id": None, "last_cycle_time": None}
            if hasattr(storage, 'list_dream_logs'):
                logs = storage.list_dream_logs(request.graph_id or "default", limit=1)
                if logs:
                    last = logs[0]
                    dream_status = {
                        "status": last.get("status", "completed"),
                        "last_cycle_id": last.get("cycle_id"),
                        "last_cycle_time": last.get("started_at") or last.get("created_at"),
                        "entities_explored": last.get("entities_explored", 0),
                        "relations_created": last.get("relations_created", 0),
                    }
                else:
                    dream_status["status"] = "no_cycles"

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

            if dream_status["status"] == "no_cycles":
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
            if hasattr(storage, 'get_all_entities'):
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
                "dream": dream_status,
                "recommendations": recommendations,
                "recommendation_count": len(recommendations),
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/butler/execute", methods=["POST"])
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
                    if hasattr(storage, 'get_isolated_entities'):
                        isolated = storage.get_isolated_entities(limit=10000)
                        family_ids = list({e.family_id for e in isolated if e.family_id})
                        if dry_run:
                            results[action] = {"status": "preview", "count": len(family_ids), "family_ids": family_ids[:20]}
                        else:
                            deleted = 0
                            for fid in family_ids:
                                deleted += storage.delete_entity_all_versions(fid)
                            results[action] = {"status": "done", "deleted_families": len(family_ids), "deleted_versions": deleted}
                    else:
                        results[action] = {"status": "skipped", "reason": "当前存储后端不支持"}

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
                                import asyncio
                                loop = asyncio.new_event_loop()
                                try:
                                    summary = loop.run_until_complete(
                                        processor.llm_client.evolve_entity_summary(e)
                                    )
                                finally:
                                    loop.close()
                                storage.update_entity_summary(e.family_id, summary)
                                evolved += 1
                            except Exception:
                                failed += 1
                    results[action] = {"status": "done", "evolved": evolved, "failed": failed, "dry_run": dry_run}

                else:
                    results[action] = {"status": "unknown", "reason": f"未知操作: {action}"}

            return ok({"actions": results, "dry_run": dry_run})
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Chat Sessions — claude CLI 多会话管理
    # =========================================================

    _chat_mgr = None  # Lazy-init SessionManager
    _chat_mgr_lock = threading.Lock()

    def _get_chat_mgr():
        nonlocal _chat_mgr
        if _chat_mgr is not None:
            return _chat_mgr
        with _chat_mgr_lock:
            if _chat_mgr is not None:
                return _chat_mgr
            from server.chat_session import SessionManager
            _chat_mgr = SessionManager()
            _chat_mgr.start()
        return _chat_mgr

    @app.route("/api/v1/chat/sessions", methods=["GET"])
    def chat_list_sessions():
        """List all chat sessions."""
        include_closed = request.args.get("include_closed", "0") == "1"
        try:
            mgr = _get_chat_mgr()
            return ok(mgr.list_sessions(include_closed=include_closed))
        except Exception as e:
            logger.error("GET /api/v1/chat/sessions failed: %s", e)
            return err(str(e), 500)

    @app.route("/api/v1/chat/sessions", methods=["POST"])
    def chat_create_session():
        """Create a new chat session."""
        body = request.get_json(silent=True) or {}
        graph_id = body.get("graph_id", "default")
        title = body.get("title")
        try:
            mgr = _get_chat_mgr()
            result = mgr.create_session(graph_id=graph_id, title=title)
            return ok(result)
        except Exception as e:
            logger.error("POST /api/v1/chat/sessions failed: %s", e, exc_info=True)
            return err(str(e), 500)

    @app.route("/api/v1/chat/sessions/<sid>", methods=["GET"])
    def chat_get_session(sid):
        """Get session details."""
        mgr = _get_chat_mgr()
        result = mgr.get_session(sid)
        if not result:
            return err("Session not found", 404)
        return ok(result)

    @app.route("/api/v1/chat/sessions/<sid>", methods=["PUT"])
    def chat_update_session(sid):
        """Update session metadata (graph_id, title)."""
        body = request.get_json(silent=True) or {}
        mgr = _get_chat_mgr()
        if not mgr.update_session(sid, **body):
            return err("Session not found", 404)
        return ok(mgr.get_session(sid))

    @app.route("/api/v1/chat/sessions/<sid>", methods=["DELETE"])
    def chat_delete_session(sid):
        """Delete a session completely."""
        mgr = _get_chat_mgr()
        if not mgr.delete_session(sid):
            return err("Session not found", 404)
        return ok({"deleted": True})

    @app.route("/api/v1/chat/sessions/<sid>/close", methods=["POST"])
    def chat_close_session(sid):
        """Close a session (keep history, terminate process)."""
        mgr = _get_chat_mgr()
        if not mgr.close_session(sid):
            return err("Session not found", 404)
        return ok({"status": "closed"})

    @app.route("/api/v1/chat/sessions/<sid>/stream", methods=["POST"])
    def chat_send_message(sid):
        """Send a message to a session. Returns SSE stream of events."""
        body = request.get_json(silent=True) or {}
        message = body.get("message", "")
        attachments = body.get("attachments")

        if not message:
            return err("message is required", 400)

        mgr = _get_chat_mgr()
        sentinel = mgr.get_event_sentinel()
        resp_queue = mgr.send_message(sid, message, attachments=attachments)

        if resp_queue is None:
            return err("Session not found or closed", 404)

        return sse_response(queue_to_generator(resp_queue, sentinel=sentinel))

    # =========================================================
    # 聚合 API：减少 Agent 调用次数（1 次替代 3-5 次）
    # =========================================================

    @app.route("/api/v1/find/entities/<family_id>/profile", methods=["GET"])
    def entity_profile(family_id: str):
        """聚合返回：实体详情 + 最新版本 + 关系列表 + 版本数。"""
        try:
            processor = _get_processor()
            entity = processor.storage.get_entity_by_family_id(family_id)
            if entity is None:
                return err(f"未找到实体: {family_id}", 404)
            relations = processor.storage.get_entity_relations_by_family_id(family_id)
            version_count = 0
            if hasattr(processor.storage, 'get_entity_version_count'):
                version_count = processor.storage.get_entity_version_count(family_id)
            else:
                version_count = len(processor.storage.get_entity_versions(family_id))
            rels = [relation_to_dict(r) for r in relations]
            enrich_relations(rels, processor)
            return ok({
                "entity": entity_to_dict(entity),
                "relations": rels,
                "relation_count": len(rels),
                "version_count": version_count,
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/graph-summary", methods=["GET"])
    def graph_summary():
        """聚合返回：图谱统计 + 健康状态。"""
        try:
            processor = _get_processor()
            stats = processor.storage.get_graph_statistics()
            embedding_available = (
                processor.embedding_client is not None
                and processor.embedding_client.is_available()
            )
            storage_backend = "neo4j" if hasattr(processor.storage, 'is_neo4j') else "sqlite"
            return ok({
                "graph_id": request.graph_id,
                "storage_backend": storage_backend,
                "embedding_available": embedding_available,
                "statistics": stats,
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/maintenance/health", methods=["GET"])
    def maintenance_health():
        """数据健康度报告：孤立实体数/失效版本数/质量统计。"""
        try:
            processor = _get_processor()
            stats = processor.storage.get_graph_statistics()
            quality = None
            if hasattr(processor.storage, 'get_data_quality_report'):
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

    @app.route("/api/v1/find/maintenance/cleanup", methods=["POST"])
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
            # 清理孤立实体（仅 Neo4j 后端支持）
            if hasattr(processor.storage, 'get_isolated_entities'):
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
                        deleted = 0
                        for fid in family_ids:
                            deleted += processor.storage.delete_entity_all_versions(fid)
                        results["isolated_entities"] = {
                            "message": f"已删除 {len(family_ids)} 个孤立实体（{deleted} 个版本）",
                            "deleted_families": len(family_ids),
                            "deleted_versions": deleted,
                        }
                else:
                    results["isolated_entities"] = {"message": "没有孤立实体", "deleted": 0}
            else:
                results["isolated_entities"] = {"message": "当前存储后端不支持孤立实体检测", "deleted": 0}
            return ok(results)
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Convenience endpoints for Agent workflows
    # =========================================================

    @app.route("/api/v1/find/quick-search", methods=["POST"])
    def quick_search():
        """One-shot search: hybrid BM25+embedding RRF fusion with name boosting.

        Phase 1: exact name match (highest confidence)
        Phase 2: BM25 + embedding via HybridSearcher RRF fusion
        Phase 3: relation search via HybridSearcher RRF fusion
        """
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            query = body.get("query", "").strip()
            if not query:
                return err("query is required", 400)
            max_entities = min(int(body.get("max_entities", 10)), 50)
            max_relations = min(int(body.get("max_relations", 20)), 100)
            threshold = max(0.0, min(1.0, float(body.get("similarity_threshold", 0.4))))

            # Phase 1: Exact name match (instant, highest confidence)
            exact_entities = []
            seen_fids = set()
            exact_map = processor.storage.get_family_ids_by_names([query])
            if exact_map:
                fid = list(exact_map.values())[0]
                ent = processor.storage.get_entity_by_family_id(fid)
                if ent:
                    exact_entities.append(ent)
                    seen_fids.add(ent.family_id)

            # Phase 2: BM25 + embedding RRF fusion via HybridSearcher
            searcher = HybridSearcher(processor.storage)

            fused_entities = searcher.search_entities(
                query_text=query,
                top_k=max_entities,
                semantic_threshold=threshold,
            )
            # Dedup: skip entities already found by exact match
            rrf_entities = []
            for ent, score in fused_entities:
                if ent.family_id not in seen_fids:
                    rrf_entities.append(ent)
                    seen_fids.add(ent.family_id)

            entities = exact_entities + rrf_entities
            entities = entities[:max_entities]

            # Phase 3: Relation search via HybridSearcher RRF fusion
            fused_relations = searcher.search_relations(
                query_text=query,
                top_k=max_relations,
                semantic_threshold=max(0.2, threshold - 0.1),
            )
            relations = [r for r, _ in fused_relations]

            entity_dicts = [entity_to_dict(e) for e in entities]
            rel_dicts = [relation_to_dict(r) for r in relations]
            enrich_relations(rel_dicts, processor)

            return ok({
                "query": query,
                "entities": entity_dicts,
                "entity_count": len(entity_dicts),
                "relations": rel_dicts,
                "relation_count": len(rel_dicts),
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/by-name/<name>", methods=["GET"])
    def find_entity_by_name(name: str):
        """Quick entity lookup by exact or fuzzy name match.
        Cascade: exact match → core-name match (strip parentheses) → BM25 text → embedding fallback.
        """
        try:
            import re as _re
            processor = _get_processor()
            threshold = float(request.args.get("threshold", "0.5"))
            limit = int(request.args.get("limit", "5"))
            best = None

            # Step 1: Exact name match
            exact_map = processor.storage.get_family_ids_by_names([name])
            if exact_map:
                fid = list(exact_map.values())[0]
                best = processor.storage.get_entity_by_family_id(fid)

            # Step 2: Core-name match (strip parentheses for alias matching)
            if not best:
                core_name = _re.sub(r'[（(][^）)]+[）)]', '', name).strip()
                if core_name and core_name != name:
                    core_map = processor.storage.get_family_ids_by_names([core_name])
                    if core_map:
                        fid = list(core_map.values())[0]
                        best = processor.storage.get_entity_by_family_id(fid)

            # Step 2.5: Name prefix match (e.g., "Go语言" → "Go语言（Golang）")
            if not best:
                prefix_entities = processor.storage.find_entity_by_name_prefix(name, limit=3)
                if prefix_entities:
                    best = prefix_entities[0]

            # Step 3: BM25 text search (fast, keyword-based)
            if not best:
                bm25_results = processor.storage.search_entities_by_bm25(name, limit=10)
                if bm25_results:
                    # Priority 1: name contains query
                    for r in bm25_results:
                        rn = (r.name or "").strip()
                        if name.lower() in rn.lower():
                            best = r
                            break
                    # Priority 2: core name (strip parens) matches query
                    if not best:
                        for r in bm25_results:
                            rn = (r.name or "").strip()
                            core_rn = _re.sub(r'[（(][^）)]+[）)]', '', rn).strip()
                            if core_rn.lower() == name.lower():
                                best = r
                                break
                    # Priority 3: first BM25 result
                    if not best:
                        best = bm25_results[0]

            # Step 4: Embedding fallback (semantic, slower)
            if not best:
                entities = processor.storage.search_entities_by_similarity(
                    query_name=name,
                    query_content=name,
                    threshold=threshold,
                    max_results=limit,
                    text_mode="name_only",
                    similarity_method="embedding",
                )
                if entities:
                    best = entities[0]

            if not best:
                return ok({"entity": None, "message": f"No entity found matching '{name}'"})
            rels = processor.storage.get_entity_relations_by_family_id(best.family_id)
            rel_dicts = [relation_to_dict(r) for r in rels]
            enrich_relations(rel_dicts, processor)
            return ok({
                "entity": entity_to_dict(best),
                "relations": rel_dicts,
                "relation_count": len(rel_dicts),
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/batch-profiles", methods=["POST"])
    def batch_profiles():
        """Get multiple entity profiles in one call."""
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            family_ids = body.get("family_ids", [])
            if not family_ids:
                return err("family_ids is required", 400)
            if len(family_ids) > 20:
                return err("Maximum 20 entities per batch", 400)

            # Use batch method if available (eliminates N+1 queries)
            if hasattr(processor.storage, 'batch_get_entity_profiles'):
                batch_results = processor.storage.batch_get_entity_profiles(family_ids)
                profiles = []
                for item in batch_results:
                    entity = item.get("entity")
                    relations = item.get("relations", [])
                    rel_dicts = [relation_to_dict(r) for r in relations]
                    enrich_relations(rel_dicts, processor)
                    profiles.append({
                        "family_id": item["family_id"],
                        "entity": entity_to_dict(entity) if entity else None,
                        "relations": rel_dicts,
                        "relation_count": len(rel_dicts),
                        "version_count": item.get("version_count", 0),
                    })
            else:
                # Fallback for non-Neo4j backends
                profiles = []
                for fid in family_ids:
                    entity = processor.storage.get_entity_by_family_id(fid)
                    if entity is None:
                        profiles.append({"family_id": fid, "entity": None, "error": "not found",
                                         "relations": [], "relation_count": 0, "version_count": 0})
                        continue
                    relations = processor.storage.get_entity_relations_by_family_id(fid)
                    version_count = 0
                    if hasattr(processor.storage, 'get_entity_version_count'):
                        version_count = processor.storage.get_entity_version_count(fid)
                    rel_dicts = [relation_to_dict(r) for r in relations]
                    enrich_relations(rel_dicts, processor)
                    profiles.append({
                        "family_id": fid,
                        "entity": entity_to_dict(entity),
                        "relations": rel_dicts,
                        "relation_count": len(rel_dicts),
                        "version_count": version_count,
                    })
            return ok({"profiles": profiles, "count": len(profiles)})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/recent-activity", methods=["GET"])
    def recent_activity():
        """Get recent graph activity: latest entities, relations, and episodes."""
        try:
            processor = _get_processor()
            limit = min(int(request.args.get("limit", "10")), 50)
            # Latest entities
            latest_entities = processor.storage.get_all_entities(limit=limit, exclude_embedding=True)
            # Latest relations
            latest_relations = processor.storage.get_all_relations(limit=limit, exclude_embedding=True) if hasattr(processor.storage, 'get_all_relations') else []
            # Stats
            stats = processor.storage.get_graph_statistics()

            entity_dicts = [entity_to_dict(e) for e in reversed(latest_entities)]
            rel_dicts = [relation_to_dict(r) for r in reversed(latest_relations)]
            enrich_relations(rel_dicts, processor)

            return ok({
                "statistics": stats,
                "latest_entities": entity_dicts,
                "latest_relations": rel_dicts,
            })
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # SPA: 提供前端静态页面（所有非 API 路由返回 index.html）
    # =========================================================
    _SPA_INDEX = str(Path(__file__).resolve().parent / "static" / "index.html")
    _API_PREFIXES = ("/api/", "/health")

    @app.route("/", methods=["GET"])
    def serve_index():
        return app.send_static_file("index.html")

    @app.route("/<path:path>", methods=["GET"])
    def serve_spa(path):
        # API 路由不拦截
        if any(path.startswith(p) for p in _API_PREFIXES):
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
        "embedding_model_path": model_path,
        "embedding_model_name": model_name,
        "embedding_device": embedding.get("device", "cpu"),
        "embedding_use_local": use_local,
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
    if "distill_data_dir" in pipeline_debug:
        kwargs["distill_data_dir"] = pipeline_debug["distill_data_dir"]
    return TemporalMemoryGraphProcessor(**kwargs)


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


def _kill_port_occupants(port: int) -> None:
    """Kill processes occupying the given port."""
    import subprocess
    try:
        # lsof 方式
        result = subprocess.run(
            ["lsof", "-t", "-i", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [p.strip() for p in result.stdout.strip().splitlines() if p.strip()]
        if pids:
            for pid in pids:
                # 避免杀掉自己
                if int(pid) == os.getpid():
                    continue
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    logging.info("已发送 SIGTERM 到进程 %s (占用端口 %d)", pid, port)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    logging.warning("无权限终止进程 %s", pid)
            # 等待进程退出
            for pid in pids:
                if int(pid) == os.getpid():
                    continue
                try:
                    os.waitpid(int(pid), os.WNOHANG)
                except (ChildProcessError, PermissionError):
                    pass
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # 回退到 fuser
    try:
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


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
        _kill_port_occupants(listen_port)
        time.sleep(1)
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
                except Exception:
                    pass
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
        from server.dashboard import DeepDreamDashboard
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
            except Exception:
                pass
        # 关闭所有图谱的数据库连接
        for gid in registry.list_graphs():
            try:
                proc = registry.get_processor(gid)
                if hasattr(proc, 'storage') and hasattr(proc.storage, 'close'):
                    proc.storage.close()
            except Exception:
                pass
        os._exit(0)

    signal.signal(signal.SIGTERM, _on_signal)

    @atexit.register
    def _cleanup():
        if dashboard_ref is not None:
            try:
                dashboard_ref.stop()
            except Exception:
                pass
        for gid in registry.list_graphs():
            try:
                proc = registry.get_processor(gid)
                if hasattr(proc, 'storage') and hasattr(proc.storage, 'close'):
                    proc.storage.close()
            except Exception:
                pass

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
