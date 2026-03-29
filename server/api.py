"""
Temporal_Memory_Graph 自然语言记忆图 API（多图谱模式）

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
from server.queue import RememberTask, RememberTaskQueue
from server.registry import GraphRegistry
from processor import TemporalMemoryGraphProcessor
from processor.llm.client import LLM_PRIORITY_STEP6
from processor.models import Entity, MemoryCache, Relation


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


def entity_to_dict(e: Entity) -> Dict[str, Any]:
    return {
        "id": e.absolute_id,  # 向后兼容
        "absolute_id": e.absolute_id,
        "entity_id": e.entity_id,
        "name": e.name,
        "content": e.content,
        "event_time": e.event_time.isoformat() if e.event_time else None,
        "processed_time": e.processed_time.isoformat() if e.processed_time else None,
        "memory_cache_id": e.memory_cache_id,
        "source_document": getattr(e, "source_document", "") or getattr(e, "doc_name", "") or "",
        "doc_name": getattr(e, "source_document", "") or getattr(e, "doc_name", "") or "",
    }


def relation_to_dict(r: Relation) -> Dict[str, Any]:
    return {
        "id": r.absolute_id,  # 向后兼容
        "absolute_id": r.absolute_id,
        "relation_id": r.relation_id,
        "entity1_absolute_id": r.entity1_absolute_id,
        "entity2_absolute_id": r.entity2_absolute_id,
        "content": r.content,
        "event_time": r.event_time.isoformat() if r.event_time else None,
        "processed_time": r.processed_time.isoformat() if r.processed_time else None,
        "memory_cache_id": r.memory_cache_id,
        "source_document": getattr(r, "source_document", "") or getattr(r, "doc_name", "") or "",
        "doc_name": getattr(r, "source_document", "") or getattr(r, "doc_name", "") or "",
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


def memory_cache_to_dict(c: MemoryCache) -> Dict[str, Any]:
    return {
        "id": c.absolute_id,  # 向后兼容
        "absolute_id": c.absolute_id,
        "content": c.content,
        "event_time": c.event_time.isoformat() if c.event_time else None,
        "source_document": getattr(c, "source_document", "") or getattr(c, "doc_name", "") or "",
        "doc_name": getattr(c, "source_document", "") or getattr(c, "doc_name", "") or "",
        "activity_type": getattr(c, "activity_type", None),
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
    # 对 500 错误隐藏内部细节，只返回通用提示
    if status >= 500:
        import logging
        logging.getLogger(__name__).error("API error: %s", message)
        message = "服务器内部错误，请稍后重试"
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
        entities = storage.get_all_entities_before_time(time_before_dt, limit=max_entities)
        for e in entities:
            entity_absolute_ids.add(e.absolute_id)
    else:
        entities = storage.get_all_entities(limit=max_entities)
        for e in entities:
            entity_absolute_ids.add(e.absolute_id)

    if not entity_absolute_ids:
        return entity_absolute_ids, relation_absolute_ids

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
        return response

    config = config or {}

    # 不需要 graph_id 的路由（白名单）
    _NO_GRAPH_ID_ROUTES = {"/api/v1/graphs", "/api/v1/routes"}
    # 系统 API 前缀（不需要 graph_id）
    _SYSTEM_API_PREFIX = "/api/v1/system/"

    # 简单内存限流（按 IP，滑动窗口）
    _rate_limit_store: Dict[str, List[float]] = {}
    _rate_limit_lock = threading.Lock()
    _RATE_LIMIT = int(config.get("rate_limit_per_minute", 600))
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

    def _score_entity_versions_against_time(entity_id: str, time_point: datetime, proc=None) -> List[Tuple[float, int, Entity]]:
        if proc is None:
            proc = _get_processor()
        target = _normalize_time_for_compare(time_point)
        scored: List[Tuple[float, int, Entity]] = []
        for version in proc.storage.get_entity_versions(entity_id):
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
            return ok({
                "graph_id": request.graph_id,
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
          - graph_id（必填）：目标图谱 ID
          - text（可选）：正文
          - file（可选）：上传文件（multipart）
          - file_path（可选）：服务端本地文件路径
          - source_name / doc_name（可选）：来源名称，默认 api_input
          - load_cache_memory（可选）：
            true = 接续图谱中已有缓存链（同图任务需串行）
            false = 不接续外部缓存链，但任务内部滑窗仍续写自己的 cache 链（可并行）
          - event_time（可选）：ISO 8601 事件时间

        返回：HTTP 202。查询进度：GET /api/v1/remember/tasks/<task_id>
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

            # 原文由处理器在 save_memory_cache 阶段保存到 docs/{timestamp}_{hash}/original.txt
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
            limit = request.args.get("limit", 50, type=int)
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
                    "path": "/api/v1/find/entities/<entity_id>/as-of-time",
                    "methods": ["GET"],
                    "summary": "返回该实体在指定时间点的最近过去版本",
                    "query": {"time_point": "ISO 8601 string，必填"},
                },
                {
                    "path": "/api/v1/find/entities/<entity_id>/nearest-to-time",
                    "methods": ["GET"],
                    "summary": "返回该实体距离指定时间点最近的版本",
                    "query": {
                        "time_point": "ISO 8601 string，必填",
                        "max_delta_seconds": "float，可选；超出该误差则返回 404",
                    },
                },
                {
                    "path": "/api/v1/find/entities/<entity_id>/around-time",
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
                    "path": "/api/v1/find/entities/<entity_id>/relations",
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
                    "path": "/api/v1/find/relations/by-entity-id/<entity_id>",
                    "methods": ["GET"],
                    "summary": "按实体业务 ID 查询相关关系",
                    "aliases": ["/api/v1/find/entities/<entity_id>/relations"],
                },
                {
                    "path": "/api/v1/find/relations/between",
                    "methods": ["GET", "POST"],
                    "summary": "查询两个实体之间的关系",
                    "body_or_query": {
                        "entity_id_a": "string，必填",
                        "entity_id_b": "string，必填",
                    },
                },
                {
                    "path": "/api/v1/find/paths/shortest",
                    "methods": ["GET", "POST"],
                    "summary": "查找两个实体之间的最短路径",
                    "body_or_query": {
                        "entity_id_a": "string，必填",
                        "entity_id_b": "string，必填",
                        "max_depth": "int，可选，默认6",
                        "max_paths": "int，可选，默认10",
                    },
                },
            ],
            "memory_cache": [
                {
                    "path": "/api/v1/find/memory-caches/latest",
                    "methods": ["GET"],
                    "summary": "读取最新记忆缓存",
                },
                {
                    "path": "/api/v1/find/memory-caches/latest/metadata",
                    "methods": ["GET"],
                    "summary": "读取最新记忆缓存元数据",
                },
                {
                    "path": "/api/v1/find/memory-caches/<cache_id>",
                    "methods": ["GET"],
                    "summary": "按 cache_id 读取记忆缓存",
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
            total_entities = len(processor.storage.get_all_entities(limit=None))
            total_relations = len(processor.storage.get_all_relations())

            cache_json_dir = processor.storage.cache_json_dir
            cache_dir = processor.storage.cache_dir
            json_files = list(cache_json_dir.glob("*.json"))
            # 优先从 docs/ 新结构计数
            docs_meta_files = list(processor.storage.docs_dir.glob("*/meta.json")) if processor.storage.docs_dir.is_dir() else []
            if docs_meta_files:
                total_memory_caches = len(docs_meta_files)
            elif json_files:
                total_memory_caches = len(json_files)
            else:
                total_memory_caches = len(list(cache_dir.glob("*.json")))

            return ok({
                "total_entities": total_entities,
                "total_relations": total_relations,
                "total_memory_caches": total_memory_caches,
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

            try:
                time_before_dt = parse_time_point(time_before) if time_before else None
                time_after_dt = parse_time_point(time_after) if time_after else None
            except ValueError as ve:
                return err(str(ve), 400)

            processor = _get_processor()
            storage = processor.storage

            # --- 第一步：语义召回实体 ---
            matched_entities = storage.search_entities_by_similarity(
                query_name=query,
                query_content=query,
                threshold=similarity_threshold,
                max_results=max_entities,
                text_mode="name_and_content",
                similarity_method="embedding",
            )

            # --- 第二步：语义召回关系 ---
            matched_relations = storage.search_relations_by_similarity(
                query_text=query,
                threshold=similarity_threshold,
                max_results=max_relations,
            )

            entity_abs_ids: Set[str] = {e.absolute_id for e in matched_entities}
            relation_abs_ids: Set[str] = {r.absolute_id for r in matched_relations}
            entities_by_abs: Dict[str, Entity] = {e.absolute_id: e for e in matched_entities}

            # --- 第三步：从语义命中的关系中补充关联实体 ---
            for r in list(matched_relations):
                for abs_id in (r.entity1_absolute_id, r.entity2_absolute_id):
                    if abs_id not in entity_abs_ids:
                        e = storage.get_entity_by_absolute_id(abs_id)
                        if e:
                            entities_by_abs[e.absolute_id] = e
                            entity_abs_ids.add(e.absolute_id)

            # --- 第四步：图谱邻域扩展 ---
            if expand and entity_abs_ids:
                expanded_rels = storage.get_relations_by_entity_absolute_ids(
                    list(entity_abs_ids), limit=max_relations
                )
                for r in expanded_rels:
                    if r.absolute_id not in relation_abs_ids:
                        relation_abs_ids.add(r.absolute_id)
                        matched_relations.append(r)
                    for abs_id in (r.entity1_absolute_id, r.entity2_absolute_id):
                        if abs_id not in entity_abs_ids:
                            e = storage.get_entity_by_absolute_id(abs_id)
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
                entity_ids, relation_ids = _extract_candidate_ids(
                    processor.storage, body, parse_time_point
                )
            except ValueError as ve:
                return err(str(ve), 400)
            storage = processor.storage
            entities_data: List[Dict[str, Any]] = []
            relations_data: List[Dict[str, Any]] = []
            if include_entities:
                for eid in entity_ids:
                    e = storage.get_entity_by_absolute_id(eid)
                    if e:
                        entities_data.append(entity_to_dict(e))
            if include_relations:
                for rid in relation_ids:
                    r = storage.get_relation_by_absolute_id(rid)
                    if r:
                        relations_data.append(relation_to_dict(r))
            return ok({"entities": entities_data, "relations": relations_data})
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Find: 实体原子接口
    # =========================================================
    @app.route("/api/v1/find/entities", methods=["GET"])
    def find_entities_all():
        try:
            processor = _get_processor()
            limit = request.args.get("limit", type=int)
            entities = processor.storage.get_all_entities(limit=limit)
            return ok([entity_to_dict(e) for e in entities])
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
            entities = processor.storage.get_all_entities_before_time(time_point, limit=limit)
            return ok([entity_to_dict(e) for e in entities])
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/version-counts", methods=["POST"])
    def find_entity_version_counts():
        try:
            processor = _get_processor()
            body = request.get_json(silent=True) or {}
            entity_ids = body.get("entity_ids")
            if not isinstance(entity_ids, list) or not all(isinstance(x, str) for x in entity_ids):
                return err("请求体需包含 entity_ids 字符串数组", 400)
            counts = processor.storage.get_entity_version_counts(entity_ids)
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

    @app.route("/api/v1/find/entities/<entity_id>/versions", methods=["GET"])
    def find_entity_versions(entity_id: str):
        try:
            processor = _get_processor()
            versions = processor.storage.get_entity_versions(entity_id)
            return ok([entity_to_dict(e) for e in versions])
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<entity_id>/as-of-time", methods=["GET"])
    def find_entity_at_time(entity_id: str):
        try:
            processor = _get_processor()
            time_point_str = request.args.get("time_point")
            if not time_point_str:
                return err("time_point 为必填参数（ISO 格式）", 400)
            try:
                time_point = parse_time_point(time_point_str)
            except ValueError as ve:
                return err(str(ve), 400)
            entity = processor.storage.get_entity_version_at_time(entity_id, time_point)
            if entity is None:
                return err(f"未找到该时间点版本: {entity_id}", 404)
            return ok(entity_to_dict(entity))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<entity_id>/nearest-to-time", methods=["GET"])
    def find_entity_nearest_to_time(entity_id: str):
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

            scored = _score_entity_versions_against_time(entity_id, time_point, proc=processor)
            if not scored:
                return err(f"未找到实体: {entity_id}", 404)

            delta_seconds, _, entity = scored[0]
            if max_delta_seconds is not None and delta_seconds > max_delta_seconds:
                return err(f"最近版本超出允许误差: {delta_seconds:.3f}s > {max_delta_seconds:.3f}s", 404)

            return ok({
                "entity_id": entity_id,
                "query_time": time_point.isoformat(),
                "matched": entity_to_dict(entity),
                "delta_seconds": round(delta_seconds, 6),
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<entity_id>/around-time", methods=["GET"])
    def find_entity_around_time(entity_id: str):
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
            for delta_seconds, _, entity in _score_entity_versions_against_time(entity_id, time_point, proc=processor):
                if delta_seconds > within_seconds:
                    continue
                item = entity_to_dict(entity)
                item["delta_seconds"] = round(delta_seconds, 6)
                direction = _normalize_time_for_compare(entity.event_time) - target
                item["relative_position"] = "before_or_exact" if direction.total_seconds() <= 0 else "after"
                matches.append(item)

            if not matches:
                return err(f"未找到 {within_seconds:.3f} 秒范围内的实体版本: {entity_id}", 404)

            return ok({
                "entity_id": entity_id,
                "query_time": time_point.isoformat(),
                "within_seconds": within_seconds,
                "count": len(matches),
                "matches": matches,
            })
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<entity_id>/version-count", methods=["GET"])
    def find_entity_version_count(entity_id: str):
        try:
            processor = _get_processor()
            count = processor.storage.get_entity_version_count(entity_id)
            if count <= 0:
                return err(f"未找到实体: {entity_id}", 404)
            return ok({"entity_id": entity_id, "version_count": count})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<entity_id>", methods=["GET"])
    def find_entity_by_id(entity_id: str):
        try:
            processor = _get_processor()
            entity = processor.storage.get_entity_by_entity_id(entity_id)
            if entity is None:
                return err(f"未找到实体: {entity_id}", 404)
            return ok(entity_to_dict(entity))
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
            relations = processor.storage.get_all_relations()
            if offset > 0:
                relations = relations[offset:]
            if limit is not None:
                relations = relations[:limit]
            dicts = [relation_to_dict(r) for r in relations]
            enrich_relations(dicts, processor)
            return ok(dicts)
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
            entity_id_a = str(body.get("entity_id_a") or body.get("from_entity_id") or request.args.get("entity_id_a") or request.args.get("from_entity_id") or "").strip()
            entity_id_b = str(body.get("entity_id_b") or body.get("to_entity_id") or request.args.get("entity_id_b") or request.args.get("to_entity_id") or "").strip()
            if not entity_id_a or not entity_id_b:
                return err("entity_id_a 与 entity_id_b 为必填参数", 400)
            relations = processor.storage.get_relations_by_entities(entity_id_a, entity_id_b)
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
            entity_id_a = str(body.get("entity_id_a") or body.get("from_entity_id")
                             or request.args.get("entity_id_a")
                             or request.args.get("from_entity_id") or "").strip()
            entity_id_b = str(body.get("entity_id_b") or body.get("to_entity_id")
                             or request.args.get("entity_id_b")
                             or request.args.get("to_entity_id") or "").strip()
            if not entity_id_a or not entity_id_b:
                return err("entity_id_a 与 entity_id_b 为必填参数", 400)

            max_depth = body.get("max_depth") if body else None
            if max_depth is None:
                max_depth = request.args.get("max_depth", type=int)
            max_depth = max_depth or 6

            max_paths = body.get("max_paths") if body else None
            if max_paths is None:
                max_paths = request.args.get("max_paths", type=int)
            max_paths = max_paths or 10

            result = processor.storage.find_shortest_paths(
                source_entity_id=entity_id_a,
                target_entity_id=entity_id_b,
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

    @app.route("/api/v1/find/relations/<relation_id>/versions", methods=["GET"])
    def find_relation_versions(relation_id: str):
        try:
            processor = _get_processor()
            versions = processor.storage.get_relation_versions(relation_id)
            dicts = [relation_to_dict(r) for r in versions]
            enrich_relations(dicts, processor)
            return ok(dicts)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/entities/<entity_id>/relations", methods=["GET"])
    def find_relations_by_entity(entity_id: str):
        try:
            processor = _get_processor()
            limit = request.args.get("limit", type=int)
            time_point_str = request.args.get("time_point")
            try:
                time_point = parse_time_point(time_point_str)
            except ValueError as ve:
                return err(str(ve), 400)
            max_version_absolute_id = (request.args.get("max_version_absolute_id") or "").strip() or None
            relations = processor.storage.get_entity_relations_by_entity_id(
                entity_id=entity_id,
                limit=limit,
                time_point=time_point,
                max_version_absolute_id=max_version_absolute_id,
            )
            dicts = [relation_to_dict(r) for r in relations]
            enrich_relations(dicts, processor)
            return ok(dicts)
        except Exception as e:
            return err(str(e), 500)

    # =========================================================
    # Find: 记忆缓存原子接口
    # =========================================================
    @app.route("/api/v1/find/memory-caches/latest/metadata", methods=["GET"])
    def find_latest_memory_cache_metadata():
        try:
            processor = _get_processor()
            activity_type = request.args.get("activity_type")
            metadata = processor.storage.get_latest_memory_cache_metadata(activity_type=activity_type)
            if metadata is None:
                return err("未找到记忆缓存元数据", 404)
            return ok(metadata)
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/memory-caches/latest", methods=["GET"])
    def find_latest_memory_cache():
        try:
            processor = _get_processor()
            activity_type = request.args.get("activity_type")
            cache = processor.storage.get_latest_memory_cache(activity_type=activity_type)
            if cache is None:
                return err("未找到记忆缓存", 404)
            return ok(memory_cache_to_dict(cache))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/memory-caches/<cache_id>/text", methods=["GET"])
    def find_memory_cache_text(cache_id: str):
        try:
            processor = _get_processor()
            text = processor.storage.get_memory_cache_text(cache_id)
            if text is None:
                return err(f"未找到记忆缓存或原文: {cache_id}", 404)
            return ok({"cache_id": cache_id, "text": text})
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/memory-caches/<cache_id>", methods=["GET"])
    def find_memory_cache(cache_id: str):
        try:
            processor = _get_processor()
            cache = processor.storage.load_memory_cache(cache_id)
            if cache is None:
                return err(f"未找到记忆缓存: {cache_id}", 404)
            return ok(memory_cache_to_dict(cache))
        except Exception as e:
            return err(str(e), 500)

    @app.route("/api/v1/find/memory-caches/<cache_id>/doc", methods=["GET"])
    def find_memory_cache_doc(cache_id: str):
        """获取记忆缓存对应的完整文档内容（原文 + 缓存摘要）。"""
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
        "entity_post_enhancement", "prompt_memory_cache_max_chars",
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
    parser = argparse.ArgumentParser(description="Temporal_Memory_Graph 自然语言记忆图 API（Remember + Find）")
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
        from server.dashboard import TMGDashboard
        system_monitor.event_log.info("System", "监控面板已启用；任务细节日志已收敛为总览。")
        dashboard = TMGDashboard(system_monitor, refresh_interval=monitor_refresh)
        dashboard.start()
    else:
        stats = system_monitor.graph_detail("default")
        entities = stats["storage"]["entities"] if stats else 0
        relations = stats["storage"]["relations"] if stats else 0
        caches = stats["storage"]["memory_caches"] if stats else 0
        print(f"""
╔══════════════════════════════════════════════════════════╗
║     Temporal_Memory_Graph — 自然语言记忆图 API           ║
╚══════════════════════════════════════════════════════════╝

  当前大脑记忆库 (default):
    实体: {entities}  关系: {relations}  记忆缓存: {caches}

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
