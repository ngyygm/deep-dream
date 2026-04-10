"""Remember pipeline routes — POST /api/v1/remember and task management."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from flask import Blueprint, current_app, jsonify, make_response, request

from server.blueprints.helpers import (
    _get_processor,
    _get_queue,
    _parse_bool_query,
    err,
    ok,
)
from server.monitor import LOG_MODE_DETAIL
from server.task_queue import RememberTask

logger = logging.getLogger(__name__)

remember_bp = Blueprint("remember", __name__)


def _get_system_monitor():
    """Retrieve the SystemMonitor stored on the Flask app config."""
    return current_app.config.get("system_monitor")


# ── POST /api/v1/remember ─────────────────────────────────────────────────

@remember_bp.route("/api/v1/remember", methods=["POST"])
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

        # 以"首次接收请求的时间"为基准：若未传 event_time，则使用当前接收时间并持久化到 journal。
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
        system_monitor = _get_system_monitor()
        if system_monitor is not None:
            system_monitor.event_log.info(
                "Remember",
                f"收到({request.method}): source_name={source_name!r}, "
                f"文本长度={len(text)} 字符, event_time={event_time_display}"
            )
            if system_monitor.mode == LOG_MODE_DETAIL:
                system_monitor.event_log.info("Remember", f"内容预览: {preview!r}")
        else:
            logger.debug(
                "[Remember] 收到(%s): source_name=%r, 文本长度=%d 字符, event_time=%s",
                request.method, source_name, len(text), event_time_display,
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


# ── GET/DELETE /api/v1/remember/tasks/<task_id> ───────────────────────────

@remember_bp.route("/api/v1/remember/tasks/<task_id>", methods=["GET", "DELETE"])
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


# ── POST /api/v1/remember/tasks/<task_id>/pause ───────────────────────────

@remember_bp.route("/api/v1/remember/tasks/<task_id>/pause", methods=["POST"])
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


# ── POST /api/v1/remember/tasks/<task_id>/resume ──────────────────────────

@remember_bp.route("/api/v1/remember/tasks/<task_id>/resume", methods=["POST"])
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


# ── GET /api/v1/remember/tasks ────────────────────────────────────────────

@remember_bp.route("/api/v1/remember/tasks", methods=["GET"])
def remember_queue_list():
    """查看记忆写入任务队列；推荐使用 /api/v1/remember/tasks。"""
    try:
        remember_queue = _get_queue()
        limit = min(request.args.get("limit", 50, type=int), 200)
        tasks = remember_queue.list_tasks(limit=limit)
        return ok({"tasks": tasks, "count": len(tasks)})
    except Exception as e:
        return err(str(e), 500)


# ── GET /api/v1/remember/monitor ──────────────────────────────────────────

@remember_bp.route("/api/v1/remember/monitor", methods=["GET"])
def remember_monitor():
    """返回 remember 的实时监控快照，适合 watch 或外部面板轮询。"""
    try:
        system_monitor = _get_system_monitor()
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
