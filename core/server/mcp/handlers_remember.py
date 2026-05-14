#!/usr/bin/env python3
"""
Remember handlers for Deep Dream MCP Server.

Handles: remember, remember_tasks, remember_task_status, delete_remember_task,
         pause_remember_task, resume_remember_task, remember_monitor
"""

from .transport import _get, _post, _delete
from .response_format import _result, _hint, _inner
from .dispatch_helpers import _arg, _req


def remember(args):
    text = _req(args, "content")
    if len(text.strip()) < 5:
        raise ValueError("content is too short to extract meaningful entities (minimum 5 characters). Combine with surrounding context and retry.")
    body = {"text": text}
    if _arg(args, "source"):
        body["source_name"] = args["source"]
    if _arg(args, "metadata"):
        body["metadata"] = args["metadata"]
    data, code = _post("/api/v1/remember", body)
    if code < 400 and isinstance(data, dict):
        task_id = _inner(data).get("task_id", "")
        if task_id:
            _hint(data, f"\n→ Poll with remember_task_status(task_id='{task_id}') to check extraction progress.")
    return _result(data, code)


def remember_tasks(args):
    qp = {}
    if _arg(args, "status"):
        qp["status"] = args["status"]
    data, code = _get("/api/v1/remember/tasks", **qp)
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        tasks = inner.get("tasks", [])
        if isinstance(tasks, list) and tasks:
            # Compact tasks to save tokens
            for i, t in enumerate(tasks):
                if isinstance(t, dict):
                    tasks[i] = {k: v for k, v in t.items() if k in ("task_id", "status", "source_name", "phase", "progress", "created_at")}
            first = tasks[0] if isinstance(tasks[0], dict) else {}
            tid = first.get("task_id", "")
            status = first.get("status", "")
            if tid:
                _hint(data, f"\n→ {len(tasks)} task(s). Latest: {status}. Check with remember_task_status(task_id='{tid}').")
        elif isinstance(tasks, list) and not tasks:
            _hint(data, "\n→ No tasks in queue. Use remember(content='...') to submit text for extraction.")
    return _result(data, code)


def remember_task_status(args):
    data, code = _get(f"/api/v1/remember/tasks/{args['task_id']}")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        status = inner.get("status", "")
        if status == "completed":
            parts = ["Extraction complete."]
            # Extract entity names from result if available
            result = inner.get("result", {})
            if isinstance(result, dict):
                entities = result.get("entities", result.get("new_entities", []))
                if isinstance(entities, int):
                    # remember_text() returns integer counts
                    relations = result.get("relations", 0)
                    chunks = result.get("chunks_processed", "?")
                    parts.append(f"Extracted {entities} entities, {relations} relations from {chunks} chunk(s).")
                elif isinstance(entities, list) and entities:
                    names = [e.get("name", "") for e in entities if isinstance(e, dict) and e.get("name")]
                    if names:
                        sample = ", ".join(names[:5])
                        suffix = f" (+{len(names)-5} more)" if len(names) > 5 else ""
                        parts.append(f"Entities: {sample}{suffix}.")
            parts.append("Use quick_search or graph_summary to explore.")
            _hint(data, "\n→ " + " ".join(parts))
        elif status in ("pending", "processing"):
            phase = inner.get("phase", "")
            progress = inner.get("progress", "")
            phase_info = f" (phase: {phase})" if phase else ""
            progress_info = f" ({progress})" if progress else ""
            _hint(data, f"\n→ Still {status}{phase_info}{progress_info}. Poll again with remember_task_status(task_id='{args['task_id']}').")
    return _result(data, code)


def delete_remember_task(args):
    return _result(*_delete(f"/api/v1/remember/tasks/{args['task_id']}"))


def pause_remember_task(args):
    return _result(*_post(f"/api/v1/remember/tasks/{args['task_id']}/pause"))


def resume_remember_task(args):
    return _result(*_post(f"/api/v1/remember/tasks/{args['task_id']}/resume"))


def remember_monitor(args):
    data, code = _get("/api/v1/remember/monitor")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        pending = inner.get("pending", 0)
        processing = inner.get("processing", 0)
        if pending or processing:
            _hint(data, f"\n→ {pending} pending, {processing} processing. Use remember_tasks(status='pending') to view queue.")
    return _result(data, code)


def register_handlers(tool_map):
    tool_map["remember"] = remember
    tool_map["remember_tasks"] = remember_tasks
    tool_map["remember_task_status"] = remember_task_status
    tool_map["delete_remember_task"] = delete_remember_task
    tool_map["pause_remember_task"] = pause_remember_task
    tool_map["resume_remember_task"] = resume_remember_task
    tool_map["remember_monitor"] = remember_monitor
