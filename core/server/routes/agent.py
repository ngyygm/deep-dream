"""
Agent routes — Agent 辅助检索/查询端点。

P4.3 自 concepts.py 纯移动而来（URL/方法/状态码/响应字段逐一不变）；
本蓝图经 concepts_bp.register_blueprint(agent_bp) 嵌套挂载（见
concepts.py），对 URL map 无影响。
"""
from __future__ import annotations

import logging
import sqlite3

from flask import Blueprint

from core.server.routes.helpers import (
    ok,
    err,
    _get_processor,
    _get_graph_id,
    get_json_body,
)

logger = logging.getLogger(__name__)

agent_bp = Blueprint("agent", __name__)


@agent_bp.route("/api/v1/agent/sql", methods=["POST"])
def agent_read_sql():
    """Agent-facing graph-local read-only SQL workbench."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "read_sql"):
            return err("当前存储后端不支持 Agent SQL 查询", 400)
        body = get_json_body()
        sql = (body.get("sql") or "").strip()
        if not sql:
            return err("sql 不能为空", 400)
        params = body.get("params")
        try:
            limit = min(max(int(body.get("limit", 200)), 1), 10000)
        except (ValueError, TypeError):
            return err("limit 必须为整数", 400)
        try:
            timeout_seconds = float(body.get("timeout_seconds", 5.0))
        except (ValueError, TypeError):
            return err("timeout_seconds 必须为数字", 400)
        if timeout_seconds <= 0 or timeout_seconds > 60:
            return err("timeout_seconds 必须在 0-60 之间", 400)
        explain = bool(body.get("explain") or body.get("include_query_plan"))
        result = storage.read_sql(
            sql,
            params=params,
            limit=limit,
            timeout_seconds=timeout_seconds,
            include_query_plan=explain,
        )
        result["graph_id"] = _get_graph_id()
        return ok(result)
    except (ValueError, TypeError, sqlite3.Error, TimeoutError) as exc:
        return err(str(exc), 400)
    except Exception as exc:
        logger.exception("Agent SQL query failed: %s", exc)
        return err("Agent SQL 查询失败", 500)


@agent_bp.route("/api/v1/agent/semantic-search", methods=["POST"])
def agent_semantic_search():
    """Agent-facing semantic candidate recall helper."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "agent_semantic_search"):
            return err("当前存储后端不支持 Agent 语义检索", 400)
        body = get_json_body()
        role = body.get("role") or None
        try:
            top_k = min(max(int(body.get("top_k", body.get("limit", 20))), 1), 1000)
        except (ValueError, TypeError):
            return err("top_k/limit 必须为整数", 400)
        try:
            threshold = float(body.get("threshold", 0.3))
        except (ValueError, TypeError):
            return err("threshold 必须为数字", 400)
        if not (0.0 <= threshold <= 1.0):
            return err("threshold 必须在 0-1 之间", 400)
        result = storage.agent_semantic_search(
            body.get("query") or "",
            role=role,
            top_k=top_k,
            threshold=threshold,
            source_document=(body.get("source_document") or "").strip() or None,
        )
        result["graph_id"] = _get_graph_id()
        return ok(result)
    except (ValueError, TypeError) as exc:
        return err(str(exc), 400)
    except Exception as exc:
        logger.exception("Agent semantic search failed: %s", exc)
        return err("Agent 语义检索失败", 500)
