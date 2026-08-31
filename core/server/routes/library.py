"""Library 路由：scope 沙箱圈范围 + ingest 文件/文本入库。

- ``POST /api/v1/scope``   图限定文档范围（可选物化沙箱），实现见
  :func:`core.find.scope.build_document_scope`
- ``POST /api/v1/ingest``  统一入库（prose 全管线 / log 零 LLM 快速通道），
  实现见 :func:`core.ingest.ingest_text`
"""
from __future__ import annotations

import logging
from pathlib import Path

from flask import Blueprint, current_app, request

from core.server.routes.helpers import (
    _get_processor,
    get_json_body,
    err,
    ok,
)

logger = logging.getLogger(__name__)

library_bp = Blueprint("library", __name__)

_VALID_SCOPE_MODES = ("bm25", "semantic", "hybrid")
_VALID_PROFILES = ("prose", "log")


def _body_int(body: dict, field: str, default: int, lo: int, hi: int) -> int:
    value = body.get(field, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} 必须为整数")
    if not (lo <= value <= hi):
        raise ValueError(f"{field} 需在 {lo}-{hi} 之间")
    return value


def _body_bool(body: dict, field: str, default: bool = False) -> bool:
    value = body.get(field, default)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} 必须为布尔值")


# =========================================================
# Scope — 图限定文档沙箱
# =========================================================

@library_bp.route("/api/v1/scope", methods=["POST"])
def build_scope():
    """用检索 + 图回溯圈出与查询相关的有界文档范围（可选物化沙箱）。"""
    from core.find.scope import build_document_scope, materialize_scope

    try:
        processor = _get_processor()
        storage = processor.storage
        body = get_json_body()
        query = str(body.get("query") or "").strip()
        if not query:
            return err("query 不能为空", 400)
        if len(query) > 4096:
            return err("query 过长（最多 4096 个字符）", 400)
        mode = str(body.get("mode") or "hybrid").strip().lower()
        if mode not in _VALID_SCOPE_MODES:
            return err(f"mode '{mode}' 无效，可选: {', '.join(_VALID_SCOPE_MODES)}", 400)
        max_concepts = _body_int(body, "max_concepts", 20, 1, 100)
        max_docs = _body_int(body, "max_docs", 30, 1, 200)
        materialize = _body_bool(body, "materialize")

        result = build_document_scope(
            storage, query, mode=mode,
            max_concepts=max_concepts, max_docs=max_docs,
        )
        if materialize:
            registry = current_app.config["registry"]
            root = str(Path(registry.graph_dir(request.graph_id)) / "sandboxes")
            result["sandbox"] = materialize_scope(result, root)
        return ok(result)
    except ValueError as exc:
        return err(str(exc), 400)
    except Exception as exc:  # noqa: BLE001 - 路由边界统一报错
        logger.exception("scope failed")
        return err(f"scope failed: {exc}", 500)


# =========================================================
# Ingest — 统一入库入口
# =========================================================

@library_bp.route("/api/v1/ingest", methods=["POST"])
def ingest():
    """统一入库：prose 走完整 remember 管线，log 走零 LLM 快速通道。"""
    from core.ingest import ingest_text

    try:
        body = get_json_body()
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return err("text 不能为空", 400)
        if len(text) > 4_000_000:
            return err("text 过长（最多 4M 字符）", 400)
        name = str(body.get("name") or "").strip()
        if not name:
            return err("name 不能为空", 400)
        if len(name) > 512:
            return err("name 过长（最多 512 个字符）", 400)
        profile = str(body.get("profile") or "prose").strip().lower()
        if profile not in _VALID_PROFILES:
            return err(f"profile '{profile}' 无效，可选: {', '.join(_VALID_PROFILES)}", 400)

        kwargs = {}
        if profile == "log":
            kwargs["time_window_s"] = float(_body_int(body, "time_window_s", 300, 1, 86400))
            kwargs["line_window"] = _body_int(body, "line_window", 400, 1, 10000)
            kwargs["distill"] = _body_bool(body, "distill", True)
            # log 快速通道零 LLM，只需要 storage。
            processor = _get_processor()
            report = ingest_text(
                text, name, profile, storage=processor.storage,
                graph_id=request.graph_id, **kwargs,
            )
        else:
            processor = _get_processor()
            report = ingest_text(
                text, name, profile, processor=processor,
                graph_id=request.graph_id, **kwargs,
            )
        return ok(report)
    except ValueError as exc:
        return err(str(exc), 400)
    except Exception as exc:  # noqa: BLE001 - 路由边界统一报错
        logger.exception("ingest failed")
        return err(f"ingest failed: {exc}", 500)
