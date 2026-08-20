"""
Shared helpers for all route modules.

Provides access to the request-scoped processor, queue, and graph_id,
as well as common response helpers and serialization functions.
"""
from __future__ import annotations

import time
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from flask import current_app, jsonify, request

from core.find.hybrid import HybridSearcher

from core.models import Entity, Relation
from core.content_schema import parse_markdown_sections

_BOOL_TRUE = frozenset(("1", "true", "yes", "on"))
_BOOL_FALSE = frozenset(("0", "false", "no", "off"))

logger = logging.getLogger(__name__)


# ── Validation helpers ─────────────────────────────────────────────────────

def _validate_graph_id(graph_id):
    """Validate graph_id is a safe string (no path traversal).

    Raises:
        ValueError: If graph_id is invalid
    """
    if not graph_id or not isinstance(graph_id, str):
        raise ValueError("graph_id is required")
    if '/' in graph_id or '\\' in graph_id or '..' in graph_id:
        raise ValueError("Invalid graph_id")
    return graph_id


def _validate_text_input(text, field_name="text", min_len=1, max_len=100000):
    """Validate text input.

    Args:
        text: The text to validate
        field_name: Name of the field for error messages
        min_len: Minimum length (default 1)
        max_len: Maximum length (default 100000)

    Raises:
        ValueError: If text is invalid
    """
    if not text or not isinstance(text, str):
        raise ValueError(f"{field_name} is required")
    if len(text.strip()) < min_len:
        raise ValueError(f"{field_name} must be at least {min_len} characters")
    if len(text) > max_len:
        raise ValueError(f"{field_name} must be at most {max_len} characters")
    return text


def get_json_body():
    """Parse JSON body with malformed-JSON detection.

    Returns parsed dict, or empty dict if no body.
    Raises ValueError if request has body content that isn't valid JSON.
    """
    if not request.data:
        return {}
    body = request.get_json(silent=True)
    if body is not None:
        return body
    # Body was present but not valid JSON
    raise ValueError("请求体不是有效的 JSON（请检查格式）")


def _validate_positive_int(value, field_name="value"):
    """Validate positive integer.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages

    Raises:
        ValueError: If value is not a positive integer
    """
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a positive integer")
    if v <= 0:
        raise ValueError(f"{field_name} must be positive")
    return v


# ── Response helpers ──────────────────────────────────────────────────────

def ok(data: Any) -> tuple:
    out: Dict[str, Any] = {"success": True, "data": data}
    try:
        if hasattr(request, "start_time"):
            out["elapsed_ms"] = round((time.time() - request.start_time) * 1000, 2)
    except RuntimeError:
        pass
    return jsonify(out), 200


def err(message: str, status: int = 400, hint: str = None) -> tuple:
    if status >= 500:
        # 501 (Not Implemented) and 503 (Service Unavailable) are operational
        # errors with user-facing messages -- don't sanitize them.
        # 500 (Internal Server Error) is sanitized to avoid leaking internals.
        if status == 500:
            # Security: Log full error details server-side, but don't expose them to client
            logger.error("API error (%d): %s", status, message, exc_info=True)
            # Sanitize error message for client - don't expose internal details
            message = "Internal server error. Please check the logs for details."
        else:
            # 501, 503, etc. -- log but preserve the message for the client
            logger.warning("API error (%d): %s", status, message)
    else:
        # For 4xx errors, log at warning level
        logger.warning("API error (%d): %s", status, message)
    out: Dict[str, Any] = {"success": False, "error": message}
    # Use explicit hint if provided, otherwise auto-detect from error message
    if hint:
        out["hint"] = hint
    else:
        from core.server.agent_api import error_hint
        _hint = error_hint(message)
        if _hint:
            out["hint"] = _hint
    try:
        if hasattr(request, "start_time"):
            out["elapsed_ms"] = round((time.time() - request.start_time) * 1000, 2)
    except RuntimeError:
        pass
    return jsonify(out), status


# ── Serialization helpers ─────────────────────────────────────────────────

def _fmt_dt(dt) -> Optional[str]:
    """Fast datetime→isoformat with None guard. Inlined pattern avoids per-call method lookup."""
    return dt.isoformat() if dt is not None else None


def entity_to_dict(e: Entity, max_content_length: int = 2000,
                   _score: Optional[float] = None,
                   version_count: Optional[int] = None,
                   skip_sections: bool = False) -> Dict[str, Any]:
    # Only parse markdown sections for markdown-format content (skip regex on plain text)
    # List endpoints pass skip_sections=True to avoid per-entity regex overhead
    _fmt = e.content_format
    sections = {} if skip_sections else (parse_markdown_sections(e.content) if e.content and _fmt == "markdown" else {})
    content = e.content or ""
    truncated = len(content) > max_content_length
    content_display = content[:max_content_length] + ("..." if truncated else "")
    _src_doc = e.source_document or getattr(e, 'doc_name', '') or ""
    _fd = _fmt_dt
    d: Dict[str, Any] = {
        "id": e.absolute_id,  # 向后兼容
        "absolute_id": e.absolute_id,
        "family_id": e.family_id,
        "name": e.name,
        "content": content_display,
        "content_truncated": truncated,
        "content_format": _fmt,
        "content_sections": sections if sections else None,
        "event_time": _fd(e.event_time),
        "processed_time": _fd(e.processed_time),
        "episode_id": e.episode_id,
        "source_document": _src_doc,
        "doc_name": _src_doc,
        "attributes": e.attributes,
        "confidence": e.confidence,
        "community_id": e.community_id,
        "valid_at": _fd(e.valid_at),
        "version_seq": e.version_seq,
    }
    if _score is not None:
        d["_score"] = round(_score, 4)
    vc = version_count if version_count is not None else getattr(e, 'version_count', None)
    if vc is not None:
        d["version_count"] = vc
    return d


def relation_to_dict(r: Relation, _score: Optional[float] = None,
                     version_count: Optional[int] = None) -> Dict[str, Any]:
    _src_doc = r.source_document or getattr(r, 'doc_name', '') or ""
    _fd = _fmt_dt
    d: Dict[str, Any] = {
        "id": r.absolute_id,  # 向后兼容
        "absolute_id": r.absolute_id,
        "family_id": r.family_id,
        "entity1_absolute_id": r.entity1_absolute_id,
        "entity2_absolute_id": r.entity2_absolute_id,
        "entity1_family_id": getattr(r, "entity1_family_id", "") or "",
        "entity2_family_id": getattr(r, "entity2_family_id", "") or "",
        "content": r.content,
        "event_time": _fd(r.event_time),
        "processed_time": _fd(r.processed_time),
        "episode_id": r.episode_id,
        "source_document": _src_doc,
        "doc_name": _src_doc,
        "relation_type": getattr(r, "relation_type", None),
        "attributes": r.attributes,
        "confidence": r.confidence,
        "valid_at": _fd(r.valid_at),
        "version_seq": r.version_seq,
    }
    if _score is not None:
        d["_score"] = round(_score, 4)
    if version_count is not None:
        d["version_count"] = version_count
    return d


def enrich_relations(relations_dicts, processor):
    """为关系列表补充 entity1_name / entity2_name 及缺失的 family_id。"""
    abs_ids = set()
    needs_family_id = False
    for rd in relations_dicts:
        if rd.get('entity1_absolute_id'):
            abs_ids.add(rd['entity1_absolute_id'])
        if rd.get('entity2_absolute_id'):
            abs_ids.add(rd['entity2_absolute_id'])
        if not rd.get('entity1_family_id') or not rd.get('entity2_family_id'):
            needs_family_id = True
    if not abs_ids:
        return relations_dicts
    name_map = processor.storage.get_entity_names_by_absolute_ids(list(abs_ids))
    fid_map = {}
    if needs_family_id and hasattr(processor.storage, 'get_family_ids_by_absolute_ids'):
        fid_map = processor.storage.get_family_ids_by_absolute_ids(list(abs_ids))
    for rd in relations_dicts:
        rd['entity1_name'] = name_map.get(rd.get('entity1_absolute_id'), '')
        rd['entity2_name'] = name_map.get(rd.get('entity2_absolute_id'), '')
        if needs_family_id:
            if not rd.get('entity1_family_id'):
                rd['entity1_family_id'] = fid_map.get(rd.get('entity1_absolute_id'), '')
            if not rd.get('entity2_family_id'):
                rd['entity2_family_id'] = fid_map.get(rd.get('entity2_absolute_id'), '')
    return relations_dicts


def enrich_entity_version_counts(entity_dicts, storage):
    """批量补充实体 version_count（按 family_id 批量查询）。"""
    family_ids = [d["family_id"] for d in entity_dicts if d.get("family_id")]
    if not family_ids:
        return entity_dicts
    counts = storage.get_entity_version_counts(family_ids)
    for d in entity_dicts:
        fid = d.get("family_id")
        if fid and fid in counts:
            d["version_count"] = counts[fid]
    return entity_dicts


def enrich_relation_version_counts(relation_dicts, storage):
    """批量补充关系 version_count（按 family_id 批量查询）。"""
    family_ids = [d["family_id"] for d in relation_dicts if d.get("family_id")]
    if not family_ids:
        return relation_dicts
    counts = storage.get_relation_version_counts(family_ids)
    for d in relation_dicts:
        fid = d.get("family_id")
        if fid and fid in counts:
            d["version_count"] = counts[fid]
    return relation_dicts


# ── Request-scoped accessors ─────────────────────────────────────────────

def _get_graph_id() -> str:
    """获取当前请求的 graph_id（由 before_request 解析）。"""
    return request.graph_id


def _get_processor():
    """获取当前请求对应的 Processor（带重试以应对瞬态连接问题）。"""
    return current_app.config["registry"].get_processor_with_retry(request.graph_id)


def _get_queue():
    """获取当前请求对应的 RememberTaskQueue。"""
    return current_app.config["registry"].get_queue(request.graph_id)


def _get_searcher(storage):
    """Get or create a cached HybridSearcher for the given storage."""
    searcher = getattr(storage, '_hybrid_searcher', None)
    if searcher is None:
        searcher = HybridSearcher(storage)
        storage._hybrid_searcher = searcher
    return searcher


def _get_system_monitor():
    """Get the SystemMonitor from app config."""
    return current_app.config.get("system_monitor")


# ── Time parsing helpers ─────────────────────────────────────────────────

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


def _parse_bool_query(name: str) -> Optional[bool]:
    v = request.args.get(name)
    if v is None or v == "":
        return None
    s = v.strip().lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    return None
