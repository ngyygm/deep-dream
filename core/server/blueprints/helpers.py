"""
Shared helpers for all blueprint modules.

Provides access to the request-scoped processor, queue, and graph_id,
as well as common response helpers and serialization functions.
"""
from __future__ import annotations

import asyncio
import time
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from flask import current_app, jsonify, request

_BOOL_TRUE = frozenset(("1", "true", "yes", "on"))
_BOOL_FALSE = frozenset(("0", "false", "no", "off"))

from core.models import Entity, Episode, Relation
from core.content_schema import parse_markdown_sections
from core.perf import _perf_timer

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


# ── Safe endpoint decorator ─────────────────────────────────────────────────

def safe_endpoint(func):
    """Decorator that standardizes error handling for blueprint endpoints.

    - ValueError / TypeError → 400 with the message (client errors)
    - Other exceptions → 500 with generic message (internal errors, details logged server-side)

    Usage:
        @entities_bp.route("/api/v1/...")
        @safe_endpoint
        def my_endpoint():
            ...
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, TypeError) as e:
            return err(str(e), 400)
        except Exception as e:
            logger.exception("Unhandled error in %s: %s", func.__name__, e)
            return err(str(e), 500)

    return wrapper


# ── Response helpers ──────────────────────────────────────────────────────

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
        # Security: Log full error details server-side, but don't expose them to client
        logger.error("API error (%d): %s", status, message, exc_info=True)
        # Sanitize error message for client - don't expose internal details
        message = "Internal server error. Please check the logs for details."
    else:
        # For 4xx errors, log at warning level
        logger.warning("API error (%d): %s", status, message)
    out: Dict[str, Any] = {"success": False, "error": message}
    try:
        if hasattr(request, "start_time"):
            out["elapsed_ms"] = round((time.time() - request.start_time) * 1000, 2)
    except RuntimeError:
        pass
    return jsonify(out), status


# ── Async sync bridge ────────────────────────────────────────────────────

# Module-level shared event loop for running async functions from sync Flask routes.
# Avoids creating and destroying a new loop per request.
_shared_loop: Optional[asyncio.AbstractEventLoop] = None


def run_async(coro):
    """Run an async coroutine from synchronous Flask route handlers.

    Uses a shared event loop to avoid creating/destroying per-request,
    which is wasteful and can leak resources on exceptions.
    """
    global _shared_loop
    if _shared_loop is None or _shared_loop.is_closed():
        _shared_loop = asyncio.new_event_loop()
    return _shared_loop.run_until_complete(coro)


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
        "summary": e.summary,
        "attributes": e.attributes,
        "confidence": e.confidence,
        "community_id": e.community_id,
        "valid_at": _fd(e.valid_at),
        "invalid_at": _fd(e.invalid_at),
    }
    if _score is not None:
        d["_score"] = round(_score, 4)
    if version_count is not None:
        d["version_count"] = version_count
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
        "content": r.content,
        "event_time": _fd(r.event_time),
        "processed_time": _fd(r.processed_time),
        "episode_id": r.episode_id,
        "source_document": _src_doc,
        "doc_name": _src_doc,
        "relation_type": getattr(r, "relation_type", None),
        "summary": r.summary,
        "attributes": r.attributes,
        "confidence": r.confidence,
        "valid_at": _fd(r.valid_at),
        "invalid_at": _fd(r.invalid_at),
    }
    if _score is not None:
        d["_score"] = round(_score, 4)
    if version_count is not None:
        d["version_count"] = version_count
    return d


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


def episode_to_dict(c: Episode) -> Dict[str, Any]:
    return {
        "id": c.absolute_id,  # 向后兼容
        "absolute_id": c.absolute_id,
        "content": c.content,
        "source_text": getattr(c, "source_text", "") or "",
        "event_time": c.event_time.isoformat() if c.event_time else None,
        "processed_time": c.processed_time.isoformat() if hasattr(c, 'processed_time') and c.processed_time else None,
        "source_document": getattr(c, "source_document", "") or getattr(c, "doc_name", "") or "",
        "doc_name": getattr(c, "source_document", "") or getattr(c, "doc_name", "") or "",
        "activity_type": getattr(c, "activity_type", None),
        "episode_type": getattr(c, "episode_type", None),
    }


# ── Request-scoped accessors ─────────────────────────────────────────────

def _get_graph_id() -> str:
    """获取当前请求的 graph_id（由 before_request 解析）。"""
    return request.graph_id


def _get_processor():
    """获取当前请求对应的 Processor。"""
    return current_app.config["registry"].get_processor(request.graph_id)


def _get_queue():
    """获取当前请求对应的 RememberTaskQueue。"""
    return current_app.config["registry"].get_queue(request.graph_id)


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
    def _sort_key(item):
        pt = item[2].processed_time
        ts = _normalize_time_for_compare(pt).timestamp() if pt else 0.0
        return (item[0], item[1], -ts)

    scored.sort(key=_sort_key)
    return scored


def _extract_candidate_ids(
    storage: Any,
    body: Dict[str, Any],
) -> Tuple[Set[str], Set[str]]:
    """按 query_text / 时间等条件从主图抽取实体与关系的 absolute id 集合。"""
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
        elif time_before_dt:
            entities = storage.get_all_entities_before_time(time_before_dt, limit=max_entities, exclude_embedding=True)
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
        rel_time_map: Dict[str, float] = {}
        for r in relations:
            relation_absolute_ids.add(r.absolute_id)
            if r.processed_time:
                rel_time_map[r.absolute_id] = _normalize_time_for_compare(r.processed_time).timestamp()

    if time_after_dt:
        after_ts = _normalize_time_for_compare(time_after_dt).timestamp()
        relation_absolute_ids = {
            r_abs_id for r_abs_id in relation_absolute_ids
            if rel_time_map.get(r_abs_id, 0.0) >= after_ts
        }

    return entity_absolute_ids, relation_absolute_ids
