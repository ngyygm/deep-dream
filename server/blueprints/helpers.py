"""
Shared helpers for all blueprint modules.

Provides access to the request-scoped processor, queue, and graph_id,
as well as common response helpers and serialization functions.
"""
from __future__ import annotations

import time
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from flask import jsonify, request

from processor.models import Entity, Episode, Relation
from processor.perf import _perf_timer

logger = logging.getLogger(__name__)


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
        logger.error("API error (%d): %s", status, message, exc_info=True)
    out: Dict[str, Any] = {"success": False, "error": message}
    try:
        if hasattr(request, "start_time"):
            out["elapsed_ms"] = round((time.time() - request.start_time) * 1000, 2)
    except RuntimeError:
        pass
    return jsonify(out), status


# ── Serialization helpers ─────────────────────────────────────────────────

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
        "episode_id": getattr(r, "episode_id", None),
        "source_document": getattr(r, "source_document", "") or getattr(r, "doc_name", "") or "",
        "doc_name": getattr(r, "source_document", "") or getattr(r, "doc_name", "") or "",
        "relation_type": getattr(r, "relation_type", None),
        "summary": getattr(r, "summary", None),
        "attributes": getattr(r, "attributes", None),
        "confidence": getattr(r, "confidence", None),
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


# ── Request-scoped accessors ─────────────────────────────────────────────

def _get_graph_id() -> str:
    """获取当前请求的 graph_id（由 before_request 解析）。"""
    return request.graph_id


def _get_processor():
    """获取当前请求对应的 Processor。"""
    return request.app.config["registry"].get_processor(request.graph_id)


def _get_queue():
    """获取当前请求对应的 RememberTaskQueue。"""
    return request.app.config["registry"].get_queue(request.graph_id)


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
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
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
    scored.sort(key=lambda item: (item[0], item[1], -_normalize_time_for_compare(item[2].processed_time).timestamp()))
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
            relation_absolute_ids.add(r.absolute_id)

    if time_after_dt:
        relation_absolute_ids = {
            r_abs_id for r_abs_id in relation_absolute_ids
            if True  # Will filter below
        }

    return entity_absolute_ids, relation_absolute_ids
