"""
Entities blueprint — Entity CRUD, search, relations, timeline, intelligence.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re as _re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import Blueprint, current_app, request

from processor.models import Entity
from processor.perf import _perf_timer
from processor.search.hybrid import HybridSearcher

logger = logging.getLogger(__name__)

entities_bp = Blueprint("entities", __name__)


# ── Shared helpers (imported from helpers) ─────────────────────────────────

def _get_processor():
    return current_app.config["registry"].get_processor(request.graph_id)


def ok(data: Any) -> tuple:
    from server.blueprints.helpers import ok as _ok
    return _ok(data)


def err(message: str, status: int = 400) -> tuple:
    from server.blueprints.helpers import err as _err
    return _err(message, status)


def _helpers():
    """Lazy import to avoid circular deps."""
    import server.blueprints.helpers as h
    return h


# ── Entity listing ────────────────────────────────────────────────────────

@entities_bp.route("/api/v1/find/entities", methods=["GET"])
def find_entities_all():
    try:
        processor = _get_processor()
        limit = min(request.args.get("limit", type=int) or 500, 500)
        offset = request.args.get("offset", type=int, default=0) or 0
        total = processor.storage.count_unique_entities()
        entities = processor.storage.get_all_entities(limit=limit, offset=offset if offset > 0 else None, exclude_embedding=True)
        h = _helpers()
        return ok({
            "entities": [h.entity_to_dict(e) for e in entities],
            "total": total,
            "offset": offset,
            "limit": limit,
        })
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/as-of-time", methods=["GET"])
def find_entities_all_before_time():
    try:
        processor = _get_processor()
        h = _helpers()
        time_point_str = request.args.get("time_point")
        if not time_point_str:
            return err("time_point 为必填参数（ISO 格式）", 400)
        try:
            time_point = h.parse_time_point(time_point_str)
        except ValueError as ve:
            return err(str(ve), 400)
        limit = request.args.get("limit", type=int)
        entities = processor.storage.get_all_entities_before_time(time_point, limit=limit, exclude_embedding=True)
        return ok([h.entity_to_dict(e) for e in entities])
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/version-counts", methods=["POST"])
def find_entity_version_counts():
    try:
        processor = _get_processor()
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            body = {}
        family_ids = body.get("family_ids")
        if not family_ids or not isinstance(family_ids, list):
            return ok({})
        family_ids = [x for x in family_ids if isinstance(x, str)]
        if not family_ids:
            return ok({})
        counts = processor.storage.get_entity_version_counts(family_ids)
        return ok(counts)
    except Exception as e:
        return err(str(e), 500)


# ── Entity by absolute_id ─────────────────────────────────────────────────

@entities_bp.route("/api/v1/find/entities/absolute/<absolute_id>/embedding-preview", methods=["GET"])
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


@entities_bp.route("/api/v1/find/entities/absolute/<absolute_id>", methods=["GET"])
def find_entity_by_absolute_id(absolute_id: str):
    try:
        processor = _get_processor()
        h = _helpers()
        entity = processor.storage.get_entity_by_absolute_id(absolute_id)
        if entity is None:
            return err(f"未找到实体版本: {absolute_id}", 404)
        return ok(h.entity_to_dict(entity))
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/absolute/<absolute_id>", methods=["PUT"])
def update_entity_absolute(absolute_id: str):
    try:
        processor = _get_processor()
        h = _helpers()
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
        return ok(h.entity_to_dict(updated))
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/absolute/<absolute_id>", methods=["DELETE"])
def delete_entity_absolute(absolute_id: str):
    try:
        processor = _get_processor()
        h = _helpers()
        blocking = processor.storage.get_relations_referencing_absolute_id(absolute_id)
        if blocking:
            blocking_dicts = [h.relation_to_dict(r) for r in blocking[:10]]
            return err(
                f"该版本仍有 {len(blocking)} 条关联关系，请先删除或重定向这些关系",
                409,
            )
        success = processor.storage.delete_entity_by_absolute_id(absolute_id)
        if not success:
            return err(f"未找到实体版本: {absolute_id}", 404)
        return ok({"absolute_id": absolute_id, "deleted": True})
    except Exception as e:
        return err(str(e), 500)


# ── Entity search ─────────────────────────────────────────────────────────

@entities_bp.route("/api/v1/find/entities/search", methods=["GET", "POST"])
def find_entities_search():
    try:
        processor = _get_processor()
        h = _helpers()
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
        return ok([h.entity_to_dict(e) for e in entities])
    except Exception as e:
        return err(str(e), 500)


# ── Entity by family_id (static paths MUST come before parameterized) ─────

@entities_bp.route("/api/v1/find/entities/create", methods=["POST"])
def create_entity():
    try:
        processor = _get_processor()
        h = _helpers()
        body = request.get_json(silent=True) or {}
        name = (body.get("name") or "").strip()
        content = (body.get("content") or "").strip()
        if not name:
            return err("name 为必填", 400)

        now = datetime.now()
        ts = now.strftime("%Y%m%d_%H%M%S")
        family_id = f"ent_{uuid.uuid4().hex[:12]}"
        absolute_id = f"entity_{ts}_{uuid.uuid4().hex[:8]}"
        for _ in range(10):
            existing = processor.storage.get_entity_by_absolute_id(absolute_id)
            if not existing:
                break
            absolute_id = f"entity_{ts}_{uuid.uuid4().hex[:8]}"
        for _ in range(10):
            existing = processor.storage.get_entity_by_family_id(family_id)
            if not existing:
                break
            family_id = f"ent_{uuid.uuid4().hex[:12]}"

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
        return ok(h.entity_to_dict(entity))
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/batch-delete", methods=["POST"])
def batch_delete_entities():
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        family_ids = body.get("family_ids") or body.get("entity_ids", [])
        if not isinstance(family_ids, list) or not family_ids:
            return err("family_ids 需为非空数组", 400)
        if len(family_ids) > 100:
            return err("单次批量删除上限 100 个", 400)
        total = processor.storage.batch_delete_entities(family_ids)
        return ok({"message": f"已删除 {total} 个实体版本", "count": len(family_ids)})
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/batch-delete-versions", methods=["POST"])
def batch_delete_entity_versions():
    try:
        processor = _get_processor()
        h = _helpers()
        body = request.get_json(silent=True) or {}
        absolute_ids = body.get("absolute_ids", [])
        if not isinstance(absolute_ids, list) or not absolute_ids:
            return err("absolute_ids 需为非空数组", 400)
        deleted = []
        blocked = {}
        blocking_map = processor.storage.batch_get_relations_referencing_absolute_ids(absolute_ids)
        to_delete = []
        for aid in absolute_ids:
            blocking = blocking_map.get(aid, [])
            if blocking:
                blocked[aid] = {
                    "blocking_count": len(blocking),
                    "blocking_relations": [h.relation_to_dict(r) for r in blocking[:5]],
                }
            else:
                to_delete.append(aid)
        if to_delete:
            batch_deleted = processor.storage.batch_delete_entity_versions_by_absolute_ids(to_delete)
            deleted_set = set(to_delete) if batch_deleted == len(to_delete) else set()
            if batch_deleted != len(to_delete):
                for aid in to_delete:
                    entity = processor.storage.get_entity_by_absolute_id(aid)
                    if not entity:
                        deleted.append(aid)
                    else:
                        blocked[aid] = {"blocking_count": 0, "reason": "未找到"}
            else:
                deleted = to_delete
        return ok({
            "deleted": deleted,
            "blocked": blocked,
            "summary": {"deleted_count": len(deleted), "blocked_count": len(blocked)},
        })
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/split-version", methods=["POST"])
def split_entity_version():
    try:
        processor = _get_processor()
        h = _helpers()
        body = request.get_json(silent=True) or {}
        absolute_id = (body.get("absolute_id") or "").strip()
        if not absolute_id:
            return err("absolute_id 为必填", 400)
        new_family_id = (body.get("new_family_id") or "").strip()
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
            "entity": h.entity_to_dict(updated),
        })
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/merge", methods=["POST"])
def merge_entities():
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        target_id = (body.get("target_family_id") or "").strip()
        source_ids = body.get("source_family_ids", [])
        if not target_id or not isinstance(source_ids, list) or not source_ids:
            return err("target_family_id 和 source_family_ids 为必填", 400)
        target = processor.storage.get_entity_by_family_id(target_id)
        if target is None:
            return err(f"目标实体不存在: {target_id}", 404)
        result = processor.storage.merge_entity_families(target_id, source_ids)
        return ok({"message": "实体合并完成", "target_family_id": target_id, "source_family_ids": source_ids, "merged_count": result})
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/isolated", methods=["GET"])
def find_isolated_entities():
    try:
        processor = _get_processor()
        h = _helpers()
        if not hasattr(processor.storage, 'get_isolated_entities'):
            return ok({"entities": [], "total": 0, "message": "当前存储后端不支持孤立实体检测"})
        limit = request.args.get("limit", type=int, default=100)
        offset = request.args.get("offset", type=int, default=0) or 0
        isolated = processor.storage.get_isolated_entities(limit=limit, offset=offset)
        total = processor.storage.count_isolated_entities()
        return ok({
            "entities": [h.entity_to_dict(e) for e in isolated],
            "total": total,
            "offset": offset,
            "limit": limit,
        })
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/delete-isolated", methods=["POST"])
def delete_isolated_entities():
    try:
        processor = _get_processor()
        h = _helpers()
        if not hasattr(processor.storage, 'get_isolated_entities'):
            return ok({"message": "当前存储后端不支持孤立实体检测", "deleted": 0})
        dry_run_body = request.get_json(silent=True) or {}
        dry_run = dry_run_body.get("dry_run", False) if isinstance(dry_run_body, dict) else False
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
        deleted = processor.storage.batch_delete_entities(family_ids)
        return ok({
            "message": f"已删除 {len(family_ids)} 个孤立实体（{deleted} 个版本）",
            "deleted_families": len(family_ids),
            "deleted_versions": deleted,
        })
    except Exception as e:
        return err(str(e), 500)


# ── Entity by family_id (parameterized — MUST come last among /entities/ routes) ─

@entities_bp.route("/api/v1/find/entities/<family_id>/versions", methods=["GET"])
def find_entity_versions(family_id: str):
    try:
        processor = _get_processor()
        h = _helpers()
        versions = processor.storage.get_entity_versions(family_id)
        return ok([h.entity_to_dict(e) for e in versions])
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/<family_id>/as-of-time", methods=["GET"])
def find_entity_at_time(family_id: str):
    try:
        processor = _get_processor()
        h = _helpers()
        time_point_str = request.args.get("time_point")
        if not time_point_str:
            return err("time_point 为必填参数（ISO 格式）", 400)
        try:
            time_point = h.parse_time_point(time_point_str)
        except ValueError as ve:
            return err(str(ve), 400)
        entity = processor.storage.get_entity_version_at_time(family_id, time_point)
        if entity is None:
            return err(f"未找到该时间点版本: {family_id}", 404)
        return ok(h.entity_to_dict(entity))
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/<family_id>/nearest-to-time", methods=["GET"])
def find_entity_nearest_to_time(family_id: str):
    try:
        processor = _get_processor()
        h = _helpers()
        time_point_str = request.args.get("time_point")
        if not time_point_str:
            return err("time_point 为必填参数（ISO 格式）", 400)
        try:
            time_point = h.parse_time_point(time_point_str)
            max_delta_seconds = h._parse_non_negative_seconds("max_delta_seconds")
        except ValueError as ve:
            return err(str(ve), 400)

        scored = h._score_entity_versions_against_time(family_id, time_point, proc=processor)
        if not scored:
            return err(f"未找到实体: {family_id}", 404)

        delta_seconds, _, entity = scored[0]
        if max_delta_seconds is not None and delta_seconds > max_delta_seconds:
            return err(f"最近版本超出允许误差: {delta_seconds:.3f}s > {max_delta_seconds:.3f}s", 404)

        return ok({
            "family_id": family_id,
            "query_time": time_point.isoformat(),
            "matched": h.entity_to_dict(entity),
            "delta_seconds": round(delta_seconds, 6),
        })
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/<family_id>/around-time", methods=["GET"])
def find_entity_around_time(family_id: str):
    try:
        processor = _get_processor()
        h = _helpers()
        time_point_str = request.args.get("time_point")
        if not time_point_str:
            return err("time_point 为必填参数（ISO 格式）", 400)
        try:
            time_point = h.parse_time_point(time_point_str)
            within_seconds = h._parse_non_negative_seconds("within_seconds")
        except ValueError as ve:
            return err(str(ve), 400)
        if within_seconds is None:
            return err("within_seconds 为必填参数（秒）", 400)

        target = h._normalize_time_for_compare(time_point)
        matches: List[Dict[str, Any]] = []
        for delta_seconds, _, entity in h._score_entity_versions_against_time(family_id, time_point, proc=processor):
            if delta_seconds > within_seconds:
                continue
            item = h.entity_to_dict(entity)
            item["delta_seconds"] = round(delta_seconds, 6)
            direction = h._normalize_time_for_compare(entity.event_time) - target
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


@entities_bp.route("/api/v1/find/entities/<family_id>/version-count", methods=["GET"])
def find_entity_version_count(family_id: str):
    try:
        processor = _get_processor()
        count = processor.storage.get_entity_version_count(family_id)
        if count <= 0:
            return err(f"未找到实体: {family_id}", 404)
        return ok({"family_id": family_id, "version_count": count})
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/<family_id>", methods=["GET"])
def find_entity_by_family_id(family_id: str):
    try:
        processor = _get_processor()
        h = _helpers()
        entity = processor.storage.get_entity_by_family_id(family_id)
        if entity is None:
            return err(f"未找到实体: {family_id}", 404)
        return ok(h.entity_to_dict(entity))
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/<family_id>", methods=["DELETE"])
def delete_entity_family(family_id: str):
    try:
        processor = _get_processor()
        cascade = request.args.get("cascade", "false").lower() == "true"
        count = processor.storage.delete_entity_all_versions(family_id)
        if count == 0:
            return err(f"未找到实体: {family_id}", 404)
        return ok({"message": f"已删除 {count} 个实体版本", "family_id": family_id, "cascade": cascade})
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/<family_id>", methods=["PUT"])
def update_entity_v2(family_id: str):
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


# ── Entity timeline & version control ─────────────────────────────────────

@entities_bp.route("/api/v1/find/entities/<family_id>/timeline", methods=["GET"])
def find_entity_timeline(family_id: str):
    try:
        processor = _get_processor()
        h = _helpers()
        with _perf_timer("find_entity_timeline"):
            versions = processor.storage.get_entity_versions(family_id)
        if not versions:
            return err(f"未找到实体: {family_id}", 404)

        relations_timeline = []
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

        seen = set()
        unique_rels = []
        for r in relations_timeline:
            if r["absolute_id"] not in seen:
                seen.add(r["absolute_id"])
                unique_rels.append(r)

        return ok({
            "family_id": family_id,
            "versions": [h.entity_to_dict(v) for v in versions],
            "relations_timeline": unique_rels,
        })
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/<family_id>/section-history", methods=["GET"])
def entity_section_history(family_id: str):
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


@entities_bp.route("/api/v1/find/entities/<family_id>/version-diff", methods=["GET"])
def entity_version_diff(family_id: str):
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


@entities_bp.route("/api/v1/find/entities/<family_id>/patches", methods=["GET"])
def entity_patches(family_id: str):
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


# ── Entity intelligence ───────────────────────────────────────────────────

@entities_bp.route("/api/v1/find/entities/<family_id>/evolve-summary", methods=["POST"])
def evolve_entity_summary(family_id: str):
    try:
        processor = _get_processor()
        entity = processor.storage.get_entity_by_family_id(family_id)
        if entity is None:
            return err(f"未找到实体: {family_id}", 404)

        versions = processor.storage.get_entity_versions(family_id)
        old_version = versions[1] if len(versions) > 1 else None

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


@entities_bp.route("/api/v1/find/entities/<family_id>/provenance", methods=["GET"])
def get_entity_provenance(family_id: str):
    try:
        processor = _get_processor()
        provenance = processor.storage.get_entity_provenance(family_id)
        return ok(provenance)
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/<family_id>/contradictions", methods=["GET"])
def get_entity_contradictions(family_id: str):
    try:
        processor = _get_processor()
        versions = processor.storage.get_entity_versions(family_id)
        if len(versions) < 2:
            return ok([])

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


@entities_bp.route("/api/v1/find/entities/<family_id>/resolve-contradiction", methods=["POST"])
def resolve_entity_contradiction(family_id: str):
    try:
        body = request.get_json(silent=True) or {}
        contradiction = body.get("contradiction")
        if not contradiction or not isinstance(contradiction, dict):
            return err("contradiction 为必填字段", 400)

        processor = _get_processor()
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


# ── Entity neighbors (Neo4j) ──────────────────────────────────────────────

@entities_bp.route("/api/v1/find/entities/<entity_uuid>/neighbors", methods=["GET"])
def find_entity_neighbors(entity_uuid: str):
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


# ── Entity aggregation: profile ───────────────────────────────────────────

@entities_bp.route("/api/v1/find/entities/<family_id>/profile", methods=["GET"])
def entity_profile(family_id: str):
    try:
        processor = _get_processor()
        h = _helpers()
        entity = processor.storage.get_entity_by_family_id(family_id)
        if entity is None:
            return err(f"未找到实体: {family_id}", 404)
        relations = processor.storage.get_entity_relations_by_family_id(family_id)
        version_count = processor.storage.get_entity_version_count(family_id)
        rels = [h.relation_to_dict(r) for r in relations]
        h.enrich_relations(rels, processor)
        return ok({
            "entity": h.entity_to_dict(entity),
            "relations": rels,
            "relation_count": len(rels),
            "version_count": version_count,
        })
    except Exception as e:
        return err(str(e), 500)


# ── Entity lookup by name ─────────────────────────────────────────────────

@entities_bp.route("/api/v1/find/entities/by-name/<name>", methods=["GET"])
def find_entity_by_name(name: str):
    try:
        processor = _get_processor()
        h = _helpers()
        threshold = float(request.args.get("threshold", "0.5"))
        limit = int(request.args.get("limit", "5"))
        best = None

        # Step 1: Exact name match
        exact_map = processor.storage.get_family_ids_by_names([name])
        if exact_map:
            fid = list(exact_map.values())[0]
            best = processor.storage.get_entity_by_family_id(fid)

        # Step 2: Core-name match (strip parentheses)
        if not best:
            core = _re.sub(r'\s*[\(（].*?[\)）]\s*', '', name).strip()
            if core and core != name:
                core_map = processor.storage.get_family_ids_by_names([core])
                if core_map:
                    fid = list(core_map.values())[0]
                    best = processor.storage.get_entity_by_family_id(fid)

        # Step 3: Name prefix match
        if not best:
            # Try BM25 text search
            try:
                entities = processor.storage.search_entities_by_bm25(name, limit=1)
                if entities:
                    best = entities[0]
            except Exception:
                pass

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
        rel_dicts = [h.relation_to_dict(r) for r in rels]
        h.enrich_relations(rel_dicts, processor)
        return ok({
            "entity": h.entity_to_dict(best),
            "relations": rel_dicts,
            "relation_count": len(rel_dicts),
        })
    except Exception as e:
        return err(str(e), 500)


# ── Batch profiles ────────────────────────────────────────────────────────

@entities_bp.route("/api/v1/find/batch-profiles", methods=["POST"])
def batch_profiles():
    try:
        processor = _get_processor()
        h = _helpers()
        body = request.get_json(silent=True) or {}
        family_ids = body.get("family_ids", [])
        if not family_ids:
            return err("family_ids is required", 400)
        if len(family_ids) > 20:
            return err("Maximum 20 entities per batch", 400)

        batch_results = processor.storage.batch_get_entity_profiles(family_ids)
        profiles = []
        for item in batch_results:
            entity = item.get("entity")
            relations = item.get("relations", [])
            rel_dicts = [h.relation_to_dict(r) for r in relations]
            h.enrich_relations(rel_dicts, processor)
            profiles.append({
                "family_id": item["family_id"],
                "entity": h.entity_to_dict(entity) if entity else None,
                "relations": rel_dicts,
                "relation_count": len(rel_dicts),
                "version_count": item.get("version_count", 0),
            })
        return ok({"profiles": profiles, "count": len(profiles)})
    except Exception as e:
        return err(str(e), 500)


# ── Recent activity ───────────────────────────────────────────────────────

@entities_bp.route("/api/v1/find/recent-activity", methods=["GET"])
def recent_activity():
    try:
        processor = _get_processor()
        h = _helpers()
        limit = min(int(request.args.get("limit", "10")), 50)
        latest_entities = processor.storage.get_all_entities(limit=limit, exclude_embedding=True)
        latest_relations = processor.storage.get_all_relations(limit=limit, exclude_embedding=True)
        stats = processor.storage.get_graph_statistics()

        entity_dicts = [h.entity_to_dict(e) for e in reversed(latest_entities)]
        rel_dicts = [h.relation_to_dict(r) for r in reversed(latest_relations)]
        h.enrich_relations(rel_dicts, processor)

        return ok({
            "statistics": stats,
            "latest_entities": entity_dicts,
            "latest_relations": rel_dicts,
        })
    except Exception as e:
        return err(str(e), 500)
