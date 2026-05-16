"""
Entity CRUD operations — create, update, delete, batch ops, merge, isolated entities.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from flask import request

from core.models import Entity
from core.server.blueprints import helpers as _h
from core.server.blueprints.entities import entities_bp

ok, err = _h.ok, _h.err
safe_endpoint = _h.safe_endpoint
_get_processor = _h._get_processor

# Import validation helpers
_validate_text_input = _h._validate_text_input
get_json_body = _h.get_json_body

logger = logging.getLogger(__name__)


# -- Entity by absolute_id (update / delete) ---------------------------------

@entities_bp.route("/api/v1/find/entities/absolute/<absolute_id>", methods=["PUT"])
@safe_endpoint
def update_entity_absolute(absolute_id: str):
    try:
        processor = _get_processor()
        h = _h
        body = get_json_body()
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
@safe_endpoint
def delete_entity_absolute(absolute_id: str):
    try:
        processor = _get_processor()
        h = _h
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


# -- Create entity -----------------------------------------------------------

@entities_bp.route("/api/v1/find/entities/create", methods=["POST"])
@safe_endpoint
def create_entity():
    try:
        processor = _get_processor()
        h = _h
        body = get_json_body()
        name = str(body.get("name") or "").strip()
        content = str(body.get("content") or "").strip()
        if not name:
            return err("name 为必填", 400)

        # Validate inputs
        _validate_text_input(name, "name", min_len=1, max_len=500)
        if content:
            _validate_text_input(content, "content", min_len=0, max_len=100000)

        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")
        for _ in range(10):
            family_id = f"ent_{uuid.uuid4().hex[:12]}"
            absolute_id = f"entity_{ts}_{uuid.uuid4().hex[:8]}"
            if (not processor.storage.get_entity_by_absolute_id(absolute_id)
                    and not processor.storage.get_entity_by_family_id(family_id)):
                break

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
    except ValueError as ve:
        return err(str(ve), 400)
    except Exception as e:
        return err(str(e), 500)


# -- Batch delete ------------------------------------------------------------

@entities_bp.route("/api/v1/find/entities/batch-delete", methods=["POST"])
@safe_endpoint
def batch_delete_entities():
    try:
        processor = _get_processor()
        body = get_json_body()
        family_ids = body.get("family_ids") or body.get("entity_ids", [])
        if not isinstance(family_ids, list) or not family_ids:
            return err("family_ids 需为非空数组", 400)
        if len(family_ids) > 100:
            return err("单次批量删除上限 100 个", 400)
        total = processor.storage.batch_delete_entities(family_ids)
        return ok({"message": f"已删除 {total} 个实体版本", "count": total, "requested": len(family_ids)})
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/batch-delete-versions", methods=["POST"])
@safe_endpoint
def batch_delete_entity_versions():
    try:
        processor = _get_processor()
        h = _h
        body = get_json_body()
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


# -- Split version -----------------------------------------------------------

@entities_bp.route("/api/v1/find/entities/split-version", methods=["POST"])
@safe_endpoint
def split_entity_version():
    try:
        processor = _get_processor()
        h = _h
        body = get_json_body()
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


# -- Merge entities ----------------------------------------------------------

@entities_bp.route("/api/v1/find/entities/merge", methods=["POST"])
@safe_endpoint
def merge_entities():
    try:
        processor = _get_processor()
        body = get_json_body()
        target_id = (body.get("target_family_id") or "").strip()
        source_ids = body.get("source_family_ids", [])
        if not target_id or not isinstance(source_ids, list) or not source_ids:
            return err("target_family_id 和 source_family_ids 为必填", 400)
        if target_id in source_ids:
            return err("目标实体不能同时是源实体（不能合并自身）", 400)
        target = processor.storage.get_entity_by_family_id(target_id)
        if target is None:
            return err(f"目标实体不存在: {target_id}", 404)
        skip_name_check = body.get("skip_name_check", False)
        result = processor.storage.merge_entity_families(target_id, source_ids, skip_name_check=skip_name_check)
        response = {"message": "实体合并完成", "target_family_id": target_id, "source_family_ids": source_ids, "merged_count": result}
        if result.get("entities_updated", 0) == 0 and not skip_name_check:
            response["warning"] = "No entities were actually merged — name similarity check may have rejected all sources. Pass skip_name_check: true to override."
            response["hint"] = "Retrying with skip_name_check:true will bypass the name similarity check (useful for cross-language synonyms like 中文↔English)."
        return ok(response)
    except ValueError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(str(e), 500)


# -- Refresh edges -----------------------------------------------------------

@entities_bp.route("/api/v1/find/entities/refresh-edges", methods=["POST"])
@safe_endpoint
def refresh_edges():
    """Rebuild RELATES_TO edges from valid Relation nodes.

    Useful after entity alignment, merges, or dream cycles to ensure
    graph traversal stays consistent. Idempotent - safe to call repeatedly.
    """
    processor = _get_processor()
    try:
        if hasattr(processor.storage, 'refresh_relates_to_edges'):
            result = processor.storage.refresh_relates_to_edges()
            return ok({"message": "RELATES_TO edges refreshed", "result": result})
        else:
            return err("Storage backend does not support refresh_relates_to_edges", 501)
    except Exception as e:
        return err(str(e), 500)


# -- Isolated entities -------------------------------------------------------

@entities_bp.route("/api/v1/find/entities/isolated", methods=["GET"])
@safe_endpoint
def find_isolated_entities():
    try:
        processor = _get_processor()
        h = _h
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
@safe_endpoint
def delete_isolated_entities():
    try:
        processor = _get_processor()
        h = _h
        if not hasattr(processor.storage, 'get_isolated_entities'):
            return ok({"message": "当前存储后端不支持孤立实体检测", "deleted": 0})
        dry_run_body = get_json_body()
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


# -- Entity by family_id: update / delete (parameterized — MUST come last) ---

@entities_bp.route("/api/v1/find/entities/<family_id>", methods=["DELETE"])
@safe_endpoint
def delete_entity_family(family_id: str):
    try:
        processor = _get_processor()
        cascade = request.args.get("cascade", "false").lower() == "true"
        if cascade:
            # Delete related relations first, then the entity
            related = processor.storage.get_entity_relations_by_family_id(family_id)
            rel_fids = list({r.family_id for r in related})
            rel_count = processor.storage.batch_delete_relations(rel_fids) if rel_fids else 0
            count = processor.storage.delete_entity_all_versions(family_id)
            if count == 0:
                return err(f"未找到实体: {family_id}", 404)
            return ok({"message": f"已删除 {count} 个实体版本和 {rel_count} 个关系", "family_id": family_id, "cascade": cascade, "relations_deleted": rel_count})
        count = processor.storage.delete_entity_all_versions(family_id)
        if count == 0:
            return err(f"未找到实体: {family_id}", 404)
        return ok({"message": f"已删除 {count} 个实体版本", "family_id": family_id, "cascade": cascade})
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/<family_id>", methods=["PUT"])
@safe_endpoint
def update_entity_v2(family_id: str):
    try:
        processor = _get_processor()
        body = get_json_body()
        summary = body.get("summary")
        attributes = body.get("attributes")
        name = body.get("name")
        content = body.get("content")

        has_metadata_update = summary is not None or attributes is not None
        has_version_update = name is not None or content is not None

        if not has_metadata_update and not has_version_update:
            return err("name, content, summary 或 attributes 至少需要提供一个", 400)

        if name is not None and not str(name).strip():
            return err("name 不能为空字符串", 400)
        if content is not None and not str(content).strip():
            return err("content 不能为空字符串", 400)

        current = processor.storage.get_entity_by_family_id(family_id)
        if current is None:
            return err(f"未找到实体: {family_id}", 404)

        # Apply metadata updates in-place first
        if summary is not None:
            processor.storage.update_entity_summary(family_id, str(summary))
        if attributes is not None:
            attr_str = json.dumps(attributes, ensure_ascii=False) if isinstance(attributes, dict) else str(attributes)
            processor.storage.update_entity_attributes(family_id, attr_str)

        # Create new version if name/content changed
        if has_version_update:
            # Re-fetch to get updated summary/attributes
            refreshed = processor.storage.get_entity_by_family_id(family_id)
            now = datetime.now(timezone.utc)
            ts = now.strftime("%Y%m%d_%H%M%S")
            updated = Entity(
                absolute_id=f"entity_{ts}_{uuid.uuid4().hex[:8]}",
                family_id=family_id,
                name=name if name else refreshed.name,
                content=content if content else refreshed.content,
                event_time=now, processed_time=now,
                episode_id=refreshed.episode_id,
                source_document=refreshed.source_document,
                valid_at=now,
                summary=getattr(refreshed, 'summary', None),
                attributes=getattr(refreshed, 'attributes', None),
                confidence=getattr(refreshed, 'confidence', None),
                content_format=getattr(refreshed, 'content_format', 'plain'),
                community_id=getattr(refreshed, 'community_id', None),
            )
            processor.storage.save_entity(updated)
            # Refresh RELATES_TO edges: new version has a new absolute_id,
            # stale edges still point to the old one
            try:
                if hasattr(processor.storage, 'refresh_relates_to_edges'):
                    processor.storage.refresh_relates_to_edges(family_ids=[family_id])
            except Exception:
                pass  # non-critical; full refresh available via POST /refresh-edges
            vc_map = processor.storage.get_entity_version_counts([family_id]) if family_id else {}
            return ok(_h.entity_to_dict(updated, version_count=vc_map.get(family_id)))

        # Metadata-only path: return updated entity
        updated_entity = processor.storage.get_entity_by_family_id(family_id)
        vc_map = processor.storage.get_entity_version_counts([family_id]) if family_id else {}
        return ok(_h.entity_to_dict(updated_entity, version_count=vc_map.get(family_id)))
    except Exception as e:
        return err(str(e), 500)
