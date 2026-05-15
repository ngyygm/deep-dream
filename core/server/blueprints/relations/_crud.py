"""
Relation CRUD operations — create, update, delete, version lookup, batch ops.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import List

from flask import request

from core.models import Relation
from core.server.blueprints import helpers as _h
from core.server.blueprints.relations import relations_bp

ok, err = _h.ok, _h.err
_get_processor = _h._get_processor
_get_searcher = _h._get_searcher
relation_to_dict = _h.relation_to_dict
enrich_relations = _h.enrich_relations
parse_time_point = _h.parse_time_point

logger = logging.getLogger(__name__)

_VALID_RELATION_SCOPES = frozenset(("accumulated", "version_only", "all_versions"))


# -- Relation by absolute_id (GET) -------------------------------------------

@relations_bp.route("/api/v1/find/relations/absolute/<absolute_id>", methods=["GET"])
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


# -- Relation versions -------------------------------------------------------

@relations_bp.route("/api/v1/find/relations/<family_id>/versions", methods=["GET"])
def find_relation_versions(family_id: str):
    try:
        processor = _get_processor()
        versions = processor.storage.get_relation_versions(family_id)
        dicts = [relation_to_dict(r) for r in versions]
        enrich_relations(dicts, processor)
        return ok(dicts)
    except Exception as e:
        return err(str(e), 500)


# -- Update relation (by family_id — creates new version) --------------------

@relations_bp.route("/api/v1/find/relations/<family_id>", methods=["PUT"])
def update_relation_by_family(family_id: str):
    """Edit relation: create new version."""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        content = body.get("content")
        if not content:
            return err("content 为必填字段", 400)

        current_versions = processor.storage.get_relation_versions(family_id)
        if not current_versions:
            return err(f"未找到关系: {family_id}", 404)
        current = current_versions[0]  # latest version

        now = datetime.now(timezone.utc)
        updated = Relation(
            absolute_id=str(uuid.uuid4()),
            family_id=family_id,
            entity1_absolute_id=current.entity1_absolute_id,
            entity2_absolute_id=current.entity2_absolute_id,
            content=content,
            event_time=now,
            processed_time=now,
            episode_id=current.episode_id,
            source_document=current.source_document,
            valid_at=now,
        )
        processor.storage.save_relation(updated)
        return ok({"message": "关系已更新", "absolute_id": updated.absolute_id, "family_id": family_id})
    except Exception as e:
        return err(str(e), 500)


# -- Delete relation (all versions) ------------------------------------------

@relations_bp.route("/api/v1/find/relations/<family_id>", methods=["DELETE"])
def delete_relation_family(family_id: str):
    """Delete all versions of a relation."""
    try:
        processor = _get_processor()
        count = processor.storage.delete_relation_all_versions(family_id)
        if count == 0:
            return err(f"未找到关系: {family_id}", 404)
        return ok({"message": f"已删除 {count} 个关系版本", "family_id": family_id})
    except Exception as e:
        return err(str(e), 500)


# -- Batch delete ------------------------------------------------------------

@relations_bp.route("/api/v1/find/relations/batch-delete", methods=["POST"])
def batch_delete_relations():
    """Batch delete relations."""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        family_ids = body.get("family_ids") or body.get("relation_ids", [])
        if not isinstance(family_ids, list) or not family_ids:
            return err("family_ids 需为非空数组", 400)
        if len(family_ids) > 100:
            return err("单次批量删除上限 100 个", 400)
        total = processor.storage.batch_delete_relations(family_ids)
        return ok({"message": f"已删除 {total} 个关系版本", "count": len(family_ids)})
    except Exception as e:
        return err(str(e), 500)


# -- Relations by entity family_id (with scope support) ----------------------

@relations_bp.route("/api/v1/find/entities/<family_id>/relations", methods=["GET"])
def find_relations_by_entity(family_id: str):
    try:
        processor = _get_processor()
        limit = request.args.get("limit", type=int)
        time_point_str = request.args.get("time_point")
        try:
            time_point = parse_time_point(time_point_str)
        except ValueError as ve:
            return err(str(ve), 400)
        max_version_absolute_id = (request.args.get("max_version_absolute_id") or "").strip() or None
        relation_scope = (request.args.get("relation_scope") or "accumulated").strip()

        if relation_scope not in _VALID_RELATION_SCOPES:
            relation_scope = "accumulated"

        # When no max_version_absolute_id, all modes degenerate to returning all relations
        if not max_version_absolute_id:
            relations = processor.storage.get_entity_relations_by_family_id(
                family_id=family_id,
                limit=limit,
                time_point=time_point,
                max_version_absolute_id=None,
            )
            dicts = [relation_to_dict(r) for r in relations]
            enrich_relations(dicts, processor)
            return ok(dicts)

        # ---- Shared queries ----
        current_rels = processor.storage.get_entity_relations(
            max_version_absolute_id,
            limit=limit,
            time_point=time_point,
        )
        accum_rels = processor.storage.get_entity_relations_by_family_id(
            family_id=family_id,
            limit=limit,
            time_point=time_point,
            max_version_absolute_id=max_version_absolute_id,
        )

        # Dedup by family_id
        accum_by_rid = {r.family_id: r for r in accum_rels}
        current_by_rid = {r.family_id: r for r in current_rels}
        accum_rids = set(accum_by_rid)
        current_rids = set(current_by_rid)

        # ---- version_only: only relations directly linked to this version ----
        if relation_scope == "version_only":
            dicts = [relation_to_dict(r) for r in current_rels]
            enrich_relations(dicts, processor)
            return ok(dicts)

        # ---- shared latest_rels (used by both accumulated and all_versions) ----
        latest_rels = processor.storage.get_entity_relations_by_family_id(
            family_id=family_id,
            limit=limit,
            time_point=time_point,
            max_version_absolute_id=None,
        )
        latest_by_rid = {r.family_id: r for r in latest_rels}
        latest_rids = set(latest_by_rid)
        union_rids = accum_rids | latest_rids

        # ---- accumulated: v1..vN union + future from latest ----
        if relation_scope == "accumulated":
            all_rels = []
            for rid in union_rids:
                if rid in current_rids:
                    all_rels.append(current_by_rid[rid])
                elif rid in accum_rids:
                    all_rels.append(accum_by_rid[rid])
                else:
                    all_rels.append(latest_by_rid[rid])

            dicts = [relation_to_dict(r) for r in all_rels]
            enrich_relations(dicts, processor)

            for d in dicts:
                rid = d["family_id"]
                if rid not in current_rids:
                    if rid not in accum_rids:
                        d["_future"] = True
                    else:
                        d["_inherited"] = True

            return ok(dicts)

        # ---- all_versions: (v1..vN) union latest, classify as current/inherited/future ----
        all_rels = []
        for rid in union_rids:
            if rid in latest_rids:
                all_rels.append(latest_by_rid[rid])
            else:
                all_rels.append(accum_by_rid[rid])

        dicts = [relation_to_dict(r) for r in all_rels]
        enrich_relations(dicts, processor)

        for d in dicts:
            rid = d["family_id"]
            if rid in current_rids:
                d["_version_scope"] = "current"
            elif rid in accum_rids:
                d["_version_scope"] = "inherited"
            else:
                d["_version_scope"] = "future"

        return ok(dicts)
    except Exception as e:
        return err(str(e), 500)


# -- Create relation ---------------------------------------------------------

@relations_bp.route("/api/v1/find/relations/create", methods=["POST"])
def create_relation():
    """Manually create a relation (generates new family_id + absolute_id)."""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}

        # Resolve entity IDs: prefer absolute_id, fall back to family_id
        e1 = (body.get("entity1_absolute_id") or "").strip()
        e2 = (body.get("entity2_absolute_id") or "").strip()

        if not e1:
            e1_fid = (body.get("entity1_family_id") or "").strip()
            if e1_fid:
                entity1 = processor.storage.get_entity_by_family_id(e1_fid)
                if entity1:
                    e1 = entity1.absolute_id
                else:
                    return err(f"entity1_family_id '{e1_fid}' 未找到对应实体", 404)

        if not e2:
            e2_fid = (body.get("entity2_family_id") or "").strip()
            if e2_fid:
                entity2 = processor.storage.get_entity_by_family_id(e2_fid)
                if entity2:
                    e2 = entity2.absolute_id
                else:
                    return err(f"entity2_family_id '{e2_fid}' 未找到对应实体", 404)

        if not e1 or not e2:
            return err("需要 entity1_absolute_id 或 entity1_family_id（entity2 同理）", 400)

        content = (body.get("content") or "").strip()
        if not content:
            return err("content 为必填", 400)

        now = datetime.now()
        ts = now.strftime("%Y%m%d_%H%M%S")
        # Ensure entity1 < entity2 (undirected relation)
        if e1 > e2:
            e1, e2 = e2, e1
        family_id = f"rel_{uuid.uuid4().hex[:12]}"
        absolute_id = f"relation_{ts}_{uuid.uuid4().hex[:8]}"
        # Single loop: check both IDs together (collision probability ~0)
        for _ in range(10):
            family_id = f"rel_{uuid.uuid4().hex[:12]}"
            absolute_id = f"relation_{ts}_{uuid.uuid4().hex[:8]}"
            if (not processor.storage.get_relation_by_absolute_id(absolute_id)
                    and not processor.storage.get_relation_by_family_id(family_id)):
                break

        # Resolve family_ids for entity1 and entity2
        _e1_ent = processor.storage.get_entity_by_absolute_id(e1) if e1 else None
        _e2_ent = processor.storage.get_entity_by_absolute_id(e2) if e2 else None
        e1_fid_resolved = _e1_ent.family_id if _e1_ent else ''
        e2_fid_resolved = _e2_ent.family_id if _e2_ent else ''

        relation = Relation(
            absolute_id=absolute_id,
            family_id=family_id,
            entity1_absolute_id=e1,
            entity2_absolute_id=e2,
            entity1_family_id=e1_fid_resolved,
            entity2_family_id=e2_fid_resolved,
            content=content,
            event_time=now,
            processed_time=now,
            episode_id=body.get("episode_id", ""),
            source_document=body.get("source_document", ""),
        )
        processor.storage.save_relation(relation)
        return ok(relation_to_dict(relation))
    except Exception as e:
        return err(str(e), 500)


# -- Update relation by absolute_id ------------------------------------------

@relations_bp.route("/api/v1/find/relations/absolute/<absolute_id>", methods=["PUT"])
def update_relation_absolute(absolute_id: str):
    """Update a specific relation version."""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        fields = {}
        for key in ("content", "summary", "attributes", "confidence"):
            if key in body:
                fields[key] = body[key]
        if not fields:
            return err("至少提供一个可更新字段", 400)
        updated = processor.storage.update_relation_by_absolute_id(absolute_id, **fields)
        if not updated:
            return err(f"未找到关系版本: {absolute_id}", 404)
        return ok(relation_to_dict(updated))
    except Exception as e:
        return err(str(e), 500)


# -- Delete relation by absolute_id ------------------------------------------

@relations_bp.route("/api/v1/find/relations/absolute/<absolute_id>", methods=["DELETE"])
def delete_relation_absolute(absolute_id: str):
    """Delete a specific relation version."""
    try:
        processor = _get_processor()
        success = processor.storage.delete_relation_by_absolute_id(absolute_id)
        if not success:
            return err(f"未找到关系版本: {absolute_id}", 404)
        return ok({"absolute_id": absolute_id, "deleted": True})
    except Exception as e:
        return err(str(e), 500)


# -- Batch delete relation versions ------------------------------------------

@relations_bp.route("/api/v1/find/relations/batch-delete-versions", methods=["POST"])
def batch_delete_relation_versions():
    """Batch delete relation versions."""
    try:
        processor = _get_processor()
        body = request.get_json(silent=True) or {}
        absolute_ids = body.get("absolute_ids", [])
        if not isinstance(absolute_ids, list) or not absolute_ids:
            return err("absolute_ids 需为非空数组", 400)
        deleted_count = processor.storage.batch_delete_relation_versions_by_absolute_ids(absolute_ids)
        return ok({
            "deleted": absolute_ids[:deleted_count],
            "failed": absolute_ids[deleted_count:] if deleted_count < len(absolute_ids) else [],
            "summary": {"deleted_count": deleted_count, "failed_count": len(absolute_ids) - deleted_count},
        })
    except Exception as e:
        return err(str(e), 500)
