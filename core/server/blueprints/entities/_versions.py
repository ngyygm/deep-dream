"""
Entity version queries, timeline, diff, patches, confidence, contradictions,
neighbors, and profile routes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from flask import request

from core.perf import _perf_timer
from core.server.blueprints import helpers as _h
from core.server.blueprints.entities import entities_bp

ok, err, run_async = _h.ok, _h.err, _h.run_async
safe_endpoint = _h.safe_endpoint
_get_processor = _h._get_processor
get_json_body = _h.get_json_body

logger = logging.getLogger(__name__)


# -- Entity by absolute_id (GET) ---------------------------------------------

@entities_bp.route("/api/v1/find/entities/absolute/<absolute_id>", methods=["GET"])
@safe_endpoint
def find_entity_by_absolute_id(absolute_id: str):
    try:
        processor = _get_processor()
        h = _h
        entity = processor.storage.get_entity_by_absolute_id(absolute_id)
        if entity is None:
            return err(f"未找到实体版本: {absolute_id}", 404)
        return ok(h.entity_to_dict(entity))
    except Exception as e:
        return err(str(e), 500)


# -- Entity by family_id (version list, GET) ---------------------------------

@entities_bp.route("/api/v1/find/entities/<family_id>/versions", methods=["GET"])
@safe_endpoint
def find_entity_versions(family_id: str):
    try:
        processor = _get_processor()
        h = _h
        versions = processor.storage.get_entity_versions(family_id)
        return ok([h.entity_to_dict(e) for e in versions])
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/<family_id>/as-of-time", methods=["GET"])
@safe_endpoint
def find_entity_at_time(family_id: str):
    try:
        processor = _get_processor()
        h = _h
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
@safe_endpoint
def find_entity_nearest_to_time(family_id: str):
    try:
        processor = _get_processor()
        h = _h
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
@safe_endpoint
def find_entity_around_time(family_id: str):
    try:
        processor = _get_processor()
        h = _h
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
@safe_endpoint
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
@safe_endpoint
def find_entity_by_family_id(family_id: str):
    try:
        processor = _get_processor()
        h = _h
        entity = processor.storage.get_entity_by_family_id(family_id)
        if entity is None:
            return err(f"未找到实体: {family_id}", 404)
        return ok(h.entity_to_dict(entity))
    except Exception as e:
        return err(str(e), 500)


# -- Timeline & patches ------------------------------------------------------

@entities_bp.route("/api/v1/find/entities/<family_id>/timeline", methods=["GET"])
@safe_endpoint
def find_entity_timeline(family_id: str):
    try:
        processor = _get_processor()
        h = _h
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
@safe_endpoint
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
@safe_endpoint
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
@safe_endpoint
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


# -- Intelligence: evolve summary, provenance, confidence, contradictions ----

@entities_bp.route("/api/v1/find/entities/<family_id>/evolve-summary", methods=["POST"])
@safe_endpoint
def evolve_entity_summary(family_id: str):
    try:
        processor = _get_processor()
        entity = processor.storage.get_entity_by_family_id(family_id)
        if entity is None:
            return err(f"未找到实体: {family_id}", 404)

        versions = processor.storage.get_entity_versions(family_id)
        old_version = versions[1] if len(versions) > 1 else None

        summary = run_async(
            processor.llm_client.evolve_entity_summary(entity, old_version)
        )

        processor.storage.update_entity_summary(family_id, summary)
        return ok({"family_id": family_id, "summary": summary})
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/<family_id>/provenance", methods=["GET"])
@safe_endpoint
def get_entity_provenance(family_id: str):
    try:
        processor = _get_processor()
        provenance = processor.storage.get_entity_provenance(family_id)
        return ok(provenance)
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/<family_id>/confidence", methods=["PUT"])
@safe_endpoint
def update_entity_confidence(family_id: str):
    """手动设置实体置信度（覆盖自动演化值）。"""
    try:
        body = get_json_body()
        confidence = body.get("confidence")
        if confidence is None:
            return err("confidence 为必填字段", 400)
        confidence = float(confidence)
        if not (0.0 <= confidence <= 1.0):
            return err("confidence 必须在 0.0 ~ 1.0 之间", 400)
        processor = _get_processor()
        entity = processor.storage.get_entity_by_family_id(family_id)
        if not entity:
            return err(f"实体不存在: {family_id}", 404)
        processor.storage.update_entity_confidence(family_id, confidence)
        # Patch in-memory instead of re-reading from DB
        entity.confidence = confidence
        h = _h
        return ok(h.entity_to_dict(entity))
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/<family_id>/contradictions", methods=["GET"])
@safe_endpoint
def get_entity_contradictions(family_id: str):
    try:
        processor = _get_processor()
        versions = processor.storage.get_entity_versions(family_id)
        if len(versions) < 2:
            return ok([])

        contradictions = run_async(
            processor.llm_client.detect_contradictions(family_id, versions)
        )

        return ok(contradictions)
    except Exception as e:
        return err(str(e), 500)


@entities_bp.route("/api/v1/find/entities/<family_id>/resolve-contradiction", methods=["POST"])
@safe_endpoint
def resolve_entity_contradiction(family_id: str):
    try:
        body = get_json_body()
        contradiction = body.get("contradiction")
        if not contradiction or not isinstance(contradiction, dict):
            return err("contradiction 为必填字段", 400)

        processor = _get_processor()
        resolution = run_async(
            processor.llm_client.resolve_contradiction(contradiction)
        )

        return ok(resolution)
    except Exception as e:
        return err(str(e), 500)


# -- Neighbors (Neo4j) -------------------------------------------------------

@entities_bp.route("/api/v1/find/entities/<entity_uuid>/neighbors", methods=["GET"])
@safe_endpoint
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


# -- Profile -----------------------------------------------------------------

@entities_bp.route("/api/v1/find/entities/<family_id>/profile", methods=["GET"])
@safe_endpoint
def entity_profile(family_id: str):
    try:
        processor = _get_processor()
        h = _h
        # Use batch_get_entity_profiles if available (single Cypher query)
        if hasattr(processor.storage, 'batch_get_entity_profiles'):
            profiles = processor.storage.batch_get_entity_profiles([family_id])
            if not profiles or not profiles[0].get("entity"):
                return err(f"未找到实体: {family_id}", 404)
            p = profiles[0]
            entity = p["entity"]
            relations = p.get("relations", [])
            version_count = p.get("version_count", 0)
        else:
            entity = processor.storage.get_entity_by_family_id(family_id)
            if entity is None:
                return err(f"未找到实体: {family_id}", 404)
            relations = processor.storage.get_entity_relations_by_family_id(family_id)
            version_count = processor.storage.get_entity_version_count(family_id)
        rels = [h.relation_to_dict(r) for r in relations]
        h.enrich_relations(rels, processor)
        return ok({
            "entity": h.entity_to_dict(entity, version_count=version_count),
            "relations": rels,
            "relation_count": len(rels),
            "version_count": version_count,
        })
    except Exception as e:
        return err(str(e), 500)
