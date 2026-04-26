"""
Episodes blueprint — Episode CRUD, search, batch ingest, snapshots, changes,
Neo4j episode list, and document management routes.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from flask import Blueprint, request

from core.server.blueprints.helpers import (
    ok,
    err,
    _get_processor,
    _get_graph_id,
    entity_to_dict,
    relation_to_dict,
    enrich_relations,
    episode_to_dict,
    parse_time_point,
)
from core.models import Episode

logger = logging.getLogger(__name__)

episodes_bp = Blueprint("episodes", __name__)


# =========================================================
# Find: Episode 原子接口
# =========================================================

@episodes_bp.route("/api/v1/find/episodes/latest/metadata", methods=["GET"])
def find_latest_episode_metadata():
    try:
        processor = _get_processor()
        activity_type = request.args.get("activity_type")
        metadata = processor.storage.get_latest_episode_metadata(activity_type=activity_type)
        if metadata is None:
            return err("未找到 Episode 元数据", 404)
        return ok(metadata)
    except Exception as e:
        return err(str(e), 500)


@episodes_bp.route("/api/v1/find/episodes/latest", methods=["GET"])
def find_latest_episode():
    try:
        processor = _get_processor()
        activity_type = request.args.get("activity_type")
        cache = processor.storage.get_latest_episode(activity_type=activity_type)
        if cache is None:
            return err("未找到 Episode", 404)
        return ok(episode_to_dict(cache))
    except Exception as e:
        return err(str(e), 500)


@episodes_bp.route("/api/v1/find/episodes/<cache_id>/text", methods=["GET"])
def find_episode_text(cache_id: str):
    try:
        processor = _get_processor()
        text = processor.storage.get_episode_text(cache_id)
        if text is None:
            return err(f"未找到 Episode 或原文: {cache_id}", 404)
        return ok({"cache_id": cache_id, "text": text})
    except Exception as e:
        return err(str(e), 500)


@episodes_bp.route("/api/v1/find/episodes/<cache_id>", methods=["GET"])
def find_episode(cache_id: str):
    try:
        processor = _get_processor()
        cache = processor.storage.load_episode(cache_id)
        if cache is None:
            return err(f"未找到 Episode: {cache_id}", 404)
        return ok(episode_to_dict(cache))
    except Exception as e:
        return err(str(e), 500)


@episodes_bp.route("/api/v1/find/episodes/<cache_id>/doc", methods=["GET"])
def find_episode_doc(cache_id: str):
    """获取 Episode 对应的完整文档内容（原文 + 缓存摘要）。"""
    try:
        processor = _get_processor()
        doc_hash = processor.storage.get_doc_hash_by_cache_id(cache_id)
        if not doc_hash:
            return err(f"未找到文档: {cache_id}", 404)
        content = processor.storage.get_doc_content(doc_hash)
        if content is None:
            return err("文档内容不存在", 404)
        # 不返回 meta 中的大文本字段
        meta = content.get("meta") or {}
        meta_for_response = {k: v for k, v in meta.items() if k not in ("text", "document_path")}
        return ok({
            "meta": meta_for_response,
            "original": content.get("original"),
            "cache": content.get("cache"),
        })
    except Exception as e:
        return err(str(e), 500)


@episodes_bp.route("/api/v1/find/episodes/search", methods=["POST"])
def search_episodes():
    """搜索 Episode。"""
    try:
        body = request.get_json(silent=True) or {}
        query = (body.get("query") or "").strip()
        if not query:
            return err("query 为必填", 400)
        limit = int(body.get("limit", 20))
        processor = _get_processor()
        results = processor.storage.search_episodes_by_bm25(query, limit=limit)
        return ok([episode_to_dict(c) for c in results])
    except Exception as e:
        return err(str(e), 500)


@episodes_bp.route("/api/v1/find/episodes/<cache_id>", methods=["DELETE"])
def find_episode_delete(cache_id: str):
    """删除 Episode。"""
    try:
        processor = _get_processor()
        processor.storage.delete_episode_mentions(cache_id)
        count = processor.storage.delete_episode(cache_id)
        if count == 0:
            return err(f"未找到 Episode: {cache_id}", 404)
        return ok({"message": "已删除", "cache_id": cache_id})
    except Exception as e:
        return err(str(e), 500)


@episodes_bp.route("/api/v1/find/episodes/batch-ingest", methods=["POST"])
def batch_ingest_episodes():
    """批量导入 Episode。使用 bulk_save 优化写入性能。"""
    try:
        body = request.get_json(silent=True) or {}
        episodes = body.get("episodes", [])
        if not isinstance(episodes, list):
            return err("episodes 需为数组", 400)
        processor = _get_processor()
        episode_objects = []
        for ep in episodes:
            text = (ep.get("content") or ep.get("text") or "").strip()
            if not text:
                continue
            source = ep.get("source_document", "batch_ingest")
            ep_type = ep.get("episode_type")
            mc = Episode(
                absolute_id=str(uuid.uuid4()),
                content=text,
                event_time=datetime.now(timezone.utc),
                source_document=source,
                episode_type=ep_type,
            )
            episode_objects.append(mc)
        if episode_objects:
            # Use bulk save for Neo4j, fallback to loop for SQLite
            if hasattr(processor.storage, 'bulk_save_episodes'):
                processor.storage.bulk_save_episodes(episode_objects)
            else:
                for mc in episode_objects:
                    processor.storage.save_episode(mc)
        return ok({"message": f"已导入 {len(episode_objects)} 条 Episode", "count": len(episode_objects)})
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Time Travel / Snapshot / Changes
# =========================================================

@episodes_bp.route("/api/v1/find/snapshot", methods=["GET"])
def find_snapshot():
    """获取指定时间点的图谱快照"""
    try:
        time_str = request.args.get("time")
        limit = request.args.get("limit", type=int)

        processor = _get_processor()
        if time_str:
            time_point = parse_time_point(time_str)
            snapshot = processor.storage.get_snapshot(time_point, limit=limit)
            time_label = time_point.isoformat()
        else:
            # 无 time 参数时返回最新快照
            now = datetime.now(timezone.utc)
            snapshot = processor.storage.get_snapshot(now, limit=limit)
            time_label = now.isoformat()

        return ok({
            "time": time_label,
            "entities": [entity_to_dict(e) for e in snapshot["entities"]],
            "relations": [relation_to_dict(r) for r in snapshot["relations"]],
            "entity_count": len(snapshot["entities"]),
            "relation_count": len(snapshot["relations"]),
        })
    except Exception as e:
        return err(str(e), 500)


@episodes_bp.route("/api/v1/find/changes", methods=["GET"])
def find_changes():
    """获取时间范围内的变更记录"""
    try:
        since_str = request.args.get("since")
        until_str = request.args.get("until")
        if not since_str:
            return err("since 为必填参数（ISO 格式）", 400)
        since = parse_time_point(since_str)
        until = parse_time_point(until_str) if until_str else None
        limit = request.args.get("limit", type=int)

        processor = _get_processor()
        changes = processor.storage.get_changes(since, until=until)
        return ok({
            "since": since.isoformat(),
            "until": (until or datetime.now(timezone.utc)).isoformat(),
            "entities": [entity_to_dict(e) for e in changes["entities"]],
            "relations": [relation_to_dict(r) for r in changes["relations"]],
            "entity_count": len(changes["entities"]),
            "relation_count": len(changes["relations"]),
        })
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Neo4j Episodes
# =========================================================

@episodes_bp.route("/api/v1/episodes", methods=["GET"])
def list_episodes():
    """分页列出 Episodes。"""
    try:
        processor = _get_processor()
        if not hasattr(processor.storage, 'list_episodes'):
            return err("此功能需要 Neo4j 或 SQLite 后端", 400)
        limit = min(max(int(request.args.get('limit', 20)), 1), 100)
        offset = max(int(request.args.get('offset', 0)), 0)
        include_text = request.args.get('include_text', '0') in ('1', 'true')
        episodes = processor.storage.list_episodes(limit=limit, offset=offset, include_text=include_text)
        total = processor.storage.count_episodes()
        return ok({"episodes": episodes, "total": total, "limit": limit, "offset": offset})
    except Exception as e:
        return err(str(e), 500)


@episodes_bp.route("/api/v1/episodes/<uuid>", methods=["GET"])
def get_episode(uuid: str):
    """获取 Episode 详情。"""
    try:
        processor = _get_processor()
        if not hasattr(processor.storage, 'get_episode'):
            return err("此功能需要 Neo4j 或 SQLite 后端", 400)
        episode = processor.storage.get_episode(uuid)
        if episode is None:
            return err("Episode 不存在", 404)
        return ok(episode)
    except Exception as e:
        return err(str(e), 500)


@episodes_bp.route("/api/v1/episodes/<uuid>/entities", methods=["GET"])
def get_episode_entities(uuid: str):
    """获取 Episode 关联实体和关系。"""
    try:
        processor = _get_processor()
        if not hasattr(processor.storage, 'get_episode_entities'):
            return err("此功能需要 Neo4j 或 SQLite 后端", 400)
        entities = processor.storage.get_episode_entities(uuid)
        return ok({"entities": entities})
    except Exception as e:
        return err(str(e), 500)


@episodes_bp.route("/api/v1/episodes/search", methods=["POST"])
def neo4j_search_episodes():
    """搜索 Episodes（Neo4j 专属）。"""
    try:
        processor = _get_processor()
        if not hasattr(processor.storage, 'search_episodes'):
            return err("此功能需要 Neo4j 后端", 400)
        body = request.get_json(silent=True) or {}
        query = (body.get("query") or "").strip()
        if not query:
            return err("query 不能为空", 400)
        limit = min(max(int(body.get("limit", 20)), 1), 100)
        episodes = processor.storage.search_episodes(query, limit=limit)
        return ok({"episodes": episodes})
    except Exception as e:
        return err(str(e), 500)


@episodes_bp.route("/api/v1/episodes/<uuid>", methods=["DELETE"])
def neo4j_delete_episode(uuid: str):
    """删除 Episode（Neo4j 专属）。"""
    try:
        processor = _get_processor()
        if not hasattr(processor.storage, 'delete_episode'):
            return err("此功能需要 Neo4j 后端", 400)
        success = processor.storage.delete_episode(uuid)
        if not success:
            return err("Episode 不存在或删除失败", 404)
        return ok({"deleted": True})
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# 文档管理
# =========================================================

@episodes_bp.route("/api/v1/docs", methods=["GET"])
def list_docs():
    """列出当前图谱的所有文档。"""
    try:
        processor = _get_processor()
        docs = processor.storage.list_docs()
        # 返回精简字段
        result = []
        for d in docs:
            result.append({
                "doc_hash": d.get("doc_hash", ""),
                "source_document": d.get("source_document") or d.get("doc_name", ""),
                "doc_name": d.get("source_document") or d.get("doc_name", ""),
                "source_name": d.get("source_name", ""),
                "event_time": d.get("event_time"),
                "processed_time": d.get("processed_time"),
                "activity_type": d.get("activity_type", ""),
                "text_length": d.get("text_length", 0),
                "filename": d.get("filename", ""),
            })
        return ok({"docs": result, "count": len(result)})
    except Exception as e:
        return err(str(e), 500)


@episodes_bp.route("/api/v1/docs/<path:filename>", methods=["GET"])
def get_doc_content(filename):
    """获取文档内容（原始文本和缓存摘要）。"""
    try:
        processor = _get_processor()
        content = processor.storage.get_doc_content(filename)
        if content is None:
            return err("Document not found", 404)
        # 不返回 meta 中的大文本字段
        meta = content.get("meta") or {}
        meta_for_response = {k: v for k, v in meta.items() if k not in ("text", "document_path")}
        return ok({
            "meta": meta_for_response,
            "original": content.get("original"),
            "cache": content.get("cache"),
        })
    except Exception as e:
        return err(str(e), 500)
