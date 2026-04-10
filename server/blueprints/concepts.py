"""
Concepts blueprint — Concept CRUD/search/traverse, communities, graphs management,
and chat session routes.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict

from flask import Blueprint, request

from server.blueprints.helpers import (
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
from server.sse import sse_response, queue_to_generator

logger = logging.getLogger(__name__)

concepts_bp = Blueprint("concepts", __name__)


# =========================================================
# Concepts — 统一概念查询接口（Phase 4）
# =========================================================

@concepts_bp.route("/api/v1/concepts/search", methods=["POST"])
def search_concepts():
    """统一概念搜索（可选 role 过滤）。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'search_concepts_by_bm25'):
            return err("此功能需要 SQLite >= Phase 3 或 Neo4j 后端", 400)
        body = request.get_json(silent=True) or {}
        query = (body.get("query") or "").strip()
        if not query:
            return err("query 不能为空", 400)
        role = body.get("role") or None
        limit = min(max(int(body.get("limit", 20)), 1), 100)
        results = storage.search_concepts_by_bm25(query, role=role, limit=limit)
        return ok({"concepts": results, "total": len(results)})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts", methods=["GET"])
def list_concepts():
    """列出概念（分页 + 可选 role 过滤）。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'list_concepts'):
            return err("此功能需要 SQLite >= Phase 3 或 Neo4j 后端", 400)
        role = request.args.get("role") or None
        limit = min(max(int(request.args.get('limit', 50)), 1), 100)
        offset = max(int(request.args.get('offset', 0)), 0)
        concepts = storage.list_concepts(role=role, limit=limit, offset=offset)
        total = storage.count_concepts(role=role) if hasattr(storage, 'count_concepts') else len(concepts)
        return ok({"concepts": concepts, "total": total, "limit": limit, "offset": offset})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/<family_id>", methods=["GET"])
def get_concept(family_id: str):
    """获取概念（任意 role）。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'get_concept_by_family_id'):
            return err("此功能需要 SQLite >= Phase 3 或 Neo4j 后端", 400)
        concept = storage.get_concept_by_family_id(family_id)
        if concept is None:
            return err("概念不存在", 404)
        return ok(concept)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/<family_id>/neighbors", methods=["GET"])
def get_concept_neighbors(family_id: str):
    """获取概念邻居（无论 role）。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'get_concept_neighbors'):
            return err("此功能需要 SQLite >= Phase 3 或 Neo4j 后端", 400)
        max_depth = min(max(int(request.args.get('max_depth', 1)), 1), 3)
        neighbors = storage.get_concept_neighbors(family_id, max_depth=max_depth)
        return ok({"family_id": family_id, "neighbors": neighbors})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/<family_id>/provenance", methods=["GET"])
def get_concept_provenance(family_id: str):
    """溯源：返回所有提及此概念的 observation。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'get_concept_provenance'):
            return err("此功能需要 SQLite >= Phase 3 或 Neo4j 后端", 400)
        provenance = storage.get_concept_provenance(family_id)
        return ok({"family_id": family_id, "provenance": provenance})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/traverse", methods=["POST"])
def traverse_concepts():
    """BFS 遍历概念图。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'traverse_concepts'):
            return err("此功能需要 SQLite >= Phase 3 或 Neo4j 后端", 400)
        body = request.get_json(silent=True) or {}
        start_ids = body.get("start_family_ids") or []
        if not start_ids:
            return err("start_family_ids 不能为空", 400)
        max_depth = min(max(int(body.get('max_depth', 2)), 1), 5)
        result = storage.traverse_concepts(start_ids, max_depth=max_depth)
        return ok(result)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/<family_id>/mentions", methods=["GET"])
def get_concept_mentions(family_id: str):
    """获取提及此概念的所有 Episode。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'get_concept_mentions'):
            return err("此功能需要 SQLite >= Phase 3 或 Neo4j 后端", 400)
        mentions = storage.get_concept_mentions(family_id)
        return ok({"family_id": family_id, "mentions": mentions})
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Communities
# =========================================================

@concepts_bp.route("/api/v1/communities/detect", methods=["POST"])
def detect_communities():
    """运行社区检测（Neo4j 专属）。"""
    try:
        processor = _get_processor()
        if not hasattr(processor.storage, 'detect_communities'):
            return err("此功能需要 Neo4j 后端", 400)
        body = request.get_json(silent=True) or {}
        algorithm = (body.get("algorithm") or "louvain").strip()
        resolution = float(body.get("resolution", 1.0))
        resolution = min(max(resolution, 0.1), 10.0)
        result = processor.storage.detect_communities(algorithm=algorithm, resolution=resolution)
        return ok(result)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/communities", methods=["GET"])
def list_communities():
    """列出社区（Neo4j 专属）。"""
    try:
        processor = _get_processor()
        if not hasattr(processor.storage, 'get_communities'):
            return err("此功能需要 Neo4j 后端", 400)
        min_size = max(int(request.args.get('min_size', 3)), 1)
        limit = min(max(int(request.args.get('limit', 50)), 1), 200)
        offset = max(int(request.args.get('offset', 0)), 0)
        communities, total = processor.storage.get_communities(limit=limit, min_size=min_size, offset=offset)
        return ok({"communities": communities, "count": len(communities), "total": total})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/communities/<int:cid>", methods=["GET"])
def get_community(cid: int):
    """获取社区详情（Neo4j 专属）。"""
    try:
        processor = _get_processor()
        if not hasattr(processor.storage, 'get_community'):
            return err("此功能需要 Neo4j 后端", 400)
        community = processor.storage.get_community(cid)
        if community is None:
            return err("社区不存在", 404)
        return ok(community)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/communities/<int:cid>/graph", methods=["GET"])
def get_community_graph(cid: int):
    """获取社区子图数据（Neo4j 专属）。"""
    try:
        processor = _get_processor()
        if not hasattr(processor.storage, 'get_community_graph'):
            return err("此功能需要 Neo4j 后端", 400)
        graph_data = processor.storage.get_community_graph(cid)
        return ok(graph_data)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/communities", methods=["DELETE"])
def clear_communities():
    """清除所有 community_id（Neo4j 专属）。"""
    try:
        processor = _get_processor()
        if not hasattr(processor.storage, 'clear_communities'):
            return err("此功能需要 Neo4j 后端", 400)
        cleared = processor.storage.clear_communities()
        return ok({"cleared": cleared})
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Graphs management
# =========================================================

@concepts_bp.route("/api/v1/graphs", methods=["GET", "POST"])
def handle_graphs():
    """GET: 列出所有图谱。POST: 创建新图谱。"""
    if request.method == "POST":
        try:
            from server.registry import GraphRegistry
            data = request.get_json(force=True) or {}
            graph_id = (data.get("graph_id") or "").strip()
            registry = request.app.config["registry"]
            GraphRegistry.validate_graph_id(graph_id)
            # 检查是否已存在
            existing = registry.list_graphs()
            if graph_id in existing:
                return err(f"图谱 '{graph_id}' 已存在", 409)
            # 触发懒创建：访问 processor 即会初始化 graph.db
            registry.get_processor(graph_id)
            return ok({"graph_id": graph_id, "message": "图谱创建成功"})
        except ValueError as e:
            return err(str(e), 400)
        except Exception as e:
            return err(str(e), 500)
    try:
        registry = request.app.config["registry"]
        graphs = registry.list_graphs()
        return ok({"graphs": graphs, "count": len(graphs)})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/graphs/<graph_id>", methods=["DELETE"])
def delete_graph(graph_id: str):
    """删除指定图谱（含所有数据）。"""
    try:
        from server.registry import GraphRegistry
        registry = request.app.config["registry"]
        GraphRegistry.validate_graph_id(graph_id)
        existing = registry.list_graphs()
        if graph_id not in existing:
            return err(f"图谱 '{graph_id}' 不存在", 404)
        registry.delete_graph(graph_id)
        return ok({"graph_id": graph_id, "message": "图谱已删除"})
    except ValueError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Chat Sessions — claude CLI 多会话管理
# =========================================================

_chat_mgr = None  # Lazy-init SessionManager
_chat_mgr_lock = threading.Lock()


def _get_chat_mgr():
    global _chat_mgr
    if _chat_mgr is not None:
        return _chat_mgr
    with _chat_mgr_lock:
        if _chat_mgr is not None:
            return _chat_mgr
        from server.chat_session import SessionManager
        _chat_mgr = SessionManager()
        _chat_mgr.start()
    return _chat_mgr


@concepts_bp.route("/api/v1/chat/sessions", methods=["GET"])
def chat_list_sessions():
    """List all chat sessions."""
    include_closed = request.args.get("include_closed", "0") == "1"
    try:
        mgr = _get_chat_mgr()
        return ok(mgr.list_sessions(include_closed=include_closed))
    except Exception as e:
        logger.error("GET /api/v1/chat/sessions failed: %s", e)
        return err(str(e), 500)


@concepts_bp.route("/api/v1/chat/sessions", methods=["POST"])
def chat_create_session():
    """Create a new chat session."""
    body = request.get_json(silent=True) or {}
    graph_id = body.get("graph_id", "default")
    title = body.get("title")
    try:
        mgr = _get_chat_mgr()
        result = mgr.create_session(graph_id=graph_id, title=title)
        return ok(result)
    except Exception as e:
        logger.error("POST /api/v1/chat/sessions failed: %s", e, exc_info=True)
        return err(str(e), 500)


@concepts_bp.route("/api/v1/chat/sessions/<sid>", methods=["GET"])
def chat_get_session(sid):
    """Get session details."""
    mgr = _get_chat_mgr()
    result = mgr.get_session(sid)
    if not result:
        return err("Session not found", 404)
    return ok(result)


@concepts_bp.route("/api/v1/chat/sessions/<sid>", methods=["PUT"])
def chat_update_session(sid):
    """Update session metadata (graph_id, title)."""
    body = request.get_json(silent=True) or {}
    mgr = _get_chat_mgr()
    if not mgr.update_session(sid, **body):
        return err("Session not found", 404)
    return ok(mgr.get_session(sid))


@concepts_bp.route("/api/v1/chat/sessions/<sid>", methods=["DELETE"])
def chat_delete_session(sid):
    """Delete a session completely."""
    mgr = _get_chat_mgr()
    if not mgr.delete_session(sid):
        return err("Session not found", 404)
    return ok({"deleted": True})


@concepts_bp.route("/api/v1/chat/sessions/<sid>/close", methods=["POST"])
def chat_close_session(sid):
    """Close a session (keep history, terminate process)."""
    mgr = _get_chat_mgr()
    if not mgr.close_session(sid):
        return err("Session not found", 404)
    return ok({"status": "closed"})


@concepts_bp.route("/api/v1/chat/sessions/<sid>/stream", methods=["POST"])
def chat_send_message(sid):
    """Send a message to a session. Returns SSE stream of events."""
    body = request.get_json(silent=True) or {}
    message = body.get("message", "")
    attachments = body.get("attachments")

    if not message:
        return err("message is required", 400)

    mgr = _get_chat_mgr()
    sentinel = mgr.get_event_sentinel()
    resp_queue = mgr.send_message(sid, message, attachments=attachments)

    if resp_queue is None:
        return err("Session not found or closed", 404)

    return sse_response(queue_to_generator(resp_queue, sentinel=sentinel))
