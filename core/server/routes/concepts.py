"""
Concept routes — Concept CRUD/search/traverse, communities, graphs management,
and chat session routes.
"""
from __future__ import annotations

import logging
import re as _re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict

from flask import Blueprint, current_app, request

from core.server.routes.helpers import (
    ok,
    err,
    _get_processor,
    _get_graph_id,
    entity_to_dict,
    relation_to_dict,
    enrich_relations,
    episode_to_dict,
    parse_time_point,
    _get_searcher,
    get_json_body,
)
from core.server.routes._constants import _VALID_SEARCH_MODES
from core.server.sse import sse_response, queue_to_generator
from core.server.registry import GraphRegistry

logger = logging.getLogger(__name__)

concepts_bp = Blueprint("concepts", __name__)

_shared_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="concept")

# Pre-compiled regex for duplicate entity name normalization
_BOOK_MARKS_RE = _re.compile(r'[《》]')
_PAREN_ANNOTATION_RE = _re.compile(r'\s*[（(][^）)]+[）)]\s*')


# =========================================================
# Concepts — 统一概念查询接口（Phase 4）
# =========================================================

@concepts_bp.route("/api/v1/concepts/search", methods=["POST"])
@concepts_bp.route("/api/v1/find", methods=["POST"])
def search_concepts():
    """统一概念搜索（可选 role 过滤，支持 semantic/bm25/hybrid 模式）。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'search_concepts_by_bm25'):
            return err("此功能暂不可用", 400)
        body = get_json_body()
        query = (body.get("query") or "").strip()
        if not query:
            return err("query 不能为空", 400)
        role = body.get("role") or None
        limit = min(max(int(body.get("limit", 20)), 1), 100)
        threshold = float(body.get("threshold", 0.5))
        search_mode = str(body.get("search_mode", "bm25") or "bm25").strip().lower()
        if search_mode not in _VALID_SEARCH_MODES:
            return err(f"search_mode '{search_mode}' 无效，可选: {', '.join(_VALID_SEARCH_MODES)}", 400)
        time_point = (body.get("time_point") or "").strip() or None

        def _search(role_filter, result_limit):
            if search_mode == "bm25":
                return storage.search_concepts_by_bm25(query, role=role_filter, limit=result_limit, time_point=time_point)
            if search_mode == "semantic":
                return storage.search_concepts_by_similarity(
                    query_text=query, role=role_filter, threshold=threshold, max_results=result_limit, time_point=time_point
                )
            return _hybrid_concept_search(storage, query, role_filter, result_limit, threshold, time_point=time_point)

        if request.path == "/api/v1/find":
            max_entities = min(max(int(body.get("max_entities", body.get("maxEntities", 20))), 1), 100)
            max_relations = min(max(int(body.get("max_relations", body.get("maxRelations", 50))), 1), 100)
            entities = _search("entity", max_entities)
            relations = _search("relation", max_relations)
            return ok({
                "entities": entities,
                "relations": relations,
                "concepts": entities + relations,
                "total": len(entities) + len(relations),
            })

        results = _search(role, limit)
        return ok({"concepts": results, "total": len(results)})
    except Exception as e:
        return err(str(e), 500)


def _hybrid_concept_search(storage, query: str, role, limit: int,
                           threshold: float, time_point: str = None) -> list:
    """Hybrid concept search: BM25 + semantic embedding, fused via RRF."""

    def _bm25():
        try:
            return storage.search_concepts_by_bm25(query, role=role, limit=limit * 2, time_point=time_point)
        except Exception:
            return []

    def _semantic():
        try:
            return storage.search_concepts_by_similarity(
                query_text=query, role=role, threshold=threshold, max_results=limit * 2, time_point=time_point
            )
        except Exception:
            return []

    bm25_fut = _shared_pool.submit(_bm25)
    sem_fut = _shared_pool.submit(_semantic)
    bm25_results = bm25_fut.result()
    semantic_results = sem_fut.result()

    if not bm25_results and not semantic_results:
        return []

    # RRF fusion on dict results (keyed by family_id)
    k = 60
    scores: Dict[str, float] = {}
    items: Dict[str, dict] = {}
    best_contrib: Dict[str, float] = {}
    bm25_weight = 0.3
    sem_weight = 0.7

    for rank, item in enumerate(bm25_results):
        fid = item.get("family_id", "") or item.get("id", "")
        rrf = bm25_weight / (k + rank + 1)
        scores[fid] = scores.get(fid, 0.0) + rrf
        if fid not in items or rrf > best_contrib.get(fid, 0.0):
            items[fid] = item
            best_contrib[fid] = rrf

    for rank, item in enumerate(semantic_results):
        fid = item.get("family_id", "") or item.get("id", "")
        rrf = sem_weight / (k + rank + 1)
        scores[fid] = scores.get(fid, 0.0) + rrf
        if fid not in items or rrf > best_contrib.get(fid, 0.0):
            items[fid] = item
            best_contrib[fid] = rrf

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused = [items[fid] for fid, _ in sorted_items[:limit]]
    return fused


@concepts_bp.route("/api/v1/concepts", methods=["GET"])
def list_concepts():
    """列出概念（分页 + 可选 role 过滤）。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'list_concepts'):
            return err("此功能暂不可用", 400)
        role = request.args.get("role") or None
        limit = min(max(int(request.args.get('limit', 50)), 1), 100)
        offset = max(int(request.args.get('offset', 0)), 0)
        time_point = (request.args.get("time_point") or "").strip() or None
        concepts = storage.list_concepts(role=role, limit=limit, offset=offset, time_point=time_point)
        total = storage.count_concepts(role=role, time_point=time_point) if hasattr(storage, 'count_concepts') else len(concepts)
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
            return err("此功能暂不可用", 400)
        time_point = (request.args.get("time_point") or "").strip() or None
        concept = storage.get_concept_by_family_id(family_id, time_point=time_point)
        if concept is None:
            return err("概念不存在", 404)
        return ok(concept)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/<family_id>/versions", methods=["GET"])
def get_concept_versions(family_id: str):
    """List all versions for a concept family."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "get_concept_versions"):
            return err("此功能暂不可用", 400)
        versions = storage.get_concept_versions(family_id)
        if not versions:
            return err("概念不存在", 404)
        return ok({"family_id": family_id, "versions": versions, "total": len(versions)})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/<family_id>/neighbors", methods=["GET"])
def get_concept_neighbors(family_id: str):
    """获取概念邻居（无论 role）。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'get_concept_neighbors'):
            return err("此功能暂不可用", 400)
        max_depth = min(max(int(request.args.get('max_depth', 1)), 1), 3)
        time_point = (request.args.get("time_point") or "").strip() or None
        neighbors = storage.get_concept_neighbors(family_id, max_depth=max_depth, time_point=time_point)
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
            return err("此功能暂不可用", 400)
        time_point = (request.args.get("time_point") or "").strip() or None
        if hasattr(storage, "get_concept_by_family_id") and storage.get_concept_by_family_id(family_id, time_point=time_point) is None:
            return err("概念不存在", 404)
        provenance = storage.get_concept_provenance(family_id, time_point=time_point)
        return ok({"family_id": family_id, "provenance": provenance})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/concepts/traverse", methods=["POST"])
@concepts_bp.route("/api/v1/traverse", methods=["POST"])
def traverse_concepts():
    """BFS 遍历概念图。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, 'traverse_concepts'):
            return err("此功能暂不可用", 400)
        body = get_json_body()
        start_ids = body.get("start_family_ids") or []
        if not start_ids:
            return err("start_family_ids 不能为空", 400)
        max_depth = min(max(int(body.get('max_depth', 2)), 1), 5)
        time_point = (body.get("time_point") or "").strip() or None
        edge_types = body.get("edge_types") or body.get("edge_type") or None
        if isinstance(edge_types, str):
            edge_types = [edge_types]
        result = storage.traverse_concepts(start_ids, max_depth=max_depth, time_point=time_point, edge_types=edge_types)
        return ok(result)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/documents", methods=["GET"])
def list_documents():
    """List indexed Markdown documents for the current graph."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "list_documents"):
            return err("此功能暂不可用", 400)
        limit = min(max(int(request.args.get("limit", 50)), 1), 200)
        offset = max(int(request.args.get("offset", 0)), 0)
        documents = storage.list_documents(limit=limit, offset=offset)
        return ok({"documents": documents, "total": len(documents), "limit": limit, "offset": offset})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/documents/graph", methods=["POST"])
def get_documents_graph():
    """Return a Document -> Episode -> Concept subgraph for selected documents."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "get_document_graph"):
            return err("此功能暂不可用", 400)
        body = get_json_body()
        document_version_ids = body.get("document_version_ids") or []
        document_family_ids = body.get("document_family_ids") or []
        if isinstance(document_version_ids, str):
            document_version_ids = [document_version_ids]
        if isinstance(document_family_ids, str):
            document_family_ids = [document_family_ids]
        if not document_version_ids and not document_family_ids:
            return err("document_version_ids 或 document_family_ids 至少提供一个", 400)
        include_relations = bool(body.get("include_relations", True))
        include_versions = bool(body.get("include_versions", True))
        max_episodes = min(max(int(body.get("max_episodes", 5000)), 1), 10000)
        max_concepts = min(max(int(body.get("max_concepts", 20000)), 1), 50000)
        result = storage.get_document_graph(
            document_version_ids=document_version_ids,
            document_family_ids=document_family_ids,
            include_relations=include_relations,
            include_versions=include_versions,
            max_episodes=max_episodes,
            max_concepts=max_concepts,
        )
        return ok(result)
    except ValueError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/documents/graph/outline", methods=["POST"])
def get_documents_graph_outline():
    """Return the fast Document -> Episode skeleton for progressive graph rendering."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "get_document_graph_outline"):
            return err("此功能暂不可用", 400)
        body = get_json_body()
        document_version_ids = body.get("document_version_ids") or []
        document_family_ids = body.get("document_family_ids") or []
        if isinstance(document_version_ids, str):
            document_version_ids = [document_version_ids]
        if isinstance(document_family_ids, str):
            document_family_ids = [document_family_ids]
        if not document_version_ids and not document_family_ids:
            return err("document_version_ids 或 document_family_ids 至少提供一个", 400)
        max_episodes = min(max(int(body.get("max_episodes", 10000)), 1), 10000)
        result = storage.get_document_graph_outline(
            document_version_ids=document_version_ids,
            document_family_ids=document_family_ids,
            max_episodes=max_episodes,
        )
        return ok(result)
    except ValueError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/documents/graph/chunk", methods=["POST"])
def get_documents_graph_chunk():
    """Return one episode-ordered concept batch for progressive graph rendering."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "get_document_graph_chunk"):
            return err("此功能暂不可用", 400)
        body = get_json_body()
        document_version_ids = body.get("document_version_ids") or []
        document_family_ids = body.get("document_family_ids") or []
        if isinstance(document_version_ids, str):
            document_version_ids = [document_version_ids]
        if isinstance(document_family_ids, str):
            document_family_ids = [document_family_ids]
        if not document_version_ids and not document_family_ids:
            return err("document_version_ids 或 document_family_ids 至少提供一个", 400)
        cursor = max(int(body.get("cursor", 0)), 0)
        limit = min(max(int(body.get("limit", 12)), 1), 100)
        include_relations = bool(body.get("include_relations", True))
        include_versions = bool(body.get("include_versions", True))
        max_concepts = min(max(int(body.get("max_concepts", 8000)), 1), 50000)
        result = storage.get_document_graph_chunk(
            document_version_ids=document_version_ids,
            document_family_ids=document_family_ids,
            cursor=cursor,
            limit=limit,
            include_relations=include_relations,
            include_versions=include_versions,
            max_concepts=max_concepts,
        )
        return ok(result)
    except ValueError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/documents/<document_version_id>/content", methods=["GET"])
def get_document_content(document_version_id: str):
    """Return Markdown source content for a document version."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "get_document_content"):
            return err("此功能暂不可用", 400)
        offset = max(int(request.args.get("offset", 0)), 0)
        limit = min(max(int(request.args.get("limit", 20000)), 1), 200000)
        result = storage.get_document_content(document_version_id, offset=offset, limit=limit)
        return ok(result)
    except KeyError as e:
        return err(str(e), 404)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/vaults/index", methods=["POST"])
def index_vault():
    """Index a read-only Markdown/Obsidian vault into the current graph."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "index_vault"):
            return err("此功能暂不可用", 400)
        body = get_json_body()
        path = (body.get("path") or body.get("vault_path") or "").strip()
        if not path:
            return err("path 不能为空", 400)
        force = bool(body.get("force", False))
        result = storage.index_vault(path, force=force)
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
            return err("此功能暂不可用", 400)
        time_point = (request.args.get("time_point") or "").strip() or None
        mentions = storage.get_concept_mentions(family_id, time_point=time_point)
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
            return err("此功能暂不可用", 400)
        body = get_json_body()
        algorithm = (body.get("algorithm") or "louvain").strip()
        resolution = float(body.get("resolution", 1.0))
        resolution = min(max(resolution, 0.1), 10.0)
        result = processor.storage.detect_communities(algorithm=algorithm, resolution=resolution)
        return ok(result)
    except ValueError as ve:
        return err(str(ve), 400)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/communities", methods=["GET"])
def list_communities():
    """列出社区（Neo4j 专属）。"""
    try:
        processor = _get_processor()
        if not hasattr(processor.storage, 'get_communities'):
            return err("此功能暂不可用", 404)
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
            return err("此功能暂不可用", 404)
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
            return err("此功能暂不可用", 404)
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
            return err("此功能暂不可用", 400)
        cleared = processor.storage.clear_communities()
        return ok({"cleared": cleared})
    except Exception as e:
        return err(str(e), 500)


# =========================================================
# Graphs management
# =========================================================

@concepts_bp.route("/api/v1/graphs", methods=["GET", "POST"])
def handle_graphs():
    """GET: 列出所有图谱（含元数据和统计信息）。POST: 创建新图谱。"""
    if request.method == "POST":
        try:
            data = request.get_json(force=True) or {}
            graph_id = (data.get("graph_id") or "").strip()
            registry = current_app.config["registry"]
            GraphRegistry.validate_graph_id(graph_id)
            # 检查是否已存在
            existing = registry.list_graphs()
            if graph_id in existing:
                return err(f"图谱 '{graph_id}' 已存在", 409)
            # 触发懒创建：访问 processor 即会初始化存储
            registry.get_processor(graph_id)
            # 持久化元数据
            metadata = registry.set_graph_metadata(
                graph_id,
                name=data.get("name", ""),
                description=data.get("description", ""),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            metadata["graph_id"] = graph_id
            return ok({"graph_id": graph_id, "message": "图谱创建成功", "metadata": metadata})
        except ValueError as e:
            return err(str(e), 400)
        except Exception as e:
            return err(str(e), 500)
    try:
        registry = current_app.config["registry"]
        graphs = registry.list_graphs_info()
        # Also return flat list for backward compat
        graph_ids = [g["graph_id"] for g in graphs]
        return ok({"graphs": graph_ids, "graphs_info": graphs, "count": len(graph_ids)})
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/graphs/<graph_id>/clear", methods=["POST"])
def clear_graph(graph_id: str):
    """清空图谱数据（保留图谱本身）。"""
    try:
        registry = current_app.config["registry"]
        GraphRegistry.validate_graph_id(graph_id)
        if graph_id not in registry.list_graphs():
            return err(f"图谱 '{graph_id}' 不存在", 404)
        registry.clear_graph(graph_id)
        return ok({"graph_id": graph_id, "message": "图谱已清空"})
    except KeyError as e:
        return err(str(e), 404)
    except ValueError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(str(e), 500)


@concepts_bp.route("/api/v1/graphs/<graph_id>", methods=["GET", "DELETE"])
def handle_single_graph(graph_id: str):
    """GET: 获取单个图谱详情。DELETE: 删除指定图谱。"""
    if request.method == "DELETE":
        try:
            registry = current_app.config["registry"]
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
    # GET
    try:
        registry = current_app.config["registry"]
        GraphRegistry.validate_graph_id(graph_id)
        info = registry.get_graph_info(graph_id)
        if info is None:
            return err(f"图谱 '{graph_id}' 不存在", 404)
        return ok(info)
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
        from core.server.chat_session import SessionManager
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
    body = get_json_body()
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
    body = get_json_body()
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
    body = get_json_body()
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


# ── Duplicate detection & merge ────────────────────────────────────────────

@concepts_bp.route("/api/v1/concepts/duplicates", methods=["GET"])
def find_duplicate_entities():
    """Detect potential duplicate entities by core-name matching.

    Groups entities whose names normalize to the same core (stripping
    parenthetical annotations and book marks) but have different family_ids.
    Returns groups with relation counts to help decide merge direction.
    """
    try:
        processor = _get_processor()

        limit = min(int(request.args.get("limit", 500)), 2000)
        entities = processor.storage.get_all_entities(limit=limit, exclude_embedding=True)

        def _normalize(name: str) -> str:
            n = _BOOK_MARKS_RE.sub('', name)
            n = _PAREN_ANNOTATION_RE.sub('', n)
            return n.strip()

        groups: Dict[str, list] = {}
        for e in entities:
            core = _normalize(getattr(e, 'name', ''))
            if not core or len(core) < 2:
                continue
            groups.setdefault(core, []).append(e)

        duplicates = []
        for core, items in sorted(groups.items()):
            fids = {getattr(e, 'family_id', '') for e in items}
            if len(fids) < 2:
                continue

            # Batch-fetch relation counts and version counts (avoid N+1 per entity)
            all_fids = list(fids)
            if hasattr(processor.storage, 'count_entity_relations_by_family_ids'):
                rel_counts = processor.storage.count_entity_relations_by_family_ids(all_fids)
            else:
                rel_counts = {fid: len(processor.storage.get_entity_relations_by_family_id(fid)) for fid in all_fids}
            if hasattr(processor.storage, 'get_entity_version_counts'):
                ver_counts = processor.storage.get_entity_version_counts(all_fids)
            else:
                ver_counts = {fid: processor.storage.get_entity_version_count(fid) for fid in all_fids}

            group = {
                "core_name": core,
                "entities": [],
            }
            for e in items:
                group["entities"].append({
                    "family_id": e.family_id,
                    "name": getattr(e, 'name', ''),
                    "relation_count": rel_counts.get(e.family_id, 0),
                    "version_count": ver_counts.get(e.family_id, 0),
                })
            # Sort by relation_count desc (first entity is merge target)
            group["entities"].sort(key=lambda x: x["relation_count"], reverse=True)
            duplicates.append(group)

        return ok({"duplicates": duplicates, "count": len(duplicates)})
    except Exception as e:
        return err(str(e), 500)

