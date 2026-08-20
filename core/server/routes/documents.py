"""Document-first API routes for Deep-Dream Vault.

P4.3：/documents*、/episodes*、/vaults/index 图谱路由自 concepts.py 纯移动
而来（URL/方法/状态码/响应字段逐一不变）；本模块原有的文件面路由不变。
"""
from __future__ import annotations

import logging

from flask import Blueprint, current_app, request

from core.documents import DocumentService
from core.server.routes.helpers import _get_processor, err, get_json_body, ok

logger = logging.getLogger(__name__)

documents_bp = Blueprint("documents", __name__)


def _document_service() -> DocumentService:
    processor = _get_processor()
    return DocumentService(processor.storage)


@documents_bp.route("/api/v1/documents/map", methods=["GET"])
def map_document_path():
    """Map a local file path back to indexed Deep-Dream documents."""
    try:
        path = (request.args.get("path") or "").strip()
        try:
            limit = min(max(int(request.args.get("limit", 20)), 1), 100)
        except (TypeError, ValueError):
            return err("limit 必须为整数", 400)
        return ok(_document_service().map_path(path, limit=limit))
    except ValueError as exc:
        return err(str(exc), 400)
    except Exception as exc:
        return err(str(exc), 500)


@documents_bp.route("/api/v1/documents/search", methods=["GET"])
def search_document_files():
    """Search raw readable files before entering the graph layer."""
    try:
        query = (request.args.get("q") or request.args.get("query") or "").strip()
        regex = (request.args.get("regex") or "").lower() in {"1", "true", "yes", "on"}
        try:
            limit = min(max(int(request.args.get("limit", 50)), 1), 500)
        except (TypeError, ValueError):
            return err("limit 必须为整数", 400)
        return ok(_document_service().search_files(query, regex=regex, limit=limit))
    except ValueError as exc:
        return err(str(exc), 400)
    except Exception as exc:
        return err(str(exc), 500)


@documents_bp.route("/api/v1/documents/<document_version_id>/content", methods=["GET"])
def read_document_content(document_version_id: str):
    """Read a document slice from raw file, managed file, or snapshot fallback."""
    try:
        try:
            offset = max(int(request.args.get("offset", 0)), 0)
            limit = min(max(int(request.args.get("limit", 20000)), 1), 10_000_000)
        except (TypeError, ValueError):
            return err("offset/limit 必须为整数", 400)
        return ok(_document_service().read_document(document_version_id, offset=offset, limit=limit))
    except KeyError:
        return err(f"文档版本不存在: {document_version_id}", 404)
    except FileNotFoundError as exc:
        return err(str(exc), 404)
    except ValueError as exc:
        return err(str(exc), 400)
    except Exception as exc:
        return err(str(exc), 500)


@documents_bp.route("/api/v1/documents/<document_version_id>/file", methods=["GET"])
def document_file_info(document_version_id: str):
    """Return file metadata (path, size, hash, verification) for a document version."""
    try:
        ds = _document_service()
        rows = ds.storage.read_sql(
            """
            SELECT document_version_id, document_family_id, title, source_mode,
                   absolute_path, managed_path, snapshot_path, relative_path,
                   vault_root, read_path, content_hash, byte_size, char_count,
                   line_count, processed_time
            FROM v_document_files
            WHERE document_version_id = :document_version_id
            LIMIT 1
            """,
            params={"document_version_id": document_version_id},
            limit=1,
        )["rows"]
        if not rows:
            return err(f"文档版本不存在: {document_version_id}", 404)
        doc = rows[0]
        payload = ds._document_file_payload(doc)
        return ok(payload)
    except KeyError as exc:
        return err(str(exc), 404)
    except Exception as exc:
        return err(str(exc), 500)


@documents_bp.route("/api/v1/vaults/tree", methods=["GET"])
def get_vault_tree():
    """Return a file-tree friendly view of indexed vault documents."""
    try:
        vault_root = (request.args.get("vault_root") or "").strip() or None
        try:
            limit = min(max(int(request.args.get("limit", 5000)), 1), 20000)
        except (TypeError, ValueError):
            return err("limit 必须为整数", 400)
        return ok(_document_service().vault_tree(vault_root=vault_root, limit=limit))
    except Exception as exc:
        return err(str(exc), 500)


# ── 以下路由 P4.3 自 concepts.py 纯移动而来（逻辑零改动） ────────────────────

@documents_bp.route("/api/v1/documents", methods=["GET"])
def list_documents():
    """List indexed Markdown documents for the current graph."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "list_documents"):
            return err("此功能暂不可用", 400)
        try:
            limit = min(max(int(request.args.get("limit", 50)), 1), 200)
        except (ValueError, TypeError):
            return err("limit 必须为整数", 400)
        try:
            offset = max(int(request.args.get("offset", 0)), 0)
        except (ValueError, TypeError):
            return err("offset 必须为整数", 400)
        source_document = (request.args.get("source_document") or "").strip() or None
        documents = storage.list_documents(limit=limit, offset=offset, source_document=source_document)
        runtime = (current_app.config.get("config") or {}).get("runtime") or {}
        integrity_cfg = runtime.get("integrity") or {}
        # P3.7：列表页默认不再逐文档做完整性全量评估（每篇都要读全文 + 重切块 +
        # 逐窗口查库，代价随文档数×窗口数线性叠加，且 chunk_hash 无索引时每次
        # 查询都是 episodes 全表扫描——列表页不可承受）。完整性改为按需拉取：
        # GET /api/v1/documents/<id>/integrity（memory 页已有"检查"按钮走该端点）。
        # 前端对缺失的 integrity 字段已有降级显示（memory.js 显示"未检查"，
        # graph.js 的窗口数改用列表自带的轻量 episode_count 字段）。
        # 仅当用户显式配置 runtime.integrity.auto_check_documents=true 时保留旧行为。
        if bool(integrity_cfg.get("auto_check_documents", False)) and documents:
            try:
                from core.server.routes.remember import _get_queue as _get_remember_queue
                remember_queue = _get_remember_queue()
                for doc in documents:
                    doc_id = doc.get("document_version_id")
                    if not doc_id:
                        continue
                    try:
                        integrity = remember_queue.assess_document_integrity(doc_id)
                        doc["integrity"] = {
                            "complete": bool(integrity.get("complete")),
                            "total_windows": integrity.get("total_windows", 0),
                            "complete_windows": integrity.get("complete_windows", 0),
                            "missing_windows": integrity.get("missing_windows", 0),
                            "missing_window_indices": integrity.get("missing_window_indices", [])[:20],
                        }
                        if hasattr(storage, "update_document_integrity_metadata"):
                            storage.update_document_integrity_metadata(doc_id, integrity)
                    except Exception as exc:
                        doc["integrity"] = {"complete": None, "error": str(exc)}
            except Exception as exc:
                logger.debug("document integrity auto-check skipped: %s", exc)
        # Get actual total count (independent of pagination)
        total = storage.count_documents(source_document=source_document) if hasattr(storage, "count_documents") else len(documents)
        return ok({"documents": documents, "total": total, "limit": limit, "offset": offset})
    except Exception as e:
        return err(str(e), 500)


@documents_bp.route("/api/v1/documents/graph", methods=["POST"])
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
        try:
            max_episodes = min(max(int(body.get("max_episodes", 5000)), 1), 10000)
        except (ValueError, TypeError):
            return err("max_episodes 必须为整数", 400)
        try:
            max_concepts = min(max(int(body.get("max_concepts", 20000)), 1), 50000)
        except (ValueError, TypeError):
            return err("max_concepts 必须为整数", 400)
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


@documents_bp.route("/api/v1/documents/graph/outline", methods=["POST"])
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
        try:
            max_episodes = min(max(int(body.get("max_episodes", 10000)), 1), 10000)
        except (ValueError, TypeError):
            return err("max_episodes 必须为整数", 400)
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


@documents_bp.route("/api/v1/documents/graph/chunk", methods=["POST"])
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
        try:
            cursor = max(int(body.get("cursor", 0)), 0)
        except (ValueError, TypeError):
            return err("cursor 必须为整数", 400)
        try:
            limit = min(max(int(body.get("limit", 12)), 1), 100)
        except (ValueError, TypeError):
            return err("limit 必须为整数", 400)
        include_relations = bool(body.get("include_relations", True))
        include_versions = bool(body.get("include_versions", True))
        try:
            max_concepts = min(max(int(body.get("max_concepts", 8000)), 1), 50000)
        except (ValueError, TypeError):
            return err("max_concepts 必须为整数", 400)
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


@documents_bp.route("/api/v1/episodes/<episode_version_id>/content", methods=["GET"])
def get_episode_content(episode_version_id: str):
    """Return source content for an episode version."""
    try:
        processor = _get_processor()
        storage = processor.storage
        if hasattr(storage, "get_episode_content_detail"):
            detail = storage.get_episode_content_detail(episode_version_id)
            if detail is None:
                return err(f"episode_version_id 不存在: {episode_version_id}", 404)
            return ok(detail)
        if not hasattr(storage, "load_episode"):
            return err("此功能暂不可用", 400)
        episode = storage.load_episode(episode_version_id)
        if episode is None:
            return err(f"episode_version_id 不存在: {episode_version_id}", 404)
        return ok({
            "episode_id": episode.absolute_id,
            "content": episode.content,
            "source_document": episode.source_document or "",
            "event_time": episode.event_time.isoformat() if episode.event_time else None,
            "processed_time": episode.processed_time.isoformat() if episode.processed_time else None,
            "activity_type": getattr(episode, "activity_type", None),
            "episode_type": getattr(episode, "episode_type", None),
        })
    except Exception as e:
        return err(str(e), 500)


@documents_bp.route("/api/v1/documents/batch", methods=["DELETE"])
def batch_delete_documents():
    """批量删除文档版本。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "delete_document_version"):
            return err("此功能暂不可用", 400)
        body = get_json_body()
        ids = body.get("document_version_ids") or []
        if not ids:
            return err("document_version_ids 不能为空", 400)
        if not isinstance(ids, list) or len(ids) > 100:
            return err("document_version_ids 必须为列表，最多 100 个", 400)
        results = []
        for doc_id in ids:
            try:
                result = storage.delete_document_version(doc_id)
                if isinstance(result, dict) and not result.get("deleted", True):
                    results.append({"id": doc_id, "success": False, "error": result.get("reason", "not found")})
                else:
                    results.append({"id": doc_id, "success": True})
            except Exception as e:
                results.append({"id": doc_id, "success": False, "error": str(e)})
        return ok({"results": results, "deleted": sum(1 for r in results if r["success"])})
    except Exception as e:
        return err(str(e), 500)


@documents_bp.route("/api/v1/documents/<document_version_id>", methods=["DELETE"])
def delete_document_version(document_version_id: str):
    """删除文档版本，以及该文档下的 episode/concept version/edge。"""
    try:
        processor = _get_processor()
        storage = processor.storage
        if not hasattr(storage, "delete_document_version"):
            return err("此功能暂不可用", 400)
        result = storage.delete_document_version(document_version_id)
        if isinstance(result, dict) and not result.get("deleted", True):
            reason = result.get("reason", "not found")
            return err(f"文档版本不存在: {document_version_id} ({reason})", 404)
        return ok(result)
    except KeyError as e:
        return err(str(e.args[0]) if e.args else str(e), 404)
    except ValueError as e:
        return err(str(e), 400)
    except Exception as e:
        return err(str(e), 500)


@documents_bp.route("/api/v1/vaults/index", methods=["POST"])
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
