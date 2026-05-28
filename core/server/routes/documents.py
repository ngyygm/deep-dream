"""Document-first API routes for Deep-Dream Vault."""
from __future__ import annotations

from flask import Blueprint, request

from core.documents import DocumentService
from core.server.routes.helpers import _get_processor, err, ok

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
    except KeyError as exc:
        return err(str(exc), 404)
    except FileNotFoundError as exc:
        return err(str(exc), 404)
    except ValueError as exc:
        return err(str(exc), 400)
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
