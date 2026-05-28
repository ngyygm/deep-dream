"""Document-first service facade.

This module keeps the agent-facing document workflow out of the SQLite storage
manager. Storage still owns persistence; this facade owns path mapping, raw-file
search, source fallback, and vault tree shaping.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Optional


class DocumentService:
    """High-level document operations for files-first Deep-Dream usage."""

    def __init__(self, storage: Any):
        self.storage = storage

    def list_documents(self, *, limit: int = 100, offset: int = 0, query: Optional[str] = None) -> dict:
        documents = self.storage.list_documents(limit=limit, offset=offset, source_document=query)
        total = (
            self.storage.count_documents(source_document=query)
            if hasattr(self.storage, "count_documents")
            else len(documents)
        )
        return {"documents": documents, "total": total, "limit": limit, "offset": offset}

    def map_path(self, path: str, *, limit: int = 20) -> dict:
        raw = str(path or "").strip()
        if not raw:
            raise ValueError("path 不能为空")
        resolved = str(Path(raw).expanduser().resolve())
        rows = self.storage.read_sql(
            """
            SELECT *
            FROM v_document_files
            WHERE absolute_path IN (:raw, :resolved)
               OR managed_path IN (:raw, :resolved)
               OR snapshot_path IN (:raw, :resolved)
               OR read_path IN (:raw, :resolved)
               OR relative_path = :raw
            ORDER BY processed_time DESC
            """,
            params={"raw": raw, "resolved": resolved},
            limit=limit,
        )["rows"]
        if not rows:
            rows = []
            for doc in self._document_rows(limit=5000):
                payload = self._document_file_payload(doc)
                doc_path = payload.get("resolved_path")
                if doc_path and str(Path(doc_path).resolve()) == resolved:
                    rows.append(doc)
                    if len(rows) >= limit:
                        break
        return {
            "path": raw,
            "resolved_path": resolved,
            "documents": [self._document_file_payload(row) for row in rows],
            "total": len(rows),
        }

    def read_document(self, document_version_id: str, *, offset: int = 0, limit: int = 10_000_000) -> dict:
        return self.storage.get_document_content(document_version_id, offset=offset, limit=limit)

    def search_files(self, query: str, *, regex: bool = False, limit: int = 50) -> dict:
        pattern = str(query or "").strip()
        if not pattern:
            raise ValueError("query 不能为空")
        matcher = re.compile(pattern, re.IGNORECASE) if regex else None
        hits: list[dict] = []
        for doc in self._iter_searchable_documents():
            path = Path(doc["resolved_path"])
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                try:
                    lines = path.read_text(encoding="utf-8-sig").splitlines()
                except Exception:
                    continue
            except OSError:
                continue
            for line_no, line in enumerate(lines, start=1):
                matched = bool(matcher.search(line)) if matcher else pattern.lower() in line.lower()
                if not matched:
                    continue
                hits.append({
                    "document": {
                        "document_version_id": doc.get("document_version_id", ""),
                        "document_family_id": doc.get("document_family_id", ""),
                        "title": doc.get("title", ""),
                        "read_path": doc.get("resolved_path") or doc.get("read_path", ""),
                        "source_mode": doc.get("source_mode", ""),
                        "line_start": line_no,
                        "line_end": line_no,
                    },
                    "verification": doc.get("verification", "raw_file"),
                    "text": line,
                })
                if len(hits) >= limit:
                    return self._search_payload(pattern, regex, hits)
        return self._search_payload(pattern, regex, hits)

    def vault_tree(self, *, vault_root: Optional[str] = None, limit: int = 5000) -> dict:
        docs = self._document_rows(limit=limit)
        if vault_root:
            root_norm = str(Path(vault_root).expanduser().resolve())
            docs = [
                doc for doc in docs
                if str(Path(doc.get("vault_root") or "").expanduser().resolve()) == root_norm
            ]
        roots: dict[str, dict] = {}
        loose_files: list[dict] = []
        for doc in docs:
            root = doc.get("vault_root") or ""
            rel = doc.get("relative_path") or doc.get("title") or doc.get("read_path") or ""
            item = {
                "name": Path(rel).name,
                "relative_path": rel,
                "document_version_id": doc.get("document_version_id", ""),
                "document_family_id": doc.get("document_family_id", ""),
                "title": doc.get("title", ""),
                "source_mode": doc.get("source_mode", ""),
                "read_path": doc.get("read_path", ""),
            }
            if not root:
                loose_files.append(item)
                continue
            root_entry = roots.setdefault(root, {"vault_root": root, "files": []})
            root_entry["files"].append(item)
        for root_entry in roots.values():
            root_entry["files"].sort(key=lambda item: item["relative_path"].lower())
        loose_files.sort(key=lambda item: item["relative_path"].lower())
        return {
            "vaults": sorted(roots.values(), key=lambda item: item["vault_root"].lower()),
            "loose_files": loose_files,
            "total": len(docs),
        }

    def _search_payload(self, query: str, regex: bool, hits: list[dict]) -> dict:
        return {
            "query": query,
            "regex": regex,
            "hits": hits,
            "total": len(hits),
            "used": {"raw_files": True, "episodes": False, "concepts": False},
        }

    def _document_rows(self, *, limit: int = 5000) -> list[dict]:
        return self.storage.read_sql(
            """
            SELECT document_version_id, document_family_id, title, source_mode,
                   absolute_path, managed_path, snapshot_path, relative_path,
                   vault_root, read_path, content_hash, byte_size, char_count,
                   line_count, processed_time, complete_windows, total_windows,
                   missing_windows
            FROM v_document_files
            ORDER BY processed_time DESC
            """,
            limit=limit,
        )["rows"]

    def _iter_searchable_documents(self) -> Iterable[dict]:
        for doc in self._document_rows():
            payload = self._document_file_payload(doc)
            if payload.get("resolved_path"):
                yield payload

    def _document_file_payload(self, doc: dict) -> dict:
        path, verification = self._readable_document_path(doc)
        item = dict(doc)
        item["resolved_path"] = str(path) if path else ""
        item["verification"] = verification
        return item

    def _readable_document_path(self, doc: dict) -> tuple[Optional[Path], str]:
        candidates: list[tuple[str, str]] = []
        if doc.get("source_mode") == "external" and doc.get("absolute_path"):
            candidates.append((doc["absolute_path"], "raw_file"))
        for key, label in (
            ("read_path", "raw_file"),
            ("managed_path", "raw_file"),
            ("snapshot_path", "snapshot"),
            ("absolute_path", "raw_file"),
        ):
            value = doc.get(key) or ""
            if value:
                candidates.append((value, label))
        seen: set[str] = set()
        for value, label in candidates:
            if value in seen:
                continue
            seen.add(value)
            path = self._resolve_path(value)
            if path.is_file():
                return path, label
        return None, "missing"

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        resolver = getattr(self.storage, "_resolve_storage_path", None)
        if resolver is not None:
            try:
                return resolver(value)
            except Exception:
                pass
        return Path(self.storage.storage_path) / value
