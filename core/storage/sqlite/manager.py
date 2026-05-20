"""Concept-backed SQLite storage manager.

The remember pipeline still passes legacy Entity/Relation/Episode DTOs through
step9/step10. This manager treats those DTOs as adapters and persists them into
the v1 Document-first Concept graph tables.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ...text_chunking import find_text_evidence, split_markdown_chunks
from ...models import Entity, Episode, Relation
from ..cache import QueryCache
from .helpers import _encode_and_normalize, _fmt_dt, _parse_dt
from .schema import init_schema

logger = logging.getLogger(__name__)


ROLE_DOCUMENT = "document"
ROLE_EPISODE = "episode"
ROLE_ENTITY = "entity"
ROLE_RELATION = "relation"

EDGE_DOCUMENT_LINK = "DOCUMENT_LINK"
EDGE_HAS_EPISODE = "HAS_EPISODE"
EDGE_MENTIONS = "MENTIONS"
EDGE_ASSERTS = "ASSERTS"
EDGE_CONNECTS = "CONNECTS"


def _json_dumps(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return {} if default is None else default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return {} if default is None else default


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _stable_id(prefix: str, value: str, length: int = 16) -> str:
    return f"{prefix}_{_sha256_text(value)[:length]}"


def _now() -> datetime:
    return datetime.now()


def _sort_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _is_task_original_path(path: str) -> bool:
    if not path:
        return False
    parts = {p.lower() for p in Path(path).parts}
    return "tasks" in parts and "originals" in parts


class SQLiteGraphStorageManager:
    """SQLite storage facade used by the existing remember pipeline."""

    def __init__(
        self,
        storage_path: str = "./graph/default",
        embedding_client=None,
        entity_content_snippet_length: int = 50,
        relation_content_snippet_length: int = 50,
        vector_dim: int = 1024,
        graph_id: str = "default",
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._db_path = self.storage_path / "graph.db"
        self._graph_id = graph_id
        self.embedding_client = embedding_client
        self.entity_content_snippet_length = entity_content_snippet_length
        self.relation_content_snippet_length = relation_content_snippet_length
        self._vector_dim = vector_dim

        self.blobs_dir = self.storage_path / "blobs" / "documents" / "sha256"
        self.artifacts_dir = self.storage_path / "artifacts" / "episodes"
        self.indexes_dir = self.storage_path / "indexes"
        self.logs_dir = self.storage_path / "logs"
        for d in (self.blobs_dir, self.artifacts_dir, self.indexes_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Legacy attributes still read by tools/monitoring.
        self.docs_dir = self.storage_path / "artifacts" / "legacy_docs"
        self.cache_dir = self.storage_path / "artifacts" / "legacy_episodes"
        self.cache_json_dir = self.cache_dir / "json"
        self.cache_md_dir = self.cache_dir / "md"
        for d in (self.docs_dir, self.cache_json_dir, self.cache_md_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._thread_local = threading.local()
        self._all_conns: List[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()
        self._write_lock = threading.RLock()
        self._entity_name_cache: Dict[str, str] = {}
        self._cache = QueryCache(default_ttl=30, max_size=4096)

        conn = self._connect()
        init_schema(conn)

    # ------------------------------------------------------------------
    # Connection and low-level helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._thread_local, "conn", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        self._thread_local.conn = conn
        with self._conn_lock:
            self._all_conns.append(conn)
        return conn

    def close(self):
        with self._conn_lock:
            conns = list(self._all_conns)
            self._all_conns.clear()
        for conn in conns:
            try:
                conn.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _cache_entity_name(self, absolute_id: str, name: str):
        if absolute_id:
            self._entity_name_cache[absolute_id] = name or ""

    def _resolve_redirect(self, family_id: str) -> str:
        if not family_id:
            return family_id
        conn = self._connect()
        seen = set()
        current = family_id
        for _ in range(16):
            if current in seen:
                break
            seen.add(current)
            row = conn.execute(
                "SELECT target_family_id FROM concept_redirect WHERE source_family_id = ? AND graph_id = ?",
                (current, self._graph_id),
            ).fetchone()
            if not row:
                break
            current = row["target_family_id"]
        return current

    def resolve_family_id(self, family_id: str) -> str:
        return self._resolve_redirect(family_id)

    def resolve_family_ids(self, family_ids: Iterable[str]) -> Dict[str, str]:
        return {fid: self._resolve_redirect(fid) for fid in family_ids if fid}

    def _upsert_family(self, family_id: str, role: str, canonical_name: str = "", metadata: Any = None):
        now = _fmt_dt(_now())
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO concept_family
              (family_id, graph_id, role, canonical_name, status, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, 'active', ?, ?, ?)
            ON CONFLICT(family_id) DO UPDATE SET
              role = excluded.role,
              canonical_name = CASE
                WHEN excluded.canonical_name != '' THEN excluded.canonical_name
                ELSE concept_family.canonical_name
              END,
              updated_at = excluded.updated_at,
              metadata = excluded.metadata
            """,
            (family_id, self._graph_id, role, canonical_name or "", now, now, _json_dumps(metadata or {})),
        )

    def _latest_version_row(self, family_id: str, role: Optional[str] = None):
        family_id = self._resolve_redirect(family_id)
        conn = self._connect()
        sql = "SELECT * FROM concept_version WHERE family_id = ? AND graph_id = ?"
        params: List[Any] = [family_id, self._graph_id]
        if role:
            sql += " AND role = ?"
            params.append(role)
        sql += " ORDER BY version_seq DESC LIMIT 1"
        return conn.execute(sql, params).fetchone()

    def _episode_version_row(self, family_id: str, role: str, episode_version_id: str):
        if not family_id or not episode_version_id:
            return None
        family_id = self._resolve_redirect(family_id)
        return self._connect().execute(
            """
            SELECT * FROM concept_version
            WHERE family_id = ? AND role = ? AND episode_version_id = ? AND graph_id = ?
            ORDER BY version_seq ASC
            LIMIT 1
            """,
            (family_id, role, episode_version_id, self._graph_id),
        ).fetchone()

    def _document_version_for_episode(self, episode_version_id: str) -> str:
        if not episode_version_id:
            return ""
        row = self._connect().execute(
            "SELECT document_version_id FROM concept_version WHERE version_id = ? AND role = ? AND graph_id = ?",
            (episode_version_id, ROLE_EPISODE, self._graph_id),
        ).fetchone()
        return row["document_version_id"] if row and row["document_version_id"] else ""

    def _next_version_seq(self, family_id: str) -> int:
        row = self._connect().execute(
            "SELECT COALESCE(MAX(version_seq), 0) AS max_seq FROM concept_version WHERE family_id = ? AND graph_id = ?",
            (family_id, self._graph_id),
        ).fetchone()
        return int(row["max_seq"] or 0) + 1

    def _insert_fts(self, version_id: str, family_id: str, role: str, name: str, content: str):
        conn = self._connect()
        try:
            conn.execute("DELETE FROM concept_version_fts WHERE version_id = ?", (version_id,))
        except Exception:
            pass
        conn.execute(
            "INSERT INTO concept_version_fts (name, content, role, family_id, version_id, graph_id) VALUES (?, ?, ?, ?, ?, ?)",
            (name or "", content or "", role, family_id, version_id, self._graph_id),
        )

    def _create_edge(
        self,
        edge_type: str,
        source_family_id: str = "",
        source_version_id: str = "",
        target_family_id: str = "",
        target_version_id: str = "",
        relation_family_id: str = "",
        relation_version_id: str = "",
        episode_version_id: str = "",
        document_version_id: str = "",
        weight: float = 1.0,
        confidence: Optional[float] = None,
        provenance: Any = None,
    ) -> str:
        edge_id = f"edge_{uuid.uuid4().hex}"
        self._connect().execute(
            """
            INSERT OR REPLACE INTO concept_edge
              (edge_id, graph_id, edge_type, source_family_id, source_version_id,
               target_family_id, target_version_id, relation_family_id,
               relation_version_id, episode_version_id, document_version_id,
               weight, confidence, provenance, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge_id, self._graph_id, edge_type,
                source_family_id or "", source_version_id or "",
                target_family_id or "", target_version_id or "",
                relation_family_id or "", relation_version_id or "",
                episode_version_id or "", document_version_id or "",
                weight, confidence, _json_dumps(provenance or {}), _fmt_dt(_now()),
            ),
        )
        return edge_id

    def _blob_write_markdown(self, content: str) -> Tuple[str, str]:
        content_hash = _sha256_text(content)
        bucket = self.blobs_dir / content_hash[:2]
        bucket.mkdir(parents=True, exist_ok=True)
        path = bucket / f"{content_hash}.md"
        if not path.exists():
            path.write_text(content or "", encoding="utf-8")
        rel = str(path.relative_to(self.storage_path)).replace("\\", "/")
        self._connect().execute(
            "INSERT OR IGNORE INTO blob_manifest (content_hash, graph_id, blob_path, size, created_at) VALUES (?, ?, ?, ?, ?)",
            (content_hash, self._graph_id, rel, len((content or "").encode("utf-8")), _fmt_dt(_now())),
        )
        return content_hash, rel

    def _save_document_snapshot(
        self,
        source_key: str,
        title: str,
        content: str,
        *,
        absolute_path: str = "",
        relative_path: str = "",
        frontmatter: Any = None,
        tags: Any = None,
        aliases: Any = None,
        mtime: Any = None,
        metadata: Any = None,
    ) -> Tuple[str, str, str, str]:
        content_hash, blob_rel = self._blob_write_markdown(content)
        doc_family_id = _stable_id("docfam", source_key)
        doc_version_id = f"docver_{content_hash[:16]}"
        source_id = _stable_id("docsrc", source_key)
        now_iso = _fmt_dt(_now())

        self._upsert_family(doc_family_id, ROLE_DOCUMENT, title, {"source_key": source_key})
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO document_source
              (source_id, graph_id, document_family_id, vault_id, absolute_path,
               relative_path, uri, title, created_at, updated_at, metadata)
            VALUES (?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
              absolute_path = excluded.absolute_path,
              relative_path = excluded.relative_path,
              uri = excluded.uri,
              title = excluded.title,
              updated_at = excluded.updated_at,
              metadata = excluded.metadata
            """,
            (
                source_id, self._graph_id, doc_family_id,
                absolute_path or "", relative_path or "",
                source_key if source_key.startswith("api://") else "",
                title or "", now_iso, now_iso, _json_dumps(metadata or {}),
            ),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO document_version
              (document_version_id, graph_id, document_family_id, source_id,
               content_hash, blob_path, title, frontmatter_json, tags_json,
               aliases_json, mtime, size, processed_time, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_version_id, self._graph_id, doc_family_id, source_id,
                content_hash, blob_rel, title or "",
                _json_dumps(frontmatter or {}), _json_dumps(tags or []),
                _json_dumps(aliases or []), _fmt_dt(mtime),
                len((content or "").encode("utf-8")), now_iso,
                _json_dumps(metadata or {}),
            ),
        )
        existing_doc_version = conn.execute(
            "SELECT 1 FROM concept_version WHERE version_id = ? AND graph_id = ?",
            (doc_version_id, self._graph_id),
        ).fetchone()
        if not existing_doc_version:
            self._append_version(
                doc_family_id, ROLE_DOCUMENT, doc_version_id, title, content,
                event_time=_now(),
                processed_time=_now(),
                document_version_id=doc_version_id,
                source_document=title,
                metadata={"content_hash": content_hash, "blob_path": blob_rel, **(metadata or {})},
            )
        return doc_family_id, doc_version_id, content_hash, blob_rel

    def _save_episode_chunk(
        self,
        *,
        document_family_id: str,
        document_version_id: str,
        title: str,
        chunk: str,
        heading_path: str,
        start_offset: int,
        end_offset: int,
        chunk_index: int,
        source_document: str,
    ) -> str:
        heading_hash = _sha256_text(heading_path or "root")[:12]
        ep_family_id = f"epfam_{document_family_id}_{heading_hash}_{chunk_index}"
        ep_version_id = f"epver_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        chunk_hash = _sha256_text(chunk)
        self._upsert_family(
            ep_family_id,
            ROLE_EPISODE,
            heading_path or title,
            {"document_family_id": document_family_id, "heading_path": heading_path, "chunk_index": chunk_index},
        )
        self._append_version(
            ep_family_id, ROLE_EPISODE, ep_version_id,
            heading_path or title, chunk,
            event_time=_now(),
            processed_time=_now(),
            episode_version_id=ep_version_id,
            document_version_id=document_version_id,
            source_document=source_document,
            metadata={
                "document_family_id": document_family_id,
                "heading_path": heading_path,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "chunk_index": chunk_index,
                "chunk_hash": chunk_hash,
            },
        )
        self._create_edge(
            EDGE_HAS_EPISODE,
            source_family_id=document_family_id,
            source_version_id=document_version_id,
            target_family_id=ep_family_id,
            target_version_id=ep_version_id,
            episode_version_id=ep_version_id,
            document_version_id=document_version_id,
        )
        artifact_dir = self.artifacts_dir / ep_version_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "source.md").write_text(chunk or "", encoding="utf-8")
        return ep_version_id

    def _compute_embedding(self, role: str, name: str, content: str) -> Optional[bytes]:
        if role == ROLE_ENTITY:
            text = f"# {name}\n{(content or '')[:512]}"
        elif role == ROLE_RELATION:
            text = (content or "")[:1024]
        else:
            text = f"# {role}\n{(content or '')[:1024]}"
        result = _encode_and_normalize(self.embedding_client, text)
        return result[0] if result else None

    def _compute_entity_embedding(self, entity: Entity):
        text = f"# {getattr(entity, 'name', '')}\n{(getattr(entity, 'content', '') or '')[:512]}"
        return _encode_and_normalize(self.embedding_client, text)

    def _compute_relation_embedding(self, relation: Relation):
        return self._compute_embedding(ROLE_RELATION, "", getattr(relation, "content", ""))

    def _append_version(
        self,
        family_id: str,
        role: str,
        version_id: str,
        name: str = "",
        content: str = "",
        event_time: Any = None,
        processed_time: Any = None,
        episode_version_id: str = "",
        document_version_id: str = "",
        source_document: str = "",
        summary: Optional[str] = None,
        attributes: Any = None,
        confidence: Optional[float] = None,
        content_format: str = "markdown",
        embedding: Optional[bytes] = None,
        metadata: Any = None,
    ) -> int:
        latest = self._latest_version_row(family_id, role=role)
        latest_content = latest["content"] if latest else None
        content_changed = 1 if latest is None or (latest_content or "") != (content or "") else 0
        version_seq = self._next_version_seq(family_id)
        processed = processed_time or _now()
        if embedding is None:
            embedding = self._compute_embedding(role, name, content)
        self._connect().execute(
            """
            INSERT OR REPLACE INTO concept_version
              (version_id, family_id, graph_id, role, name, content, summary,
               attributes, confidence, content_format, content_changed,
               version_seq, valid_at, event_time, processed_time,
               episode_version_id, document_version_id, source_document,
               embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id, family_id, self._graph_id, role, name or "", content or "",
                summary, _json_dumps(attributes), confidence, content_format,
                content_changed, version_seq, _fmt_dt(event_time),
                _fmt_dt(event_time), _fmt_dt(processed),
                episode_version_id or "", document_version_id or "",
                source_document or "", embedding, _json_dumps(metadata or {}),
            ),
        )
        self._insert_fts(version_id, family_id, role, name, content)
        return version_seq

    # ------------------------------------------------------------------
    # Document / episode storage
    # ------------------------------------------------------------------

    def save_episode(
        self,
        cache: Episode,
        text: str = "",
        document_path: str = "",
        doc_hash: str = "",
        start_offset: int = 0,
        end_offset: Optional[int] = None,
    ) -> str:
        episode_content = text or getattr(cache, "content", "") or ""
        document_content = episode_content
        if document_path and _is_task_original_path(document_path):
            try:
                p = Path(document_path)
                if p.exists():
                    document_content = p.read_text(encoding="utf-8")
            except Exception as exc:
                logger.debug("Failed to read remember original document %s: %s", document_path, exc)
        content_hash, blob_rel = self._blob_write_markdown(document_content)
        doc_hash = doc_hash or content_hash[:12]
        source_doc_name = getattr(cache, "source_document", "") or ""
        source_key = document_path if document_path and not _is_task_original_path(document_path) else ""
        source_key = source_key or source_doc_name or doc_hash
        doc_family_id = _stable_id("docfam", source_key)
        doc_version_id = f"docver_{content_hash[:16]}"
        source_id = _stable_id("docsrc", source_key)
        title = Path(source_key).name if source_key and not source_key.startswith("api:") else (source_doc_name or source_key)
        now_iso = _fmt_dt(_now())

        with self._write_lock:
            conn = self._connect()
            self._upsert_family(doc_family_id, ROLE_DOCUMENT, title, {"source_key": source_key})
            conn.execute(
                """
                INSERT INTO document_source
                  (source_id, graph_id, document_family_id, vault_id, absolute_path,
                   relative_path, uri, title, created_at, updated_at, metadata)
                VALUES (?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                  absolute_path = excluded.absolute_path,
                  relative_path = excluded.relative_path,
                  uri = excluded.uri,
                  title = excluded.title,
                  updated_at = excluded.updated_at,
                  metadata = excluded.metadata
                """,
                (
                    source_id, self._graph_id, doc_family_id,
                    str(Path(document_path).resolve()) if document_path and not document_path.startswith("api://") and not _is_task_original_path(document_path) else "",
                    source_key or "", source_key if source_key.startswith("api://") else "",
                    title, now_iso, now_iso, _json_dumps({"source_document": getattr(cache, "source_document", "")}),
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO document_version
                  (document_version_id, graph_id, document_family_id, source_id,
                   content_hash, blob_path, title, frontmatter_json, tags_json,
                   aliases_json, mtime, size, processed_time, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, '{}', '[]', '[]', ?, ?, ?, ?)
                """,
                (
                    doc_version_id, self._graph_id, doc_family_id, source_id,
                    content_hash, blob_rel, title, None, len(document_content.encode("utf-8")),
                    now_iso, _json_dumps({"document_path": document_path, "doc_hash": doc_hash}),
                ),
            )
            existing_doc_version = conn.execute(
                "SELECT 1 FROM concept_version WHERE version_id = ? AND graph_id = ?",
                (doc_version_id, self._graph_id),
            ).fetchone()
            if not existing_doc_version:
                self._append_version(
                    doc_family_id, ROLE_DOCUMENT, doc_version_id, title, document_content,
                    event_time=getattr(cache, "event_time", None),
                    processed_time=getattr(cache, "processed_time", None) or _now(),
                    document_version_id=doc_version_id,
                    source_document=title,
                    metadata={"content_hash": content_hash, "blob_path": blob_rel},
                )

            ep_family_id = _stable_id("epfam", f"{doc_family_id}:{doc_hash}:{cache.absolute_id}")
            ep_version_id = cache.absolute_id
            self._upsert_family(ep_family_id, ROLE_EPISODE, title, {"document_family_id": doc_family_id})
            self._append_version(
                ep_family_id, ROLE_EPISODE, ep_version_id, title,
                getattr(cache, "content", "") or episode_content,
                event_time=getattr(cache, "event_time", None),
                processed_time=getattr(cache, "processed_time", None) or _now(),
                episode_version_id=ep_version_id,
                document_version_id=doc_version_id,
                source_document=title,
                metadata={
                    "source_text": episode_content,
                    "doc_hash": doc_hash,
                    "document_path": document_path,
                    "document_family_id": doc_family_id,
                    "heading_path": "",
                    "start_offset": max(0, int(start_offset or 0)),
                    "end_offset": max(0, int(end_offset if end_offset is not None else start_offset + len(episode_content))),
                    "chunk_index": 0,
                    "chunk_hash": doc_hash,
                    "activity_type": getattr(cache, "activity_type", None),
                    "episode_type": getattr(cache, "episode_type", None),
                },
            )
            self._create_edge(
                EDGE_HAS_EPISODE,
                source_family_id=doc_family_id,
                source_version_id=doc_version_id,
                target_family_id=ep_family_id,
                target_version_id=ep_version_id,
                episode_version_id=ep_version_id,
                document_version_id=doc_version_id,
            )
            artifact_dir = self.artifacts_dir / ep_version_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "source.md").write_text(episode_content, encoding="utf-8")
            (artifact_dir / "prompt_context.md").write_text(getattr(cache, "content", "") or "", encoding="utf-8")
            conn.commit()
        return doc_hash

    def find_cache_and_extraction_by_doc_hash(self, doc_hash: str, document_path: str = "") -> Tuple[Optional[Episode], Optional[tuple]]:
        ep = self.find_cache_by_doc_hash(doc_hash, document_path=document_path)
        extraction = self.load_extraction_result(doc_hash, document_path=document_path)
        return ep, extraction

    def find_cache_by_doc_hash(self, doc_hash: str, document_path: str = "") -> Optional[Episode]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM concept_version WHERE role = ? AND graph_id = ? ORDER BY processed_time DESC",
            (ROLE_EPISODE, self._graph_id),
        ).fetchall()
        requested_path = str(document_path or "")
        for row in rows:
            meta = _json_loads(row["metadata"], {})
            if requested_path and str(meta.get("document_path") or "") != requested_path:
                continue
            if meta.get("doc_hash") == doc_hash or meta.get("chunk_hash") == doc_hash:
                return self._row_to_episode(row)
        return None

    def load_extraction_result(self, doc_hash: str, document_path: str = "") -> Optional[tuple]:
        ep = self.find_cache_by_doc_hash(doc_hash, document_path=document_path)
        if not ep:
            return None
        path = self.artifacts_dir / ep.absolute_id / "extraction.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("entities", []), data.get("relations", [])
        except Exception:
            return None

    def save_extraction_result(self, doc_hash: str, entities: list, relations: list, document_path: str = "") -> bool:
        ep = self.find_cache_by_doc_hash(doc_hash, document_path=document_path)
        if not ep:
            return False
        path = self.artifacts_dir / ep.absolute_id
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "entities": [
                {"absolute_id": e.absolute_id, "family_id": e.family_id, "name": e.name, "content": e.content}
                if hasattr(e, "absolute_id") else e
                for e in entities
            ],
            "relations": [
                {"absolute_id": r.absolute_id, "family_id": r.family_id, "content": r.content}
                if hasattr(r, "absolute_id") else r
                for r in relations
            ],
        }
        (path / "extraction.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True

    def _row_to_episode(self, row) -> Episode:
        meta = _json_loads(row["metadata"], {})
        return Episode(
            absolute_id=row["version_id"],
            content=row["content"] or "",
            event_time=_parse_dt(row["event_time"]) or _now(),
            processed_time=_parse_dt(row["processed_time"]),
            source_document=row["source_document"] or "",
            activity_type=meta.get("activity_type"),
            episode_type=meta.get("episode_type"),
        )

    def load_episode(self, cache_id: str) -> Optional[Episode]:
        row = self._connect().execute(
            "SELECT * FROM concept_version WHERE version_id = ? AND role = ? AND graph_id = ?",
            (cache_id, ROLE_EPISODE, self._graph_id),
        ).fetchone()
        return self._row_to_episode(row) if row else None

    def _episode_source_text_for_evidence(self, episode_id: str) -> Tuple[str, int]:
        row = self._connect().execute(
            "SELECT content, metadata FROM concept_version WHERE version_id = ? AND role = ? AND graph_id = ?",
            (episode_id, ROLE_EPISODE, self._graph_id),
        ).fetchone()
        if not row:
            return "", 0
        meta = _json_loads(row["metadata"], {})
        text = row["content"] or meta.get("source_text") or ""
        base_offset = max(0, int(meta.get("start_offset") or 0))
        if text.startswith("[文档元数据]"):
            marker = "\n\n"
            marker_idx = text.find(marker)
            if marker_idx >= 0:
                text = text[marker_idx + len(marker):]
        return text, base_offset

    def _mention_provenance(self, episode_id: str, concept_row, *, context: str = "", target_type: str = "") -> dict:
        text, base_offset = self._episode_source_text_for_evidence(episode_id)
        candidates = []
        name = (concept_row["name"] or "").strip()
        if name:
            candidates.append(name)
        family = self._connect().execute(
            "SELECT canonical_name FROM concept_family WHERE family_id = ? AND graph_id = ?",
            (concept_row["family_id"], self._graph_id),
        ).fetchone()
        canonical = (family["canonical_name"] if family else "") or ""
        if canonical and canonical not in candidates:
            candidates.append(canonical)
        attrs = _json_loads(concept_row["attributes"], {})
        if isinstance(attrs, dict):
            aliases = attrs.get("aliases") or attrs.get("alias") or attrs.get("variants") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            if isinstance(aliases, list):
                candidates.extend(str(alias) for alias in aliases if str(alias).strip())
        evidence = find_text_evidence(text, candidates, base_offset=base_offset, limit=3)
        provenance = {"context": context, "target_type": target_type}
        if evidence:
            provenance.update({
                "evidence": evidence,
                "evidence_count": len(evidence),
                "source_sentence": evidence[0]["sentence"],
                "source_quote": evidence[0]["quote"],
                "match_type": evidence[0]["match_type"],
                "confidence": evidence[0]["confidence"],
            })
        return provenance

    def get_latest_episode_metadata(self, activity_type: Optional[str] = None) -> Optional[Dict]:
        rows = self._connect().execute(
            "SELECT * FROM concept_version WHERE role = ? AND graph_id = ? ORDER BY processed_time DESC LIMIT 50",
            (ROLE_EPISODE, self._graph_id),
        ).fetchall()
        for row in rows:
            meta = _json_loads(row["metadata"], {})
            if activity_type and meta.get("activity_type") != activity_type:
                continue
            return {
                "absolute_id": row["version_id"],
                "event_time": row["event_time"],
                "processed_time": row["processed_time"],
                "source_document": row["source_document"],
                **meta,
            }
        return None

    def save_episode_mentions(self, episode_id: str, entity_absolute_ids: List[str], context: str = "", target_type: str = "entity"):
        if not entity_absolute_ids:
            return
        role = ROLE_ENTITY if target_type == "entity" else ROLE_RELATION
        with self._write_lock:
            conn = self._connect()
            for vid in set(entity_absolute_ids):
                row = conn.execute(
                    "SELECT * FROM concept_version WHERE version_id = ? AND graph_id = ?",
                    (vid, self._graph_id),
                ).fetchone()
                if not row:
                    continue
                provenance = self._mention_provenance(episode_id, row, context=context, target_type=role)
                self._create_edge(
                    EDGE_MENTIONS,
                    source_version_id=episode_id,
                    target_family_id=row["family_id"],
                    target_version_id=vid,
                    episode_version_id=episode_id,
                    document_version_id=row["document_version_id"] or "",
                    provenance=provenance,
                )
            conn.commit()

    # ------------------------------------------------------------------
    # Entity adapter
    # ------------------------------------------------------------------

    def _row_to_entity(self, row) -> Entity:
        meta = _json_loads(row["metadata"], {})
        return Entity(
            absolute_id=row["version_id"],
            family_id=row["family_id"],
            name=row["name"] or "",
            content=row["content"] or "",
            event_time=_parse_dt(row["event_time"]) or _now(),
            processed_time=_parse_dt(row["processed_time"]) or _now(),
            episode_id=row["episode_version_id"] or "",
            source_document=row["source_document"] or "",
            embedding=row["embedding"],
            valid_at=_parse_dt(row["valid_at"]),
            version_seq=row["version_seq"] or 1,
            summary=row["summary"],
            attributes=row["attributes"],
            confidence=float(row["confidence"]) if row["confidence"] is not None else None,
            content_format=row["content_format"] or "markdown",
            community_id=meta.get("community_id"),
        )

    def save_entity(self, entity: Entity, _precomputed_embedding=None):
        with self._write_lock:
            conn = self._connect()
            self._upsert_family(entity.family_id, ROLE_ENTITY, entity.name)
            existing = self._episode_version_row(entity.family_id, ROLE_ENTITY, entity.episode_id)
            if existing:
                entity.absolute_id = existing["version_id"]
                entity.version_seq = existing["version_seq"] or 1
            else:
                document_version_id = self._document_version_for_episode(entity.episode_id)
                version_seq = self._append_version(
                    entity.family_id, ROLE_ENTITY, entity.absolute_id,
                    entity.name, entity.content,
                    event_time=entity.event_time,
                    processed_time=entity.processed_time or _now(),
                    episode_version_id=entity.episode_id,
                    document_version_id=document_version_id,
                    source_document=entity.source_document,
                    summary=entity.summary,
                    attributes=entity.attributes,
                    confidence=entity.confidence,
                    content_format=getattr(entity, "content_format", "markdown"),
                    embedding=_precomputed_embedding or entity.embedding,
                    metadata={"community_id": getattr(entity, "community_id", None)},
                )
                entity.version_seq = version_seq
            self._cache_entity_name(entity.absolute_id, entity.name)
            conn.commit()

    def bulk_save_entities(self, entities: List[Entity]):
        for entity in entities or []:
            self.save_entity(entity)

    def bulk_save_entities_with_embedding(self, entities: List[Entity]):
        self.bulk_save_entities(entities)

    def get_entity_by_family_id(self, family_id: str) -> Optional[Entity]:
        row = self._latest_version_row(family_id, role=ROLE_ENTITY)
        return self._row_to_entity(row) if row else None

    def get_entities_by_family_ids(self, family_ids: List[str]) -> Dict[str, Entity]:
        result: Dict[str, Entity] = {}
        for original in family_ids or []:
            fid = self._resolve_redirect(original)
            ent = self.get_entity_by_family_id(fid)
            if ent:
                result[fid] = ent
                result[original] = ent
        return result

    def get_entities_by_absolute_ids(self, absolute_ids: List[str]) -> List[Entity]:
        if not absolute_ids:
            return []
        ph = ",".join("?" for _ in absolute_ids)
        rows = self._connect().execute(
            f"SELECT * FROM concept_version WHERE version_id IN ({ph}) AND role = ? AND graph_id = ?",
            absolute_ids + [ROLE_ENTITY, self._graph_id],
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def get_entity_versions(self, family_id: str) -> List[Entity]:
        fid = self._resolve_redirect(family_id)
        rows = self._connect().execute(
            "SELECT * FROM concept_version WHERE family_id = ? AND role = ? AND graph_id = ? ORDER BY version_seq ASC",
            (fid, ROLE_ENTITY, self._graph_id),
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def get_entity_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
        return self._version_counts(family_ids, ROLE_ENTITY)

    def get_entity_version_count(self, family_id: str) -> int:
        return self.get_entity_version_counts([family_id]).get(family_id, 0)

    def _version_counts(self, family_ids: List[str], role: str) -> Dict[str, int]:
        result = {fid: 0 for fid in family_ids or []}
        if not family_ids:
            return result
        resolved = [self._resolve_redirect(fid) for fid in family_ids if fid]
        if not resolved:
            return result
        ph = ",".join("?" for _ in resolved)
        rows = self._connect().execute(
            f"SELECT family_id, COUNT(*) AS cnt FROM concept_version WHERE family_id IN ({ph}) AND role = ? AND graph_id = ? GROUP BY family_id",
            resolved + [role, self._graph_id],
        ).fetchall()
        counts = {r["family_id"]: int(r["cnt"]) for r in rows}
        for original in result:
            result[original] = counts.get(self._resolve_redirect(original), 0)
        return result

    def get_family_ids_by_names(self, names: List[str]) -> Dict[str, str]:
        result = {}
        for name in names or []:
            row = self._connect().execute(
                "SELECT family_id FROM concept_family WHERE role = ? AND canonical_name = ? AND graph_id = ? ORDER BY updated_at DESC LIMIT 1",
                (ROLE_ENTITY, name, self._graph_id),
            ).fetchone()
            if row:
                result[name] = row["family_id"]
        return result

    def get_entity_names_by_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, str]:
        return {e.absolute_id: e.name for e in self.get_entities_by_absolute_ids(absolute_ids)}

    def get_family_ids_by_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, str]:
        return {e.absolute_id: e.family_id for e in self.get_entities_by_absolute_ids(absolute_ids)}

    def get_latest_absolute_ids_by_family_ids(self, family_ids: List[str]) -> Dict[str, str]:
        return {fid: ent.absolute_id for fid, ent in self.get_entities_by_family_ids(family_ids).items()}

    def get_latest_entities_projection(self, content_snippet_length: Optional[int] = None) -> List[Dict[str, Any]]:
        rows = self._latest_rows_by_role(ROLE_ENTITY)
        counts = self.get_entity_version_counts([r["family_id"] for r in rows])
        projections = []
        for r in rows:
            ent = self._row_to_entity(r)
            content = ent.content or ""
            projections.append({
                "family_id": ent.family_id,
                "name": ent.name,
                "content": content,
                "content_snippet": content[: content_snippet_length or self.entity_content_snippet_length],
                "source_document": ent.source_document,
                "version_count": counts.get(ent.family_id, 1),
                "entity": ent,
                "embedding": ent.embedding,
            })
        return projections

    def get_all_entities(self, limit: int = 100, offset: Optional[int] = None, exclude_embedding: bool = False) -> List[Entity]:
        rows = self._latest_rows_by_role(ROLE_ENTITY)
        start = int(offset or 0)
        selected = rows[start: start + int(limit or 100)]
        entities = [self._row_to_entity(r) for r in selected]
        if exclude_embedding:
            for entity in entities:
                entity.embedding = None
        return entities

    def get_all_entities_before_time(self, time_point, limit: int = 100, exclude_embedding: bool = False) -> List[Entity]:
        cutoff = _fmt_dt(time_point)
        if not cutoff:
            return self.get_all_entities(limit=limit, exclude_embedding=exclude_embedding)
        rows = self._connect().execute(
            """
            SELECT v.* FROM concept_version v
            INNER JOIN (
              SELECT family_id, MAX(version_seq) AS max_seq
              FROM concept_version
              WHERE role = ? AND graph_id = ? AND processed_time <= ?
              GROUP BY family_id
            ) latest ON latest.family_id = v.family_id AND latest.max_seq = v.version_seq
            WHERE v.role = ? AND v.graph_id = ?
            ORDER BY v.processed_time DESC
            LIMIT ?
            """,
            (ROLE_ENTITY, self._graph_id, cutoff, ROLE_ENTITY, self._graph_id, int(limit or 100)),
        ).fetchall()
        entities = [self._row_to_entity(r) for r in rows]
        if exclude_embedding:
            for entity in entities:
                entity.embedding = None
        return entities

    def _latest_rows_by_role(self, role: str) -> List[sqlite3.Row]:
        return self._connect().execute(
            """
            SELECT v.* FROM concept_version v
            INNER JOIN (
              SELECT family_id, MAX(version_seq) AS max_seq
              FROM concept_version
              WHERE role = ? AND graph_id = ?
              GROUP BY family_id
            ) latest ON latest.family_id = v.family_id AND latest.max_seq = v.version_seq
            WHERE v.role = ? AND v.graph_id = ?
            ORDER BY v.processed_time DESC
            """,
            (role, self._graph_id, role, self._graph_id),
        ).fetchall()

    # ------------------------------------------------------------------
    # Relation adapter
    # ------------------------------------------------------------------

    def _row_to_relation(self, row) -> Relation:
        meta = _json_loads(row["metadata"], {})
        return Relation(
            absolute_id=row["version_id"],
            family_id=row["family_id"],
            entity1_absolute_id=meta.get("entity1_absolute_id", ""),
            entity2_absolute_id=meta.get("entity2_absolute_id", ""),
            content=row["content"] or "",
            event_time=_parse_dt(row["event_time"]) or _now(),
            processed_time=_parse_dt(row["processed_time"]) or _now(),
            episode_id=row["episode_version_id"] or "",
            source_document=row["source_document"] or "",
            entity1_family_id=meta.get("entity1_family_id", ""),
            entity2_family_id=meta.get("entity2_family_id", ""),
            embedding=row["embedding"],
            valid_at=_parse_dt(row["valid_at"]),
            version_seq=row["version_seq"] or 1,
            summary=row["summary"],
            attributes=row["attributes"],
            confidence=float(row["confidence"]) if row["confidence"] is not None else None,
            provenance=meta.get("provenance"),
            content_format=row["content_format"] or "markdown",
        )

    def _entity_family_for_version(self, version_id: str) -> str:
        row = self._connect().execute(
            "SELECT family_id FROM concept_version WHERE version_id = ? AND role = ? AND graph_id = ?",
            (version_id, ROLE_ENTITY, self._graph_id),
        ).fetchone()
        return row["family_id"] if row else ""

    def save_relation(self, relation: Relation):
        with self._write_lock:
            conn = self._connect()
            e1_fid = getattr(relation, "entity1_family_id", "") or self._entity_family_for_version(relation.entity1_absolute_id)
            e2_fid = getattr(relation, "entity2_family_id", "") or self._entity_family_for_version(relation.entity2_absolute_id)
            document_version_id = self._document_version_for_episode(relation.episode_id)
            self._upsert_family(relation.family_id, ROLE_RELATION, relation.summary or relation.content[:80])
            existing = self._episode_version_row(relation.family_id, ROLE_RELATION, relation.episode_id)
            if existing:
                relation.absolute_id = existing["version_id"]
                relation.version_seq = existing["version_seq"] or 1
                relation.entity1_family_id = e1_fid
                relation.entity2_family_id = e2_fid
                conn.commit()
                return
            metadata = {
                "entity1_absolute_id": relation.entity1_absolute_id,
                "entity2_absolute_id": relation.entity2_absolute_id,
                "entity1_family_id": e1_fid,
                "entity2_family_id": e2_fid,
                "provenance": relation.provenance,
            }
            version_seq = self._append_version(
                relation.family_id, ROLE_RELATION, relation.absolute_id,
                "", relation.content,
                event_time=relation.event_time,
                processed_time=relation.processed_time or _now(),
                episode_version_id=relation.episode_id,
                document_version_id=document_version_id,
                source_document=relation.source_document,
                summary=relation.summary,
                attributes=relation.attributes,
                confidence=relation.confidence,
                content_format=getattr(relation, "content_format", "markdown"),
                embedding=relation.embedding,
                metadata=metadata,
            )
            relation.version_seq = version_seq
            relation.entity1_family_id = e1_fid
            relation.entity2_family_id = e2_fid
            if relation.episode_id:
                self._create_edge(
                    EDGE_ASSERTS,
                    source_version_id=relation.episode_id,
                    target_family_id=relation.family_id,
                    target_version_id=relation.absolute_id,
                    relation_family_id=relation.family_id,
                    relation_version_id=relation.absolute_id,
                    episode_version_id=relation.episode_id,
                    document_version_id=document_version_id,
                    confidence=relation.confidence,
                )
            for target_fid, target_vid in ((e1_fid, relation.entity1_absolute_id), (e2_fid, relation.entity2_absolute_id)):
                if target_fid:
                    self._create_edge(
                        EDGE_CONNECTS,
                        source_family_id=relation.family_id,
                        source_version_id=relation.absolute_id,
                        target_family_id=target_fid,
                        target_version_id=target_vid,
                        relation_family_id=relation.family_id,
                        relation_version_id=relation.absolute_id,
                        episode_version_id=relation.episode_id,
                        document_version_id=document_version_id,
                        confidence=relation.confidence,
                    )
            conn.commit()

    def bulk_save_relations(self, relations: List[Relation]):
        for relation in relations or []:
            self.save_relation(relation)

    def bulk_save_relations_with_embedding(self, relations: List[Relation]):
        self.bulk_save_relations(relations)

    def get_relation_by_absolute_id(self, absolute_id: str) -> Optional[Relation]:
        row = self._connect().execute(
            "SELECT * FROM concept_version WHERE version_id = ? AND role = ? AND graph_id = ?",
            (absolute_id, ROLE_RELATION, self._graph_id),
        ).fetchone()
        return self._row_to_relation(row) if row else None

    def get_relation_by_family_id(self, family_id: str) -> Optional[Relation]:
        row = self._latest_version_row(family_id, role=ROLE_RELATION)
        return self._row_to_relation(row) if row else None

    def get_relation_versions(self, family_id: str) -> List[Relation]:
        fid = self._resolve_redirect(family_id)
        rows = self._connect().execute(
            "SELECT * FROM concept_version WHERE family_id = ? AND role = ? AND graph_id = ? ORDER BY version_seq ASC",
            (fid, ROLE_RELATION, self._graph_id),
        ).fetchall()
        return [self._row_to_relation(r) for r in rows]

    def get_relation_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
        return self._version_counts(family_ids, ROLE_RELATION)

    def _relation_fids_for_pair(self, e1_fid: str, e2_fid: str) -> List[str]:
        rows = self._connect().execute(
            """
            SELECT source_family_id
            FROM concept_edge
            WHERE graph_id = ? AND edge_type = ? AND target_family_id IN (?, ?)
            GROUP BY source_family_id
            HAVING COUNT(DISTINCT target_family_id) = 2
            """,
            (self._graph_id, EDGE_CONNECTS, e1_fid, e2_fid),
        ).fetchall()
        return [r["source_family_id"] for r in rows if r["source_family_id"]]

    def get_relations_by_entities(self, from_family_id: str, to_family_id: str, include_candidates: bool = False) -> List[Relation]:
        f1, f2 = _sort_pair(self._resolve_redirect(from_family_id), self._resolve_redirect(to_family_id))
        return [r for fid in self._relation_fids_for_pair(f1, f2) if (r := self.get_relation_by_family_id(fid))]

    def get_relations_by_entity_pairs(self, entity_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], List[Relation]]:
        result: Dict[Tuple[str, str], List[Relation]] = {}
        for a, b in entity_pairs or []:
            key = _sort_pair(self._resolve_redirect(a), self._resolve_redirect(b))
            result[key] = self.get_relations_by_entities(key[0], key[1])
        return result

    def get_relations_by_family_ids(self, family_ids: List[str], limit: int = 100, time_point: str = None, include_candidates: bool = False) -> List[Relation]:
        fids = {self._resolve_redirect(fid) for fid in family_ids or [] if fid}
        if not fids:
            return []
        ph = ",".join("?" for _ in fids)
        rows = self._connect().execute(
            f"SELECT DISTINCT source_family_id FROM concept_edge WHERE graph_id = ? AND edge_type = ? AND target_family_id IN ({ph}) LIMIT ?",
            [self._graph_id, EDGE_CONNECTS] + list(fids) + [limit],
        ).fetchall()
        rels = [self.get_relation_by_family_id(r["source_family_id"]) for r in rows]
        return [r for r in rels if r is not None][:limit]

    def get_entity_relations_by_family_id(self, family_id: str, limit: int = 100, **_ignored) -> List[Relation]:
        return self.get_relations_by_family_ids([family_id], limit=limit)

    def count_entity_relations_by_family_ids(self, family_ids: List[str]) -> Dict[str, int]:
        return {fid: len(self.get_entity_relations_by_family_id(fid, limit=10_000)) for fid in family_ids or []}

    def get_relation_embeddings(self, family_ids: List[str]) -> Dict[str, Any]:
        result = {}
        for fid in family_ids or []:
            rel = self.get_relation_by_family_id(fid)
            if rel and rel.embedding:
                result[fid] = np.frombuffer(rel.embedding, dtype=np.float32)
        return result

    # ------------------------------------------------------------------
    # Search, concepts, traversal
    # ------------------------------------------------------------------

    def _like_search(self, query: str, role: str, limit: int) -> List[sqlite3.Row]:
        q = f"%{query}%"
        return self._connect().execute(
            """
            SELECT v.* FROM concept_version v
            INNER JOIN (
              SELECT family_id, MAX(version_seq) AS max_seq
              FROM concept_version
              WHERE role = ? AND graph_id = ?
              GROUP BY family_id
            ) latest ON latest.family_id = v.family_id AND latest.max_seq = v.version_seq
            WHERE v.role = ? AND v.graph_id = ? AND (v.name LIKE ? OR v.content LIKE ?)
            ORDER BY v.processed_time DESC
            LIMIT ?
            """,
            (role, self._graph_id, role, self._graph_id, q, q, limit),
        ).fetchall()

    def search_entities_by_bm25(self, query: str, limit: int = 20, time_point: str = None) -> List[Entity]:
        return [self._row_to_entity(r) for r in self._like_search(query, ROLE_ENTITY, limit)]

    def search_relations_by_bm25(self, query: str, limit: int = 20, time_point: str = None) -> List[Relation]:
        return [self._row_to_relation(r) for r in self._like_search(query, ROLE_RELATION, limit)]

    def _similarity_search(self, query_text: str, role: str, threshold: float, max_results: int):
        if not self.embedding_client or not self.embedding_client.is_available():
            return self._like_search(query_text, role, max_results)
        encoded = _encode_and_normalize(self.embedding_client, query_text)
        if not encoded:
            return self._like_search(query_text, role, max_results)
        _, q = encoded
        scored = []
        for row in self._latest_rows_by_role(role):
            emb = row["embedding"]
            if not emb:
                continue
            arr = np.frombuffer(emb, dtype=np.float32)
            if arr.size != q.size:
                continue
            score = float(np.dot(q, arr))
            if score >= threshold:
                scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:max_results]]

    def search_entities_by_similarity(self, query_text: str, threshold: float = 0.5, max_results: int = 20) -> List[Entity]:
        return [self._row_to_entity(r) for r in self._similarity_search(query_text, ROLE_ENTITY, threshold, max_results)]

    def search_relations_by_similarity(self, query_text: str, threshold: float = 0.5, max_results: int = 20) -> List[Relation]:
        return [self._row_to_relation(r) for r in self._similarity_search(query_text, ROLE_RELATION, threshold, max_results)]

    def search_concepts_by_bm25(self, query: str, role: str = None, limit: int = 20, time_point: str = None) -> List[dict]:
        roles = [role] if role else [ROLE_DOCUMENT, ROLE_EPISODE, ROLE_ENTITY, ROLE_RELATION]
        out = []
        for r in roles:
            for row in self._like_search(query, r, limit):
                out.append(self._concept_dict(row))
        return out[:limit]

    def search_concepts_by_similarity(self, query_text: str, role: str = None, threshold: float = 0.5, max_results: int = 20, time_point: str = None) -> List[dict]:
        roles = [role] if role else [ROLE_ENTITY, ROLE_RELATION, ROLE_DOCUMENT, ROLE_EPISODE]
        out = []
        for r in roles:
            for row in self._similarity_search(query_text, r, threshold, max_results):
                out.append(self._concept_dict(row))
        return out[:max_results]

    def _concept_dict(self, row) -> dict:
        meta = _json_loads(row["metadata"], {})
        return {
            "id": row["version_id"],
            "version_id": row["version_id"],
            "family_id": row["family_id"],
            "role": row["role"],
            "name": row["name"] or "",
            "content": row["content"] or "",
            "summary": row["summary"],
            "confidence": row["confidence"],
            "version_seq": row["version_seq"],
            "content_changed": bool(row["content_changed"]),
            "event_time": row["event_time"],
            "processed_time": row["processed_time"],
            "episode_version_id": row["episode_version_id"],
            "document_version_id": row["document_version_id"],
            "source_document": row["source_document"],
            "metadata": meta,
        }

    def get_concept_by_family_id(self, family_id: str, time_point: str = None) -> Optional[dict]:
        if time_point:
            fid = self._resolve_redirect(family_id)
            row = self._connect().execute(
                """
                SELECT * FROM concept_version
                WHERE family_id = ? AND graph_id = ? AND processed_time <= ?
                ORDER BY version_seq DESC
                LIMIT 1
                """,
                (fid, self._graph_id, _fmt_dt(time_point)),
            ).fetchone()
        else:
            row = self._latest_version_row(family_id)
        return self._concept_dict(row) if row else None

    def list_concepts(self, role: str = None, limit: int = 50, offset: int = 0, time_point: str = None) -> List[dict]:
        role_clause = ""
        time_clause_inner = ""
        params: List[Any] = [self._graph_id]
        if time_point:
            time_clause_inner = "AND processed_time <= ?"
            params.append(_fmt_dt(time_point))
        if role:
            role_clause = "AND v.role = ?"
        params.extend([self._graph_id])
        if role:
            params.append(role)
        params.extend([limit, offset])
        rows = self._connect().execute(
            f"""
            SELECT v.* FROM concept_version v
            INNER JOIN (
              SELECT family_id, MAX(version_seq) AS max_seq
              FROM concept_version
              WHERE graph_id = ? {time_clause_inner}
              GROUP BY family_id
            ) latest ON latest.family_id = v.family_id AND latest.max_seq = v.version_seq
            WHERE v.graph_id = ? {role_clause}
            ORDER BY v.processed_time DESC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [self._concept_dict(r) for r in rows]

    def count_concepts(self, role: str = None, time_point: str = None) -> int:
        if role:
            row = self._connect().execute(
                "SELECT COUNT(*) AS cnt FROM concept_family WHERE role = ? AND graph_id = ?",
                (role, self._graph_id),
            ).fetchone()
        else:
            row = self._connect().execute(
                "SELECT COUNT(*) AS cnt FROM concept_family WHERE graph_id = ?",
                (self._graph_id,),
            ).fetchone()
        return int(row["cnt"] or 0)

    def get_concept_versions(self, family_id: str) -> List[dict]:
        fid = self._resolve_redirect(family_id)
        rows = self._connect().execute(
            "SELECT * FROM concept_version WHERE family_id = ? AND graph_id = ? ORDER BY version_seq ASC",
            (fid, self._graph_id),
        ).fetchall()
        return [self._concept_dict(r) for r in rows]

    def get_concept_provenance(self, family_id: str, time_point: str = None) -> List[dict]:
        fid = self._resolve_redirect(family_id)
        rows = self._connect().execute(
            """
            SELECT e.*, v.content AS episode_content, v.source_document AS episode_source,
                   v.metadata AS episode_metadata
            FROM concept_edge e
            LEFT JOIN concept_version v
              ON v.version_id = e.episode_version_id AND v.graph_id = e.graph_id
            WHERE e.graph_id = ?
              AND e.edge_type IN (?, ?, ?)
              AND (e.target_family_id = ? OR e.relation_family_id = ? OR e.source_family_id = ?)
            ORDER BY e.created_at DESC
            """,
            (self._graph_id, EDGE_MENTIONS, EDGE_ASSERTS, EDGE_CONNECTS, fid, fid, fid),
        ).fetchall()
        out = []
        for r in rows:
            episode_meta = _json_loads(r["episode_metadata"], {})
            out.append({
                "edge_type": r["edge_type"],
                "episode_id": r["episode_version_id"],
                "document_version_id": r["document_version_id"],
                "content": r["episode_content"] or "",
                "source_document": r["episode_source"] or "",
                "source_span": {
                    "heading_path": episode_meta.get("heading_path", ""),
                    "start_offset": episode_meta.get("start_offset"),
                    "end_offset": episode_meta.get("end_offset"),
                    "chunk_index": episode_meta.get("chunk_index"),
                    "chunk_hash": episode_meta.get("chunk_hash"),
                },
                "provenance": _json_loads(r["provenance"], {}),
            })
        return out

    def get_concept_mentions(self, family_id: str, time_point: str = None) -> List[dict]:
        return self.get_concept_provenance(family_id, time_point=time_point)

    def get_concept_neighbors(self, family_id: str, max_depth: int = 1, time_point: str = None, edge_types: Optional[List[str]] = None) -> List[dict]:
        fid = self._resolve_redirect(family_id)
        filters = ["graph_id = ?", "(source_family_id = ? OR target_family_id = ? OR relation_family_id = ?)"]
        params: List[Any] = [self._graph_id, fid, fid, fid]
        if edge_types:
            ph = ",".join("?" for _ in edge_types)
            filters.append(f"edge_type IN ({ph})")
            params.extend(edge_types)
        if time_point:
            filters.append("created_at <= ?")
            params.append(_fmt_dt(time_point))
        rows = self._connect().execute(
            f"SELECT * FROM concept_edge WHERE {' AND '.join(filters)} LIMIT 200",
            params,
        ).fetchall()
        nids = set()
        for r in rows:
            for col in ("source_family_id", "target_family_id", "relation_family_id"):
                val = r[col]
                if val and val != fid:
                    nids.add(val)
        return [c for nid in nids if (c := self.get_concept_by_family_id(nid, time_point=time_point))]

    def traverse_concepts(self, start_family_ids: List[str], max_depth: int = 2, time_point: str = None, edge_types: Optional[List[str]] = None) -> dict:
        visited = set()
        frontier = [self._resolve_redirect(fid) for fid in start_family_ids or []]
        concepts = {}
        edges = []
        for depth in range(max_depth + 1):
            next_frontier = []
            for fid in frontier:
                if not fid or fid in visited:
                    continue
                visited.add(fid)
                concept = self.get_concept_by_family_id(fid)
                if concept:
                    concepts[fid] = concept
                for n in self.get_concept_neighbors(fid, time_point=time_point, edge_types=edge_types):
                    nfid = n.get("family_id")
                    edges.append({"from": fid, "to": nfid, "to_role": n.get("role"), "to_name": n.get("name")})
                    if nfid not in visited:
                        next_frontier.append(nfid)
            frontier = next_frontier
        return {"concepts": concepts, "edges": edges, "relations": [c for c in concepts.values() if c.get("role") == ROLE_RELATION], "visited_count": len(visited)}

    def get_episode_concepts(self, episode_id: str) -> List[dict]:
        rows = self._connect().execute(
            "SELECT target_family_id FROM concept_edge WHERE graph_id = ? AND edge_type = ? AND episode_version_id = ?",
            (self._graph_id, EDGE_MENTIONS, episode_id),
        ).fetchall()
        return [c for r in rows if (c := self.get_concept_by_family_id(r["target_family_id"]))]

    # ------------------------------------------------------------------
    # Documents / vault indexing
    # ------------------------------------------------------------------

    def index_markdown_file(self, path: str, vault_root: str = "", force: bool = False) -> dict:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        rel = str(p.relative_to(vault_root)) if vault_root else p.name
        parsed = self.parse_markdown(text)
        doc_hash = _sha256_text(text)
        doc_family_id = _stable_id("docfam", str(p.resolve()))
        existing = self._connect().execute(
            "SELECT 1 FROM document_version WHERE graph_id = ? AND document_family_id = ? AND content_hash = ? LIMIT 1",
            (self._graph_id, doc_family_id, doc_hash),
        ).fetchone()
        if existing and not force:
            return {"document_family_id": doc_family_id, "content_hash": doc_hash, "skipped": True}
        with self._write_lock:
            doc_family_id, doc_version_id, content_hash, _blob_rel = self._save_document_snapshot(
                str(p.resolve()),
                parsed["title"] or p.name,
                text,
                absolute_path=str(p.resolve()),
                relative_path=rel,
                frontmatter=parsed["frontmatter"],
                tags=parsed["tags"],
                aliases=parsed["aliases"],
                mtime=datetime.fromtimestamp(p.stat().st_mtime),
                metadata={"vault_root": str(vault_root or ""), "relative_path": rel},
            )
            episodes = []
            for idx, section in enumerate(self.split_markdown_episodes(text)):
                ep_id = self._save_episode_chunk(
                    document_family_id=doc_family_id,
                    document_version_id=doc_version_id,
                    title=parsed["title"] or p.name,
                    chunk=section["content"],
                    heading_path=section["heading_path"],
                    start_offset=section["start_offset"],
                    end_offset=section["end_offset"],
                    chunk_index=idx,
                    source_document=p.name,
                )
                episodes.append({"episode_version_id": ep_id, **section})
            for link in parsed["links"]:
                link_path = Path(link)
                if link_path.suffix.lower() != ".md":
                    link_path = Path(f"{link}.md")
                target_family = _stable_id("docfam", str((p.parent / link_path).resolve()))
                self._upsert_family(target_family, ROLE_DOCUMENT, link)
                self._create_edge(
                    EDGE_DOCUMENT_LINK,
                    source_family_id=doc_family_id,
                    source_version_id=doc_version_id,
                    target_family_id=target_family,
                    document_version_id=doc_version_id,
                    provenance={"link": link},
                )
            self._connect().commit()
        return {
            "document_family_id": doc_family_id,
            "document_version_id": doc_version_id,
            "content_hash": content_hash,
            "episodes": len(episodes),
            "skipped": False,
            **parsed,
        }

    def index_vault(self, path: str, force: bool = False) -> dict:
        root = Path(path)
        files = sorted(root.rglob("*.md")) if root.is_dir() else [root]
        results = [self.index_markdown_file(str(p), vault_root=str(root) if root.is_dir() else "", force=force) for p in files]
        return {"path": str(root), "files": len(files), "indexed": sum(1 for r in results if not r.get("skipped")), "skipped": sum(1 for r in results if r.get("skipped")), "results": results}

    @staticmethod
    def parse_markdown(text: str) -> dict:
        frontmatter: Dict[str, Any] = {}
        body = text or ""
        if body.startswith("---"):
            parts = body.split("---", 2)
            if len(parts) >= 3:
                raw = parts[1]
                body = parts[2]
                current_key = None
                for line in raw.splitlines():
                    if not line.strip():
                        continue
                    if line.lstrip().startswith("-") and current_key:
                        frontmatter.setdefault(current_key, []).append(line.split("-", 1)[1].strip())
                    elif ":" in line:
                        k, v = line.split(":", 1)
                        current_key = k.strip()
                        val = v.strip()
                        if val.startswith("[") and val.endswith("]"):
                            frontmatter[current_key] = [x.strip().strip("'\"") for x in val[1:-1].split(",") if x.strip()]
                        elif val:
                            frontmatter[current_key] = val.strip("'\"")
                        else:
                            frontmatter[current_key] = []
        title = ""
        for line in body.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        if not title:
            title = str(frontmatter.get("title") or "")
        import re
        wikilinks = []
        for m in re.finditer(r"!?\[\[([^\]]+)\]\]", text or ""):
            target = m.group(1).split("|", 1)[0].split("#", 1)[0].strip()
            if target:
                wikilinks.append(target)
        md_links = [m.group(1).strip() for m in re.finditer(r"\[[^\]]+\]\(([^)]+\.md)(?:#[^)]+)?\)", text or "")]
        tags = set()
        fm_tags = frontmatter.get("tags", [])
        if isinstance(fm_tags, str):
            tags.add(fm_tags)
        elif isinstance(fm_tags, list):
            tags.update(str(x) for x in fm_tags)
        tags.update(m.group(1) for m in re.finditer(r"(?<!\w)#([\w\-/\u4e00-\u9fff]+)", text or ""))
        aliases = frontmatter.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, list):
            aliases = []
        return {"title": title, "frontmatter": frontmatter, "tags": sorted(tags), "aliases": aliases, "links": sorted(set(wikilinks + md_links))}

    @staticmethod
    def split_markdown_episodes(text: str, window_size: int = 4000, overlap: int = 200) -> List[dict]:
        return split_markdown_chunks(text or "", window_size=window_size, overlap=overlap)

    def list_documents(self, limit: int = 50, offset: int = 0) -> List[dict]:
        rows = self._connect().execute(
            """
            SELECT ds.*, dv.document_version_id, dv.content_hash, dv.title AS version_title, dv.processed_time
            FROM document_source ds
            LEFT JOIN document_version dv ON dv.source_id = ds.source_id AND dv.graph_id = ds.graph_id
            WHERE ds.graph_id = ?
            ORDER BY dv.processed_time DESC
            LIMIT ? OFFSET ?
            """,
            (self._graph_id, limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_document_content(self, document_version_id: str, *, offset: int = 0, limit: int = 20000) -> dict:
        """Return a slice of the original Markdown snapshot for one document version."""
        doc_id = str(document_version_id or "").strip()
        if not doc_id:
            raise ValueError("document_version_id 不能为空")
        offset = max(0, int(offset or 0))
        limit = max(1, min(int(limit or 20000), 200000))
        row = self._connect().execute(
            """
            SELECT dv.document_version_id, dv.document_family_id, dv.content_hash,
                   dv.blob_path, dv.title AS version_title, dv.size, dv.processed_time,
                   ds.title AS source_title, ds.relative_path, ds.absolute_path, ds.uri
            FROM document_version dv
            JOIN document_source ds
              ON ds.source_id = dv.source_id AND ds.graph_id = dv.graph_id
            WHERE dv.graph_id = ? AND dv.document_version_id = ?
            LIMIT 1
            """,
            (self._graph_id, doc_id),
        ).fetchone()
        if not row:
            raise KeyError(f"document_version_id 不存在: {doc_id}")
        blob_rel = row["blob_path"] or ""
        blob_path = (self.storage_path / blob_rel).resolve()
        storage_root = self.storage_path.resolve()
        if storage_root not in blob_path.parents and blob_path != storage_root:
            raise ValueError("文档 blob 路径越界")
        if not blob_path.exists():
            raise FileNotFoundError(f"文档 blob 不存在: {blob_rel}")
        content = blob_path.read_text(encoding="utf-8")
        total_chars = len(content)
        end = min(total_chars, offset + limit)
        return {
            "document_version_id": row["document_version_id"],
            "document_family_id": row["document_family_id"],
            "title": row["version_title"] or row["source_title"] or "",
            "relative_path": row["relative_path"] or "",
            "absolute_path": row["absolute_path"] or "",
            "uri": row["uri"] or "",
            "content_hash": row["content_hash"] or "",
            "blob_path": blob_rel,
            "size": row["size"] or 0,
            "processed_time": row["processed_time"],
            "offset": offset,
            "limit": limit,
            "next_offset": end if end < total_chars else None,
            "total_chars": total_chars,
            "truncated": end < total_chars,
            "content": content[offset:end],
        }

    def _document_graph_docs(self, document_version_ids: Optional[List[str]], document_family_ids: Optional[List[str]]) -> Dict[str, dict]:
        version_ids = [str(x).strip() for x in (document_version_ids or []) if str(x).strip()]
        family_ids = [str(x).strip() for x in (document_family_ids or []) if str(x).strip()]
        if not version_ids and not family_ids:
            raise ValueError("document_version_ids 或 document_family_ids 至少提供一个")

        conn = self._connect()
        docs_by_version: Dict[str, dict] = {}

        def _doc_from_row(row) -> dict:
            metadata = _json_loads(row["metadata"], {})
            return {
                "id": f"doc:{row['document_version_id']}",
                "type": ROLE_DOCUMENT,
                "role": ROLE_DOCUMENT,
                "family_id": row["document_family_id"],
                "version_id": row["document_version_id"],
                "document_version_id": row["document_version_id"],
                "source_id": row["source_id"],
                "title": row["version_title"] or row["source_title"] or "",
                "absolute_path": row["absolute_path"] or "",
                "relative_path": row["relative_path"] or "",
                "uri": row["uri"] or "",
                "content_hash": row["content_hash"] or "",
                "blob_path": row["blob_path"] or "",
                "size": row["size"] or 0,
                "processed_time": row["processed_time"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "frontmatter": _json_loads(row["frontmatter_json"], {}),
                "tags": _json_loads(row["tags_json"], []),
                "aliases": _json_loads(row["aliases_json"], []),
                "metadata": metadata,
            }

        if version_ids:
            ph = ",".join("?" for _ in version_ids)
            rows = conn.execute(
                f"""
                SELECT ds.title AS source_title, ds.absolute_path, ds.relative_path, ds.uri,
                       ds.created_at, ds.updated_at,
                       dv.*, dv.title AS version_title
                FROM document_version dv
                JOIN document_source ds
                  ON ds.source_id = dv.source_id AND ds.graph_id = dv.graph_id
                WHERE dv.graph_id = ? AND dv.document_version_id IN ({ph})
                ORDER BY dv.processed_time DESC
                """,
                [self._graph_id, *version_ids],
            ).fetchall()
            for row in rows:
                docs_by_version[row["document_version_id"]] = _doc_from_row(row)

        if family_ids:
            ph = ",".join("?" for _ in family_ids)
            rows = conn.execute(
                f"""
                SELECT ds.title AS source_title, ds.absolute_path, ds.relative_path, ds.uri,
                       ds.created_at, ds.updated_at,
                       dv.*, dv.title AS version_title
                FROM document_version dv
                JOIN (
                  SELECT document_family_id, MAX(processed_time) AS latest_time
                  FROM document_version
                  WHERE graph_id = ? AND document_family_id IN ({ph})
                  GROUP BY document_family_id
                ) latest
                  ON latest.document_family_id = dv.document_family_id
                 AND latest.latest_time = dv.processed_time
                JOIN document_source ds
                  ON ds.source_id = dv.source_id AND ds.graph_id = dv.graph_id
                WHERE dv.graph_id = ?
                ORDER BY dv.processed_time DESC
                """,
                [self._graph_id, *family_ids, self._graph_id],
            ).fetchall()
            for row in rows:
                docs_by_version[row["document_version_id"]] = _doc_from_row(row)

        return docs_by_version

    @staticmethod
    def _document_graph_edge_dict(row) -> dict:
        return {
            "id": row["edge_id"],
            "edge_id": row["edge_id"],
            "type": row["edge_type"],
            "edge_type": row["edge_type"],
            "source_family_id": row["source_family_id"] or "",
            "source_version_id": row["source_version_id"] or "",
            "target_family_id": row["target_family_id"] or "",
            "target_version_id": row["target_version_id"] or "",
            "relation_family_id": row["relation_family_id"] or "",
            "relation_version_id": row["relation_version_id"] or "",
            "episode_version_id": row["episode_version_id"] or "",
            "document_version_id": row["document_version_id"] or "",
            "weight": row["weight"],
            "confidence": row["confidence"],
            "provenance": _json_loads(row["provenance"], {}),
            "created_at": row["created_at"],
        }

    def _document_graph_episode_dict(self, row) -> dict:
        ep = self._concept_dict(row)
        ep_meta = ep.get("metadata") or {}
        ep.update({
            "id": f"episode:{ep['version_id']}",
            "type": ROLE_EPISODE,
            "heading_path": ep_meta.get("heading_path", ""),
            "start_offset": ep_meta.get("start_offset"),
            "end_offset": ep_meta.get("end_offset"),
            "chunk_index": ep_meta.get("chunk_index"),
            "chunk_hash": ep_meta.get("chunk_hash"),
            "source_span": {
                "heading_path": ep_meta.get("heading_path", ""),
                "start_offset": ep_meta.get("start_offset"),
                "end_offset": ep_meta.get("end_offset"),
                "chunk_index": ep_meta.get("chunk_index"),
                "chunk_hash": ep_meta.get("chunk_hash"),
            },
        })
        return ep

    def get_document_graph_outline(
        self,
        document_version_ids: Optional[List[str]] = None,
        document_family_ids: Optional[List[str]] = None,
        max_episodes: int = 10000,
    ) -> dict:
        """Return the fast Document -> Episode skeleton for progressive graph rendering."""
        docs_by_version = self._document_graph_docs(document_version_ids, document_family_ids)
        if not docs_by_version:
            return {
                "documents": [],
                "episodes": [],
                "concepts": [],
                "edges": [],
                "versions": {},
                "episode_counts": {},
                "cursor": 0,
                "next_cursor": None,
                "counts": {"documents": 0, "episodes": 0, "concepts": 0, "edges": 0, "relations": 0},
            }

        conn = self._connect()
        selected_doc_versions = list(docs_by_version.keys())
        doc_ph = ",".join("?" for _ in selected_doc_versions)
        has_episode_rows = conn.execute(
            f"""
            SELECT e.*, ep.*
            FROM concept_edge e
            JOIN concept_version ep
              ON ep.version_id = e.target_version_id
             AND ep.graph_id = e.graph_id
             AND ep.role = ?
            WHERE e.graph_id = ?
              AND e.edge_type = ?
              AND e.document_version_id IN ({doc_ph})
            ORDER BY e.document_version_id ASC, ep.version_seq ASC, ep.processed_time ASC
            LIMIT ?
            """,
            [ROLE_EPISODE, self._graph_id, EDGE_HAS_EPISODE, *selected_doc_versions, max_episodes],
        ).fetchall()

        episodes_by_version: Dict[str, dict] = {}
        edges_by_id: Dict[str, dict] = {}
        for row in has_episode_rows:
            ep = self._document_graph_episode_dict(row)
            episodes_by_version[ep["version_id"]] = ep
            edge = self._document_graph_edge_dict(row)
            edge["from"] = f"doc:{edge['document_version_id']}"
            edge["to"] = f"episode:{edge['target_version_id']}"
            edges_by_id[edge["id"]] = edge

        episode_ids = list(episodes_by_version.keys())
        episode_counts = {eid: {"entities": 0, "relations": 0, "edges": 0} for eid in episode_ids}
        total_concepts = 0
        total_relations = 0
        if episode_ids:
            ep_ph = ",".join("?" for _ in episode_ids)
            rows = conn.execute(
                f"""
                SELECT episode_version_id, edge_type,
                       COUNT(*) AS edge_count,
                       COUNT(DISTINCT COALESCE(NULLIF(relation_family_id, ''), target_family_id)) AS concept_count
                FROM concept_edge
                WHERE graph_id = ?
                  AND episode_version_id IN ({ep_ph})
                  AND edge_type IN (?, ?)
                GROUP BY episode_version_id, edge_type
                """,
                [self._graph_id, *episode_ids, EDGE_MENTIONS, EDGE_ASSERTS],
            ).fetchall()
            for row in rows:
                eid = row["episode_version_id"]
                if eid not in episode_counts:
                    continue
                count = int(row["concept_count"] or 0)
                episode_counts[eid]["edges"] += int(row["edge_count"] or 0)
                if row["edge_type"] == EDGE_MENTIONS:
                    episode_counts[eid]["entities"] = count
                elif row["edge_type"] == EDGE_ASSERTS:
                    episode_counts[eid]["relations"] = count

            total_row = conn.execute(
                f"""
                SELECT
                  COUNT(DISTINCT CASE WHEN e.edge_type = ? AND cf.role = ? THEN e.target_family_id END) AS entities,
                  COUNT(DISTINCT CASE WHEN e.edge_type = ? THEN COALESCE(NULLIF(e.relation_family_id, ''), e.target_family_id) END) AS relations
                FROM concept_edge e
                LEFT JOIN concept_family cf
                  ON cf.graph_id = e.graph_id AND cf.family_id = e.target_family_id
                WHERE e.graph_id = ?
                  AND e.episode_version_id IN ({ep_ph})
                  AND e.edge_type IN (?, ?)
                """,
                [EDGE_MENTIONS, ROLE_ENTITY, EDGE_ASSERTS, self._graph_id, *episode_ids, EDGE_MENTIONS, EDGE_ASSERTS],
            ).fetchone()
            total_concepts = int(total_row["entities"] or 0) if total_row else 0
            total_relations = int(total_row["relations"] or 0) if total_row else 0

        selected_families = [d["family_id"] for d in docs_by_version.values()]
        if selected_families:
            fam_ph = ",".join("?" for _ in selected_families)
            rows = conn.execute(
                f"""
                SELECT *
                FROM concept_edge
                WHERE graph_id = ?
                  AND edge_type = ?
                  AND source_family_id IN ({fam_ph})
                  AND target_family_id IN ({fam_ph})
                LIMIT 200
                """,
                [self._graph_id, EDGE_DOCUMENT_LINK, *selected_families, *selected_families],
            ).fetchall()
            doc_node_by_family = {d["family_id"]: d["id"] for d in docs_by_version.values()}
            for row in rows:
                edge = self._document_graph_edge_dict(row)
                edge["from"] = doc_node_by_family.get(edge["source_family_id"], "")
                edge["to"] = doc_node_by_family.get(edge["target_family_id"], "")
                if edge["from"] and edge["to"]:
                    edges_by_id[edge["id"]] = edge

        documents = sorted(docs_by_version.values(), key=lambda d: d.get("processed_time") or "", reverse=True)
        episodes = sorted(episodes_by_version.values(), key=lambda e: (e.get("document_version_id") or "", e.get("chunk_index") or 0, e.get("processed_time") or ""))
        return {
            "documents": documents,
            "episodes": episodes,
            "concepts": [],
            "edges": list(edges_by_id.values()),
            "versions": {},
            "episode_counts": episode_counts,
            "cursor": 0,
            "next_cursor": 0 if episodes else None,
            "counts": {
                "documents": len(documents),
                "episodes": len(episodes),
                "concepts": total_concepts,
                "edges": len(edges_by_id),
                "relations": total_relations,
            },
        }

    def get_document_graph_chunk(
        self,
        document_version_ids: Optional[List[str]] = None,
        document_family_ids: Optional[List[str]] = None,
        cursor: int = 0,
        limit: int = 12,
        include_relations: bool = True,
        include_versions: bool = True,
        max_concepts: int = 8000,
    ) -> dict:
        """Return one episode-ordered concept batch for progressive graph rendering."""
        outline = self.get_document_graph_outline(document_version_ids, document_family_ids, max_episodes=10000)
        all_episodes = outline["episodes"]
        cursor = max(0, int(cursor or 0))
        limit = max(1, min(int(limit or 12), 100))
        batch_episodes = all_episodes[cursor:cursor + limit]
        next_cursor = cursor + len(batch_episodes)
        if next_cursor >= len(all_episodes):
            next_cursor = None
        if not batch_episodes:
            return {
                "documents": outline["documents"],
                "episodes": [],
                "concepts": [],
                "edges": [],
                "versions": {},
                "cursor": cursor,
                "next_cursor": None,
                "totals": outline["counts"],
                "loaded_counts": {"episodes": 0, "concepts": 0, "edges": 0, "relations": 0},
            }

        conn = self._connect()
        episode_ids = [ep["version_id"] for ep in batch_episodes]
        ep_ph = ",".join("?" for _ in episode_ids)
        concept_edge_types = [EDGE_MENTIONS, EDGE_ASSERTS]
        if include_relations:
            concept_edge_types.append(EDGE_CONNECTS)
        type_ph = ",".join("?" for _ in concept_edge_types)
        rows = conn.execute(
            f"""
            SELECT *
            FROM concept_edge
            WHERE graph_id = ?
              AND episode_version_id IN ({ep_ph})
              AND edge_type IN ({type_ph})
            ORDER BY created_at ASC
            LIMIT ?
            """,
            [self._graph_id, *episode_ids, *concept_edge_types, max_concepts * 4],
        ).fetchall()

        needed_families = set()
        edges_by_id: Dict[str, dict] = {}
        relation_families = set()
        for row in rows:
            edge = self._document_graph_edge_dict(row)
            if edge["edge_type"] == EDGE_MENTIONS:
                edge["from"] = f"episode:{edge['episode_version_id']}"
                edge["to"] = f"concept:{edge['target_family_id']}"
                needed_families.add(edge["target_family_id"])
            elif edge["edge_type"] == EDGE_ASSERTS:
                rel_fid = edge["relation_family_id"] or edge["target_family_id"]
                edge["from"] = f"episode:{edge['episode_version_id']}"
                edge["to"] = f"concept:{rel_fid}"
                needed_families.add(rel_fid)
                relation_families.add(rel_fid)
            elif edge["edge_type"] == EDGE_CONNECTS:
                edge["from"] = f"concept:{edge['source_family_id']}"
                edge["to"] = f"concept:{edge['target_family_id']}"
                needed_families.add(edge["source_family_id"])
                needed_families.add(edge["target_family_id"])
                if edge["relation_family_id"]:
                    relation_families.add(edge["relation_family_id"])
                    needed_families.add(edge["relation_family_id"])
            if edge.get("from") and edge.get("to"):
                edges_by_id[edge["id"]] = edge

        concepts_by_family: Dict[str, dict] = {}
        fids = [fid for fid in needed_families if fid][:max_concepts]
        if fids:
            ph = ",".join("?" for _ in fids)
            concept_rows = conn.execute(
                f"""
                SELECT v.* FROM concept_version v
                INNER JOIN (
                  SELECT family_id, MAX(version_seq) AS max_seq
                  FROM concept_version
                  WHERE graph_id = ? AND family_id IN ({ph})
                  GROUP BY family_id
                ) latest
                  ON latest.family_id = v.family_id AND latest.max_seq = v.version_seq
                WHERE v.graph_id = ?
                """,
                [self._graph_id, *fids, self._graph_id],
            ).fetchall()
            for row in concept_rows:
                concept = self._concept_dict(row)
                concept["id"] = f"concept:{concept['family_id']}"
                concept["type"] = concept.get("role") or "concept"
                concepts_by_family[concept["family_id"]] = concept

        versions = {}
        if include_versions and fids:
            ph = ",".join("?" for _ in fids)
            version_rows = conn.execute(
                f"""
                SELECT family_id, COUNT(*) AS total, MAX(version_seq) AS latest_seq
                FROM concept_version
                WHERE graph_id = ? AND family_id IN ({ph})
                GROUP BY family_id
                """,
                [self._graph_id, *fids],
            ).fetchall()
            versions = {
                row["family_id"]: {
                    "total": int(row["total"] or 0),
                    "latest_seq": int(row["latest_seq"] or 0),
                    "latest_version_id": concepts_by_family.get(row["family_id"], {}).get("version_id", ""),
                }
                for row in version_rows
            }

        concepts = sorted(concepts_by_family.values(), key=lambda c: (c.get("role") or "", c.get("name") or c.get("content") or ""))
        edges = list(edges_by_id.values())
        return {
            "documents": outline["documents"],
            "episodes": batch_episodes,
            "concepts": concepts,
            "edges": edges,
            "versions": versions,
            "cursor": cursor,
            "next_cursor": next_cursor,
            "totals": outline["counts"],
            "loaded_counts": {
                "episodes": len(batch_episodes),
                "concepts": len([c for c in concepts if c.get("role") != ROLE_RELATION]),
                "edges": len(edges),
                "relations": len([c for c in concepts if c.get("role") == ROLE_RELATION]),
            },
        }

    def get_document_graph(
        self,
        document_version_ids: Optional[List[str]] = None,
        document_family_ids: Optional[List[str]] = None,
        include_relations: bool = True,
        include_versions: bool = True,
        max_episodes: int = 500,
        max_concepts: int = 1000,
    ) -> dict:
        """Return a Document -> Episode -> Concept subgraph for selected documents."""
        version_ids = [str(x).strip() for x in (document_version_ids or []) if str(x).strip()]
        family_ids = [str(x).strip() for x in (document_family_ids or []) if str(x).strip()]
        if not version_ids and not family_ids:
            raise ValueError("document_version_ids 或 document_family_ids 至少提供一个")

        conn = self._connect()
        docs_by_version: Dict[str, dict] = {}

        def _doc_from_row(row) -> dict:
            metadata = _json_loads(row["metadata"], {})
            return {
                "id": f"doc:{row['document_version_id']}",
                "type": ROLE_DOCUMENT,
                "role": ROLE_DOCUMENT,
                "family_id": row["document_family_id"],
                "version_id": row["document_version_id"],
                "document_version_id": row["document_version_id"],
                "source_id": row["source_id"],
                "title": row["version_title"] or row["source_title"] or "",
                "absolute_path": row["absolute_path"] or "",
                "relative_path": row["relative_path"] or "",
                "uri": row["uri"] or "",
                "content_hash": row["content_hash"] or "",
                "blob_path": row["blob_path"] or "",
                "size": row["size"] or 0,
                "processed_time": row["processed_time"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "frontmatter": _json_loads(row["frontmatter_json"], {}),
                "tags": _json_loads(row["tags_json"], []),
                "aliases": _json_loads(row["aliases_json"], []),
                "metadata": metadata,
            }

        if version_ids:
            ph = ",".join("?" for _ in version_ids)
            rows = conn.execute(
                f"""
                SELECT ds.title AS source_title, ds.absolute_path, ds.relative_path, ds.uri,
                       ds.created_at, ds.updated_at,
                       dv.*, dv.title AS version_title
                FROM document_version dv
                JOIN document_source ds
                  ON ds.source_id = dv.source_id AND ds.graph_id = dv.graph_id
                WHERE dv.graph_id = ? AND dv.document_version_id IN ({ph})
                ORDER BY dv.processed_time DESC
                """,
                [self._graph_id, *version_ids],
            ).fetchall()
            for row in rows:
                docs_by_version[row["document_version_id"]] = _doc_from_row(row)

        if family_ids:
            ph = ",".join("?" for _ in family_ids)
            rows = conn.execute(
                f"""
                SELECT ds.title AS source_title, ds.absolute_path, ds.relative_path, ds.uri,
                       ds.created_at, ds.updated_at,
                       dv.*, dv.title AS version_title
                FROM document_version dv
                JOIN (
                  SELECT document_family_id, MAX(processed_time) AS latest_time
                  FROM document_version
                  WHERE graph_id = ? AND document_family_id IN ({ph})
                  GROUP BY document_family_id
                ) latest
                  ON latest.document_family_id = dv.document_family_id
                 AND latest.latest_time = dv.processed_time
                JOIN document_source ds
                  ON ds.source_id = dv.source_id AND ds.graph_id = dv.graph_id
                WHERE dv.graph_id = ?
                ORDER BY dv.processed_time DESC
                """,
                [self._graph_id, *family_ids, self._graph_id],
            ).fetchall()
            for row in rows:
                docs_by_version[row["document_version_id"]] = _doc_from_row(row)

        if not docs_by_version:
            return {
                "documents": [],
                "episodes": [],
                "concepts": [],
                "edges": [],
                "versions": {},
                "counts": {"documents": 0, "episodes": 0, "concepts": 0, "edges": 0},
            }

        selected_doc_versions = list(docs_by_version.keys())
        doc_ph = ",".join("?" for _ in selected_doc_versions)

        has_episode_rows = conn.execute(
            f"""
            SELECT e.*, ep.*
            FROM concept_edge e
            JOIN concept_version ep
              ON ep.version_id = e.target_version_id
             AND ep.graph_id = e.graph_id
             AND ep.role = ?
            WHERE e.graph_id = ?
              AND e.edge_type = ?
              AND e.document_version_id IN ({doc_ph})
            ORDER BY ep.processed_time ASC
            LIMIT ?
            """,
            [ROLE_EPISODE, self._graph_id, EDGE_HAS_EPISODE, *selected_doc_versions, max_episodes],
        ).fetchall()

        episodes_by_version: Dict[str, dict] = {}
        edges_by_id: Dict[str, dict] = {}

        def _edge_dict(row) -> dict:
            return {
                "id": row["edge_id"],
                "edge_id": row["edge_id"],
                "type": row["edge_type"],
                "edge_type": row["edge_type"],
                "source_family_id": row["source_family_id"] or "",
                "source_version_id": row["source_version_id"] or "",
                "target_family_id": row["target_family_id"] or "",
                "target_version_id": row["target_version_id"] or "",
                "relation_family_id": row["relation_family_id"] or "",
                "relation_version_id": row["relation_version_id"] or "",
                "episode_version_id": row["episode_version_id"] or "",
                "document_version_id": row["document_version_id"] or "",
                "weight": row["weight"],
                "confidence": row["confidence"],
                "provenance": _json_loads(row["provenance"], {}),
                "created_at": row["created_at"],
            }

        for row in has_episode_rows:
            ep = self._concept_dict(row)
            ep_meta = ep.get("metadata") or {}
            ep.update({
                "id": f"episode:{ep['version_id']}",
                "type": ROLE_EPISODE,
                "heading_path": ep_meta.get("heading_path", ""),
                "start_offset": ep_meta.get("start_offset"),
                "end_offset": ep_meta.get("end_offset"),
                "chunk_index": ep_meta.get("chunk_index"),
                "chunk_hash": ep_meta.get("chunk_hash"),
                "source_span": {
                    "heading_path": ep_meta.get("heading_path", ""),
                    "start_offset": ep_meta.get("start_offset"),
                    "end_offset": ep_meta.get("end_offset"),
                    "chunk_index": ep_meta.get("chunk_index"),
                    "chunk_hash": ep_meta.get("chunk_hash"),
                },
            })
            episodes_by_version[ep["version_id"]] = ep
            edge = _edge_dict(row)
            edge["from"] = f"doc:{edge['document_version_id']}"
            edge["to"] = f"episode:{edge['target_version_id']}"
            edges_by_id[edge["id"]] = edge

        episode_ids = list(episodes_by_version.keys())
        concepts_by_family: Dict[str, dict] = {}
        if episode_ids:
            ep_ph = ",".join("?" for _ in episode_ids)
            concept_edge_types = [EDGE_MENTIONS, EDGE_ASSERTS]
            if include_relations:
                concept_edge_types.append(EDGE_CONNECTS)
            type_ph = ",".join("?" for _ in concept_edge_types)
            rows = conn.execute(
                f"""
                SELECT *
                FROM concept_edge
                WHERE graph_id = ?
                  AND episode_version_id IN ({ep_ph})
                  AND edge_type IN ({type_ph})
                ORDER BY created_at ASC
                LIMIT ?
                """,
                [self._graph_id, *episode_ids, *concept_edge_types, max_concepts * 4],
            ).fetchall()

            needed_families = set()
            for row in rows:
                edge = _edge_dict(row)
                if edge["edge_type"] == EDGE_MENTIONS:
                    edge["from"] = f"episode:{edge['episode_version_id']}"
                    edge["to"] = f"concept:{edge['target_family_id']}"
                    needed_families.add(edge["target_family_id"])
                elif edge["edge_type"] == EDGE_ASSERTS:
                    edge["from"] = f"episode:{edge['episode_version_id']}"
                    edge["to"] = f"concept:{edge['relation_family_id'] or edge['target_family_id']}"
                    needed_families.add(edge["relation_family_id"] or edge["target_family_id"])
                elif edge["edge_type"] == EDGE_CONNECTS:
                    edge["from"] = f"concept:{edge['source_family_id']}"
                    edge["to"] = f"concept:{edge['target_family_id']}"
                    needed_families.add(edge["source_family_id"])
                    needed_families.add(edge["target_family_id"])
                if edge.get("from") and edge.get("to"):
                    edges_by_id[edge["id"]] = edge

            for fid in list(needed_families)[:max_concepts]:
                if fid and (concept := self.get_concept_by_family_id(fid)):
                    concept["id"] = f"concept:{fid}"
                    concept["type"] = concept.get("role") or "concept"
                    concepts_by_family[fid] = concept

        # Document links between selected documents are useful context but not
        # required for the radial episode layout.
        selected_families = [d["family_id"] for d in docs_by_version.values()]
        if selected_families:
            fam_ph = ",".join("?" for _ in selected_families)
            rows = conn.execute(
                f"""
                SELECT *
                FROM concept_edge
                WHERE graph_id = ?
                  AND edge_type = ?
                  AND source_family_id IN ({fam_ph})
                  AND target_family_id IN ({fam_ph})
                LIMIT 200
                """,
                [self._graph_id, EDGE_DOCUMENT_LINK, *selected_families, *selected_families],
            ).fetchall()
            doc_node_by_family = {d["family_id"]: d["id"] for d in docs_by_version.values()}
            for row in rows:
                edge = _edge_dict(row)
                edge["from"] = doc_node_by_family.get(edge["source_family_id"], "")
                edge["to"] = doc_node_by_family.get(edge["target_family_id"], "")
                if edge["from"] and edge["to"]:
                    edges_by_id[edge["id"]] = edge

        versions = {}
        if include_versions and concepts_by_family:
            fids = list(concepts_by_family.keys())
            ph = ",".join("?" for _ in fids)
            rows = conn.execute(
                f"""
                SELECT family_id, COUNT(*) AS total, MAX(version_seq) AS latest_seq
                FROM concept_version
                WHERE graph_id = ? AND family_id IN ({ph})
                GROUP BY family_id
                """,
                [self._graph_id, *fids],
            ).fetchall()
            versions = {
                row["family_id"]: {
                    "total": int(row["total"] or 0),
                    "latest_seq": int(row["latest_seq"] or 0),
                    "latest_version_id": concepts_by_family.get(row["family_id"], {}).get("version_id", ""),
                }
                for row in rows
            }

        documents = sorted(docs_by_version.values(), key=lambda d: d.get("processed_time") or "", reverse=True)
        episodes = sorted(episodes_by_version.values(), key=lambda e: (e.get("document_version_id") or "", e.get("chunk_index") or 0, e.get("processed_time") or ""))
        concepts = sorted(concepts_by_family.values(), key=lambda c: (c.get("role") or "", c.get("name") or c.get("content") or ""))
        edges = list(edges_by_id.values())
        return {
            "documents": documents,
            "episodes": episodes,
            "concepts": concepts,
            "edges": edges,
            "versions": versions,
            "counts": {
                "documents": len(documents),
                "episodes": len(episodes),
                "concepts": len(concepts),
                "edges": len(edges),
            },
        }

    # ------------------------------------------------------------------
    # Misc compatibility methods
    # ------------------------------------------------------------------

    def register_entity_redirect(self, source_id: str, target_id: str):
        self.register_entity_redirects_batch({source_id: target_id})

    def register_entity_redirects_batch(self, redirects: Dict[str, str]):
        now = _fmt_dt(_now())
        with self._write_lock:
            conn = self._connect()
            for source, target in (redirects or {}).items():
                if source and target and source != target:
                    conn.execute(
                        "INSERT OR REPLACE INTO concept_redirect (source_family_id, target_family_id, graph_id, updated_at) VALUES (?, ?, ?, ?)",
                        (source, target, self._graph_id, now),
                    )
            conn.commit()

    def merge_entity_families(self, target_family_id: str, source_family_ids: List[str], skip_name_check: bool = False) -> Dict[str, Any]:
        self.register_entity_redirects_batch({sid: target_family_id for sid in source_family_ids if sid and sid != target_family_id})
        return {"entities_updated": len(source_family_ids or []), "relations_updated": 0}

    def redirect_entity_relations(self, old_family_id: str, new_family_id: str):
        self.register_entity_redirect(old_family_id, new_family_id)

    def delete_entity_all_versions(self, family_id: str) -> int:
        with self._write_lock:
            conn = self._connect()
            rows = conn.execute("DELETE FROM concept_version WHERE family_id = ? AND graph_id = ?", (family_id, self._graph_id))
            conn.execute("DELETE FROM concept_family WHERE family_id = ? AND graph_id = ?", (family_id, self._graph_id))
            conn.commit()
            return rows.rowcount

    def refresh_relates_to_edges(self, family_ids: List[str] = None):
        return None

    def save_content_patches(self, patches):
        return 0

    def update_entity_summary(self, family_id: str, summary: str):
        row = self._latest_version_row(family_id, role=ROLE_ENTITY)
        if row:
            self._connect().execute("UPDATE concept_version SET summary = ? WHERE version_id = ?", (summary, row["version_id"]))
            self._connect().commit()

    def adjust_confidence_on_corroboration(self, family_id: str, source_type: str = "entity", **_ignored):
        return None

    def adjust_confidence_on_corroboration_batch(self, family_ids: List[str], source_type: str = "entity", **_ignored):
        return None

    def adjust_confidence_on_contradiction(self, family_id: str, source_type: str = "entity"):
        return None

    def batch_get_source_text_snippets(self, episode_ids: List[str], snippet_length: int = 200) -> Dict[str, str]:
        result = {}
        for eid in episode_ids or []:
            ep = self.load_episode(eid)
            result[eid] = (ep.content if ep else "")[:snippet_length]
        return result

    def batch_bfs_traverse(self, seed_family_ids: List[str], max_depth: int = 2, max_nodes: int = 50, time_point: Optional[str] = None):
        result = self.traverse_concepts(seed_family_ids, max_depth=max_depth, time_point=time_point)
        entities = []
        relations = []
        hops = {}
        for fid, c in result["concepts"].items():
            hops[fid] = 0
            if c["role"] == ROLE_ENTITY:
                ent = self.get_entity_by_family_id(fid)
                if ent:
                    entities.append(ent)
            elif c["role"] == ROLE_RELATION:
                rel = self.get_relation_by_family_id(fid)
                if rel:
                    relations.append(rel)
        return entities[:max_nodes], relations[: max_nodes * 3], hops

    def get_stats(self) -> dict:
        return {
            "documents": self.count_concepts(ROLE_DOCUMENT),
            "episodes": self.count_concepts(ROLE_EPISODE),
            "entities": self.count_concepts(ROLE_ENTITY),
            "relations": self.count_concepts(ROLE_RELATION),
            "concepts": self.count_concepts(),
        }

    def count_unique_entities(self) -> int:
        return self.count_concepts(ROLE_ENTITY)

    def count_unique_relations(self) -> int:
        return self.count_concepts(ROLE_RELATION)

    def count_episodes(self) -> int:
        return self.count_concepts(ROLE_EPISODE)

    def get_graph_statistics(self) -> dict:
        return self.get_stats()

    def get_data_quality_report(self) -> dict:
        return {"issues": [], "warnings": [], "stats": self.get_stats()}

    def count_isolated_entities(self) -> int:
        rows = self._connect().execute(
            """
            SELECT COUNT(*) AS cnt FROM concept_family cf
            WHERE cf.graph_id = ? AND cf.role = ?
              AND NOT EXISTS (
                SELECT 1 FROM concept_edge e
                WHERE e.graph_id = cf.graph_id
                  AND (e.source_family_id = cf.family_id OR e.target_family_id = cf.family_id)
              )
            """,
            (self._graph_id, ROLE_ENTITY),
        ).fetchone()
        return int(rows["cnt"] or 0)

    def clear_graph_data(self):
        with self._write_lock:
            conn = self._connect()
            for table in ("concept_edge", "concept_version", "concept_family", "document_version", "document_source", "blob_manifest", "concept_redirect"):
                conn.execute(f"DELETE FROM {table} WHERE graph_id = ?", (self._graph_id,))
            try:
                conn.execute("DELETE FROM concept_version_fts WHERE graph_id = ?", (self._graph_id,))
            except Exception:
                pass
            conn.commit()
        for sub in ("blobs", "artifacts", "indexes", "logs"):
            path = self.storage_path / sub
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)

    def delete_graph_data(self):
        self.clear_graph_data()
