"""V1.5 Library Manager — facade over repository layer.

Replaces the old SQLiteGraphStorageManager (5200-line monolith).
All DB operations delegate to V1.5 repository functions.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ...models import Entity, Episode, Relation
from ..cache import QueryCache
from .dto_mapping import assertion_to_relation, episode_row_to_dto, observation_to_entity
from .helpers import _encode_and_normalize, _fmt_dt, _parse_dt
from .schema_v15 import init_schema_v15

from .repositories import (
    documents as doc_repo,
    embeddings as emb_repo,
    episodes as ep_repo,
    entities as ent_repo,
    relations as rel_repo,
    search as search_repo,
)

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_str() -> str:
    return _now().isoformat()


class LibraryManager:
    """V1.5 storage facade used by the remember pipeline and server."""

    def __init__(
        self,
        library_path: str = None,
        embedding_client=None,
        entity_content_snippet_length: int = 50,
        relation_content_snippet_length: int = 50,
        # Old compat kwargs
        storage_path: str = None,
        vector_dim: int = 1024,
        graph_id: str = None,
    ):
        if library_path is None and storage_path is not None:
            library_path = storage_path
        if library_path is None:
            library_path = "./library"
        self.library_path = Path(library_path)
        self.library_path.mkdir(parents=True, exist_ok=True)
        self._db_path = self.library_path / "library.db"
        self.embedding_client = embedding_client
        self.entity_content_snippet_length = entity_content_snippet_length
        self.relation_content_snippet_length = relation_content_snippet_length

        # Directory layout for content files
        self.storage_path = self.library_path
        self.documents_dir = self.library_path / "documents"
        self.extraction_cache_dir = self.library_path / "tasks" / "extraction_cache"
        self.content_dir = self.library_path / "content"
        for d in (self.documents_dir, self.extraction_cache_dir,
                  self.content_dir, self.content_dir / "versions"):
            d.mkdir(parents=True, exist_ok=True)

        # Compat aliases (old manager attributes read by server/CLI)
        self.cache_dir = self.extraction_cache_dir
        self.cache_json_dir = self.extraction_cache_dir
        self.docs_dir = self.documents_dir
        self.artifacts_dir = self.extraction_cache_dir
        self.snapshots_dir = self.library_path / "snapshots" / "sha256"
        self.blobs_dir = self.snapshots_dir

        self._thread_local = threading.local()
        self._all_conns: List[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()
        self._write_lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._closed = False
        self._entity_name_cache: Dict[str, str] = {}
        self._cache = QueryCache(default_ttl=30, max_size=4096)
        self._vector_cache_lock = threading.RLock()
        self._vector_role_cache: Dict[str, dict] = {}

        conn = self._conn()
        init_schema_v15(conn)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("LibraryManager is closed")
            conn = getattr(self._thread_local, "conn", None)
            if conn is not None:
                return conn
            conn = sqlite3.connect(str(self._db_path), timeout=30, check_same_thread=False)
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
        with self._lifecycle_lock:
            self._closed = True
            with self._conn_lock:
                conns = list(self._all_conns)
                self._all_conns.clear()
            if getattr(self._thread_local, "conn", None) is not None:
                self._thread_local.conn = None
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

    def _in_write_batch(self) -> bool:
        return int(getattr(self._thread_local, "write_batch_depth", 0) or 0) > 0

    def _commit_if_not_batched(self, conn: sqlite3.Connection) -> None:
        if not self._in_write_batch():
            conn.commit()

    @contextmanager
    def _write_batch(self):
        with self._write_lock:
            conn = self._conn()
            depth = int(getattr(self._thread_local, "write_batch_depth", 0) or 0)
            self._thread_local.write_batch_depth = depth + 1
            try:
                yield conn
            finally:
                self._thread_local.write_batch_depth = depth
                if depth == 0:
                    conn.commit()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_entity_name(self, absolute_id: str, name: str):
        if absolute_id:
            self._entity_name_cache[absolute_id] = name or ""

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def list_documents(self, limit: int = 50, offset: int = 0,
                       source_document: str = None) -> List[dict]:
        conn = self._conn()
        docs = doc_repo.list_documents(conn, status="active",
                                       limit=limit, offset=offset)
        for d in docs:
            d["role"] = "document"
            if d.get("current_version_id"):
                d["document_version_id"] = d["current_version_id"]
        # Enrich with size from document_versions and counts
        for d in docs:
            ver_id = d.get("document_version_id")
            if ver_id:
                ver = conn.execute(
                    "SELECT byte_size, char_count FROM document_versions WHERE document_version_id = ?",
                    (ver_id,),
                ).fetchone()
                if ver:
                    d["size"] = ver[0] or 0
                    d["char_count"] = ver[1] or 0
            doc_id = d.get("document_id")
            if doc_id:
                ep_cnt = conn.execute(
                    "SELECT COUNT(DISTINCT eo.entity_family_id) FROM entity_mentions em "
                    "JOIN entity_observations eo ON eo.entity_id = em.entity_id AND eo.status = 'active' "
                    "JOIN episodes ep ON ep.episode_id = em.episode_id AND ep.status = 'active' "
                    "WHERE ep.document_id = ?",
                    (doc_id,),
                ).fetchone()[0]
                rel_cnt = conn.execute(
                    "SELECT COUNT(DISTINCT ra.relation_family_id) FROM relation_assertions ra "
                    "JOIN episodes ep ON ep.episode_id = ra.episode_id AND ep.status = 'active' "
                    "WHERE ra.status = 'active' AND ep.document_id = ?",
                    (doc_id,),
                ).fetchone()[0]
                d["entity_count"] = ep_cnt
                d["relation_count"] = rel_cnt
        return docs

    def count_documents(self, source_document: str = None) -> int:
        conn = self._conn()
        if source_document:
            row = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE status = 'active' "
                "AND (title LIKE ? OR managed_path LIKE ? OR absolute_path LIKE ?)",
                (f"%{source_document}%", f"%{source_document}%", f"%{source_document}%"),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE status = 'active'"
            ).fetchone()
        return row[0]

    def get_document(self, document_id: str) -> Optional[dict]:
        doc = doc_repo.get_document(self._conn(), document_id)
        if doc:
            doc["role"] = "document"
            if doc.get("current_version_id"):
                doc["document_version_id"] = doc["current_version_id"]
        return doc

    def get_document_content(self, document_version_id: str, *,
                             offset: int = 0, limit: int = 10_000_000) -> dict:
        conn = self._conn()
        ver = conn.execute(
            "SELECT * FROM document_versions WHERE document_version_id = ?",
            (document_version_id,),
        ).fetchone()
        if not ver:
            return {"content": "", "read_path": ""}
        ver = dict(ver)
        doc = doc_repo.get_document(conn, ver["document_id"]) or {}

        # Read content from managed file or snapshot
        content = ""
        read_path = ""
        managed = doc.get("managed_path", "")
        if managed:
            full = self.library_path / managed
            if full.exists():
                content = full.read_text(encoding="utf-8")
                read_path = managed
        if not content and ver.get("version_content_path"):
            full = self.library_path / ver["version_content_path"]
            if full.exists():
                content = full.read_text(encoding="utf-8")
                read_path = ver["version_content_path"]

        if offset > 0 or limit < len(content):
            content = content[offset:offset + limit]
        return {
            "content": content,
            "read_path": read_path,
            "source_mode": doc.get("source_mode", ""),
            "title": doc.get("title") or "",
        }

    def get_document_file_info(self, document_version_id: str) -> dict:
        conn = self._conn()
        ver = conn.execute(
            "SELECT * FROM document_versions WHERE document_version_id = ?",
            (document_version_id,),
        ).fetchone()
        if not ver:
            return {}
        ver = dict(ver)
        doc = doc_repo.get_document(conn, ver["document_id"]) or {}
        return {
            "document_version_id": document_version_id,
            "document_id": ver.get("document_id"),
            "title": ver.get("title", ""),
            "content_hash": ver.get("content_hash", ""),
            "char_count": ver.get("char_count", 0),
            "line_count": ver.get("line_count", 0),
            "byte_size": ver.get("byte_size", 0),
            "managed_path": doc.get("managed_path", ""),
            "absolute_path": doc.get("absolute_path", ""),
        }

    def delete_document_version(self, document_version_id: str) -> dict:
        conn = self._conn()
        # Get document_id
        ver = conn.execute(
            "SELECT document_id FROM document_versions WHERE document_version_id = ?",
            (document_version_id,),
        ).fetchone()
        if not ver:
            return {"deleted": False, "reason": "not found"}
        doc_id = ver[0]
        now = _now_str()

        # 1. Cascade-delete episodes belonging to this document
        ep_ids = [r[0] for r in conn.execute(
            "SELECT episode_id FROM episodes WHERE document_id = ?", (doc_id,)
        ).fetchall()]

        if ep_ids:
            ph = ",".join("?" for _ in ep_ids)

            # Delete relation_assertions linked to these episodes
            orphan_rel_fam_ids = {r[0] for r in conn.execute(
                f"SELECT DISTINCT relation_family_id FROM relation_assertions WHERE episode_id IN ({ph})", ep_ids
            ).fetchall()}

            # Delete entity_observations linked to these episodes
            orphan_ent_fam_ids = {r[0] for r in conn.execute(
                f"SELECT DISTINCT entity_family_id FROM entity_observations WHERE episode_id IN ({ph})", ep_ids
            ).fetchall()}

            # Delete entity_mentions linked to these episodes
            conn.execute(f"DELETE FROM entity_mentions WHERE episode_id IN ({ph})", ep_ids)
            # Delete relation_assertions linked to these episodes
            conn.execute(f"DELETE FROM relation_assertions WHERE episode_id IN ({ph})", ep_ids)
            # Delete entity_observations linked to these episodes
            conn.execute(f"DELETE FROM entity_observations WHERE episode_id IN ({ph})", ep_ids)
            # Delete embeddings linked to these episodes
            conn.execute(f"DELETE FROM embeddings WHERE owner_type = 'episode' AND owner_id IN ({ph})", ep_ids)

            # Delete episodes
            conn.execute(f"DELETE FROM episodes WHERE episode_id IN ({ph})", ep_ids)

            # For entity families: only delete if no observations remain
            if orphan_ent_fam_ids:
                fam_ph = ",".join("?" for _ in orphan_ent_fam_ids)
                surviving = {r[0] for r in conn.execute(
                    f"SELECT DISTINCT entity_family_id FROM entity_observations WHERE entity_family_id IN ({fam_ph})",
                    list(orphan_ent_fam_ids),
                ).fetchall()}
                to_delete = orphan_ent_fam_ids - surviving
                if to_delete:
                    del_ph = ",".join("?" for _ in to_delete)
                    # Check for relation_families referencing these entity families
                    rel_fams_blocked = {r[0] for r in conn.execute(
                        f"SELECT DISTINCT rf.relation_family_id FROM relation_families rf "
                        f"WHERE rf.subject_entity_family_id IN ({del_ph}) "
                        f"OR rf.object_entity_family_id IN ({del_ph})",
                        list(to_delete) + list(to_delete),
                    ).fetchall()}
                    # Delete assertions for these relation families first
                    if rel_fams_blocked:
                        rel_blocked_ph = ",".join("?" for _ in rel_fams_blocked)
                        conn.execute(f"DELETE FROM relation_assertions WHERE relation_family_id IN ({rel_blocked_ph})", list(rel_fams_blocked))
                        conn.execute(f"DELETE FROM embeddings WHERE owner_type = 'relation_assert' AND owner_id IN ({rel_blocked_ph})", list(rel_fams_blocked))
                        conn.execute(f"DELETE FROM relation_families WHERE relation_family_id IN ({rel_blocked_ph})", list(rel_fams_blocked))
                    conn.execute(f"DELETE FROM entity_mentions WHERE entity_family_id IN ({del_ph})", list(to_delete))
                    conn.execute(f"DELETE FROM embeddings WHERE owner_type = 'entity_obs' AND owner_id IN ({del_ph})", list(to_delete))
                    conn.execute(f"DELETE FROM embeddings WHERE owner_type = 'entity_family' AND owner_id IN ({del_ph})", list(to_delete))
                    conn.execute(f"DELETE FROM entity_families WHERE entity_family_id IN ({del_ph})", list(to_delete))

            # For relation families: only delete if no assertions remain
            if orphan_rel_fam_ids:
                fam_ph = ",".join("?" for _ in orphan_rel_fam_ids)
                surviving = {r[0] for r in conn.execute(
                    f"SELECT DISTINCT relation_family_id FROM relation_assertions WHERE relation_family_id IN ({fam_ph})",
                    list(orphan_rel_fam_ids),
                ).fetchall()}
                to_delete = orphan_rel_fam_ids - surviving
                if to_delete:
                    del_ph = ",".join("?" for _ in to_delete)
                    conn.execute(f"DELETE FROM embeddings WHERE owner_type = 'relation_assert' AND owner_id IN ({del_ph})", list(to_delete))
                    conn.execute(f"DELETE FROM relation_families WHERE relation_family_id IN ({del_ph})", list(to_delete))

        # Delete document_links for this document
        conn.execute("DELETE FROM document_links WHERE from_document_id = ?", (doc_id,))

        # Soft-delete document_version and document
        conn.execute(
            "UPDATE document_versions SET status = 'deleted', processed_at = ? WHERE document_version_id = ?",
            (now, document_version_id),
        )
        doc_repo.soft_delete_document(conn, doc_id, updated_at=now)
        conn.commit()
        return {"deleted": True, "document_id": doc_id}

    # ------------------------------------------------------------------
    # Episode operations
    # ------------------------------------------------------------------

    def load_episode(self, cache_id: str) -> Optional[Episode]:
        row = ep_repo.get_episode(self._conn(), cache_id)
        if not row:
            return None
        return episode_row_to_dto(row)

    def get_episode(self, cache_id: str) -> Optional[dict]:
        return ep_repo.get_episode(self._conn(), cache_id)

    def get_episode_content_detail(self, cache_id: str) -> Optional[dict]:
        row = ep_repo.get_episode(self._conn(), cache_id)
        if not row:
            return None
        return {
            "episode_id": row.get("episode_id", ""),
            "source_text": row.get("source_text", ""),
            "memory_text": row.get("memory_text", ""),
            "heading_path": row.get("heading_path", ""),
            "start_offset": row.get("start_offset", 0),
            "end_offset": row.get("end_offset", 0),
            "line_start": row.get("line_start", 0),
            "line_end": row.get("line_end", 0),
            "source_path": "",
        }

    def get_latest_episode_metadata(self, activity_type: str = None) -> Optional[dict]:
        conn = self._conn()
        if activity_type:
            row = conn.execute(
                "SELECT episode_id, activity_type, processed_at "
                "FROM episodes WHERE status = 'active' AND activity_type = ? "
                "ORDER BY processed_at DESC LIMIT 1",
                (activity_type,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT episode_id, activity_type, processed_at "
                "FROM episodes WHERE status = 'active' "
                "ORDER BY processed_at DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def count_episodes(self) -> int:
        return self._conn().execute(
            "SELECT COUNT(*) FROM episodes WHERE status = 'active'"
        ).fetchone()[0]

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    def get_entity_by_family_id(self, family_id: str) -> Optional[Entity]:
        conn = self._conn()
        fam = ent_repo.get_entity_family(conn, family_id)
        if not fam:
            return None
        obs = conn.execute(
            "SELECT * FROM entity_observations "
            "WHERE entity_family_id = ? AND status = 'active' "
            "ORDER BY processed_at DESC, rowid DESC LIMIT 1",
            (family_id,),
        ).fetchone()
        if not obs:
            return None
        emb = self._get_embedding_blob("entity_obs", dict(obs)["entity_id"])
        version_seq = conn.execute(
            "SELECT COUNT(*) FROM entity_observations "
            "WHERE entity_family_id = ? AND processed_at <= ?",
            (family_id, dict(obs).get("processed_at", "")),
        ).fetchone()[0]
        return observation_to_entity(fam, dict(obs), embedding_blob=emb, version_seq=version_seq)

    def get_entities_by_family_ids(self, family_ids: List[str]) -> Dict[str, Entity]:
        if not family_ids:
            return {}
        result = {}
        for fid in family_ids:
            e = self.get_entity_by_family_id(fid)
            if e:
                result[fid] = e
        return result

    def get_entities_by_absolute_ids(self, absolute_ids: List[str]) -> List[Entity]:
        if not absolute_ids:
            return []
        conn = self._conn()
        placeholders = ",".join("?" for _ in absolute_ids)
        rows = conn.execute(
            f"SELECT eo.*, ef.entity_family_id, ef.canonical_name, ef.canonical_content "
            f"FROM entity_observations eo "
            f"JOIN entity_families ef ON ef.entity_family_id = eo.entity_family_id "
            f"WHERE eo.entity_id IN ({placeholders}) AND eo.status = 'active' "
            f"ORDER BY eo.processed_at DESC",
            absolute_ids,
        ).fetchall()
        entities = []
        for row in rows:
            row = dict(row)
            fam = {"entity_family_id": row["entity_family_id"],
                   "canonical_name": row["canonical_name"],
                   "canonical_content": row["canonical_content"]}
            emb = self._get_embedding_blob("entity_obs", row["entity_id"])
            entities.append(observation_to_entity(fam, row, embedding_blob=emb))
        return entities

    def get_entity_versions(self, family_id: str) -> List[Entity]:
        conn = self._conn()
        fam = ent_repo.get_entity_family(conn, family_id)
        if not fam:
            return []
        rows = conn.execute(
            "SELECT * FROM entity_observations "
            "WHERE entity_family_id = ? AND status != 'deleted' "
            "ORDER BY processed_at ASC",
            (family_id,),
        ).fetchall()
        entities = []
        for i, row in enumerate(rows, 1):
            row = dict(row)
            emb = self._get_embedding_blob("entity_obs", row["entity_id"])
            entities.append(observation_to_entity(fam, row, embedding_blob=emb, version_seq=i))
        return entities

    def get_entity_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
        if not family_ids:
            return {}
        conn = self._conn()
        placeholders = ",".join("?" for _ in family_ids)
        rows = conn.execute(
            f"SELECT entity_family_id, COUNT(*) as cnt "
            f"FROM entity_observations "
            f"WHERE entity_family_id IN ({placeholders}) AND status != 'deleted' "
            f"GROUP BY entity_family_id",
            family_ids,
        ).fetchall()
        result = {fid: 0 for fid in family_ids}
        for row in rows:
            result[row[0]] = row[1]
        return result

    def get_entity_version_count(self, family_id: str) -> int:
        return self.get_entity_version_counts([family_id]).get(family_id, 0)

    def get_family_ids_by_names(self, names: List[str]) -> Dict[str, str]:
        result = {}
        for name in names:
            fam = ent_repo.find_entity_family_by_name(self._conn(), name)
            if fam:
                result[name] = fam["entity_family_id"]
        return result

    def get_entity_names_by_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, str]:
        if not absolute_ids:
            return {}
        conn = self._conn()
        placeholders = ",".join("?" for _ in absolute_ids)
        rows = conn.execute(
            f"SELECT entity_id, name FROM entity_observations "
            f"WHERE entity_id IN ({placeholders})",
            absolute_ids,
        ).fetchall()
        result = {}
        for row in rows:
            result[row[0]] = row[1]
        # Also check cache
        for aid in absolute_ids:
            if aid not in result and aid in self._entity_name_cache:
                result[aid] = self._entity_name_cache[aid]
        return result

    def get_family_ids_by_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, str]:
        if not absolute_ids:
            return {}
        conn = self._conn()
        placeholders = ",".join("?" for _ in absolute_ids)
        rows = conn.execute(
            f"SELECT entity_id, entity_family_id FROM entity_observations "
            f"WHERE entity_id IN ({placeholders})",
            absolute_ids,
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_latest_absolute_ids_by_family_ids(self, family_ids: List[str]) -> Dict[str, str]:
        if not family_ids:
            return {}
        conn = self._conn()
        result = {}
        for fid in family_ids:
            row = conn.execute(
                "SELECT entity_id FROM entity_observations "
                "WHERE entity_family_id = ? AND status = 'active' "
                "ORDER BY processed_at DESC LIMIT 1",
                (fid,),
            ).fetchone()
            if row:
                result[fid] = row[0]
        return result

    def get_latest_entities_projection(self, content_snippet_length: int = None) -> List[dict]:
        snippet_len = content_snippet_length or self.entity_content_snippet_length
        conn = self._conn()
        rows = conn.execute(
            "SELECT ef.entity_family_id, ef.canonical_name, ef.canonical_content, "
            "  eo.entity_id, eo.content, eo.processed_at, "
            "  (SELECT COUNT(*) FROM entity_observations eo2 "
            "   WHERE eo2.entity_family_id = ef.entity_family_id AND eo2.status != 'deleted') as version_count "
            "FROM entity_families ef "
            "JOIN entity_observations eo ON eo.entity_family_id = ef.entity_family_id AND eo.status = 'active' "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM entity_redirects r WHERE r.source_family_id = ef.entity_family_id"
            ") "
            "ORDER BY ef.updated_at DESC"
        ).fetchall()
        results = []
        seen = set()
        for row in rows:
            row = dict(row)
            fid = row["entity_family_id"]
            if fid in seen:
                continue
            seen.add(fid)
            content = row.get("content") or row.get("canonical_content", "")
            results.append({
                "family_id": fid,
                "name": row["canonical_name"],
                "content": content[:snippet_len] if content else "",
                "content_snippet": content[:snippet_len] if content else "",
                "version_count": row.get("version_count", 1),
                "entity": observation_to_entity(
                    {"entity_family_id": fid, "canonical_name": row["canonical_name"],
                     "canonical_content": row.get("canonical_content", "")},
                    row,
                ),
            })
        return results

    def get_all_entities(self, limit: int = 100, offset: int = None,
                         exclude_embedding: bool = False) -> List[Entity]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT ef.entity_family_id, ef.canonical_name, ef.canonical_content, "
            "  eo.entity_id, eo.name, eo.content, eo.episode_id, eo.processed_at "
            "FROM entity_families ef "
            "JOIN entity_observations eo ON eo.entity_family_id = ef.entity_family_id AND eo.status = 'active' "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM entity_redirects r WHERE r.source_family_id = ef.entity_family_id"
            ") "
            "ORDER BY ef.updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset or 0),
        ).fetchall()
        entities = []
        seen = set()
        for row in rows:
            row = dict(row)
            fid = row["entity_family_id"]
            if fid in seen:
                continue
            seen.add(fid)
            emb = None if exclude_embedding else self._get_embedding_blob("entity_obs", row["entity_id"])
            entities.append(observation_to_entity(
                {"entity_family_id": fid, "canonical_name": row["canonical_name"],
                 "canonical_content": row.get("canonical_content", "")},
                row,
                embedding_blob=emb,
            ))
        return entities

    def get_all_entities_before_time(self, time_point, limit: int = 100,
                                     exclude_embedding: bool = False) -> List[Entity]:
        ts = _fmt_dt(time_point) or _now_str()
        conn = self._conn()
        rows = conn.execute(
            "SELECT ef.entity_family_id, ef.canonical_name, ef.canonical_content, "
            "  eo.entity_id, eo.name, eo.content, eo.episode_id, eo.processed_at "
            "FROM entity_families ef "
            "JOIN entity_observations eo ON eo.entity_family_id = ef.entity_family_id AND eo.status = 'active' "
            "WHERE eo.processed_at <= ? "
            "AND NOT EXISTS (SELECT 1 FROM entity_redirects r WHERE r.source_family_id = ef.entity_family_id) "
            "ORDER BY eo.processed_at DESC LIMIT ?",
            (ts, limit),
        ).fetchall()
        entities = []
        seen = set()
        for row in rows:
            row = dict(row)
            fid = row["entity_family_id"]
            if fid in seen:
                continue
            seen.add(fid)
            emb = None if exclude_embedding else self._get_embedding_blob("entity_obs", row["entity_id"])
            entities.append(observation_to_entity(
                {"entity_family_id": fid, "canonical_name": row["canonical_name"],
                 "canonical_content": row.get("canonical_content", "")},
                row,
                embedding_blob=emb,
            ))
        return entities

    # ------------------------------------------------------------------
    # Relation operations
    # ------------------------------------------------------------------

    def get_relation_by_absolute_id(self, absolute_id: str) -> Optional[Relation]:
        conn = self._conn()
        row = conn.execute(
            "SELECT ra.*, rf.relation_family_id, rf.subject_entity_family_id, "
            "  rf.object_entity_family_id, rf.predicate, rf.canonical_content "
            "FROM relation_assertions ra "
            "JOIN relation_families rf ON rf.relation_family_id = ra.relation_family_id "
            "WHERE ra.relation_id = ?",
            (absolute_id,),
        ).fetchone()
        if not row:
            return None
        row = dict(row)
        fam = {k: row[k] for k in ("relation_family_id", "subject_entity_family_id",
                                     "object_entity_family_id", "predicate", "canonical_content")}
        sub_abs = self._latest_obs_id_for_family(row["subject_entity_family_id"])
        obj_abs = self._latest_obs_id_for_family(row["object_entity_family_id"])
        emb = self._get_embedding_blob("relation_assert", absolute_id)
        return assertion_to_relation(fam, row, subject_entity_id=sub_abs,
                                     object_entity_id=obj_abs, embedding_blob=emb)

    def get_relation_by_family_id(self, family_id: str) -> Optional[Relation]:
        conn = self._conn()
        fam = rel_repo.get_relation_family(conn, family_id)
        if not fam:
            return None
        row = conn.execute(
            "SELECT * FROM relation_assertions "
            "WHERE relation_family_id = ? AND status = 'active' "
            "ORDER BY processed_at DESC LIMIT 1",
            (family_id,),
        ).fetchone()
        if not row:
            return None
        row = dict(row)
        sub_abs = self._latest_obs_id_for_family(row["subject_entity_family_id"])
        obj_abs = self._latest_obs_id_for_family(row["object_entity_family_id"])
        emb = self._get_embedding_blob("relation_assert", row["relation_id"])
        return assertion_to_relation(fam, row, subject_entity_id=sub_abs,
                                     object_entity_id=obj_abs, embedding_blob=emb)

    def get_relation_versions(self, family_id: str) -> List[Relation]:
        conn = self._conn()
        fam = rel_repo.get_relation_family(conn, family_id)
        if not fam:
            return []
        rows = conn.execute(
            "SELECT * FROM relation_assertions "
            "WHERE relation_family_id = ? AND status != 'deleted' "
            "ORDER BY processed_at ASC",
            (family_id,),
        ).fetchall()
        relations = []
        for i, row in enumerate(rows, 1):
            row = dict(row)
            sub_abs = self._latest_obs_id_for_family(row["subject_entity_family_id"])
            obj_abs = self._latest_obs_id_for_family(row["object_entity_family_id"])
            emb = self._get_embedding_blob("relation_assert", row["relation_id"])
            relations.append(assertion_to_relation(fam, row, subject_entity_id=sub_abs,
                                                    object_entity_id=obj_abs,
                                                    embedding_blob=emb, version_seq=i))
        return relations

    def get_relation_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
        if not family_ids:
            return {}
        conn = self._conn()
        placeholders = ",".join("?" for _ in family_ids)
        rows = conn.execute(
            f"SELECT relation_family_id, COUNT(*) as cnt "
            f"FROM relation_assertions "
            f"WHERE relation_family_id IN ({placeholders}) AND status != 'deleted' "
            f"GROUP BY relation_family_id",
            family_ids,
        ).fetchall()
        result = {fid: 0 for fid in family_ids}
        for row in rows:
            result[row[0]] = row[1]
        return result

    def get_relations_by_entities(self, from_family_id: str, to_family_id: str,
                                  include_candidates: bool = False) -> List[Relation]:
        conn = self._conn()
        fam = rel_repo.find_relation_family(conn, from_family_id, to_family_id)
        if not fam:
            return []
        rows = conn.execute(
            "SELECT * FROM relation_assertions "
            "WHERE relation_family_id = ? AND status = 'active' "
            "ORDER BY processed_at DESC",
            (fam["relation_family_id"],),
        ).fetchall()
        relations = []
        for row in rows:
            row = dict(row)
            sub_abs = self._latest_obs_id_for_family(row["subject_entity_family_id"])
            obj_abs = self._latest_obs_id_for_family(row["object_entity_family_id"])
            emb = self._get_embedding_blob("relation_assert", row["relation_id"])
            relations.append(assertion_to_relation(fam, row, subject_entity_id=sub_abs,
                                                    object_entity_id=obj_abs, embedding_blob=emb))
        return relations

    def get_relations_by_entity_pairs(self, entity_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], List[Relation]]:
        result = {}
        for pair in entity_pairs:
            result[pair] = self.get_relations_by_entities(pair[0], pair[1])
        return result

    def get_relations_by_family_ids(self, family_ids: List[str], limit: int = 100,
                                    time_point: str = None,
                                    include_candidates: bool = False) -> List[Relation]:
        if not family_ids:
            return []
        conn = self._conn()
        placeholders = ",".join("?" for _ in family_ids)
        # Find relation_families where subject or object is in family_ids
        rows = conn.execute(
            f"SELECT DISTINCT rf.relation_family_id "
            f"FROM relation_families rf "
            f"WHERE rf.subject_entity_family_id IN ({placeholders}) "
            f"   OR rf.object_entity_family_id IN ({placeholders})",
            family_ids + family_ids,
        ).fetchall()
        rel_fids = [row[0] for row in rows]
        relations = []
        for fid in rel_fids[:limit]:
            rel = self.get_relation_by_family_id(fid)
            if rel:
                relations.append(rel)
        return relations

    def get_entity_relations_by_family_id(self, family_id: str, limit: int = 100,
                                          **_ignored) -> List[Relation]:
        return self.get_relations_by_family_ids([family_id], limit=limit)

    def get_entity_relations(self, entity_id: str, limit: int = 100, **_ignored) -> List[Relation]:
        conn = self._conn()
        row = conn.execute(
            "SELECT entity_family_id FROM entity_observations WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
        if not row:
            return []
        return self.get_entity_relations_by_family_id(row[0], limit=limit)

    def count_entity_relations_by_family_ids(self, family_ids: List[str]) -> Dict[str, int]:
        if not family_ids:
            return {}
        conn = self._conn()
        placeholders = ",".join("?" for _ in family_ids)
        rows = conn.execute(
            f"SELECT entity_family_id, COUNT(*) as cnt FROM ("
            f"  SELECT rf.subject_entity_family_id AS entity_family_id "
            f"  FROM relation_families rf "
            f"  WHERE rf.subject_entity_family_id IN ({placeholders}) "
            f"  UNION ALL "
            f"  SELECT rf.object_entity_family_id AS entity_family_id "
            f"  FROM relation_families rf "
            f"  WHERE rf.object_entity_family_id IN ({placeholders})"
            f") GROUP BY entity_family_id",
            family_ids + family_ids,
        ).fetchall()
        result = {fid: 0 for fid in family_ids}
        for row in rows:
            result[row[0]] = row[1]
        return result

    def get_relation_embeddings(self, family_ids: List[str]) -> Dict[str, Any]:
        result = {}
        for fid in family_ids:
            rel = self.get_relation_by_family_id(fid)
            if rel and rel.embedding:
                result[fid] = np.frombuffer(rel.embedding, dtype=np.float32)
        return result

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_concepts_by_bm25(self, query: str, role: str = None,
                                limit: int = 20, time_point: str = None,
                                source_document: str = None) -> List[dict]:
        raw = search_repo.search_fts(self._conn(), query, limit=limit)
        if raw:
            if len(raw) == 1:
                raw[0]["_score"] = 0.5
            else:
                scores = [r.get("score", 0) for r in raw]
                min_s = min(scores)
                max_s = max(scores)
                span = max_s - min_s
                for r in raw:
                    r["_score"] = (r.get("score", 0) - min_s) / span
        return raw

    def search_entities_by_bm25(self, query: str, limit: int = 20,
                                time_point: str = None) -> List[Entity]:
        results = search_repo.search_fts(self._conn(), query, limit=limit)
        # Normalize BM25 scores (FTS5 returns negative, more negative = more relevant)
        if results:
            scores = [r.get("score", 0) for r in results]
            min_s, max_s = min(scores), max(scores)
            span = max_s - min_s
            for r in results:
                r["_score"] = (r.get("score", 0) - min_s) / span if span else 0.5
        entities = []
        for r in results:
            ep_id = r.get("episode_id")
            if not ep_id:
                continue
            conn = self._conn()
            obs = conn.execute(
                "SELECT eo.*, ef.canonical_name, ef.canonical_content "
                "FROM entity_observations eo "
                "JOIN entity_families ef ON ef.entity_family_id = eo.entity_family_id "
                "WHERE eo.episode_id = ? AND eo.status = 'active'",
                (ep_id,),
            ).fetchone()
            if obs:
                obs = dict(obs)
                fam = {"entity_family_id": obs["entity_family_id"],
                       "canonical_name": obs["canonical_name"],
                       "canonical_content": obs["canonical_content"]}
                e = observation_to_entity(fam, obs)
                e._score = r.get("_score", 0.0)
                entities.append(e)
        return entities

    def search_relations_by_bm25(self, query: str, limit: int = 20,
                                 time_point: str = None) -> List[Relation]:
        results = search_repo.search_fts(self._conn(), query, limit=limit)
        # Normalize BM25 scores (FTS5 returns negative, more negative = more relevant)
        if results:
            scores = [r.get("score", 0) for r in results]
            min_s, max_s = min(scores), max(scores)
            span = max_s - min_s
            for r in results:
                r["_score"] = (r.get("score", 0) - min_s) / span if span else 0.5
        relations = []
        for r in results:
            ep_id = r.get("episode_id")
            if not ep_id:
                continue
            conn = self._conn()
            ra = conn.execute(
                "SELECT ra.* FROM relation_assertions ra "
                "WHERE ra.episode_id = ? AND ra.status = 'active'",
                (ep_id,),
            ).fetchone()
            if ra:
                ra = dict(ra)
                fam = rel_repo.get_relation_family(conn, ra["relation_family_id"])
                if fam:
                    rel = assertion_to_relation(fam, ra)
                    rel._pending_patches = []
                    rel._score = r.get("_score", 0.0)
                    relations.append(rel)
        return relations

    def search_entities_by_similarity(self, query_text: str, threshold: float = 0.3,
                                      max_results: int = 20, **kwargs) -> List[Entity]:
        if not self.embedding_client or not self.embedding_client.is_available():
            return []
        result = _encode_and_normalize(self.embedding_client, query_text)
        if not result:
            return []
        query_vec, query_nd = result
        candidates = emb_repo.search_entity_embeddings(
            self._conn(), query_vec,
            embedding_model=getattr(self.embedding_client, 'model_name', ''),
            limit=max_results * 3,
        )
        scored = []
        for c in candidates:
            vec = np.frombuffer(c["vector"], dtype=np.float32)
            sim = float(np.dot(query_nd, vec))
            if sim >= threshold:
                scored.append((sim, c))
        scored.sort(key=lambda x: -x[0])
        entities = []
        for sim, c in scored[:max_results]:
            conn = self._conn()
            obs = conn.execute(
                "SELECT eo.*, ef.canonical_name, ef.canonical_content "
                "FROM entity_observations eo "
                "JOIN entity_families ef ON ef.entity_family_id = eo.entity_family_id "
                "WHERE eo.entity_id = ? AND eo.status = 'active'",
                (c["owner_id"],),
            ).fetchone()
            if obs:
                obs = dict(obs)
                fam = {"entity_family_id": obs["entity_family_id"],
                       "canonical_name": obs["canonical_name"],
                       "canonical_content": obs["canonical_content"]}
                e = observation_to_entity(fam, obs, embedding_blob=c["vector"])
                e._score = sim
                entities.append(e)
        return entities

    def search_relations_by_similarity(self, query_text: str, threshold: float = 0.3,
                                       max_results: int = 20, **kwargs) -> List[Relation]:
        if not self.embedding_client or not self.embedding_client.is_available():
            return []
        result = _encode_and_normalize(self.embedding_client, query_text)
        if not result:
            return []
        query_vec, query_nd = result
        candidates = emb_repo.search_episode_embeddings(
            self._conn(), query_vec,
            embedding_model=getattr(self.embedding_client, 'model_name', ''),
            limit=max_results * 3,
        )
        scored = []
        for c in candidates:
            vec = np.frombuffer(c["vector"], dtype=np.float32)
            sim = float(np.dot(query_nd, vec))
            if sim >= threshold:
                scored.append((sim, c))
        scored.sort(key=lambda x: -x[0])
        relations = []
        for sim, c in scored[:max_results]:
            rel = self.get_relation_by_absolute_id(c["owner_id"])
            if rel:
                rel._pending_patches = []
                relations.append(rel)
        return relations

    def search_concepts_by_similarity(self, query_text: str, role: str = None,
                                      threshold: float = 0.3, max_results: int = 20,
                                      **kwargs) -> List[dict]:
        results = []
        if role is None or role == "entity":
            for e in self.search_entities_by_similarity(query_text, threshold, max_results):
                results.append({
                    "family_id": e.family_id, "id": e.absolute_id,
                    "name": e.name, "content": e.content,
                    "role": "entity", "_score": getattr(e, "_score", 0.0),
                })
        if role is None or role == "relation":
            for r in self.search_relations_by_similarity(query_text, threshold, max_results):
                results.append({
                    "family_id": r.family_id, "id": r.absolute_id,
                    "name": "", "content": r.content,
                    "entity1_name": "", "entity2_name": "",
                    "role": "relation", "_score": getattr(r, "_score", 0.0),
                })
        return results

    def suggest_concepts(self, query: str, role: str = "entity", limit: int = 10,
                         source_document: str = None) -> List[dict]:
        entities = self.search_entities_by_similarity(query, threshold=0.3, max_results=limit)
        return [{"family_id": e.family_id, "name": e.name, "relevance": e._score, "role": "entity"}
                for e in entities[:limit]]

    # ------------------------------------------------------------------
    # Concept unified API (server compatibility)
    # ------------------------------------------------------------------

    def get_concept_by_family_id(self, family_id: str, time_point: str = None) -> Optional[dict]:
        # Try entity
        fam = ent_repo.get_entity_family(self._conn(), family_id)
        if fam:
            obs = self._conn().execute(
                "SELECT * FROM entity_observations "
                "WHERE entity_family_id = ? AND status = 'active' "
                "ORDER BY processed_at DESC LIMIT 1",
                (family_id,),
            ).fetchone()
            result = dict(fam) if fam else None
            if result:
                result["role"] = "entity"
                result["family_id"] = result["entity_family_id"]
                result["name"] = result["canonical_name"]
                if obs:
                    result["content"] = dict(obs).get("content", "")
                return result
        # Try relation
        fam = rel_repo.get_relation_family(self._conn(), family_id)
        if fam:
            result = dict(fam)
            result["role"] = "relation"
            result["family_id"] = result["relation_family_id"]
            return result
        return None

    def get_concepts_by_family_ids(self, family_ids: Iterable[str],
                                   time_point: str = None) -> Dict[str, dict]:
        result = {}
        for fid in family_ids:
            c = self.get_concept_by_family_id(fid, time_point=time_point)
            if c:
                result[fid] = c
        return result

    def list_concepts(self, role: str = None, limit: int = 50, offset: int = 0,
                      time_point: str = None, name: str = None) -> List[dict]:
        if role == "entity":
            return [{"role": "entity", "family_id": r["entity_family_id"],
                      "name": r["canonical_name"]}
                     for r in ent_repo.list_entity_families(self._conn(), limit=limit, offset=offset)]
        elif role == "relation":
            return [{"role": "relation", "family_id": r["relation_family_id"]}
                     for r in rel_repo.list_relation_families(self._conn(), limit=limit, offset=offset)]
        else:
            ents = self.list_concepts(role="entity", limit=limit, offset=offset)
            rels = self.list_concepts(role="relation", limit=limit, offset=offset)
            return ents + rels

    def count_concepts(self, role: str = None, time_point: str = None,
                       name: str = None) -> int:
        if role == "entity":
            return self.count_unique_entities()
        elif role == "relation":
            return self.count_unique_relations()
        elif role == "episode":
            return self.count_episodes()
        return self.count_unique_entities() + self.count_unique_relations()

    def get_concept_versions(self, family_id: str) -> List[dict]:
        entities = self.get_entity_versions(family_id)
        versions = []
        for i, e in enumerate(entities):
            versions.append({
                "absolute_id": e.absolute_id,
                "family_id": e.family_id,
                "name": e.name,
                "content": e.content,
                "processed_time": e.processed_time.isoformat() if e.processed_time else None,
                "source_document": e.source_document or "",
                "episode_id": e.episode_id or "",
                "content_changed": i == 0 or (e.content or "") != (entities[i - 1].content or ""),
            })
        return versions

    def get_concept_provenance(self, family_id: str, time_point: str = None) -> List[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT em.episode_id, em.surface_text, em.start_offset, em.end_offset "
            "FROM entity_mentions em "
            "WHERE em.entity_family_id = ?",
            (family_id,),
        ).fetchall()
        return [{"edge_type": "MENTIONS", "episode_id": r[0],
                 "evidence": {"surface_text": r[1], "start_offset": r[2], "end_offset": r[3]}}
                for r in rows]

    def get_concept_mentions(self, family_id: str, time_point: str = None) -> List[dict]:
        return self.get_concept_provenance(family_id, time_point=time_point)

    def get_concept_neighbors(self, family_id: str, max_depth: int = 1,
                              time_point: str = None, edge_types: Optional[List[str]] = None,
                              max_results: int = 200) -> List[dict]:
        from .graph_traversal import get_concept_neighbors
        return get_concept_neighbors(self._conn(), family_id, max_depth=max_depth,
                                     max_results=max_results, edge_types=edge_types)

    def traverse_concepts(self, start_family_ids: List[str], max_depth: int = 2,
                          time_point: str = None, edge_types: Optional[List[str]] = None,
                          max_results: int = 500, _timeout_seconds: float = 30.0) -> dict:
        from .graph_traversal import traverse_concepts
        return traverse_concepts(self._conn(), start_family_ids, max_depth=max_depth,
                                 max_results=max_results, edge_types=edge_types,
                                 timeout_seconds=_timeout_seconds)

    def batch_get_entity_degrees(self, family_ids: List[str]) -> Dict[str, int]:
        return self.count_entity_relations_by_family_ids(family_ids)

    def get_episode_concepts(self, episode_id: str) -> List[dict]:
        mentions = ent_repo.get_mentions_by_episode(self._conn(), episode_id)
        return [{"family_id": m["entity_family_id"], "role": "entity",
                 "name": m.get("surface_text", "")}
                for m in mentions]

    def update_concept_manual(self, family_id: str, updates: dict) -> dict:
        conn = self._conn()
        fam = ent_repo.get_entity_family(conn, family_id)
        if not fam:
            return {"updated": False, "reason": "not found"}
        name = updates.get("name", fam["canonical_name"])
        content = updates.get("content", fam.get("canonical_content", ""))
        ent_repo.upsert_entity_family(conn, family_id, name, content,
                                       updated_at=_now_str())
        conn.commit()
        return {"updated": True, "family_id": family_id}

    def find_duplicate_entities_fast(self, limit: int = 500) -> List[dict]:
        return []

    # ------------------------------------------------------------------
    # Document graph rendering
    # ------------------------------------------------------------------

    def get_document_graph(self, document_version_ids: List[str] = None,
                           document_family_ids: List[str] = None,
                           include_relations: bool = True,
                           include_versions: bool = True,
                           max_episodes: int = 500,
                           max_concepts: int = 1000) -> dict:
        from .graph_traversal import get_document_graph
        return get_document_graph(self._conn(), document_version_ids, document_family_ids)

    def get_document_graph_outline(self, document_version_ids: List[str] = None,
                                    document_family_ids: List[str] = None,
                                    max_episodes: int = 10000) -> dict:
        from .graph_traversal import get_document_graph_outline
        return get_document_graph_outline(self._conn(), document_version_ids, document_family_ids)

    def get_document_graph_chunk(self, document_version_ids: List[str] = None,
                                  document_family_ids: List[str] = None,
                                  cursor: int = 0, limit: int = 12,
                                  include_relations: bool = True,
                                  include_versions: bool = True,
                                  max_concepts: int = 8000) -> dict:
        from .graph_traversal import get_document_graph_chunk
        return get_document_graph_chunk(self._conn(), document_version_ids,
                                         document_family_ids, cursor, limit)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "documents": self.count_documents(),
            "episodes": self.count_episodes(),
            "entities": self.count_unique_entities(),
            "relations": self.count_unique_relations(),
            "concepts": self.count_unique_entities() + self.count_unique_relations(),
        }

    def get_graph_statistics(self) -> dict:
        return self.get_stats()

    def count_unique_entities(self) -> int:
        return self._conn().execute(
            "SELECT COUNT(*) FROM entity_families"
        ).fetchone()[0]

    def count_unique_relations(self) -> int:
        return self._conn().execute(
            "SELECT COUNT(*) FROM relation_families"
        ).fetchone()[0]

    def count_isolated_entities(self) -> int:
        conn = self._conn()
        return conn.execute(
            "SELECT COUNT(*) FROM entity_families ef "
            "WHERE NOT EXISTS (SELECT 1 FROM entity_mentions em WHERE em.entity_family_id = ef.entity_family_id) "
            "AND NOT EXISTS (SELECT 1 FROM relation_families rf WHERE rf.subject_entity_family_id = ef.entity_family_id OR rf.object_entity_family_id = ef.entity_family_id)"
        ).fetchone()[0]

    # ------------------------------------------------------------------
    # Redirect / merge / delete (stubs — delegate to merge.py)
    # ------------------------------------------------------------------

    def resolve_family_id(self, family_id: str) -> str:
        from .merge import resolve_family_id
        return resolve_family_id(self._conn(), family_id)

    def resolve_family_ids(self, family_ids: Iterable[str]) -> Dict[str, str]:
        from .merge import resolve_family_ids
        return resolve_family_ids(self._conn(), family_ids)

    def register_entity_redirect(self, source_id: str, target_id: str):
        from .merge import register_redirect
        register_redirect(self._conn(), source_id, target_id)

    def register_entity_redirects_batch(self, redirects: Dict[str, str]):
        from .merge import register_redirects_batch
        register_redirects_batch(self._conn(), redirects)

    def merge_entity_families(self, target_family_id: str,
                              source_family_ids: List[str],
                              skip_name_check: bool = False) -> Dict[str, Any]:
        from .merge import merge_entity_families
        return merge_entity_families(self._conn(), target_family_id, source_family_ids)

    def redirect_entity_relations(self, old_family_id: str, new_family_id: str):
        from .merge import redirect_entity_relations
        redirect_entity_relations(self._conn(), old_family_id, new_family_id)

    def delete_entity_all_versions(self, family_id: str) -> int:
        from .merge import delete_entity_all_versions
        return delete_entity_all_versions(self._conn(), family_id)

    def dedup_merge_batch(self, pairs: List[Tuple[str, str]]) -> int:
        from .merge import dedup_merge_batch
        return dedup_merge_batch(self._conn(), pairs)

    # ------------------------------------------------------------------
    # Vault indexing (stubs — delegate to vault_indexer.py)
    # ------------------------------------------------------------------

    def index_vault(self, path: str, force: bool = False) -> dict:
        from .vault_indexer import index_vault
        return index_vault(self._conn(), self.library_path, path, force=force)

    def index_markdown_file(self, path: str, vault_root: str = "",
                            force: bool = False) -> dict:
        from .vault_indexer import index_markdown_file
        return index_markdown_file(self._conn(), self.library_path, path,
                                    vault_root=vault_root, force=force)

    @staticmethod
    def parse_markdown(text: str) -> dict:
        from .vault_indexer import parse_markdown
        return parse_markdown(text)

    @staticmethod
    def split_markdown_episodes(text: str, window_size: int = 4000,
                                overlap: int = 200) -> list:
        from ...text_chunking import split_markdown_chunks
        return split_markdown_chunks(text, window_size=window_size, overlap=overlap)

    # ------------------------------------------------------------------
    # Agent query (stub — delegate to agent_query.py)
    # ------------------------------------------------------------------

    def read_sql(self, sql: str, params: Any = None, *, limit: int = 200,
                 timeout_seconds: float = 5.0, include_query_plan: bool = False) -> dict:
        from .agent_query import execute_readonly_query
        return execute_readonly_query(self._conn(), sql, params, limit=limit,
                                       timeout_seconds=timeout_seconds,
                                       include_query_plan=include_query_plan)

    def agent_semantic_search(self, query: str, *, role: str = None,
                              top_k: int = 20, threshold: float = 0.3,
                              source_document: str = None) -> dict:
        results = self.search_concepts_by_similarity(query, role=role,
                                                      threshold=threshold, max_results=top_k)
        # Fallback: when embedding search returns no results (e.g. no
        # embedding client available), try a LIKE-based name lookup.
        if not results and (role is None or role == "entity"):
            conn = self._conn()
            like = f"%{query}%"
            rows = conn.execute(
                "SELECT ef.entity_family_id, ef.canonical_name, ef.canonical_content "
                "FROM entity_families ef "
                "WHERE ef.canonical_name LIKE ? "
                "ORDER BY ef.updated_at DESC LIMIT ?",
                (like, top_k),
            ).fetchall()
            for row in rows:
                e = Entity(
                    absolute_id="",
                    family_id=row[0],
                    name=row[1],
                    content=row[2] or "",
                    event_time=_now(),
                    processed_time=_now(),
                    episode_id="",
                    source_document="",
                )
                e._score = threshold * 0.95
                results.append(e)
        return {"results": results, "total": len(results)}

    # ------------------------------------------------------------------
    # Agent query (prewarm)
    # ------------------------------------------------------------------

    def prewarm_vector_search(self, roles: Optional[List[str]] = None):
        pass

    def get_entity_by_absolute_id(self, absolute_id: str) -> Optional[Entity]:
        """Get single entity by absolute_id (observation ID)."""
        conn = self._conn()
        obs = conn.execute(
            "SELECT eo.*, ef.entity_family_id, ef.canonical_name, ef.canonical_content "
            "FROM entity_observations eo "
            "JOIN entity_families ef ON ef.entity_family_id = eo.entity_family_id "
            "WHERE eo.entity_id = ? AND eo.status = 'active'",
            (absolute_id,),
        ).fetchone()
        if not obs:
            return None
        obs = dict(obs)
        fam = {"entity_family_id": obs["entity_family_id"],
               "canonical_name": obs["canonical_name"],
               "canonical_content": obs["canonical_content"]}
        emb = self._get_embedding_blob("entity_obs", absolute_id)
        return observation_to_entity(fam, obs, embedding_blob=emb)

    def get_relations_by_entity_absolute_ids(self, absolute_ids: List[str],
                                              limit: int = 100) -> List[Relation]:
        """Get relations involving entities with the given absolute IDs."""
        fam_map = self.get_family_ids_by_absolute_ids(absolute_ids)
        family_ids = list(set(fam_map.values()))
        return self.get_relations_by_family_ids(family_ids, limit=limit)

    def get_entity_absolute_ids_up_to_version(self, family_id: str,
                                               max_version_seq: int) -> List[str]:
        """Get entity observation IDs up to a given version sequence."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT entity_id FROM entity_observations "
            "WHERE entity_family_id = ? AND status = 'active' "
            "ORDER BY processed_at ASC LIMIT ?",
            (family_id, max_version_seq),
        ).fetchall()
        return [row[0] for row in rows]

    # ------------------------------------------------------------------
    # No-op stubs (same as old manager)
    # ------------------------------------------------------------------

    def save_content_patches(self, patches):
        return 0

    def adjust_confidence_on_corroboration(self, family_id: str, source_type: str = "entity",
                                           **_ignored):
        pass

    def adjust_confidence_on_corroboration_batch(self, family_ids: List[str],
                                                  source_type: str = "entity", **_ignored):
        pass

    def adjust_confidence_on_contradiction(self, family_id: str, source_type: str = "entity"):
        pass

    def refresh_relates_to_edges(self, family_ids: List[str] = None):
        pass

    def get_data_quality_report(self) -> dict:
        return {"issues": [], "warnings": [], "stats": self.get_stats()}

    def batch_get_source_text_snippets(self, episode_ids: List[str],
                                        snippet_length: int = 200) -> Dict[str, str]:
        if not episode_ids:
            return {}
        conn = self._conn()
        placeholders = ",".join("?" for _ in episode_ids)
        rows = conn.execute(
            f"SELECT episode_id, source_text FROM episodes WHERE episode_id IN ({placeholders})",
            episode_ids,
        ).fetchall()
        return {row[0]: (row[1] or "")[:snippet_length] for row in rows}

    def batch_bfs_traverse(self, seed_family_ids: List[str], max_depth: int = 2,
                           max_nodes: int = 50, time_point: str = None):
        from .graph_traversal import batch_bfs_traverse
        return batch_bfs_traverse(self._conn(), seed_family_ids, max_depth, max_nodes)

    def clear_graph_data(self):
        conn = self._conn()
        for table in ("entity_mentions", "relation_assertions", "relation_families",
                       "entity_observations", "entity_families", "entity_redirects",
                       "document_links", "embeddings", "pipeline_runs",
                       "episodes", "document_versions", "documents"):
            conn.execute(f"DELETE FROM {table}")
        conn.execute("INSERT INTO episodes_fts(episodes_fts) VALUES('rebuild')")
        conn.commit()

    def delete_graph_data(self):
        self.clear_graph_data()

    # ------------------------------------------------------------------
    # Write methods (pipeline-facing)
    # ------------------------------------------------------------------

    def save_episode(self, cache: Episode, text: str = "",
                     document_path: str = "", doc_hash: str = "",
                     start_offset: int = 0, end_offset = None) -> str:
        """Persist an Episode DTO and its source document."""
        import hashlib, uuid
        from . import content_fs

        conn = self._conn()
        text = text or cache.content
        source = cache.source_document or ""

        # Determine document identity from source or path
        source_key = document_path or source or text[:64]
        doc_id = f"doc_{hashlib.sha256(source_key.encode()).hexdigest()[:16]}"

        # Read full document content if available, otherwise fall back to text
        doc_text = text
        if document_path and Path(document_path).exists():
            doc_text = Path(document_path).read_text(encoding="utf-8")
        content_hash = content_fs.compute_content_hash(doc_text)

        # Ensure document exists
        doc = doc_repo.get_document(conn, doc_id)
        if not doc:
            title = source or Path(document_path).stem if document_path else ""
            safe_name = content_fs._safe_title(source) if source else ""
            content_md = f"content/{safe_name}.md" if safe_name else ""
            doc_repo.insert_document(
                conn, doc_id, title,
                managed_path=content_md,
                source_mode="managed" if source else "external",
                created_at=_now_str(), updated_at=_now_str(),
            )

        # Reuse existing version with same content hash, or create new one
        old_ver = doc_repo.get_active_version(conn, doc_id)
        if old_ver and old_ver.get("content_hash") == content_hash:
            ver_id = old_ver["document_version_id"]
        else:
            if old_ver:
                doc_repo.supersede_active_version_cascade(conn, doc_id)
            ver_id = f"docver_{doc_id}_{content_hash[:16]}"
            content_fs.write_version_snapshot(str(self.library_path), doc_id, content_hash, doc_text)
            doc_repo.insert_document_version(
                conn, ver_id, doc_id, content_hash,
                version_content_path=f"content/versions/{doc_id}/{content_hash}.md",
                title=source, char_count=len(doc_text), line_count=len(doc_text.splitlines()),
                byte_size=len(doc_text.encode("utf-8")),
                processed_at=_now_str(),
            )
            doc_repo.update_current_version(conn, doc_id, ver_id, updated_at=_now_str())

        # Create episode
        ep_id = cache.absolute_id or f"ep_{uuid.uuid4().hex[:16]}"
        ep_fam = f"epfam_{doc_id}_{doc_hash or ep_id}"
        # Compute next chunk_index for this version
        existing_chunks = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE document_version_id = ?",
            (ver_id,),
        ).fetchone()[0]
        ep_repo.insert_episode(
            conn, ep_id, ep_fam, doc_id, ver_id,
            source_text=text,
            memory_text=cache.content or "",
            start_offset=start_offset,
            end_offset=end_offset if end_offset is not None else len(text),
            chunk_index=existing_chunks,
            chunk_hash=doc_hash or content_hash[:16],
            name=source,
            event_time=_fmt_dt(cache.event_time) or _now_str(),
            processed_at=_fmt_dt(cache.processed_time) or _now_str(),
            activity_type=cache.activity_type or "",
            episode_type=cache.episode_type or "",
        )
        ep_repo.fts_sync_episode(conn, ep_id, doc_id, ver_id,
                                  name=source, source_text=text,
                                  memory_text=cache.content or "")
        self._commit_if_not_batched(conn)
        return doc_hash

    def save_entity(self, entity: Entity, _precomputed_embedding=None) -> None:
        """Persist an Entity DTO as entity_family + entity_observation."""
        import uuid
        conn = self._conn()
        fid = entity.family_id
        ent_repo.upsert_entity_family(
            conn, fid, entity.name, entity.content,
            created_at=_now_str(), updated_at=_now_str(),
        )
        # Resolve episode_id — null it if the referenced episode doesn't exist (FK safety)
        raw_ep_id = entity.episode_id or ""
        ep_id = raw_ep_id
        if ep_id:
            has_ep = conn.execute(
                "SELECT 1 FROM episodes WHERE episode_id = ?", (ep_id,)
            ).fetchone()
            if not has_ep:
                ep_id = None
        # Check for existing active observation for same episode+family (only when ep exists)
        if ep_id is not None:
            existing = ent_repo.get_active_observation(conn, ep_id, fid)
            if existing:
                return
        obs_id = entity.absolute_id or f"entobs_{uuid.uuid4().hex[:16]}"
        ent_repo.insert_entity_observation(
            conn, obs_id, fid, ep_id,
            name=entity.name, content=entity.content,
            processed_at=_fmt_dt(entity.processed_time) or _now_str(),
        )
        # Store embedding if available
        emb = _precomputed_embedding or entity.embedding
        if emb:
            self._store_embedding_if_available("entity_obs", obs_id, "content",
                                                entity.name or entity.content, emb)
        self._cache_entity_name(obs_id, entity.name)
        self._commit_if_not_batched(conn)

    def bulk_save_entities(self, entities: List[Entity]) -> None:
        with self._write_batch():
            for e in entities:
                self.save_entity(e)

    def bulk_save_entities_with_embedding(self, entities: List[Entity]) -> None:
        self.bulk_save_entities(entities)

    def save_relation(self, relation: Relation) -> None:
        """Persist a Relation DTO as relation_family + relation_assertion."""
        import uuid
        conn = self._conn()

        sub_fid = relation.entity1_family_id
        obj_fid = relation.entity2_family_id
        if not sub_fid or not obj_fid:
            # Look up family IDs from absolute IDs
            fam_map = self.get_family_ids_by_absolute_ids(
                [relation.entity1_absolute_id, relation.entity2_absolute_id]
            )
            sub_fid = sub_fid or fam_map.get(relation.entity1_absolute_id, "")
            obj_fid = obj_fid or fam_map.get(relation.entity2_absolute_id, "")

        # Ensure entity families exist
        for fid, name in [(sub_fid, ""), (obj_fid, "")]:
            if fid:
                fam = ent_repo.get_entity_family(conn, fid)
                if not fam:
                    ent_repo.upsert_entity_family(conn, fid, name, "",
                                                   created_at=_now_str(), updated_at=_now_str())

        # Upsert relation family
        fam = rel_repo.find_relation_family(conn, sub_fid, obj_fid)
        if not fam:
            rel_fid = relation.family_id or f"relfam_{uuid.uuid4().hex[:16]}"
            rel_repo.upsert_relation_family(
                conn, rel_fid, sub_fid, obj_fid,
                canonical_content=relation.content,
                created_at=_now_str(), updated_at=_now_str(),
            )
        else:
            rel_fid = fam["relation_family_id"]

        # Insert assertion — resolve episode_id for FK safety
        rel_ep_id = relation.episode_id or ""
        if rel_ep_id:
            has_ep = conn.execute(
                "SELECT 1 FROM episodes WHERE episode_id = ?", (rel_ep_id,)
            ).fetchone()
            if not has_ep:
                rel_ep_id = None
        sub_abs = relation.entity1_absolute_id
        obj_abs = relation.entity2_absolute_id
        if not sub_abs:
            sub_abs = self._latest_obs_id_for_family(sub_fid)
        if not obj_abs:
            obj_abs = self._latest_obs_id_for_family(obj_fid)

        ra_id = relation.absolute_id or f"rel_{uuid.uuid4().hex[:16]}"
        rel_repo.insert_relation_assertion(
            conn, ra_id, rel_fid, rel_ep_id,
            sub_abs, obj_abs, sub_fid, obj_fid,
            content=relation.content,
            processed_at=_fmt_dt(relation.processed_time) or _now_str(),
        )
        # Store embedding if available
        if relation.embedding:
            self._store_embedding_if_available("relation_assert", ra_id, "content",
                                                relation.content, relation.embedding)
        self._commit_if_not_batched(conn)

    def bulk_save_relations(self, relations: List[Relation]) -> None:
        with self._write_batch():
            for r in relations:
                self.save_relation(r)

    def bulk_save_relations_with_embedding(self, relations: List[Relation]) -> None:
        self.bulk_save_relations(relations)

    def save_episode_mentions(self, episode_id: str, entity_absolute_ids: List[str],
                              context: str = "", target_type: str = "entity") -> None:
        """Create entity_mention rows for entities mentioned in an episode."""
        import uuid
        from ...text_chunking import find_text_evidence
        conn = self._conn()

        # Get episode source text
        ep = ep_repo.get_episode(conn, episode_id)
        if not ep:
            return
        source_text = ep.get("source_text", "")
        start_offset = ep.get("start_offset", 0)

        # Build candidate list for text evidence
        candidates = []
        cand_info = {}  # name -> {family_id, absolute_id}
        for abs_id in entity_absolute_ids:
            fam = self.get_family_ids_by_absolute_ids([abs_id])
            fid = fam.get(abs_id, "")
            name = self._entity_name_cache.get(abs_id, "")
            if not name and fid:
                ef = ent_repo.get_entity_family(conn, fid)
                name = ef["canonical_name"] if ef else ""
            candidates.append(name)
            if name:
                cand_info[name] = {"family_id": fid, "absolute_id": abs_id}

        # Find text evidence per candidate to avoid limit truncation
        evidence_map = {}
        if source_text:
            for name, info in cand_info.items():
                hits = find_text_evidence(source_text, [name], base_offset=start_offset, limit=1)
                if hits:
                    evidence_map[info["absolute_id"]] = hits[0]

        import json
        for abs_id in entity_absolute_ids:
            fam_map = self.get_family_ids_by_absolute_ids([abs_id])
            fid = fam_map.get(abs_id, "")
            if not fid:
                continue

            ev = evidence_map.get(abs_id, {})
            mention_id = f"ment_{uuid.uuid4().hex[:16]}"
            ent_repo.insert_entity_mention(
                conn, mention_id, abs_id, fid, episode_id,
                surface_text=ev.get("quote", "") or ev.get("name", ""),
                start_offset=ev.get("start_offset", 0),
                end_offset=ev.get("end_offset", 0),
                line_start=ev.get("line_start", 0),
                line_end=ev.get("line_end", 0),
                created_at=_now_str(),
            )
        self._commit_if_not_batched(conn)

    def save_extraction_result(self, doc_hash: str, entities: list,
                               relations: list, document_path: str = "") -> bool:
        """Save extraction results to task extraction cache."""
        ep = self.find_cache_by_doc_hash(doc_hash, document_path=document_path)
        if not ep:
            return False
        cache_dir = self.extraction_cache_dir / ep.absolute_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "extraction.json").write_text(
            json.dumps({"entities": entities, "relations": relations}, ensure_ascii=False),
            encoding="utf-8",
        )
        return True

    def load_extraction_result(self, doc_hash: str,
                               document_path: str = "") -> Optional[tuple]:
        ep = self.find_cache_by_doc_hash(doc_hash, document_path=document_path)
        if not ep:
            return None
        path = self.extraction_cache_dir / ep.absolute_id / "extraction.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return (data.get("entities", []), data.get("relations", []))

    def find_cache_by_doc_hash(self, doc_hash: str,
                                document_path: str = "") -> Optional[Episode]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM episodes WHERE chunk_hash = ? AND status = 'active' "
            "ORDER BY processed_at DESC LIMIT 1",
            (doc_hash,),
        ).fetchone()
        if row:
            return episode_row_to_dto(dict(row))
        return None

    def find_cache_and_extraction_by_doc_hash(self, doc_hash: str,
                                               document_path: str = ""):
        ep = self.find_cache_by_doc_hash(doc_hash, document_path)
        if not ep:
            return None, None
        extraction = self.load_extraction_result(doc_hash, document_path)
        return ep, extraction

    def assess_remember_window_statuses(self, doc_hashes: List[str],
                                         document_path: str = "") -> List[dict]:
        results = []
        for idx, h in enumerate(doc_hashes):
            ep = self.find_cache_by_doc_hash(h, document_path)
            ext = self.load_extraction_result(h, document_path) if ep else None
            results.append({
                "doc_hash": h,
                "window_index": idx,
                "complete": ep is not None and ext is not None,
                "episode_exists": ep is not None,
                "extraction_exists": ext is not None,
            })
        return results

    def assess_document_integrity(self, document_version_id: str, *,
                                   window_hashes: List[str] = None) -> dict:
        conn = self._conn()
        ver = conn.execute(
            "SELECT document_id FROM document_versions WHERE document_version_id = ?",
            (document_version_id,),
        ).fetchone()
        if not ver:
            return {"complete": False, "missing_windows": []}
        ep_count = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE document_version_id = ? AND status = 'active'",
            (document_version_id,),
        ).fetchone()[0]
        return {"complete": True, "episode_count": ep_count, "missing_windows": []}

    def update_document_integrity_metadata(self, document_version_id: str,
                                            integrity: dict) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embedding_blob(self, owner_type: str, owner_id: str) -> Optional[bytes]:
        row = self._conn().execute(
            "SELECT vector FROM embeddings WHERE owner_type = ? AND owner_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (owner_type, owner_id),
        ).fetchone()
        return row[0] if row else None

    def _latest_obs_id_for_family(self, family_id: str) -> str:
        if not family_id:
            return ""
        row = self._conn().execute(
            "SELECT entity_id FROM entity_observations "
            "WHERE entity_family_id = ? AND status = 'active' "
            "ORDER BY processed_at DESC LIMIT 1",
            (family_id,),
        ).fetchone()
        return row[0] if row else ""

    def _vector_cache_for_role(self, role: str) -> dict:
        return {"matrix": None, "rows": []}

    def _document_version_for_episode(self, episode_id: str) -> str:
        row = self._conn().execute(
            "SELECT document_version_id FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        return row[0] if row else ""

    def _store_embedding_if_available(self, owner_type: str, owner_id: str,
                                       text_kind: str, text: str,
                                       embedding_blob: bytes) -> None:
        if not self.embedding_client or not self.embedding_client.is_available():
            return
        import hashlib as _hashlib
        conn = self._conn()
        text_hash = _hashlib.sha256((text or "").encode("utf-8")).hexdigest()
        model_name = getattr(self.embedding_client, 'model_name', 'unknown')
        dim = len(embedding_blob) // 4
        emb_repo.insert_embedding(
            conn, f"emb_{owner_id[:16]}", owner_type, owner_id,
            text_kind, text_hash, model_name, dim, embedding_blob,
            created_at=_now_str(),
        )

    def _compute_entity_embedding(self, entity: Entity):
        text = entity.name
        if entity.content:
            text = f"{entity.name}: {entity.content}"
        return _encode_and_normalize(self.embedding_client, text)

    def _compute_entity_embeddings_batch(self, entities: List[Entity]):
        results = []
        for e in entities:
            results.append(self._compute_entity_embedding(e))
        return results

    def _compute_relation_embedding(self, relation: Relation):
        return _encode_and_normalize(self.embedding_client, relation.content)

    def _compute_relation_embeddings_batch(self, relations: List[Relation]):
        results = []
        for r in relations:
            results.append(self._compute_relation_embedding(r))
        return results
