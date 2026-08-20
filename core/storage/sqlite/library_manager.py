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
from .dto_mapping import assertion_to_relation, episode_row_to_dto, observation_to_entity
from .helpers import _encode_and_normalize, _fmt_dt
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


def _escape_like(value: str) -> str:
    """Escape LIKE wildcard characters (%_) so they match literally.

    Uses '!' as the ESCAPE character to avoid backslash quoting issues
    in Python triple-quoted SQL strings.
    """
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


class LibraryManager:
    """V1.5 storage facade used by the remember pipeline and server."""

    def __init__(
        self,
        library_path: str = None,
        embedding_client=None,
        entity_content_snippet_length: int = 50,
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
            _ok = False
            try:
                yield conn
                _ok = True
            finally:
                self._thread_local.write_batch_depth = depth
                if depth == 0:
                    if _ok:
                        conn.commit()
                    else:
                        conn.rollback()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_entity_name(self, absolute_id: str, name: str):
        if absolute_id:
            self._entity_name_cache[absolute_id] = name or ""

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def set_document_ingestion_state(
        self, document_id: str, state: str, *, total_windows: int = 0,
        complete_windows: int = 0, missing_windows=None,
    ) -> None:
        """Atomically gate document visibility during remember publication."""
        import json as _json
        if state not in {"processing", "active", "failed", "incomplete"}:
            raise ValueError(f"Invalid ingestion state: {state}")
        conn = self._conn()
        conn.execute(
            """INSERT INTO document_ingestion_state
               (document_id,state,total_windows,complete_windows,missing_windows,updated_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(document_id) DO UPDATE SET
                 state=excluded.state,total_windows=excluded.total_windows,
                 complete_windows=excluded.complete_windows,missing_windows=excluded.missing_windows,
                 updated_at=excluded.updated_at""",
            (
                document_id, state, int(total_windows), int(complete_windows),
                _json.dumps(sorted(set(missing_windows or []))), _now_str(),
            ),
        )
        self._commit_if_not_batched(conn)

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
            esc = _escape_like(source_document)
            row = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE status = 'active' "
                "AND (title LIKE ? ESCAPE '!' OR managed_path LIKE ? ESCAPE '!' OR absolute_path LIKE ? ESCAPE '!')",
                (f"%{esc}%", f"%{esc}%", f"%{esc}%"),
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

    def _resolve_version(self, conn, identifier: str):
        """Resolve a document_id or document_version_id to a version row.

        Accepts either a ``document_version_id`` (e.g. ``docver_doc_...``)
        or a ``document_id`` (e.g. ``doc_...``).  For the latter, looks up
        ``current_version_id`` in the ``documents`` table first.
        """
        ver = conn.execute(
            "SELECT * FROM document_versions WHERE document_version_id = ?",
            (identifier,),
        ).fetchone()
        if ver:
            return ver
        # Maybe it's a document_id — resolve to current version
        doc = doc_repo.get_document(conn, identifier)
        if doc and doc.get("current_version_id"):
            ver = conn.execute(
                "SELECT * FROM document_versions WHERE document_version_id = ?",
                (doc["current_version_id"],),
            ).fetchone()
        return ver

    def get_document_content(self, document_version_id: str, *,
                             offset: int = 0, limit: int = 10_000_000) -> dict:
        conn = self._conn()
        ver = self._resolve_version(conn, document_version_id)
        if not ver:
            raise KeyError(document_version_id)
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
            "doc_id": ver.get("document_id", ""),
        }

    def get_document_file_info(self, document_version_id: str) -> dict:
        conn = self._conn()
        ver = self._resolve_version(conn, document_version_id)
        if not ver:
            return {}
        ver = dict(ver)
        doc = doc_repo.get_document(conn, ver["document_id"]) or {}
        return {
            "document_version_id": document_version_id,
            "document_id": ver.get("document_id"),
            "title": ver.get("title", ""),
            "source_mode": doc.get("source_mode", "managed"),
            "content_hash": ver.get("content_hash", ""),
            "char_count": ver.get("char_count", 0),
            "line_count": ver.get("line_count", 0),
            "byte_size": ver.get("byte_size", 0),
            "managed_path": doc.get("managed_path", ""),
            "absolute_path": doc.get("absolute_path", ""),
            "snapshot_path": ver.get("version_content_path", ""),
            "vault_root": doc.get("vault_root", ""),
            "relative_path": doc.get("relative_path", ""),
            "read_path": doc.get("managed_path", "") or ver.get("version_content_path", ""),
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
            # Collect assertion IDs BEFORE deleting (needed for embedding cleanup)
            rel_assert_ids_to_delete = [r[0] for r in conn.execute(
                f"SELECT relation_id FROM relation_assertions WHERE episode_id IN ({ph})", ep_ids
            ).fetchall()]
            # Delete relation_assertions linked to these episodes
            conn.execute(f"DELETE FROM relation_assertions WHERE episode_id IN ({ph})", ep_ids)
            # Collect observation IDs for embedding cleanup
            obs_ids_to_delete = [r[0] for r in conn.execute(
                f"SELECT entity_id FROM entity_observations WHERE episode_id IN ({ph})", ep_ids
            ).fetchall()]
            # Delete entity_observations linked to these episodes
            conn.execute(f"DELETE FROM entity_observations WHERE episode_id IN ({ph})", ep_ids)
            # Delete embeddings linked to these episodes
            conn.execute(f"DELETE FROM embeddings WHERE owner_type = 'episode' AND owner_id IN ({ph})", ep_ids)
            # Delete embeddings for the observations and assertions we just removed
            if obs_ids_to_delete:
                obs_ph = ",".join("?" for _ in obs_ids_to_delete)
                conn.execute(f"DELETE FROM embeddings WHERE owner_type = 'entity_obs' AND owner_id IN ({obs_ph})", obs_ids_to_delete)
            if rel_assert_ids_to_delete:
                rass_ph = ",".join("?" for _ in rel_assert_ids_to_delete)
                conn.execute(f"DELETE FROM embeddings WHERE owner_type = 'relation_assert' AND owner_id IN ({rass_ph})", rel_assert_ids_to_delete)

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
                        # Collect assertion IDs before deleting for embedding cleanup
                        blocked_assert_ids = [r[0] for r in conn.execute(
                            f"SELECT relation_id FROM relation_assertions WHERE relation_family_id IN ({rel_blocked_ph})",
                            list(rel_fams_blocked),
                        ).fetchall()]
                        conn.execute(f"DELETE FROM relation_assertions WHERE relation_family_id IN ({rel_blocked_ph})", list(rel_fams_blocked))
                        if blocked_assert_ids:
                            ba_ph = ",".join("?" for _ in blocked_assert_ids)
                            conn.execute(f"DELETE FROM embeddings WHERE owner_type = 'relation_assert' AND owner_id IN ({ba_ph})", blocked_assert_ids)
                        conn.execute(f"DELETE FROM relation_families WHERE relation_family_id IN ({rel_blocked_ph})", list(rel_fams_blocked))
                    # Collect entity observation IDs for these families before deleting
                    orphan_obs_ids = [r[0] for r in conn.execute(
                        f"SELECT entity_id FROM entity_observations WHERE entity_family_id IN ({del_ph})",
                        list(to_delete),
                    ).fetchall()] if to_delete else []
                    conn.execute(f"DELETE FROM entity_mentions WHERE entity_family_id IN ({del_ph})", list(to_delete))
                    conn.execute(f"DELETE FROM entity_observations WHERE entity_family_id IN ({del_ph})", list(to_delete))
                    if orphan_obs_ids:
                        oobs_ph = ",".join("?" for _ in orphan_obs_ids)
                        conn.execute(f"DELETE FROM embeddings WHERE owner_type = 'entity_obs' AND owner_id IN ({oobs_ph})", orphan_obs_ids)
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
                    # Collect assertion IDs before deleting for embedding cleanup
                    orphan_assert_ids = [r[0] for r in conn.execute(
                        f"SELECT relation_id FROM relation_assertions WHERE relation_family_id IN ({del_ph})",
                        list(to_delete),
                    ).fetchall()]
                    conn.execute(f"DELETE FROM relation_assertions WHERE relation_family_id IN ({del_ph})", list(to_delete))
                    if orphan_assert_ids:
                        oa_ph = ",".join("?" for _ in orphan_assert_ids)
                        conn.execute(f"DELETE FROM embeddings WHERE owner_type = 'relation_assert' AND owner_id IN ({oa_ph})", orphan_assert_ids)
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
            "  rf.object_entity_family_id, rf.canonical_content "
            "FROM relation_assertions ra "
            "JOIN relation_families rf ON rf.relation_family_id = ra.relation_family_id "
            "WHERE ra.relation_id = ?",
            (absolute_id,),
        ).fetchone()
        if not row:
            return None
        row = dict(row)
        fam = {k: row[k] for k in ("relation_family_id", "subject_entity_family_id",
                                     "object_entity_family_id", "canonical_content")}
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
        # 双向查询：管线的 pair key 是排序规范化的，存库时的 subject/object 方向
        # 可能与查询方向相反——只查单方向会让已有关系不可见，导致同一实体对
        # 重复建 relation family（正反两条）。
        fams = []
        fam = rel_repo.find_relation_family(conn, from_family_id, to_family_id)
        if fam:
            fams.append(fam)
        fam_rev = rel_repo.find_relation_family(conn, to_family_id, from_family_id)
        if fam_rev and (not fam or fam_rev["relation_family_id"] != fam["relation_family_id"]):
            fams.append(fam_rev)
        relations = []
        for fam in fams:
            rows = conn.execute(
                "SELECT * FROM relation_assertions "
                "WHERE relation_family_id = ? AND status = 'active' "
                "ORDER BY processed_at DESC",
                (fam["relation_family_id"],),
            ).fetchall()
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
            scores = [r.get("score", 0) for r in raw]
            min_s, max_s = min(scores), max(scores)
            span = max_s - min_s
            for r in raw:
                # FTS5 bm25() returns negative values; most negative = most relevant.
                # Invert so most relevant → 1.0.
                r["_score"] = (max_s - r.get("score", 0)) / span if span else 0.5
        return raw

    def search_entities_by_bm25(self, query: str, limit: int = 20,
                                time_point: str = None) -> List[Entity]:
        results = search_repo.search_fts(self._conn(), query, limit=limit)
        # Normalize BM25 scores (FTS5 returns negative, more negative = more relevant)
        # Invert so that most relevant → 1.0, least relevant → 0.0
        if results:
            scores = [r.get("score", 0) for r in results]
            min_s, max_s = min(scores), max(scores)
            span = max_s - min_s
            for r in results:
                r["_score"] = (max_s - r.get("score", 0)) / span if span else 0.5
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
        # Invert so that most relevant → 1.0, least relevant → 0.0
        if results:
            scores = [r.get("score", 0) for r in results]
            min_s, max_s = min(scores), max(scores)
            span = max_s - min_s
            for r in results:
                r["_score"] = (max_s - r.get("score", 0)) / span if span else 0.5
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
            embedding_model=(getattr(self.embedding_client, 'model_name', None)
                             or getattr(self.embedding_client, 'model_path', None)
                             or 'unknown'),
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
        candidates = emb_repo.search_relation_embeddings(
            self._conn(), query_vec,
            embedding_model=(getattr(self.embedding_client, 'model_name', None)
                             or getattr(self.embedding_client, 'model_path', None)
                             or 'unknown'),
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
        conn = self._conn()
        like = _escape_like(query) + "%"
        rows = conn.execute(
            "SELECT entity_family_id, canonical_name FROM entity_families "
            "WHERE canonical_name LIKE ? ESCAPE '!' "
            "ORDER BY updated_at DESC LIMIT ?",
            (like, limit),
        ).fetchall()
        return [{"family_id": r[0], "name": r[1], "relevance": 1.0, "role": "entity"}
                for r in rows]

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
                # Prefer canonical_content (manually updated) over
                # observation content so that `concept update --content`
                # is reflected in `concept get`.
                canon = result.get("canonical_content") or ""
                if canon:
                    result["content"] = canon
                elif obs:
                    result["content"] = dict(obs).get("content", "")
                # Extract confidence from latest active observation extra_json
                if obs:
                    from .dto_mapping import _extract_confidence
                    result["confidence"] = _extract_confidence(dict(obs).get("extra_json", "{}"))
                return result
        # Try relation
        fam = rel_repo.get_relation_family(self._conn(), family_id)
        if fam:
            result = dict(fam)
            result["role"] = "relation"
            result["family_id"] = result["relation_family_id"]
            # Extract confidence from latest active assertion extra_json
            assert_row = self._conn().execute(
                "SELECT extra_json FROM relation_assertions "
                "WHERE relation_family_id = ? AND status = 'active' "
                "ORDER BY processed_at DESC LIMIT 1",
                (family_id,),
            ).fetchone()
            if assert_row:
                from .dto_mapping import _extract_confidence
                result["confidence"] = _extract_confidence(assert_row[0])
            return result
        # Try episode (by episode_id / absolute_id)
        ep_row = ep_repo.get_episode(self._conn(), family_id)
        if ep_row:
            result = dict(ep_row)
            result["role"] = "episode"
            result["family_id"] = result.get("episode_family_id", family_id)
            result["absolute_id"] = result["episode_id"]
            result["uuid"] = result["episode_id"]
            result["name"] = result.get("heading_path", "") or result.get("name", "")
            result["content"] = result.get("memory_text", "")
            result["source_text"] = result.get("source_text", "")
            result["source_document"] = result.get("document_id", "")
            result["event_time"] = result.get("event_time")
            result["processed_time"] = result.get("processed_at")
            return result
        return None

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
            remaining = max(0, limit - len(ents))
            rels = self.list_concepts(role="relation", limit=remaining, offset=max(0, offset - len(ents))) if remaining > 0 else []
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
        # Episode: episodes don't have version chains, return single version
        ep = ep_repo.get_episode(self._conn(), family_id)
        if ep:
            return [{
                "absolute_id": ep["episode_id"],
                "family_id": ep.get("episode_family_id", family_id),
                "name": ep.get("heading_path", "") or ep.get("name", ""),
                "content": ep.get("memory_text", ""),
                "processed_time": ep.get("processed_at"),
                "source_document": ep.get("document_id", ""),
                "episode_id": ep["episode_id"],
                "content_changed": True,
            }]
        # Entity version chain
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
        """Return episodes mentioning a concept, with document metadata.

        Returns list of dicts with keys: episode_id, document_id, title,
        heading_path, line_start, line_end, source_text, surface_text.
        """
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT em.episode_id,
                   em.surface_text,
                   em.line_start,
                   em.line_end,
                   ep.document_id,
                   ep.heading_path,
                   ep.source_text,
                   d.title
            FROM entity_mentions em
            JOIN episodes ep ON ep.episode_id = em.episode_id
            JOIN documents d ON d.document_id = ep.document_id
            WHERE em.entity_family_id = ?
            ORDER BY ep.document_id, em.line_start
            """,
            (family_id,),
        ).fetchall()
        return [
            {
                "episode_id": r[0],
                "surface_text": r[1],
                "line_start": r[2],
                "line_end": r[3],
                "document_id": r[4],
                "heading_path": r[5],
                "source_text": r[6],
                "title": r[7],
            }
            for r in rows
        ]

    def get_concept_neighbors(self, family_id: str, max_depth: int = 1,
                              time_point: str = None, edge_types: Optional[List[str]] = None,
                              max_results: int = 200) -> List[dict]:
        # Episode absolute_id: return entity mentions + relation assertions
        ep_row = ep_repo.get_episode(self._conn(), family_id)
        if ep_row:
            return self._get_episode_neighbors(family_id)
        # Standard concept BFS
        from .graph_traversal import get_concept_neighbors
        return get_concept_neighbors(self._conn(), family_id, max_depth=max_depth,
                                     max_results=max_results, edge_types=edge_types)

    def _get_episode_neighbors(self, episode_id: str) -> List[dict]:
        """Return entity mentions and relation assertions for an episode."""
        neighbors = []
        # Entity mentions
        mentions = ent_repo.get_mentions_by_episode(self._conn(), episode_id)
        for m in mentions:
            neighbors.append({
                "family_id": m.get("entity_family_id", ""),
                "role": "entity",
                "absolute_id": m.get("entity_id", ""),
                "name": m.get("surface_text", ""),
                "target_type": "entity",
                "edge_type": "MENTIONS",
                "depth": 1,
            })
        # Relation assertions
        rows = self._conn().execute(
            "SELECT relation_id, relation_family_id, subject_entity_family_id, "
            "object_entity_family_id, content FROM relation_assertions "
            "WHERE episode_id = ? AND status = 'active'",
            (episode_id,)
        ).fetchall()
        for r in rows:
            neighbors.append({
                "family_id": r[1],
                "role": "relation",
                "absolute_id": r[0],
                "entity1_absolute_id": r[2],
                "entity2_absolute_id": r[3],
                "content": r[4],
                "target_type": "relation",
                "edge_type": "ASSERTS",
                "depth": 1,
            })
        return neighbors

    def traverse_concepts(self, start_family_ids: List[str], max_depth: int = 2,
                          time_point: str = None, edge_types: Optional[List[str]] = None,
                          max_results: int = 500, _timeout_seconds: float = 30.0) -> dict:
        from .graph_traversal import traverse_concepts
        return traverse_concepts(self._conn(), start_family_ids, max_depth=max_depth,
                                 max_results=max_results, edge_types=edge_types,
                                 timeout_seconds=_timeout_seconds)

    def batch_get_entity_degrees(self, family_ids: List[str]) -> Dict[str, int]:
        return self.count_entity_relations_by_family_ids(family_ids)

    def update_concept_manual(self, family_id: str, updates: dict) -> dict:
        import json as _json
        conn = self._conn()
        fam = ent_repo.get_entity_family(conn, family_id)
        if not fam:
            return {"updated": False, "reason": "not found"}
        name = updates.get("name", fam["canonical_name"])
        content = updates.get("content", fam.get("canonical_content", ""))
        ent_repo.upsert_entity_family(conn, family_id, name, content,
                                       updated_at=_now_str())
        # Update confidence on latest active observation if provided
        new_confidence = updates.get("confidence")
        if new_confidence is not None:
            obs_row = conn.execute(
                "SELECT entity_id, extra_json FROM entity_observations "
                "WHERE entity_family_id = ? AND status = 'active' "
                "ORDER BY processed_at DESC LIMIT 1",
                (family_id,),
            ).fetchone()
            if obs_row:
                obs_id, extra_json = obs_row[0], obs_row[1] or "{}"
                try:
                    extra = _json.loads(extra_json)
                except (ValueError, TypeError):
                    extra = {}
                extra["confidence"] = float(new_confidence)
                conn.execute(
                    "UPDATE entity_observations SET extra_json = ? WHERE entity_id = ?",
                    (_json.dumps(extra, ensure_ascii=False), obs_id),
                )
        conn.commit()
        return {"updated": True, "family_id": family_id}

    def find_duplicate_entities_fast(self, limit: int = 500) -> List[dict]:
        """Find entity families sharing the same canonical_name."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT ef1.entity_family_id AS family_a_id,
                   ef1.canonical_name    AS name_a,
                   ef2.entity_family_id AS family_b_id,
                   ef2.canonical_name    AS name_b
            FROM entity_families ef1
            JOIN entity_families ef2
              ON ef1.canonical_name = ef2.canonical_name
             AND ef1.entity_family_id < ef2.entity_family_id
            ORDER BY ef1.canonical_name
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "family_a_id": r[0],
                "name_a": r[1],
                "family_b_id": r[2],
                "name_b": r[3],
                "similarity": 1.0,  # exact name match
            }
            for r in rows
        ]

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
        return get_document_graph(self._conn(), document_version_ids,
                                  document_family_ids, max_episodes, max_concepts)

    def get_document_graph_outline(self, document_version_ids: List[str] = None,
                                    document_family_ids: List[str] = None,
                                    max_episodes: int = 10000) -> dict:
        from .graph_traversal import get_document_graph_outline
        return get_document_graph_outline(self._conn(), document_version_ids,
                                          document_family_ids, max_episodes)

    def get_document_graph_chunk(self, document_version_ids: List[str] = None,
                                  document_family_ids: List[str] = None,
                                  cursor: int = 0, limit: int = 12,
                                  include_relations: bool = True,
                                  include_versions: bool = True,
                                  max_concepts: int = 8000) -> dict:
        from .graph_traversal import get_document_graph_chunk
        return get_document_graph_chunk(self._conn(), document_version_ids,
                                         document_family_ids, cursor, limit,
                                         include_relations, include_versions,
                                         max_concepts)

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

    def count_unique_entities(self) -> int:
        return self._conn().execute(
            "SELECT COUNT(*) FROM entity_families"
        ).fetchone()[0]

    def count_unique_relations(self) -> int:
        return self._conn().execute(
            "SELECT COUNT(*) FROM relation_families"
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
        self._commit_if_not_batched()

    def delete_entity_all_versions(self, family_id: str) -> int:
        from .merge import delete_entity_all_versions
        result = delete_entity_all_versions(self._conn(), family_id)
        self._commit_if_not_batched()
        return result

    def dedup_merge_batch(self, pairs: List[Tuple[str, str]]) -> int:
        from .merge import dedup_merge_batch
        with self._write_batch():
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
            like = f"%{_escape_like(query)}%"
            rows = conn.execute(
                "SELECT ef.entity_family_id, ef.canonical_name, ef.canonical_content "
                "FROM entity_families ef "
                "WHERE ef.canonical_name LIKE ? ESCAPE '!' "
                "ORDER BY ef.updated_at DESC LIMIT ?",
                (like, top_k),
            ).fetchall()
            for row in rows:
                results.append({
                    "family_id": row[0],
                    "name": row[1],
                    "content": row[2] or "",
                    "role": "entity",
                    "_score": threshold * 0.95,
                })
        return {"results": results, "total": len(results)}

    # ------------------------------------------------------------------
    # Agent query (prewarm)
    # ------------------------------------------------------------------

    def prewarm_vector_search(self, roles: Optional[List[str]] = None):
        """Pre-load vector caches for the given roles (or ['entity'] by default).

        Called from pipeline_workers.py as a prefetch future and from
        registry.py in a background thread after server startup.
        """
        if roles is None:
            roles = ["entity"]
        warmed = {}
        for role in roles:
            try:
                cache = self._vector_cache_for_role(role)
                matrix = cache.get("matrix")
                warmed[role] = matrix.shape[0] if matrix is not None else 0
            except Exception as exc:
                logger.debug("prewarm_vector_search(%s) failed: %s", role, exc)
                warmed[role] = -1
        return {"warmed": warmed}

    def get_relations_by_entity_absolute_ids(self, absolute_ids: List[str],
                                              limit: int = 100) -> List[Relation]:
        """Get relations involving entities with the given absolute IDs."""
        fam_map = self.get_family_ids_by_absolute_ids(absolute_ids)
        family_ids = list(set(fam_map.values()))
        return self.get_relations_by_family_ids(family_ids, limit=limit)

    # ------------------------------------------------------------------
    # No-op stubs (same as old manager)
    # ------------------------------------------------------------------

    def save_content_patches(self, patches):
        return 0

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

    # ------------------------------------------------------------------
    # Write methods (pipeline-facing)
    # ------------------------------------------------------------------

    def save_episode(self, cache: Episode, text: str = "",
                     document_path: str = "", doc_hash: str = "",
                     start_offset: int = 0, end_offset = None,
                     override_doc_id: str = "",
                     heading_path: str = "",
                     episode_type: str = "",
                     run_id: str = "",
                     retrieval_slice_chars: int = 0) -> str:
        """Persist an Episode DTO and its source document."""
        import hashlib
        import uuid
        from . import content_fs

        conn = self._conn()
        text = text or cache.content
        source = cache.source_document or ""

        # Determine document identity from source or path
        if override_doc_id:
            doc_id = override_doc_id
        else:
            source_key = document_path or source or text[:64]
            doc_id = f"doc_{hashlib.sha256(source_key.encode()).hexdigest()[:16]}"

        # Read full document content if available, otherwise fall back to text
        doc_text = text
        if document_path and Path(document_path).exists():
            doc_text = Path(document_path).read_text(encoding="utf-8")
        content_hash = content_fs.compute_content_hash(doc_text)

        # Cross-document content-hash dedup: prevent duplicate documents when the
        # same content is submitted multiple times (e.g. re-uploading same file,
        # or text-submit with a fresh random UUID as source_key).
        # When override_doc_id is set (repair / targeted retry), skip dedup —
        # the caller already knows the correct doc_id.
        if not override_doc_id:
            _dedup_row = conn.execute(
                "SELECT dv.document_id FROM document_versions dv "
                "JOIN documents d ON d.document_id = dv.document_id "
                "WHERE dv.content_hash = ? AND dv.status = 'active' AND d.status = 'active' "
                "LIMIT 1",
                (content_hash,)
            ).fetchone()
            if _dedup_row and _dedup_row[0] != doc_id:
                doc_id = _dedup_row[0]

        # Ensure document exists
        doc = doc_repo.get_document(conn, doc_id)
        if not doc:
            title = source or Path(document_path).stem if document_path else ""
            content_md = content_fs.write_current_file(
                str(self.library_path), title or doc_id, doc_text, doc_id=doc_id,
            )
            doc_repo.insert_document(
                conn, doc_id, title,
                managed_path=content_md,
                source_mode="managed" if source else "external",
                created_at=_now_str(), updated_at=_now_str(),
            )
        else:
            # Document already exists (possibly via content-hash dedup):
            # use the existing document's title for episode name consistency.
            title = doc.get("title") or source
            content_md = content_fs.write_current_file(
                str(self.library_path), title or doc_id, doc_text, doc_id=doc_id,
            )
            conn.execute(
                "UPDATE documents SET managed_path = ?, status = 'active', updated_at = ? WHERE document_id = ?",
                (content_md, _now_str(), doc_id),
            )

        # Reuse existing version with same content hash, or create new one
        old_ver = doc_repo.get_active_version(conn, doc_id)
        if old_ver and old_ver.get("content_hash") == content_hash:
            ver_id = old_ver["document_version_id"]
        elif override_doc_id and old_ver:
            # Repair / targeted mode: reuse existing version to avoid
            # cascading supersede of existing episodes/entities/relations.
            # The repair text is a window fragment, not the full document,
            # so content_hash will always differ — but we must not replace
            # the version.
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
            heading_path=heading_path or getattr(cache, 'heading_path', '') or "",
            start_offset=start_offset,
            end_offset=end_offset if end_offset is not None else len(text),
            chunk_index=existing_chunks,
            chunk_hash=doc_hash or content_hash[:16],
            name=title,
            event_time=_fmt_dt(cache.event_time) or _now_str(),
            processed_at=_fmt_dt(cache.processed_time) or _now_str(),
            activity_type=cache.activity_type or "",
            episode_type=episode_type or getattr(cache, 'episode_type', '') or "",
            run_id=run_id,
        )
        ep_repo.fts_sync_episode(conn, ep_id, doc_id, ver_id,
                                  name=title, source_text=text,
                                  memory_text=cache.content or "")
        if retrieval_slice_chars and episode_type != "retrieval_slice":
            self._write_retrieval_slices(
                conn, text, doc_id, ver_id, ep_fam, title,
                heading_path=heading_path,
                base_chunk_index=existing_chunks,
                run_id=run_id, slice_chars=int(retrieval_slice_chars),
            )
        self._commit_if_not_batched(conn)
        return doc_hash

    def _write_retrieval_slices(self, conn, text: str, doc_id: str, ver_id: str,
                                ep_fam: str, title: str, *, heading_path: str,
                                base_chunk_index: int, run_id: str,
                                slice_chars: int) -> int:
        """在窗口 episode 之外追加"检索切片" episode 行（episode_type=retrieval_slice）。

        大窗口（strong-v1 6000 字）episode 因 FTS5 bm25 长度归一化处于劣势，
        其中的证据 turn 难以进入检索候选集；按 ~slice_chars 在对话行边界切出
        薄切片行作为纯 FTS 检索单元（memory_text 为空，不参与实体锚定语义）。
        窗口 episode 原样保留：实体锚定、版本链、溯源全部不动。
        行必须整行切分——检索层的 turn 解析依赖 "[turn_id] role: text" 行格式。

        Returns:
            写入的切片行数
        """
        import hashlib
        import uuid
        if slice_chars <= 0 or len(text or "") <= slice_chars:
            return 0
        lines = (text or "").splitlines()
        line_offsets, running = [], 0
        for line in lines:
            line_offsets.append(running)
            running += len(line) + 1
        slices: list[tuple[int, list[str]]] = []  # (start_line_index, lines)
        cur: list[str] = []
        cur_start = 0
        cur_len = 0
        for i, line in enumerate(lines):
            add = len(line) + 1
            if cur and cur_len + add > slice_chars:
                slices.append((cur_start, cur))
                cur, cur_start, cur_len = [], i, 0
            cur.append(line)
            cur_len += add
        if cur:
            slices.append((cur_start, cur))
        written = 0
        for n, (start_line, slice_lines) in enumerate(slices, 1):
            slice_text = "\n".join(slice_lines)
            if not slice_text.strip():
                continue
            start_off = line_offsets[start_line]
            slice_id = f"ep_{uuid.uuid4().hex[:16]}"
            slice_hash = hashlib.sha256(slice_text.encode("utf-8")).hexdigest()[:16]
            ep_repo.insert_episode(
                conn, slice_id, ep_fam, doc_id, ver_id,
                source_text=slice_text, memory_text="",
                heading_path=heading_path,
                start_offset=start_off,
                end_offset=start_off + len(slice_text),
                line_start=start_line,
                line_end=start_line + len(slice_lines) - 1,
                chunk_index=base_chunk_index + n,
                chunk_hash=slice_hash,
                name=title,
                episode_type="retrieval_slice",
                activity_type="",
                event_time=_now_str(), processed_at=_now_str(),
                run_id=run_id,
            )
            ep_repo.fts_sync_episode(conn, slice_id, doc_id, ver_id,
                                      name=title, source_text=slice_text,
                                      memory_text="")
            written += 1
        if written:
            logger.info("检索切片｜%s｜窗口 %d 字 → %d 片（≤%d 字/片）",
                        title, len(text or ""), written, slice_chars)
        return written

    def save_entity(self, entity: Entity, _precomputed_embedding=None,
                     run_id: str = "", extra_json: str = "") -> None:
        """Persist an Entity DTO as entity_family + entity_observation."""
        import uuid
        import json as _json
        # Auto-fill run_id from storage context if not explicitly provided
        if not run_id:
            run_id = getattr(self, '_current_run_id', '') or ''
        # Auto-fill extra_json from entity metadata if not explicitly provided
        if (not extra_json or extra_json == "{}") and hasattr(entity, 'confidence') and entity.confidence is not None:
            extra_json = _json.dumps({"confidence": entity.confidence})
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
                logger.warning("save_entity: episode_id %r not found in episodes table, "
                               "nullifying for entity %s (family=%s)",
                               raw_ep_id, entity.absolute_id, fid)
                ep_id = None
        # Normalize embedding to bytes: handle both raw bytes and (bytes, ndarray) tuples.
        _emb_raw = _precomputed_embedding or entity.embedding
        if isinstance(_emb_raw, (list, tuple)):
            _emb_raw = _emb_raw[0]  # extract bytes from (bytes, ndarray)

        # Check for existing active observation for same episode+family (only when ep exists)
        obs_id = entity.absolute_id or f"entobs_{uuid.uuid4().hex[:16]}"
        if ep_id is not None:
            existing = ent_repo.get_active_observation(conn, ep_id, fid)
            if existing:
                # Observation already exists — but we may still need to store the embedding
                # that was pre-computed during the pipeline batch step.
                if _emb_raw:
                    existing_obs_id = existing.get("entity_id", "") if isinstance(existing, dict) else str(existing)
                    self._store_embedding_if_available(
                        "entity_obs", existing_obs_id, "content",
                        entity.name or entity.content, _emb_raw)
                return
        ent_repo.insert_entity_observation(
            conn, obs_id, fid, ep_id,
            name=entity.name, content=entity.content,
            extra_json=extra_json or "{}",
            processed_at=_fmt_dt(entity.processed_time) or _now_str(),
            run_id=run_id,
        )
        # Store embedding if available
        if _emb_raw:
            self._store_embedding_if_available("entity_obs", obs_id, "content",
                                                entity.name or entity.content, _emb_raw)
        self._cache_entity_name(obs_id, entity.name)
        self._commit_if_not_batched(conn)

    def bulk_save_entities(self, entities: List[Entity]) -> None:
        with self._write_batch():
            for e in entities:
                self.save_entity(e)

    def save_relation(self, relation: Relation,
                       run_id: str = "", extra_json: str = "") -> None:
        """Persist a Relation DTO as relation_family + relation_assertion."""
        import uuid
        import json as _json
        # Auto-fill run_id from storage context if not explicitly provided
        if not run_id:
            run_id = getattr(self, '_current_run_id', '') or ''
        # Auto-fill extra_json from relation confidence if not explicitly provided
        if (not extra_json or extra_json == "{}") and hasattr(relation, 'confidence') and relation.confidence is not None:
            extra_json = _json.dumps({"confidence": relation.confidence})
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
                logger.warning("save_relation: episode_id %r not found in episodes table, "
                               "nullifying for relation %s (family=%s)",
                               rel_ep_id, relation.absolute_id, rel_fid)
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
            evidence_text=relation.evidence_text or "",
            evidence_start_offset=relation.evidence_start_offset or 0,
            evidence_end_offset=relation.evidence_end_offset or 0,
            evidence_line_start=relation.evidence_line_start or 0,
            evidence_line_end=relation.evidence_line_end or 0,
            extra_json=extra_json or "{}",
            processed_at=_fmt_dt(relation.processed_time) or _now_str(),
            run_id=run_id,
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
        conn = self._conn()
        results = []
        for idx, h in enumerate(doc_hashes):
            ep = self.find_cache_by_doc_hash(h, document_path)
            ext = self.load_extraction_result(h, document_path) if ep else None

            ep_exists = ep is not None
            ext_exists = ext is not None

            # Entity persistence check: if extraction found entities but
            # none were persisted (step9 crashed after step5), mark incomplete.
            # This catches the case where episode + extraction cache both exist
            # but entity alignment (step9) didn't finish.
            entities_complete = True
            if ep_exists and ext_exists:
                ep_id = getattr(ep, 'absolute_id', '')
                if ep_id:
                    # Count extracted entities from the extraction result
                    _ext_ents = 0
                    if isinstance(ext, (list, tuple)) and len(ext) > 0:
                        _ext_ents = len(ext[0]) if isinstance(ext[0], list) else 0
                    # Only verify DB if extraction found entities
                    if _ext_ents > 0:
                        _db_count = conn.execute(
                            "SELECT COUNT(*) FROM entity_observations WHERE episode_id = ?",
                            (ep_id,)
                        ).fetchone()[0]
                        if _db_count == 0:
                            entities_complete = False

            results.append({
                "doc_hash": h,
                "window_index": idx,
                "complete": ep_exists and ext_exists and entities_complete,
                "episode_exists": ep_exists,
                "extraction_exists": ext_exists,
                "entities_complete": entities_complete,
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
        """Return cached vector matrix for a role, loading on first access."""
        with self._vector_cache_lock:
            cached = self._vector_role_cache.get(role)
            if cached is not None:
                return cached
            result = self._build_vector_cache_for_role(role)
            self._vector_role_cache[role] = result
            return result

    # Role-to-SQL configuration for vector cache loading
    _VECTOR_ROLE_CONFIG = {
        "entity": {
            "owner_type": "entity_obs",
            "join_fragment": (
                "JOIN entity_observations eo ON eo.entity_id = e.owner_id AND eo.status = 'active'"
            ),
            "family_col": "eo.entity_family_id",
            "owner_col": "eo.entity_id",
            "dedup_fragment": (
                "SELECT 1 FROM entity_observations eo2 "
                "WHERE eo2.entity_family_id = eo.entity_family_id "
                "  AND eo2.status = 'active' AND eo2.rowid > eo.rowid"
            ),
        },
        "relation": {
            "owner_type": "relation_assert",
            "join_fragment": (
                "JOIN relation_assertions ra ON ra.relation_id = e.owner_id AND ra.status = 'active'"
            ),
            "family_col": "ra.relation_family_id",
            "owner_col": "ra.relation_id",
            "dedup_fragment": (
                "SELECT 1 FROM relation_assertions ra2 "
                "WHERE ra2.relation_family_id = ra.relation_family_id "
                "  AND ra2.status = 'active' AND ra2.rowid > ra.rowid"
            ),
        },
    }

    def _build_vector_cache_for_role(self, role: str) -> dict:
        """Load all embeddings for a role from SQLite into a numpy matrix.

        Returns {"matrix": np.ndarray(N,D), "rows": [{"family_id": str}, ...], "_loaded": True}
        or {"matrix": None, "rows": [], "_loaded": True} if no data / error.
        """
        import numpy as np

        config = self._VECTOR_ROLE_CONFIG.get(role)
        if config is None:
            return {"matrix": None, "rows": [], "_loaded": True}

        sql = (
            f"SELECT e.vector, {config['family_col']}, {config['owner_col']} "
            f"FROM embeddings e "
            f"{config['join_fragment']} "
            f"WHERE e.owner_type = ? "
            f"  AND NOT EXISTS ({config['dedup_fragment']})"
        )

        try:
            conn = self._conn()
            rows = conn.execute(sql, (config["owner_type"],)).fetchall()
        except Exception as exc:
            logger.debug("_build_vector_cache_for_role(%s) SQL failed: %s", role, exc)
            return {"matrix": None, "rows": [], "_loaded": True}

        if not rows:
            logger.debug("_build_vector_cache_for_role(%s): no embeddings found", role)
            return {"matrix": None, "rows": [], "_loaded": True}

        # Build rows metadata and matrix
        meta_rows = []
        vectors = []
        dim = None
        for row in rows:
            vec_bytes = row[0]
            if not vec_bytes or len(vec_bytes) < 4:
                continue
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            if dim is None:
                dim = vec.shape[0]
            elif vec.shape[0] != dim:
                continue  # skip dimension mismatches
            meta_rows.append({"family_id": row[1], "owner_id": row[2]})
            vectors.append(vec)

        if not vectors:
            return {"matrix": None, "rows": [], "_loaded": True}

        matrix = np.vstack(vectors)
        logger.info("_build_vector_cache_for_role(%s): loaded %d vectors, dim=%d",
                     role, matrix.shape[0], matrix.shape[1])
        return {"matrix": matrix, "rows": meta_rows, "_loaded": True}

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
            logger.debug("_store_embedding_if_available: embedding client not available, "
                         "skipping embedding for %s/%s", owner_type, owner_id)
            return
        import hashlib as _hashlib
        conn = self._conn()
        text_hash = _hashlib.sha256((text or "").encode("utf-8")).hexdigest()
        model_name = (getattr(self.embedding_client, 'model_name', None)
                      or getattr(self.embedding_client, 'model_path', None)
                      or 'unknown')
        dim = len(embedding_blob) // 4
        emb_repo.insert_embedding(
            conn, f"emb_{owner_id}", owner_type, owner_id,
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
