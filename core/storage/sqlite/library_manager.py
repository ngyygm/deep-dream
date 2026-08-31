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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ...models import Entity, Episode, Relation
from .dto_mapping import assertion_to_relation, episode_row_to_dto, observation_to_entity
from .helpers import (
    _encode_and_normalize,
    _fmt_dt,
    _parse_dt,
    _time_bounds_sql,
    escape_like,
    now_utc_str,
)
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


def _placeholders(values: List[str]) -> str:
    """生成 IN (...) 的占位符串（P3.5 批量查询用）。"""
    return ",".join("?" for _ in values)


def _in_clause_chunks(values: List[str], chunk_size: int = 400):
    """把 id 列表切成不超过 chunk_size 的段，规避 SQLite 变量数上限（P3.5）。"""
    for i in range(0, len(values), chunk_size):
        yield values[i:i + chunk_size]


class LibraryManager:
    """V1.5 storage facade used by the remember pipeline and server.

    返回契约（P4.6 文档化——只记录现状，不改任何返回类型）：

    - 返回 ``Entity`` DTO：``get_entity_by_family_id`` / ``get_all_entities`` /
      ``get_all_entities_before_time``、检索系的 ``search_entities_by_bm25`` /
      ``search_entities_by_similarity``。检索系 DTO 附带动态属性 ``_score``
      （BM25/余弦得分）与 ``embedding``（语义检索路径水合时才有值）。
    - 返回 ``Relation`` DTO：``get_relation_by_family_id`` /
      ``get_relations_by_family_ids`` / ``get_relations_by_entity_absolute_ids``、
      检索系的 ``search_relations_by_bm25`` / ``search_relations_by_similarity``。
    - 返回 ``dict``：概念统一面（``get_concept_by_family_id`` / ``list_concepts`` /
      ``suggest_concepts`` / ``search_concepts_by_bm25``）与文档面
      （``list_documents`` 等）。检索 dict 统一携带 ``family_id`` / ``role`` /
      ``_score`` 键；``agent_semantic_search`` 为
      ``{"results": List[dict], "total": int}`` 包装（语义检索单入口，P4.2）。
    - 做 ``.get()`` 前先判类型：DTO 与 dict 混用是已知坑（见 CLAUDE.md）。
    """

    def __init__(
        self,
        library_path: str = None,
        embedding_client=None,
        entity_content_snippet_length: int = 50,
        # Old compat kwargs（P4.6：vector_dim/graph_id 经 grep 确认 __init__
        # 不消费、也不落 self；core/tests/test_stress.py 仍按关键字传入且
        # 该文件不在本次改动范围，故保留为显式 no-op 兼容参数）
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
        # 向量缓存代数：结构性变更（合并/重定向/删除）时递增，
        # 供 remember 管线的 run 级候选表缓存感知库结构变化并整体重建
        self._vector_cache_generation: int = 0
        # P3.2 per-run 文档级解析缓存：run_id → {解析前 doc_id: (doc_id, ver_id, title, content_hash)}
        # 同一 run 的窗口复用首个窗口的文档级结果，避免每窗口重复
        # 全文件读 / content hash / 去重查询 / current 重写 / 版本检查。
        self._run_doc_lock = threading.Lock()
        self._run_doc_cache: Dict[str, Dict[str, tuple]] = {}
        # 文档级解析的 double-check 锁：与 _run_doc_lock 分开，锁序恒为
        # _run_doc_init_lock → _run_doc_lock，读缓存（_run_doc_resolve）不排队。
        self._run_doc_init_lock = threading.Lock()

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
                _json.dumps(sorted(set(missing_windows or []))), now_utc_str(),
            ),
        )
        self._commit_if_not_batched(conn)

    def list_documents(self, limit: int = 50, offset: int = 0,
                       source_document: str = None) -> List[dict]:
        conn = self._conn()
        docs = doc_repo.list_documents(conn, status="active",
                                       limit=limit, offset=offset,
                                       source_document=source_document)
        for d in docs:
            d["role"] = "document"
            if d.get("current_version_id"):
                d["document_version_id"] = d["current_version_id"]
        # P3.5：原实现逐文档 3 条 SQL（版本尺寸 + 实体数 + 关系数），
        # 50 文档 ≈ 150 查询；改为 IN() 批量 + GROUP BY 聚合，恒定 4 条以内。
        ver_ids = [d["document_version_id"] for d in docs if d.get("document_version_id")]
        ver_sizes: Dict[str, Tuple] = {}
        for chunk in _in_clause_chunks(ver_ids):
            rows = conn.execute(
                f"SELECT document_version_id, byte_size, char_count FROM document_versions "
                f"WHERE document_version_id IN ({_placeholders(chunk)})",
                chunk,
            ).fetchall()
            ver_sizes.update({r[0]: (r[1], r[2]) for r in rows})
        doc_ids = list({d["document_id"] for d in docs if d.get("document_id")})
        entity_counts: Dict[str, int] = {}
        relation_counts: Dict[str, int] = {}
        # P3.7：轻量 episode_count（走 idx_episodes_one_active_chunk 部分索引的批量
        # COUNT）——integrity 字段从列表响应移除后，graph 页的窗口数改用它展示。
        episode_counts: Dict[str, int] = {}
        for chunk in _in_clause_chunks(ver_ids):
            for row in conn.execute(
                f"SELECT document_version_id, COUNT(*) FROM episodes "
                f"WHERE status = 'active' AND document_version_id IN ({_placeholders(chunk)}) "
                f"GROUP BY document_version_id",
                chunk,
            ).fetchall():
                episode_counts[row[0]] = row[1]
        for chunk in _in_clause_chunks(doc_ids):
            for row in conn.execute(
                f"SELECT ep.document_id, COUNT(DISTINCT eo.entity_family_id) "
                f"FROM entity_mentions em "
                f"JOIN entity_observations eo ON eo.entity_id = em.entity_id AND eo.status = 'active' "
                f"JOIN episodes ep ON ep.episode_id = em.episode_id AND ep.status = 'active' "
                f"WHERE ep.document_id IN ({_placeholders(chunk)}) "
                f"GROUP BY ep.document_id",
                chunk,
            ).fetchall():
                entity_counts[row[0]] = row[1]
            for row in conn.execute(
                f"SELECT ep.document_id, COUNT(DISTINCT ra.relation_family_id) "
                f"FROM relation_assertions ra "
                f"JOIN episodes ep ON ep.episode_id = ra.episode_id AND ep.status = 'active' "
                f"WHERE ra.status = 'active' AND ep.document_id IN ({_placeholders(chunk)}) "
                f"GROUP BY ep.document_id",
                chunk,
            ).fetchall():
                relation_counts[row[0]] = row[1]
        for d in docs:
            ver_id = d.get("document_version_id")
            if ver_id and ver_id in ver_sizes:
                size, char_count = ver_sizes[ver_id]
                d["size"] = size or 0
                d["char_count"] = char_count or 0
            doc_id = d.get("document_id")
            if doc_id:
                d["entity_count"] = entity_counts.get(doc_id, 0)
                d["relation_count"] = relation_counts.get(doc_id, 0)
            if ver_id:
                d["episode_count"] = episode_counts.get(ver_id, 0)
        return docs

    def count_documents(self, source_document: str = None) -> int:
        conn = self._conn()
        if source_document:
            esc = escape_like(source_document)
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

        def _read_text_slice(path: Path, start: int, length: int) -> str:
            """Read only the requested character window from a UTF-8 file."""
            start = min(max(int(start or 0), 0), 10_000_000)
            length = min(max(int(length or 0), 0), 10_000_000)
            if length <= 0:
                return ""
            chunks: list[str] = []
            skipped = 0
            remaining = length
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                while remaining > 0:
                    block = handle.read(min(64 * 1024, remaining + max(0, start - skipped)))
                    if not block:
                        break
                    if skipped < start:
                        drop = min(start - skipped, len(block))
                        skipped += drop
                        block = block[drop:]
                    if block:
                        take = block[:remaining]
                        chunks.append(take)
                        remaining -= len(take)
                    if skipped >= start and remaining <= 0:
                        break
            return "".join(chunks)

        library_root = self.library_path.resolve()

        def _library_file(raw_path: str) -> Path | None:
            if not raw_path:
                return None
            candidate = (self.library_path / raw_path).resolve()
            if not candidate.is_relative_to(library_root):
                logger.warning("拒绝读取 library 外部路径: %s", raw_path)
                return None
            return candidate

        # Current versions may use the mutable managed file.  Historical (or
        # deleted) versions must prefer their immutable snapshot; falling back
        # to ``documents.managed_path`` there would silently show today's
        # content for an old-version request.
        content = ""
        read_path = ""
        managed = doc.get("managed_path", "")
        is_current = (
            ver.get("status") == "active"
            and ver.get("document_version_id") == doc.get("current_version_id")
        )
        if is_current and managed:
            full = _library_file(managed)
            if full and full.exists():
                content = _read_text_slice(full, offset, limit)
                read_path = managed
        if not content and ver.get("version_content_path"):
            full = _library_file(ver["version_content_path"])
            if full and full.exists():
                content = _read_text_slice(full, offset, limit)
                read_path = ver["version_content_path"]
        # A current version created by an older database may not have a
        # snapshot.  Only that current-version case may safely fall back to
        # the managed file; historical requests remain empty rather than
        # leaking the wrong version.
        if not content and is_current and managed and not read_path:
            full = _library_file(managed)
            if full and full.exists():
                content = _read_text_slice(full, offset, limit)
                read_path = managed
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
        is_current = (
            ver.get("status") == "active"
            and ver.get("document_version_id") == doc.get("current_version_id")
        )
        preferred_read_path = (
            doc.get("managed_path", "") if is_current
            else ver.get("version_content_path", "")
        )
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
            "read_path": preferred_read_path,
        }

    @staticmethod
    def _delete_observations_safe(conn, obs_ids: list) -> None:
        """删除 entity_observations 前先清掉指向它们的外键行。

        对齐会把别的文档/episode 的 entity_mentions（entity_id 锚点）和
        relation_assertions（subject/object_entity_id）挂到这些观察上；
        mentions.episode_id 还可为 NULL。只按 episode 过滤删 mention、
        只按 episode 删 assertion 会漏掉这些交叉引用，随后删除观察时触发
        FOREIGN KEY 约束（部分失败文档重灌时暴露）。relation_mentions 对
        relation_assertions 有 ON DELETE CASCADE，无需手动清。
        """
        if not obs_ids:
            return
        obs_ph = ",".join("?" for _ in obs_ids)
        conn.execute(f"DELETE FROM entity_mentions WHERE entity_id IN ({obs_ph})", obs_ids)
        cross_assert_ids = [r[0] for r in conn.execute(
            f"SELECT relation_id FROM relation_assertions "
            f"WHERE subject_entity_id IN ({obs_ph}) OR object_entity_id IN ({obs_ph})",
            obs_ids + obs_ids,
        ).fetchall()]
        if cross_assert_ids:
            cross_ph = ",".join("?" for _ in cross_assert_ids)
            conn.execute(f"DELETE FROM relation_assertions WHERE relation_id IN ({cross_ph})", cross_assert_ids)
            conn.execute(
                f"DELETE FROM embeddings WHERE owner_type = 'relation_assert' AND owner_id IN ({cross_ph})",
                cross_assert_ids,
            )
        conn.execute(f"DELETE FROM entity_observations WHERE entity_id IN ({obs_ph})", obs_ids)
        conn.execute(
            f"DELETE FROM embeddings WHERE owner_type = 'entity_obs' AND owner_id IN ({obs_ph})",
            obs_ids,
        )

    def delete_document_version(self, document_version_id: str) -> dict:
        conn = self._conn()
        # Get document_id
        ver = conn.execute(
            "SELECT document_id, status FROM document_versions WHERE document_version_id = ?",
            (document_version_id,),
        ).fetchone()
        if not ver:
            return {"deleted": False, "reason": "not found"}
        doc_id = ver[0]
        now = now_utc_str()

        # 1. Cascade-delete episodes belonging to this *version*.  Filtering
        # only by document_id used to erase every historical version when a
        # user removed one old snapshot.
        ep_ids = [r[0] for r in conn.execute(
            "SELECT episode_id FROM episodes WHERE document_version_id = ?", (document_version_id,)
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
            # Collect observation IDs, then delete them FK-safely: alignment
            # can anchor other episodes' mentions / assertions onto them.
            obs_ids_to_delete = [r[0] for r in conn.execute(
                f"SELECT entity_id FROM entity_observations WHERE episode_id IN ({ph})", ep_ids
            ).fetchall()]
            self._delete_observations_safe(conn, obs_ids_to_delete)
            # Delete embeddings linked to these episodes
            conn.execute(f"DELETE FROM embeddings WHERE owner_type = 'episode' AND owner_id IN ({ph})", ep_ids)
            # Delete embeddings for the assertions we just removed
            if rel_assert_ids_to_delete:
                rass_ph = ",".join("?" for _ in rel_assert_ids_to_delete)
                conn.execute(f"DELETE FROM embeddings WHERE owner_type = 'relation_assert' AND owner_id IN ({rass_ph})", rel_assert_ids_to_delete)

            # Delete episodes. document_links.from_episode_id 外键指向本版本
            # episodes（vault_indexer 写入），且无 ON DELETE 级联——必须先删
            # 这些链接行再删 episodes，否则 FOREIGN KEY 约束失败并留下
            # 部分 DML（版本级链接清理在函数尾部另行兜底）。
            conn.execute(f"DELETE FROM document_links WHERE from_episode_id IN ({ph})", ep_ids)
            conn.execute(f"DELETE FROM episodes_fts WHERE episode_id IN ({ph})", ep_ids)
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
                    self._delete_observations_safe(conn, orphan_obs_ids)
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

        # Delete links originating from this version only; other snapshots of
        # the same document retain their provenance.
        conn.execute("DELETE FROM document_links WHERE from_document_version_id = ?", (document_version_id,))

        # Soft-delete document_version and document
        conn.execute(
            "UPDATE document_versions SET status = 'deleted', processed_at = ? WHERE document_version_id = ?",
            (now, document_version_id),
        )
        active = conn.execute(
            "SELECT document_version_id FROM document_versions WHERE document_id = ? AND status = 'active' ORDER BY processed_at DESC LIMIT 1",
            (doc_id,),
        ).fetchone()
        if active:
            doc_repo.update_current_version(conn, doc_id, active[0], updated_at=now)
        else:
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

    def get_episode_content_detail(self, cache_id: str, *, max_chars: int = 200_000) -> Optional[dict]:
        row = ep_repo.get_episode(self._conn(), cache_id)
        if not row:
            return None
        max_chars = min(max(int(max_chars or 200_000), 1), 1_000_000)
        source_text = row.get("source_text", "") or ""
        memory_text = row.get("memory_text", "") or ""
        return {
            "episode_id": row.get("episode_id", ""),
            "source_text": source_text[:max_chars],
            "memory_text": memory_text[:max_chars],
            "truncated": len(source_text) > max_chars or len(memory_text) > max_chars,
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

    def find_family_id_by_name(self, name: str) -> Optional[str]:
        """按统一名称语义查 family（FamilyWriteGate 存储腿，原文名入参）。

        解析逻辑与 registry 注入的短连接版本共用
        core.judge.models.resolve_family_id_from_conn（变体召回 + 归一过滤）。
        """
        from core.judge.models import resolve_family_id_from_conn
        return resolve_family_id_from_conn(self._conn(), name)

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

    @staticmethod
    def _assemble_entities_projection(rows, snippet_len: int) -> List[dict]:
        """把投影 SQL 行组装为候选检索用 dict 列表（全量/子集查询共用）。"""
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
        return self._assemble_entities_projection(rows, snippet_len)

    def get_entities_projection_for_families(self, family_ids: List[str],
                                             content_snippet_length: int = None) -> List[dict]:
        """按 family_id 子集重取 latest 投影（remember run 级候选表缓存的增量刷新用）。

        与 get_latest_entities_projection 同一 JOIN/排序/去重语义，只是限定
        family 范围，避免 run 内每窗口全库重扫。
        """
        if not family_ids:
            return []
        snippet_len = content_snippet_length or self.entity_content_snippet_length
        conn = self._conn()
        placeholders = ",".join("?" for _ in family_ids)
        rows = conn.execute(
            "SELECT ef.entity_family_id, ef.canonical_name, ef.canonical_content, "
            "  eo.entity_id, eo.content, eo.processed_at, "
            "  (SELECT COUNT(*) FROM entity_observations eo2 "
            "   WHERE eo2.entity_family_id = ef.entity_family_id AND eo2.status != 'deleted') as version_count "
            "FROM entity_families ef "
            "JOIN entity_observations eo ON eo.entity_family_id = ef.entity_family_id AND eo.status = 'active' "
            f"WHERE ef.entity_family_id IN ({placeholders}) "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM entity_redirects r WHERE r.source_family_id = ef.entity_family_id"
            ") "
            "ORDER BY ef.updated_at DESC",
            family_ids,
        ).fetchall()
        return self._assemble_entities_projection(rows, snippet_len)

    def get_changed_entity_families_since_obs_rowid(self, after_obs_rowid: Optional[int] = None) -> Tuple[List[str], int]:
        """返回 after_obs_rowid 之后写入过 observation 的 family_id 列表与当前最大 rowid。

        run 内新建实体/已有实体新版本都会插入 entity_observations 行，rowid 单调
        递增 → 一次轻量定位即可找到需增量刷新的 family（合并/重定向这类结构性
        变更不产生新行，由向量缓存代数兜底触发整体重建）。
        after_obs_rowid=None 表示只取当前最大 rowid 作初始 marker，不扫变更行。
        """
        conn = self._conn()
        max_rowid = int(conn.execute(
            "SELECT COALESCE(MAX(rowid), 0) FROM entity_observations").fetchone()[0])
        if after_obs_rowid is None or after_obs_rowid >= max_rowid:
            return [], max_rowid
        rows = conn.execute(
            "SELECT DISTINCT entity_family_id FROM entity_observations WHERE rowid > ?",
            (after_obs_rowid,),
        ).fetchall()
        return [r[0] for r in rows], max_rowid

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
        ts = _fmt_dt(time_point) or now_utc_str()
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
            # rowid DESC 显式决出并列 processed_at 的胜者（后插入者），
            # 与 _batch_latest_assertions_by_family / _VECTOR_ROLE_CONFIG 一致
            "ORDER BY processed_at DESC, rowid DESC LIMIT 1",
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
        placeholders = _placeholders(family_ids)
        # Find relation_families where subject or object is in family_ids
        rows = conn.execute(
            f"SELECT DISTINCT rf.relation_family_id "
            f"FROM relation_families rf "
            f"WHERE rf.subject_entity_family_id IN ({placeholders}) "
            f"   OR rf.object_entity_family_id IN ({placeholders})",
            family_ids + family_ids,
        ).fetchall()
        rel_fids = [row[0] for row in rows][:limit]
        if not rel_fids:
            return []
        # P3.5：原实现逐 fid 调 get_relation_by_family_id（每个 5-6 条 SQL，
        # 20 fid ≈ 180 查询）；改为批量：families + 最新断言 + 最新观测 + embeddings。
        fams = self._batch_relation_families(rel_fids)
        asserts = self._batch_latest_assertions_by_family(rel_fids)
        obs_ids = self._batch_latest_obs_ids_by_families(
            [f.get("subject_entity_family_id", "") for f in fams.values()]
            + [f.get("object_entity_family_id", "") for f in fams.values()])
        embs = self._batch_relation_embedding_blobs(
            [a.get("relation_id", "") for a in asserts.values()])
        relations = []
        for fid in rel_fids:
            fam = fams.get(fid)
            row = asserts.get(fid)
            if not fam or not row:
                continue
            relations.append(assertion_to_relation(
                fam, row,
                subject_entity_id=obs_ids.get(row.get("subject_entity_family_id") or "", ""),
                object_entity_id=obs_ids.get(row.get("object_entity_family_id") or "", ""),
                embedding_blob=embs.get(row.get("relation_id") or "")))
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
                                source_document: str = None,
                                time_after: str = None,
                                time_before: str = None) -> List[dict]:
        """Return normalized concept DTOs rather than raw episode FTS rows.

        The old role-less path returned ``episodes_fts`` records with no
        ``family_id``/``role``.  Consumers then treated an episode as a
        concept and RRF collapsed unrelated hits under an empty key.  Keep
        the episode FTS index as the evidence source, but hydrate both concept
        roles and expose one stable DTO shape.  Pure document rows with no
        extracted concept receive an explicit ``role='episode'`` DTO so
        document search remains useful without masquerading as a concept.
        """
        roles = [role] if role in {"entity", "relation"} else ["entity", "relation"]
        results: List[dict] = []
        per_role_limit = max(limit, (limit + len(roles) - 1) // len(roles))
        for current_role in roles:
            objects = (
                self.search_entities_by_bm25(
                    query, limit=per_role_limit, time_point=time_point,
                    source_document=source_document, time_after=time_after,
                    time_before=time_before,
                )
                if current_role == "entity" else
                self.search_relations_by_bm25(
                    query, limit=per_role_limit, time_point=time_point,
                    source_document=source_document, time_after=time_after,
                    time_before=time_before,
                )
            )
            for obj in objects:
                if current_role == "entity":
                    results.append({
                        "family_id": obj.family_id,
                        "id": obj.absolute_id,
                        "name": obj.name,
                        "content": obj.content,
                        "role": "entity",
                        "_score": getattr(obj, "_score", 0.0),
                    })
                else:
                    results.append({
                        "family_id": obj.family_id,
                        "id": obj.absolute_id,
                        "name": "",
                        "content": obj.content,
                        "role": "relation",
                        "entity1_name": "",
                        "entity2_name": "",
                        "_score": getattr(obj, "_score", 0.0),
                    })
        if not results and role is None:
            # A newly indexed document may legitimately have no entity or
            # relation observations yet. Preserve document search with a
            # typed evidence DTO instead of returning raw FTS rows.
            evidence = search_repo.search_fts(
                self._conn(), query, limit=limit,
                source_document=source_document,
                time_after=time_after, time_before=time_before,
            )
            # 与实体/关系腿同款 BM25 归一（FTS5 原始分为负、越小越相关）。
            # 不归一会把负分原样塞进 _score，下游阈值过滤（0.5 / CJK 0.15）
            # 会把 episode 兜底腿的真命中全部丢掉。
            if evidence:
                scores = [r.get("score", 0) for r in evidence]
                min_s, max_s = min(scores), max(scores)
                span = max_s - min_s
                for r in evidence:
                    r["_score"] = (max_s - r.get("score", 0)) / span if span else 0.5
            for row in evidence:
                episode_id = row.get("episode_id")
                if not episode_id:
                    continue
                text = row.get("source_text") or row.get("memory_text") or ""
                results.append({
                    "family_id": episode_id,
                    "id": episode_id,
                    "episode_id": episode_id,
                    "name": text[:120],
                    "content": text,
                    "role": "episode",
                    "_score": row.get("_score", 0.0),
                })
        results.sort(key=lambda item: item.get("_score", 0.0), reverse=True)
        return results[:limit]

    def search_episodes_by_fts(self, query: str, limit: int = 20,
                               source_document: str = None,
                               time_after: str = None,
                               time_before: str = None) -> List[dict]:
        """Episode 原文 FTS 检索（搜索单入口的文本通道）。

        与 ``search_concepts_by_bm25``（概念 DTO，无 episode 维度）互补：
        返回 ``episodes_fts`` 命中行（episode_id/document_id/source_text），
        供 explore 的 episode-bm25 通道按原文取证。
        """
        rows = search_repo.search_fts(
            self._conn(), query, limit=limit,
            source_document=source_document,
            time_after=time_after, time_before=time_before,
        )
        out: List[dict] = []
        for row in rows:
            episode_id = row.get("episode_id")
            if not episode_id:
                continue
            out.append({
                "episode_id": episode_id,
                "document_id": row.get("document_id"),
                "source_text": row.get("source_text") or row.get("memory_text") or "",
                "score": row.get("score", 0.0),
            })
        return out

    def search_entities_by_bm25(self, query: str, limit: int = 20,
                                time_point: str = None,
                                source_document: str = None,
                                time_after: str = None,
                                time_before: str = None) -> List[Entity]:
        results = search_repo.search_fts(self._conn(), query, limit=limit,
                                         source_document=source_document,
                                         time_after=time_after, time_before=time_before)
        # Normalize BM25 scores (FTS5 returns negative, more negative = more relevant)
        # Invert so that most relevant → 1.0, least relevant → 0.0
        if results:
            scores = [r.get("score", 0) for r in results]
            min_s, max_s = min(scores), max(scores)
            span = max_s - min_s
            for r in results:
                r["_score"] = (max_s - r.get("score", 0)) / span if span else 0.5
        # 概念本身的时间列是 entity_observations.processed_at（P2.8 双界下推）
        obs_time_sql, obs_time_params = _time_bounds_sql(
            "eo.processed_at", time_after, time_before)
        # P3.5：原实现逐结果 1 条观测查询（N+1）；改为按 episode_id 一批 IN() 取回。
        obs_by_ep = self._batch_active_observations(
            [r.get("episode_id") for r in results], "episode_id",
            obs_time_sql, tuple(obs_time_params))
        entities = []
        for r in results:
            ep_id = r.get("episode_id")
            if not ep_id:
                continue
            obs = obs_by_ep.get(ep_id)
            if obs:
                fam = {"entity_family_id": obs["entity_family_id"],
                       "canonical_name": obs["canonical_name"],
                       "canonical_content": obs["canonical_content"]}
                e = observation_to_entity(fam, obs)
                e._score = r.get("_score", 0.0)
                entities.append(e)
        return entities

    def search_relations_by_bm25(self, query: str, limit: int = 20,
                                 time_point: str = None,
                                 source_document: str = None,
                                 time_after: str = None,
                                 time_before: str = None) -> List[Relation]:
        results = search_repo.search_fts(self._conn(), query, limit=limit,
                                         source_document=source_document,
                                         time_after=time_after, time_before=time_before)
        # Normalize BM25 scores (FTS5 returns negative, more negative = more relevant)
        # Invert so that most relevant → 1.0, least relevant → 0.0
        if results:
            scores = [r.get("score", 0) for r in results]
            min_s, max_s = min(scores), max(scores)
            span = max_s - min_s
            for r in results:
                r["_score"] = (max_s - r.get("score", 0)) / span if span else 0.5
        # 概念本身的时间列是 relation_assertions.processed_at（P2.8 双界下推）
        ra_time_sql, ra_time_params = _time_bounds_sql(
            "ra.processed_at", time_after, time_before)
        # P3.5：原实现逐结果 1 条断言查询 + 1 次 family 探测（N+1）；改为两批 IN()。
        ra_by_ep: Dict[str, dict] = {}
        conn = self._conn()
        ep_ids = [r.get("episode_id") for r in results]
        for chunk in _in_clause_chunks([e for e in dict.fromkeys(ep_ids) if e]):
            rows = conn.execute(
                f"SELECT ra.* FROM relation_assertions ra "
                f"WHERE ra.episode_id IN ({_placeholders(chunk)}) "
                f"  AND ra.status = 'active'{ra_time_sql} "
                f"ORDER BY ra.rowid",
                tuple(chunk) + tuple(ra_time_params),
            ).fetchall()
            for row in rows:
                ra_by_ep.setdefault(row["episode_id"], dict(row))
        fams = self._batch_relation_families(
            [ra["relation_family_id"] for ra in ra_by_ep.values()])
        relations = []
        for r in results:
            ep_id = r.get("episode_id")
            if not ep_id:
                continue
            ra = ra_by_ep.get(ep_id)
            if not ra:
                continue
            fam = fams.get(ra["relation_family_id"])
            if fam:
                rel = assertion_to_relation(fam, ra)
                rel._pending_patches = []
                rel._score = r.get("_score", 0.0)
                relations.append(rel)
        return relations

    def _semantic_candidate_rows(self, search_fn, owner_type: str,
                                 query_vec: bytes, limit: int) -> list:
        """SQL 语义检索腿的候选行获取（f6：多数模型回退）。

        与 ``_build_vector_cache_for_role`` 的回退语义对齐：active model
        在 embeddings 表无行时回退到该 owner_type 的多数模型并告警——
        否则换模型后（未 backfill）公开语义检索静默返回空，而 remember
        管线的向量缓存腿仍靠回退继续工作。跨模型余弦无意义，仍提示
        backfill-embeddings 重建。
        """
        model = self._active_embedding_model()
        conn = self._conn()
        rows = search_fn(conn, query_vec, embedding_model=model, limit=limit)
        if not rows:
            maj = conn.execute(
                "SELECT embedding_model, COUNT(*) AS c FROM embeddings "
                "WHERE owner_type = ? GROUP BY embedding_model "
                "ORDER BY c DESC LIMIT 1",
                (owner_type,),
            ).fetchone()
            if maj and maj[1] > 0 and maj[0] != model:
                rows = search_fn(conn, query_vec, embedding_model=maj[0], limit=limit)
                if rows:
                    logger.warning(
                        "语义检索：active model %r 无 %s embeddings，"
                        "回退多数模型 %r（%d 行）——换模型后请运行 "
                        "backfill-embeddings 重建",
                        model, owner_type, maj[0], len(rows))
        return rows

    def _semantic_entity_search(self, query_text: str, threshold: float = 0.3,
                                max_results: int = 20,
                                time_after: str = None,
                                time_before: str = None) -> List[Entity]:
        """实体语义检索单一实现（P4.2 收敛点）。

        嵌入召回 + 观测水合；agent_semantic_search 的实体腿走这里，
        search_entities_by_similarity 是它的兼容薄别名（保持 List[Entity] 形状，
        且不触发 agent_semantic_search 的 LIKE 兜底/关系腿——嵌入水合结果
        携带 embedding，走 dict 往返会丢失该字段）。
        """
        if not self.embedding_client or not self.embedding_client.is_available():
            return []
        result = _encode_and_normalize(self.embedding_client, query_text)
        if not result:
            return []
        query_vec, query_nd = result
        candidates = self._semantic_candidate_rows(
            emb_repo.search_entity_embeddings, "entity_obs",
            query_vec, limit=max_results * 3,
        )
        scored = []
        for c in candidates:
            vec = np.frombuffer(c["vector"], dtype=np.float32)
            sim = float(np.dot(query_nd, vec))
            if sim >= threshold:
                scored.append((sim, c))
        scored.sort(key=lambda x: -x[0])
        # 概念本身的时间列是 entity_observations.processed_at（P2.8 双界下推）
        obs_time_sql, obs_time_params = _time_bounds_sql(
            "eo.processed_at", time_after, time_before)
        # P3.5：原实现逐候选 1 条观测查询（N+1）；改为按 entity_id 一批 IN() 取回。
        obs_by_id = self._batch_active_observations(
            [c["owner_id"] for _sim, c in scored[:max_results]], "entity_id",
            obs_time_sql, tuple(obs_time_params))
        entities = []
        for sim, c in scored[:max_results]:
            obs = obs_by_id.get(c["owner_id"])
            if obs:
                fam = {"entity_family_id": obs["entity_family_id"],
                       "canonical_name": obs["canonical_name"],
                       "canonical_content": obs["canonical_content"]}
                e = observation_to_entity(fam, obs, embedding_blob=c["vector"])
                e._score = sim
                entities.append(e)
        return entities

    def search_entities_by_similarity(self, query_text: str, threshold: float = 0.3,
                                      max_results: int = 20, **kwargs) -> List[Entity]:
        """P4.2 兼容薄别名：实现统一收敛至 agent_semantic_search 单入口的实体腿。

        返回形状不变（List[Entity]，带 embedding/_score）；
        time_after/time_before 经 kwargs 透传（兼容旧签名）。
        """
        return self._semantic_entity_search(query_text, threshold=threshold,
                                            max_results=max_results,
                                            time_after=kwargs.get("time_after"),
                                            time_before=kwargs.get("time_before"))

    def search_relations_by_similarity(self, query_text: str, threshold: float = 0.3,
                                       max_results: int = 20, **kwargs) -> List[Relation]:
        # P2.8 双界下推：断言读取后按 processed_at 双界过滤（闭区间）
        lo_dt = _parse_dt(kwargs.get("time_after"))
        hi_dt = _parse_dt(kwargs.get("time_before"))
        if not self.embedding_client or not self.embedding_client.is_available():
            return []
        result = _encode_and_normalize(self.embedding_client, query_text)
        if not result:
            return []
        query_vec, query_nd = result
        candidates = self._semantic_candidate_rows(
            emb_repo.search_relation_embeddings, "relation_assert",
            query_vec, limit=max_results * 3,
        )
        scored = []
        for c in candidates:
            vec = np.frombuffer(c["vector"], dtype=np.float32)
            sim = float(np.dot(query_nd, vec))
            if sim >= threshold:
                scored.append((sim, c))
        scored.sort(key=lambda x: -x[0])
        # P3.5：原实现逐候选调 get_relation_by_absolute_id（每个 4-5 条 SQL，N+1）；
        # 改为批量：断言+family 一批、最新观测 id 一批、embeddings 一批。
        owner_ids = [c["owner_id"] for _sim, c in scored[:max_results]]
        rows_by_id: Dict[str, dict] = {}
        conn = self._conn()
        for chunk in _in_clause_chunks([i for i in dict.fromkeys(owner_ids) if i]):
            rows = conn.execute(
                f"SELECT ra.*, rf.relation_family_id, rf.subject_entity_family_id, "
                f"  rf.object_entity_family_id, rf.canonical_content "
                f"FROM relation_assertions ra "
                f"JOIN relation_families rf ON rf.relation_family_id = ra.relation_family_id "
                f"WHERE ra.relation_id IN ({_placeholders(chunk)})",
                chunk,
            ).fetchall()
            rows_by_id.update({r["relation_id"]: dict(r) for r in rows})
        obs_ids = self._batch_latest_obs_ids_by_families(
            [r.get("subject_entity_family_id") or "" for r in rows_by_id.values()]
            + [r.get("object_entity_family_id") or "" for r in rows_by_id.values()])
        embs = self._batch_relation_embedding_blobs(list(rows_by_id))
        relations = []
        for sim, c in scored[:max_results]:
            row = rows_by_id.get(c["owner_id"])
            if not row:
                continue
            fam = {k: row[k] for k in ("relation_family_id", "subject_entity_family_id",
                                       "object_entity_family_id", "canonical_content")}
            rel = assertion_to_relation(
                fam, row,
                subject_entity_id=obs_ids.get(row.get("subject_entity_family_id") or "", ""),
                object_entity_id=obs_ids.get(row.get("object_entity_family_id") or "", ""),
                embedding_blob=embs.get(c["owner_id"]))
            # 概念本身的时间列是 relation_assertions.processed_at（P2.8 双界下推）
            when = rel.processed_time
            if lo_dt and (when is None or when < lo_dt):
                continue
            if hi_dt and (when is None or when > hi_dt):
                continue
            rel._pending_patches = []
            rel._score = sim
            relations.append(rel)
        return relations

    def suggest_concepts(self, query: str, role: str = "entity", limit: int = 10,
                         source_document: str = None) -> List[dict]:
        conn = self._conn()
        like = escape_like(query) + "%"
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

    def get_concept_names_by_family_ids(self, family_ids: List[str]) -> Dict[str, str]:
        """批量解析概念显示名（CLI concept get/neighbors 的逐邻居名字查询收敛，P3.5）。

        取名语义与 get_concept_by_family_id 一致，按同优先级探测三类表：
        entity → canonical_name；episode → heading_path 或 name；
        relation family 无名称列，返回 fid 本身（原 ``.get("name", fid)`` 同值）。
        未命中任何表的 fid 不出现在结果里——调用方保持"名字未解析"状态。
        """
        uniq = [f for f in dict.fromkeys(family_ids) if f]
        if not uniq:
            return {}
        conn = self._conn()
        names: Dict[str, str] = {}
        remaining = list(uniq)
        for chunk in _in_clause_chunks(remaining):
            rows = conn.execute(
                f"SELECT entity_family_id, canonical_name FROM entity_families "
                f"WHERE entity_family_id IN ({_placeholders(chunk)})",
                chunk,
            ).fetchall()
            names.update({r[0]: r[1] or "" for r in rows})
        remaining = [f for f in remaining if f not in names]
        for chunk in _in_clause_chunks(remaining):
            rows = conn.execute(
                f"SELECT relation_family_id FROM relation_families "
                f"WHERE relation_family_id IN ({_placeholders(chunk)})",
                chunk,
            ).fetchall()
            names.update({r[0]: r[0] for r in rows})
        remaining = [f for f in remaining if f not in names]
        for chunk in _in_clause_chunks(remaining):
            rows = conn.execute(
                f"SELECT episode_id, heading_path, name FROM episodes "
                f"WHERE episode_id IN ({_placeholders(chunk)})",
                chunk,
            ).fetchall()
            names.update({r[0]: (r[1] or "") or (r[2] or "") for r in rows})
        return names

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
                                       updated_at=now_utc_str())
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

    def adjust_confidence_on_corroboration(self, family_id: str,
                                           source_type: str = "entity",
                                           **_ignored) -> None:
        """独立来源印证 → 概念置信度 +0.05（封顶 1.0）。

        remember 管线（alignment / entity_alignment / relation*）在
        try/except 里调用；方法缺失时 AttributeError 被吞成静默 no-op。
        """
        self.adjust_confidence_on_corroboration_batch([family_id],
                                                      source_type=source_type)

    def adjust_confidence_on_corroboration_batch(self, family_ids: List[str],
                                                 source_type: str = "entity",
                                                 **_ignored) -> None:
        """批量印证调整：最新 active 观测/断言的 extra_json["confidence"] +0.05。

        v1.5 schema 无独立 confidence 列——置信度随观测/断言行存
        extra_json（get_concept_by_family_id 同一读取口），因此只更新
        family 最新 active 行；extra_json 未记录置信度时不动
        （对应旧实现的 ``confidence IS NOT NULL`` 语义）。
        """
        import json as _json
        from .dto_mapping import _extract_confidence
        if not family_ids:
            return
        conn = self._conn()
        if source_type == "relation":
            pick_sql = ("SELECT relation_id, extra_json FROM relation_assertions "
                        "WHERE relation_family_id = ? AND status = 'active' "
                        "ORDER BY processed_at DESC, rowid DESC LIMIT 1")
            update_sql = ("UPDATE relation_assertions SET extra_json = ? "
                          "WHERE relation_id = ?")
        else:
            pick_sql = ("SELECT entity_id, extra_json FROM entity_observations "
                        "WHERE entity_family_id = ? AND status = 'active' "
                        "ORDER BY processed_at DESC, rowid DESC LIMIT 1")
            update_sql = ("UPDATE entity_observations SET extra_json = ? "
                          "WHERE entity_id = ?")
        for fid in family_ids:
            if not fid:
                continue
            row = conn.execute(pick_sql, (fid,)).fetchone()
            if not row:
                continue
            owner_id, extra_json = row[0], row[1] or "{}"
            current = _extract_confidence(extra_json)
            if current is None:
                continue
            try:
                extra = _json.loads(extra_json)
            except (ValueError, TypeError):
                extra = {}
            if not isinstance(extra, dict):
                extra = {}
            extra["confidence"] = min(float(current) + 0.05, 1.0)
            conn.execute(update_sql,
                         (_json.dumps(extra, ensure_ascii=False), owner_id))
        self._commit_if_not_batched(conn)

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
                                  document_family_ids, max_episodes, max_concepts,
                                  include_relations=include_relations,
                                  include_versions=include_versions)

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
        with self._write_batch() as conn:
            register_redirect(conn, source_id, target_id)

    def register_entity_redirects_batch(self, redirects: Dict[str, str]):
        from .merge import register_redirects_batch
        with self._write_batch() as conn:
            register_redirects_batch(conn, redirects)
        self.invalidate_vector_caches()

    def merge_entity_families(self, target_family_id: str,
                              source_family_ids: List[str],
                              skip_name_check: bool = False) -> Dict[str, Any]:
        from .merge import merge_entity_families
        with self._write_batch() as conn:
            result = merge_entity_families(
                conn, target_family_id, source_family_ids,
                skip_name_check=skip_name_check,
            )
        self.invalidate_vector_caches()
        return result

    def redirect_entity_relations(self, old_family_id: str, new_family_id: str):
        from .merge import redirect_entity_relations
        with self._write_batch() as conn:
            redirect_entity_relations(conn, old_family_id, new_family_id)
        self.invalidate_vector_caches()

    def delete_entity_all_versions(self, family_id: str) -> int:
        from .merge import delete_entity_all_versions
        with self._write_batch() as conn:
            result = delete_entity_all_versions(conn, family_id)
        self.invalidate_vector_caches()
        return result

    def dedup_merge_batch(self, pairs: List[Tuple[str, str]]) -> int:
        from .merge import dedup_merge_batch
        with self._write_batch():
            result = dedup_merge_batch(self._conn(), pairs)
        self.invalidate_vector_caches()
        return result

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
                              source_document: str = None,
                              time_point: str = None,
                              time_after: str = None,
                              time_before: str = None) -> dict:
        """实体/概念语义检索唯一入口（P4.2：server 路由、CLI、explore 共用）。

        过滤器收敛到此一处（此前由 server 路由向多个存储方法逐个透传）：
          - role：None = 实体 + 关系两腿；"entity"/"relation" 只走对应腿；
            其他 role（document/episode）语义检索无对应腿，返回空。
          - time_after/time_before：闭区间双界下推（概念时间列 =
            entity_observations.processed_at / relation_assertions.processed_at）。
          - time_point / source_document：仅收口兼容参数，存储层当前不消费
            （与收敛前行为一致；见 test_hybrid_ignored_filters_current_behavior
            钉住的现状——实现它们属于精准性工作，不在结构收敛内）。

        返回 ``{"results": List[dict], "total": int}``；检索 dict 统一携带
        family_id/id/name/content/role/_score（关系额外带 entity1_name/entity2_name）。
        """
        results: List[dict] = []
        if role is None or role == "entity":
            for e in self._semantic_entity_search(query, threshold=threshold,
                                                  max_results=top_k,
                                                  time_after=time_after,
                                                  time_before=time_before):
                results.append({
                    "family_id": e.family_id, "id": e.absolute_id,
                    "name": e.name, "content": e.content,
                    "role": "entity", "_score": getattr(e, "_score", 0.0),
                })
        if role is None or role == "relation":
            for r in self.search_relations_by_similarity(query, threshold=threshold,
                                                         max_results=top_k,
                                                         time_after=time_after,
                                                         time_before=time_before):
                results.append({
                    "family_id": r.family_id, "id": r.absolute_id,
                    "name": "", "content": r.content,
                    "entity1_name": "", "entity2_name": "",
                    "role": "relation", "_score": getattr(r, "_score", 0.0),
                })
        # Fallback: when embedding search returns no results (e.g. no
        # embedding client available), try a LIKE-based name lookup.
        if not results and (role is None or role == "entity"):
            conn = self._conn()
            like = f"%{escape_like(query)}%"
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

    def save_content_patches(self, patches):
        """Persist section-level entity/relation content diffs.

        ContentPatch objects are deliberately kept independent from the
        version tables: a patch describes the transition *to* a new absolute
        version and can therefore be queried even after the previous version
        is superseded.  ``INSERT OR IGNORE`` makes retries idempotent while
        the write-batch boundary keeps a multi-patch flush atomic.
        """
        if not patches:
            return 0
        inserted = 0
        with self._write_batch() as conn:
            for patch in patches:
                get = patch.get if isinstance(patch, dict) else lambda key, default=None: getattr(patch, key, default)
                patch_id = str(get("uuid", "") or "").strip()
                target_type = str(get("target_type", "") or "").strip()
                target_absolute_id = str(get("target_absolute_id", "") or "").strip()
                target_family_id = str(get("target_family_id", "") or "").strip()
                section_key = str(get("section_key", "") or "").strip()
                if not patch_id or target_type not in {"Entity", "Relation"}:
                    raise ValueError("content patch requires a valid uuid and target_type")
                if not target_absolute_id or not target_family_id or not section_key:
                    raise ValueError("content patch target/section cannot be empty")
                event_time = get("event_time")
                if hasattr(event_time, "isoformat"):
                    event_time = event_time.isoformat()
                event_time = str(event_time or now_utc_str())
                cursor = conn.execute(
                    """INSERT OR IGNORE INTO content_patches
                       (patch_id,target_type,target_absolute_id,target_family_id,
                        section_key,change_type,old_hash,new_hash,diff_summary,
                        source_document,event_time)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        patch_id,
                        target_type,
                        target_absolute_id,
                        target_family_id,
                        section_key,
                        str(get("change_type", "modified") or "modified"),
                        str(get("old_hash", "") or ""),
                        str(get("new_hash", "") or ""),
                        str(get("diff_summary", "") or ""),
                        str(get("source_document", "") or ""),
                        event_time,
                    ),
                )
                inserted += max(0, int(cursor.rowcount or 0))
        return inserted

    def get_content_patches(self, *, target_family_id: str = "",
                            target_absolute_id: str = "", limit: int = 100) -> list[dict]:
        """Return persisted section diffs, newest first, with a hard limit."""
        limit = min(max(int(limit or 100), 1), 1000)
        conditions = []
        params = []
        if target_family_id:
            conditions.append("target_family_id = ?")
            params.append(target_family_id)
        if target_absolute_id:
            conditions.append("target_absolute_id = ?")
            params.append(target_absolute_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._conn().execute(
            f"SELECT * FROM content_patches {where} ORDER BY event_time DESC, patch_id DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [dict(row) for row in rows]

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
        for table in ("content_patches", "entity_mentions", "relation_mentions",
                      "relation_assertions", "relation_families",
                       "entity_observations", "entity_families", "entity_redirects",
                       "document_links", "embeddings", "pipeline_runs",
                       "episodes", "document_ingestion_state", "document_versions", "documents"):
            conn.execute(f"DELETE FROM {table}")
        # FTS5's ``rebuild`` command only refreshes rows backed by the content
        # table; this index is maintained explicitly, so stale standalone
        # rows would survive a graph clear.  Delete them before committing.
        conn.execute("DELETE FROM episodes_fts")
        conn.commit()
        self.invalidate_vector_caches()

    # ------------------------------------------------------------------
    # Write methods (pipeline-facing)
    # ------------------------------------------------------------------

    def _run_doc_resolve(self, run_id: str, doc_key: str) -> Optional[tuple]:
        """P3.2：读取 per-run 文档级解析缓存，未命中返回 None。"""
        if not run_id:
            return None
        with self._run_doc_lock:
            return self._run_doc_cache.get(run_id, {}).get(doc_key)

    def _run_doc_store(self, run_id: str, doc_key: str,
                       doc_id: str, ver_id: str, title: str,
                       content_hash: str) -> None:
        """P3.2：写入 per-run 文档级解析缓存（锁内 setdefault，保证恰好一次生效）。"""
        if not run_id:
            return
        with self._run_doc_lock:
            # 防膨胀：保留的 run 数超上限时整体清空（最坏情况重做一次文档级解析）
            if len(self._run_doc_cache) >= 8:
                self._run_doc_cache.clear()
            self._run_doc_cache.setdefault(run_id, {}).setdefault(
                doc_key, (doc_id, ver_id, title, content_hash)
            )

    def _resolve_episode_document(self, conn, text: str, source: str,
                                  document_path: str, doc_id: str,
                                  override_doc_id: str) -> tuple:
        """save_episode 的文档级解析部分（P3.2：同一 run 只执行一次）。

        全文件读 → content hash → 跨文档去重 → 文档行 ensure → current 重写
        → 版本复用/快照。返回 (doc_id, ver_id, title, content_hash)。
        """
        from . import content_fs

        # A targeted retry receives one window as ``text`` and an explicit
        # document identity.  Reuse the existing active version *before* any
        # current-file write: treating that window as the full document used
        # to truncate the managed file during repair.
        doc = doc_repo.get_document(conn, doc_id)
        if override_doc_id and doc:
            old_ver = doc_repo.get_active_version(conn, doc_id)
            if old_ver:
                return (
                    doc_id,
                    old_ver["document_version_id"],
                    doc.get("title") or source,
                    old_ver.get("content_hash") or "",
                )
            # 版本已不存在（benchmark resume 重置 / 用户删除快照后重灌）：
            # 不再 raise，落回下面的常规路径从 document_path 重建版本，
            # 保持稳定 doc_id。

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
            title = (source or Path(document_path).stem) if document_path else (source or doc_id)
            content_md = content_fs.write_current_file(
                str(self.library_path), title or doc_id, doc_text, doc_id=doc_id,
            )
            doc_repo.insert_document(
                conn, doc_id, title,
                managed_path=content_md,
                source_mode="managed" if source else "external",
                # absolute_path 落库是窗口缓存作用域命中（断点续传）的前提：
                # find_cache_by_doc_hash 按 document_path 关联 documents 行，
                # 缺失时同内容重跑永远 miss，整 doc 从头重算。
                absolute_path=document_path or "",
                created_at=now_utc_str(), updated_at=now_utc_str(),
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
                (content_md, now_utc_str(), doc_id),
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
                processed_at=now_utc_str(),
            )
            doc_repo.update_current_version(conn, doc_id, ver_id, updated_at=now_utc_str())
        return doc_id, ver_id, title, content_hash

    def save_episode(self, *args, **kwargs) -> str:
        """Serialize episode/document writes across processor connections.

        Document initialization, version switching, FTS replacement, and the
        episode insert must be one writer critical section.  Without this,
        concurrent windows can hold SQLite WAL snapshots while each tries to
        upgrade to a writer, causing an intermittent lock/deadlock and
        partially initialized per-run document caches.
        """
        # A window write spans document/version resolution, episode replacement,
        # FTS maintenance, retrieval slices, and optional graph retirement.  Keep
        # the whole unit rollback-safe: an error in a late slice insert must not
        # leave the connection holding a half-published transaction.
        with self._write_batch():
            return self._save_episode_unlocked(*args, **kwargs)

    def _save_episode_unlocked(self, cache: Episode, text: str = "",
                     document_path: str = "", doc_hash: str = "",
                     start_offset: int = 0, end_offset = None,
                     override_doc_id: str = "",
                     heading_path: str = "",
                     episode_type: str = "",
                     run_id: str = "",
                     chunk_index: Optional[int] = None,
                     retrieval_slice_chars: int = 0) -> str:
        """Persist an Episode DTO and its source document."""
        import hashlib
        import uuid

        conn = self._conn()
        text = text or cache.content
        source = cache.source_document or ""

        # Determine document identity from source or path
        if override_doc_id:
            doc_id = override_doc_id
        else:
            source_key = document_path or source or text[:64]
            doc_id = f"doc_{hashlib.sha256(source_key.encode()).hexdigest()[:16]}"

        # P3.2 文档级工作每 run 一次：同一 run 的首个窗口完成文档级解析
        # （全文件读、content hash、跨文档去重、current 重写、版本快照）后
        # 记入 per-run 缓存，后续窗口直接复用 (doc_id, ver_id, title)，
        # 每窗口只剩 episode 行写入 + FTS 同步。step1 串行路径不再被
        # 文档级 I/O 逐窗口阻塞。run_id 为空（直连调用/测试）不启用，
        # 保持原有逐次完整解析语义。
        # 并发竞态：锁 + double-check 保证文档级初始化恰发生一次——管线下
        # step1 本就在 _cache_lock 下串行（首窗口先行），这里是直连并发
        # 调用路径的兜底；解析 + 入缓存同在 _run_doc_init_lock 内完成，
        # _run_doc_store 的 setdefault 保证首个完成者的结果生效。
        _doc_key = doc_id
        _doc_resolved = self._run_doc_resolve(run_id, _doc_key)
        if _doc_resolved is None:
            if run_id:
                with self._run_doc_init_lock:
                    _doc_resolved = self._run_doc_resolve(run_id, _doc_key)
                    if _doc_resolved is None:
                        _doc_resolved = self._resolve_episode_document(
                            conn, text, source, document_path, doc_id, override_doc_id)
                        self._run_doc_store(run_id, _doc_key, *_doc_resolved)
            else:
                _doc_resolved = self._resolve_episode_document(
                    conn, text, source, document_path, doc_id, override_doc_id)
        doc_id, ver_id, title, content_hash = _doc_resolved

        # Create/update episode.  Pipeline callers provide the absolute window
        # index so targeted repair replaces the same slot instead of appending
        # a duplicate at ``COUNT(*)``.
        ep_id = cache.absolute_id or f"ep_{uuid.uuid4().hex[:16]}"
        ep_fam = f"epfam_{doc_id}_{doc_hash or ep_id}"
        _explicit_chunk_index = chunk_index is not None
        if chunk_index is None:
            chunk_index = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE document_version_id = ?",
                (ver_id,),
            ).fetchone()[0]
        else:
            chunk_index = max(0, int(chunk_index))
        values = (
            doc_id, ver_id, text, cache.content or "",
            heading_path or getattr(cache, 'heading_path', '') or "",
            start_offset, end_offset if end_offset is not None else len(text),
            chunk_index, doc_hash or content_hash[:16], title,
            episode_type or getattr(cache, 'episode_type', '') or "",
            cache.activity_type or "", _fmt_dt(cache.event_time) or now_utc_str(),
            _fmt_dt(cache.processed_time) or now_utc_str(), run_id,
        )
        existing_id = conn.execute(
            "SELECT episode_id FROM episodes WHERE episode_id = ?", (ep_id,)
        ).fetchone()
        if existing_id:
            conn.execute(
                """UPDATE episodes SET document_id=?, document_version_id=?,
                   episode_family_id=?, source_text=?, memory_text=?,
                   heading_path=?, start_offset=?, end_offset=?, chunk_index=?,
                   chunk_hash=?, name=?, episode_type=?, activity_type=?,
                   status='active', event_time=?, processed_at=?, run_id=?
                   WHERE episode_id=?""",
                (doc_id, ver_id, ep_fam, values[2], values[3], values[4],
                 values[5], values[6], values[7], values[8], values[9],
                 values[10], values[11], values[12], values[13], values[14], ep_id),
            )
        else:
            # Replace an incomplete active row occupying this window.  Its
            # downstream observations remain tied to the superseded episode
            # and therefore stay out of active graph views.
            collision = None
            if _explicit_chunk_index:
                _slot_rows = conn.execute(
                    "SELECT episode_id, chunk_hash, status FROM episodes "
                    "WHERE document_version_id=? AND chunk_index=?",
                    (ver_id, chunk_index),
                ).fetchall()
                # 同 hash 槽位优先：表级 UNIQUE 按 (ver, chunk, hash) 拦截，真正
                # 冲突的只有内容相同的那一行；不同 hash 的历史行不构成冲突。
                collision = next(
                    (r for r in _slot_rows if (r[1] or "") == (values[8] or "")),
                    _slot_rows[0] if _slot_rows else None,
                )
            if collision and collision[0] != ep_id:
                old_episode_id = collision[0]
                if (collision[1] or "") == (values[8] or ""):
                    # 同 (ver, chunk, hash) 槽位在表级 UNIQUE 下只能容纳一行——
                    # 无论 active 还是墓碑。稳定 ID 重试（reset 后同内容重灌）、
                    # 簇收敛合并的墓碑、以及同窗口内容被两个不同 Episode 身份
                    #（一有 absolute_id 一用 uuid）先后保存的竞态，都会占着槽位
                    # 拦下 INSERT；退位（只改 status）救不回来。同内容换身份重存
                    # 一律连子观测硬删后按新身份重插（同 hash 子行描述的就是同
                    # 一窗口，由新保存方的步骤 9/10 重新生成）。
                    conn.execute(
                        "DELETE FROM entity_observations WHERE episode_id=?",
                        (old_episode_id,),
                    )
                    conn.execute(
                        "DELETE FROM relation_assertions WHERE episode_id=?",
                        (old_episode_id,),
                    )
                    conn.execute(
                        "DELETE FROM entity_mentions WHERE episode_id=?",
                        (old_episode_id,),
                    )
                    conn.execute(
                        "DELETE FROM episodes WHERE episode_id=?",
                        (old_episode_id,),
                    )
                    ep_repo.fts_delete_episodes(conn, [old_episode_id])
                else:
                    # Replacing a window must retire its graph observations too;
                    # otherwise the old facts remain visible beside the repaired
                    # facts even though the episode itself is superseded.
                    conn.execute(
                        "UPDATE entity_observations SET status='superseded' "
                        "WHERE episode_id=? AND status='active'",
                        (old_episode_id,),
                    )
                    conn.execute(
                        "UPDATE relation_assertions SET status='superseded' "
                        "WHERE episode_id=? AND status='active'",
                        (old_episode_id,),
                    )
                    conn.execute(
                        "DELETE FROM entity_mentions WHERE episode_id=?",
                        (old_episode_id,),
                    )
                    conn.execute(
                        "UPDATE episodes SET status='superseded' WHERE episode_id=?",
                        (old_episode_id,),
                    )
                    ep_repo.fts_delete_episodes(conn, [old_episode_id])
            ep_repo.insert_episode(
                conn, ep_id, ep_fam, doc_id, ver_id,
                source_text=values[2], memory_text=values[3], heading_path=values[4],
                start_offset=values[5], end_offset=values[6], chunk_index=values[7],
                chunk_hash=values[8], name=values[9], episode_type=values[10],
                activity_type=values[11], event_time=values[12],
                processed_at=values[13], run_id=values[14],
            )
        ep_repo.fts_sync_episode(conn, ep_id, doc_id, ver_id,
                                  name=title, source_text=text,
                                  memory_text=cache.content or "")
        if retrieval_slice_chars and episode_type != "retrieval_slice":
            self._write_retrieval_slices(
                conn, text, doc_id, ver_id, ep_fam, title,
                heading_path=heading_path,
                base_chunk_index=chunk_index,
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
        # Main document windows use non-negative chunk indexes.  Retrieval-only
        # rows need a disjoint namespace; using ``base + n`` made the next main
        # window collide with the previous window's slices under
        # idx_episodes_one_active_chunk.  Reserve a deterministic negative range
        # per base window so rewrites can also retire stale slices reliably.
        slice_stride = 1_000_000
        if len(slices) >= slice_stride:
            raise ValueError("too many retrieval slices for one episode")
        slice_prefix = (max(0, int(base_chunk_index)) + 1) * slice_stride
        slice_low = -(slice_prefix + slice_stride - 1)
        slice_high = -(slice_prefix + 1)
        stale_slice_ids = [
            row[0] for row in conn.execute(
                "SELECT episode_id FROM episodes WHERE document_version_id=? "
                "AND episode_type='retrieval_slice' AND status='active' "
                "AND chunk_index BETWEEN ? AND ?",
                (ver_id, slice_low, slice_high),
            ).fetchall()
        ]
        if stale_slice_ids:
            placeholders = ",".join("?" for _ in stale_slice_ids)
            conn.execute(
                f"UPDATE episodes SET status='superseded' "
                f"WHERE episode_id IN ({placeholders})",
                stale_slice_ids,
            )
            ep_repo.fts_delete_episodes(conn, stale_slice_ids)

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
                chunk_index=-(slice_prefix + n),
                chunk_hash=slice_hash,
                name=title,
                episode_type="retrieval_slice",
                activity_type="",
                event_time=now_utc_str(), processed_at=now_utc_str(),
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
            created_at=now_utc_str(), updated_at=now_utc_str(),
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
        if _emb_raw is None:
            # 管线的单条持久化路径（对齐命中/更新）不预计算向量；这里兜底
            # 编码一次，保证 embeddings 表有行可查（agent_semantic_search /
            # hybrid scope 依赖）。批量路径已填 entity.embedding，不会走到。
            _computed = self._compute_entity_embedding(entity)
            _emb_raw = _computed[0] if _computed else None

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
                self._commit_if_not_batched(conn)
                return
        ent_repo.insert_entity_observation(
            conn, obs_id, fid, ep_id,
            name=entity.name, content=entity.content,
            extra_json=extra_json or "{}",
            processed_at=_fmt_dt(entity.processed_time) or now_utc_str(),
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
                       run_id: str = "", extra_json: str = "") -> str | None:
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
                                                   created_at=now_utc_str(), updated_at=now_utc_str())

        # Upsert relation family
        fam = rel_repo.find_relation_family(conn, sub_fid, obj_fid)
        if not fam:
            rel_fid = relation.family_id or f"relfam_{uuid.uuid4().hex[:16]}"
            rel_repo.upsert_relation_family(
                conn, rel_fid, sub_fid, obj_fid,
                canonical_content=relation.content,
                created_at=now_utc_str(), updated_at=now_utc_str(),
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
        persisted_relation_id = rel_repo.insert_relation_assertion(
            conn, ra_id, rel_fid, rel_ep_id,
            sub_abs, obj_abs, sub_fid, obj_fid,
            content=relation.content,
            evidence_text=relation.evidence_text or "",
            evidence_start_offset=relation.evidence_start_offset or 0,
            evidence_end_offset=relation.evidence_end_offset or 0,
            evidence_line_start=relation.evidence_line_start or 0,
            evidence_line_end=relation.evidence_line_end or 0,
            extra_json=extra_json or "{}",
            processed_at=_fmt_dt(relation.processed_time) or now_utc_str(),
            run_id=run_id,
        )
        # Store embedding if available（relation 的向量同样只在合并检查分支被
        # 预计算；这里兜底编码，见 save_entity 同款注释）
        _rel_emb = relation.embedding
        if isinstance(_rel_emb, (list, tuple)):
            _rel_emb = _rel_emb[0]
        if not _rel_emb:
            _computed_rel = self._compute_relation_embedding(relation)
            _rel_emb = _computed_rel[0] if _computed_rel else None
        if _rel_emb and persisted_relation_id:
            self._store_embedding_if_available("relation_assert", persisted_relation_id, "content",
                                                relation.content, _rel_emb)
        self._commit_if_not_batched(conn)
        return persisted_relation_id

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

        if target_type == "relation":
            # Relation assertions already carry episode provenance, but the
            # pipeline also emits an explicit MENTIONS edge.  Store it in a
            # relation-specific table instead of violating entity_mentions FK.
            inserted = 0
            for relation_id in dict.fromkeys(entity_absolute_ids):
                row = conn.execute(
                    "SELECT relation_family_id, content, evidence_start_offset, "
                    "evidence_end_offset, evidence_line_start, evidence_line_end "
                    "FROM relation_assertions WHERE relation_id=? AND episode_id=? "
                    "AND status='active'",
                    (relation_id, episode_id),
                ).fetchone()
                if not row:
                    continue
                mention_id = f"rment_{uuid.uuid4().hex[:16]}"
                cur = conn.execute(
                    "INSERT OR IGNORE INTO relation_mentions "
                    "(mention_id, relation_id, relation_family_id, episode_id, "
                    "surface_text, start_offset, end_offset, line_start, line_end, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (mention_id, relation_id, row[0], episode_id, row[1] or "",
                     row[2] or 0, row[3] or 0, row[4] or 0, row[5] or 0,
                     now_utc_str()),
                )
                inserted += int(cur.rowcount or 0)
            self._commit_if_not_batched(conn)
            return inserted

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
                created_at=now_utc_str(),
            )
        self._commit_if_not_batched(conn)

    def save_extraction_result(self, doc_hash: str, entities: list,
                               relations: list, document_path: str = "",
                               document_version_id: str = "") -> bool:
        """Save extraction results to task extraction cache."""
        ep = self.find_cache_by_doc_hash(
            doc_hash, document_path=document_path,
            document_version_id=document_version_id,
        )
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
                               document_path: str = "",
                               document_version_id: str = "") -> Optional[tuple]:
        ep = self.find_cache_by_doc_hash(
            doc_hash, document_path=document_path,
            document_version_id=document_version_id,
        )
        if not ep:
            return None
        path = self.extraction_cache_dir / ep.absolute_id / "extraction.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return (data.get("entities", []), data.get("relations", []))

    def find_cache_by_doc_hash(self, doc_hash: str,
                                document_path: str = "",
                                document_version_id: str = "") -> Optional[Episode]:
        conn = self._conn()
        params: list[Any] = [doc_hash]
        scope_sql = ""
        if document_version_id:
            scope_sql = " AND e.document_version_id = ?"
            params.append(document_version_id)
        elif document_path:
            # Cache hashes are not globally unique: two documents can share a
            # paragraph.  Scope lookup to the document identity/path first so
            # extraction results cannot silently cross-contaminate tasks.
            import hashlib
            derived_id = "doc_" + hashlib.sha256(document_path.encode("utf-8")).hexdigest()[:16]
            scope_sql = (
                " AND (e.document_id = ? OR d.absolute_path = ? "
                "OR d.managed_path = ? OR d.relative_path = ?)"
            )
            params.extend([derived_id, document_path, document_path, document_path])
        row = conn.execute(
            "SELECT e.* FROM episodes e JOIN documents d ON d.document_id = e.document_id "
            "WHERE e.chunk_hash = ? AND e.status = 'active'" + scope_sql +
            " ORDER BY e.processed_at DESC LIMIT 1",
            params,
        ).fetchone()
        if row:
            return episode_row_to_dto(dict(row))
        return None

    def find_cache_and_extraction_by_doc_hash(self, doc_hash: str,
                                               document_path: str = "",
                                               document_version_id: str = ""):
        ep = self.find_cache_by_doc_hash(doc_hash, document_path, document_version_id)
        if not ep:
            return None, None
        extraction = self.load_extraction_result(
            doc_hash, document_path, document_version_id
        )
        return ep, extraction

    def assess_remember_window_statuses(self, doc_hashes: List[str],
                                         document_path: str = "",
                                         document_version_id: str = "") -> List[dict]:
        conn = self._conn()
        results = []
        for idx, h in enumerate(doc_hashes):
            ep = self.find_cache_by_doc_hash(h, document_path, document_version_id)
            ext = self.load_extraction_result(
                h, document_path, document_version_id
            ) if ep else None

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
                            "SELECT COUNT(*) FROM entity_observations "
                            "WHERE episode_id = ? AND status = 'active'",
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
            # 同上：rowid 决并列，与 _batch_latest_obs_ids_by_families 一致
            "ORDER BY processed_at DESC, rowid DESC LIMIT 1",
            (family_id,),
        ).fetchone()
        return row[0] if row else ""

    # ── P3.5 批量读取 helpers（N+1 收敛用，语义与上面的单条版一致）──

    def _batch_active_observations(self, ids: List[str], key_column: str,
                                   time_sql: str = "",
                                   time_params: Tuple = ()) -> Dict[str, dict]:
        """按 key_column（episode_id / entity_id）批量取 active 观测 + family 名称列。

        key_column='episode_id' 时每组保留 rowid 最小行：原单条查询走
        idx_entityobs_episode(episode_id) 索引序 fetchone，显式 ORDER BY rowid
        固化该语义；key_column='entity_id' 是主键，本就至多一行。
        """
        uniq = [i for i in dict.fromkeys(ids) if i]
        if not uniq:
            return {}
        conn = self._conn()
        out: Dict[str, dict] = {}
        order_sql = " ORDER BY eo.rowid" if key_column == "episode_id" else ""
        for chunk in _in_clause_chunks(uniq):
            rows = conn.execute(
                f"SELECT eo.*, ef.canonical_name, ef.canonical_content "
                f"FROM entity_observations eo "
                f"JOIN entity_families ef ON ef.entity_family_id = eo.entity_family_id "
                f"WHERE eo.{key_column} IN ({_placeholders(chunk)}) "
                f"  AND eo.status = 'active'{time_sql}{order_sql}",
                tuple(chunk) + tuple(time_params),
            ).fetchall()
            for row in rows:
                out.setdefault(row[key_column], dict(row))
        return out

    def _batch_relation_families(self, family_ids: List[str]) -> Dict[str, dict]:
        """批量取 relation_families 行（rel_repo.get_relation_family 的批量版）。"""
        uniq = [f for f in dict.fromkeys(family_ids) if f]
        if not uniq:
            return {}
        conn = self._conn()
        out: Dict[str, dict] = {}
        for chunk in _in_clause_chunks(uniq):
            rows = conn.execute(
                f"SELECT * FROM relation_families "
                f"WHERE relation_family_id IN ({_placeholders(chunk)})",
                chunk,
            ).fetchall()
            out.update({r["relation_family_id"]: dict(r) for r in rows})
        return out

    def _batch_latest_assertions_by_family(self, family_ids: List[str]) -> Dict[str, dict]:
        """批量取每个 relation family 的最新 active 断言。

        等价于逐 fid ``ORDER BY processed_at DESC LIMIT 1``；NOT EXISTS 反连接
        以 (processed_at, rowid) 决定次序（与 _VECTOR_ROLE_CONFIG 同模式），
        走 idx_relassert_family(relation_family_id, processed_at DESC)。
        """
        uniq = [f for f in dict.fromkeys(family_ids) if f]
        if not uniq:
            return {}
        conn = self._conn()
        out: Dict[str, dict] = {}
        for chunk in _in_clause_chunks(uniq):
            rows = conn.execute(
                f"SELECT ra.* FROM relation_assertions ra "
                f"WHERE ra.relation_family_id IN ({_placeholders(chunk)}) "
                f"  AND ra.status = 'active' "
                f"  AND NOT EXISTS ("
                f"    SELECT 1 FROM relation_assertions ra2 "
                f"    WHERE ra2.relation_family_id = ra.relation_family_id "
                f"      AND ra2.status = 'active' "
                f"      AND (ra2.processed_at > ra.processed_at "
                f"           OR (ra2.processed_at = ra.processed_at "
                f"               AND ra2.rowid > ra.rowid)))",
                chunk,
            ).fetchall()
            for row in rows:
                out[row["relation_family_id"]] = dict(row)
        return out

    def _batch_latest_obs_ids_by_families(self, family_ids: List[str]) -> Dict[str, str]:
        """批量取每个 entity family 的最新 active 观测 id（_latest_obs_id_for_family 的批量版）。"""
        uniq = [f for f in dict.fromkeys(family_ids) if f]
        if not uniq:
            return {}
        conn = self._conn()
        out: Dict[str, str] = {}
        for chunk in _in_clause_chunks(uniq):
            rows = conn.execute(
                f"SELECT eo.entity_family_id, eo.entity_id FROM entity_observations eo "
                f"WHERE eo.entity_family_id IN ({_placeholders(chunk)}) "
                f"  AND eo.status = 'active' "
                f"  AND NOT EXISTS ("
                f"    SELECT 1 FROM entity_observations eo2 "
                f"    WHERE eo2.entity_family_id = eo.entity_family_id "
                f"      AND eo2.status = 'active' "
                f"      AND (eo2.processed_at > eo.processed_at "
                f"           OR (eo2.processed_at = eo.processed_at "
                f"               AND eo2.rowid > eo.rowid)))",
                chunk,
            ).fetchall()
            out.update({r[0]: r[1] for r in rows})
        return out

    def _batch_relation_embedding_blobs(self, assertion_ids: List[str]) -> Dict[str, bytes]:
        """批量取断言 embedding（_get_embedding_blob('relation_assert', …) 的批量版）。

        ORDER BY created_at DESC 后每组取首行，与单条版"最新一条"一致。
        """
        uniq = [i for i in dict.fromkeys(assertion_ids) if i]
        if not uniq:
            return {}
        conn = self._conn()
        out: Dict[str, bytes] = {}
        for chunk in _in_clause_chunks(uniq):
            rows = conn.execute(
                f"SELECT owner_id, vector FROM embeddings "
                f"WHERE owner_type = 'relation_assert' "
                f"  AND owner_id IN ({_placeholders(chunk)}) "
                f"ORDER BY created_at DESC",
                chunk,
            ).fetchall()
            for owner_id, vector in rows:
                out.setdefault(owner_id, vector)
        return out

    def _vector_cache_for_role(self, role: str) -> dict:
        """Return cached vector matrix for a role, loading on first access.

        按 active embedding model 缓存——模型切换后自动重建（不变式 e）。
        """
        model = self._active_embedding_model()
        with self._vector_cache_lock:
            cached = self._vector_role_cache.get(role)
            if cached is not None and cached.get("model") == model:
                return cached
            result = self._build_vector_cache_for_role(role)
            self._vector_role_cache[role] = result
            return result

    def invalidate_vector_caches(self) -> None:
        """清除向量矩阵缓存（合并/重定向/删除后调用——懒重建）。

        不在每次 embedding 插入时失效：新增观测只是缓存暂时缺行，
        事后一致；结构性变更（family 消失/改挂）才会让缓存指向死 fid。
        """
        with self._vector_cache_lock:
            self._vector_role_cache.clear()
            # 代数递增：run 级候选表缓存据此感知结构变更并整体重建
            self._vector_cache_generation += 1

    def _active_embedding_model(self) -> str:
        client = getattr(self, "embedding_client", None)
        return (getattr(client, 'model_name', None)
                or getattr(client, 'model_path', None)
                or 'unknown')

    def embedding_model_report(self) -> Dict[str, Any]:
        """P6.1：embedding 模型一致性报告（只读 GROUP BY，走 embedding_model 索引）。

        返回 {"active": str, "models": {model: 行数}, "consistent": bool,
        "warning": Optional[str]}——判定逻辑见 helpers.embedding_consistency，
        registry 启动预热线程与 doctor 均消费此报告。
        """
        from .helpers import embedding_consistency
        active = self._active_embedding_model()
        report: Dict[str, Any] = {
            "active": active, "models": {}, "consistent": True, "warning": None}
        try:
            conn = self._conn()
            rows = conn.execute(
                "SELECT embedding_model, COUNT(*) AS c FROM embeddings "
                "GROUP BY embedding_model ORDER BY c DESC"
            ).fetchall()
        except Exception:
            return report
        report["models"] = {r[0]: int(r[1]) for r in rows}
        consistent, warning = embedding_consistency(active, report["models"])
        report["consistent"] = consistent
        report["warning"] = warning
        return report

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

        Returns {"matrix": np.ndarray(N,D), "rows": [{"family_id": str}, ...],
        "_loaded": True, "model": str} or {"matrix": None, "rows": [], ...} if
        no data / error.

        按 active embedding model 过滤（与 search_entity_embeddings 对齐）。
        混模型库：active model 无行时回退到多数模型并告警（不变式 e）——
        此前不过滤，换模型后余弦是跨模型垃圾值且永不失效。
        """
        import numpy as np

        config = self._VECTOR_ROLE_CONFIG.get(role)
        if config is None:
            return {"matrix": None, "rows": [], "_loaded": True, "model": ""}

        sql = (
            f"SELECT e.vector, {config['family_col']}, {config['owner_col']} "
            f"FROM embeddings e "
            f"{config['join_fragment']} "
            f"WHERE e.owner_type = ? AND e.embedding_model = ? "
            f"  AND NOT EXISTS ({config['dedup_fragment']})"
        )

        model = self._active_embedding_model()
        try:
            conn = self._conn()
            rows = conn.execute(sql, (config["owner_type"], model)).fetchall()
            if not rows:
                # 多数模型回退：存量库的 embeddings 大多属旧模型时仍可用
                maj = conn.execute(
                    "SELECT embedding_model, COUNT(*) AS c FROM embeddings "
                    "WHERE owner_type = ? GROUP BY embedding_model "
                    "ORDER BY c DESC LIMIT 1",
                    (config["owner_type"],),
                ).fetchone()
                if maj and maj[1] > 0:
                    rows = conn.execute(sql, (config["owner_type"], maj[0])).fetchall()
                    if rows:
                        logger.warning(
                            "向量缓存：active model %r 无 %s embeddings，"
                            "回退多数模型 %r（%d 行）——换模型后请运行 "
                            "backfill-embeddings 重建",
                            model, config["owner_type"], maj[0], len(rows))
                        model = maj[0]
        except Exception as exc:
            logger.debug("_build_vector_cache_for_role(%s) SQL failed: %s", role, exc)
            return {"matrix": None, "rows": [], "_loaded": True, "model": model}

        if not rows:
            logger.debug("_build_vector_cache_for_role(%s): no embeddings found", role)
            return {"matrix": None, "rows": [], "_loaded": True, "model": model}

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
            return {"matrix": None, "rows": [], "_loaded": True, "model": model}

        matrix = np.vstack(vectors)
        logger.info("_build_vector_cache_for_role(%s): loaded %d vectors, dim=%d, model=%s",
                     role, matrix.shape[0], matrix.shape[1], model)
        return {"matrix": matrix, "rows": meta_rows, "_loaded": True, "model": model}

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
            created_at=now_utc_str(),
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

    # ==================================================================
    # Scope sandbox（任务 A 追加区块）——概念 → episode → document 批量回溯
    # 仅供 core/find/scope.py 的 build_document_scope 消费；上方方法未改动。
    # ==================================================================

    def concept_source_documents(self, family_ids: List[str], *,
                                 include_offsets: bool = True) -> List[dict]:
        """批量回溯 family → episode → document 的裸数据（scope 沙箱用）。

        对 entity family 走 entity_mentions + entity_observations，对
        relation family 走 relation_assertions（family 命中哪张表由 id 空间
        自然区分）；IN 分块查询后按 (family_id, episode_id) 去重——
        mention / assertion 行（带证据偏移）优先于 observation 行。

        可见性过滤与搜索视图（repositories/search.py::search_fts、
        v_document_files）保持一致：episodes.status='active' AND
        documents.status='active' AND document_versions.status='active' AND
        COALESCE(document_ingestion_state.state, 'active')='active'。

        每行返回：family_id / role('entity'|'relation') / episode_id /
        episode_name / source_text（前 512 字符，供 snippet 用）/ document_id
        / title / source_mode / managed_path / absolute_path；
        include_offsets=True 时另带 episode_start_offset / episode_end_offset
        （episode 在文档中的位置）与 evidence_start_offset /
        evidence_end_offset（mention/断言在 episode 中的证据位置，observation
        来源为 None）。
        """
        uniq = [f for f in dict.fromkeys(family_ids or []) if f]
        if not uniq:
            return []
        conn = self._conn()
        # (表, family 列, role, 证据偏移投影, 额外 status 过滤)
        # entity_mentions 无 status 列；mention 行不 join entity_observations
        # 的 active 校验——旧观测被 superseded 不影响"该 family 出现在该
        # episode"这一事实（scope 回溯要召回不要精排）。
        sources = [
            ("entity_mentions", "entity_family_id", "entity",
             "src.start_offset AS evidence_start_offset, "
             "src.end_offset AS evidence_end_offset, ", "1=1"),
            ("relation_assertions", "relation_family_id", "relation",
             "src.evidence_start_offset AS evidence_start_offset, "
             "src.evidence_end_offset AS evidence_end_offset, ",
             "src.status = 'active'"),
            ("entity_observations", "entity_family_id", "entity",
             "NULL AS evidence_start_offset, NULL AS evidence_end_offset, ",
             "src.status = 'active'"),
        ]
        _source_sql = (
            "SELECT {fid_col} AS family_id, '{role}' AS role, "
            "        src.episode_id, ep.name AS episode_name, "
            "        {evidence}"
            "        ep.start_offset AS episode_start_offset, "
            "        ep.end_offset AS episode_end_offset, "
            "        substr(ep.source_text, 1, 512) AS source_text, "
            "        d.document_id, d.title, d.source_mode, "
            "        d.managed_path, d.absolute_path "
            "FROM {table} src "
            "JOIN episodes ep ON ep.episode_id = src.episode_id "
            "               AND ep.status = 'active' "
            "JOIN documents d ON d.document_id = ep.document_id "
            "                AND d.status = 'active' "
            "LEFT JOIN document_ingestion_state dis "
            "       ON dis.document_id = d.document_id "
            "JOIN document_versions dv "
            "       ON dv.document_id = ep.document_id "
            "      AND dv.document_version_id = ep.document_version_id "
            "      AND dv.status = 'active' "
            "WHERE src.{fid_col} IN ({ph}) "
            "  AND src.episode_id IS NOT NULL "
            "  AND {status_filter} "
            "  AND COALESCE(dis.state, 'active') = 'active'"
        )
        rows: Dict[Tuple[str, str], dict] = {}
        for table, fid_col, role, evidence_sql, extra_filter in sources:
            for chunk in _in_clause_chunks(uniq):
                sql = _source_sql.format(
                    table=table, fid_col=fid_col, role=role,
                    evidence=evidence_sql, ph=_placeholders(chunk),
                    status_filter=extra_filter)
                for r in conn.execute(sql, chunk).fetchall():
                    row = dict(r)
                    key = (row["family_id"], row["episode_id"])
                    existing = rows.get(key)
                    # observation 行（证据偏移 None）不覆盖 mention/断言行
                    if existing is None or (
                            existing.get("evidence_start_offset") is None
                            and row.get("evidence_start_offset") is not None):
                        rows[key] = row
        result = []
        for key in sorted(rows):
            row = rows[key]
            if not include_offsets:
                for offset_key in ("evidence_start_offset", "evidence_end_offset",
                                   "episode_start_offset", "episode_end_offset"):
                    row.pop(offset_key, None)
            result.append(row)
        return result
