"""Base mixin for SQLiteGraphStorageManager — constructor, DB helpers, family-id
resolution, redirect logic, dedup/merge, and dream-candidate filtering."""

import hashlib
import json
import logging
import shutil
import sqlite3
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ...models import Entity, Relation
from ...perf import _perf_timer
from ...utils import clean_markdown_code_blocks
from ..cache import QueryCache
from .helpers import (
    ENTITY_COLUMNS,
    EPISODE_COLUMNS,
    RELATION_COLUMNS,
    _encode_and_normalize,
    _fmt_dt,
    _get_cached_now,
    _parse_dt,
    _row_to_entity,
    _row_to_relation,
)
from .schema import init_schema

try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
except ImportError:
    nx = None
    louvain_communities = None

try:
    import jieba as _jieba
except ImportError:
    _jieba = None

try:
    import hnswlib as _hnswlib
except ImportError:
    _hnswlib = None

import re as _re

_TOKEN_SPLIT_RE = _re.compile(r'[\s,;，；、]+')

logger = logging.getLogger(__name__)

_EMB_CONTENT_MAX = 512


# ---------------------------------------------------------------------------
# Dream helpers (shared with Neo4j backend)
# ---------------------------------------------------------------------------

_DREAM_TIERS = frozenset(("candidate", "verified", "rejected"))


def _new_record_id():
    return f"relation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _dream_source(cycle_id):
    return f"dream:{cycle_id}" if cycle_id else "dream"


# ---------------------------------------------------------------------------
# Time-point parsing (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _cached_tp_to_datetime(tp):
    if tp is None:
        return None
    if isinstance(tp, datetime):
        return tp
    try:
        dt = datetime.fromisoformat(str(tp).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ===========================================================================
# _BaseMixin
# ===========================================================================


class _BaseMixin:
    """Constructor, DB helpers, family-id resolution, redirect logic, dedup/merge,
    and dream-candidate filtering."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        storage_path: str = "./graph",
        embedding_client=None,
        entity_content_snippet_length: int = 50,
        relation_content_snippet_length: int = 50,
        vector_dim: int = 1024,
        graph_id: str = "default",
    ):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self.storage_path = self._storage_path
        self._db_path = self._storage_path / "graph.db"
        self._graph_id = graph_id
        self.embedding_client = embedding_client
        self.entity_content_snippet_length = entity_content_snippet_length
        self.relation_content_snippet_length = relation_content_snippet_length
        self._vector_dim = vector_dim

        # Persistent thread-local connection pool
        self._thread_local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()

        # Thread-safe write locks
        self._write_lock = threading.Lock()
        self._entity_write_lock = threading.Lock()
        self._relation_write_lock = threading.Lock()
        self._episode_write_lock = threading.Lock()

        # Query cache
        self._cache = QueryCache(default_ttl=30, max_size=4096)

        # Entity name cache (absolute_id -> name)
        self._entity_name_cache: Dict[str, str] = {}

        # Entity embedding cache
        self._entity_emb_cache: Optional[List[tuple]] = None
        self._entity_emb_cache_ts: float = 0.0
        self._entity_emb_fid_idx: Optional[Dict[str, int]] = None
        self._emb_cache_ttl: float = 120.0
        self._emb_cache_max_size: int = 10000

        # Relation embedding cache
        self._relation_emb_cache: Optional[List[tuple]] = None
        self._relation_emb_cache_ts: float = 0.0
        self._relation_emb_fid_idx: Optional[Dict[str, int]] = None

        # HNSW indices (lazy-built from embedding caches)
        self._entity_hnsw = None          # hnswlib.Index or None
        self._entity_hnsw_items = None    # list[Entity] aligned with index
        self._relation_hnsw = None
        self._relation_hnsw_items = None

        # Entity remap cache (old abs_id -> latest abs_id per family)
        self._entity_remap_cache: Optional[dict] = None
        self._entity_remap_cache_ts: float = 0.0
        self._entity_remap_cache_ttl: float = 120.0

        # Episode filesystem dirs
        self.docs_dir = self._storage_path / "docs"
        self.docs_dir.mkdir(exist_ok=True)
        self.cache_dir = self._storage_path / "episodes"
        self.cache_json_dir = self.cache_dir / "json"
        self.cache_md_dir = self.cache_dir / "md"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_json_dir.mkdir(exist_ok=True)
        self.cache_md_dir.mkdir(exist_ok=True)

        # Doc hash caches
        self._id_to_doc_hash: Dict[str, str] = {}
        self._doc_hash_to_dirname: Dict[str, str] = {}
        self._build_doc_hash_cache()

        # Episode meta caches
        self._meta_files_cache: tuple = (0.0, None)
        self._bm25_lower_cache: tuple = (0.0, None)
        self._meta_json_cache: dict = {}
        self._META_FILES_TTL: float = 2.0

        # Initialize schema
        self._init_schema()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return the thread-local persistent connection (creates on first call)."""
        conn = getattr(self._thread_local, '_sqlite_conn', None)
        if conn is not None:
            try:
                conn.execute("SELECT 1")
                return conn
            except sqlite3.Error:
                self._discard_conn(conn)
                conn = None
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")
        conn.execute("PRAGMA mmap_size=268435456")
        conn.execute("PRAGMA wal_autocheckpoint=1000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA busy_timeout=5000")
        self._thread_local._sqlite_conn = conn
        with self._conn_lock:
            self._all_conns.append(conn)
        return conn

    def _discard_conn(self, conn):
        with self._conn_lock:
            if conn in self._all_conns:
                self._all_conns.remove(conn)
        try:
            conn.rollback()
        except Exception:
            pass

    def close(self):
        """Close all thread-local connections (call on shutdown)."""
        with self._conn_lock:
            for c in self._all_conns:
                try:
                    c.close()
                except Exception:
                    pass
            self._all_conns.clear()
        if hasattr(self._thread_local, '_sqlite_conn'):
            del self._thread_local._sqlite_conn

    @staticmethod
    def _build_hnsw(items_with_emb: list, dim: int, ef_construction=200, M=16, ef_search=50):
        """Build an hnswlib Index from a list of (item, np.array) tuples.

        Returns (index, items) where items are aligned with index positions,
        or (None, None) if hnswlib is unavailable or no valid vectors exist.
        """
        if _hnswlib is None:
            return None, None
        valid = [(item, emb) for item, emb in items_with_emb if emb is not None]
        if len(valid) < 50:
            return None, None
        n = len(valid)
        index = _hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
        items = []
        vectors = []
        for item, emb in valid:
            items.append(item)
            vectors.append(emb)
        data = np.vstack(vectors).astype(np.float32)
        index.add_items(data, list(range(n)))
        index.set_ef(ef_search)
        return index, items

    def _init_schema(self):
        conn = self._connect()
        try:
            init_schema(conn)
        finally:
            conn.rollback()

    def _build_doc_hash_cache(self):
        """Scan docs/ dirs to build doc_hash -> dirname reverse map."""
        self._doc_hash_to_dirname.clear()
        if not self.docs_dir.is_dir():
            return
        for d in self.docs_dir.iterdir():
            if d.is_dir():
                meta_path = d / "meta.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        dh = meta.get("doc_hash", "")
                        if dh:
                            self._doc_hash_to_dirname[dh] = d.name
                            aid = meta.get("absolute_id", "")
                            if aid:
                                self._id_to_doc_hash[aid] = d.name
                    except Exception:
                        pass

    def clear_graph_data(self):
        """Delete all entities, relations, episodes, and edges for this graph_id.

        Keeps the database file and schema intact. Used by GraphRegistry.clear_graph().
        """
        with self._connect() as conn:
            gid = self._graph_id
            # Delete in dependency order: edges first, then nodes
            conn.execute("DELETE FROM relates_to WHERE graph_id = ?", (gid,))
            conn.execute("DELETE FROM mentions WHERE graph_id = ?", (gid,))
            conn.execute("DELETE FROM content_patch WHERE target_family_id IN (SELECT family_id FROM entity WHERE graph_id = ?)", (gid,))
            conn.execute("DELETE FROM dream_log WHERE graph_id = ?", (gid,))
            conn.execute("DELETE FROM relation WHERE graph_id = ?", (gid,))
            conn.execute("DELETE FROM episode WHERE graph_id = ?", (gid,))
            conn.execute("DELETE FROM entity WHERE graph_id = ?", (gid,))
            conn.commit()
        # Clear in-memory caches
        self._entity_name_cache.clear()
        if self._entity_emb_cache is not None:
            self._entity_emb_cache.clear()
        if self._relation_emb_cache is not None:
            self._relation_emb_cache.clear()
        self.invalidate_entity_remap_cache()
        logger.info("Cleared all data for graph_id=%s", self._graph_id)

    def delete_graph_data(self):
        """Same as clear_graph_data for SQLite (file-level delete handled by registry)."""
        self.clear_graph_data()

    # ------------------------------------------------------------------
    # Internal: cache helpers
    # ------------------------------------------------------------------

    def _cache_entity_name(self, absolute_id: str, name: str):
        if absolute_id and name:
            self._entity_name_cache[absolute_id] = name

    def invalidate_entity_remap_cache(self):
        self._entity_remap_cache = None
        self._entity_remap_cache_ts = 0.0

    # ------------------------------------------------------------------
    # Base mixin: dream candidate filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _is_dream_candidate(relation) -> bool:
        if not relation.attributes:
            return False
        if isinstance(relation.attributes, str) and (
            '"candidate"' not in relation.attributes or '"tier"' not in relation.attributes
        ):
            return False
        try:
            attrs = json.loads(relation.attributes) if isinstance(relation.attributes, str) else relation.attributes
            tier = attrs.get("tier")
            status = attrs.get("status")
            return tier == "candidate" and status != "verified"
        except (json.JSONDecodeError, TypeError, AttributeError):
            return False

    def _filter_dream_candidates(self, relations: list, include_candidates: bool = False) -> list:
        if include_candidates or not relations:
            return relations
        return [r for r in relations if not self._is_dream_candidate(r)]

    # ------------------------------------------------------------------
    # Base mixin: family_id resolution (redirect chain)
    # ------------------------------------------------------------------

    def _resolve_family_id_in_conn(self, conn, family_id: str) -> str:
        current_id = (family_id or "").strip()
        if not current_id:
            return ""
        seen: Set[str] = set()
        while current_id and current_id not in seen:
            seen.add(current_id)
            row = conn.execute(
                "SELECT target_id FROM entity_redirect WHERE source_id = ?",
                (current_id,),
            ).fetchone()
            if not row or not row["target_id"] or row["target_id"] == current_id:
                break
            current_id = row["target_id"]
        return current_id

    def resolve_family_id(self, family_id: str) -> str:
        cache_key = f"resolve:{family_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with _perf_timer("resolve_family_id"):
            conn = self._connect()
            try:
                resolved = self._resolve_family_id_in_conn(conn, family_id)
            finally:
                conn.rollback()
        self._cache.set(cache_key, resolved, ttl=600)
        return resolved

    def resolve_family_ids(self, family_ids: List[str]) -> Dict[str, str]:
        if not family_ids:
            return {}
        unique_ids = list({_f for fid in family_ids if fid and (_f := fid.strip())})
        if not unique_ids:
            return {}
        result: Dict[str, str] = {}
        uncached: List[str] = []
        for fid in unique_ids:
            cache_key = f"resolve:{fid}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                result[fid] = cached
            else:
                uncached.append(fid)
        if uncached:
            conn = self._connect()
            try:
                for fid in uncached:
                    resolved = self._resolve_family_id_in_conn(conn, fid)
                    result[fid] = resolved
                    self._cache.set(f"resolve:{fid}", resolved, ttl=600)
            finally:
                conn.rollback()
        output: Dict[str, str] = {}
        for fid in family_ids:
            key = fid.strip() if fid else ""
            output[fid] = result.get(key, key)
        return output

    def register_entity_redirect(self, source_family_id: str, target_family_id: str) -> str:
        source_id = (source_family_id or "").strip()
        target_id = (target_family_id or "").strip()
        if not source_id or not target_id:
            return target_id
        with self._write_lock:
            conn = self._connect()
            try:
                canonical_target = self._resolve_family_id_in_conn(conn, target_id)
                if not canonical_target:
                    canonical_target = target_id
                canonical_source = self._resolve_family_id_in_conn(conn, source_id)
                if canonical_source == canonical_target:
                    return canonical_target
                now_iso = datetime.now().isoformat()
                conn.execute(
                    "INSERT OR REPLACE INTO entity_redirect (source_id, target_id, updated_at) VALUES (?, ?, ?)",
                    (source_id, canonical_target, now_iso),
                )
                conn.commit()
                return canonical_target
            finally:
                conn.rollback()

    def register_entity_redirects_batch(self, pairs: List[Tuple[str, str]]) -> None:
        if not pairs:
            return
        now_iso = datetime.now().isoformat()
        with self._write_lock:
            conn = self._connect()
            try:
                rows = []
                for source_id, target_id in pairs:
                    source_id = (source_id or "").strip()
                    target_id = (target_id or "").strip()
                    if not source_id or not target_id or source_id == target_id:
                        continue
                    canonical_target = self._resolve_family_id_in_conn(conn, target_id)
                    canonical_source = self._resolve_family_id_in_conn(conn, source_id)
                    if canonical_source == canonical_target:
                        continue
                    rows.append((source_id, canonical_target, now_iso))
                if rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO entity_redirect (source_id, target_id, updated_at) VALUES (?, ?, ?)",
                        rows,
                    )
                    conn.commit()
            finally:
                conn.rollback()

    def redirect_entity_relations(self, old_family_id: str, new_family_id: str) -> int:
        old_family_id = (old_family_id or "").strip()
        new_family_id = (new_family_id or "").strip()
        if not old_family_id or not new_family_id:
            return 0
        with self._write_lock:
            conn = self._connect()
            try:
                # Get new abs_id
                row = conn.execute(
                    "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                    (new_family_id, self._graph_id),
                ).fetchone()
                if not row:
                    return 0
                new_abs_id = row["uuid"]

                # Get old abs_ids
                old_rows = conn.execute(
                    "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?",
                    (old_family_id, self._graph_id),
                ).fetchall()
                old_abs_ids = [r["uuid"] for r in old_rows]
                if not old_abs_ids:
                    return 0

                # Update relation references
                placeholders = ",".join("?" * len(old_abs_ids))
                cnt1 = conn.execute(
                    f"UPDATE relation SET entity1_absolute_id = ? WHERE entity1_absolute_id IN ({placeholders}) AND graph_id = ?",
                    [new_abs_id] + old_abs_ids + [self._graph_id],
                ).rowcount
                cnt2 = conn.execute(
                    f"UPDATE relation SET entity2_absolute_id = ? WHERE entity2_absolute_id IN ({placeholders}) AND graph_id = ?",
                    [new_abs_id] + old_abs_ids + [self._graph_id],
                ).rowcount

                # Update relates_to edges
                for old_aid in old_abs_ids:
                    conn.execute(
                        "UPDATE relates_to SET entity1_uuid = ? WHERE entity1_uuid = ? AND graph_id = ?",
                        (new_abs_id, old_aid, self._graph_id),
                    )
                    conn.execute(
                        "UPDATE relates_to SET entity2_uuid = ? WHERE entity2_uuid = ? AND graph_id = ?",
                        (new_abs_id, old_aid, self._graph_id),
                    )

                conn.commit()
                self._cache.invalidate_keys(["graph_stats"])
                return cnt1 + cnt2
            finally:
                conn.rollback()

    def dedup_merge_batch(self, pairs: List[Tuple[str, str]]) -> int:
        if not pairs:
            return 0
        all_ids = list(set(fid for pair in pairs for fid in pair if fid and fid.strip()))
        resolved = self.resolve_family_ids(all_ids) if all_ids else {}
        resolved_pairs: List[Tuple[str, str]] = []
        for old_fid, new_fid in pairs:
            old_fid = (old_fid or "").strip()
            new_fid = (new_fid or "").strip()
            if not old_fid or not new_fid:
                continue
            old_r = resolved.get(old_fid, old_fid)
            new_r = resolved.get(new_fid, new_fid)
            if old_r == new_r:
                continue
            resolved_pairs.append((old_r, new_r))
        if not resolved_pairs:
            return 0
        total_deleted = 0
        now_iso = datetime.now().isoformat()
        with self._write_lock:
            conn = self._connect()
            try:
                for old_fid, new_fid in resolved_pairs:
                    # Get new abs_id
                    row = conn.execute(
                        "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                        (new_fid, self._graph_id),
                    ).fetchone()
                    if not row:
                        continue
                    new_abs_id = row["uuid"]
                    # Get old abs_ids
                    old_rows = conn.execute(
                        "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?",
                        (old_fid, self._graph_id),
                    ).fetchall()
                    old_abs_ids = [r["uuid"] for r in old_rows]
                    # Redirect relations
                    for old_aid in old_abs_ids:
                        conn.execute(
                            "UPDATE relation SET entity1_absolute_id = ? WHERE entity1_absolute_id = ? AND graph_id = ?",
                            (new_abs_id, old_aid, self._graph_id),
                        )
                        conn.execute(
                            "UPDATE relation SET entity2_absolute_id = ? WHERE entity2_absolute_id = ? AND graph_id = ?",
                            (new_abs_id, old_aid, self._graph_id),
                        )
                    # Delete old entities
                    for old_aid in old_abs_ids:
                        conn.execute("DELETE FROM relates_to WHERE entity1_uuid = ? OR entity2_uuid = ?", (old_aid, old_aid))
                    cnt = conn.execute(
                        "DELETE FROM entity WHERE family_id = ? AND graph_id = ?",
                        (old_fid, self._graph_id),
                    ).rowcount
                    total_deleted += cnt
                    # Register redirect
                    conn.execute(
                        "INSERT OR REPLACE INTO entity_redirect (source_id, target_id, updated_at) VALUES (?, ?, ?)",
                        (old_fid, new_fid, now_iso),
                    )
                conn.commit()
            finally:
                conn.rollback()
        self._cache.invalidate("sim_search:")
        self._cache.invalidate_keys(["graph_stats"])
        return total_deleted

    # ------------------------------------------------------------------
    # Time point helper
    # ------------------------------------------------------------------

    @staticmethod
    def _tp_to_datetime(tp):
        return _cached_tp_to_datetime(tp)
