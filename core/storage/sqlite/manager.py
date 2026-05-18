"""SQLite-based native graph storage manager for Deep Dream.

Drop-in replacement for Neo4jStorageManager. Implements all methods across
10 Neo4j mixin interfaces (Base, Entity, Relation, Episode, Search, Stats,
GraphTraversal, Community, Dream, Concepts) using SQLite + FTS5 + numpy
brute-force cosine similarity.
"""
import hashlib
import json
import logging
import random
import shutil
import sqlite3
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np

from ...models import Concept, ContentPatch, Entity, Episode, Relation
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
# SQLiteGraphStorageManager
# ===========================================================================


class SQLiteGraphStorageManager:
    """SQLite-based graph storage replacing Neo4j.

    Constructor mirrors the factory signature in ``core/storage/__init__.py``.
    """

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

    def close(self):
        """Cleanup resources."""
        pass

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

    # ==================================================================
    # ENTITY STORE
    # ==================================================================

    def _compute_entity_embedding(self, entity: Entity) -> Optional[tuple]:
        content = entity.content or ""
        if len(content) > _EMB_CONTENT_MAX:
            content = content[:_EMB_CONTENT_MAX]
        text = f"# {entity.name}\n{content}"
        return _encode_and_normalize(self.embedding_client, text)

    def _bulk_compute_entity_embeddings(self, entities: List[Entity]) -> Dict[str, bytes]:
        if not self.embedding_client or not self.embedding_client.is_available():
            return {}
        texts = []
        uuids = []
        for e in entities:
            content = e.content or ""
            if len(content) > _EMB_CONTENT_MAX:
                content = content[:_EMB_CONTENT_MAX]
            texts.append(f"# {e.name}\n{content}")
            uuids.append(e.absolute_id)
        try:
            embeddings = self.embedding_client.encode(texts)
        except Exception:
            return {}
        if embeddings is None:
            return {}
        result = {}
        for idx, uuid in enumerate(uuids):
            try:
                emb = np.array(embeddings[idx], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                result[uuid] = emb.tobytes()
            except Exception:
                pass
        return result

    # --- Cache helpers ---

    def _update_entity_emb_cache(self, entity: Entity, emb_array: Optional[np.ndarray]):
        if self._entity_emb_cache is None:
            return
        if self._entity_emb_fid_idx is None:
            self._entity_emb_fid_idx = {e.family_id: i for i, (e, _) in enumerate(self._entity_emb_cache)}
        idx = self._entity_emb_fid_idx.get(entity.family_id)
        if idx is not None:
            self._entity_emb_cache[idx] = (entity, emb_array)
        else:
            self._entity_emb_cache.append((entity, emb_array))
            self._entity_emb_fid_idx[entity.family_id] = len(self._entity_emb_cache) - 1

    def _update_entity_emb_cache_batch(self, items: List[tuple]):
        if self._entity_emb_cache is None or not items:
            return
        if self._entity_emb_fid_idx is not None:
            fid_to_idx = self._entity_emb_fid_idx
        else:
            fid_to_idx = {e.family_id: i for i, (e, _) in enumerate(self._entity_emb_cache)}
            self._entity_emb_fid_idx = fid_to_idx
        for entity, emb_array in items:
            idx = fid_to_idx.get(entity.family_id)
            if idx is not None:
                self._entity_emb_cache[idx] = (entity, emb_array)
            else:
                self._entity_emb_cache.append((entity, emb_array))
                fid_to_idx[entity.family_id] = len(self._entity_emb_cache) - 1

    def _invalidate_entity_cache(self, family_id: str):
        self._cache.invalidate_keys([
            f"entity:by_fid:{family_id}",
            f"resolve:{family_id}",
        ])
        # Invalidate all absolute-id caches for this family (version chain)
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?",
                (family_id, self._graph_id),
            ).fetchall()
        finally:
            conn.rollback()
        if rows:
            self._cache.invalidate_keys([f"entity:by_abs:{r[0]}" for r in rows])

    def _invalidate_entity_cache_bulk(self):
        self._cache.invalidate("entity:")
        self._cache.invalidate("resolve:")
        self._cache.invalidate("sim_search:")
        self.invalidate_entity_remap_cache()

    def _get_entities_with_embeddings(self) -> List[tuple]:
        now = time.time()
        if self._entity_emb_cache is not None and (now - self._entity_emb_cache_ts) < self._emb_cache_ttl:
            return self._entity_emb_cache
        with _perf_timer("_get_entities_with_embeddings"):
            result = self._get_entities_with_embeddings_impl()
        self._entity_emb_cache = result
        self._entity_emb_fid_idx = None
        self._entity_emb_cache_ts = time.time()
        # Build HNSW index alongside cache
        if result and self._vector_dim > 0:
            self._entity_hnsw, self._entity_hnsw_items = self._build_hnsw(result, self._vector_dim)
        else:
            self._entity_hnsw = None
            self._entity_hnsw_items = None
        return result

    def _get_entities_with_embeddings_impl(self) -> List[tuple]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT e.* FROM entity e "
                "INNER JOIN ("
                "  SELECT family_id, MAX(version_seq) AS max_vs FROM entity "
                "  WHERE graph_id = ? GROUP BY family_id"
                ") latest ON e.family_id = latest.family_id AND e.version_seq = latest.max_vs "
                "WHERE e.graph_id = ? ORDER BY e.processed_time DESC LIMIT ?",
                (self._graph_id, self._graph_id, self._emb_cache_max_size),
            ).fetchall()
        finally:
            conn.rollback()
        entities = []
        for row in rows:
            entity = _row_to_entity(dict(row))
            emb_array = np.frombuffer(entity.embedding, dtype=np.float32) if entity.embedding else None
            entities.append((entity, emb_array))
        return entities

    # --- Save ---

    def save_entity(self, entity: Entity, _precomputed_embedding=None):
        with _perf_timer("save_entity"):
            _emb_array = None
            if _precomputed_embedding is not None:
                embedding_blob = _precomputed_embedding
            else:
                _emb_result = self._compute_entity_embedding(entity)
                if _emb_result is not None:
                    embedding_blob, _emb_array = _emb_result
                else:
                    embedding_blob = None
            entity.embedding = embedding_blob
            entity.processed_time = datetime.now()
            valid_at = _fmt_dt(entity.valid_at or entity.event_time)
            attrs = json.dumps(entity.attributes, ensure_ascii=False) if isinstance(entity.attributes, (dict, list)) else entity.attributes
            with self._write_lock:
                conn = self._connect()
                try:
                    # Compute version_seq: max existing + 1
                    if entity.version_seq <= 1:
                        row = conn.execute(
                            "SELECT MAX(version_seq) FROM entity WHERE family_id = ? AND graph_id = ?",
                            (entity.family_id, self._graph_id),
                        ).fetchone()
                        entity.version_seq = (row[0] or 0) + 1
                    conn.execute(
                        f"INSERT OR REPLACE INTO entity ({', '.join(ENTITY_COLUMNS)}) VALUES ({', '.join('?' * len(ENTITY_COLUMNS))})",
                        (
                            entity.absolute_id, entity.family_id, self._graph_id,
                            entity.name, entity.content, entity.summary,
                            attrs, entity.confidence,
                            getattr(entity, "content_format", "plain"),
                            entity.community_id,
                            entity.version_seq,
                            valid_at,
                            _fmt_dt(entity.event_time), _fmt_dt(entity.processed_time),
                            entity.episode_id, entity.source_document,
                            embedding_blob,
                        ),
                    )
                    # Update FTS
                    conn.execute("DELETE FROM entity_fts WHERE rowid = (SELECT rowid FROM entity WHERE uuid = ?)",
                                 (entity.absolute_id,))
                    conn.execute(
                        "INSERT INTO entity_fts (rowid, name, content, graph_id) VALUES ((SELECT rowid FROM entity WHERE uuid = ?), ?, ?, ?)",
                        (entity.absolute_id, entity.name, entity.content, self._graph_id),
                    )
                    conn.commit()
                finally:
                    conn.rollback()
            # Cache update
            if embedding_blob:
                emb_array = _emb_array if _emb_array is not None else np.frombuffer(embedding_blob, dtype=np.float32)
                self._update_entity_emb_cache(entity, emb_array)
            else:
                self._update_entity_emb_cache(entity, None)
            self._cache_entity_name(entity.absolute_id, entity.name)
        self._invalidate_entity_cache(entity.family_id)
        self._cache.invalidate("sim_search:")

    def bulk_save_entities(self, entities: List[Entity]):
        if not entities:
            return
        _now = datetime.now()
        # Compute embeddings synchronously before writing
        emb_map = self._bulk_compute_entity_embeddings(entities)
        rows = []
        cache_items = []
        with self._write_lock:
            conn = self._connect()
            try:
                # Batch compute version_seq for all family_ids
                fid_map = {}
                for entity in entities:
                    fid_map.setdefault(entity.family_id, []).append(entity)
                for fid, fid_entities in fid_map.items():
                    row = conn.execute(
                        "SELECT MAX(version_seq) FROM entity WHERE family_id = ? AND graph_id = ?",
                        (fid, self._graph_id),
                    ).fetchone()
                    next_seq = (row[0] or 0) + 1
                    for entity in fid_entities:
                        if entity.version_seq <= 1:
                            entity.version_seq = next_seq
                            next_seq += 1
            finally:
                conn.rollback()
        for entity in entities:
            entity.processed_time = _now
            emb_blob = emb_map.get(entity.absolute_id)
            entity.embedding = emb_blob
            attrs = json.dumps(entity.attributes, ensure_ascii=False) if isinstance(entity.attributes, (dict, list)) else entity.attributes
            rows.append((
                entity.absolute_id, entity.family_id, self._graph_id,
                entity.name, entity.content, entity.summary, attrs,
                entity.confidence, getattr(entity, "content_format", "plain"),
                entity.community_id,
                entity.version_seq,
                _fmt_dt(entity.valid_at or entity.event_time),
                _fmt_dt(entity.event_time), _fmt_dt(entity.processed_time),
                entity.episode_id, entity.source_document, emb_blob,
            ))
            if emb_blob is not None:
                cache_items.append((entity, np.frombuffer(emb_blob, dtype=np.float32)))
            else:
                cache_items.append((entity, None))
        with self._write_lock:
            conn = self._connect()
            try:
                conn.executemany(
                    f"INSERT OR REPLACE INTO entity ({', '.join(ENTITY_COLUMNS)}) VALUES ({', '.join('?' * len(ENTITY_COLUMNS))})",
                    rows,
                )
                conn.commit()
            finally:
                conn.rollback()
        self._update_entity_emb_cache_batch(cache_items)
        for entity in entities:
            self._invalidate_entity_cache(entity.family_id)
            self._cache_entity_name(entity.absolute_id, entity.name)

    def bulk_save_entities_with_embedding(self, entities: List[Entity]):
        if not entities:
            return
        _now = datetime.now()
        rows = []
        cache_items = []
        # Pre-compute version_seq per family_id
        with self._write_lock:
            conn = self._connect()
            try:
                fid_map = {}
                for entity in entities:
                    fid_map.setdefault(entity.family_id, []).append(entity)
                for fid, fid_entities in fid_map.items():
                    row = conn.execute(
                        "SELECT MAX(version_seq) FROM entity WHERE family_id = ? AND graph_id = ?",
                        (fid, self._graph_id),
                    ).fetchone()
                    next_seq = (row[0] or 0) + 1
                    for entity in fid_entities:
                        if entity.version_seq <= 1:
                            entity.version_seq = next_seq
                            next_seq += 1
            finally:
                conn.rollback()
        for entity in entities:
            entity.processed_time = _now
            emb_blob = getattr(entity, 'embedding', None)
            emb_array = None
            if emb_blob is not None:
                if isinstance(emb_blob, np.ndarray):
                    emb_array = emb_blob
                else:
                    emb_array = np.frombuffer(emb_blob, dtype=np.float32)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_array = emb_array / norm
                entity.embedding = emb_array.tobytes()
                cache_items.append((entity, emb_array))
            else:
                cache_items.append((entity, None))
            attrs = json.dumps(entity.attributes, ensure_ascii=False) if isinstance(entity.attributes, (dict, list)) else entity.attributes
            rows.append((
                entity.absolute_id, entity.family_id, self._graph_id,
                entity.name, entity.content, entity.summary, attrs,
                entity.confidence, getattr(entity, "content_format", "plain"),
                entity.community_id,
                entity.version_seq,
                _fmt_dt(entity.valid_at or entity.event_time),
                _fmt_dt(entity.event_time), _fmt_dt(entity.processed_time),
                entity.episode_id, entity.source_document, entity.embedding,
            ))
        with self._write_lock:
            conn = self._connect()
            try:
                conn.executemany(
                    f"INSERT OR REPLACE INTO entity ({', '.join(ENTITY_COLUMNS)}) VALUES ({', '.join('?' * len(ENTITY_COLUMNS))})",
                    rows,
                )
                conn.commit()
            finally:
                conn.rollback()
        for entity in entities:
            self._invalidate_entity_cache(entity.family_id)
            self._cache_entity_name(entity.absolute_id, entity.name)
        self._update_entity_emb_cache_batch(cache_items)

    # --- Get ---

    def get_entity_by_absolute_id(self, absolute_id: str) -> Optional[Entity]:
        cache_key = f"entity:by_abs:{absolute_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE uuid = ? AND graph_id = ?",
                (absolute_id, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        if not row:
            return None
        entity = _row_to_entity(dict(row))
        self._cache.set(cache_key, entity, ttl=60)
        return entity

    def get_entity_by_family_id(self, family_id: str) -> Optional[Entity]:
        cache_key = f"entity:by_fid:{family_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        resolved_fid = self.resolve_family_id(family_id)
        if resolved_fid != family_id:
            cache_key2 = f"entity:by_fid:{resolved_fid}"
            cached = self._cache.get(cache_key2)
            if cached is not None:
                self._cache.set(cache_key, cached, ttl=60)
                return cached
        family_id = resolved_fid
        if not family_id:
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                (family_id, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        if not row:
            return None
        entity = _row_to_entity(dict(row))
        self._cache.set(cache_key, entity, ttl=60)
        return entity

    def get_entities_by_family_ids(self, family_ids: List[str]) -> Dict[str, Entity]:
        if not family_ids:
            return {}
        resolved_map = self.resolve_family_ids(list(family_ids))
        valid_fids = set(resolved_map.keys()) | set(resolved_map.values())
        if not valid_fids:
            return {}
        result: Dict[str, Entity] = {}
        uncached = set()
        for fid in valid_fids:
            cached = self._cache.get(f"entity:by_fid:{fid}")
            if cached is not None:
                result[fid] = cached
            else:
                uncached.add(fid)
        if uncached:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(uncached))
                rows = conn.execute(
                    f"SELECT e.* FROM entity e "
                    f"INNER JOIN ("
                    f"  SELECT family_id, MAX(version_seq) AS max_vs FROM entity "
                    f"  WHERE family_id IN ({placeholders}) AND graph_id = ? GROUP BY family_id"
                    f") latest ON e.family_id = latest.family_id AND e.version_seq = latest.max_vs "
                    f"WHERE e.graph_id = ?",
                    list(uncached) + [self._graph_id, self._graph_id],
                ).fetchall()
            finally:
                conn.rollback()
            for row in rows:
                entity = _row_to_entity(dict(row))
                result[entity.family_id] = entity
                self._cache.set(f"entity:by_fid:{entity.family_id}", entity, ttl=60)
        for orig_fid, resolved_fid in resolved_map.items():
            if resolved_fid in result and orig_fid not in result:
                result[orig_fid] = result[resolved_fid]
        return result

    def get_entities_by_absolute_ids(self, absolute_ids: List[str], valid_only: bool = False) -> List[Entity]:
        if not absolute_ids:
            return []
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(absolute_ids))
            extra = " AND version_seq = (SELECT MAX(e2.version_seq) FROM entity e2 WHERE e2.family_id = entity.family_id AND e2.graph_id = entity.graph_id)" if valid_only else ""
            rows = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE uuid IN ({placeholders}) AND graph_id = ?{extra}",
                absolute_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        return [_row_to_entity(dict(r)) for r in rows]

    def get_latest_absolute_ids_by_family_ids(self, family_ids: List[str]) -> Dict[str, str]:
        if not family_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(family_ids))
            rows = conn.execute(
                f"SELECT family_id, uuid FROM entity "
                f"WHERE family_id IN ({placeholders}) AND graph_id = ? "
                f"GROUP BY family_id HAVING version_seq = MAX(version_seq)",
                family_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        return {r["family_id"]: r["uuid"] for r in rows}

    def _get_all_absolute_ids_for_entity(self, family_id: str) -> List[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?",
                (family_id, self._graph_id),
            ).fetchall()
        finally:
            conn.rollback()
        return [r["uuid"] for r in rows]

    def get_entity_versions(self, family_id: str) -> List[Entity]:
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY processed_time ASC",
                (family_id, self._graph_id),
            ).fetchall()
        finally:
            conn.rollback()
        return [_row_to_entity(dict(r)) for r in rows]

    def get_entity_versions_batch(self, family_ids: List[str]) -> Dict[str, List[Entity]]:
        if not family_ids:
            return {}
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = list({r for r in resolved_map.values() if r})
        if not canonical_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(canonical_ids))
            rows = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE family_id IN ({placeholders}) AND graph_id = ? ORDER BY processed_time ASC",
                canonical_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        versions_map: Dict[str, List[Entity]] = {fid: [] for fid in canonical_ids}
        for row in rows:
            entity = _row_to_entity(dict(row))
            if entity.family_id in versions_map:
                versions_map[entity.family_id].append(entity)
        return {orig: versions_map.get(resolved, []) for orig, resolved in resolved_map.items() if resolved}

    def get_entity_version_count(self, family_id: str) -> int:
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM entity WHERE family_id = ? AND graph_id = ?",
                (family_id, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def get_entity_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
        if not family_ids:
            return {}
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = list({r for r in resolved_map.values() if r})
        if not canonical_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(canonical_ids))
            rows = conn.execute(
                "SELECT family_id, COUNT(*) AS cnt FROM entity WHERE family_id IN ({}) AND graph_id = ? GROUP BY family_id".format(placeholders),
                canonical_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        counts = {r["family_id"]: r["cnt"] for r in rows}
        return {fid: counts.get(canonical, 0) for fid, canonical in resolved_map.items() if canonical}

    def get_entity_absolute_ids_up_to_version(self, family_id: str, max_absolute_id: str) -> List[str]:
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT processed_time FROM entity WHERE uuid = ? AND graph_id = ?",
                (max_absolute_id, self._graph_id),
            ).fetchone()
            if not row:
                return []
            max_pt = row["processed_time"]
            rows = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND processed_time <= ? AND graph_id = ? ORDER BY processed_time ASC",
                (family_id, max_pt, self._graph_id),
            ).fetchall()
        finally:
            conn.rollback()
        return [r["uuid"] for r in rows]

    def get_entity_version_at_time(self, family_id: str, time_point: datetime) -> Optional[Entity]:
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE family_id = ? AND event_time <= ? AND graph_id = ? ORDER BY processed_time DESC LIMIT 1",
                (family_id, time_point.isoformat(), self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        if not row:
            return None
        return _row_to_entity(dict(row))

    # --- Relations for entity ---

    def get_entity_relations(self, entity_absolute_id: str, limit: Optional[int] = None,
                              time_point: Optional[datetime] = None,
                              include_candidates: bool = False) -> List[Relation]:
        conn = self._connect()
        try:
            tp_filter = " AND r.event_time <= ?" if time_point else ""
            params: list = [entity_absolute_id, entity_absolute_id, self._graph_id]
            if time_point:
                params.append(time_point.isoformat())
            query = (
                "SELECT r.* FROM relation r "
                "WHERE (r.entity1_absolute_id = ? OR r.entity2_absolute_id = ?) "
                "AND r.graph_id = ? "
                f"{tp_filter} "
                "ORDER BY r.version_seq DESC"
            )
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        relations = [_row_to_relation(dict(r)) for r in rows]
        return self._filter_dream_candidates(relations, include_candidates)

    def get_entity_relations_by_family_id(self, family_id: str, limit: Optional[int] = None,
                                           time_point: Optional[datetime] = None,
                                           max_version_absolute_id: Optional[str] = None,
                                           include_candidates: bool = False) -> List[Relation]:
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        conn = self._connect()
        try:
            # Get all abs_ids for this family (current versions only)
            abs_rows = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC",
                (family_id, self._graph_id),
            ).fetchall()
            abs_ids = [r["uuid"] for r in abs_rows]
        finally:
            conn.rollback()
        if not abs_ids:
            return []
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(abs_ids))
            query = (
                f"SELECT r.* FROM relation r "
                f"WHERE (r.entity1_absolute_id IN ({placeholders}) OR r.entity2_absolute_id IN ({placeholders})) "
                f"AND r.graph_id = ? "
            )
            params: list = abs_ids + abs_ids + [self._graph_id]
            if time_point:
                query += " AND r.event_time <= ?"
                params.append(time_point.isoformat())
            query += " ORDER BY r.processed_time DESC"
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        relations = [_row_to_relation(dict(r)) for r in rows]
        return self._filter_dream_candidates(relations, include_candidates)

    def get_entity_relations_timeline(self, family_id: str, version_abs_ids: List[str]) -> List[Dict]:
        family_id = self.resolve_family_id(family_id)
        if not family_id or not version_abs_ids:
            return []
        conn = self._connect()
        try:
            abs_rows = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?",
                (family_id, self._graph_id),
            ).fetchall()
            abs_ids = [r["uuid"] for r in abs_rows]
            if not abs_ids:
                return []
            placeholders_abs = ",".join("?" * len(abs_ids))
            placeholders_ver = ",".join("?" * len(version_abs_ids))
            version_rows = conn.execute(
                f"SELECT uuid, processed_time FROM entity WHERE uuid IN ({placeholders_ver}) AND graph_id = ? ORDER BY processed_time ASC",
                version_abs_ids + [self._graph_id],
            ).fetchall()
            version_times = [(r["uuid"], r["processed_time"]) for r in version_rows]
            if not version_times:
                return []
            rel_rows = conn.execute(
                f"SELECT uuid, family_id, content, event_time, processed_time FROM relation "
                f"WHERE (entity1_absolute_id IN ({placeholders_abs}) OR entity2_absolute_id IN ({placeholders_abs})) "
                f"AND graph_id = ?",
                abs_ids + abs_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        timeline = []
        seen = set()
        for rel in rel_rows:
            rel_uuid = rel["uuid"]
            if rel_uuid in seen:
                continue
            rel_pt = rel["processed_time"]
            for _, v_pt in version_times:
                if rel_pt and v_pt and rel_pt <= v_pt:
                    seen.add(rel_uuid)
                    timeline.append({
                        "family_id": rel["family_id"],
                        "content": rel["content"],
                        "event_time": rel["event_time"],
                        "absolute_id": rel_uuid,
                    })
                    break
        return timeline

    def count_entity_relations_by_family_ids(self, family_ids: List[str]) -> Dict[str, int]:
        if not family_ids:
            return {}
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = list({r for r in resolved_map.values() if r})
        if not canonical_ids:
            return {}
        conn = self._connect()
        try:
            counts = {}
            for fid in canonical_ids:
                abs_rows = conn.execute(
                    "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?",
                    (fid, self._graph_id),
                ).fetchall()
                abs_ids = [r["uuid"] for r in abs_rows]
                if abs_ids:
                    placeholders = ",".join("?" * len(abs_ids))
                    row = conn.execute(
                        f"SELECT COUNT(*) AS cnt FROM relation WHERE (entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})) AND graph_id = ?",
                        abs_ids + abs_ids + [self._graph_id],
                    ).fetchone()
                    counts[fid] = row["cnt"] if row else 0
                else:
                    counts[fid] = 0
        finally:
            conn.rollback()
        return {orig: counts.get(resolved, 0) for orig, resolved in resolved_map.items() if resolved}

    # --- Batch profiles ---

    def batch_get_entity_profiles(self, family_ids: List[str]) -> List[Dict[str, Any]]:
        if not family_ids:
            return []
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_map: Dict[str, str] = {}
        canonical_set: List[str] = []
        _seen_resolved: set = set()
        for fid in family_ids:
            resolved = resolved_map.get(fid, fid)
            if resolved and resolved not in _seen_resolved:
                canonical_map[fid] = resolved
                canonical_set.append(resolved)
                _seen_resolved.add(resolved)
        if not canonical_set:
            return [{"family_id": fid, "entity": None, "relations": [], "version_count": 0} for fid in family_ids]

        entity_map: Dict[str, tuple] = {}
        fid_to_aids: Dict[str, List[str]] = {}
        all_aids: Set[str] = set()

        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(canonical_set))
            rows = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE family_id IN ({placeholders}) AND graph_id = ? ORDER BY version_seq DESC",
                canonical_set + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()

        for row in rows:
            entity = _row_to_entity(dict(row))
            fid = entity.family_id
            if fid not in entity_map:
                entity_map[fid] = (entity, 1)
                fid_to_aids[fid] = [entity.absolute_id]
                all_aids.add(entity.absolute_id)
            else:
                ent, vc = entity_map[fid]
                entity_map[fid] = (ent, vc + 1)
                fid_to_aids[fid].append(entity.absolute_id)
                all_aids.add(entity.absolute_id)

        relations_map: Dict[str, List] = {fid: [] for fid in canonical_set}
        if all_aids:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(all_aids))
                rel_rows = conn.execute(
                    f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE (entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})) AND graph_id = ?",
                    list(all_aids) + list(all_aids) + [self._graph_id],
                ).fetchall()
            finally:
                conn.rollback()
            aid_to_fid = {}
            for fid, aids in fid_to_aids.items():
                for aid in aids:
                    aid_to_fid[aid] = fid
            for r in rel_rows:
                rel = _row_to_relation(dict(r))
                fid1 = aid_to_fid.get(rel.entity1_absolute_id)
                fid2 = aid_to_fid.get(rel.entity2_absolute_id)
                if fid1:
                    relations_map[fid1].append(rel)
                if fid2 and fid2 != fid1:
                    relations_map[fid2].append(rel)

        results = []
        seen_fids = set()
        for fid in family_ids:
            canonical = canonical_map.get(fid, fid)
            if canonical in seen_fids:
                results.append({"family_id": fid, "entity": None, "relations": [], "version_count": 0})
                continue
            seen_fids.add(canonical)
            if canonical in entity_map:
                entity, vc = entity_map[canonical]
                results.append({
                    "family_id": canonical,
                    "entity": entity,
                    "relations": relations_map.get(canonical, []),
                    "version_count": vc,
                })
            else:
                results.append({"family_id": fid, "entity": None, "relations": [], "version_count": 0})
        return results

    # --- Delete ---

    def delete_entity_all_versions(self, family_id: str) -> int:
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        abs_ids = self._get_all_absolute_ids_for_entity(family_id)
        with self._write_lock:
            conn = self._connect()
            try:
                for aid in abs_ids:
                    conn.execute("DELETE FROM relates_to WHERE entity1_uuid = ? OR entity2_uuid = ?", (aid, aid))
                count = conn.execute(
                    "DELETE FROM entity WHERE family_id = ? AND graph_id = ?",
                    (family_id, self._graph_id),
                ).rowcount
                conn.commit()
            finally:
                conn.rollback()
            self._invalidate_entity_cache(family_id)
            self._cache.invalidate("sim_search:")
            self._cache.invalidate_keys(["graph_stats"])
        return count

    def delete_entity_by_absolute_id(self, absolute_id: str) -> bool:
        deleted = False
        fid = None
        with self._write_lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (absolute_id, self._graph_id)).fetchone()
                if row:
                    fid = row["family_id"]
                    conn.execute("DELETE FROM relates_to WHERE entity1_uuid = ? OR entity2_uuid = ?", (absolute_id, absolute_id))
                    deleted = conn.execute("DELETE FROM entity WHERE uuid = ? AND graph_id = ?", (absolute_id, self._graph_id)).rowcount > 0
                conn.commit()
            finally:
                conn.rollback()
        if fid:
            self._invalidate_entity_cache(fid)
        else:
            self._invalidate_entity_cache_bulk()
        self._cache.invalidate_keys(["graph_stats"])
        return deleted

    def batch_delete_entities(self, family_ids: List[str]) -> int:
        resolved_map = self.resolve_family_ids(family_ids)
        resolved = list({r for r in resolved_map.values() if r})
        if not resolved:
            return 0
        count = 0
        with self._write_lock:
            conn = self._connect()
            try:
                for fid in resolved:
                    abs_rows = conn.execute("SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?", (fid, self._graph_id)).fetchall()
                    for r in abs_rows:
                        conn.execute("DELETE FROM relates_to WHERE entity1_uuid = ? OR entity2_uuid = ?", (r["uuid"], r["uuid"]))
                    count += conn.execute("DELETE FROM entity WHERE family_id = ? AND graph_id = ?", (fid, self._graph_id)).rowcount
                conn.commit()
            finally:
                conn.rollback()
            self._invalidate_entity_cache_bulk()
            self._cache.invalidate_keys(["graph_stats"])
        return count

    def batch_delete_entity_versions_by_absolute_ids(self, absolute_ids: List[str]) -> int:
        if not absolute_ids:
            return 0
        with self._write_lock:
            conn = self._connect()
            try:
                for aid in absolute_ids:
                    conn.execute("DELETE FROM relates_to WHERE entity1_uuid = ? OR entity2_uuid = ?", (aid, aid))
                placeholders = ",".join("?" * len(absolute_ids))
                deleted = conn.execute(
                    f"DELETE FROM entity WHERE uuid IN ({placeholders}) AND graph_id = ?",
                    absolute_ids + [self._graph_id],
                ).rowcount
                conn.commit()
            finally:
                conn.rollback()
            self._invalidate_entity_cache_bulk()
            self._cache.invalidate_keys(["graph_stats"])
        return deleted

    def delete_entity_by_id(self, family_id: str) -> int:
        return self.delete_entity_all_versions(family_id)

    # --- Update ---

    def update_entity_by_absolute_id(self, absolute_id: str, **fields) -> Optional[Entity]:
        valid_keys = {"name", "content", "summary", "attributes", "confidence"}
        filtered = {k: v for k, v in fields.items() if k in valid_keys and v is not None}
        if not filtered:
            return self.get_entity_by_absolute_id(absolute_id)
        needs_emb_update = "name" in filtered or "content" in filtered
        _precomputed_emb = None
        if needs_emb_update and self.embedding_client and self.embedding_client.is_available():
            current = self.get_entity_by_absolute_id(absolute_id)
            if current:
                merged_name = filtered.get("name", current.name)
                merged_content = filtered.get("content", current.content)
                if len(merged_content or "") > _EMB_CONTENT_MAX:
                    merged_content = merged_content[:_EMB_CONTENT_MAX]
                text = f"# {merged_name}\n{merged_content}"
                _emb_result = _encode_and_normalize(self.embedding_client, text)
                if _emb_result is not None:
                    _precomputed_emb = _emb_result
        embedding_blob = _precomputed_emb[0] if _precomputed_emb else None
        with self._write_lock:
            conn = self._connect()
            try:
                set_parts = [f"{k} = ?" for k in filtered]
                params = list(filtered.values())
                if _precomputed_emb:
                    set_parts.append("embedding = ?")
                    params.append(embedding_blob)
                params.append(absolute_id)
                params.append(self._graph_id)
                conn.execute(
                    f"UPDATE entity SET {', '.join(set_parts)} WHERE uuid = ? AND graph_id = ?",
                    params,
                )
                conn.commit()
            finally:
                conn.rollback()
        # Invalidate caches BEFORE re-reading
        self._cache.invalidate(f"entity:by_abs:{absolute_id}")
        entity = self.get_entity_by_absolute_id(absolute_id)
        if entity:
            self._invalidate_entity_cache(entity.family_id)
            if _precomputed_emb:
                entity.embedding = embedding_blob
                self._update_entity_emb_cache(entity, _precomputed_emb[1])
        return entity

    def update_entity_confidence(self, family_id: str, confidence: float):
        confidence = max(0.0, min(1.0, confidence))
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                (family_id, self._graph_id),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE entity SET confidence = ? WHERE uuid = ? AND graph_id = ?",
                    (confidence, row["uuid"], self._graph_id),
                )
                conn.commit()
        finally:
            conn.rollback()
        self._invalidate_entity_cache(family_id)

    def update_entity_summary(self, family_id: str, summary: str):
        resolved = self.resolve_family_id(family_id)
        if not resolved:
            return
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                (resolved, self._graph_id),
            ).fetchone()
            if row:
                conn.execute("UPDATE entity SET summary = ? WHERE uuid = ? AND graph_id = ?", (summary, row["uuid"], self._graph_id))
                conn.commit()
        finally:
            conn.rollback()
        self._invalidate_entity_cache(resolved)

    def batch_update_entity_summaries(self, updates: Dict[str, str]):
        if not updates:
            return
        resolved_map = self.resolve_family_ids(list(updates))
        conn = self._connect()
        try:
            for orig_fid, summary in updates.items():
                resolved = resolved_map.get(orig_fid)
                if resolved:
                    row = conn.execute(
                        "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                        (resolved, self._graph_id),
                    ).fetchone()
                    if row:
                        conn.execute("UPDATE entity SET summary = ? WHERE uuid = ? AND graph_id = ?", (summary, row["uuid"], self._graph_id))
            conn.commit()
        finally:
            conn.rollback()
        self._invalidate_entity_cache_bulk()

    def update_entity_attributes(self, family_id: str, attributes: str):
        resolved = self.resolve_family_id(family_id)
        if not resolved:
            return
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                (resolved, self._graph_id),
            ).fetchone()
            if row:
                conn.execute("UPDATE entity SET attributes = ? WHERE uuid = ? AND graph_id = ?", (attributes, row["uuid"], self._graph_id))
                conn.commit()
        finally:
            conn.rollback()
        self._invalidate_entity_cache(resolved)

    def split_entity_version(self, absolute_id: str, new_family_id: str = "") -> Optional[Entity]:
        if not new_family_id:
            new_family_id = f"ent_{uuid.uuid4().hex[:12]}"
        with self._write_lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE entity SET family_id = ? WHERE uuid = ? AND graph_id = ?",
                    (new_family_id, absolute_id, self._graph_id),
                )
                conn.commit()
                row = conn.execute(
                    f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE uuid = ? AND graph_id = ?",
                    (absolute_id, self._graph_id),
                ).fetchone()
            finally:
                conn.rollback()
        if not row:
            return None
        entity = _row_to_entity(dict(row))
        self._invalidate_entity_cache(new_family_id)
        return entity

    # --- Confidence ---

    def adjust_confidence_on_contradiction(self, family_id: str, source_type: str = "entity"):
        table = "entity" if source_type == "entity" else "relation"
        conn = self._connect()
        try:
            conn.execute(
                f"UPDATE {table} SET confidence = MAX(confidence - 0.1, 0.0) "
                f"WHERE family_id = ? AND confidence IS NOT NULL AND graph_id = ? "
                f"AND uuid = (SELECT uuid FROM {table} WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1)",
                (family_id, self._graph_id, family_id, self._graph_id),
            )
            conn.commit()
        finally:
            conn.rollback()
        if source_type == "entity":
            self._invalidate_entity_cache(family_id)
        else:
            self._invalidate_relation_cache_bulk()

    def adjust_confidence_on_contradiction_batch(self, family_ids: List[str], source_type: str = "entity"):
        if not family_ids:
            return
        table = "entity" if source_type == "entity" else "relation"
        conn = self._connect()
        try:
            for fid in family_ids:
                conn.execute(
                    f"UPDATE {table} SET confidence = MAX(confidence - 0.1, 0.0) "
                    f"WHERE family_id = ? AND confidence IS NOT NULL AND graph_id = ? "
                    f"AND uuid = (SELECT uuid FROM {table} WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1)",
                    (fid, self._graph_id, fid, self._graph_id),
                )
            conn.commit()
        finally:
            conn.rollback()
        if source_type == "entity":
            self._invalidate_entity_cache_bulk()
        else:
            self._invalidate_relation_cache_bulk()

    def adjust_confidence_on_corroboration(self, family_id: str, source_type: str = "entity", is_dream: bool = False):
        table = "entity" if source_type == "entity" else "relation"
        delta = 0.025 if is_dream else 0.05
        conn = self._connect()
        try:
            conn.execute(
                f"UPDATE {table} SET confidence = MIN(confidence + ?, 1.0) "
                f"WHERE family_id = ? AND confidence IS NOT NULL AND graph_id = ? "
                f"AND uuid = (SELECT uuid FROM {table} WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1)",
                (delta, family_id, self._graph_id, family_id, self._graph_id),
            )
            conn.commit()
        finally:
            conn.rollback()
        if source_type == "entity":
            self._invalidate_entity_cache(family_id)
        else:
            self._invalidate_relation_cache_bulk()

    def adjust_confidence_on_corroboration_batch(self, family_ids: List[str], source_type: str = "entity", is_dream: bool = False):
        if not family_ids:
            return
        table = "entity" if source_type == "entity" else "relation"
        delta = 0.025 if is_dream else 0.05
        conn = self._connect()
        try:
            for fid in family_ids:
                conn.execute(
                    f"UPDATE {table} SET confidence = MIN(confidence + ?, 1.0) "
                    f"WHERE family_id = ? AND confidence IS NOT NULL AND graph_id = ? "
                    f"AND uuid = (SELECT uuid FROM {table} WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1)",
                    (delta, fid, self._graph_id, fid, self._graph_id),
                )
            conn.commit()
        finally:
            conn.rollback()
        if source_type == "entity":
            self._invalidate_entity_cache_bulk()
        else:
            self._invalidate_relation_cache_bulk()

    # --- Content patches ---

    def save_content_patches(self, patches: list):
        if not patches:
            return
        rows = []
        now_iso = datetime.now().isoformat()
        for p in patches:
            if isinstance(p, dict):
                rows.append((
                    p["uuid"], p["target_type"], p["target_absolute_id"],
                    p["target_family_id"], p["section_key"], p["change_type"],
                    p["old_hash"], p["new_hash"], p["diff_summary"],
                    p["source_document"],
                    _fmt_dt(p["event_time"]) if p.get("event_time") else now_iso,
                ))
            else:
                rows.append((
                    p.uuid, p.target_type, p.target_absolute_id,
                    p.target_family_id, p.section_key, p.change_type,
                    p.old_hash, p.new_hash, p.diff_summary,
                    p.source_document,
                    _fmt_dt(p.event_time) if p.event_time else now_iso,
                ))
        conn = self._connect()
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO content_patch (uuid, target_type, target_absolute_id, target_family_id, section_key, change_type, old_hash, new_hash, diff_summary, source_document, event_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        finally:
            conn.rollback()

    def get_content_patches(self, family_id: str, section_key: str = None) -> list:
        conn = self._connect()
        try:
            if section_key:
                rows = conn.execute(
                    "SELECT * FROM content_patch WHERE target_family_id = ? AND section_key = ? ORDER BY event_time DESC",
                    (family_id, section_key),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM content_patch WHERE target_family_id = ? ORDER BY event_time DESC",
                    (family_id,),
                ).fetchall()
        finally:
            conn.rollback()
        patches = []
        for row in rows:
            rd = dict(row)
            patches.append(ContentPatch(
                uuid=rd["uuid"],
                target_type=rd["target_type"],
                target_absolute_id=rd["target_absolute_id"],
                target_family_id=rd["target_family_id"],
                section_key=rd["section_key"],
                change_type=rd["change_type"],
                old_hash=rd.get("old_hash", ""),
                new_hash=rd.get("new_hash", ""),
                diff_summary=rd.get("diff_summary", ""),
                source_document=rd.get("source_document", ""),
                event_time=_parse_dt(rd.get("event_time")),
            ))
        return patches

    def get_section_history(self, family_id: str, section_key: str) -> list:
        return self.get_content_patches(family_id, section_key=section_key)

    # --- Misc queries ---

    def find_entity_by_name_prefix(self, prefix: str, limit: int = 5) -> list:
        if not prefix:
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE (name LIKE ? OR name = ?) AND graph_id = ? ORDER BY version_seq DESC LIMIT ?",
                (prefix + "%", prefix, self._graph_id, limit),
            ).fetchall()
        finally:
            conn.rollback()
        entities = []
        seen_fids = set()
        for row in rows:
            entity = _row_to_entity(dict(row))
            if entity.family_id not in seen_fids:
                seen_fids.add(entity.family_id)
                entities.append(entity)
        return entities

    def get_entity_provenance(self, family_id: str) -> List[dict]:
        conn = self._connect()
        try:
            abs_rows = conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?",
                (family_id, self._graph_id),
            ).fetchall()
            abs_ids = [r["uuid"] for r in abs_rows]
            if not abs_ids:
                return []
            placeholders = ",".join("?" * len(abs_ids))
            mention_rows = conn.execute(
                f"SELECT episode_uuid, context FROM mentions WHERE entity_absolute_id IN ({placeholders}) AND graph_id = ?",
                abs_ids + [self._graph_id],
            ).fetchall()
            if mention_rows:
                return [{"episode_id": r["episode_uuid"], "context": dict(r).get("context", "")} for r in mention_rows]
            # Fallback: indirect via relations
            rel_rows = conn.execute(
                f"SELECT DISTINCT m.episode_uuid, m.context FROM mentions m "
                f"INNER JOIN relation r ON m.target_uuid = r.uuid "
                f"WHERE (r.entity1_absolute_id IN ({placeholders}) OR r.entity2_absolute_id IN ({placeholders})) "
                f"AND m.graph_id = ?",
                abs_ids + abs_ids + [self._graph_id],
            ).fetchall()
            return [{"episode_id": r["episode_uuid"], "context": dict(r).get("context", "")} for r in rel_rows]
        finally:
            conn.rollback()

    def get_entity_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT embedding FROM entity WHERE uuid = ? AND graph_id = ?", (absolute_id, self._graph_id)).fetchone()
        finally:
            conn.rollback()
        if row and row["embedding"]:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            return emb[:num_values].tolist()
        return None

    def get_entity_names_by_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, str]:
        if not absolute_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(absolute_ids))
            rows = conn.execute(f"SELECT uuid, name FROM entity WHERE uuid IN ({placeholders}) AND graph_id = ?", absolute_ids + [self._graph_id]).fetchall()
        finally:
            conn.rollback()
        return {r["uuid"]: r["name"] for r in rows}

    def get_all_entity_names_map(self) -> Dict[str, str]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT uuid, name FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchall()
        finally:
            conn.rollback()
        return {r["uuid"]: r["name"] for r in rows}

    def get_family_ids_by_names(self, names: list) -> dict:
        if not names:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(names))
            rows = conn.execute(
                f"SELECT name, family_id FROM entity WHERE name IN ({placeholders}) AND graph_id = ? ORDER BY version_seq DESC",
                names + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        result = {}
        seen = set()
        for r in rows:
            if r["name"] not in seen:
                seen.add(r["name"])
                result[r["name"]] = r["family_id"]
        return result

    def get_all_entities(self, limit: Optional[int] = None, offset: Optional[int] = None, exclude_embedding: bool = False) -> List[Entity]:
        cols = [c for c in ENTITY_COLUMNS if not (exclude_embedding and c == "embedding")]
        conn = self._connect()
        try:
            query = (
                f"SELECT e.* FROM entity e "
                f"INNER JOIN ("
                f"  SELECT family_id, MAX(version_seq) AS max_vs FROM entity "
                f"  WHERE graph_id = ? GROUP BY family_id"
                f") latest ON e.family_id = latest.family_id AND e.version_seq = latest.max_vs "
                f"WHERE e.graph_id = ? ORDER BY e.processed_time DESC"
            )
            params: list = [self._graph_id, self._graph_id]
            if offset:
                query += f" OFFSET {int(offset)}"
            if limit:
                query += f" LIMIT {int(limit)}"
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [_row_to_entity(dict(r)) for r in rows]

    def stream_all_entities(self, exclude_embedding: bool = True, since: Optional[str] = None):
        entities = self.get_all_entities(exclude_embedding=exclude_embedding)
        # Batch-fetch version counts to avoid N+1 queries
        family_ids = [e.family_id for e in entities]
        vc_map = self.get_entity_version_counts(family_ids) if family_ids else {}
        for entity in entities:
            if since:
                pt = _fmt_dt(entity.processed_time)
                if pt and pt <= since:
                    continue
            yield entity, vc_map.get(entity.family_id, 1)

    def count_entities_since(self, since: str) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT family_id) AS cnt FROM entity "
                "WHERE processed_time > ? AND graph_id = ?",
                (since, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def get_all_entities_before_time(self, time_point: datetime, limit: Optional[int] = None,
                                      exclude_embedding: bool = False) -> List[Entity]:
        conn = self._connect()
        try:
            query = (
                "SELECT e.* FROM entity e "
                "INNER JOIN ("
                "  SELECT family_id, MAX(version_seq) AS max_vs FROM entity "
                "  WHERE event_time <= ? AND graph_id = ? GROUP BY family_id"
                ") latest ON e.family_id = latest.family_id AND e.version_seq = latest.max_vs "
                "WHERE e.graph_id = ? ORDER BY e.processed_time DESC"
            )
            params = [time_point.isoformat(), self._graph_id, self._graph_id]
            if limit:
                query += f" LIMIT {int(limit)}"
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [_row_to_entity(dict(r)) for r in rows]

    def get_latest_entities_projection(self, content_snippet_length: Optional[int] = None) -> List[Dict[str, Any]]:
        snippet_length = content_snippet_length or self.entity_content_snippet_length
        entities_with_emb = self._get_entities_with_embeddings()
        version_counts = self.get_entity_version_counts([e.family_id for e, _ in entities_with_emb])
        results = []
        for entity, embedding_array in entities_with_emb:
            results.append({
                "entity": entity,
                "family_id": entity.family_id,
                "name": entity.name,
                "content": entity.content,
                "content_snippet": (entity.content or "")[:snippet_length],
                "version_count": version_counts.get(entity.family_id, 1),
                "embedding_array": embedding_array,
            })
        return results

    def get_isolated_entities(self, limit: int = 100, offset: int = 0) -> List[Entity]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT e.* FROM entity e "
                "WHERE e.family_id IS NOT NULL AND e.graph_id = ? "
                "AND e.uuid NOT IN (SELECT entity1_uuid FROM relates_to WHERE graph_id = ? UNION SELECT entity2_uuid FROM relates_to WHERE graph_id = ?) "
                "ORDER BY e.processed_time DESC LIMIT ? OFFSET ?",
                (self._graph_id, self._graph_id, self._graph_id, limit, offset),
            ).fetchall()
        finally:
            conn.rollback()
        return [_row_to_entity(dict(r)) for r in rows]

    def count_isolated_entities(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT e.family_id) AS cnt FROM entity e "
                "WHERE e.family_id IS NOT NULL AND e.graph_id = ? "
                "AND e.uuid NOT IN (SELECT entity1_uuid FROM relates_to WHERE graph_id = ? UNION SELECT entity2_uuid FROM relates_to WHERE graph_id = ?)",
                (self._graph_id, self._graph_id, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def count_unique_entities(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def cleanup_old_versions(self, before_date: str = None, dry_run: bool = False) -> Dict[str, Any]:
        """Delete old entity/relation versions while keeping the latest per family_id."""
        conn = self._connect()
        try:
            date_filter = f" AND processed_time < '{before_date}'" if before_date else ""
            # Count entity versions that are NOT the latest per family_id
            ent_count = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM entity WHERE graph_id = ?{date_filter} "
                f"AND uuid NOT IN (SELECT e2.uuid FROM entity e2 "
                f"WHERE e2.family_id = entity.family_id AND e2.graph_id = entity.graph_id "
                f"ORDER BY e2.version_seq DESC LIMIT 1)",
                (self._graph_id,),
            ).fetchone()["cnt"]
            rel_count = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM relation WHERE graph_id = ?{date_filter} "
                f"AND uuid NOT IN (SELECT r2.uuid FROM relation r2 "
                f"WHERE r2.family_id = relation.family_id AND r2.graph_id = relation.graph_id "
                f"ORDER BY r2.version_seq DESC LIMIT 1)",
                (self._graph_id,),
            ).fetchone()["cnt"]
            if dry_run:
                return {"dry_run": True, "entities_to_remove": ent_count, "relations_to_remove": rel_count,
                        "message": f"Preview: will delete {ent_count} old entity versions and {rel_count} old relation versions"}
            # Delete old entity versions, keeping latest per family_id
            conn.execute(
                f"DELETE FROM entity WHERE graph_id = ?{date_filter} "
                f"AND uuid NOT IN (SELECT e2.uuid FROM entity e2 "
                f"WHERE e2.family_id = entity.family_id AND e2.graph_id = entity.graph_id "
                f"ORDER BY e2.version_seq DESC LIMIT 1)",
                (self._graph_id,),
            )
            conn.execute(
                f"DELETE FROM relation WHERE graph_id = ?{date_filter} "
                f"AND uuid NOT IN (SELECT r2.uuid FROM relation r2 "
                f"WHERE r2.family_id = relation.family_id AND r2.graph_id = relation.graph_id "
                f"ORDER BY r2.version_seq DESC LIMIT 1)",
                (self._graph_id,),
            )
            conn.commit()
            return {"dry_run": False, "deleted_entity_versions": ent_count, "deleted_relation_versions": rel_count,
                    "message": f"Deleted {ent_count} old entity versions and {rel_count} old relation versions"}
        finally:
            conn.rollback()

    # Keep old name as alias for backward compatibility
    def cleanup_invalidated_versions(self, before_date: str = None, dry_run: bool = False) -> Dict[str, Any]:
        return self.cleanup_old_versions(before_date=before_date, dry_run=dry_run)

    def get_version_diff(self, family_id: str, v1: str, v2: str) -> dict:
        from ...content_schema import parse_markdown_sections, compute_section_diff
        conn = self._connect()
        try:
            rows = conn.execute("SELECT uuid, content FROM entity WHERE uuid IN (?, ?)", (v1, v2)).fetchall()
        finally:
            conn.rollback()
        v1_content = ""
        v2_content = ""
        for r in rows:
            if r["uuid"] == v1:
                v1_content = r["content"] or ""
            else:
                v2_content = r["content"] or ""
        s1 = parse_markdown_sections(v1_content)
        s2 = parse_markdown_sections(v2_content)
        return compute_section_diff(s1, s2)

    # ==================================================================
    # STATISTICS
    # ==================================================================

    def get_stats(self) -> Dict[str, Any]:
        try:
            conn = self._connect()
            try:
                ec = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchone()
                rc = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM relation WHERE graph_id = ?", (self._graph_id,)).fetchone()
            finally:
                conn.rollback()
            return {"entities": ec["cnt"] if ec else 0, "relations": rc["cnt"] if rc else 0}
        except Exception as e:
            logger.warning("get_stats failed: %s", e)
            return {"entities": 0, "relations": 0}

    def get_graph_version(self) -> dict:
        conn = self._connect()
        try:
            ec = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchone()
            rc = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM relation WHERE graph_id = ?", (self._graph_id,)).fetchone()
            et = conn.execute("SELECT MAX(processed_time) AS pt FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchone()
            rt = conn.execute("SELECT MAX(processed_time) AS pt FROM relation WHERE graph_id = ?", (self._graph_id,)).fetchone()
            e_pt = et["pt"] if et and et["pt"] else None
            r_pt = rt["pt"] if rt and rt["pt"] else None
            lm = r_pt if r_pt and (not e_pt or r_pt > e_pt) else e_pt
        finally:
            conn.rollback()
        return {"entity_count": ec["cnt"] if ec else 0, "relation_count": rc["cnt"] if rc else 0, "last_modified": lm}

    def get_graph_statistics(self) -> Dict[str, Any]:
        cached = self._cache.get("graph_stats")
        if cached is not None:
            return cached
        conn = self._connect()
        try:
            total_e = conn.execute("SELECT COUNT(*) AS cnt FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            total_r = conn.execute("SELECT COUNT(*) AS cnt FROM relation WHERE graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            valid_e = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            valid_r = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM relation WHERE graph_id = ?", (self._graph_id,)).fetchone()["cnt"]

            stats = {
                "entity_count": valid_e,
                "relation_count": valid_r,
                "total_entity_versions": total_e,
                "total_relation_versions": total_r,
                "avg_relations_per_entity": 0,
                "max_relations_per_entity": 0,
                "isolated_entities": 0,
                "graph_density": 0.0,
            }

            if valid_e > 0:
                # Degree calculation
                degree_rows = conn.execute(
                    "SELECT e.family_id, COUNT(DISTINCT r.uuid) AS degree "
                    "FROM entity e "
                    "LEFT JOIN relation r ON (r.entity1_absolute_id = e.uuid OR r.entity2_absolute_id = e.uuid) AND r.graph_id = ? "
                    "WHERE e.family_id IS NOT NULL AND e.graph_id = ? "
                    "GROUP BY e.family_id",
                    (self._graph_id, self._graph_id),
                ).fetchall()
                degrees = [r["degree"] for r in degree_rows]
                isolated = sum(1 for d in degrees if d == 0)
                stats["avg_relations_per_entity"] = round(sum(degrees) / len(degrees), 2) if degrees else 0
                stats["max_relations_per_entity"] = max(degrees) if degrees else 0
                stats["isolated_entities"] = isolated
                if valid_e > 1:
                    stats["graph_density"] = round(valid_r / (valid_e * (valid_e - 1) / 2), 4)
                else:
                    stats["graph_density"] = 0.0

            # Time trends
            e_trend = conn.execute(
                "SELECT DATE(event_time) AS d, COUNT(DISTINCT family_id) AS cnt FROM entity WHERE event_time IS NOT NULL AND graph_id = ? GROUP BY d ORDER BY d LIMIT 30",
                (self._graph_id,),
            ).fetchall()
            stats["entity_count_over_time"] = [{"date": r["d"], "count": r["cnt"]} for r in e_trend]
            r_trend = conn.execute(
                "SELECT DATE(event_time) AS d, COUNT(DISTINCT family_id) AS cnt FROM relation WHERE event_time IS NOT NULL AND graph_id = ? GROUP BY d ORDER BY d LIMIT 30",
                (self._graph_id,),
            ).fetchall()
            stats["relation_count_over_time"] = [{"date": r["d"], "count": r["cnt"]} for r in r_trend]
        finally:
            conn.rollback()
        self._cache.set("graph_stats", stats, ttl=60)
        return stats

    def get_data_quality_report(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            valid_families = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM entity WHERE family_id IS NOT NULL AND graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            valid_nodes = conn.execute("SELECT COUNT(*) AS cnt FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            no_fid = conn.execute("SELECT COUNT(*) AS cnt FROM entity WHERE family_id IS NULL AND graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            valid_r_families = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM relation WHERE graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            valid_r_nodes = conn.execute("SELECT COUNT(*) AS cnt FROM relation WHERE graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            # Compute old versions count (all versions minus latest per family)
            inv_e = conn.execute(
                "SELECT COUNT(*) AS cnt FROM entity WHERE graph_id = ? "
                "AND uuid NOT IN (SELECT e2.uuid FROM entity e2 WHERE e2.family_id = entity.family_id AND e2.graph_id = entity.graph_id ORDER BY e2.version_seq DESC LIMIT 1)",
                (self._graph_id,),
            ).fetchone()["cnt"]
            inv_r = conn.execute(
                "SELECT COUNT(*) AS cnt FROM relation WHERE graph_id = ? "
                "AND uuid NOT IN (SELECT r2.uuid FROM relation r2 WHERE r2.family_id = relation.family_id AND r2.graph_id = relation.graph_id ORDER BY r2.version_seq DESC LIMIT 1)",
                (self._graph_id,),
            ).fetchone()["cnt"]
        finally:
            conn.rollback()
        isolated = self.count_isolated_entities()
        return {
            "entities": {"valid_unique": valid_families, "valid_versions": valid_nodes, "old_versions": inv_e, "no_family_id": no_fid, "isolated": isolated},
            "relations": {"valid_unique": valid_r_families, "valid_versions": valid_r_nodes, "old_versions": inv_r},
            "total_nodes": valid_nodes + inv_e + valid_r_nodes + inv_r + no_fid,
        }

    # ==================================================================
    # RELATION STORE
    # ==================================================================

    def _invalidate_relation_cache(self, family_id: str = None):
        keys = ["graph_stats"]
        if family_id:
            keys.append(f"relation:by_fid:{family_id}")
        self._cache.invalidate_keys(keys)

    def _invalidate_relation_cache_bulk(self):
        self._cache.invalidate_keys(["graph_stats"])

    def _build_relation_embedding_text(self, relation: Relation, entity1_name: str = "", entity2_name: str = "") -> str:
        content = relation.content or ""
        if entity1_name and entity2_name:
            return f"# {entity1_name} → {entity2_name}\n{content}"
        elif entity1_name or entity2_name:
            return f"# {entity1_name or entity2_name}\n{content}"
        return content

    def _resolve_entity_names_for_embedding(self, relation: Relation, names: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
        aid1, aid2 = relation.entity1_absolute_id, relation.entity2_absolute_id
        _enc = self._entity_name_cache
        if names:
            for k, v in names.items():
                if k not in _enc:
                    self._cache_entity_name(k, v)
        name1 = _enc.get(aid1, ...)
        name2 = _enc.get(aid2, ...)
        if name1 is not ... and name2 is not ...:
            return name1, name2
        try:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT uuid, name FROM entity WHERE uuid IN (?, ?)", (aid1, aid2)).fetchall()
            finally:
                conn.rollback()
            for r in rows:
                self._cache_entity_name(r["uuid"], r["name"] or "")
                if r["uuid"] == aid1:
                    name1 = r["name"] or ""
                else:
                    name2 = r["name"] or ""
        except Exception:
            if name1 is ...:
                name1 = ""
            if name2 is ...:
                name2 = ""
        return name1, name2

    def _compute_relation_embedding(self, relation: Relation, names: Optional[Dict[str, str]] = None) -> Optional[bytes]:
        name1, name2 = self._resolve_entity_names_for_embedding(relation, names=names)
        text = self._build_relation_embedding_text(relation, name1, name2)
        result = _encode_and_normalize(self.embedding_client, text)
        return result[0] if result else None

    def _bulk_compute_relation_embeddings(self, relations: List[Relation]) -> Dict[str, bytes]:
        if not self.embedding_client or not self.embedding_client.is_available():
            return {}
        # Batch resolve entity names
        all_abs_ids = set()
        for r in relations:
            if r.entity1_absolute_id:
                all_abs_ids.add(r.entity1_absolute_id)
            if r.entity2_absolute_id:
                all_abs_ids.add(r.entity2_absolute_id)
        entity_names = {}
        if all_abs_ids:
            try:
                conn = self._connect()
                try:
                    ph = ",".join("?" * len(all_abs_ids))
                    rows = conn.execute(f"SELECT uuid, name FROM entity WHERE uuid IN ({ph})", list(all_abs_ids)).fetchall()
                finally:
                    conn.rollback()
                for r in rows:
                    entity_names[r["uuid"]] = r["name"] or ""
            except Exception:
                pass
        # Build texts
        texts = []
        uuids = []
        for r in relations:
            name1 = entity_names.get(r.entity1_absolute_id, "")
            name2 = entity_names.get(r.entity2_absolute_id, "")
            texts.append(self._build_relation_embedding_text(r, name1, name2))
            uuids.append(r.absolute_id)
        try:
            embeddings = self.embedding_client.encode(texts)
        except Exception:
            return {}
        if embeddings is None:
            return {}
        result = {}
        for idx, uuid in enumerate(uuids):
            try:
                emb = np.array(embeddings[idx], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                result[uuid] = emb.tobytes()
            except Exception:
                pass
        return result

    def _update_relation_emb_cache(self, relation: Relation, emb_array: Optional[np.ndarray]):
        if self._relation_emb_cache is None:
            return
        if not hasattr(self, '_relation_emb_fid_idx') or self._relation_emb_fid_idx is None:
            self._relation_emb_fid_idx = {r.family_id: i for i, (r, _) in enumerate(self._relation_emb_cache)}
        idx = self._relation_emb_fid_idx.get(relation.family_id)
        if idx is not None:
            self._relation_emb_cache[idx] = (relation, emb_array)
        else:
            self._relation_emb_cache.append((relation, emb_array))
            self._relation_emb_fid_idx[relation.family_id] = len(self._relation_emb_cache) - 1

    def _update_relation_emb_cache_batch(self, items: List[tuple]):
        if self._relation_emb_cache is None or not items:
            return
        if not hasattr(self, '_relation_emb_fid_idx') or self._relation_emb_fid_idx is None:
            self._relation_emb_fid_idx = {r.family_id: i for i, (r, _) in enumerate(self._relation_emb_cache)}
        for relation, emb_array in items:
            idx = self._relation_emb_fid_idx.get(relation.family_id)
            if idx is not None:
                self._relation_emb_cache[idx] = (relation, emb_array)
            else:
                self._relation_emb_cache.append((relation, emb_array))
                self._relation_emb_fid_idx[relation.family_id] = len(self._relation_emb_cache) - 1

    def _get_relations_with_embeddings(self) -> List[tuple]:
        now = time.time()
        if self._relation_emb_cache is not None and (now - self._relation_emb_cache_ts) < self._emb_cache_ttl:
            return self._relation_emb_cache
        with _perf_timer("_get_relations_with_embeddings"):
            result = self._get_relations_with_embeddings_impl()
        self._relation_emb_cache = result
        self._relation_emb_fid_idx = None
        self._relation_emb_cache_ts = time.time()
        # Build HNSW index alongside cache
        if result and self._vector_dim > 0:
            self._relation_hnsw, self._relation_hnsw_items = self._build_hnsw(result, self._vector_dim)
        else:
            self._relation_hnsw = None
            self._relation_hnsw_items = None
        return result

    def _get_relations_with_embeddings_impl(self) -> List[tuple]:
        conn = self._connect()
        try:
            limit = getattr(self, '_emb_cache_max_size', 10000)
            rows = conn.execute(
                f"SELECT r.* FROM relation r "
                f"INNER JOIN ("
                f"  SELECT family_id, MAX(version_seq) AS max_vs FROM relation "
                f"  WHERE graph_id = ? GROUP BY family_id"
                f") latest ON r.family_id = latest.family_id AND r.version_seq = latest.max_vs "
                f"WHERE r.graph_id = ? ORDER BY r.processed_time DESC LIMIT ?",
                (self._graph_id, self._graph_id, limit),
            ).fetchall()
        finally:
            conn.rollback()
        relations = []
        for row in rows:
            relation = _row_to_relation(dict(row))
            emb_array = np.frombuffer(relation.embedding, dtype=np.float32) if relation.embedding else None
            relations.append((relation, emb_array))
        return relations

    def save_relation(self, relation: Relation):
        with _perf_timer("save_relation"):
            emb_array = self._save_relation_impl(relation)
            if emb_array is not None:
                self._update_relation_emb_cache(relation, emb_array)

    def _save_relation_impl(self, relation: Relation, names: Optional[Dict[str, str]] = None):
        valid_at = _fmt_dt(relation.valid_at or relation.event_time)
        embedding_blob = self._compute_relation_embedding(relation, names=names)
        if embedding_blob is not None:
            relation.embedding = embedding_blob
        with self._relation_write_lock:
            conn = self._connect()
            try:
                # Auto-increment version_seq
                max_vs_row = conn.execute(
                    "SELECT MAX(version_seq) AS max_vs FROM relation WHERE family_id = ? AND graph_id = ?",
                    (relation.family_id, self._graph_id),
                ).fetchone()
                version_seq = (max_vs_row["max_vs"] or 0) + 1
                relation.version_seq = version_seq
                attrs = json.dumps(relation.attributes, ensure_ascii=False) if isinstance(relation.attributes, (dict, list)) else relation.attributes
                prov = json.dumps(relation.provenance, ensure_ascii=False) if isinstance(relation.provenance, (dict, list)) else relation.provenance
                conn.execute(
                    f"INSERT OR REPLACE INTO relation ({', '.join(RELATION_COLUMNS)}) "
                    f"VALUES ({', '.join('?' * len(RELATION_COLUMNS))})",
                    (
                        relation.absolute_id, relation.family_id, self._graph_id,
                        relation.entity1_absolute_id, relation.entity2_absolute_id,
                        None, None,  # entity1_family_id, entity2_family_id filled below
                        relation.content, relation.summary, attrs,
                        relation.confidence, prov,
                        getattr(relation, "content_format", "plain"),
                        version_seq, valid_at,
                        _fmt_dt(relation.event_time), _fmt_dt(relation.processed_time),
                        relation.episode_id, relation.source_document,
                        relation.embedding,
                    ),
                )
                # Look up entity family_ids
                e1_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (relation.entity1_absolute_id, self._graph_id)).fetchone()
                e2_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (relation.entity2_absolute_id, self._graph_id)).fetchone()
                e1_fid = e1_row["family_id"] if e1_row else None
                e2_fid = e2_row["family_id"] if e2_row else None
                conn.execute(
                    "UPDATE relation SET entity1_family_id = ?, entity2_family_id = ? WHERE uuid = ? AND graph_id = ?",
                    (e1_fid, e2_fid, relation.absolute_id, self._graph_id),
                )
                # Update RELATES_TO edges
                if e1_fid and e2_fid:
                    # Get latest entities for each family
                    e1_latest = conn.execute(
                        "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                        (e1_fid, self._graph_id),
                    ).fetchone()
                    e2_latest = conn.execute(
                        "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                        (e2_fid, self._graph_id),
                    ).fetchone()
                    if e1_latest and e2_latest:
                        conn.execute(
                            "INSERT OR REPLACE INTO relates_to (entity1_uuid, entity2_uuid, relation_uuid, fact, graph_id) VALUES (?, ?, ?, ?, ?)",
                            (e1_latest["uuid"], e2_latest["uuid"], relation.absolute_id, relation.content, self._graph_id),
                        )
                conn.commit()
            finally:
                conn.rollback()
        self._invalidate_relation_cache_bulk()
        emb_array = None
        if embedding_blob:
            emb_array = np.frombuffer(embedding_blob, dtype=np.float32)
        return emb_array

    def bulk_save_relations(self, relations: List[Relation]):
        if not relations:
            return
        # Compute embeddings synchronously before writing
        emb_map = self._bulk_compute_relation_embeddings(relations)
        rows = []
        cache_items = []
        for relation in relations:
            _attrs = getattr(relation, 'attributes', None)
            _prov = getattr(relation, 'provenance', None)
            emb_blob = emb_map.get(relation.absolute_id)
            relation.embedding = emb_blob
            rows.append((
                relation.absolute_id, relation.family_id, self._graph_id,
                relation.entity1_absolute_id, relation.entity2_absolute_id,
                None, None,
                relation.content, getattr(relation, 'summary', None),
                json.dumps(_attrs, ensure_ascii=False) if isinstance(_attrs, (dict, list)) else _attrs,
                getattr(relation, 'confidence', None),
                json.dumps(_prov, ensure_ascii=False) if isinstance(_prov, (dict, list)) else _prov,
                getattr(relation, 'content_format', None),
                _fmt_dt(relation.valid_at or relation.event_time) if (relation.valid_at or relation.event_time) else None,
                None,  # placeholder - will be replaced with version_seq below
                _fmt_dt(relation.event_time), _fmt_dt(relation.processed_time),
                relation.episode_id, relation.source_document,
                emb_blob,
            ))
            if emb_blob is not None:
                cache_items.append((relation, np.frombuffer(emb_blob, dtype=np.float32)))
            else:
                cache_items.append((relation, None))
        with self._relation_write_lock:
            conn = self._connect()
            try:
                # Auto-increment version_seq per family_id
                fid_to_max_vs = {}
                for relation in relations:
                    fid = relation.family_id
                    if fid not in fid_to_max_vs:
                        row = conn.execute(
                            "SELECT MAX(version_seq) AS max_vs FROM relation WHERE family_id = ? AND graph_id = ?",
                            (fid, self._graph_id),
                        ).fetchone()
                        fid_to_max_vs[fid] = row["max_vs"] or 0

                # Replace the version_seq placeholder (index 13) in each row
                final_rows = []
                for i, relation in enumerate(relations):
                    row = list(rows[i])
                    fid = relation.family_id
                    fid_to_max_vs[fid] += 1
                    row[13] = fid_to_max_vs[fid]
                    relation.version_seq = fid_to_max_vs[fid]
                    final_rows.append(tuple(row))

                conn.executemany(
                    f"INSERT OR REPLACE INTO relation ({', '.join(RELATION_COLUMNS)}) VALUES ({', '.join('?' * len(RELATION_COLUMNS))})",
                    final_rows,
                )
                # Batch-lookup entity family_ids: collect all unique abs_ids
                all_abs_ids = set()
                for r in relations:
                    all_abs_ids.add(r.entity1_absolute_id)
                    all_abs_ids.add(r.entity2_absolute_id)
                uuid_to_fid = {}
                if all_abs_ids:
                    id_list = list(all_abs_ids)
                    ph = ",".join("?" * len(id_list))
                    for row in conn.execute(
                        f"SELECT uuid, family_id FROM entity WHERE uuid IN ({ph}) AND graph_id = ?",
                        id_list + [self._graph_id],
                    ).fetchall():
                        uuid_to_fid[row["uuid"]] = row["family_id"]

                # Batch-lookup latest entity UUIDs per family_id
                all_fids = set(fid for fid in uuid_to_fid.values() if fid)
                fid_to_latest = {}
                if all_fids:
                    fid_list = list(all_fids)
                    ph = ",".join("?" * len(fid_list))
                    for row in conn.execute(
                        f"SELECT family_id, uuid FROM entity WHERE family_id IN ({ph}) AND graph_id = ? ORDER BY version_seq DESC",
                        fid_list + [self._graph_id],
                    ).fetchall():
                        fid_to_latest.setdefault(row["family_id"], row["uuid"])

                # Apply family_ids and RELATES_TO edges
                fid_update_rows = []
                relates_to_rows = []
                for relation in relations:
                    e1_fid = uuid_to_fid.get(relation.entity1_absolute_id)
                    e2_fid = uuid_to_fid.get(relation.entity2_absolute_id)
                    fid_update_rows.append((e1_fid, e2_fid, relation.absolute_id, self._graph_id))
                    if e1_fid and e2_fid:
                        e1_latest = fid_to_latest.get(e1_fid)
                        e2_latest = fid_to_latest.get(e2_fid)
                        if e1_latest and e2_latest:
                            relates_to_rows.append((e1_latest, e2_latest, relation.absolute_id, relation.content, self._graph_id))
                conn.executemany(
                    "UPDATE relation SET entity1_family_id = ?, entity2_family_id = ? WHERE uuid = ? AND graph_id = ?",
                    fid_update_rows,
                )
                if relates_to_rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO relates_to (entity1_uuid, entity2_uuid, relation_uuid, fact, graph_id) VALUES (?, ?, ?, ?, ?)",
                        relates_to_rows,
                    )
                conn.commit()
            finally:
                conn.rollback()
        self._update_relation_emb_cache_batch(cache_items)
        self._invalidate_relation_cache_bulk()

    def bulk_save_relations_with_embedding(self, relations: List[Relation]):
        if not relations:
            return
        rows = []
        cache_items = []
        for relation in relations:
            emb_blob = getattr(relation, 'embedding', None)
            emb_array = None
            if emb_blob is not None:
                if isinstance(emb_blob, np.ndarray):
                    emb_array = emb_blob
                else:
                    emb_array = np.frombuffer(emb_blob, dtype=np.float32)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_array = emb_array / norm
                relation.embedding = emb_array.tobytes()
                cache_items.append((relation, emb_array))
            else:
                cache_items.append((relation, None))
            _attrs = getattr(relation, 'attributes', None)
            _prov = getattr(relation, 'provenance', None)
            rows.append((
                relation.absolute_id, relation.family_id, self._graph_id,
                relation.entity1_absolute_id, relation.entity2_absolute_id,
                None, None,
                relation.content, getattr(relation, 'summary', None),
                json.dumps(_attrs, ensure_ascii=False) if isinstance(_attrs, (dict, list)) else _attrs,
                getattr(relation, 'confidence', None),
                json.dumps(_prov, ensure_ascii=False) if isinstance(_prov, (dict, list)) else _prov,
                getattr(relation, 'content_format', None),
                _fmt_dt(relation.valid_at or relation.event_time) if (relation.valid_at or relation.event_time) else None,
                None,  # placeholder - will be replaced with version_seq below
                _fmt_dt(relation.event_time), _fmt_dt(relation.processed_time),
                relation.episode_id, relation.source_document,
                relation.embedding,
            ))
        with self._relation_write_lock:
            conn = self._connect()
            try:
                # Auto-increment version_seq per family_id
                fid_to_max_vs = {}
                for relation in relations:
                    fid = relation.family_id
                    if fid not in fid_to_max_vs:
                        row = conn.execute(
                            "SELECT MAX(version_seq) AS max_vs FROM relation WHERE family_id = ? AND graph_id = ?",
                            (fid, self._graph_id),
                        ).fetchone()
                        fid_to_max_vs[fid] = row["max_vs"] or 0

                # Replace the version_seq placeholder (index 13) in each row
                final_rows = []
                for i, relation in enumerate(relations):
                    row = list(rows[i])
                    fid = relation.family_id
                    fid_to_max_vs[fid] += 1
                    row[13] = fid_to_max_vs[fid]
                    relation.version_seq = fid_to_max_vs[fid]
                    final_rows.append(tuple(row))

                conn.executemany(
                    f"INSERT OR REPLACE INTO relation ({', '.join(RELATION_COLUMNS)}) VALUES ({', '.join('?' * len(RELATION_COLUMNS))})",
                    final_rows,
                )
                for relation in relations:
                    e1_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (relation.entity1_absolute_id, self._graph_id)).fetchone()
                    e2_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (relation.entity2_absolute_id, self._graph_id)).fetchone()
                    e1_fid = e1_row["family_id"] if e1_row else None
                    e2_fid = e2_row["family_id"] if e2_row else None
                    conn.execute(
                        "UPDATE relation SET entity1_family_id = ?, entity2_family_id = ? WHERE uuid = ? AND graph_id = ?",
                        (e1_fid, e2_fid, relation.absolute_id, self._graph_id),
                    )
                    if e1_fid and e2_fid:
                        e1_latest = conn.execute(
                            "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                            (e1_fid, self._graph_id),
                        ).fetchone()
                        e2_latest = conn.execute(
                            "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                            (e2_fid, self._graph_id),
                        ).fetchone()
                        if e1_latest and e2_latest:
                            conn.execute(
                                "INSERT OR REPLACE INTO relates_to (entity1_uuid, entity2_uuid, relation_uuid, fact, graph_id) VALUES (?, ?, ?, ?, ?)",
                                (e1_latest["uuid"], e2_latest["uuid"], relation.absolute_id, relation.content, self._graph_id),
                            )
                conn.commit()
            finally:
                conn.rollback()
        if self._relation_emb_cache is not None and cache_items:
            if self._relation_emb_fid_idx is not None:
                fid_to_idx = self._relation_emb_fid_idx
            else:
                fid_to_idx = {r.family_id: i for i, (r, _) in enumerate(self._relation_emb_cache)} if self._relation_emb_cache else {}
                self._relation_emb_fid_idx = fid_to_idx
            for relation, emb_array in cache_items:
                idx = fid_to_idx.get(relation.family_id)
                if idx is not None:
                    self._relation_emb_cache[idx] = (relation, emb_array)
                else:
                    self._relation_emb_cache.append((relation, emb_array))
                    fid_to_idx[relation.family_id] = len(self._relation_emb_cache) - 1
        self._invalidate_relation_cache_bulk()

    def get_relation_by_absolute_id(self, relation_absolute_id: str) -> Optional[Relation]:
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE uuid = ? AND graph_id = ?",
                (relation_absolute_id, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        return _row_to_relation(dict(row)) if row else None

    def get_relation_by_family_id(self, family_id: str) -> Optional[Relation]:
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                (family_id, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        return _row_to_relation(dict(row)) if row else None

    def get_relation_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT embedding FROM relation WHERE uuid = ? AND graph_id = ?", (absolute_id, self._graph_id)).fetchone()
        finally:
            conn.rollback()
        if row and row["embedding"]:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            return emb[:num_values].tolist()
        return None

    def get_relation_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
        if not family_ids:
            return {}
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = list({r for r in resolved_map.values() if r})
        if not canonical_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(canonical_ids))
            rows = conn.execute(
                f"SELECT family_id, COUNT(*) AS cnt FROM relation WHERE family_id IN ({placeholders}) AND graph_id = ? GROUP BY family_id",
                canonical_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        return {r["family_id"]: r["cnt"] for r in rows}

    def get_relation_versions(self, family_id: str) -> List[Relation]:
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE family_id = ? AND graph_id = ? ORDER BY processed_time ASC",
                (family_id, self._graph_id),
            ).fetchall()
        finally:
            conn.rollback()
        return [_row_to_relation(dict(r)) for r in rows]

    def get_relation_versions_batch(self, family_ids: List[str]) -> Dict[str, List[Relation]]:
        if not family_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(family_ids))
            rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE family_id IN ({placeholders}) AND graph_id = ? ORDER BY processed_time ASC",
                family_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        versions_map: Dict[str, List[Relation]] = {fid: [] for fid in family_ids}
        for r in rows:
            rel = _row_to_relation(dict(r))
            if rel.family_id in versions_map:
                versions_map[rel.family_id].append(rel)
        return versions_map

    def get_relations_by_absolute_ids(self, absolute_ids: List[str], valid_only: bool = False) -> List[Relation]:
        if not absolute_ids:
            return []
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(absolute_ids))
            extra = " AND version_seq = (SELECT MAX(r2.version_seq) FROM relation r2 WHERE r2.family_id = relation.family_id AND r2.graph_id = relation.graph_id)" if valid_only else ""
            rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE uuid IN ({placeholders}){extra}",
                absolute_ids,
            ).fetchall()
        finally:
            conn.rollback()
        return [_row_to_relation(dict(r)) for r in rows]

    def get_relations_by_entities(self, from_family_id: str, to_family_id: str, include_candidates: bool = False) -> List[Relation]:
        with _perf_timer("get_relations_by_entities"):
            result = self._get_relations_by_entities_impl(from_family_id, to_family_id)
            return self._filter_dream_candidates(result, include_candidates)

    def _get_relations_by_entities_impl(self, from_family_id: str, to_family_id: str) -> List[Relation]:
        from_family_id = self.resolve_family_id(from_family_id)
        to_family_id = self.resolve_family_id(to_family_id)
        if not from_family_id or not to_family_id:
            return []
        conn = self._connect()
        try:
            from_ids = [r["uuid"] for r in conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?", (from_family_id, self._graph_id),
            ).fetchall()]
            to_ids = [r["uuid"] for r in conn.execute(
                "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?", (to_family_id, self._graph_id),
            ).fetchall()]
            if not from_ids or not to_ids:
                return []
            all_ids = from_ids + to_ids
            placeholders = ",".join("?" * len(all_ids))
            from_ph = ",".join("?" * len(from_ids))
            to_ph = ",".join("?" * len(to_ids))
            rows = conn.execute(
                f"SELECT r.* FROM relation r "
                f"INNER JOIN ("
                f"  SELECT family_id, MAX(version_seq) AS max_vs FROM relation "
                f"  WHERE graph_id = ? GROUP BY family_id"
                f") latest ON r.family_id = latest.family_id AND r.version_seq = latest.max_vs "
                f"WHERE r.graph_id = ? "
                f"AND ((r.entity1_absolute_id IN ({from_ph}) AND r.entity2_absolute_id IN ({to_ph})) "
                f"  OR (r.entity1_absolute_id IN ({to_ph}) AND r.entity2_absolute_id IN ({from_ph}))) "
                f"ORDER BY r.processed_time DESC",
                [self._graph_id, self._graph_id] + from_ids + to_ids + to_ids + from_ids,
            ).fetchall()
        finally:
            conn.rollback()
        return [_row_to_relation(dict(r)) for r in rows]

    def get_relations_by_entity_absolute_ids(self, entity_absolute_ids: List[str], limit: Optional[int] = None,
                                              include_candidates: bool = False) -> List[Relation]:
        if not entity_absolute_ids:
            return []
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(entity_absolute_ids))
            query = (
                f"SELECT r.* FROM relation r "
                f"INNER JOIN ("
                f"  SELECT family_id, MAX(version_seq) AS max_vs FROM relation "
                f"  WHERE graph_id = ? GROUP BY family_id"
                f") latest ON r.family_id = latest.family_id AND r.version_seq = latest.max_vs "
                f"WHERE r.graph_id = ? "
                f"AND (r.entity1_absolute_id IN ({placeholders}) OR r.entity2_absolute_id IN ({placeholders})) "
                f"ORDER BY r.processed_time DESC"
            )
            params = [self._graph_id, self._graph_id] + entity_absolute_ids + entity_absolute_ids
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        relations = [_row_to_relation(dict(r)) for r in rows]
        return self._filter_dream_candidates(relations, include_candidates)

    def get_relations_by_entity_pairs(self, entity_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], List[Relation]]:
        if not entity_pairs:
            return {}
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)}, entity1_family_id, entity2_family_id "
                f"FROM relation WHERE graph_id = ?",
                (self._graph_id,),
            ).fetchall()
        finally:
            conn.rollback()
        _rel_by_pair: Dict[Tuple[str, str], List[Relation]] = defaultdict(list)
        seen_rel_fids: set = set()
        for r in rows:
            rel = _row_to_relation(dict(r))
            if rel.family_id in seen_rel_fids:
                continue
            seen_rel_fids.add(rel.family_id)
            f1 = r["entity1_family_id"]
            f2 = r["entity2_family_id"]
            if f1 and f2:
                pk = (f1, f2) if f1 <= f2 else (f2, f1)
                _rel_by_pair[pk].append(rel)
        results: Dict[Tuple[str, str], List[Relation]] = {}
        for e1, e2 in entity_pairs:
            pk = (e1, e2) if e1 <= e2 else (e2, e1)
            if pk not in results:
                results[pk] = _rel_by_pair.get(pk, [])
        return results

    def get_relation_embeddings(self, family_ids: List[str]) -> Dict[str, Any]:
        if not family_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(family_ids))
            rows = conn.execute(
                f"SELECT family_id, embedding FROM relation WHERE family_id IN ({placeholders}) AND embedding IS NOT NULL AND graph_id = ?",
                family_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        result = {}
        for r in rows:
            emb = r["embedding"]
            if emb:
                result[r["family_id"]] = np.frombuffer(emb, dtype=np.float32).copy()
        return result

    def get_relations_by_family_ids(self, family_ids: List[str], limit: int = 100, time_point: Optional[str] = None) -> List[Relation]:
        if not family_ids:
            return []
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(family_ids))
            tp_filter = ""
            tp_params = []
            if time_point:
                tp_filter = " AND (r.valid_at IS NULL OR r.valid_at <= ?)"
                tp_params.append(time_point)
            query = (
                f"SELECT DISTINCT r.* FROM entity e "
                f"INNER JOIN relation r ON (r.entity1_absolute_id = e.uuid OR r.entity2_absolute_id = e.uuid) "
                f"WHERE e.family_id IN ({placeholders}) AND e.graph_id = ? "
                f"AND r.graph_id = ?{tp_filter} "
                f"LIMIT ?"
            )
            params = family_ids + [self._graph_id, self._graph_id] + tp_params + [limit]
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [_row_to_relation(dict(r)) for r in rows]

    def get_relations_referencing_absolute_id(self, absolute_id: str) -> List[Relation]:
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation "
                f"WHERE (entity1_absolute_id = ? OR entity2_absolute_id = ?) AND graph_id = ?",
                (absolute_id, absolute_id, self._graph_id),
            ).fetchall()
        finally:
            conn.rollback()
        rels = [_row_to_relation(dict(r)) for r in rows]
        remap = self._build_entity_abs_id_remap()
        if remap:
            for rel in rels:
                rel.entity1_absolute_id = remap.get(rel.entity1_absolute_id, rel.entity1_absolute_id)
                rel.entity2_absolute_id = remap.get(rel.entity2_absolute_id, rel.entity2_absolute_id)
        return rels

    def get_all_relations(self, limit: Optional[int] = None, offset: Optional[int] = None,
                           exclude_embedding: bool = False, include_candidates: bool = False) -> List[Relation]:
        cols = [c for c in RELATION_COLUMNS if not (exclude_embedding and c == "embedding")]
        conn = self._connect()
        try:
            query = (
                f"SELECT r.* FROM relation r "
                f"INNER JOIN ("
                f"  SELECT family_id, MAX(version_seq) AS max_vs FROM relation "
                f"  WHERE graph_id = ? GROUP BY family_id"
                f") latest ON r.family_id = latest.family_id AND r.version_seq = latest.max_vs "
                f"WHERE r.graph_id = ? ORDER BY r.processed_time DESC"
            )
            params: list = [self._graph_id, self._graph_id]
            if offset:
                query += f" OFFSET {int(offset)}"
            if limit:
                query += f" LIMIT {int(limit)}"
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        relations = [_row_to_relation(dict(r)) for r in rows]
        return self._filter_dream_candidates(relations, include_candidates)

    def stream_all_relations(self, exclude_embedding: bool = True, include_candidates: bool = False, since: Optional[str] = None):
        relations = self.get_all_relations(exclude_embedding=exclude_embedding)
        remap = self._build_entity_abs_id_remap()
        for rel in relations:
            if since:
                pt = _fmt_dt(rel.processed_time)
                if pt and pt <= since:
                    continue
            if remap:
                rel.entity1_absolute_id = remap.get(rel.entity1_absolute_id, rel.entity1_absolute_id)
                rel.entity2_absolute_id = remap.get(rel.entity2_absolute_id, rel.entity2_absolute_id)
            if not self._is_dream_candidate(rel) or include_candidates:
                yield rel

    def count_relations_since(self, since: str) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT family_id) AS cnt FROM relation "
                "WHERE processed_time > ? AND graph_id = ?",
                (since, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def count_unique_relations(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT family_id) AS cnt FROM relation WHERE graph_id = ?",
                (self._graph_id,),
            ).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def get_old_relation_versions(self, limit: int = 100) -> List[Relation]:
        """Get relation versions that are not the latest per family_id."""
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE graph_id = ? "
                f"AND uuid NOT IN (SELECT r2.uuid FROM relation r2 "
                f"WHERE r2.family_id = relation.family_id AND r2.graph_id = relation.graph_id "
                f"ORDER BY r2.version_seq DESC LIMIT 1) "
                f"ORDER BY processed_time DESC LIMIT ?",
                (self._graph_id, limit),
            ).fetchall()
        finally:
            conn.rollback()
        return [_row_to_relation(dict(r)) for r in rows]

    def delete_relation_all_versions(self, family_id: str) -> int:
        return self.delete_relation_by_id(family_id)

    def delete_relation_by_absolute_id(self, absolute_id: str) -> bool:
        with self._relation_write_lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM relates_to WHERE relation_uuid = ? AND graph_id = ?", (absolute_id, self._graph_id))
                cursor = conn.execute("DELETE FROM relation WHERE uuid = ? AND graph_id = ?", (absolute_id, self._graph_id))
                conn.commit()
                deleted = cursor.rowcount > 0
            finally:
                conn.rollback()
        self._invalidate_relation_cache_bulk()
        return deleted

    def delete_relation_by_id(self, family_id: str) -> int:
        with self._relation_write_lock:
            conn = self._connect()
            try:
                abs_ids = [r["uuid"] for r in conn.execute(
                    "SELECT uuid FROM relation WHERE family_id = ? AND graph_id = ?", (family_id, self._graph_id),
                ).fetchall()]
                count = 0
                if abs_ids:
                    placeholders = ",".join("?" * len(abs_ids))
                    conn.execute(f"DELETE FROM relates_to WHERE relation_uuid IN ({placeholders}) AND graph_id = ?", abs_ids + [self._graph_id])
                    cursor = conn.execute("DELETE FROM relation WHERE family_id = ? AND graph_id = ?", (family_id, self._graph_id))
                    count = cursor.rowcount
                conn.commit()
            finally:
                conn.rollback()
        self._invalidate_relation_cache_bulk()
        return count

    def batch_delete_relation_versions_by_absolute_ids(self, absolute_ids: List[str]) -> int:
        if not absolute_ids:
            return 0
        with self._relation_write_lock:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(absolute_ids))
                conn.execute(f"DELETE FROM relates_to WHERE relation_uuid IN ({placeholders}) AND graph_id = ?", absolute_ids + [self._graph_id])
                cursor = conn.execute(f"DELETE FROM relation WHERE uuid IN ({placeholders}) AND graph_id = ?", absolute_ids + [self._graph_id])
                conn.commit()
                deleted = cursor.rowcount
            finally:
                conn.rollback()
        self._invalidate_relation_cache_bulk()
        return deleted

    def batch_delete_relations(self, family_ids: List[str]) -> int:
        if not family_ids:
            return 0
        with self._relation_write_lock:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(family_ids))
                all_uuids = [r["uuid"] for r in conn.execute(
                    f"SELECT uuid FROM relation WHERE family_id IN ({placeholders}) AND graph_id = ?",
                    family_ids + [self._graph_id],
                ).fetchall()]
                count = 0
                if all_uuids:
                    uuid_ph = ",".join("?" * len(all_uuids))
                    conn.execute(f"DELETE FROM relates_to WHERE relation_uuid IN ({uuid_ph}) AND graph_id = ?", all_uuids + [self._graph_id])
                cursor = conn.execute(
                    f"DELETE FROM relation WHERE family_id IN ({placeholders}) AND graph_id = ?",
                    family_ids + [self._graph_id],
                )
                count = cursor.rowcount
                conn.commit()
            finally:
                conn.rollback()
        self._invalidate_relation_cache_bulk()
        return count

    def batch_get_relations_referencing_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, List[Relation]]:
        if not absolute_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(absolute_ids))
            rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation "
                f"WHERE (entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})) AND graph_id = ?",
                absolute_ids + absolute_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        result_map: Dict[str, List[Relation]] = {aid: [] for aid in absolute_ids}
        for r in rows:
            rel = _row_to_relation(dict(r))
            if rel.entity1_absolute_id in result_map:
                result_map[rel.entity1_absolute_id].append(rel)
            if rel.entity2_absolute_id in result_map:
                result_map[rel.entity2_absolute_id].append(rel)
        return result_map

    def invalidate_relation(self, family_id: str, reason: str = "") -> int:
        """Delete all but the latest version of a relation."""
        conn = self._connect()
        try:
            # Delete old versions, keep latest per version_seq
            cursor = conn.execute(
                "DELETE FROM relation WHERE family_id = ? AND graph_id = ? "
                "AND uuid NOT IN (SELECT r2.uuid FROM relation r2 WHERE r2.family_id = ? AND r2.graph_id = ? ORDER BY r2.version_seq DESC LIMIT 1)",
                (family_id, self._graph_id, family_id, self._graph_id),
            )
            conn.commit()
        finally:
            conn.rollback()
        self._invalidate_relation_cache_bulk()
        return cursor.rowcount

    def redirect_relation(self, family_id: str, side: str, new_family_id: str) -> int:
        if side not in ("entity1", "entity2"):
            raise ValueError(f"side must be 'entity1' or 'entity2', got '{side}'")
        side_field = f"{side}_absolute_id"
        with self._relation_write_lock:
            conn = self._connect()
            try:
                target_row = conn.execute(
                    "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY processed_time DESC LIMIT 1",
                    (new_family_id, self._graph_id),
                ).fetchone()
                if not target_row:
                    return 0
                new_abs_id = target_row["uuid"]
                cursor = conn.execute(
                    f"UPDATE relation SET {side_field} = ? WHERE family_id = ? AND graph_id = ?",
                    (new_abs_id, family_id, self._graph_id),
                )
                conn.commit()
                count = cursor.rowcount
            finally:
                conn.rollback()
        self._invalidate_relation_cache_bulk()
        return count

    def update_relation_by_absolute_id(self, absolute_id: str, **fields) -> Optional[Relation]:
        valid_keys = {"content", "summary", "attributes", "confidence"}
        filtered = {k: v for k, v in fields.items() if k in valid_keys and v is not None}
        if not filtered:
            return None
        needs_emb_update = "content" in filtered
        _precomputed_emb = None
        if needs_emb_update and self.embedding_client and self.embedding_client.is_available():
            current = self.get_relation_by_absolute_id(absolute_id)
            if current:
                merged = Relation(
                    name="", content=filtered.get("content", current.content),
                    entity1_absolute_id=current.entity1_absolute_id,
                    entity2_absolute_id=current.entity2_absolute_id,
                )
                _precomputed_emb = self._compute_relation_embedding(merged)
        with self._relation_write_lock:
            conn = self._connect()
            try:
                set_parts = [f"{k} = ?" for k in filtered]
                params = list(filtered.values())
                if _precomputed_emb is not None:
                    set_parts.append("embedding = ?")
                    params.append(_precomputed_emb)
                params.extend([absolute_id, self._graph_id])
                cursor = conn.execute(
                    f"UPDATE relation SET {', '.join(set_parts)} WHERE uuid = ? AND graph_id = ?",
                    params,
                )
                conn.commit()
                if cursor.rowcount == 0:
                    return None
                row = conn.execute(
                    f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE uuid = ? AND graph_id = ?",
                    (absolute_id, self._graph_id),
                ).fetchone()
            finally:
                conn.rollback()
            if not row:
                return None
            relation = _row_to_relation(dict(row))
            if _precomputed_emb is not None:
                relation.embedding = _precomputed_emb
            self._invalidate_relation_cache_bulk()
        if _precomputed_emb is not None:
            emb_array = np.frombuffer(_precomputed_emb, dtype=np.float32)
            self._update_relation_emb_cache(relation, emb_array)
        elif needs_emb_update:
            self._update_relation_emb_cache(relation, None)
        return relation

    def update_relation_confidence(self, family_id: str, confidence: float):
        confidence = max(0.0, min(1.0, confidence))
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE relation SET confidence = ? WHERE family_id = ? AND graph_id = ? "
                "AND uuid = (SELECT uuid FROM relation WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1)",
                (confidence, family_id, self._graph_id, family_id, self._graph_id),
            )
            conn.commit()
        finally:
            conn.rollback()
        self._invalidate_relation_cache_bulk()

    def refresh_relates_to_edges(self, family_ids: List[str] = None):
        conn = self._connect()
        try:
            if family_ids:
                # Incremental: refresh edges for specified entity families
                placeholders = ",".join("?" * len(family_ids))
                # Delete stale edges
                conn.execute(
                    f"DELETE FROM relates_to WHERE graph_id = ? AND "
                    f"(entity1_uuid IN (SELECT uuid FROM entity WHERE family_id IN ({placeholders}) AND graph_id = ?) "
                    f"OR entity2_uuid IN (SELECT uuid FROM entity WHERE family_id IN ({placeholders}) AND graph_id = ?))",
                    [self._graph_id] + family_ids + [self._graph_id] + family_ids + [self._graph_id],
                )
                # Recreate edges for valid relations involving these families
                rows = conn.execute(
                    f"SELECT r.uuid AS r_uuid, r.content, r.entity1_absolute_id, r.entity2_absolute_id "
                    f"FROM relation r "
                    f"WHERE r.graph_id = ? "
                    f"AND (r.entity1_absolute_id IN (SELECT uuid FROM entity WHERE family_id IN ({placeholders}) AND graph_id = ?) "
                    f"OR r.entity2_absolute_id IN (SELECT uuid FROM entity WHERE family_id IN ({placeholders}) AND graph_id = ?))",
                    [self._graph_id] + family_ids + [self._graph_id] + family_ids + [self._graph_id],
                ).fetchall()
                for r in rows:
                    e1_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (r["entity1_absolute_id"], self._graph_id)).fetchone()
                    e2_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (r["entity2_absolute_id"], self._graph_id)).fetchone()
                    if e1_row and e2_row:
                        e1_latest = conn.execute(
                            "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                            (e1_row["family_id"], self._graph_id),
                        ).fetchone()
                        e2_latest = conn.execute(
                            "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                            (e2_row["family_id"], self._graph_id),
                        ).fetchone()
                        if e1_latest and e2_latest:
                            conn.execute(
                                "INSERT OR REPLACE INTO relates_to (entity1_uuid, entity2_uuid, relation_uuid, fact, graph_id) VALUES (?, ?, ?, ?, ?)",
                                (e1_latest["uuid"], e2_latest["uuid"], r["r_uuid"], r["content"], self._graph_id),
                            )
                refreshed = len(rows)
                conn.commit()
                if refreshed > 0:
                    logger.info("refresh_relates_to_edges: incremental refresh for %d families, %d edges", len(family_ids), refreshed)
                return {"refreshed": refreshed}
            else:
                # Full refresh
                conn.execute("DELETE FROM relates_to WHERE graph_id = ?", (self._graph_id,))
                rows = conn.execute(
                    "SELECT r.uuid AS r_uuid, r.content, r.entity1_absolute_id, r.entity2_absolute_id "
                    "FROM relation r WHERE r.graph_id = ?",
                    (self._graph_id,),
                ).fetchall()
                created = 0
                for r in rows:
                    e1_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (r["entity1_absolute_id"], self._graph_id)).fetchone()
                    e2_row = conn.execute("SELECT family_id FROM entity WHERE uuid = ? AND graph_id = ?", (r["entity2_absolute_id"], self._graph_id)).fetchone()
                    if e1_row and e2_row:
                        e1_latest = conn.execute(
                            "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                            (e1_row["family_id"], self._graph_id),
                        ).fetchone()
                        e2_latest = conn.execute(
                            "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ? ORDER BY version_seq DESC LIMIT 1",
                            (e2_row["family_id"], self._graph_id),
                        ).fetchone()
                        if e1_latest and e2_latest:
                            conn.execute(
                                "INSERT OR REPLACE INTO relates_to (entity1_uuid, entity2_uuid, relation_uuid, fact, graph_id) VALUES (?, ?, ?, ?, ?)",
                                (e1_latest["uuid"], e2_latest["uuid"], r["r_uuid"], r["content"], self._graph_id),
                            )
                            created += 1
                conn.commit()
                if created > 0:
                    logger.info("refresh_relates_to_edges: full refresh, created=%d new edges", created)
                return {"deleted": 0, "created": created}
        finally:
            conn.rollback()

    def save_dream_relation(self, entity1_id: str, entity2_id: str,
                            content: str, confidence: float, reasoning: str,
                            dream_cycle_id: Optional[str] = None,
                            episode_id: Optional[str] = None) -> Dict[str, Any]:
        resolved_map = self.resolve_family_ids([entity1_id, entity2_id])
        resolved1 = resolved_map.get(entity1_id, entity1_id)
        resolved2 = resolved_map.get(entity2_id, entity2_id)
        if not resolved1:
            raise ValueError(f"Entity not found: {entity1_id}")
        if not resolved2:
            raise ValueError(f"Entity not found: {entity2_id}")
        entities_map = self.get_entities_by_family_ids([resolved1, resolved2])
        entity1 = entities_map.get(resolved1)
        entity2 = entities_map.get(resolved2)
        if not entity1:
            raise ValueError(f"Entity not found: {entity1_id}")
        if not entity2:
            raise ValueError(f"Entity not found: {entity2_id}")
        existing = self.get_relations_by_entities(resolved1, resolved2, include_candidates=True)
        source_doc = f"dream:{dream_cycle_id}" if dream_cycle_id else "dream"
        if existing:
            latest = existing[0]
            new_confidence = max(latest.confidence or 0, confidence)
            new_prov_entry = {"source": "dream", "dream_cycle_id": dream_cycle_id, "confidence": confidence, "reasoning": reasoning}
            try:
                old_prov = json.loads(latest.provenance) if latest.provenance else []
            except Exception:
                old_prov = []
            old_prov.append(new_prov_entry)
            now = datetime.now()
            record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            merged_content = f"{latest.content}\n[Dream update] {content}" if content != latest.content else latest.content
            try:
                merged_attrs = json.loads(latest.attributes) if latest.attributes else {}
            except (json.JSONDecodeError, TypeError):
                merged_attrs = {}
            if dream_cycle_id:
                merged_attrs.setdefault("additional_dream_cycles", [])
                merged_attrs["additional_dream_cycles"].append(dream_cycle_id)
            relation = Relation(
                absolute_id=record_id, family_id=latest.family_id,
                entity1_absolute_id=latest.entity1_absolute_id,
                entity2_absolute_id=latest.entity2_absolute_id,
                content=merged_content, event_time=now, processed_time=now,
                episode_id=episode_id or latest.episode_id or "",
                source_document=source_doc, confidence=new_confidence,
                provenance=json.dumps(old_prov, ensure_ascii=False),
                attributes=json.dumps(merged_attrs) if merged_attrs else latest.attributes,
            )
            self.save_relation(relation)
            return {"family_id": latest.family_id, "entity1_family_id": resolved1, "entity2_family_id": resolved2,
                    "entity1_name": entity1.name, "entity2_name": entity2.name, "action": "merged"}
        if entity1.name <= entity2.name:
            e1_abs, e2_abs = entity1.absolute_id, entity2.absolute_id
        else:
            e1_abs, e2_abs = entity2.absolute_id, entity1.absolute_id
        now = datetime.now()
        family_id = f"rel_{uuid.uuid4().hex[:12]}"
        record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        provenance_data = {"source": "dream", "dream_cycle_id": dream_cycle_id, "confidence": confidence, "reasoning": reasoning}
        relation = Relation(
            absolute_id=record_id, family_id=family_id,
            entity1_absolute_id=e1_abs, entity2_absolute_id=e2_abs,
            content=content, event_time=now, processed_time=now,
            episode_id=episode_id or "", source_document=source_doc,
            confidence=min(confidence, 0.5),
            provenance=json.dumps([provenance_data], ensure_ascii=False),
            attributes=json.dumps({
                "tier": "candidate", "status": "hypothesized", "corroboration_count": 0,
                "created_by_dream": dream_cycle_id or "unknown", "created_at": now.isoformat(),
            }),
        )
        self.save_relation(relation)
        return {"family_id": family_id, "entity1_family_id": resolved1, "entity2_family_id": resolved2,
                "entity1_name": entity1.name, "entity2_name": entity2.name, "action": "created"}

    # ==================================================================
    # EPISODE STORE
    # ==================================================================

    def _compute_episode_embedding(self, content: str) -> Optional[bytes]:
        if not content:
            return None
        result = _encode_and_normalize(self.embedding_client, f"# Episode\n{content}")
        return result[0] if result else None

    def _get_cache_dir_by_doc_hash(self, doc_hash: str, document_path: str = "") -> Optional[Path]:
        if not doc_hash:
            return None
        doc_dir = self.docs_dir / doc_hash
        if doc_dir.is_dir():
            return doc_dir
        dirname = self._doc_hash_to_dirname.get(doc_hash)
        if dirname:
            candidate = self.docs_dir / dirname
            if candidate.is_dir():
                return candidate
        # Filesystem fallback: scan for directory ending with _{doc_hash}
        if self.docs_dir.is_dir():
            for d in self.docs_dir.iterdir():
                if d.is_dir() and d.name.endswith(f"_{doc_hash}"):
                    self._doc_hash_to_dirname[doc_hash] = d.name
                    return d
        return None

    _meta_files_cache: tuple = (0.0, None)
    _bm25_lower_cache: tuple = (0.0, None)
    _meta_json_cache: dict = {}
    _META_FILES_TTL: float = 2.0

    def _iter_cache_meta_files(self) -> List[Path]:
        now = time.monotonic()
        cached_ts, cached_files = self._meta_files_cache
        if cached_files is not None and now - cached_ts < self._META_FILES_TTL:
            return cached_files
        if not self.docs_dir.is_dir():
            files = []
        else:
            files = sorted(self.docs_dir.glob("*/meta.json"))
        self._meta_files_cache = (now, files)
        return files

    def save_episode(self, cache: Episode, text: str = "", document_path: str = "", doc_hash: str = "") -> str:
        if not doc_hash and text:
            doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        if not doc_hash:
            doc_hash = "unknown"
        _now = datetime.now()
        ts_prefix = cache.event_time.strftime("%Y%m%d_%H%M%S") if cache.event_time else _now.strftime("%Y%m%d_%H%M%S")
        dir_name = f"{ts_prefix}_{doc_hash}"
        doc_dir = self.docs_dir / dir_name
        self.docs_dir.mkdir(exist_ok=True)
        doc_dir.mkdir(parents=True, exist_ok=True)
        if text:
            original_path = doc_dir / "original.txt"
            if not original_path.exists():
                original_path.write_text(text, encoding="utf-8")
        content = clean_markdown_code_blocks(cache.content)
        (doc_dir / "cache.md").write_text(content, encoding="utf-8")
        _proc_time = (cache.processed_time or _now).isoformat()
        meta = {
            "absolute_id": cache.absolute_id,
            "event_time": cache.event_time.isoformat(),
            "processed_time": _proc_time,
            "activity_type": cache.activity_type,
            "source_document": cache.source_document,
            "text": text, "document_path": document_path, "doc_hash": doc_hash,
        }
        (doc_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if cache.absolute_id:
            self._id_to_doc_hash[cache.absolute_id] = doc_dir.name
        self._doc_hash_to_dirname[doc_hash] = dir_name
        embedding_blob = self._compute_episode_embedding(cache.content)
        with self._episode_write_lock:
            conn = self._connect()
            try:
                conn.execute(
                    f"INSERT OR REPLACE INTO episode ({', '.join(EPISODE_COLUMNS)}) VALUES ({', '.join('?' * len(EPISODE_COLUMNS))})",
                    (
                        cache.absolute_id, self._graph_id, cache.content,
                        text or "", cache.source_document,
                        _fmt_dt(cache.event_time), _proc_time,
                        getattr(cache, 'episode_type', None),
                        getattr(cache, 'activity_type', None),
                        doc_hash, _now.isoformat(), embedding_blob,
                    ),
                )
                conn.commit()
            finally:
                conn.rollback()
        return doc_hash

    def bulk_save_episodes(self, episodes: list) -> int:
        if not episodes:
            return 0
        _now_iso = datetime.now().isoformat()
        embeddings = None
        if self.embedding_client and self.embedding_client.is_available():
            texts = [f"# Episode\n{ep.content}" for ep in episodes if ep.content]
            if texts:
                embeddings = self.embedding_client.encode(texts)
        rows = []
        ep_idx = 0
        for ep in episodes:
            embedding_blob = None
            if embeddings is not None and ep.content and ep.absolute_id:
                if ep_idx < len(embeddings):
                    try:
                        emb_arr = np.array(embeddings[ep_idx], dtype=np.float32)
                        norm = np.linalg.norm(emb_arr)
                        if norm > 0:
                            emb_arr = emb_arr / norm
                        embedding_blob = emb_arr.tobytes()
                    except Exception:
                        pass
                ep_idx += 1
            rows.append((
                ep.absolute_id, self._graph_id, ep.content or "",
                "", getattr(ep, "source_document", "") or "",
                ep.event_time.isoformat() if ep.event_time else _now_iso,
                _now_iso, getattr(ep, "episode_type", None),
                getattr(ep, "activity_type", None), None, _now_iso,
                embedding_blob,
            ))
        with self._episode_write_lock:
            conn = self._connect()
            try:
                conn.executemany(
                    f"INSERT OR REPLACE INTO episode ({', '.join(EPISODE_COLUMNS)}) VALUES ({', '.join('?' * len(EPISODE_COLUMNS))})",
                    rows,
                )
                conn.commit()
            finally:
                conn.rollback()
        return len(rows)

    def count_episodes(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM episode WHERE graph_id = ?", (self._graph_id,)).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def delete_episode(self, cache_id: str) -> int:
        doc_hash = self._resolve_doc_hash(cache_id)
        if doc_hash:
            doc_dir = self.docs_dir / doc_hash
            if doc_dir.is_dir():
                shutil.rmtree(doc_dir, ignore_errors=True)
                self._id_to_doc_hash.pop(cache_id, None)
        with self._episode_write_lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM mentions WHERE episode_uuid = ? AND graph_id = ?", (cache_id, self._graph_id))
                cursor = conn.execute("DELETE FROM episode WHERE uuid = ? AND graph_id = ?", (cache_id, self._graph_id))
                conn.commit()
                deleted = cursor.rowcount
            finally:
                conn.rollback()
        if deleted > 0:
            return 1
        for base_dir in (self.cache_json_dir, self.cache_dir):
            meta_path = base_dir / f"{cache_id}.json"
            if meta_path.exists():
                meta_path.unlink(missing_ok=True)
                return 1
        return 0

    def delete_episode_mentions(self, episode_id: str):
        conn = self._connect()
        try:
            conn.execute("DELETE FROM mentions WHERE episode_uuid = ? AND graph_id = ?", (episode_id, self._graph_id))
            conn.commit()
        finally:
            conn.rollback()

    def _resolve_doc_hash(self, cache_id: str) -> Optional[str]:
        # 1. Direct mapping from absolute_id to directory name
        doc_hash = self._id_to_doc_hash.get(cache_id)
        if doc_hash:
            return doc_hash
        # 2. Look up by uuid in episode table
        try:
            conn = self._connect()
            try:
                row = conn.execute("SELECT doc_hash FROM episode WHERE uuid = ? AND graph_id = ?", (cache_id, self._graph_id)).fetchone()
            finally:
                conn.rollback()
            if row and row["doc_hash"]:
                doc_hash = row["doc_hash"]
                self._id_to_doc_hash[cache_id] = doc_hash
                return doc_hash
        except Exception:
            pass
        # 3. Try as a bare doc_hash — find matching directory on filesystem
        for d in self.docs_dir.iterdir():
            if d.is_dir() and d.name.endswith(f"_{cache_id}"):
                self._id_to_doc_hash[cache_id] = d.name
                return d.name
        return None

    def get_doc_hash_by_cache_id(self, cache_id: str) -> Optional[str]:
        return self._resolve_doc_hash(cache_id)

    def find_cache_by_doc_hash(self, doc_hash: str, document_path: str = "") -> Optional[Episode]:
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT uuid, source_document, event_time, processed_time, activity_type FROM episode WHERE doc_hash = ? AND graph_id = ? LIMIT 1",
                    (doc_hash, self._graph_id),
                ).fetchone()
            finally:
                conn.rollback()
            if row and row["uuid"]:
                cache_id = row["uuid"]
                cache_md = self.cache_md_dir / f"{cache_id}.md"
                content = cache_md.read_text(encoding="utf-8") if cache_md.exists() else ""
                return Episode(
                    absolute_id=cache_id, content=content,
                    event_time=_parse_dt(row["event_time"]) or datetime.now(),
                    processed_time=_parse_dt(row["processed_time"]),
                    source_document=row["source_document"] or "",
                    activity_type=row["activity_type"],
                )
        except Exception:
            pass
        for meta_file in self._iter_cache_meta_files():
            try:
                mf_key = str(meta_file)
                meta = self._meta_json_cache.get(mf_key)
                if meta is None:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    self._meta_json_cache[mf_key] = meta
                if meta.get("doc_hash") == doc_hash:
                    _cache_md = meta_file.parent / "cache.md"
                    return Episode(
                        absolute_id=meta.get("absolute_id", ""),
                        content=_cache_md.read_text(encoding="utf-8") if _cache_md.exists() else "",
                        event_time=_parse_dt(meta.get("event_time")) or datetime.now(),
                        processed_time=_parse_dt(meta.get("processed_time")),
                        source_document=meta.get("source_document", ""),
                        activity_type=meta.get("activity_type"),
                    )
            except Exception:
                continue
        return None

    def find_cache_and_extraction_by_doc_hash(self, doc_hash: str, document_path: str = "") -> Tuple[Optional[Episode], Optional[tuple]]:
        if not doc_hash:
            return None, None
        episode = None
        extraction = None
        doc_dir = self._get_cache_dir_by_doc_hash(doc_hash, document_path)
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT uuid, source_document, event_time, processed_time, activity_type FROM episode WHERE doc_hash = ? AND graph_id = ? LIMIT 1",
                    (doc_hash, self._graph_id),
                ).fetchone()
            finally:
                conn.rollback()
            if row and row["uuid"]:
                cache_id = row["uuid"]
                cache_md = self.cache_md_dir / f"{cache_id}.md"
                content = cache_md.read_text(encoding="utf-8") if cache_md.exists() else ""
                episode = Episode(
                    absolute_id=cache_id, content=content,
                    event_time=_parse_dt(row["event_time"]) or datetime.now(),
                    processed_time=_parse_dt(row["processed_time"]),
                    source_document=row["source_document"] or "",
                    activity_type=row["activity_type"],
                )
        except Exception:
            pass
        if doc_dir:
            extraction_path = doc_dir / "extraction.json"
            if extraction_path.exists():
                try:
                    data = json.loads(extraction_path.read_text(encoding="utf-8"))
                    extraction = (data.get("entities", []), data.get("relations", []))
                except Exception:
                    pass
        if episode is None:
            if doc_dir:
                meta_file = doc_dir / "meta.json"
                if meta_file.exists():
                    try:
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                        _cache_md = doc_dir / "cache.md"
                        episode = Episode(
                            absolute_id=meta.get("absolute_id", ""),
                            content=_cache_md.read_text(encoding="utf-8") if _cache_md.exists() else "",
                            event_time=_parse_dt(meta.get("event_time")) or datetime.now(),
                            processed_time=_parse_dt(meta.get("processed_time")),
                            source_document=meta.get("source_document", ""),
                            activity_type=meta.get("activity_type"),
                        )
                    except Exception:
                        pass
            else:
                for meta_file in self._iter_cache_meta_files():
                    try:
                        mf_key = str(meta_file)
                        meta = self._meta_json_cache.get(mf_key)
                        if meta is None:
                            meta = json.loads(meta_file.read_text(encoding="utf-8"))
                            self._meta_json_cache[mf_key] = meta
                        if meta.get("doc_hash") == doc_hash:
                            _cache_md = meta_file.parent / "cache.md"
                            episode = Episode(
                                absolute_id=meta.get("absolute_id", ""),
                                content=_cache_md.read_text(encoding="utf-8") if _cache_md.exists() else "",
                                event_time=_parse_dt(meta.get("event_time")) or datetime.now(),
                                processed_time=_parse_dt(meta.get("processed_time")),
                                source_document=meta.get("source_document", ""),
                                activity_type=meta.get("activity_type"),
                            )
                            break
                    except Exception:
                        continue
        return episode, extraction

    def get_doc_content(self, filename: str) -> Optional[Dict[str, Any]]:
        doc_dir = self.docs_dir / filename
        if not doc_dir.is_dir():
            return None
        try:
            original_path = doc_dir / "original.txt"
            cache_path = doc_dir / "cache.md"
            meta_path = doc_dir / "meta.json"
            return {
                "original": original_path.read_text(encoding="utf-8") if original_path.exists() else "",
                "cache": cache_path.read_text(encoding="utf-8") if cache_path.exists() else "",
                "meta": json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {},
            }
        except Exception:
            return None

    def get_doc_dir(self, doc_hash: str) -> Optional[Path]:
        return self._get_cache_dir_by_doc_hash(doc_hash)

    def get_episode(self, uuid: str) -> Optional[Dict]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM episode WHERE uuid = ? AND graph_id = ?",
                (uuid, self._graph_id),
            ).fetchone()
            if not row:
                return None
            mentions_count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM mentions WHERE episode_uuid = ? AND graph_id = ?",
                (uuid, self._graph_id),
            ).fetchone()["cnt"]
        finally:
            conn.rollback()
        rd = dict(row)
        return {
            "uuid": rd["uuid"],
            "content": rd["content"] or "",
            "source_text": rd.get("source_text") or "",
            "source_document": rd["source_document"] or "",
            "event_time": _fmt_dt(rd.get("event_time")),
            "episode_id": rd.get("uuid", ""),
            "created_at": _fmt_dt(rd.get("created_at")),
            "mentions_count": mentions_count,
        }

    def get_episode_entities(self, episode_id: str) -> List[dict]:
        results = []
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT m.target_uuid, m.target_type, m.context, m.entity_absolute_id "
                "FROM mentions m WHERE m.episode_uuid = ? AND m.graph_id = ?",
                (episode_id, self._graph_id),
            ).fetchall()
            target_uuids = [r["target_uuid"] for r in rows]
            ent_map = {}
            rel_map = {}
            if target_uuids:
                placeholders = ",".join("?" * len(target_uuids))
                ent_rows = conn.execute(
                    f"SELECT uuid, family_id, name FROM entity WHERE uuid IN ({placeholders}) AND graph_id = ?",
                    target_uuids + [self._graph_id],
                ).fetchall()
                for r in ent_rows:
                    ent_map[r["uuid"]] = {"family_id": r["family_id"], "name": r["name"]}
                rel_rows = conn.execute(
                    f"SELECT uuid, family_id FROM relation WHERE uuid IN ({placeholders}) AND graph_id = ?",
                    target_uuids + [self._graph_id],
                ).fetchall()
                for r in rel_rows:
                    rel_map[r["uuid"]] = {"family_id": r["family_id"]}
        finally:
            conn.rollback()
        for r in rows:
            target_type = r["target_type"]
            target_uuid = r["target_uuid"]
            info = ent_map.get(target_uuid) if target_type == "entity" else rel_map.get(target_uuid)
            results.append({
                "absolute_id": target_uuid,
                "target_type": target_type,
                "name": (info.get("name", "") if info else ""),
                "family_id": (info.get("family_id", "") if info else ""),
                "mention_context": r["context"] or "",
            })
        return results

    def get_episode_text(self, cache_id: str) -> Optional[str]:
        doc_hash = self._resolve_doc_hash(cache_id)
        if doc_hash:
            doc_dir = self.docs_dir / doc_hash
            original_path = doc_dir / "original.txt"
            if original_path.exists():
                return original_path.read_text(encoding="utf-8")
            meta_path = doc_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    return meta.get("text")
                except Exception:
                    pass
        metadata_path = self.cache_json_dir / f"{cache_id}.json"
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                return meta.get("text", "")
            except Exception:
                pass
        return None

    def get_latest_episode(self, activity_type: Optional[str] = None) -> Optional[Episode]:
        conn = self._connect()
        try:
            if activity_type:
                row = conn.execute(
                    "SELECT * FROM episode WHERE activity_type = ? AND graph_id = ? ORDER BY created_at DESC LIMIT 1",
                    (activity_type, self._graph_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM episode WHERE graph_id = ? ORDER BY created_at DESC LIMIT 1",
                    (self._graph_id,),
                ).fetchone()
        finally:
            conn.rollback()
        if row:
            rd = dict(row)
            return Episode(
                absolute_id=rd["uuid"], content=rd["content"] or "",
                event_time=_parse_dt(rd["event_time"]) or datetime.now(),
                processed_time=_parse_dt(rd["processed_time"]),
                source_document=rd["source_document"] or "",
                activity_type=rd.get("activity_type"),
            )
        return None

    def get_latest_episode_metadata(self, activity_type: Optional[str] = None) -> Optional[Dict]:
        cache_key = f"latest_episode_meta:{activity_type or ''}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        cache_files = self._iter_cache_meta_files()
        if not cache_files:
            self._cache.set(cache_key, None, ttl=60)
            return None
        latest_metadata = None
        latest_time = None
        for cache_file in cache_files:
            try:
                cf_key = str(cache_file)
                metadata = self._meta_json_cache.get(cf_key)
                if metadata is None:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    self._meta_json_cache[cf_key] = metadata
            except Exception:
                continue
            if activity_type and metadata.get("activity_type") != activity_type:
                continue
            evt_str = metadata.get("event_time")
            try:
                cache_time = _parse_dt(evt_str) or datetime.now(timezone.utc)
            except (TypeError, ValueError):
                cache_time = datetime.now(timezone.utc)
            if latest_time is None or cache_time > latest_time:
                latest_time = cache_time
                latest_metadata = metadata
        self._cache.set(cache_key, latest_metadata, ttl=60)
        return latest_metadata

    def list_docs(self) -> List[Dict[str, Any]]:
        results = []
        for meta_file in self._iter_cache_meta_files():
            try:
                mf_key = str(meta_file)
                meta = self._meta_json_cache.get(mf_key)
                if meta is None:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    self._meta_json_cache[mf_key] = meta
                doc_dir = meta_file.parent
                original_path = doc_dir / "original.txt"
                text_length = 0
                original_size = 0
                if original_path.exists():
                    try:
                        raw = original_path.read_text(encoding="utf-8")
                        text_length = len(raw)
                        original_size = original_path.stat().st_size
                    except Exception:
                        pass
                cache_size = 0
                cache_path = doc_dir / "cache.md"
                if cache_path.exists():
                    try:
                        cache_size = cache_path.stat().st_size
                    except Exception:
                        pass
                results.append({
                    "id": meta.get("absolute_id", ""), "doc_hash": meta.get("doc_hash", ""),
                    "event_time": meta.get("event_time", ""), "processed_time": meta.get("processed_time", ""),
                    "source_document": meta.get("source_document", ""), "document_path": meta.get("document_path", ""),
                    "dir_name": doc_dir.name, "activity_type": meta.get("activity_type", ""),
                    "text_length": text_length, "original_size": original_size, "cache_size": cache_size,
                })
            except Exception:
                continue
        return results

    def list_episodes(self, limit: int = 20, offset: int = 0, include_text: bool = False) -> List[Dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT uuid, content, source_document, event_time, processed_time, uuid as episode_id, created_at"
                + (", source_text" if include_text else "") +
                " FROM episode WHERE graph_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (self._graph_id, limit, offset),
            ).fetchall()
        finally:
            conn.rollback()
        episodes = []
        for r in rows:
            ep = {
                "uuid": r["uuid"], "content": r["content"] or "",
                "source_document": r["source_document"] or "",
                "event_time": _fmt_dt(r["event_time"]),
                "processed_time": _fmt_dt(r["processed_time"]),
                "episode_id": r["episode_id"] or "",
                "created_at": _fmt_dt(r["created_at"]),
            }
            if include_text:
                ep["source_text"] = r["source_text"] if "source_text" in r.keys() else ""
            episodes.append(ep)
        return episodes

    def load_episode(self, cache_id: str) -> Optional[Episode]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT content, event_time, processed_time, source_document FROM episode WHERE uuid = ? AND graph_id = ?",
                (cache_id, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        if row:
            return Episode(
                absolute_id=cache_id, content=row["content"] or "",
                event_time=_parse_dt(row["event_time"]) or datetime.now(),
                processed_time=_parse_dt(row["processed_time"]),
                source_document=row["source_document"] or "",
            )
        doc_hash = self._resolve_doc_hash(cache_id)
        if doc_hash:
            doc_dir = self.docs_dir / doc_hash
            meta_path = doc_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    _cache_md = doc_dir / "cache.md"
                    return Episode(
                        absolute_id=cache_id,
                        content=_cache_md.read_text(encoding="utf-8") if _cache_md.exists() else "",
                        event_time=_parse_dt(meta.get("event_time")) or datetime.now(),
                        processed_time=_parse_dt(meta.get("processed_time")),
                        source_document=meta.get("source_document", ""),
                        activity_type=meta.get("activity_type"),
                    )
                except Exception:
                    pass
        return None

    def load_episodes(self, cache_ids: List[str]) -> List[Episode]:
        if not cache_ids:
            return []
        results_map: Dict[str, Episode] = {}
        try:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(cache_ids))
                rows = conn.execute(
                    f"SELECT uuid, content, event_time, processed_time, source_document FROM episode WHERE uuid IN ({placeholders}) AND graph_id = ?",
                    cache_ids + [self._graph_id],
                ).fetchall()
            finally:
                conn.rollback()
            for r in rows:
                results_map[r["uuid"]] = Episode(
                    absolute_id=r["uuid"], content=r["content"] or "",
                    event_time=_parse_dt(r["event_time"]) or datetime.now(),
                    processed_time=_parse_dt(r["processed_time"]),
                    source_document=r["source_document"] or "",
                )
        except Exception:
            pass
        missing = [cid for cid in cache_ids if cid not in results_map]
        for cache_id in missing:
            doc_hash = self._resolve_doc_hash(cache_id)
            if doc_hash:
                doc_dir = self.docs_dir / doc_hash
                meta_path = doc_dir / "meta.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        _cache_md = doc_dir / "cache.md"
                        results_map[cache_id] = Episode(
                            absolute_id=cache_id,
                            content=_cache_md.read_text(encoding="utf-8") if _cache_md.exists() else "",
                            event_time=_parse_dt(meta.get("event_time")) or datetime.now(),
                            processed_time=_parse_dt(meta.get("processed_time")),
                            source_document=meta.get("source_document", ""),
                        )
                    except Exception:
                        pass
        return [results_map[cid] for cid in cache_ids if cid in results_map]

    def load_extraction_result(self, doc_hash: str, document_path: str = "") -> Optional[tuple]:
        doc_dir = self._get_cache_dir_by_doc_hash(doc_hash, document_path)
        if not doc_dir:
            return None
        extraction_path = doc_dir / "extraction.json"
        if not extraction_path.exists():
            return None
        try:
            data = json.loads(extraction_path.read_text(encoding="utf-8"))
            return data.get("entities", []), data.get("relations", [])
        except Exception:
            return None

    def save_extraction_result(self, doc_hash: str, entities: list, relations: list, document_path: str = "") -> bool:
        doc_dir = self._get_cache_dir_by_doc_hash(doc_hash, document_path)
        if not doc_dir:
            return False
        try:
            result = {
                "entities": [{"absolute_id": e.absolute_id, "family_id": e.family_id, "name": e.name, "content": e.content} for e in entities],
                "relations": [{"absolute_id": r.absolute_id, "family_id": r.family_id, "content": r.content} for r in relations],
            }
            (doc_dir / "extraction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return True
        except Exception:
            return False

    def save_episode_mentions(self, episode_id: str, entity_absolute_ids: List[str], context: str = "", target_type: str = "entity"):
        if not entity_absolute_ids:
            return
        with self._episode_write_lock:
            conn = self._connect()
            try:
                rows = []
                for aid in entity_absolute_ids:
                    entity_abs_id = None
                    if target_type == "entity":
                        r = conn.execute("SELECT uuid, family_id FROM entity WHERE uuid = ? AND graph_id = ?", (aid, self._graph_id)).fetchone()
                        if r:
                            entity_abs_id = aid
                    rows.append((episode_id, aid, target_type, context, entity_abs_id, self._graph_id))
                conn.executemany(
                    "INSERT OR REPLACE INTO mentions (episode_uuid, target_uuid, target_type, context, entity_absolute_id, graph_id) VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
            finally:
                conn.rollback()

    def batch_get_source_text_snippets(self, episode_ids: List[str], snippet_length: int = 200) -> Dict[str, str]:
        if not episode_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(episode_ids))
            rows = conn.execute(
                f"SELECT uuid, source_text FROM episode WHERE uuid IN ({placeholders}) AND graph_id = ?",
                episode_ids + [self._graph_id],
            ).fetchall()
        finally:
            conn.rollback()
        return {r["uuid"]: (dict(r).get("source_text") or "")[:snippet_length] for r in rows}

    def search_episodes(self, query: str, limit: int = 20) -> List[Dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT uuid, content, source_text, source_document, event_time, uuid as episode_id, created_at "
                "FROM episode WHERE content LIKE ? AND graph_id = ? ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", self._graph_id, limit),
            ).fetchall()
        finally:
            conn.rollback()
        episodes = []
        for r in rows:
            episodes.append({
                "uuid": r["uuid"], "content": r["content"] or "",
                "source_text": dict(r).get("source_text") or "",
                "source_document": r["source_document"] or "",
                "event_time": _fmt_dt(r["event_time"]),
                "episode_id": r["episode_id"] or "",
                "created_at": _fmt_dt(r["created_at"]),
            })
        return episodes

    def search_episodes_by_bm25(self, query: str, limit: int = 20) -> List[Episode]:
        if not query:
            return []
        query_lower = query.lower()
        now = time.monotonic()
        _cache_ts, _cache_map = self._bm25_lower_cache
        if _cache_map is None or now - _cache_ts > self._META_FILES_TTL:
            _cache_map = {}
            self._bm25_lower_cache = (now, _cache_map)
            self._meta_json_cache.clear()
        scored: List[Tuple[int, str]] = []
        for meta_file in self._iter_cache_meta_files():
            try:
                mf_key = str(meta_file)
                meta = self._meta_json_cache.get(mf_key)
                if meta is None:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    self._meta_json_cache[mf_key] = meta
                cache_id = meta.get("absolute_id") or meta.get("id") or meta_file.parent.name
            except Exception:
                continue
            content_lower = _cache_map.get(mf_key)
            if content_lower is None:
                content_path = meta_file.parent / "cache.md"
                try:
                    content = content_path.read_text(encoding="utf-8") if content_path.exists() else ""
                except Exception:
                    content = ""
                content_lower = content.lower()
                _cache_map[mf_key] = content_lower
            if query_lower in content_lower:
                score = content_lower.count(query_lower)
                scored.append((score, cache_id))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_ids = [cid for _, cid in scored[:limit]]
        if not top_ids:
            return []
        return self.load_episodes(top_ids)

    # ==================================================================
    # SEARCH (BM25 + Vector Similarity)
    # ==================================================================

    def search_entities_by_bm25(self, query: str, limit: int = 20) -> List[Entity]:
        if not query:
            return []
        cache_key = f"bm25_entity:{hash(query)}:{limit}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            conn = self._connect()
            try:
                raw_limit = min(limit * 5, 500)
                # FTS5 MATCH query - split into tokens and join with OR for broad matching
                tokens = query.replace('"', '').split()
                if tokens:
                    fts_query = ' OR '.join(f'"{t}"' for t in tokens[:10])
                else:
                    fts_query = '""'
                rows = conn.execute(
                    f"SELECT rowid FROM entity_fts WHERE entity_fts MATCH ? AND graph_id = ? ORDER BY rank LIMIT ?",
                    (fts_query, self._graph_id, raw_limit),
                ).fetchall()
                if not rows:
                    # Fallback to LIKE
                    rows = conn.execute(
                        f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity "
                        f"WHERE (name LIKE ? OR content LIKE ?) AND graph_id = ? "
                        f"ORDER BY processed_time DESC LIMIT ?",
                        (f"%{query}%", f"%{query}%", self._graph_id, raw_limit),
                    ).fetchall()
                else:
                    # Get actual entity data from the matched FTS rows
                    rowids = [r["rowid"] for r in rows]
                    # Get the corresponding entity uuids by joining
                    rid_ph = ",".join("?" * len(rowids))
                    rows = conn.execute(
                        f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity "
                        f"WHERE rowid IN ({rid_ph}) AND graph_id = ? "
                        f"ORDER BY processed_time DESC",
                        rowids + [self._graph_id],
                    ).fetchall()
            finally:
                conn.rollback()
            seen_fids = set()
            entities = []
            for row in rows:
                entity = _row_to_entity(dict(row))
                if entity.family_id and entity.family_id in seen_fids:
                    continue
                if entity.family_id:
                    seen_fids.add(entity.family_id)
                entities.append(entity)
                if len(entities) >= limit:
                    break
            # Prefix match supplement
            _has_core_match = False
            for ent in entities:
                name = ent.name
                if name == query or name.startswith(query + "(") or name.startswith(query + "("):
                    _has_core_match = True
                    break
            if not _has_core_match and len(query) >= 2:
                conn = self._connect()
                try:
                    prefix_rows = conn.execute(
                        f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity "
                        f"WHERE (name LIKE ? OR name = ?) AND graph_id = ? "
                        f"ORDER BY processed_time DESC LIMIT 5",
                        (query + "%", query, self._graph_id),
                    ).fetchall()
                finally:
                    conn.rollback()
                for r in prefix_rows:
                    entity = _row_to_entity(dict(r))
                    if entity.family_id and entity.family_id not in seen_fids:
                        seen_fids.add(entity.family_id)
                        entities.append(entity)
            # Resolve redirects
            raw_fids = [e.family_id for e in entities if e.family_id]
            if raw_fids:
                resolved_map = self.resolve_family_ids(raw_fids)
                for ent in entities:
                    resolved_fid = resolved_map.get(ent.family_id, ent.family_id) if ent.family_id else ent.family_id
                    if resolved_fid != ent.family_id:
                        ent.family_id = resolved_fid
            result = entities[:limit]
            self._cache.set(cache_key, result, ttl=30)
            return result
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)
            self._cache.set(cache_key, [], ttl=10)
            return []

    def _search_with_embedding(self, query_text: str, entities_with_embeddings: List[tuple],
                                threshold: float, use_content: bool = False,
                                max_results: int = 10, content_snippet_length: int = 50,
                                text_mode: str = "name_and_content", query_embedding=None) -> List[Entity]:
        if query_embedding is None:
            query_embedding = self.embedding_client.encode(query_text)
        if query_embedding is None:
            return self.search_entities_by_bm25(query_text, limit=max_results * 3)[:max_results]
        query_emb = np.asarray(query_embedding, dtype=np.float32)
        if query_emb.ndim > 1:
            query_emb = query_emb[0]
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm
        entities_with_emb = self._get_entities_with_embeddings()
        if not entities_with_emb:
            return self.search_entities_by_bm25(query_text, limit=max_results * 3)[:max_results]

        # Fast path: HNSW approximate nearest neighbor search
        if self._entity_hnsw is not None and self._entity_hnsw_items is not None:
            k = min(max_results * 3, len(self._entity_hnsw_items))
            try:
                labels, distances = self._entity_hnsw.knn_query(query_emb.reshape(1, -1), k=k)
                seen = set()
                results = []
                for idx, dist in zip(labels[0], distances[0]):
                    score = 1.0 - dist  # cosine distance -> similarity
                    if score < threshold:
                        break
                    entity = self._entity_hnsw_items[idx]
                    if entity.family_id in seen:
                        continue
                    seen.add(entity.family_id)
                    results.append(entity)
                    if len(results) >= max_results:
                        break
                if results:
                    return results
            except Exception:
                pass  # fall through to brute-force

        # Brute-force cosine similarity against all entity embeddings
        scored = []
        for entity, emb_array in entities_with_emb:
            if emb_array is None:
                continue
            score = float(np.dot(query_emb, emb_array))
            if score >= threshold:
                scored.append((score, entity))
        scored.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        results = []
        for score, entity in scored:
            if entity.family_id in seen:
                continue
            seen.add(entity.family_id)
            results.append(entity)
            if len(results) >= max_results:
                break
        return results

    def search_entities_by_similarity(self, query_name: str, query_content: Optional[str] = None,
                                       threshold: float = 0.7, max_results: int = 10,
                                       content_snippet_length: int = 50,
                                       text_mode: Literal["name_only", "content_only", "name_and_content"] = "name_and_content",
                                       similarity_method: Literal["embedding", "text", "jaccard", "bleu"] = "embedding",
                                       query_embedding=None) -> List[Entity]:
        cache_key = f"sim_search:{hash(query_name)}:{hash(query_content or '')}:{threshold}:{max_results}:{text_mode}:{similarity_method}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with _perf_timer("search_entities_by_similarity"):
            if text_mode == "name_only":
                query_text = query_name
            elif text_mode == "content_only":
                if not query_content:
                    self._cache.set(cache_key, [], ttl=30)
                    return []
                query_text = query_content
            else:
                query_text = f"{query_name} {query_content}" if query_content else query_name
            if similarity_method == "embedding" and self.embedding_client and self.embedding_client.is_available():
                result = self._search_with_embedding(query_text, [], threshold, False, max_results, content_snippet_length, text_mode, query_embedding=query_embedding)
            else:
                result = self.search_entities_by_bm25(query_text, limit=max_results * 3)[:max_results]
            self._cache.set(cache_key, result, ttl=30)
            return result

    def search_relations_by_bm25(self, query: str, limit: int = 20, include_candidates: bool = False) -> List[Relation]:
        if not query:
            return []
        cache_key = f"bm25_relation:{hash(query)}:{limit}:{include_candidates}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            conn = self._connect()
            try:
                raw_limit = min(limit * 5, 500)
                # FTS5 MATCH query - split into tokens and join with OR for broad matching
                tokens = query.replace('"', '').split()
                if tokens:
                    fts_query = ' OR '.join(f'"{t}"' for t in tokens[:10])
                else:
                    fts_query = '""'
                fts_rows = conn.execute(
                    "SELECT rowid FROM relation_fts WHERE relation_fts MATCH ? AND graph_id = ? ORDER BY rank LIMIT ?",
                    (fts_query, self._graph_id, raw_limit),
                ).fetchall()
                if not fts_rows:
                    rows = conn.execute(
                        f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation "
                        f"WHERE content LIKE ? AND graph_id = ? "
                        f"ORDER BY processed_time DESC LIMIT ?",
                        (f"%{query}%", self._graph_id, raw_limit),
                    ).fetchall()
                else:
                    rowids = [r["rowid"] for r in fts_rows]
                    rid_ph = ",".join("?" * len(rowids))
                    rows = conn.execute(
                        f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation "
                        f"WHERE rowid IN ({rid_ph}) AND graph_id = ? "
                        f"ORDER BY processed_time DESC",
                        rowids + [self._graph_id],
                    ).fetchall()
            finally:
                conn.rollback()
            seen_fids = set()
            relations = []
            for r in rows:
                rel = _row_to_relation(dict(r))
                if rel.family_id and rel.family_id in seen_fids:
                    continue
                if rel.family_id:
                    seen_fids.add(rel.family_id)
                relations.append(rel)
                if len(relations) >= limit:
                    break
            result = self._filter_dream_candidates(relations, include_candidates)
            self._cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.warning("Relation BM25 search failed: %s", e)
            return []

    def _search_relations_with_embedding(self, query_text: str, relations_with_embeddings: List[tuple],
                                          threshold: float, max_results: int, query_embedding=None) -> List[Relation]:
        if query_embedding is None:
            query_embedding = self.embedding_client.encode(query_text)
        if query_embedding is None:
            return []
        query_emb = np.asarray(query_embedding, dtype=np.float32)
        if query_emb.ndim > 1:
            query_emb = query_emb[0]
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm
        rels_with_emb = self._get_relations_with_embeddings()
        if not rels_with_emb:
            return []

        # Fast path: HNSW approximate nearest neighbor search
        if self._relation_hnsw is not None and self._relation_hnsw_items is not None:
            k = min(max_results * 3, len(self._relation_hnsw_items))
            try:
                labels, distances = self._relation_hnsw.knn_query(query_emb.reshape(1, -1), k=k)
                seen = set()
                results = []
                for idx, dist in zip(labels[0], distances[0]):
                    score = 1.0 - dist
                    if score < threshold:
                        break
                    rel = self._relation_hnsw_items[idx]
                    if rel.family_id in seen:
                        continue
                    seen.add(rel.family_id)
                    results.append(rel)
                    if len(results) >= max_results:
                        break
                if results:
                    return results
            except Exception:
                pass

        # Brute-force cosine similarity
        scored = []
        for rel, emb_array in rels_with_emb:
            if emb_array is None:
                continue
            score = float(np.dot(query_emb, emb_array))
            if score >= threshold:
                scored.append((score, rel))
        scored.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        results = []
        for score, rel in scored:
            if rel.family_id in seen:
                continue
            seen.add(rel.family_id)
            results.append(rel)
            if len(results) >= max_results:
                break
        return results

    def search_relations_by_similarity(self, query_text: str, threshold: float = 0.3,
                                       max_results: int = 10, include_candidates: bool = False,
                                       query_embedding=None) -> List[Relation]:
        if self.embedding_client and self.embedding_client.is_available():
            results = self._search_relations_with_embedding(query_text, [], threshold, max_results, query_embedding=query_embedding)
            return self._filter_dream_candidates(results, include_candidates)
        else:
            return self.search_relations_by_bm25(query_text, limit=max_results, include_candidates=include_candidates)

    # ==================================================================
    # GRAPH TRAVERSAL
    # ==================================================================

    def _build_entity_abs_id_remap(self) -> dict:
        now = time.time()
        if self._entity_remap_cache is not None and (now - self._entity_remap_cache_ts) < self._entity_remap_cache_ttl:
            return self._entity_remap_cache
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT family_id, uuid, processed_time FROM entity WHERE graph_id = ? ORDER BY processed_time DESC",
                (self._graph_id,),
            ).fetchall()
        finally:
            conn.rollback()
        fid_to_uuids: Dict[str, List[str]] = defaultdict(list)
        for r in rows:
            fid_to_uuids[r["family_id"]].append(r["uuid"])
        remap = {}
        for fid, uuids in fid_to_uuids.items():
            latest = uuids[0]
            for u in uuids[1:]:
                if u != latest:
                    remap[u] = latest
        self._entity_remap_cache = remap
        self._entity_remap_cache_ts = now
        return remap

    def batch_bfs_traverse(self, seed_family_ids: List[str], max_depth: int = 2, max_nodes: int = 50,
                           time_point: Optional[str] = None) -> Tuple[List[Entity], List[Relation], Dict[str, int]]:
        if not seed_family_ids:
            return [], [], {}
        conn = self._connect()
        try:
            # Get seed absolute_ids
            placeholders = ",".join("?" * len(seed_family_ids))
            seed_rows = conn.execute(
                "SELECT family_id, uuid FROM entity WHERE family_id IN ({}) AND graph_id = ?".format(placeholders),
                seed_family_ids + [self._graph_id],
            ).fetchall()
            seed_abs_to_fid = {r["uuid"]: r["family_id"] for r in seed_rows}
            seed_fids = [r["family_id"] for r in seed_rows]
            if not seed_fids:
                return [], [], {}
            # BFS via relates_to table
            visited_uuids = set()
            visited_fids = set()
            hop_map: Dict[str, int] = {}
            entities = []
            # Add seeds at hop 0
            current_frontier = list(seed_abs_to_fid.keys())
            for uuid_val in current_frontier:
                visited_uuids.add(uuid_val)
            for fid in seed_fids:
                if fid not in hop_map:
                    hop_map[fid] = 0
                    visited_fids.add(fid)
            for depth in range(1, max_depth + 1):
                if not current_frontier:
                    break
                ph = ",".join("?" * len(current_frontier))
                edge_rows = conn.execute(
                    f"SELECT entity1_uuid, entity2_uuid FROM relates_to WHERE (entity1_uuid IN ({ph}) OR entity2_uuid IN ({ph})) AND graph_id = ?",
                    current_frontier + current_frontier + [self._graph_id],
                ).fetchall()
                next_frontier = []
                for r in edge_rows:
                    e1, e2 = r["entity1_uuid"], r["entity2_uuid"]
                    neighbor = e2 if e1 in visited_uuids else e1
                    if neighbor not in visited_uuids:
                        visited_uuids.add(neighbor)
                        next_frontier.append(neighbor)
                current_frontier = next_frontier
                # Get entities for this frontier
                if current_frontier:
                    ph = ",".join("?" * len(current_frontier))
                    frontier_rows = conn.execute(
                        f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE uuid IN ({ph}) AND graph_id = ?",
                        current_frontier + [self._graph_id],
                    ).fetchall()
                    for r in frontier_rows:
                        entity = _row_to_entity(dict(r))
                        if entity.family_id not in visited_fids:
                            visited_fids.add(entity.family_id)
                            hop_map[entity.family_id] = depth
                            entities.append(entity)
                if len(visited_fids) >= max_nodes:
                    break
            # Fetch seed entities
            missing_seed_fids = [fid for fid in seed_fids if fid not in hop_map]
            if missing_seed_fids:
                seed_entities = self.get_entities_by_family_ids(missing_seed_fids)
                for fid, entity in seed_entities.items():
                    if fid not in visited_fids:
                        visited_fids.add(fid)
                        hop_map[fid] = 0
                        entities.insert(0, entity)
            # Get seed entities that were already in hop_map
            seed_entity_map = self.get_entities_by_family_ids(seed_fids)
            seed_entities_list = []
            for fid in seed_fids:
                ent = seed_entity_map.get(fid)
                if ent and ent not in entities:
                    seed_entities_list.append(ent)
            all_entities = seed_entities_list + entities
            # Get relations
            discovered_fids = list(hop_map.keys())
            relations = self.get_relations_by_family_ids(discovered_fids, limit=max_nodes * 3, time_point=time_point) if discovered_fids else []
            return all_entities, relations, hop_map
        finally:
            conn.rollback()

    def batch_get_entity_degrees(self, family_ids: List[str]) -> Dict[str, int]:
        if not family_ids:
            return {}
        conn = self._connect()
        try:
            degree_map = {}
            for fid in family_ids:
                rows = conn.execute(
                    "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?",
                    (fid, self._graph_id),
                ).fetchall()
                abs_ids = [r["uuid"] for r in rows]
                if abs_ids:
                    ph = ",".join("?" * len(abs_ids))
                    cnt = conn.execute(
                        f"SELECT COUNT(DISTINCT uuid) AS cnt FROM relation WHERE (entity1_absolute_id IN ({ph}) OR entity2_absolute_id IN ({ph})) AND graph_id = ?",
                        abs_ids + abs_ids + [self._graph_id],
                    ).fetchone()["cnt"]
                    degree_map[fid] = cnt
                else:
                    degree_map[fid] = 0
        finally:
            conn.rollback()
        for fid in family_ids:
            degree_map.setdefault(fid, 0)
        return degree_map

    def find_shortest_path_cypher(self, source_family_id: str, target_family_id: str, max_depth: int = 6) -> List[List[str]]:
        result = self.find_shortest_paths(source_family_id, target_family_id, max_depth=max_depth, max_paths=1)
        if result.get("paths"):
            return [[n.name for n in p["entities"]] for p in result["paths"]]
        return []

    def find_shortest_paths(self, source_family_id: str, target_family_id: str,
                             max_depth: int = 6, max_paths: int = 10) -> Dict[str, Any]:
        result_empty = {"source_entity": None, "target_entity": None, "path_length": -1, "total_shortest_paths": 0, "paths": []}
        _ents = self.get_entities_by_family_ids([source_family_id, target_family_id])
        source_entity = _ents.get(source_family_id)
        target_entity = _ents.get(target_family_id)
        if not source_entity or not target_entity:
            result_empty["source_entity"] = source_entity
            result_empty["target_entity"] = target_entity
            return result_empty
        if source_family_id == target_family_id:
            return {"source_entity": source_entity, "target_entity": target_entity, "path_length": 0,
                    "total_shortest_paths": 1, "paths": [{"entities": [source_entity], "relations": [], "length": 0}]}
        # BFS using relates_to table
        conn = self._connect()
        try:
            source_uuid = source_entity.absolute_id
            target_uuid = target_entity.absolute_id
            # BFS with path tracking
            queue = [(source_uuid, [source_uuid])]
            visited = {source_uuid}
            found_paths = []
            while queue and len(found_paths) < max_paths:
                next_queue = []
                for current_uuid, path in queue:
                    ph_placeholders = "?"  # single node lookup
                    edges = conn.execute(
                        "SELECT entity1_uuid, entity2_uuid FROM relates_to WHERE (entity1_uuid = ? OR entity2_uuid = ?) AND graph_id = ?",
                        (current_uuid, current_uuid, self._graph_id),
                    ).fetchall()
                    for r in edges:
                        neighbor = r["entity2_uuid"] if r["entity1_uuid"] == current_uuid else r["entity1_uuid"]
                        if neighbor == target_uuid:
                            found_paths.append(path + [neighbor])
                        elif neighbor not in visited and len(path) < max_depth:
                            visited.add(neighbor)
                            next_queue.append((neighbor, path + [neighbor]))
                if found_paths:
                    break
                queue = next_queue
        finally:
            conn.rollback()
        if not found_paths:
            return {"source_entity": source_entity, "target_entity": target_entity, "path_length": -1,
                    "total_shortest_paths": 0, "paths": []}
        # Build entity path objects
        all_uuids = set()
        for p in found_paths:
            all_uuids.update(p)
        uuid_to_entity = {}
        if all_uuids:
            ent_map = self.get_entities_by_absolute_ids(list(all_uuids))
            for e in ent_map:
                uuid_to_entity[e.absolute_id] = e
        paths_result = []
        for p in found_paths:
            path_entities = [uuid_to_entity[uid] for uid in p if uid in uuid_to_entity]
            paths_result.append({"entities": path_entities, "relations": [], "length": len(path_entities) - 1})
        path_length = paths_result[0]["length"] if paths_result else -1
        return {"source_entity": source_entity, "target_entity": target_entity, "path_length": path_length,
                "total_shortest_paths": len(paths_result), "paths": paths_result}

    def get_entity_neighbors(self, entity_uuid: str, depth: int = 1) -> Dict:
        conn = self._connect()
        try:
            center_row = conn.execute(
                "SELECT uuid, name, family_id FROM entity WHERE uuid = ? AND graph_id = ?",
                (entity_uuid, self._graph_id),
            ).fetchone()
            center_node = None
            if center_row:
                center_node = {"uuid": center_row["uuid"], "name": center_row["name"], "family_id": center_row["family_id"]}
            neighbors = {"entity": center_node, "nodes": [], "edges": []}
            # BFS via relates_to
            visited = {entity_uuid}
            current_frontier = [entity_uuid]
            for _ in range(depth):
                if not current_frontier:
                    break
                ph = ",".join("?" * len(current_frontier))
                edge_rows = conn.execute(
                    f"SELECT entity1_uuid, entity2_uuid, relation_uuid, fact FROM relates_to WHERE (entity1_uuid IN ({ph}) OR entity2_uuid IN ({ph})) AND graph_id = ? LIMIT 500",
                    current_frontier + current_frontier + [self._graph_id],
                ).fetchall()
                next_frontier = []
                seen_nodes = set()
                seen_edges = set()
                for r in edge_rows:
                    e1, e2 = r["entity1_uuid"], r["entity2_uuid"]
                    if (e1, e2) not in seen_edges:
                        seen_edges.add((e1, e2))
                        neighbors["edges"].append({"source_uuid": e1, "target_uuid": e2, "content": r["fact"] or "", "relation_uuid": r["relation_uuid"]})
                    for uid in (e1, e2):
                        if uid not in visited and uid not in seen_nodes:
                            seen_nodes.add(uid)
                            next_frontier.append(uid)
                if next_frontier:
                    ph2 = ",".join("?" * len(next_frontier))
                    node_rows = conn.execute(
                        f"SELECT uuid, name, family_id FROM entity WHERE uuid IN ({ph2}) AND graph_id = ?",
                        next_frontier + [self._graph_id],
                    ).fetchall()
                    for r in node_rows:
                        neighbors["nodes"].append({"uuid": r["uuid"], "name": r["name"], "family_id": r["family_id"]})
                        visited.add(r["uuid"])
                current_frontier = next_frontier
        finally:
            conn.rollback()
        return neighbors

    def merge_entity_families(self, target_family_id: str, source_family_ids: List[str], skip_name_check: bool = False) -> Dict[str, Any]:
        all_ids_to_resolve = [target_family_id] + [s for s in source_family_ids if s]
        resolved_map = self.resolve_family_ids(all_ids_to_resolve)
        target_family_id = resolved_map.get(target_family_id, target_family_id)
        if not target_family_id or not source_family_ids:
            return {"entities_updated": 0, "relations_updated": 0}
        if not skip_name_check:
            resolved_sources = {s: resolved_map.get(s, s) for s in source_family_ids if s}
            unique_fids = list(set([target_family_id] + list(resolved_sources.values())))
            fid_to_entity = {}
            try:
                fid_to_entity = self.get_entities_by_family_ids(unique_fids) or {}
            except Exception:
                pass
            target_entity = fid_to_entity.get(target_family_id) or self.get_entity_by_family_id(target_family_id)
            target_name = target_entity.name if target_entity else ""
            _target_chars = set(target_name) if target_name else set()
            rejected_ids = set()
            for source_id in source_family_ids:
                resolved_source = resolved_sources.get(source_id, source_id)
                if not resolved_source:
                    continue
                source_entity = fid_to_entity.get(resolved_source) or self.get_entity_by_family_id(resolved_source)
                if not source_entity:
                    continue
                source_name = source_entity.name
                if target_name and source_name:
                    _source_chars = set(source_name)
                    shared = len(_source_chars & _target_chars)
                    total = len(_source_chars | _target_chars)
                    overlap = shared / total if total > 0 else 0
                    if overlap < 0.2:
                        logger.warning("Rejecting merge: name difference too large — target=%s(%s) source=%s(%s) overlap=%.2f",
                                       target_name, target_family_id, source_name, resolved_source, overlap)
                        rejected_ids.add(resolved_source)
            if rejected_ids:
                source_family_ids = [s for s in source_family_ids if resolved_sources.get(s, s) not in rejected_ids]
        if not source_family_ids:
            return {"entities_updated": 0, "relations_updated": 0, "rejected": True}
        with self._write_lock:
            conn = self._connect()
            try:
                entities_updated = 0
                canonical_source_ids: List[str] = []
                now_iso = datetime.now().isoformat()
                resolved_sources_in_session = {s: resolved_map.get(s, s) for s in source_family_ids if s}
                for source_id in source_family_ids:
                    source_id = resolved_sources_in_session.get(source_id, source_id)
                    if not source_id or source_id == target_family_id or source_id in canonical_source_ids:
                        continue
                    canonical_source_ids.append(source_id)
                if canonical_source_ids:
                    for sid in canonical_source_ids:
                        cursor = conn.execute(
                            "UPDATE entity SET family_id = ? WHERE family_id = ? AND graph_id = ?",
                            (target_family_id, sid, self._graph_id),
                        )
                        entities_updated += cursor.rowcount
                    for sid in canonical_source_ids:
                        conn.execute(
                            "INSERT OR REPLACE INTO entity_redirect (source_id, target_id, updated_at) VALUES (?, ?, ?)",
                            (sid, target_family_id, now_iso),
                        )
                    conn.commit()
            finally:
                conn.rollback()
        self.invalidate_entity_remap_cache()
        self._invalidate_entity_cache_bulk()
        return {"entities_updated": entities_updated, "relations_updated": 0,
                "target_family_id": target_family_id, "merged_source_ids": canonical_source_ids}

    # ==================================================================
    # COMMUNITY (Louvain)
    # ==================================================================

    def _write_community_labels(self, assignment: Dict[str, int]):
        if not assignment:
            return
        items = list(assignment.items())
        batch_size = 5000
        conn = self._connect()
        try:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                conn.executemany(
                    "UPDATE entity SET community_id = ? WHERE uuid = ? AND graph_id = ?",
                    [(str(cid), uuid_val, self._graph_id) for uuid_val, cid in batch],
                )
            conn.commit()
        finally:
            conn.rollback()

    def clear_communities(self) -> int:
        conn = self._connect()
        try:
            cursor = conn.execute("UPDATE entity SET community_id = NULL WHERE community_id IS NOT NULL AND graph_id = ?", (self._graph_id,))
            conn.commit()
            count = cursor.rowcount
        finally:
            conn.rollback()
        self._cache.invalidate_keys(["communities"])
        return count

    def count_communities(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT community_id) AS cnt FROM entity WHERE community_id IS NOT NULL AND graph_id = ?",
                (self._graph_id,),
            ).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def detect_communities(self, algorithm: str = 'louvain', resolution: float = 1.0) -> Dict:
        if nx is None:
            return {"error": "networkx not installed", "communities": 0}
        t0 = time.time()
        conn = self._connect()
        try:
            entity_rows = conn.execute("SELECT uuid, family_id, name FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchall()
            entity_map = {r["uuid"]: r["family_id"] for r in entity_rows}
            edge_rows = conn.execute(
                "SELECT entity1_uuid AS src, entity2_uuid AS tgt FROM relates_to WHERE graph_id = ?",
                (self._graph_id,),
            ).fetchall()
        finally:
            conn.rollback()
        G = nx.Graph()
        for uuid_val in entity_map:
            G.add_node(uuid_val)
        for r in edge_rows:
            if r["src"] in G and r["tgt"] in G:
                G.add_edge(r["src"], r["tgt"])
        communities = louvain_communities(G, resolution=resolution, seed=42)
        assignment = {}
        for cid, community_set in enumerate(communities):
            for uuid_val in community_set:
                assignment[uuid_val] = cid
        self._write_community_labels(assignment)
        elapsed = time.time() - t0
        community_sizes = [len(c) for c in communities]
        return {"total_communities": len(communities), "community_sizes": sorted(community_sizes, reverse=True), "elapsed_seconds": round(elapsed, 3)}

    def get_communities(self, limit: int = 50, min_size: int = 3, offset: int = 0) -> Tuple[List[Dict], int]:
        conn = self._connect()
        try:
            # Get communities with size >= min_size
            rows = conn.execute(
                "SELECT community_id, COUNT(*) AS size FROM entity "
                "WHERE community_id IS NOT NULL AND graph_id = ? GROUP BY community_id HAVING COUNT(*) >= ? "
                "ORDER BY size DESC LIMIT ? OFFSET ?",
                (self._graph_id, min_size, limit, offset),
            ).fetchall()
            total_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM (SELECT community_id FROM entity "
                "WHERE community_id IS NOT NULL AND graph_id = ? GROUP BY community_id HAVING COUNT(*) >= ?)",
                (self._graph_id, min_size),
            ).fetchone()
            total = total_row["cnt"] if total_row else 0
        finally:
            conn.rollback()
        communities = []
        for r in rows:
            cid = r["community_id"]
            members = conn.execute(
                "SELECT uuid, family_id, name FROM entity WHERE community_id = ? AND graph_id = ?",
                (cid, self._graph_id),
            ).fetchall()
            communities.append({
                "community_id": cid,
                "size": r["size"],
                "members": [{"uuid": m["uuid"], "family_id": m["family_id"], "name": m["name"]} for m in members],
            })
        return communities, total

    def get_community(self, cid: int) -> Optional[Dict]:
        cache_key = f"community:{cid}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        conn = self._connect()
        try:
            cid_str = str(cid)
            member_rows = conn.execute(
                "SELECT uuid, family_id, name, content FROM entity WHERE community_id = ? AND graph_id = ? ORDER BY name LIMIT 500",
                (cid_str, self._graph_id),
            ).fetchall()
            if not member_rows:
                return None
            members = [{"uuid": r["uuid"], "family_id": r["family_id"], "name": r["name"], "content": r["content"] or ""} for r in member_rows]
            # Get edges within community
            member_uuids = [r["uuid"] for r in member_rows]
            ph = ",".join("?" * len(member_uuids))
            edge_rows = conn.execute(
                f"SELECT r.entity1_uuid, r.entity2_uuid, r.fact, r.relation_uuid "
                f"FROM relates_to r WHERE r.graph_id = ? "
                f"AND r.entity1_uuid IN ({ph}) AND r.entity2_uuid IN ({ph})",
                [self._graph_id] + member_uuids + member_uuids,
            ).fetchall()
            uuid_to_name = {r["uuid"]: r["name"] for r in member_rows}
            relations = []
            seen_edges = set()
            for r in edge_rows:
                e_key = (r["entity1_uuid"], r["entity2_uuid"])
                if e_key not in seen_edges:
                    seen_edges.add(e_key)
                    relations.append({
                        "source_uuid": r["entity1_uuid"], "source_name": uuid_to_name.get(r["entity1_uuid"], ""),
                        "target_uuid": r["entity2_uuid"], "target_name": uuid_to_name.get(r["entity2_uuid"], ""),
                        "content": r["fact"] or "", "relation_uuid": r["relation_uuid"],
                    })
        finally:
            conn.rollback()
        result = {"community_id": cid, "size": len(members), "members": members, "relations": relations}
        self._cache.set(cache_key, result, ttl=120)
        return result

    def get_community_graph(self, cid: int) -> Dict:
        cache_key = f"community_graph:{cid}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        conn = self._connect()
        try:
            cid_str = str(cid)
            member_rows = conn.execute(
                "SELECT uuid, family_id, name FROM entity WHERE community_id = ? AND graph_id = ? LIMIT 300",
                (cid_str, self._graph_id),
            ).fetchall()
            member_uuids = [r["uuid"] for r in member_rows]
            if not member_uuids:
                return {"nodes": [], "edges": []}
            ph = ",".join("?" * len(member_uuids))
            edge_rows = conn.execute(
                f"SELECT entity1_uuid, entity2_uuid, fact FROM relates_to WHERE graph_id = ? "
                f"AND entity1_uuid IN ({ph}) AND entity2_uuid IN ({ph})",
                [self._graph_id] + member_uuids + member_uuids,
            ).fetchall()
            nodes = [{"uuid": r["uuid"], "family_id": r["family_id"], "name": r["name"]} for r in member_rows]
            seen_edges = set()
            edges = []
            for r in edge_rows:
                e_key = (r["entity1_uuid"], r["entity2_uuid"])
                if e_key not in seen_edges:
                    seen_edges.add(e_key)
                    edges.append({"source_uuid": r["entity1_uuid"], "target_uuid": r["entity2_uuid"], "content": r["fact"] or ""})
        finally:
            conn.rollback()
        result = {"nodes": nodes, "edges": edges}
        self._cache.set(cache_key, result, ttl=120)
        return result

    # ==================================================================
    # DREAM (Seeds, Corroboration, Logs)
    # ==================================================================

    def _dream_seeds_random(self, count, exclude_uuids, community_id):
        conn = self._connect()
        try:
            query = (
                "SELECT uuid, family_id, name, content, confidence, event_time, community_id FROM entity "
                "WHERE graph_id = ? "
            )
            params: list = [self._graph_id]
            if exclude_uuids:
                ph = ",".join("?" * len(exclude_uuids))
                query += f"AND uuid NOT IN ({ph}) "
                params.extend(exclude_uuids)
            if community_id is not None:
                query += "AND community_id = ? "
                params.append(str(community_id))
            query += "ORDER BY RANDOM() LIMIT ?"
            params.append(count)
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [dict(r) for r in rows]

    def _dream_seeds_orphan(self, count, exclude_uuids, community_id):
        conn = self._connect()
        try:
            query = (
                "SELECT e.uuid, e.family_id, e.name, e.content, e.confidence, e.event_time, e.community_id "
                "FROM entity e WHERE e.graph_id = ? "
                "AND e.uuid NOT IN (SELECT entity1_uuid FROM relates_to WHERE graph_id = ? UNION SELECT entity2_uuid FROM relates_to WHERE graph_id = ?) "
            )
            params: list = [self._graph_id, self._graph_id, self._graph_id]
            if exclude_uuids:
                ph = ",".join("?" * len(exclude_uuids))
                query += f"AND e.uuid NOT IN ({ph}) "
                params.extend(exclude_uuids)
            if community_id is not None:
                query += "AND e.community_id = ? "
                params.append(str(community_id))
            query += "LIMIT ?"
            params.append(count)
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [dict(r) for r in rows]

    def _dream_seeds_hub(self, count, exclude_uuids, community_id):
        conn = self._connect()
        try:
            query = (
                "SELECT e.uuid, e.family_id, e.name, e.content, e.confidence, e.event_time, e.community_id, "
                "COUNT(DISTINCT rt.entity2_uuid) AS degree "
                "FROM entity e "
                "INNER JOIN relates_to rt ON rt.entity1_uuid = e.uuid AND rt.graph_id = ? "
                "WHERE e.graph_id = ? "
            )
            params: list = [self._graph_id, self._graph_id]
            if exclude_uuids:
                ph = ",".join("?" * len(exclude_uuids))
                query += f"AND e.uuid NOT IN ({ph}) "
                params.extend(exclude_uuids)
            if community_id is not None:
                query += "AND e.community_id = ? "
                params.append(str(community_id))
            query += "GROUP BY e.uuid ORDER BY degree DESC LIMIT ?"
            params.append(count)
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [dict(r) for r in rows]

    def _dream_seeds_time_gap(self, count, exclude_uuids, community_id):
        conn = self._connect()
        try:
            query = (
                "SELECT uuid, family_id, name, content, confidence, event_time, community_id "
                "FROM entity WHERE graph_id = ? "
                "AND processed_time IS NOT NULL AND julianday('now') - julianday(processed_time) > 30 "
            )
            params: list = [self._graph_id]
            if exclude_uuids:
                ph = ",".join("?" * len(exclude_uuids))
                query += f"AND uuid NOT IN ({ph}) "
                params.extend(exclude_uuids)
            if community_id is not None:
                query += "AND community_id = ? "
                params.append(str(community_id))
            query += "ORDER BY processed_time ASC LIMIT ?"
            params.append(count)
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [dict(r) for r in rows]

    def _dream_seeds_low_confidence(self, count, exclude_uuids, community_id):
        conn = self._connect()
        try:
            query = (
                "SELECT uuid, family_id, name, content, confidence, event_time, community_id "
                "FROM entity WHERE graph_id = ? "
                "AND confidence IS NOT NULL AND confidence < 0.5 "
            )
            params: list = [self._graph_id]
            if exclude_uuids:
                ph = ",".join("?" * len(exclude_uuids))
                query += f"AND uuid NOT IN ({ph}) "
                params.extend(exclude_uuids)
            if community_id is not None:
                query += "AND community_id = ? "
                params.append(str(community_id))
            query += "ORDER BY confidence ASC LIMIT ?"
            params.append(count)
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [dict(r) for r in rows]

    def _dream_seeds_cross_community(self, count, exclude_uuids, community_id):
        communities, _ = self.get_communities(limit=10, min_size=2)
        if len(communities) < 2:
            return self._dream_seeds_random(count, exclude_uuids, community_id)
        pairs = []
        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):
                if len(pairs) >= count:
                    break
                c1_members = communities[i]["members"]
                c2_members = communities[j]["members"]
                c1_valid = [m for m in c1_members if m["uuid"] not in exclude_uuids]
                c2_valid = [m for m in c2_members if m["uuid"] not in exclude_uuids]
                if c1_valid and c2_valid:
                    e1 = random.choice(c1_valid)
                    e2 = random.choice(c2_valid)
                    pairs.extend([e1, e2])
            if len(pairs) >= count * 2:
                break
        return pairs[:count * 2]

    def get_dream_seeds(self, strategy: str = "random", count: int = 10,
                        exclude_ids: Optional[List[str]] = None,
                        community_id: Optional[int] = None) -> List[Dict[str, Any]]:
        exclude_uuids = set()
        if exclude_ids:
            resolved_map = self.resolve_family_ids(exclude_ids)
            canonical_fids = list({v for v in resolved_map.values() if v})
            if canonical_fids:
                aids_map = self.get_latest_absolute_ids_by_family_ids(canonical_fids)
                exclude_uuids = set(aids_map.values())
        strategies = {
            "random": self._dream_seeds_random, "orphan": self._dream_seeds_orphan,
            "hub": self._dream_seeds_hub, "time_gap": self._dream_seeds_time_gap,
            "low_confidence": self._dream_seeds_low_confidence, "cross_community": self._dream_seeds_cross_community,
        }
        handler = strategies.get(strategy)
        if not handler:
            raise ValueError(f"Unknown seed strategy: {strategy}")
        seeds = handler(count, exclude_uuids, community_id)
        reason_map = {"random": "Random selection", "orphan": "Orphan entity: no connections",
                      "hub": "High connectivity hub", "time_gap": "Long time without updates",
                      "low_confidence": "Low confidence entity", "cross_community": "Cross-community bridge candidate"}
        for s in seeds:
            s["reason"] = reason_map.get(strategy, "")
        return seeds

    def corroborate_dream_relation(self, entity1_family_id: str, entity2_family_id: str,
                                    corroboration_source: str = "remember") -> Optional[Dict[str, Any]]:
        rels = self.get_relations_by_entities(entity1_family_id, entity2_family_id, include_candidates=True)
        if not rels:
            return None
        for rel in rels:
            try:
                attrs = json.loads(rel.attributes) if rel.attributes else {}
            except (ValueError, TypeError):
                attrs = {}
            if (attrs.get("tier") == "candidate" and attrs.get("status") == "hypothesized"
                    and rel.source_document and rel.source_document.startswith("dream")):
                count = attrs.get("corroboration_count", 0) + 1
                attrs["corroboration_count"] = count
                attrs.setdefault("corroboration_sources", []).append(corroboration_source)
                now = datetime.now()
                record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                new_conf = min((rel.confidence or 0.5) + 0.1, 0.69)
                updated = Relation(
                    absolute_id=record_id, family_id=rel.family_id,
                    entity1_absolute_id=rel.entity1_absolute_id, entity2_absolute_id=rel.entity2_absolute_id,
                    content=rel.content, event_time=now, processed_time=now,
                    episode_id=rel.episode_id, source_document=rel.source_document,
                    confidence=new_conf, attributes=json.dumps(attrs),
                )
                self.save_relation(updated)
                if count >= 2:
                    return self.promote_candidate_relation(rel.family_id, evidence_source=f"auto:{corroboration_source}")
                return {"family_id": rel.family_id, "corroboration_count": count, "status": "hypothesized",
                        "confidence": new_conf, "message": f"Corroboration count: {count}/2"}
        return None

    def corroborate_dream_relations_batch(self, entity_pairs: List[tuple], corroboration_source: str = "remember") -> List[Dict[str, Any]]:
        if not entity_pairs:
            return []
        results = []
        for e1_fid, e2_fid in entity_pairs:
            result = self.corroborate_dream_relation(e1_fid, e2_fid, corroboration_source)
            if result:
                results.append(result)
        return results

    def count_candidate_relations(self, status: str = None) -> int:
        conn = self._connect()
        try:
            query = "SELECT COUNT(DISTINCT family_id) AS cnt FROM relation WHERE source_document LIKE 'dream%' AND graph_id = ?"
            params: list = [self._graph_id]
            if status:
                query += " AND attributes LIKE ?"
                params.append(f'%"status":"{status}"%')
            row = conn.execute(query, params).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def get_candidate_relations(self, limit: int = 50, offset: int = 0, status: str = None) -> list:
        conn = self._connect()
        try:
            query = (
                f"SELECT r.* FROM relation r "
                f"INNER JOIN ("
                f"  SELECT family_id, MAX(processed_time) AS max_pt FROM relation "
                f"  WHERE graph_id = ? AND source_document LIKE 'dream%' "
                f"  GROUP BY family_id"
                f") latest ON r.family_id = latest.family_id AND r.processed_time = latest.max_pt "
                f"WHERE r.graph_id = ? "
            )
            params: list = [self._graph_id, self._graph_id]
            if status:
                query += "AND r.attributes LIKE ? "
                params.append(f'%"status":"{status}"%')
            query += f"ORDER BY r.processed_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [_row_to_relation(dict(r)) for r in rows]

    def promote_candidate_relation(self, family_id: str, evidence_source: str = "manual", new_confidence: float = None) -> Dict[str, Any]:
        resolved = self.resolve_family_id(family_id)
        if not resolved:
            raise ValueError(f"Relation not found: {family_id}")
        rel = self.get_relation_by_family_id(resolved)
        if not rel:
            raise ValueError(f"Relation not found: {family_id}")
        try:
            attrs = json.loads(rel.attributes) if rel.attributes else {}
        except (ValueError, TypeError):
            attrs = {}
        old_status = attrs.get("status", "unknown")
        old_tier = attrs.get("tier", "unknown")
        attrs["tier"] = "verified"
        attrs["status"] = "verified"
        attrs["promoted_by"] = evidence_source
        now = datetime.now()
        attrs["promoted_at"] = now.isoformat()
        attrs["corroboration_count"] = attrs.get("corroboration_count", 0) + 1
        record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        new_conf = new_confidence if new_confidence is not None else max(rel.confidence or 0.5, 0.7)
        relation = Relation(
            absolute_id=record_id, family_id=rel.family_id,
            entity1_absolute_id=rel.entity1_absolute_id, entity2_absolute_id=rel.entity2_absolute_id,
            content=rel.content, event_time=now, processed_time=now,
            episode_id=rel.episode_id, source_document=rel.source_document,
            confidence=new_conf, attributes=json.dumps(attrs),
        )
        self.save_relation(relation)
        return {"family_id": resolved, "old_status": old_status, "old_tier": old_tier,
                "new_status": "verified", "new_tier": "verified", "confidence": new_conf}

    def promote_candidate_relations_batch(self, family_ids: List[str], evidence_source: str = "manual", new_confidence: float = None) -> List[Dict[str, Any]]:
        if not family_ids:
            return []
        results = []
        for fid in family_ids:
            try:
                result = self.promote_candidate_relation(fid, evidence_source, new_confidence)
                results.append(result)
            except Exception:
                pass
        return results

    def demote_candidate_relation(self, family_id: str, reason: str = "") -> Dict[str, Any]:
        resolved = self.resolve_family_id(family_id)
        if not resolved:
            raise ValueError(f"Relation not found: {family_id}")
        rel = self.get_relation_by_family_id(resolved)
        if not rel:
            raise ValueError(f"Relation not found: {family_id}")
        try:
            attrs = json.loads(rel.attributes) if rel.attributes else {}
        except (ValueError, TypeError):
            attrs = {}
        old_status = attrs.get("status", "unknown")
        attrs["status"] = "rejected"
        attrs["rejected_reason"] = reason
        now = datetime.now()
        attrs["rejected_at"] = now.isoformat()
        record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        relation = Relation(
            absolute_id=record_id, family_id=rel.family_id,
            entity1_absolute_id=rel.entity1_absolute_id, entity2_absolute_id=rel.entity2_absolute_id,
            content=rel.content, event_time=now, processed_time=now,
            episode_id=rel.episode_id, source_document=rel.source_document,
            confidence=min(rel.confidence or 0.3, 0.2), attributes=json.dumps(attrs),
        )
        self.save_relation(relation)
        return {"family_id": resolved, "old_status": old_status, "new_status": "rejected", "confidence": relation.confidence}

    def save_dream_log(self, report):
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO dream_log (cycle_id, graph_id, start_time, end_time, status, narrative, "
                "insights, connections, consolidations, strategy, entities_examined, relations_created, episode_ids) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    report.cycle_id, report.graph_id,
                    _fmt_dt(report.start_time), _fmt_dt(report.end_time or datetime.now()),
                    report.status, report.narrative,
                    json.dumps(report.insights, ensure_ascii=False),
                    json.dumps(getattr(report, 'new_connections', []), ensure_ascii=False),
                    json.dumps(report.consolidations, ensure_ascii=False),
                    getattr(report, 'strategy', ''),
                    getattr(report, 'entities_examined', 0),
                    getattr(report, 'relations_created', 0),
                    json.dumps(getattr(report, 'episode_ids', []), ensure_ascii=False),
                ),
            )
            conn.commit()
        finally:
            conn.rollback()

    def get_dream_log(self, cycle_id: str) -> Optional[dict]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM dream_log WHERE cycle_id = ? AND graph_id = ?",
                (cycle_id, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        if not row:
            return None
        return self._parse_dream_log_record(dict(row))

    @staticmethod
    def _parse_dream_log_record(r) -> dict:
        d = dict(r) if not isinstance(r, dict) else r
        _loads = json.loads
        _raw_ins = d.get("insights")
        _raw_con = d.get("connections")
        _raw_cns = d.get("consolidations")
        _raw_epi = d.get("episode_ids")
        return {
            "cycle_id": d["cycle_id"], "graph_id": d["graph_id"],
            "start_time": str(d.get("start_time", "")), "end_time": str(d.get("end_time", "")),
            "status": d.get("status", ""), "narrative": d.get("narrative", ""),
            "insights": () if not _raw_ins or _raw_ins == "[]" else _loads(_raw_ins),
            "connections": () if not _raw_con or _raw_con == "[]" else _loads(_raw_con),
            "consolidations": () if not _raw_cns or _raw_cns == "[]" else _loads(_raw_cns),
            "strategy": d.get("strategy", ""),
            "entities_examined": d.get("entities_examined", 0),
            "relations_created": d.get("relations_created", 0),
            "episode_ids": () if not _raw_epi or _raw_epi == "[]" else _loads(_raw_epi),
        }

    def list_dream_logs(self, graph_id: str = None, limit: int = 20) -> List[dict]:
        gid = graph_id or self._graph_id
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM dream_log WHERE graph_id = ? ORDER BY start_time DESC LIMIT ?",
                (gid, limit),
            ).fetchall()
        finally:
            conn.rollback()
        return [self._parse_dream_log_record(dict(r)) for r in rows]

    def save_dream_episode(self, content: str, entities_examined: Optional[List[str]] = None,
                           relations_created: Optional[List[Dict]] = None, strategy_used: str = "",
                           dream_cycle_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        now = datetime.now()
        episode_id = f"episode_dream_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        if not dream_cycle_id:
            dream_cycle_id = f"dream_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        _explicit_rel_count = kwargs.get("relations_created_count")
        _explicit_ent_count = kwargs.get("entities_examined_count")
        ent_count = _explicit_ent_count if _explicit_ent_count is not None else (len(entities_examined) if entities_examined else 0)
        rel_count = _explicit_rel_count if _explicit_rel_count is not None else (len(relations_created) if relations_created else 0)
        structured = {"narrative": content, "strategy": strategy_used,
                      "entities_examined_count": ent_count, "relations_created_count": rel_count}
        if relations_created:
            structured["relations_created"] = relations_created
        full_content = content
        if rel_count > 0 or ent_count > 0:
            full_content += "\n\n---\n" + json.dumps(structured, ensure_ascii=False, indent=2)
        source_doc = f"dream:{dream_cycle_id}" if dream_cycle_id else "dream"
        cache = Episode(
            absolute_id=episode_id, content=full_content, event_time=now,
            source_document=source_doc, episode_type="dream",
        )
        self.save_episode(cache)
        if entities_examined:
            abs_ids = []
            try:
                resolved_map = self.resolve_family_ids(entities_examined)
                canonical_fids = list({r for r in resolved_map.values() if r})
                if canonical_fids:
                    entities_map = self.get_entities_by_family_ids(canonical_fids)
                    for eid in entities_examined:
                        resolved = resolved_map.get(eid)
                        if resolved:
                            entity = entities_map.get(resolved)
                            if entity and entity.absolute_id:
                                abs_ids.append(entity.absolute_id)
            except Exception:
                for eid in entities_examined:
                    resolved = self.resolve_family_id(eid)
                    if resolved:
                        entity = self.get_entity_by_family_id(resolved)
                        if entity:
                            abs_ids.append(entity.absolute_id)
            if abs_ids:
                self.save_episode_mentions(episode_id, abs_ids, context=f"dream:{strategy_used}")
        report = SimpleNamespace(
            cycle_id=dream_cycle_id, graph_id=self._graph_id,
            start_time=now, end_time=now, status="completed",
            narrative=content[:2000], insights=[], new_connections=relations_created or [],
            consolidations=[], strategy=strategy_used,
            entities_examined=ent_count, relations_created=rel_count,
            episode_ids=[episode_id],
        )
        self.save_dream_log(report)
        return {"episode_id": episode_id, "episode_type": "dream", "cycle_id": dream_cycle_id}

    # ==================================================================
    # CONCEPTS (Unified concept model)
    # ==================================================================

    def count_concepts(self, role: str = None, time_point: str = None) -> int:
        tp = self._tp_to_datetime(time_point)
        tp_iso = tp.isoformat() if tp else None
        conn = self._connect()
        try:
            total = 0
            if role is None or role == "entity":
                query = "SELECT COUNT(DISTINCT family_id) AS cnt FROM entity WHERE graph_id = ?"
                params = [self._graph_id]
                if tp_iso:
                    query += " AND (valid_at IS NULL OR valid_at <= ?)"
                    params.append(tp_iso)
                total += conn.execute(query, params).fetchone()["cnt"]
            if role is None or role == "relation":
                query = "SELECT COUNT(DISTINCT family_id) AS cnt FROM relation WHERE graph_id = ?"
                params = [self._graph_id]
                if tp_iso:
                    query += " AND (valid_at IS NULL OR valid_at <= ?)"
                    params.append(tp_iso)
                total += conn.execute(query, params).fetchone()["cnt"]
            if role is None or role == "observation":
                query = "SELECT COUNT(*) AS cnt FROM episode WHERE graph_id = ?"
                params = [self._graph_id]
                total += conn.execute(query, params).fetchone()["cnt"]
        finally:
            conn.rollback()
        return total

    def get_concept_by_family_id(self, family_id: str, time_point: str = None) -> Optional[dict]:
        tp = self._tp_to_datetime(time_point)
        tp_iso = tp.isoformat() if tp else None
        # Try entity first
        conn = self._connect()
        try:
            query = f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE family_id = ? AND graph_id = ?"
            params = [family_id, self._graph_id]
            if tp_iso:
                query += " AND (valid_at IS NULL OR valid_at <= ?)"
                params.append(tp_iso)
            query += " ORDER BY version_seq DESC LIMIT 1"
            row = conn.execute(query, params).fetchone()
            if row:
                entity = _row_to_entity(dict(row))
                return {"id": entity.absolute_id, "family_id": entity.family_id, "role": "entity",
                        "name": entity.name, "content": entity.content,
                        "event_time": _fmt_dt(entity.event_time), "processed_time": _fmt_dt(entity.processed_time),
                        "source_document": entity.source_document, "summary": entity.summary, "confidence": entity.confidence}
            # Try relation
            query = f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE family_id = ? AND graph_id = ?"
            params = [family_id, self._graph_id]
            if tp_iso:
                query += " AND (valid_at IS NULL OR valid_at <= ?)"
                params.append(tp_iso)
            query += " ORDER BY version_seq DESC LIMIT 1"
            row = conn.execute(query, params).fetchone()
            if row:
                rel = _row_to_relation(dict(row))
                return {"id": rel.absolute_id, "family_id": rel.family_id, "role": "relation",
                        "name": "", "content": rel.content,
                        "event_time": _fmt_dt(rel.event_time), "processed_time": _fmt_dt(rel.processed_time),
                        "source_document": rel.source_document, "summary": rel.summary, "confidence": rel.confidence}
            # Try episode (by uuid)
            row = conn.execute("SELECT * FROM episode WHERE uuid = ? AND graph_id = ?", (family_id, self._graph_id)).fetchone()
            if row:
                rd = dict(row)
                return {"id": rd["uuid"], "family_id": rd["uuid"], "role": "observation",
                        "name": "", "content": rd["content"] or "",
                        "event_time": _fmt_dt(rd.get("event_time")), "processed_time": _fmt_dt(rd.get("processed_time")),
                        "source_document": rd.get("source_document", ""), "summary": None, "confidence": None}
        finally:
            conn.rollback()
        return None

    def get_concept_neighbors(self, family_id: str, max_depth: int = 1, time_point: str = None) -> List[dict]:
        concept = self.get_concept_by_family_id(family_id, time_point=time_point)
        if not concept:
            return []
        abs_id = concept.get("id")
        role = concept.get("role")
        if not abs_id or not role:
            return []
        neighbors = []
        conn = self._connect()
        try:
            if role == 'entity':
                # RELATES_TO neighbors
                edge_rows = conn.execute(
                    "SELECT entity1_uuid, entity2_uuid FROM relates_to WHERE (entity1_uuid = ? OR entity2_uuid = ?) AND graph_id = ?",
                    (abs_id, abs_id, self._graph_id),
                ).fetchall()
                neighbor_uuids = set()
                for r in edge_rows:
                    n = r["entity2_uuid"] if r["entity1_uuid"] == abs_id else r["entity1_uuid"]
                    neighbor_uuids.add(n)
                if neighbor_uuids:
                    ph = ",".join("?" * len(neighbor_uuids))
                    ent_rows = conn.execute(
                        f"SELECT DISTINCT family_id, uuid AS id, name, 'entity' AS role, content FROM entity WHERE uuid IN ({ph}) AND graph_id = ?",
                        list(neighbor_uuids) + [self._graph_id],
                    ).fetchall()
                    neighbors.extend([dict(r) for r in ent_rows])
                # Relations referencing this entity
                rel_rows = conn.execute(
                    "SELECT DISTINCT family_id, uuid AS id, '' AS name, 'relation' AS role, content FROM relation WHERE (entity1_absolute_id = ? OR entity2_absolute_id = ?) AND graph_id = ?",
                    (abs_id, abs_id, self._graph_id),
                ).fetchall()
                neighbors.extend([dict(r) for r in rel_rows])
            elif role == 'relation':
                # Endpoint entities
                rel = self.get_relation_by_absolute_id(abs_id)
                if rel:
                    eids = [rel.entity1_absolute_id, rel.entity2_absolute_id]
                    ph = ",".join("?" * len(eids))
                    ent_rows = conn.execute(
                        f"SELECT DISTINCT family_id, uuid AS id, name, 'entity' AS role, content FROM entity WHERE uuid IN ({ph}) AND graph_id = ?",
                        eids + [self._graph_id],
                    ).fetchall()
                    neighbors.extend([dict(r) for r in ent_rows])
                # Episodes mentioning this relation
                mention_rows = conn.execute(
                    "SELECT DISTINCT e.uuid AS id, e.content AS name, 'observation' AS role, e.content FROM episode e "
                    "INNER JOIN mentions m ON m.episode_uuid = e.uuid WHERE m.target_uuid = ? AND e.graph_id = ?",
                    (abs_id, self._graph_id),
                ).fetchall()
                neighbors.extend([dict(r) for r in mention_rows])
            elif role == 'observation':
                mention_rows = conn.execute(
                    "SELECT m.target_uuid, m.target_type, m.target_family_id FROM mentions m "
                    "WHERE m.episode_uuid = ? AND m.graph_id = ?",
                    (abs_id, self._graph_id),
                ).fetchall()
                for r in mention_rows:
                    rd = dict(r)
                    neighbors.append({"id": r["target_uuid"], "family_id": rd.get("target_family_id", ""),
                                      "role": r["target_type"] if r["target_type"] != "entity" else "entity"})
        finally:
            conn.rollback()
        seen = set()
        deduped = []
        for n in neighbors:
            fid = n.get('family_id', '')
            if fid and fid not in seen:
                seen.add(fid)
                deduped.append(n)
        return deduped

    def get_concept_provenance(self, family_id: str, time_point: str = None) -> List[dict]:
        concept = self.get_concept_by_family_id(family_id, time_point=time_point)
        if not concept:
            return []
        role = concept.get("role")
        abs_id = concept.get("id")
        if not abs_id:
            return []
        if role == 'entity':
            return self.get_entity_provenance(family_id)
        elif role == 'relation':
            conn = self._connect()
            try:
                abs_rows = conn.execute("SELECT uuid FROM relation WHERE family_id = ? AND graph_id = ?", (family_id, self._graph_id)).fetchall()
                abs_ids = [r["uuid"] for r in abs_rows]
                if not abs_ids:
                    return []
                ph = ",".join("?" * len(abs_ids))
                rows = conn.execute(
                    f"SELECT DISTINCT m.episode_uuid AS episode_id, m.context, e.content, e.source_document "
                    f"FROM mentions m INNER JOIN episode e ON e.uuid = m.episode_uuid "
                    f"WHERE m.target_uuid IN ({ph}) AND m.graph_id = ?",
                    abs_ids + [self._graph_id],
                ).fetchall()
            finally:
                conn.rollback()
            return [{"episode_id": r["episode_id"], "context": dict(r).get("context", ""),
                      "content": dict(r).get("content", ""), "source_document": dict(r).get("source_document", "")} for r in rows]
        elif role == 'observation':
            return [{"episode_id": abs_id, "role": "observation", "content": concept.get("content", "")}]
        return []

    def get_concept_mentions(self, family_id: str, time_point: str = None) -> List[dict]:
        return self.get_concept_provenance(family_id, time_point=time_point)

    def get_episode_concepts(self, episode_id: str) -> List[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT target_uuid, target_type FROM mentions WHERE episode_uuid = ? AND graph_id = ?",
                (episode_id, self._graph_id),
            ).fetchall()
            results = []
            for r in rows:
                target_uuid = r["target_uuid"]
                target_type = r["target_type"]
                if target_type == "entity":
                    ent = conn.execute(
                        "SELECT family_id, uuid AS id, 'entity' AS role, name, content FROM entity WHERE uuid = ? AND graph_id = ?",
                        (target_uuid, self._graph_id),
                    ).fetchone()
                    if ent:
                        results.append(dict(ent))
                elif target_type == "relation":
                    rel = conn.execute(
                        "SELECT family_id, uuid AS id, 'relation' AS role, '' AS name, content FROM relation WHERE uuid = ? AND graph_id = ?",
                        (target_uuid, self._graph_id),
                    ).fetchone()
                    if rel:
                        results.append(dict(rel))
        finally:
            conn.rollback()
        return results

    def list_concepts(self, role: str = None, limit: int = 50, offset: int = 0, time_point: str = None) -> List[dict]:
        results = []
        tp = self._tp_to_datetime(time_point)
        tp_iso = tp.isoformat() if tp else None
        conn = self._connect()
        try:
            if role is None or role == "entity":
                query = f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE graph_id = ?"
                params = [self._graph_id]
                if tp_iso:
                    query += " AND (valid_at IS NULL OR valid_at <= ?)"
                    params.append(tp_iso)
                query += " ORDER BY processed_time DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                rows = conn.execute(query, params).fetchall()
                for r in rows:
                    ent = _row_to_entity(dict(r))
                    results.append({"id": ent.absolute_id, "family_id": ent.family_id, "role": "entity",
                                    "name": ent.name, "content": ent.content,
                                    "event_time": _fmt_dt(ent.event_time), "processed_time": _fmt_dt(ent.processed_time)})
            if role is None or role == "relation":
                query = f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE graph_id = ?"
                params = [self._graph_id]
                if tp_iso:
                    query += " AND (valid_at IS NULL OR valid_at <= ?)"
                    params.append(tp_iso)
                query += " ORDER BY processed_time DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                rows = conn.execute(query, params).fetchall()
                for r in rows:
                    rel = _row_to_relation(dict(r))
                    results.append({"id": rel.absolute_id, "family_id": rel.family_id, "role": "relation",
                                    "name": "", "content": rel.content,
                                    "event_time": _fmt_dt(rel.event_time), "processed_time": _fmt_dt(rel.processed_time)})
            if role is None or role == "observation":
                rows = conn.execute(
                    "SELECT uuid, content, event_time, processed_time FROM episode WHERE graph_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (self._graph_id, limit, offset),
                ).fetchall()
                for r in rows:
                    results.append({"id": r["uuid"], "family_id": r["uuid"], "role": "observation",
                                    "name": "", "content": r["content"] or "",
                                    "event_time": _fmt_dt(r["event_time"]), "processed_time": _fmt_dt(r["processed_time"])})
        finally:
            conn.rollback()
        return results[:limit]

    def traverse_concepts(self, start_family_ids: List[str], max_depth: int = 2, time_point: str = None) -> dict:
        if not start_family_ids:
            return {"concepts": {}, "edges": [], "relations": [], "visited_count": 0}
        visited = set()
        queue = list(start_family_ids)
        all_concepts = {}
        all_edges = []
        for _ in range(max_depth):
            frontier = [fid for fid in queue if fid not in visited]
            if not frontier:
                break
            visited.update(frontier)
            for fid in frontier:
                concept = self.get_concept_by_family_id(fid, time_point=time_point)
                if concept:
                    all_concepts[fid] = concept
                    neighbors = self.get_concept_neighbors(fid, max_depth=1, time_point=time_point)
                    for n in neighbors:
                        nfid = n.get("family_id", "")
                        if nfid:
                            all_edges.append({"from": fid, "to": nfid, "to_role": n.get("role", ""), "to_name": n.get("name", "")})
                            if nfid not in visited:
                                queue.append(nfid)
            queue = list(set(queue))
        relation_concepts = [c for c in all_concepts.values() if c.get('role') == 'relation']
        return {"concepts": all_concepts, "edges": all_edges, "relations": relation_concepts, "visited_count": len(visited)}

    def search_concepts_by_bm25(self, query: str, role: str = None, limit: int = 20, time_point: str = None) -> List[dict]:
        if not query:
            return []
        results = []
        if role is None or role == "entity":
            entities = self.search_entities_by_bm25(query, limit=limit)
            for e in entities:
                results.append({"id": e.absolute_id, "family_id": e.family_id, "role": "entity",
                                "name": e.name, "content": e.content,
                                "event_time": _fmt_dt(e.event_time), "processed_time": _fmt_dt(e.processed_time)})
        if role is None or role == "relation":
            relations = self.search_relations_by_bm25(query, limit=limit)
            for r in relations:
                results.append({"id": r.absolute_id, "family_id": r.family_id, "role": "relation",
                                "name": "", "content": r.content,
                                "event_time": _fmt_dt(r.event_time), "processed_time": _fmt_dt(r.processed_time)})
        if role is None or role == "observation":
            episodes = self.search_episodes_by_bm25(query, limit=limit)
            for ep in episodes:
                results.append({"id": ep.absolute_id, "family_id": ep.absolute_id, "role": "observation",
                                "name": "", "content": ep.content or "",
                                "event_time": _fmt_dt(ep.event_time), "processed_time": _fmt_dt(ep.processed_time)})
        return results[:limit]

    def search_concepts_by_similarity(self, query_text: str, role: str = None,
                                       threshold: float = 0.5, max_results: int = 20,
                                       time_point: str = None) -> List[dict]:
        if not query_text or not self.embedding_client or not self.embedding_client.is_available():
            return self.search_concepts_by_bm25(query_text, role=role, limit=max_results, time_point=time_point)
        results = []
        if role is None or role == "entity":
            entities = self.search_entities_by_similarity(query_text, threshold=threshold, max_results=max_results)
            for e in entities:
                results.append({"id": e.absolute_id, "family_id": e.family_id, "role": "entity",
                                "name": e.name, "content": e.content, "score": 0.0})
        if role is None or role == "relation":
            relations = self.search_relations_by_similarity(query_text, threshold=threshold, max_results=max_results)
            for r in relations:
                results.append({"id": r.absolute_id, "family_id": r.family_id, "role": "relation",
                                "name": "", "content": r.content, "score": 0.0})
        if role == "observation" or (role is None and len(results) < max_results):
            bm25_results = self.search_concepts_by_bm25(query_text, role="observation", limit=max_results)
            for c in bm25_results:
                results.append(c)
        return results[:max_results]

    # ==================================================================
    # STATS (Additional methods)
    # ==================================================================

    def get_changes(self, since: datetime, until: Optional[datetime] = None) -> Dict[str, Any]:
        if until is None:
            until = datetime.now(timezone.utc)
        since_iso = since.isoformat()
        until_iso = until.isoformat()
        conn = self._connect()
        try:
            ent_rows = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE event_time >= ? AND event_time <= ? AND graph_id = ? ORDER BY event_time DESC",
                (since_iso, until_iso, self._graph_id),
            ).fetchall()
            entities = [_row_to_entity(dict(r)) for r in ent_rows]
            rel_rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE event_time >= ? AND event_time <= ? AND graph_id = ? ORDER BY event_time DESC",
                (since_iso, until_iso, self._graph_id),
            ).fetchall()
            relations = [_row_to_relation(dict(r)) for r in rel_rows]
        finally:
            conn.rollback()
        return {"entities": entities, "relations": relations}

    def get_snapshot(self, time_point: datetime, limit: Optional[int] = None) -> Dict[str, Any]:
        time_iso = time_point.isoformat()
        _limit = limit or 10000
        conn = self._connect()
        try:
            ent_rows = conn.execute(
                f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity "
                f"WHERE (valid_at IS NULL OR valid_at <= ?) AND graph_id = ? "
                f"ORDER BY event_time DESC LIMIT ?",
                (time_iso, self._graph_id, _limit),
            ).fetchall()
            entities = [_row_to_entity(dict(r)) for r in ent_rows]
            rel_rows = conn.execute(
                f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation "
                f"WHERE (valid_at IS NULL OR valid_at <= ?) AND graph_id = ? "
                f"ORDER BY event_time DESC LIMIT ?",
                (time_iso, self._graph_id, _limit),
            ).fetchall()
            relations = [_row_to_relation(dict(r)) for r in rel_rows]
        finally:
            conn.rollback()
        return {"entities": entities, "relations": relations}
