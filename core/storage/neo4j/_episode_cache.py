"""Neo4j EpisodeCacheMixin — cache, doc-hash, and filesystem helpers."""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...models import Episode
from ._helpers import _encode_and_normalize, _fmt_dt, _parse_dt

logger = logging.getLogger(__name__)


class EpisodeCacheMixin:
    """Episode cache / doc-hash / filesystem lookup helpers.

    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              -> Neo4j session factory
        self._run(session, cypher, **kw) -> execute Cypher with graph_id injection
        self._graph_id: str          -> active graph ID
        self.cache_dir               -> Path to episode cache dir
        self.cache_json_dir          -> Path to JSON cache dir
        self.cache_md_dir            -> Path to MD cache dir
        self.docs_dir                -> Path to docs dir
        self._id_to_doc_hash         -> Dict mapping cache_id to doc_hash
        self._doc_hash_to_dirname    -> Dict mapping doc_hash suffix to full dir name
        self.embedding_client        -> embedding client
        self._cache                  -> QueryCache
    """

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _compute_episode_embedding(self, content: str) -> Optional[bytes]:
        if not content:
            return None
        result = _encode_and_normalize(self.embedding_client, f"# Episode\n{content}")
        return result[0] if result else None

    # ------------------------------------------------------------------
    # Doc-hash directory helpers
    # ------------------------------------------------------------------

    def _get_cache_dir_by_doc_hash(self, doc_hash: str, document_path: str = "") -> Optional[Path]:
        """根据 doc_hash 获取文档目录（O(1) 查找 via reverse map）。"""
        if not doc_hash:
            return None
        # Fast path: direct match
        doc_dir = self.docs_dir / doc_hash
        if doc_dir.is_dir():
            return doc_dir
        # Reverse map lookup (O(1) instead of linear directory scan)
        dirname = self._doc_hash_to_dirname.get(doc_hash)
        if dirname:
            candidate = self.docs_dir / dirname
            if candidate.is_dir():
                return candidate
        return None

    # ------------------------------------------------------------------
    # Meta-file iteration (TTL-cached)
    # ------------------------------------------------------------------

    _meta_files_cache: tuple = (0.0, None)    # (timestamp, [Path, ...])
    _bm25_lower_cache: tuple = (0.0, None)    # (timestamp, {path_str: lowercased_content})
    _meta_json_cache: dict = {}      # {path_str: parsed_dict} — shared with BM25 TTL
    _META_FILES_TTL: float = 2.0     # seconds

    def _iter_cache_meta_files(self) -> List[Path]:
        """遍历 docs/ 目录下所有 meta.json 文件（带短 TTL 缓存）。"""
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

    # ------------------------------------------------------------------
    # Cache lookups by doc_hash
    # ------------------------------------------------------------------

    def find_cache_by_doc_hash(self, doc_hash: str, document_path: str = "") -> Optional[Episode]:
        """根据 doc_hash 查找 Episode。优先查 Neo4j，fallback 到文件系统扫描。"""
        # Fast path: check Neo4j first (O(1) index lookup)
        try:
            with self._session() as session:
                result = self._run(session, """
                    MATCH (ep:Episode) WHERE ep.doc_hash = $hash
                    RETURN ep.uuid AS uuid, ep.source_document AS source,
                           ep.event_time AS event_time, ep.processed_time AS processed_time,
                           ep.activity_type AS activity_type
                    LIMIT 1
                """, hash=doc_hash)
                record = result.single()
                if record and record["uuid"]:
                    cache_id = record["uuid"]
                    cache_md = self.cache_md_dir / f"{cache_id}.md"
                    content = cache_md.read_text(encoding="utf-8") if cache_md.exists() else ""
                    return Episode(
                        absolute_id=cache_id,
                        content=content,
                        event_time=record["event_time"] if record["event_time"] else datetime.now(),
                        processed_time=record["processed_time"],
                        source_document=record["source"] or "",
                        activity_type=record["activity_type"],
                    )
        except Exception:
            pass

        # Fallback: filesystem scan (use _meta_json_cache to avoid re-parsing)
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
            except Exception as e:
                logger.debug("Skipping meta file during doc_hash lookup: %s", e)
                continue
        return None

    def find_cache_and_extraction_by_doc_hash(
        self, doc_hash: str, document_path: str = ""
    ) -> Tuple[Optional[Episode], Optional[tuple]]:
        """Combined lookup: find Episode AND extraction result for a doc_hash in one pass.

        Returns (episode_or_None, extraction_result_or_None).
        Avoids calling find_cache_by_doc_hash + load_extraction_result separately
        (which would scan the filesystem twice for the same doc_hash).
        """
        if not doc_hash:
            return None, None

        episode = None
        extraction = None

        # 1. Resolve doc_dir once
        doc_dir = self._get_cache_dir_by_doc_hash(doc_hash, document_path)

        # 2. Try Neo4j for Episode (fast path)
        try:
            with self._session() as session:
                result = self._run(session, """
                    MATCH (ep:Episode) WHERE ep.doc_hash = $hash
                    RETURN ep.uuid AS uuid, ep.source_document AS source,
                           ep.event_time AS event_time, ep.processed_time AS processed_time,
                           ep.activity_type AS activity_type
                    LIMIT 1
                """, hash=doc_hash)
                record = result.single()
                if record and record["uuid"]:
                    cache_id = record["uuid"]
                    cache_md = self.cache_md_dir / f"{cache_id}.md"
                    content = cache_md.read_text(encoding="utf-8") if cache_md.exists() else ""
                    episode = Episode(
                        absolute_id=cache_id,
                        content=content,
                        event_time=record["event_time"] if record["event_time"] else datetime.now(),
                        processed_time=record["processed_time"],
                        source_document=record["source"] or "",
                        activity_type=record["activity_type"],
                    )
        except Exception:
            pass

        # 3. Load extraction result from the same doc_dir
        if doc_dir:
            extraction_path = doc_dir / "extraction.json"
            if extraction_path.exists():
                try:
                    data = json.loads(extraction_path.read_text(encoding="utf-8"))
                    extraction = (data.get("entities", []), data.get("relations", []))
                except Exception as e:
                    logger.debug("Failed to load extraction for doc_hash=%s: %s", doc_hash, e)

        # 4. Fallback Episode lookup via filesystem scan (only if Neo4j didn't find it)
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
                # Full filesystem scan fallback (use _meta_json_cache)
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

    # ------------------------------------------------------------------
    # Doc content / dir accessors
    # ------------------------------------------------------------------

    def get_doc_content(self, filename: str) -> Optional[Dict[str, Any]]:
        """获取文档内容。"""
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
        except Exception as e:
            logger.debug("Failed to read doc content for '%s': %s", filename, e)
            return None

    def get_doc_dir(self, doc_hash: str) -> Optional[Path]:
        """获取文档目录。"""
        return self._get_cache_dir_by_doc_hash(doc_hash)

    # ------------------------------------------------------------------
    # Doc hash resolution
    # ------------------------------------------------------------------

    def _resolve_doc_hash(self, cache_id: str) -> Optional[str]:
        """根据 cache_id 获取 doc_hash，内存缓存未命中时回退 Neo4j 查询。"""
        doc_hash = self._id_to_doc_hash.get(cache_id)
        if doc_hash:
            return doc_hash
        # Fallback: query Neo4j Episode node
        try:
            with self._session() as session:
                result = self._run(session,
                    "MATCH (ep:Episode {uuid: $uuid}) RETURN ep.doc_hash AS dh",
                    uuid=cache_id)
                record = result.single()
                if record and record["dh"]:
                    doc_hash = record["dh"]
                    self._id_to_doc_hash[cache_id] = doc_hash
                    return doc_hash
        except Exception as e:
            logger.debug("Neo4j fallback for doc_hash lookup failed: %s", e)
        return None

    def get_doc_hash_by_cache_id(self, cache_id: str) -> Optional[str]:
        """根据 cache_id 获取 doc_hash。"""
        return self._resolve_doc_hash(cache_id)
