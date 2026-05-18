import hashlib
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...models import Episode
from ...utils import clean_markdown_code_blocks
from .helpers import EPISODE_COLUMNS, _encode_and_normalize, _fmt_dt, _parse_dt


class _EpisodeMixin:

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
