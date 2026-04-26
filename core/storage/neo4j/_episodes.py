"""Neo4j EpisodeStoreMixin — extracted from neo4j_store."""
import hashlib
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...models import Episode, Entity, Relation
from ._helpers import _fmt_dt, _parse_dt
from ...utils import clean_markdown_code_blocks

logger = logging.getLogger(__name__)


class EpisodeStoreMixin:
    """EpisodeStore operations for Neo4j backend.
    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              → Neo4j session factory
        self._run(session, cypher, **kw) → execute Cypher with graph_id injection
        self._graph_id: str          → active graph ID
        self._episode_write_lock     → threading.Lock for episode writes
        self.cache_dir               → Path to episode cache dir
        self.cache_json_dir          → Path to JSON cache dir
        self.cache_md_dir            → Path to MD cache dir
        self.docs_dir                → Path to docs dir
        self._id_to_doc_hash         → Dict mapping cache_id to doc_hash
    """


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



    _meta_files_cache: tuple = ()    # (timestamp, [Path, ...])
    _bm25_lower_cache: tuple = ()    # (timestamp, {path_str: lowercased_content})
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



    def bulk_save_episodes(self, episodes: list) -> int:
        """批量保存 Episode 到 Neo4j，使用 UNWIND 单事务写入。

        Args:
            episodes: list of Episode objects

        Returns:
            保存的条数
        """
        if not episodes:
            return 0
        _now_iso = datetime.now().isoformat()
        rows = []
        for ep in episodes:
            rows.append({
                "uuid": ep.absolute_id,
                "content": ep.content or "",
                "source": getattr(ep, "source_document", "") or "",
                "event_time": ep.event_time.isoformat() if ep.event_time else _now_iso,
                "episode_type": getattr(ep, "episode_type", None),
                "activity_type": getattr(ep, "activity_type", None),
                "graph_id": self._graph_id,
            })
        with self._session() as session:
            self._run(session,
                """
                UNWIND $rows AS row
                MERGE (ep:Episode {uuid: row.uuid})
                SET ep:Concept, ep.role = 'observation',
                    ep.content = row.content,
                    ep.source_document = row.source,
                    ep.event_time = row.event_time,
                    ep.episode_type = row.episode_type,
                    ep.activity_type = row.activity_type,
                    ep.created_at = datetime(),
                    ep.graph_id = row.graph_id
                """,
                rows=rows,
            )
        return len(rows)



    def count_episodes(self) -> int:
        """统计 Episode 节点总数。"""
        with self._session() as session:
            result = self._run(session, "MATCH (ep:Episode) RETURN COUNT(ep) AS cnt")
            record = result.single()
            return record["cnt"] if record else 0



    def delete_episode(self, cache_id: str) -> int:
        """删除 docs/ 目录下的文件 + Neo4j Episode 节点。返回删除的条数。"""
        # 1. 尝试删除 docs/ 子目录
        doc_hash = self._id_to_doc_hash.get(cache_id)
        if doc_hash:
            doc_dir = self.docs_dir / doc_hash
            if doc_dir.is_dir():
                shutil.rmtree(doc_dir, ignore_errors=True)
                self._id_to_doc_hash.pop(cache_id, None)
        # 2. 删除 Neo4j Episode 节点
        with self._session() as session:
            result = self._run(session, "MATCH (ep:Episode {uuid: $uuid}) DETACH DELETE ep RETURN count(ep) AS cnt", uuid=cache_id)
            record = result.single()
            if record and record["cnt"] > 0:
                return 1
        # 3. 回退到旧结构
        for base_dir in (self.cache_json_dir, self.cache_dir):
            meta_path = base_dir / f"{cache_id}.json"
            if meta_path.exists():
                meta_path.unlink(missing_ok=True)
                return 1
        return 0



    def delete_episode_mentions(self, episode_id: str):
        """删除 Episode 的所有 MENTIONS 边。"""
        with self._session() as session:
            self._run(session, """
                MATCH (ep:Episode {uuid: $ep_id})-[m:MENTIONS]->()
                DELETE m
            """, ep_id=episode_id)



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

    def get_doc_hash_by_cache_id(self, cache_id: str) -> Optional[str]:
        """根据 cache_id 获取 doc_hash。"""
        return self._id_to_doc_hash.get(cache_id)



    def get_episode(self, uuid: str) -> Optional[Dict]:
        """获取单个 Episode 详情（含 MENTIONS 关联数量）。"""
        with self._session() as session:
            result = self._run(session,
                "MATCH (ep:Episode {uuid: $uuid}) "
                "OPTIONAL MATCH (ep)-[m:MENTIONS]->(target) "
                "RETURN ep.uuid AS uuid, ep.content AS content, "
                "ep.source_text AS source_text, "
                "ep.source_document AS source_document, ep.event_time AS event_time, "
                "ep.episode_id AS episode_id, ep.created_at AS created_at, "
                "count(m) AS mentions_count",
                uuid=uuid,
            )
            record = result.single()
            if not record:
                return None
            return {
                "uuid": record["uuid"],
                "content": record["content"] or "",
                "source_text": record.get("source_text") or "",
                "source_document": record["source_document"] or "",
                "event_time": _fmt_dt(record["event_time"]),
                "episode_id": record["episode_id"] or "",
                "created_at": _fmt_dt(record["created_at"]),
                "mentions_count": record.get("mentions_count", 0),
            }



    def get_episode_entities(self, episode_id: str) -> List[dict]:
        """获取 Episode 通过 MENTIONS 边关联的所有实体和关系。

        注意：Episode 节点可能缺少 graph_id（旧数据），因此使用 graph_id_safe=False
        避免 Episode 侧的 graph_id 过滤，仅通过 uuid 精确匹配。

        Returns:
            列表中每项包含:
              - absolute_id: 目标节点 uuid
              - target_type: "entity" 或 "relation"
              - name: 目标名称（relation 使用 family_id）
              - family_id: 目标 family_id
              - mention_context: MENTIONS 边的 context 属性
        """
        results = []
        with self._session() as session:
            # Single UNION ALL query: entity + relation mentions in one round-trip
            combined_result = self._run(session, """
                MATCH (ep:Episode {uuid: $ep_id})-[m:MENTIONS]->(e:Entity)
                WHERE e.graph_id = $graph_id
                RETURN e.uuid AS absolute_id, e.family_id AS family_id,
                       e.name AS name, m.context AS mention_context, 'entity' AS target_type
                UNION ALL
                MATCH (ep:Episode {uuid: $ep_id})-[m:MENTIONS]->(r:Relation)
                WHERE r.graph_id = $graph_id
                RETURN r.uuid AS absolute_id, r.family_id AS family_id,
                       r.family_id AS name, m.context AS mention_context, 'relation' AS target_type
            """, ep_id=episode_id, graph_id=self._graph_id, graph_id_safe=False)
            for r in combined_result:
                results.append({
                    "absolute_id": r["absolute_id"],
                    "target_type": r["target_type"],
                    "name": r.get("name", ""),
                    "family_id": r.get("family_id", ""),
                    "mention_context": r.get("mention_context", ""),
                })

        return results



    def get_episode_text(self, cache_id: str) -> Optional[str]:
        """获取记忆缓存对应的原始文本。"""
        doc_hash = self._id_to_doc_hash.get(cache_id)
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
                except Exception as e:
                    logger.debug("Failed to read episode text from meta.json: %s", e)
        # 回退旧结构
        metadata_path = self.cache_json_dir / f"{cache_id}.json"
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                return meta.get("text", "")
            except Exception as e:
                logger.debug("Failed to read episode text from fallback json: %s", e)
        return None



    def get_latest_episode(self, activity_type: Optional[str] = None) -> Optional[Episode]:
        """获取最新的记忆缓存。"""
        with self._session() as session:
            query = "MATCH (ep:Episode) "
            params: dict = {}
            if activity_type:
                query += "WHERE ep.activity_type = $atype "
                params["atype"] = activity_type
            query += "RETURN ep.uuid AS uuid, ep.content AS content, ep.event_time AS event_time, " \
                     "ep.processed_time AS processed_time, " \
                     "ep.source_document AS source_document, ep.activity_type AS activity_type " \
                     "ORDER BY ep.created_at DESC LIMIT 1"
            result = self._run(session, query, **params)
            record = result.single()
            if record:
                return Episode(
                    absolute_id=record["uuid"],
                    content=record["content"] or "",
                    event_time=_parse_dt(record["event_time"]) or datetime.now(),
                    processed_time=_parse_dt(record["processed_time"]),
                    source_document=record["source_document"] or "",
                    activity_type=record.get("activity_type"),
                )
        return None



    def get_latest_episode_metadata(self, activity_type: Optional[str] = None) -> Optional[Dict]:
        """获取最新记忆缓存的元数据（用于断点续传）。

        使用文件元数据（与文件后端一致），因为 Episode 节点
        不存储 text / document_path 等断点续传所需字段。
        Results are TTL-cached for 60s to avoid repeated filesystem scans.
        """
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
            except Exception as exc:
                logger.debug("episode meta load failed: %s", exc)
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


    def is_neo4j(self) -> bool:
        """标识当前为 Neo4j 后端。"""
        return True

    # ------------------------------------------------------------------
    # Episode 管理
    # ------------------------------------------------------------------



    def list_docs(self) -> List[Dict[str, Any]]:
        """列出所有文档。"""
        results = []
        for meta_file in self._iter_cache_meta_files():
            try:
                mf_key = str(meta_file)
                meta = self._meta_json_cache.get(mf_key)
                if meta is None:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    self._meta_json_cache[mf_key] = meta
                results.append({
                    "id": meta.get("absolute_id", ""),
                    "doc_hash": meta.get("doc_hash", ""),
                    "event_time": meta.get("event_time", ""),
                    "source_document": meta.get("source_document", ""),
                    "document_path": meta.get("document_path", ""),
                    "dir_name": meta_file.parent.name,
                })
            except Exception as e:
                logger.debug("Skipping meta file during list_docs: %s", e)
                continue
        return results

    # ------------------------------------------------------------------
    # 图遍历操作（Neo4j 原生优势）
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------

    def list_episodes(self, limit: int = 20, offset: int = 0, include_text: bool = False) -> List[Dict]:
        """分页查询 Episode 节点，按 created_at DESC。

        Args:
            include_text: 是否返回 source_text（原文），列表接口默认关闭以减少传输量。
        """
        fields = (
            "ep.uuid AS uuid, ep.content AS content, "
            "ep.source_document AS source_document, ep.event_time AS event_time, "
            "ep.processed_time AS processed_time, "
            "ep.episode_id AS episode_id, "
            "ep.created_at AS created_at"
        )
        if include_text:
            fields += ", ep.source_text AS source_text"
        with self._session() as session:
            result = self._run(session,
                f"MATCH (ep:Episode) RETURN {fields} "
                "ORDER BY ep.created_at DESC SKIP $offset LIMIT $limit",
                offset=offset, limit=limit,
            )
            episodes = []
            for r in result:
                ep = {
                    "uuid": r["uuid"],
                    "content": r["content"] or "",
                    "source_document": r["source_document"] or "",
                    "event_time": _fmt_dt(r["event_time"]),
                    "processed_time": _fmt_dt(r["processed_time"]),
                    "episode_id": r["episode_id"] or "",
                    "created_at": _fmt_dt(r["created_at"]),
                }
                if include_text:
                    ep["source_text"] = r.get("source_text") or ""
                episodes.append(ep)
            return episodes



    def load_episode(self, cache_id: str) -> Optional[Episode]:
        """从 Neo4j 或文件系统加载 Episode。"""
        # 优先从 Neo4j 加载
        with self._session() as session:
            result = self._run(session,
                "MATCH (ep:Episode {uuid: $uuid}) RETURN ep.content AS content, "
                "ep.event_time AS event_time, ep.processed_time AS processed_time, "
                "ep.source_document AS source_document",
                uuid=cache_id,
            )
            record = result.single()
            if record:
                return Episode(
                    absolute_id=cache_id,
                    content=record["content"] or "",
                    event_time=_parse_dt(record["event_time"]) or datetime.now(),
                    processed_time=_parse_dt(record["processed_time"]),
                    source_document=record["source_document"] or "",
                )

        # 回退到文件系统
        doc_hash = self._id_to_doc_hash.get(cache_id)
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
                except Exception as e:
                    logger.debug("Failed to load episode %s from file: %s", cache_id, e)
        return None


    def load_episodes(self, cache_ids: List[str]) -> List[Episode]:
        """Batch-load multiple episodes in a single Neo4j query."""
        if not cache_ids:
            return []
        results_map: Dict[str, Episode] = {}

        # Batch fetch from Neo4j
        try:
            with self._session() as session:
                result = self._run(session,
                    "UNWIND $ids AS id "
                    "MATCH (ep:Episode {uuid: id}) "
                    "RETURN ep.uuid AS uuid, ep.content AS content, "
                    "ep.event_time AS event_time, ep.processed_time AS processed_time, "
                    "ep.source_document AS source_document",
                    ids=cache_ids,
                )
                for record in result:
                    eid = record["uuid"]
                    results_map[eid] = Episode(
                        absolute_id=eid,
                        content=record["content"] or "",
                        event_time=_parse_dt(record["event_time"]) or datetime.now(),
                        processed_time=_parse_dt(record["processed_time"]),
                        source_document=record["source_document"] or "",
                    )
        except Exception:
            pass

        # Fallback: file-system for missing ids
        missing = [cid for cid in cache_ids if cid not in results_map]
        for cache_id in missing:
            doc_hash = self._id_to_doc_hash.get(cache_id)
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

        # Return in original order, skip missing
        return [results_map[cid] for cid in cache_ids if cid in results_map]



    def load_extraction_result(self, doc_hash: str,
                                document_path: str = "") -> Optional[tuple]:
        """加载抽取结果。"""
        doc_dir = self._get_cache_dir_by_doc_hash(doc_hash, document_path)
        if not doc_dir:
            return None
        extraction_path = doc_dir / "extraction.json"
        if not extraction_path.exists():
            return None
        try:
            data = json.loads(extraction_path.read_text(encoding="utf-8"))
            return data.get("entities", []), data.get("relations", [])
        except Exception as e:
            logger.debug("Failed to load extraction result for doc_hash=%s: %s", doc_hash, e)
            return None

    # ------------------------------------------------------------------
    # Embedding 计算
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------

    def save_episode(self, cache: Episode, text: str = "", document_path: str = "", doc_hash: str = "") -> str:
        """保存 Episode 到文件系统 + Neo4j。"""
        if not doc_hash and text:
            doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        if not doc_hash:
            doc_hash = "unknown"

        _now = datetime.now()
        _has_pt = hasattr(cache, 'processed_time')
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

        _proc_time = (cache.processed_time or _now).isoformat() if _has_pt else _now.isoformat()

        meta = {
            "absolute_id": cache.absolute_id,
            "event_time": cache.event_time.isoformat(),
            "processed_time": _proc_time,
            "activity_type": cache.activity_type,
            "source_document": cache.source_document,
            "text": text,
            "document_path": document_path,
            "doc_hash": doc_hash,
        }
        (doc_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if cache.absolute_id:
            self._id_to_doc_hash[cache.absolute_id] = doc_dir.name

        # 在 Neo4j 中创建 Episode 节点
        with self._session() as session:
            self._run(session,
                """
                MERGE (ep:Episode {uuid: $uuid})
                SET ep:Concept, ep.role = 'observation',
                    ep.content = $content,
                    ep.source_text = $source_text,
                    ep.source_document = $source,
                    ep.event_time = $event_time,
                    ep.processed_time = $processed_time,
                    ep.episode_type = $episode_type,
                    ep.activity_type = $activity_type,
                    ep.doc_hash = $doc_hash,
                    ep.created_at = datetime(),
                    ep.graph_id = $graph_id
                """,
                uuid=cache.absolute_id,
                content=cache.content,
                source_text=text or "",
                source=cache.source_document,
                event_time=cache.event_time.isoformat(),
                processed_time=_proc_time,
                episode_type=cache.episode_type,
                activity_type=cache.activity_type,
                doc_hash=doc_hash,
                graph_id=self._graph_id,
            )

        return doc_hash



    def save_episode_mentions(self, episode_id: str, entity_absolute_ids: List[str],
                              context: str = "", target_type: str = "entity"):
        """记录 Episode 提及的实体或关系（单次 UNWIND 批量写入）。

        Args:
            episode_id: Episode 节点的 uuid。
            entity_absolute_ids: 目标节点（Entity 或 Relation）的 absolute_id 列表。
            context: 提及上下文描述。
            target_type: "entity" 创建 (ep)-[:MENTIONS]->(e:Entity)，
                         "relation" 创建 (ep)-[:MENTIONS]->(r:Relation)。
        """
        if not entity_absolute_ids:
            return
        with self._episode_write_lock:
            with self._session() as session:
                if target_type == "relation":
                    self._run(session, """
                        MERGE (ep:Episode {uuid: $ep_id})
                        ON CREATE SET ep.graph_id = $graph_id
                        WITH ep
                        UNWIND $items AS item
                        MATCH (r:Relation {uuid: item.abs_id})
                        MERGE (ep)-[m:MENTIONS {context: item.ctx}]->(r)
                    """, ep_id=episode_id,
                         items=[{"abs_id": aid, "ctx": context} for aid in entity_absolute_ids])
                else:
                    self._run(session, """
                        MERGE (ep:Episode {uuid: $ep_id})
                        ON CREATE SET ep.graph_id = $graph_id
                        WITH ep
                        UNWIND $items AS item
                        MATCH (e:Entity {uuid: item.abs_id})
                        MERGE (ep)-[m:MENTIONS {context: item.ctx}]->(e)
                    """, ep_id=episode_id,
                         items=[{"abs_id": aid, "ctx": context} for aid in entity_absolute_ids])



    def save_extraction_result(self, doc_hash: str, entities: list, relations: list,
                                document_path: str = "") -> bool:
        """保存抽取结果到文件。"""
        doc_dir = self._get_cache_dir_by_doc_hash(doc_hash, document_path)
        if not doc_dir:
            return False
        try:
            result = {
                "entities": [
                    {
                        "absolute_id": e.absolute_id, "family_id": e.family_id,
                        "name": e.name, "content": e.content,
                    }
                    for e in entities
                ],
                "relations": [
                    {
                        "absolute_id": r.absolute_id, "family_id": r.family_id,
                        "content": r.content,
                    }
                    for r in relations
                ],
            }
            (doc_dir / "extraction.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return True
        except Exception as e:
            logger.debug("Failed to save extraction result for doc_hash=%s: %s", doc_hash, e)
            return False



    def batch_get_source_text_snippets(self, episode_ids: List[str], snippet_length: int = 200) -> Dict[str, str]:
        """Batch-fetch source_text snippets for multiple episode UUIDs.

        Returns dict mapping episode_id -> source_text[:snippet_length].
        """
        if not episode_ids:
            return {}
        with self._session() as session:
            result = self._run(session,
                "MATCH (ep:Episode) WHERE ep.uuid IN $ids "
                "RETURN ep.uuid AS uuid, ep.source_text AS source_text",
                ids=episode_ids,
            )
            return {
                r["uuid"]: (r.get("source_text") or "")[:snippet_length]
                for r in result
            }

    def search_episodes(self, query: str, limit: int = 20) -> List[Dict]:
        """通过 content LIKE 搜索 Episode。"""
        with self._session() as session:
            result = self._run(session,
                "MATCH (ep:Episode) WHERE ep.content CONTAINS $search_query "
                "RETURN ep.uuid AS uuid, ep.content AS content, "
                "ep.source_text AS source_text, "
                "ep.source_document AS source_document, ep.event_time AS event_time, "
                "ep.episode_id AS episode_id, ep.created_at AS created_at "
                "ORDER BY ep.created_at DESC LIMIT $limit",
                search_query=query, limit=limit,
            )
            episodes = []
            for r in result:
                episodes.append({
                    "uuid": r["uuid"],
                    "content": r["content"] or "",
                    "source_text": r.get("source_text") or "",
                    "source_document": r["source_document"] or "",
                    "event_time": _fmt_dt(r["event_time"]),
                    "episode_id": r["episode_id"] or "",
                    "created_at": _fmt_dt(r["created_at"]),
                })
            return episodes

    # ------------------------------------------------------------------
    # 社区检测
    # ------------------------------------------------------------------



    def search_episodes_by_bm25(self, query: str, limit: int = 20) -> List[Episode]:
        """遍历 docs/ 目录搜索 Episode（简单文本匹配，与 SQLite 版本一致）。

        优化：先从 cache.md 文件直接读取内容做 BM25 匹配（无 DB 调用），
        只对最终 top-K 结果调用 load_episode 获取完整 Episode 对象。
        缓存 lowercased content 以避免重复 .lower() 开销。
        """
        if not query:
            return []
        query_lower = query.lower()
        # Use TTL-based cache for lowercased content (same TTL as meta_files cache)
        now = time.monotonic()
        _cache_ts, _cache_map = self._bm25_lower_cache
        if _cache_map is None or now - _cache_ts > self._META_FILES_TTL:
            _cache_map = {}
            self._bm25_lower_cache = (now, _cache_map)
            self._meta_json_cache.clear()

        scored: List[Tuple[int, str]] = []  # (score, cache_id)
        for meta_file in self._iter_cache_meta_files():
            try:
                mf_key = str(meta_file)
                meta = self._meta_json_cache.get(mf_key)
                if meta is None:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    self._meta_json_cache[mf_key] = meta
                cache_id = meta.get("absolute_id") or meta.get("id") or meta_file.parent.name
            except Exception as e:
                logger.debug("Skipping meta file %s: %s", meta_file, e)
                continue
            # 使用缓存的 lowercased content
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
                # .count() is O(n) but needed for scoring; the `in` check above
                # is also O(n) worst-case but short-circuits on first match.
                # CPython optimizes the common case where both use the same
                # internal find() machinery, making the second scan fast.
                score = content_lower.count(query_lower)
                scored.append((score, cache_id))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Batch-load top-K episodes (single UNWIND query instead of N individual queries)
        top_ids = [cid for _, cid in scored[:limit]]
        if not top_ids:
            return []
        episodes = self.load_episodes(top_ids)
        return episodes

