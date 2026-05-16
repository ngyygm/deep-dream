"""Neo4j EpisodeQueryMixin — read-only episode queries and searches."""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...models import Episode
from ._helpers import _fmt_dt, _parse_dt

logger = logging.getLogger(__name__)


class EpisodeQueryMixin:
    """Episode read-only queries: count, get, list, load, search.

    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              -> Neo4j session factory
        self._run(session, cypher, **kw) -> execute Cypher with graph_id injection
        self._graph_id: str          -> active graph ID
        self.cache_dir               -> Path to episode cache dir
        self.cache_json_dir          -> Path to JSON cache dir
        self.docs_dir                -> Path to docs dir
        self._cache                  -> QueryCache
    """

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def count_episodes(self) -> int:
        """统计 Episode 节点总数。"""
        with self._session() as session:
            result = self._run(session, "MATCH (ep:Episode) RETURN COUNT(ep) AS cnt")
            record = result.single()
            return record["cnt"] if record else 0

    # ------------------------------------------------------------------
    # Single-episode lookups
    # ------------------------------------------------------------------

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

        # Fallback: query Neo4j source_text property
        try:
            with self._session() as session:
                result = self._run(session,
                    "MATCH (ep:Episode {uuid: $uuid}) "
                    "RETURN ep.source_text AS source_text",
                    uuid=cache_id,
                )
                record = result.single()
                if record and record.get("source_text"):
                    return record["source_text"]
        except Exception as e:
            logger.debug("Neo4j fallback for episode text failed: %s", e)
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

    # ------------------------------------------------------------------
    # List / enumerate
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
                doc_dir = meta_file.parent
                # 计算原文大小
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
                    "id": meta.get("absolute_id", ""),
                    "doc_hash": meta.get("doc_hash", ""),
                    "event_time": meta.get("event_time", ""),
                    "processed_time": meta.get("processed_time", ""),
                    "source_document": meta.get("source_document", ""),
                    "document_path": meta.get("document_path", ""),
                    "dir_name": doc_dir.name,
                    "activity_type": meta.get("activity_type", ""),
                    "text_length": text_length,
                    "original_size": original_size,
                    "cache_size": cache_size,
                })
            except Exception as e:
                logger.debug("Skipping meta file during list_docs: %s", e)
                continue
        return results

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

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

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

        # Return in original order, skip missing
        return [results_map[cid] for cid in cache_ids if cid in results_map]

    # ------------------------------------------------------------------
    # Batch source-text snippets
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

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

    def search_episodes_by_bm25(self, query: str, limit: int = 20) -> List[Episode]:
        """遍历 docs/ 目录搜索 Episode（简单文本匹配）。

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
