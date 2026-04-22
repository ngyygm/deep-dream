"""Neo4j EpisodeStoreMixin — extracted from neo4j_store."""
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        """根据 doc_hash 获取文档目录。"""
        if not doc_hash:
            return None
        # 直接查找
        doc_dir = self.docs_dir / doc_hash
        if doc_dir.is_dir():
            return doc_dir
        # 在子目录中搜索
        for d in self.docs_dir.iterdir():
            if d.is_dir() and d.name.endswith(f"_{doc_hash}"):
                return d
        return None



    def _iter_cache_meta_files(self) -> List[Path]:
        """遍历 docs/ 目录下所有 meta.json 文件。"""
        if not self.docs_dir.is_dir():
            return []
        return sorted(self.docs_dir.glob("*/meta.json"))



    def bulk_save_episodes(self, episodes: list) -> int:
        """批量保存 Episode 到 Neo4j，使用 UNWIND 单事务写入。

        Args:
            episodes: list of Episode objects

        Returns:
            保存的条数
        """
        if not episodes:
            return 0
        rows = []
        for ep in episodes:
            rows.append({
                "uuid": ep.absolute_id,
                "content": ep.content or "",
                "source": getattr(ep, "source_document", "") or "",
                "event_time": ep.event_time.isoformat() if ep.event_time else datetime.now().isoformat(),
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
        import shutil

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
        """根据 doc_hash 查找 Episode。"""
        for meta_file in self._iter_cache_meta_files():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if meta.get("doc_hash") == doc_hash:
                    return Episode(
                        absolute_id=meta.get("absolute_id", ""),
                        content=(meta_file.parent / "cache.md").read_text(encoding="utf-8") if (meta_file.parent / "cache.md").exists() else "",
                        event_time=datetime.fromisoformat(meta["event_time"]) if meta.get("event_time") else datetime.now(),
                        processed_time=datetime.fromisoformat(meta["processed_time"]) if meta.get("processed_time") else None,
                        source_document=meta.get("source_document", ""),
                        activity_type=meta.get("activity_type"),
                    )
            except Exception as e:
                logger.debug("Skipping meta file during doc_hash lookup: %s", e)
                continue
        return None



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
            # Entity targets（不对 Episode 注入 graph_id 过滤）
            ent_result = self._run(session, """
                MATCH (ep:Episode {uuid: $ep_id})-[m:MENTIONS]->(e:Entity)
                WHERE e.graph_id = $graph_id
                RETURN e.uuid AS absolute_id, e.family_id AS family_id,
                       e.name AS name, m.context AS mention_context
            """, ep_id=episode_id, graph_id=self._graph_id, graph_id_safe=False)
            for r in ent_result:
                results.append({
                    "absolute_id": r["absolute_id"],
                    "target_type": "entity",
                    "name": r.get("name", ""),
                    "family_id": r.get("family_id", ""),
                    "mention_context": r.get("mention_context", ""),
                })

            # Relation targets（同理）
            rel_result = self._run(session, """
                MATCH (ep:Episode {uuid: $ep_id})-[m:MENTIONS]->(r:Relation)
                WHERE r.graph_id = $graph_id
                RETURN r.uuid AS absolute_id, r.family_id AS family_id,
                       r.family_id AS name, m.context AS mention_context
            """, ep_id=episode_id, graph_id=self._graph_id, graph_id_safe=False)
            for r in rel_result:
                results.append({
                    "absolute_id": r["absolute_id"],
                    "target_type": "relation",
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
        """获取最新记忆缓存的元数据。"""
        with self._session() as session:
            query = "MATCH (ep:Episode) "
            params: dict = {}
            if activity_type:
                query += "WHERE ep.activity_type = $atype "
                params["atype"] = activity_type
            query += "RETURN ep.uuid AS uuid, ep.event_time AS event_time, " \
                     "ep.source_document AS source_document, ep.created_at AS created_at " \
                     "ORDER BY ep.created_at DESC LIMIT 1"
            result = self._run(session, query, **params)
            record = result.single()
            if record:
                evt = record["event_time"]
                cat = record["created_at"]
                return {
                    "id": record["uuid"],
                    "event_time": evt.isoformat() if hasattr(evt, 'isoformat') else (str(evt) if evt else None),
                    "source_document": record["source_document"],
                    "created_at": cat.isoformat() if hasattr(cat, 'isoformat') else (str(cat) if cat else None),
                }
        return None


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
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
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
                    return Episode(
                        absolute_id=cache_id,
                        content=(doc_dir / "cache.md").read_text(encoding="utf-8") if (doc_dir / "cache.md").exists() else "",
                        event_time=datetime.fromisoformat(meta["event_time"]) if meta.get("event_time") else datetime.now(),
                        processed_time=datetime.fromisoformat(meta["processed_time"]) if meta.get("processed_time") else None,
                        source_document=meta.get("source_document", ""),
                        activity_type=meta.get("activity_type"),
                    )
                except Exception as e:
                    logger.debug("Failed to load episode %s from file: %s", cache_id, e)
        return None



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

        ts_prefix = cache.event_time.strftime("%Y%m%d_%H%M%S") if cache.event_time else datetime.now().strftime("%Y%m%d_%H%M%S")
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

        meta = {
            "absolute_id": cache.absolute_id,
            "event_time": cache.event_time.isoformat(),
            "processed_time": (cache.processed_time or datetime.now()).isoformat() if hasattr(cache, 'processed_time') else datetime.now().isoformat(),
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
                    ep.created_at = datetime(),
                    ep.graph_id = $graph_id
                """,
                uuid=cache.absolute_id,
                content=cache.content,
                source_text=text or "",
                source=cache.source_document,
                event_time=cache.event_time.isoformat(),
                processed_time=(cache.processed_time or datetime.now()).isoformat() if hasattr(cache, 'processed_time') else datetime.now().isoformat(),
                episode_type=cache.episode_type,
                activity_type=cache.activity_type,
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
        """遍历 docs/ 目录搜索 Episode（简单文本匹配，与 SQLite 版本一致）。"""
        if not query:
            return []
        query_lower = query.lower()
        results = []
        for meta_file in self._iter_cache_meta_files():
            try:
                # 从 meta.json 读取 absolute_id 作为 cache_id（而非目录名）
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                cache_id = meta.get("absolute_id") or meta.get("id") or meta_file.parent.name
                cache = self.load_episode(cache_id)
            except Exception as e:
                logger.debug("Skipping episode file %s during search: %s", meta_file, e)
                continue
            if cache is None:
                continue
            content_lower = (cache.content or "").lower()
            if query_lower in content_lower:
                score = content_lower.count(query_lower)
                results.append((score, cache))
        results.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in results[:limit]]

