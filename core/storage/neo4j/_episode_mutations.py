"""Neo4j EpisodeMutationMixin — write / delete operations on episodes."""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np

from ...models import Episode
from ...utils import clean_markdown_code_blocks

logger = logging.getLogger(__name__)


class EpisodeMutationMixin:
    """Episode write operations: save, delete, mentions, extraction.

    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              -> Neo4j session factory
        self._run(session, cypher, **kw) -> execute Cypher with graph_id injection
        self._graph_id: str          -> active graph ID
        self._episode_write_lock     -> threading.Lock for episode writes
        self.cache_dir               -> Path to episode cache dir
        self.cache_json_dir          -> Path to JSON cache dir
        self.docs_dir                -> Path to docs dir
        self._id_to_doc_hash         -> Dict mapping cache_id to doc_hash
        self.embedding_client        -> embedding client
    """

    def bulk_save_episodes(self, episodes: list) -> int:
        """批量保存 Episode 到 Neo4j，使用 UNWIND 单事务写入。

        Args:
            episodes: list of Episode objects

        Returns:
            保存的条数
        """
        if not episodes:
            return 0
        _now_iso = datetime.now(timezone.utc).isoformat()

        # Compute embeddings BEFORE write
        embeddings = None
        if self.embedding_client and self.embedding_client.is_available():
            texts = [f"# Episode\n{ep.content}" for ep in episodes if ep.content]
            if texts:
                embeddings = self.embedding_client.encode(texts)

        rows = []
        ep_idx = 0
        for ep in episodes:
            embedding_list = None
            if embeddings is not None and ep.content and ep.absolute_id:
                if ep_idx < len(embeddings):
                    try:
                        emb_arr = np.array(embeddings[ep_idx], dtype=np.float32)
                        norm = np.linalg.norm(emb_arr)
                        if norm > 0:
                            emb_arr = emb_arr / norm
                        embedding_list = emb_arr.tolist()
                    except Exception as e:
                        logger.debug("Episode embedding decode failed for %s: %s", ep.absolute_id, e)
                ep_idx += 1

            rows.append({
                "uuid": ep.absolute_id,
                "content": ep.content or "",
                "source": getattr(ep, "source_document", "") or "",
                "event_time": ep.event_time.isoformat() if ep.event_time else _now_iso,
                "episode_type": getattr(ep, "episode_type", None),
                "activity_type": getattr(ep, "activity_type", None),
                "graph_id": self._graph_id,
                "embedding": embedding_list,
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
                    ep.graph_id = row.graph_id,
                    ep.embedding = row.embedding
                """,
                rows=rows,
            )

        return len(rows)

    def delete_episode(self, cache_id: str) -> int:
        """删除 docs/ 目录下的文件 + Neo4j Episode 节点。返回删除的条数。"""
        # 1. 尝试删除 docs/ 子目录
        doc_hash = self._resolve_doc_hash(cache_id)
        if doc_hash:
            doc_dir = self.docs_dir / doc_hash
            if doc_dir.is_dir():
                import shutil
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

    def save_episode(self, cache: Episode, text: str = "", document_path: str = "", doc_hash: str = "") -> str:
        """保存 Episode 到文件系统 + Neo4j。"""
        if not doc_hash and text:
            doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        if not doc_hash:
            doc_hash = "unknown"

        _now = datetime.now(timezone.utc)
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

        # Compute embedding BEFORE Neo4j write
        embedding_list = None
        embedding_blob = self._compute_episode_embedding(cache.content)
        if embedding_blob:
            emb_arr = np.frombuffer(embedding_blob, dtype=np.float32)
            embedding_list = emb_arr.tolist()

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
                    ep.graph_id = $graph_id,
                    ep.embedding = $embedding
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
                embedding=embedding_list,
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

    def load_extraction_result(self, doc_hash: str,
                                document_path: str = "") -> "tuple | None":
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
