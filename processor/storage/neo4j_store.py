"""
Neo4jStorageManager: Neo4j + sqlite-vec 混合存储后端。

借鉴 Graphiti (Zep) 的分层节点架构：
    Neo4j       → 图结构存储（Entity / Relation / Episode 节点及边）
    sqlite-vec  → embedding 向量存储与 KNN 搜索

与 StorageManager 保持完全相同的公共接口，可作为 drop-in replacement。
"""

import difflib
import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np

from ..models import MemoryCache, Entity, Relation
from ..utils import clean_markdown_code_blocks, wprint
from .cache import QueryCache
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neo4j 节点 / 边 属性 → Entity / Relation 转换
# ---------------------------------------------------------------------------

def _neo4j_record_to_entity(record) -> Entity:
    """将 Neo4j 查询返回的单条记录转为 Entity dataclass。"""
    return Entity(
        absolute_id=record["uuid"],
        entity_id=record["entity_id"],
        name=record.get("name", ""),
        content=record.get("content", ""),
        event_time=_parse_dt(record.get("event_time")),
        processed_time=_parse_dt(record.get("processed_time")),
        memory_cache_id=record.get("memory_cache_id", ""),
        source_document=record.get("source_document", "") or "",
        embedding=record.get("embedding"),
        valid_at=_parse_dt(record.get("valid_at")) if record.get("valid_at") is not None else None,
        invalid_at=_parse_dt(record.get("invalid_at")) if record.get("invalid_at") is not None else None,
        summary=record.get("summary"),
        attributes=record.get("attributes"),
        confidence=float(record["confidence"]) if record.get("confidence") is not None else None,
    )


def _neo4j_record_to_relation(record) -> Relation:
    """将 Neo4j 查询返回的单条记录转为 Relation dataclass。"""
    return Relation(
        absolute_id=record["uuid"],
        relation_id=record["relation_id"],
        entity1_absolute_id=record.get("entity1_absolute_id", ""),
        entity2_absolute_id=record.get("entity2_absolute_id", ""),
        content=record.get("content", ""),
        event_time=_parse_dt(record.get("event_time")),
        processed_time=_parse_dt(record.get("processed_time")),
        memory_cache_id=record.get("memory_cache_id", ""),
        source_document=record.get("source_document", "") or "",
        embedding=record.get("embedding"),
        valid_at=_parse_dt(record.get("valid_at")) if record.get("valid_at") is not None else None,
        invalid_at=_parse_dt(record.get("invalid_at")) if record.get("invalid_at") is not None else None,
        summary=record.get("summary"),
        attributes=record.get("attributes"),
        confidence=float(record["confidence"]) if record.get("confidence") is not None else None,
        provenance=record.get("provenance"),
    )


def _parse_dt(value: Any) -> datetime:
    """安全解析日期时间。"""
    if value is None:
        return datetime.now()
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return datetime.now()


def _fmt_dt(value: Any) -> Optional[str]:
    """安全格式化日期时间为 ISO 字符串。兼容 datetime 对象和字符串。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class Neo4jStorageManager:
    """Neo4j + sqlite-vec 混合存储管理器。

    实现与 StorageManager 完全相同的公共接口，用于替代 SQLite 后端。

    Usage:
        sm = Neo4jStorageManager(
            neo4j_uri="bolt://localhost:7687",
            neo4j_auth=("neo4j", "password"),
            storage_path="./graph",
            embedding_client=embedding_client,
        )
    """

    def __init__(
        self,
        storage_path: str,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_auth: Tuple[str, str] = ("neo4j", "password"),
        embedding_client=None,
        entity_content_snippet_length: int = 50,
        relation_content_snippet_length: int = 50,
        vector_dim: int = 1024,
        **_kwargs,
    ):
        import neo4j

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Neo4j 驱动
        self._neo4j_uri = neo4j_uri
        self._neo4j_auth = neo4j_auth
        self._driver = neo4j.GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        self._driver.verify_connectivity()

        # 文档目录（与 StorageManager 相同的文件存储结构）
        self.docs_dir = self.storage_path / "docs"
        self.docs_dir.mkdir(exist_ok=True)
        self.cache_dir = self.storage_path / "memory_caches"
        self.cache_json_dir = self.cache_dir / "json"
        self.cache_md_dir = self.cache_dir / "md"

        # 缓存 cache_id → doc_hash 映射
        self._id_to_doc_hash: Dict[str, str] = {}

        # 写锁
        self._write_lock = threading.Lock()

        # 查询缓存
        self._cache = QueryCache(default_ttl=30)

        # Embedding 客户端
        self.embedding_client = embedding_client
        self.entity_content_snippet_length = entity_content_snippet_length
        self.relation_content_snippet_length = relation_content_snippet_length

        # sqlite-vec 向量存储
        self._vector_store = VectorStore(
            str(self.storage_path / "vectors.db"),
            dim=vector_dim,
        )

        # 初始化 Neo4j 约束和索引
        self._init_schema()

        # 构建缓存映射
        self._build_doc_hash_cache()

    # ------------------------------------------------------------------
    # Neo4j Schema 初始化
    # ------------------------------------------------------------------

    def _init_schema(self):
        """创建 Neo4j 约束和索引（幂等）。"""
        constraints = [
            # Entity 唯一性约束
            "CREATE CONSTRAINT entity_uuid IF NOT EXISTS FOR (e:Entity) REQUIRE e.uuid IS UNIQUE",
            # Relation 唯一性约束
            "CREATE CONSTRAINT relation_uuid IF NOT EXISTS FOR (r:Relation) REQUIRE r.uuid IS UNIQUE",
            # Episode 唯一性约束
            "CREATE CONSTRAINT episode_uuid IF NOT EXISTS FOR (ep:Episode) REQUIRE ep.uuid IS UNIQUE",
            # Entity redirect 唯一性约束
            "CREATE CONSTRAINT redirect_source IF NOT EXISTS FOR (red:EntityRedirect) REQUIRE red.source_id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX entity_entity_id IF NOT EXISTS FOR (e:Entity) ON (e.entity_id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_processed_time IF NOT EXISTS FOR (e:Entity) ON (e.processed_time)",
            "CREATE INDEX entity_event_time IF NOT EXISTS FOR (e:Entity) ON (e.event_time)",
            "CREATE INDEX entity_cache_id IF NOT EXISTS FOR (e:Entity) ON (e.memory_cache_id)",
            "CREATE INDEX relation_relation_id IF NOT EXISTS FOR (r:Relation) ON (r.relation_id)",
            "CREATE INDEX relation_processed_time IF NOT EXISTS FOR (r:Relation) ON (r.processed_time)",
            "CREATE INDEX relation_entities IF NOT EXISTS FOR (r:Relation) ON (r.entity1_absolute_id, r.entity2_absolute_id)",
            "CREATE INDEX redirect_target IF NOT EXISTS FOR (red:EntityRedirect) ON (red.target_id)",
        ]
        with self._driver.session() as session:
            for c in constraints:
                try:
                    session.run(c)
                except Exception as e:
                    logger.debug("Constraint creation skipped: %s", e)
            for idx in indexes:
                try:
                    session.run(idx)
                except Exception as e:
                    logger.debug("Index creation skipped: %s", e)
            # BM25 全文搜索索引（Neo4j 5.x）
            fulltext_indexes = [
                ("entityFulltext", "CREATE FULLTEXT INDEX entityFulltext IF NOT EXISTS FOR (e:Entity) ON (e.name, e.content)"),
                ("relationFulltext", "CREATE FULLTEXT INDEX relationFulltext IF NOT EXISTS FOR (r:Relation) ON (r.content)"),
            ]
            for idx_name, idx_cypher in fulltext_indexes:
                try:
                    session.run(idx_cypher)
                except Exception as e:
                    logger.debug("Fulltext index %s creation skipped: %s", idx_name, e)

            # Performance indexes
            perf_indexes = [
                "CREATE INDEX entity_source_document IF NOT EXISTS FOR (e:Entity) ON (e.source_document)",
                "CREATE INDEX relation_source_document IF NOT EXISTS FOR (r:Relation) ON (r.source_document)",
                "CREATE INDEX episode_memory_cache_id IF NOT EXISTS FOR (ep:Episode) ON (ep.memory_cache_id)",
                # Phase C: MENTIONS edge index
                "CREATE INDEX mentions_entity IF NOT EXISTS FOR ()-[m:MENTIONS]->() ON (m.entity_absolute_id)",
                # Phase E: DreamLog
                "CREATE INDEX dream_log_graph IF NOT EXISTS FOR (d:DreamLog) ON (d.graph_id)",
            ]
            for idx in perf_indexes:
                try:
                    session.run(idx)
                except Exception as e:
                    logger.debug("Performance index creation skipped: %s", e)

    def _build_doc_hash_cache(self):
        """从 docs/ 目录构建 cache_id → doc_hash 映射。"""
        if not self.docs_dir.is_dir():
            return
        for doc_dir in self.docs_dir.iterdir():
            if not doc_dir.is_dir():
                continue
            meta_path = doc_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    cache_id = meta.get("absolute_id") or meta.get("id")
                    if cache_id:
                        self._id_to_doc_hash[cache_id] = doc_dir.name
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def close(self):
        """关闭 Neo4j 驱动和向量存储连接。"""
        try:
            self._driver.close()
        except Exception:
            pass
        self._vector_store.close()

    # ------------------------------------------------------------------
    # Entity Redirect（实体 ID 解析）
    # ------------------------------------------------------------------

    def _resolve_entity_id_in_session(self, session, entity_id: str) -> str:
        """沿 EntityRedirect 链解析到 canonical entity_id。"""
        current_id = (entity_id or "").strip()
        if not current_id:
            return ""
        seen: Set[str] = set()
        while current_id and current_id not in seen:
            seen.add(current_id)
            result = session.run(
                "MATCH (red:EntityRedirect {source_id: $sid}) RETURN red.target_id AS target",
                sid=current_id,
            )
            record = result.single()
            if not record or not record["target"] or record["target"] == current_id:
                break
            current_id = record["target"]
        return current_id

    def resolve_entity_id(self, entity_id: str) -> str:
        """解析 entity_id 到 canonical id。"""
        with self._driver.session() as session:
            return self._resolve_entity_id_in_session(session, entity_id)

    def register_entity_redirect(self, source_entity_id: str, target_entity_id: str) -> str:
        """登记旧 entity_id → canonical entity_id 映射。"""
        source_id = (source_entity_id or "").strip()
        target_id = (target_entity_id or "").strip()
        if not source_id or not target_id:
            return target_id
        with self._write_lock:
            with self._driver.session() as session:
                canonical_target = self._resolve_entity_id_in_session(session, target_id)
                if not canonical_target:
                    canonical_target = target_id
                canonical_source = self._resolve_entity_id_in_session(session, source_id)
                if canonical_source == canonical_target:
                    return canonical_target
                now_iso = datetime.now().isoformat()
                session.run(
                    """
                    MERGE (red:EntityRedirect {source_id: $sid})
                    SET red.target_id = $tid, red.updated_at = $now
                    """,
                    sid=source_id,
                    tid=canonical_target,
                    now=now_iso,
                )
            return canonical_target

    def register_entity_redirects(self, target_entity_id: str, source_entity_ids: List[str]) -> str:
        """批量登记多个旧 entity_id 指向同一 canonical id。"""
        canonical_target = (target_entity_id or "").strip()
        if not canonical_target:
            return canonical_target
        for source_id in source_entity_ids:
            if source_id and source_id != canonical_target:
                canonical_target = self.register_entity_redirect(source_id, canonical_target)
        return canonical_target

    # ------------------------------------------------------------------
    # MemoryCache 操作（文件存储，与 StorageManager 相同逻辑）
    # ------------------------------------------------------------------

    def save_memory_cache(self, cache: MemoryCache, text: str = "", document_path: str = "", doc_hash: str = "") -> str:
        """保存记忆缓存到 docs/ 目录。"""
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
        with self._driver.session() as session:
            session.run(
                """
                MERGE (ep:Episode {uuid: $uuid})
                SET ep.content = $content,
                    ep.memory_cache_id = $cache_id,
                    ep.source_document = $source,
                    ep.event_time = $event_time,
                    ep.created_at = datetime()
                """,
                uuid=cache.absolute_id,
                content=cache.content,
                cache_id=cache.absolute_id,
                source=cache.source_document,
                event_time=cache.event_time.isoformat(),
            )

        return doc_hash

    def load_memory_cache(self, cache_id: str) -> Optional[MemoryCache]:
        """从 Neo4j 或文件系统加载记忆缓存。"""
        # 优先从 Neo4j 加载
        with self._driver.session() as session:
            result = session.run(
                "MATCH (ep:Episode {uuid: $uuid}) RETURN ep.content AS content, "
                "ep.event_time AS event_time, ep.source_document AS source_document",
                uuid=cache_id,
            )
            record = result.single()
            if record:
                return MemoryCache(
                    absolute_id=cache_id,
                    content=record["content"] or "",
                    event_time=_parse_dt(record["event_time"]),
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
                    return MemoryCache(
                        absolute_id=cache_id,
                        content=(doc_dir / "cache.md").read_text(encoding="utf-8") if (doc_dir / "cache.md").exists() else "",
                        event_time=datetime.fromisoformat(meta["event_time"]) if meta.get("event_time") else datetime.now(),
                        source_document=meta.get("source_document", ""),
                        activity_type=meta.get("activity_type"),
                    )
                except Exception:
                    pass
        return None

    def _iter_cache_meta_files(self) -> List[Path]:
        """遍历 docs/ 目录下所有 meta.json 文件。"""
        if not self.docs_dir.is_dir():
            return []
        return sorted(self.docs_dir.glob("*/meta.json"))

    def get_latest_memory_cache(self, activity_type: Optional[str] = None) -> Optional[MemoryCache]:
        """获取最新的记忆缓存。"""
        with self._driver.session() as session:
            query = "MATCH (ep:Episode) "
            params: dict = {}
            if activity_type:
                query += "WHERE ep.activity_type = $atype "
                params["atype"] = activity_type
            query += "RETURN ep.uuid AS uuid, ep.content AS content, ep.event_time AS event_time, " \
                     "ep.source_document AS source_document, ep.activity_type AS activity_type " \
                     "ORDER BY ep.created_at DESC LIMIT 1"
            result = session.run(query, **params)
            record = result.single()
            if record:
                return MemoryCache(
                    absolute_id=record["uuid"],
                    content=record["content"] or "",
                    event_time=_parse_dt(record["event_time"]),
                    source_document=record["source_document"] or "",
                    activity_type=record.get("activity_type"),
                )
        return None

    def get_latest_memory_cache_metadata(self, activity_type: Optional[str] = None) -> Optional[Dict]:
        """获取最新记忆缓存的元数据。"""
        with self._driver.session() as session:
            query = "MATCH (ep:Episode) "
            params: dict = {}
            if activity_type:
                query += "WHERE ep.activity_type = $atype "
                params["atype"] = activity_type
            query += "RETURN ep.uuid AS uuid, ep.event_time AS event_time, " \
                     "ep.source_document AS source_document, ep.created_at AS created_at " \
                     "ORDER BY ep.created_at DESC LIMIT 1"
            result = session.run(query, **params)
            record = result.single()
            if record:
                return {
                    "id": record["uuid"],
                    "event_time": record["event_time"].isoformat() if record["event_time"] else None,
                    "source_document": record["source_document"],
                    "created_at": record["created_at"].isoformat() if record["created_at"] else None,
                }
        return None

    def get_entity_count(self) -> int:
        """返回实体总数。"""
        with self._driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN count(e) AS cnt")
            record = result.single()
            return record["cnt"] if record else 0

    def get_relation_count(self) -> int:
        """返回关系总数。"""
        with self._driver.session() as session:
            result = session.run("MATCH (r:Relation) RETURN count(r) AS cnt")
            record = result.single()
            return record["cnt"] if record else 0

    def search_memory_caches_by_bm25(self, query: str, limit: int = 20) -> List[MemoryCache]:
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
                cache = self.load_memory_cache(cache_id)
            except Exception:
                continue
            if cache is None:
                continue
            content_lower = (cache.content or "").lower()
            if query_lower in content_lower:
                score = content_lower.count(query_lower)
                results.append((score, cache))
        results.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in results[:limit]]

    def delete_memory_cache(self, cache_id: str) -> int:
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
        with self._driver.session() as session:
            result = session.run("MATCH (ep:Episode {uuid: $uuid}) DETACH DELETE ep RETURN count(ep) AS cnt", uuid=cache_id)
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

    def find_cache_by_doc_hash(self, doc_hash: str, document_path: str = "") -> Optional[MemoryCache]:
        """根据 doc_hash 查找记忆缓存。"""
        for meta_file in self._iter_cache_meta_files():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if meta.get("doc_hash") == doc_hash:
                    return MemoryCache(
                        absolute_id=meta.get("absolute_id", ""),
                        content=(meta_file.parent / "cache.md").read_text(encoding="utf-8") if (meta_file.parent / "cache.md").exists() else "",
                        event_time=datetime.fromisoformat(meta["event_time"]) if meta.get("event_time") else datetime.now(),
                        source_document=meta.get("source_document", ""),
                        activity_type=meta.get("activity_type"),
                    )
            except Exception:
                continue
        return None

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
                        "absolute_id": e.absolute_id, "entity_id": e.entity_id,
                        "name": e.name, "content": e.content,
                    }
                    for e in entities
                ],
                "relations": [
                    {
                        "absolute_id": r.absolute_id, "relation_id": r.relation_id,
                        "content": r.content,
                    }
                    for r in relations
                ],
            }
            (doc_dir / "extraction.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return True
        except Exception:
            return False

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
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Embedding 计算
    # ------------------------------------------------------------------

    def _compute_entity_embedding(self, entity: Entity) -> Optional[bytes]:
        """计算实体的 embedding 向量。"""
        if not self.embedding_client or not self.embedding_client.is_available():
            return None
        n = self.entity_content_snippet_length
        text = f"{entity.name} {entity.content[:n]}"
        embedding = self.embedding_client.encode(text)
        if embedding is None or len(embedding) == 0:
            return None
        emb_array = np.array(embedding[0] if isinstance(embedding, list) else embedding, dtype=np.float32)
        return emb_array.tobytes()

    def _compute_relation_embedding(self, relation: Relation) -> Optional[bytes]:
        """计算关系的 embedding 向量。"""
        if not self.embedding_client or not self.embedding_client.is_available():
            return None
        n = self.relation_content_snippet_length
        text = relation.content if n is None or n <= 0 else relation.content[:n]
        embedding = self.embedding_client.encode(text)
        if embedding is None or len(embedding) == 0:
            return None
        emb_array = np.array(embedding[0] if isinstance(embedding, list) else embedding, dtype=np.float32)
        return emb_array.tobytes()

    # ------------------------------------------------------------------
    # Entity 操作
    # ------------------------------------------------------------------

    def save_entity(self, entity: Entity):
        """保存实体到 Neo4j + sqlite-vec。"""
        embedding_blob = self._compute_entity_embedding(entity)
        entity.embedding = embedding_blob

        with self._write_lock:
            with self._driver.session() as session:
                session.run(
                    """
                    MERGE (e:Entity {uuid: $uuid})
                    SET e.entity_id = $entity_id,
                        e.name = $name,
                        e.content = $content,
                        e.event_time = datetime($event_time),
                        e.processed_time = datetime($processed_time),
                        e.memory_cache_id = $cache_id,
                        e.source_document = $source,
                        e.summary = $summary,
                        e.attributes = $attributes,
                        e.confidence = $confidence
                    """,
                    uuid=entity.absolute_id,
                    entity_id=entity.entity_id,
                    name=entity.name,
                    content=entity.content,
                    event_time=entity.event_time.isoformat(),
                    processed_time=entity.processed_time.isoformat(),
                    cache_id=entity.memory_cache_id,
                    source=entity.source_document,
                    summary=entity.summary,
                    attributes=entity.attributes,
                    confidence=entity.confidence,
                )

                # 设置 valid_at
                if entity.valid_at:
                    session.run("""
                        MATCH (e:Entity {uuid: $abs_id})
                        SET e.valid_at = $valid_at
                    """, abs_id=entity.absolute_id, valid_at=entity.valid_at.isoformat())
                else:
                    session.run("""
                        MATCH (e:Entity {uuid: $abs_id})
                        SET e.valid_at = $event_time
                    """, abs_id=entity.absolute_id, event_time=entity.event_time.isoformat())

                # 使旧版本失效
                session.run("""
                    MATCH (e:Entity {entity_id: $entity_id})
                    WHERE e.uuid <> $abs_id AND e.invalid_at IS NULL
                    SET e.invalid_at = $event_time
                """, entity_id=entity.entity_id, abs_id=entity.absolute_id, event_time=entity.event_time.isoformat())

            # 存储向量
            if embedding_blob:
                emb_list = list(np.frombuffer(embedding_blob, dtype=np.float32))
                self._vector_store.upsert("entity_vectors", entity.absolute_id, emb_list)

        self._cache.invalidate()

    def bulk_save_entities(self, entities: List[Entity]):
        """批量保存实体。"""
        if not entities:
            return

        # 批量计算 embedding
        embeddings = None
        if self.embedding_client and self.embedding_client.is_available():
            texts = [
                f"{e.name} {e.content[:self.entity_content_snippet_length]}"
                for e in entities
            ]
            embeddings = self.embedding_client.encode(texts)

        with self._write_lock:
            with self._driver.session() as session:
                vec_items = []
                for idx, entity in enumerate(entities):
                    embedding_blob = None
                    if embeddings is not None:
                        try:
                            embedding_blob = np.array(embeddings[idx], dtype=np.float32).tobytes()
                        except Exception:
                            embedding_blob = None
                    entity.embedding = embedding_blob

                    session.run(
                        """
                        MERGE (e:Entity {uuid: $uuid})
                        SET e.entity_id = $entity_id,
                            e.name = $name,
                            e.content = $content,
                            e.event_time = datetime($event_time),
                            e.processed_time = datetime($processed_time),
                            e.memory_cache_id = $cache_id,
                            e.source_document = $source,
                            e.summary = $summary,
                            e.attributes = $attributes,
                            e.confidence = $confidence
                        """,
                        uuid=entity.absolute_id,
                        entity_id=entity.entity_id,
                        name=entity.name,
                        content=entity.content,
                        event_time=entity.event_time.isoformat(),
                        processed_time=entity.processed_time.isoformat(),
                        cache_id=entity.memory_cache_id,
                        source=entity.source_document,
                        summary=entity.summary,
                        attributes=entity.attributes,
                        confidence=entity.confidence,
                    )

                    if embedding_blob:
                        emb_list = list(np.frombuffer(embedding_blob, dtype=np.float32))
                        vec_items.append((entity.absolute_id, emb_list))

                if vec_items:
                    self._vector_store.upsert_batch("entity_vectors", vec_items)

        self._cache.invalidate()

    def get_entity_by_entity_id(self, entity_id: str) -> Optional[Entity]:
        """根据 entity_id 获取最新版本的实体。"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return None
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {entity_id: $eid})
                RETURN e.uuid AS uuid, e.entity_id AS entity_id, e.name AS name,
                       e.content AS content, e.event_time AS event_time,
                       e.processed_time AS processed_time, e.memory_cache_id AS memory_cache_id,
                       e.source_document AS source_document
                ORDER BY e.processed_time DESC LIMIT 1
                """,
                eid=entity_id,
            )
            record = result.single()
            if not record:
                return None
            entity = _neo4j_record_to_entity(record)
            # 从 sqlite-vec 获取 embedding
            emb = self._vector_store.get("entity_vectors", entity.absolute_id)
            if emb:
                entity.embedding = np.array(emb, dtype=np.float32).tobytes()
            return entity

    # 别名兼容
    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """get_entity_by_entity_id 的别名。"""
        return self.get_entity_by_entity_id(entity_id)

    def get_entity_by_absolute_id(self, absolute_id: str) -> Optional[Entity]:
        """根据 absolute_id 获取实体。"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {uuid: $uuid})
                RETURN e.uuid AS uuid, e.entity_id AS entity_id, e.name AS name,
                       e.content AS content, e.event_time AS event_time,
                       e.processed_time AS processed_time, e.memory_cache_id AS memory_cache_id,
                       e.source_document AS source_document
                """,
                uuid=absolute_id,
            )
            record = result.single()
            if not record:
                return None
            entity = _neo4j_record_to_entity(record)
            emb = self._vector_store.get("entity_vectors", entity.absolute_id)
            if emb:
                entity.embedding = np.array(emb, dtype=np.float32).tobytes()
            return entity

    def get_entity_names_by_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, str]:
        """批量根据 absolute_id 查询实体名称。"""
        if not absolute_ids:
            return {}
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.uuid IN $uuids
                RETURN e.uuid AS uuid, e.name AS name
                """,
                uuids=absolute_ids,
            )
            return {record["uuid"]: record["name"] for record in result}

    def get_entity_version_at_time(self, entity_id: str, time_point: datetime) -> Optional[Entity]:
        """获取实体在指定时间点的版本。"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return None
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {entity_id: $eid})
                WHERE e.event_time <= datetime($tp)
                RETURN e.uuid AS uuid, e.entity_id AS entity_id, e.name AS name,
                       e.content AS content, e.event_time AS event_time,
                       e.processed_time AS processed_time, e.memory_cache_id AS memory_cache_id,
                       e.source_document AS source_document
                ORDER BY e.processed_time DESC LIMIT 1
                """,
                eid=entity_id,
                tp=time_point.isoformat(),
            )
            record = result.single()
            if not record:
                return None
            entity = _neo4j_record_to_entity(record)
            emb = self._vector_store.get("entity_vectors", entity.absolute_id)
            if emb:
                entity.embedding = np.array(emb, dtype=np.float32).tobytes()
            return entity

    def get_entity_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        """获取实体 embedding 预览。"""
        emb = self._vector_store.get("entity_vectors", absolute_id)
        if emb:
            return emb[:num_values]
        return None

    def get_relation_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        """获取关系 embedding 预览。"""
        emb = self._vector_store.get("relation_vectors", absolute_id)
        if emb:
            return emb[:num_values]
        return None

    def get_entity_versions(self, entity_id: str) -> List[Entity]:
        """获取实体的所有版本。"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return []
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {entity_id: $eid})
                RETURN e.uuid AS uuid, e.entity_id AS entity_id, e.name AS name,
                       e.content AS content, e.event_time AS event_time,
                       e.processed_time AS processed_time, e.memory_cache_id AS memory_cache_id,
                       e.source_document AS source_document
                ORDER BY e.processed_time ASC
                """,
                eid=entity_id,
            )
            entities = []
            for record in result:
                entities.append(_neo4j_record_to_entity(record))
            return entities

    def _get_entities_with_embeddings(self) -> List[tuple]:
        """获取所有实体的最新版本及其 embedding。"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WITH e.entity_id AS eid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH eid, e ORDER BY e.processed_time DESC
                WITH eid, HEAD(COLLECT(e)) AS e
                RETURN e.uuid AS uuid, e.entity_id AS entity_id, e.name AS name,
                       e.content AS content, e.event_time AS event_time,
                       e.processed_time AS processed_time, e.memory_cache_id AS memory_cache_id,
                       e.source_document AS source_document
                ORDER BY e.processed_time DESC
                """
            )
            entities = []
            for record in result:
                entity = _neo4j_record_to_entity(record)
                emb_list = self._vector_store.get("entity_vectors", entity.absolute_id)
                emb_array = np.array(emb_list, dtype=np.float32) if emb_list else None
                entities.append((entity, emb_array))
            return entities

    def get_latest_entities_projection(self, content_snippet_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取最新实体投影。"""
        snippet_length = content_snippet_length or self.entity_content_snippet_length
        entities_with_emb = self._get_entities_with_embeddings()
        version_counts = self.get_entity_version_counts([
            e.entity_id for e, _ in entities_with_emb
        ])
        results: List[Dict[str, Any]] = []
        for entity, embedding_array in entities_with_emb:
            results.append({
                "entity": entity,
                "entity_id": entity.entity_id,
                "name": entity.name,
                "content": entity.content,
                "content_snippet": entity.content[:snippet_length],
                "version_count": version_counts.get(entity.entity_id, 1),
                "embedding_array": embedding_array,
            })
        return results

    def get_all_entities(self, limit: Optional[int] = None, exclude_embedding: bool = False) -> List[Entity]:
        """获取所有实体的最新版本。"""
        with self._driver.session() as session:
            query = """
                MATCH (e:Entity)
                WITH e.entity_id AS eid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH eid, e ORDER BY e.processed_time DESC
                WITH eid, HEAD(COLLECT(e)) AS e
                RETURN e.uuid AS uuid, e.entity_id AS entity_id, e.name AS name,
                       e.content AS content, e.event_time AS event_time,
                       e.processed_time AS processed_time, e.memory_cache_id AS memory_cache_id,
                       e.source_document AS source_document
                ORDER BY e.processed_time DESC
            """
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = session.run(query)
            entities = []
            for record in result:
                entity = _neo4j_record_to_entity(record)
                if not exclude_embedding:
                    emb = self._vector_store.get("entity_vectors", entity.absolute_id)
                    if emb:
                        entity.embedding = np.array(emb, dtype=np.float32).tobytes()
                entities.append(entity)
            return entities

    def count_unique_entities(self) -> int:
        """统计不重复的 entity_id 数量。"""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) RETURN COUNT(DISTINCT e.entity_id) AS cnt"
            )
            record = result.single()
            return record["cnt"] if record else 0

    def count_unique_relations(self) -> int:
        """统计不重复的 relation_id 数量。"""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (r:Relation) RETURN COUNT(DISTINCT r.relation_id) AS cnt"
            )
            record = result.single()
            return record["cnt"] if record else 0

    def get_all_entities_before_time(self, time_point: datetime, limit: Optional[int] = None,
                                      exclude_embedding: bool = False) -> List[Entity]:
        """获取指定时间点之前的所有实体最新版本。"""
        with self._driver.session() as session:
            query = """
                MATCH (e:Entity)
                WHERE e.event_time <= datetime($tp)
                WITH e.entity_id AS eid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH eid, e ORDER BY e.processed_time DESC
                WITH eid, HEAD(COLLECT(e)) AS e
                RETURN e.uuid AS uuid, e.entity_id AS entity_id, e.name AS name,
                       e.content AS content, e.event_time AS event_time,
                       e.processed_time AS processed_time, e.memory_cache_id AS memory_cache_id,
                       e.source_document AS source_document
                ORDER BY e.processed_time DESC
            """
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = session.run(query, tp=time_point.isoformat())
            entities = []
            for record in result:
                entity = _neo4j_record_to_entity(record)
                if not exclude_embedding:
                    emb = self._vector_store.get("entity_vectors", entity.absolute_id)
                    if emb:
                        entity.embedding = np.array(emb, dtype=np.float32).tobytes()
                entities.append(entity)
            return entities

    def get_entity_version_count(self, entity_id: str) -> int:
        """获取指定 entity_id 的版本数量。"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return 0
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity {entity_id: $eid}) RETURN COUNT(e) AS cnt",
                eid=entity_id,
            )
            record = result.single()
            return record["cnt"] if record else 0

    def get_entity_version_counts(self, entity_ids: List[str]) -> Dict[str, int]:
        """批量获取多个 entity_id 的版本数量。"""
        if not entity_ids:
            return {}
        canonical_ids = []
        for eid in entity_ids:
            canonical = self.resolve_entity_id(eid)
            if canonical and canonical not in canonical_ids:
                canonical_ids.append(canonical)
        if not canonical_ids:
            return {}
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.entity_id IN $eids
                RETURN e.entity_id AS entity_id, COUNT(e) AS cnt
                """,
                eids=canonical_ids,
            )
            return {record["entity_id"]: record["cnt"] for record in result}

    def get_graph_statistics(self) -> Dict[str, Any]:
        """返回图谱结构统计数据"""
        cached = self._cache.get("graph_stats")
        if cached is not None:
            return cached
        with self._driver.session() as session:
            # 基础计数
            r = session.run("MATCH (e:Entity) RETURN count(e) AS cnt")
            entity_count = r.single()["cnt"]

            r = session.run("MATCH (r:Relation) RETURN count(r) AS cnt")
            relation_count = r.single()["cnt"]

            stats = {
                "entity_count": entity_count,
                "relation_count": relation_count,
            }

            if entity_count > 0:
                # 平均度数
                r = session.run("""
                    MATCH (e:Entity)-[]-(r:Relation)
                    WITH e, count(DISTINCT r) AS deg
                    RETURN avg(deg) AS avg_deg
                """)
                row = r.single()
                stats["avg_relations_per_entity"] = round(row["avg_deg"], 2) if row and row["avg_deg"] else 0

                # 最大度数
                r = session.run("""
                    MATCH (e:Entity)-[]-(r:Relation)
                    WITH e, count(DISTINCT r) AS deg
                    RETURN max(deg) AS max_deg
                """)
                row = r.single()
                stats["max_relations_per_entity"] = row["max_deg"] if row and row["max_deg"] else 0

                # 孤立实体
                r = session.run("""
                    MATCH (e:Entity)
                    WHERE NOT (e)--()
                    RETURN count(e) AS cnt
                """)
                stats["isolated_entities"] = r.single()["cnt"]

                # 图密度
                r = session.run("MATCH (e:Entity) RETURN count(DISTINCT e.entity_id) AS cnt")
                unique_entities = r.single()["cnt"]
                if unique_entities > 1:
                    max_possible = unique_entities * (unique_entities - 1) / 2
                    r = session.run("MATCH (r:Relation) RETURN count(DISTINCT r.relation_id) AS cnt")
                    unique_relations = r.single()["cnt"]
                    stats["graph_density"] = round(unique_relations / max_possible, 4)
                else:
                    stats["graph_density"] = 0.0
            else:
                stats.update({
                    "avg_relations_per_entity": 0,
                    "max_relations_per_entity": 0,
                    "isolated_entities": entity_count,
                    "graph_density": 0.0,
                })

            # 时间趋势
            r = session.run("""
                MATCH (e:Entity)
                WITH e.event_time AS t
                WHERE t IS NOT NULL
                WITH date(t) AS d
                RETURN d AS date, count(*) AS cnt
                ORDER BY d
                LIMIT 30
            """)
            stats["entity_count_over_time"] = [{"date": str(r["date"]), "count": r["cnt"]} for r in r]

            r = session.run("""
                MATCH (r:Relation)
                WITH r.event_time AS t
                WHERE t IS NOT NULL
                WITH date(t) AS d
                RETURN d AS date, count(*) AS cnt
                ORDER BY d
                LIMIT 30
            """)
            stats["relation_count_over_time"] = [{"date": str(r["date"]), "count": r["cnt"]} for r in r]

        self._cache.set("graph_stats", stats, ttl=60)
        return stats

    def entity_has_any_relation(self, entity_id: str) -> bool:
        """检查实体是否有关系。"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return False
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (r:Relation)
                WHERE r.entity1_absolute_id IN (
                    MATCH (e:Entity {entity_id: $eid}) RETURN e.uuid
                ) OR r.entity2_absolute_id IN (
                    MATCH (e:Entity {entity_id: $eid}) RETURN e.uuid
                )
                RETURN COUNT(r) AS cnt LIMIT 1
                """,
                eid=entity_id,
            )
            record = result.single()
            return (record["cnt"] or 0) > 0

    def delete_orphan_entities(self, candidate_entity_ids: list) -> list:
        """批量检查并删除没有关系的实体。"""
        if not candidate_entity_ids:
            return []
        with self._write_lock:
            with self._driver.session() as session:
                # 找出候选中无关系的 entity_id
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.entity_id IN $eids
                    AND NOT (
                        (e)-[:RELATES_TO]-(:Entity)
                    )
                    RETURN DISTINCT e.entity_id AS eid
                    """,
                    eids=candidate_entity_ids,
                )
                orphan_ids = [record["eid"] for record in result]
                if orphan_ids:
                    session.run(
                        """
                        MATCH (e:Entity)
                        WHERE e.entity_id IN $eids
                        DETACH DELETE e
                        """,
                        eids=orphan_ids,
                    )
                    # 同步删除向量
                    self._vector_store.delete_batch("entity_vectors", orphan_ids)
                return orphan_ids

    def delete_entity_by_id(self, entity_id: str) -> int:
        """删除实体的所有版本。"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return 0
        with self._write_lock:
            with self._driver.session() as session:
                # 获取所有 absolute_id
                result = session.run(
                    "MATCH (e:Entity {entity_id: $eid}) RETURN e.uuid AS uuid",
                    eid=entity_id,
                )
                uuids = [record["uuid"] for record in result]
                count = len(uuids)
                if uuids:
                    session.run(
                        """
                        MATCH (e:Entity {entity_id: $eid})
                        DETACH DELETE e
                        """,
                        eid=entity_id,
                    )
                    self._vector_store.delete_batch("entity_vectors", uuids)
                self._cache.invalidate()
                return count

    def delete_relation_by_id(self, relation_id: str) -> int:
        """删除关系的所有版本。返回删除的行数。"""
        with self._write_lock:
            with self._driver.session() as session:
                # 删除关系节点
                result = session.run(
                    "MATCH (r:Relation {relation_id: $rid}) DETACH DELETE r RETURN count(r) AS cnt",
                    rid=relation_id,
                )
                record = result.single()
                count = record["cnt"] if record else 0
                # 清理向量存储
                try:
                    self._vector_store.delete_batch("relation_vectors",
                        [r.absolute_id for r in self.get_relation_versions(relation_id)])
                except Exception:
                    pass
                self._cache.invalidate()
                return count

    def delete_entity_all_versions(self, entity_id: str) -> int:
        """删除实体的所有版本（含关系边）。返回删除的行数。"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return 0
        with self._write_lock:
            with self._driver.session() as session:
                # 删除相关关系
                session.run(
                    """MATCH (e:Entity {entity_id: $eid})-[r:RELATES_TO]-()
                       DETACH DELETE r""",
                    eid=entity_id,
                )
                # 删除实体节点
                result = session.run(
                    "MATCH (e:Entity {entity_id: $eid}) DETACH DELETE e RETURN count(e) AS cnt",
                    eid=entity_id,
                )
                record = result.single()
                count = record["cnt"] if record else 0
                # 清理向量存储
                try:
                    self._vector_store.delete_batch("entity_vectors",
                        [e.absolute_id for e in self.get_entity_versions(entity_id)])
                except Exception:
                    pass
                self._cache.invalidate()
                return count

    def delete_relation_all_versions(self, relation_id: str) -> int:
        """删除关系的所有版本。返回删除的行数。"""
        return self.delete_relation_by_id(relation_id)

    def get_entity_ids_by_names(self, names: list) -> dict:
        """按名称批量查询 entity_id。"""
        if not names:
            return {}
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.name IN $names
                RETURN e.name AS name, e.entity_id AS entity_id
                ORDER BY e.processed_time DESC
                """,
                names=names,
            )
            output = {}
            for record in result:
                name = record["name"]
                if name not in output:
                    output[name] = self.resolve_entity_id(record["entity_id"])
            return output

    def get_total_entity_count(self) -> int:
        """获取不重复 entity_id 数量。"""
        try:
            return self.count_unique_entities()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # BM25 Full-Text Search
    # ------------------------------------------------------------------

    def search_entities_by_bm25(self, query: str, limit: int = 20) -> List[Entity]:
        """BM25 全文搜索实体（Neo4j 5.x 全文索引）。"""
        if not query:
            return []
        try:
            with self._driver.session() as session:
                result = session.run(
                    """CALL db.index.fulltext.queryNodes('entityFulltext', $query)
                       YIELD node, score
                       RETURN node.uuid AS uuid, node.entity_id AS entity_id,
                              node.name AS name, node.content AS content,
                              node.event_time AS event_time,
                              node.processed_time AS processed_time,
                              node.memory_cache_id AS memory_cache_id,
                              node.source_document AS source_document,
                              score AS bm25_score
                       ORDER BY score DESC
                       LIMIT $limit""",
                    query=query, limit=limit,
                )
                entities = []
                for record in result:
                    entities.append(_neo4j_record_to_entity(record))
                return entities
        except Exception as e:
            logger.warning("BM25 search failed, falling back to empty: %s", e)
            return []

    def search_relations_by_bm25(self, query: str, limit: int = 20) -> List[Relation]:
        """BM25 全文搜索关系（Neo4j 5.x 全文索引）。"""
        if not query:
            return []
        try:
            with self._driver.session() as session:
                result = session.run(
                    """CALL db.index.fulltext.queryNodes('relationFulltext', $query)
                       YIELD node, score
                       RETURN node.uuid AS uuid, node.relation_id AS relation_id,
                              node.entity1_absolute_id AS entity1_absolute_id,
                              node.entity2_absolute_id AS entity2_absolute_id,
                              node.content AS content,
                              node.event_time AS event_time,
                              node.processed_time AS processed_time,
                              node.memory_cache_id AS memory_cache_id,
                              node.source_document AS source_document,
                              score AS bm25_score
                       ORDER BY score DESC
                       LIMIT $limit""",
                    query=query, limit=limit,
                )
                relations = []
                for record in result:
                    relations.append(_neo4j_record_to_relation(record))
                return relations
        except Exception as e:
            logger.warning("BM25 search failed, falling back to empty: %s", e)
            return []

    # ------------------------------------------------------------------
    # Entity Search
    # ------------------------------------------------------------------

    def search_entities_by_similarity(self, query_name: str, query_content: Optional[str] = None,
                                       threshold: float = 0.7, max_results: int = 10,
                                       content_snippet_length: int = 50,
                                       text_mode: Literal["name_only", "content_only", "name_and_content"] = "name_and_content",
                                       similarity_method: Literal["embedding", "text", "jaccard", "bleu"] = "embedding") -> List[Entity]:
        """根据相似度搜索实体。"""
        entities_with_embeddings = self._get_entities_with_embeddings()
        if not entities_with_embeddings:
            return []

        all_entities = [e for e, _ in entities_with_embeddings]

        if text_mode == "name_only":
            query_text = query_name
            use_content = False
        elif text_mode == "content_only":
            if not query_content:
                return []
            query_text = query_content[:content_snippet_length]
            use_content = True
        else:
            if query_content:
                query_text = f"{query_name} {query_content[:content_snippet_length]}"
            else:
                query_text = query_name
            use_content = query_content is not None

        if similarity_method == "embedding" and self.embedding_client and self.embedding_client.is_available():
            return self._search_with_embedding(
                query_text, entities_with_embeddings, threshold,
                use_content, max_results, content_snippet_length, text_mode
            )
        else:
            return self._search_with_text_similarity(
                query_text, all_entities, threshold,
                use_content, max_results, content_snippet_length,
                text_mode, similarity_method
            )

    def _search_with_embedding(self, query_text: str, entities_with_embeddings: List[tuple],
                                threshold: float, use_content: bool = False,
                                max_results: int = 10, content_snippet_length: int = 50,
                                text_mode: str = "name_and_content") -> List[Entity]:
        """使用 embedding 向量进行相似度搜索。"""
        query_embedding = self.embedding_client.encode(query_text)
        if query_embedding is None:
            all_entities = [e for e, _ in entities_with_embeddings]
            return self._search_with_text_similarity(
                query_text, all_entities, threshold, use_content, max_results, content_snippet_length, text_mode, "text"
            )

        query_emb_array = np.array(
            query_embedding[0] if isinstance(query_embedding, (list, np.ndarray)) else query_embedding,
            dtype=np.float32
        )

        stored_embeddings = []
        entities_to_encode = []
        entity_indices = []

        for idx, (entity, stored_emb) in enumerate(entities_with_embeddings):
            if stored_emb is not None:
                stored_embeddings.append((idx, stored_emb))
            else:
                entities_to_encode.append(entity)
                entity_indices.append(idx)

        if entities_to_encode:
            entity_texts = []
            for entity in entities_to_encode:
                if text_mode == "name_only":
                    entity_texts.append(entity.name)
                elif text_mode == "content_only":
                    entity_texts.append(entity.content[:content_snippet_length])
                else:
                    entity_texts.append(
                        f"{entity.name} {entity.content[:content_snippet_length]}" if use_content else entity.name
                    )
            new_embs = self.embedding_client.encode(entity_texts)
            if new_embs is not None:
                for i, entity in enumerate(entities_to_encode):
                    emb_arr = np.array(new_embs[i] if isinstance(new_embs, (list, np.ndarray)) else new_embs, dtype=np.float32)
                    stored_embeddings.append((entity_indices[i], emb_arr))

        if not stored_embeddings:
            all_entities = [e for e, _ in entities_with_embeddings]
            return self._search_with_text_similarity(
                query_text, all_entities, threshold, use_content, max_results, content_snippet_length, text_mode, "text"
            )

        similarities = []
        for idx, stored_emb in stored_embeddings:
            dot = np.dot(query_emb_array, stored_emb)
            norm_q = np.linalg.norm(query_emb_array)
            norm_s = np.linalg.norm(stored_emb)
            sim = float(dot / (norm_q * norm_s + 1e-9))
            entity = entities_with_embeddings[idx][0]
            similarities.append((entity, sim))

        scored = [(e, s) for e, s in similarities if s >= threshold]
        scored.sort(key=lambda x: x[1], reverse=True)

        entities = []
        seen = set()
        for entity, _ in scored:
            if entity.entity_id not in seen:
                entities.append(entity)
                seen.add(entity.entity_id)
                if len(entities) >= max_results:
                    break
        return entities

    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """计算 Jaccard 相似度。"""
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _calculate_bleu_similarity(self, text1: str, text2: str) -> float:
        """计算 BLEU 相似度。"""
        def get_char_ngrams(text, n):
            return [text[i:i + n] for i in range(len(text) - n + 1)]

        ngrams1_1 = set(get_char_ngrams(text1.lower(), 1))
        ngrams2_1 = set(get_char_ngrams(text2.lower(), 1))
        ngrams1_2 = set(get_char_ngrams(text1.lower(), 2))
        ngrams2_2 = set(get_char_ngrams(text2.lower(), 2))

        precision_1 = len(ngrams1_1 & ngrams2_1) / max(len(ngrams1_1), 1)
        precision_2 = len(ngrams1_2 & ngrams2_2) / max(len(ngrams1_2), 1)
        return (precision_1 + precision_2) / 2

    def _search_with_text_similarity(self, query_text: str, all_entities: List[Entity],
                                      threshold: float, use_content: bool = False,
                                      max_results: int = 10, content_snippet_length: int = 50,
                                      text_mode: str = "name_and_content",
                                      similarity_method: str = "text") -> List[Entity]:
        """使用文本相似度搜索实体。"""
        scored = []
        for entity in all_entities:
            if text_mode == "name_only":
                target = entity.name
            elif text_mode == "content_only":
                target = entity.content[:content_snippet_length]
            else:
                target = f"{entity.name} {entity.content[:content_snippet_length]}" if use_content else entity.name

            if similarity_method == "jaccard":
                sim = self._calculate_jaccard_similarity(query_text, target)
            elif similarity_method == "bleu":
                sim = self._calculate_bleu_similarity(query_text, target)
            else:
                sim = difflib.SequenceMatcher(None, query_text.lower(), target.lower()).ratio()

            if sim >= threshold:
                scored.append((entity, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        entities = []
        seen = set()
        for entity, _ in scored:
            if entity.entity_id not in seen:
                entities.append(entity)
                seen.add(entity.entity_id)
                if len(entities) >= max_results:
                    break
        return entities

    # ------------------------------------------------------------------
    # Relation 操作
    # ------------------------------------------------------------------

    def save_relation(self, relation: Relation):
        """保存关系到 Neo4j + sqlite-vec。"""
        embedding_blob = self._compute_relation_embedding(relation)
        relation.embedding = embedding_blob

        with self._write_lock:
            with self._driver.session() as session:
                session.run(
                    """
                    MERGE (r:Relation {uuid: $uuid})
                    SET r.relation_id = $relation_id,
                        r.entity1_absolute_id = $e1_abs,
                        r.entity2_absolute_id = $e2_abs,
                        r.content = $content,
                        r.event_time = datetime($event_time),
                        r.processed_time = datetime($processed_time),
                        r.memory_cache_id = $cache_id,
                        r.source_document = $source,
                        r.summary = $summary,
                        r.attributes = $attributes,
                        r.confidence = $confidence,
                        r.provenance = $provenance
                    """,
                    uuid=relation.absolute_id,
                    relation_id=relation.relation_id,
                    e1_abs=relation.entity1_absolute_id,
                    e2_abs=relation.entity2_absolute_id,
                    content=relation.content,
                    event_time=relation.event_time.isoformat(),
                    processed_time=relation.processed_time.isoformat(),
                    cache_id=relation.memory_cache_id,
                    source=relation.source_document,
                    summary=relation.summary,
                    attributes=relation.attributes,
                    confidence=relation.confidence,
                    provenance=relation.provenance,
                )

                # 设置 valid_at
                if relation.valid_at:
                    session.run("""
                        MATCH (r:Relation {uuid: $abs_id})
                        SET r.valid_at = $valid_at
                    """, abs_id=relation.absolute_id, valid_at=relation.valid_at.isoformat())
                else:
                    session.run("""
                        MATCH (r:Relation {uuid: $abs_id})
                        SET r.valid_at = $event_time
                    """, abs_id=relation.absolute_id, event_time=relation.event_time.isoformat())

                # 使旧版本失效
                session.run("""
                    MATCH (r:Relation {relation_id: $relation_id})
                    WHERE r.uuid <> $abs_id AND r.invalid_at IS NULL
                    SET r.invalid_at = $event_time
                """, relation_id=relation.relation_id, abs_id=relation.absolute_id, event_time=relation.event_time.isoformat())

                # 创建 RELATES_TO 边
                session.run(
                    """
                    MATCH (e1:Entity {uuid: $e1_abs})
                    MATCH (e2:Entity {uuid: $e2_abs})
                    MERGE (e1)-[rel:RELATES_TO {relation_uuid: $uuid}]->(e2)
                    SET rel.fact = $content
                    """,
                    e1_abs=relation.entity1_absolute_id,
                    e2_abs=relation.entity2_absolute_id,
                    uuid=relation.absolute_id,
                    content=relation.content,
                )

            if embedding_blob:
                emb_list = list(np.frombuffer(embedding_blob, dtype=np.float32))
                self._vector_store.upsert("relation_vectors", relation.absolute_id, emb_list)

        self._cache.invalidate()

    def bulk_save_relations(self, relations: List[Relation]):
        """批量保存关系。"""
        if not relations:
            return

        embeddings = None
        if self.embedding_client and self.embedding_client.is_available():
            _n = self.relation_content_snippet_length
            texts = [
                r.content if _n is None or _n <= 0 else r.content[:_n]
                for r in relations
            ]
            embeddings = self.embedding_client.encode(texts)

        with self._write_lock:
            with self._driver.session() as session:
                vec_items = []
                for idx, relation in enumerate(relations):
                    embedding_blob = None
                    if embeddings is not None:
                        try:
                            embedding_blob = np.array(embeddings[idx], dtype=np.float32).tobytes()
                        except Exception:
                            embedding_blob = None
                    relation.embedding = embedding_blob

                    session.run(
                        """
                        MERGE (r:Relation {uuid: $uuid})
                        SET r.relation_id = $relation_id,
                            r.entity1_absolute_id = $e1_abs,
                            r.entity2_absolute_id = $e2_abs,
                            r.content = $content,
                            r.event_time = datetime($event_time),
                            r.processed_time = datetime($processed_time),
                            r.memory_cache_id = $cache_id,
                            r.source_document = $source
                        """,
                        uuid=relation.absolute_id,
                        relation_id=relation.relation_id,
                        e1_abs=relation.entity1_absolute_id,
                        e2_abs=relation.entity2_absolute_id,
                        content=relation.content,
                        event_time=relation.event_time.isoformat(),
                        processed_time=relation.processed_time.isoformat(),
                        cache_id=relation.memory_cache_id,
                        source=relation.source_document,
                    )

                    # 创建 RELATES_TO 边
                    session.run(
                        """
                        MATCH (e1:Entity {uuid: $e1_abs})
                        MATCH (e2:Entity {uuid: $e2_abs})
                        MERGE (e1)-[rel:RELATES_TO {relation_uuid: $uuid}]->(e2)
                        SET rel.fact = $content
                        """,
                        e1_abs=relation.entity1_absolute_id,
                        e2_abs=relation.entity2_absolute_id,
                        uuid=relation.absolute_id,
                        content=relation.content,
                    )

                    if embedding_blob:
                        emb_list = list(np.frombuffer(embedding_blob, dtype=np.float32))
                        vec_items.append((relation.absolute_id, emb_list))

                if vec_items:
                    self._vector_store.upsert_batch("relation_vectors", vec_items)

        self._cache.invalidate()

    def get_relation_by_absolute_id(self, relation_absolute_id: str) -> Optional[Relation]:
        """根据 absolute_id 获取关系。"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (r:Relation {uuid: $uuid})
                RETURN r.uuid AS uuid, r.relation_id AS relation_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                       r.source_document AS source_document
                """,
                uuid=relation_absolute_id,
            )
            record = result.single()
            if not record:
                return None
            relation = _neo4j_record_to_relation(record)
            emb = self._vector_store.get("relation_vectors", relation.absolute_id)
            if emb:
                relation.embedding = np.array(emb, dtype=np.float32).tobytes()
            return relation

    def get_relation_by_relation_id(self, relation_id: str) -> Optional[Relation]:
        """根据 relation_id 获取最新版本的关系。"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (r:Relation {relation_id: $rid})
                RETURN r.uuid AS uuid, r.relation_id AS relation_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                       r.source_document AS source_document
                ORDER BY r.processed_time DESC LIMIT 1
                """,
                rid=relation_id,
            )
            record = result.single()
            if not record:
                return None
            relation = _neo4j_record_to_relation(record)
            emb = self._vector_store.get("relation_vectors", relation.absolute_id)
            if emb:
                relation.embedding = np.array(emb, dtype=np.float32).tobytes()
            return relation

    def get_relations_by_entities(self, from_entity_id: str, to_entity_id: str) -> List[Relation]:
        """根据两个 entity_id 获取所有关系。"""
        from_entity_id = self.resolve_entity_id(from_entity_id)
        to_entity_id = self.resolve_entity_id(to_entity_id)
        if not from_entity_id or not to_entity_id:
            return []

        from_entity = self.get_entity_by_entity_id(from_entity_id)
        to_entity = self.get_entity_by_entity_id(to_entity_id)
        if not from_entity or not to_entity:
            return []

        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id IN $from_ids AND r.entity2_absolute_id IN $to_ids)
                   OR (r.entity1_absolute_id IN $to_ids AND r.entity2_absolute_id IN $from_ids)
                WITH r.relation_id AS rid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH rid, r ORDER BY r.processed_time DESC
                WITH rid, HEAD(COLLECT(r)) AS r
                RETURN r.uuid AS uuid, r.relation_id AS relation_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                       r.source_document AS source_document
                ORDER BY r.processed_time DESC
                """,
                from_ids=self._get_all_absolute_ids_for_entity(from_entity_id),
                to_ids=self._get_all_absolute_ids_for_entity(to_entity_id),
            )
            return [_neo4j_record_to_relation(r) for r in result]

    def get_relations_by_entity_pairs(self, entity_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], List[Relation]]:
        """批量获取多个实体对的关系。"""
        results: Dict[Tuple[str, str], List[Relation]] = {}
        for e1, e2 in entity_pairs:
            pair_key = tuple(sorted((e1, e2)))
            if pair_key not in results:
                results[pair_key] = self.get_relations_by_entities(pair_key[0], pair_key[1])
        return results

    def _get_all_absolute_ids_for_entity(self, entity_id: str) -> List[str]:
        """获取实体的所有版本的 absolute_id。"""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity {entity_id: $eid}) RETURN e.uuid AS uuid",
                eid=entity_id,
            )
            return [record["uuid"] for record in result]

    def get_latest_relations_projection(self, content_snippet_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取最新关系投影。"""
        snippet_length = (
            self.relation_content_snippet_length
            if content_snippet_length is None
            else content_snippet_length
        )
        relations_with_emb = self._get_relations_with_embeddings()
        results: List[Dict[str, Any]] = []
        for relation, emb_array in relations_with_emb:
            _csnip = relation.content if snippet_length is None or snippet_length <= 0 else relation.content[:snippet_length]
            results.append({
                "relation": relation,
                "relation_id": relation.relation_id,
                "pair": tuple(sorted((relation.entity1_absolute_id, relation.entity2_absolute_id))),
                "content": relation.content,
                "content_snippet": _csnip,
                "embedding_array": emb_array,
            })
        return results

    def get_relation_versions(self, relation_id: str) -> List[Relation]:
        """获取关系的所有版本。"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (r:Relation {relation_id: $rid})
                RETURN r.uuid AS uuid, r.relation_id AS relation_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                       r.source_document AS source_document
                ORDER BY r.processed_time ASC
                """,
                rid=relation_id,
            )
            return [_neo4j_record_to_relation(r) for r in result]

    def update_relation_memory_cache_id(self, relation_id: str, memory_cache_id: str):
        """更新关系的 memory_cache_id。"""
        with self._write_lock:
            with self._driver.session() as session:
                session.run(
                    """
                    MATCH (r:Relation {relation_id: $rid})
                    SET r.memory_cache_id = $cache_id
                    """,
                    rid=relation_id,
                    cache_id=memory_cache_id,
                )

    def get_self_referential_relations(self) -> Dict[str, List[Dict]]:
        """获取自引用关系。"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (r:Relation)
                WHERE r.entity1_absolute_id = r.entity2_absolute_id
                RETURN r.uuid AS uuid, r.relation_id AS relation_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.content AS content
                """
            )
            grouped: Dict[str, List[Dict]] = {}
            for record in result:
                eid = record["entity1_absolute_id"]
                grouped.setdefault(eid, []).append({
                    "uuid": record["uuid"],
                    "relation_id": record["relation_id"],
                    "content": record["content"],
                })
            return grouped

    def delete_self_referential_relations(self) -> int:
        """删除所有自引用关系。"""
        with self._write_lock:
            with self._driver.session() as session:
                # 先获取要删除的 uuid
                result = session.run(
                    "MATCH (r:Relation) WHERE r.entity1_absolute_id = r.entity2_absolute_id RETURN r.uuid AS uuid"
                )
                uuids = [record["uuid"] for record in result]
                count = len(uuids)
                if uuids:
                    session.run(
                        """
                        MATCH (r:Relation)
                        WHERE r.uuid IN $uuids
                        DETACH DELETE r
                        """,
                        uuids=uuids,
                    )
                    self._vector_store.delete_batch("relation_vectors", uuids)
                return count

    def get_self_referential_relations_for_entity(self, entity_id: str) -> List[Dict]:
        """获取指定实体的自引用关系。"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return []
        abs_ids = self._get_all_absolute_ids_for_entity(entity_id)
        if not abs_ids:
            return []
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (r:Relation)
                WHERE r.entity1_absolute_id IN $abs_ids AND r.entity2_absolute_id IN $abs_ids
                AND r.entity1_absolute_id = r.entity2_absolute_id
                RETURN r.uuid AS uuid, r.relation_id AS relation_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.content AS content
                """,
                abs_ids=abs_ids,
            )
            return [
                {
                    "uuid": record["uuid"],
                    "relation_id": record["relation_id"],
                    "entity1_absolute_id": record["entity1_absolute_id"],
                    "content": record["content"],
                }
                for record in result
            ]

    def delete_self_referential_relations_for_entity(self, entity_id: str) -> int:
        """删除指定实体的自引用关系。"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return 0
        abs_ids = self._get_all_absolute_ids_for_entity(entity_id)
        if not abs_ids:
            return 0
        with self._write_lock:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (r:Relation)
                    WHERE r.entity1_absolute_id IN $abs_ids AND r.entity2_absolute_id IN $abs_ids
                    AND r.entity1_absolute_id = r.entity2_absolute_id
                    RETURN r.uuid AS uuid
                    """,
                    abs_ids=abs_ids,
                )
                uuids = [record["uuid"] for record in result]
                count = len(uuids)
                if uuids:
                    session.run(
                        """
                        MATCH (r:Relation)
                        WHERE r.uuid IN $uuids
                        DETACH DELETE r
                        """,
                        uuids=uuids,
                    )
                    self._vector_store.delete_batch("relation_vectors", uuids)
                return count

    def get_entity_relations(self, entity_absolute_id: str, limit: Optional[int] = None,
                              time_point: Optional[datetime] = None) -> List[Relation]:
        """获取与指定实体相关的所有关系。"""
        with self._driver.session() as session:
            if time_point:
                query = """
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id = $abs_id OR r.entity2_absolute_id = $abs_id)
                    AND r.event_time <= datetime($tp)
                    WITH r.relation_id AS rid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH rid, r ORDER BY r.processed_time DESC
                    WITH rid, HEAD(COLLECT(r)) AS r
                    RETURN r.uuid AS uuid, r.relation_id AS relation_id,
                           r.entity1_absolute_id AS entity1_absolute_id,
                           r.entity2_absolute_id AS entity2_absolute_id,
                           r.content AS content, r.event_time AS event_time,
                           r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                           r.source_document AS source_document
                    ORDER BY r.processed_time DESC
                """
                params = {"abs_id": entity_absolute_id, "tp": time_point.isoformat()}
            else:
                query = """
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id = $abs_id OR r.entity2_absolute_id = $abs_id)
                    WITH r.relation_id AS rid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH rid, r ORDER BY r.processed_time DESC
                    WITH rid, HEAD(COLLECT(r)) AS r
                    RETURN r.uuid AS uuid, r.relation_id AS relation_id,
                           r.entity1_absolute_id AS entity1_absolute_id,
                           r.entity2_absolute_id AS entity2_absolute_id,
                           r.content AS content, r.event_time AS event_time,
                           r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                           r.source_document AS source_document
                    ORDER BY r.processed_time DESC
                """
                params = {"abs_id": entity_absolute_id}

            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = session.run(query, **params)
            return [_neo4j_record_to_relation(r) for r in result]

    def get_entity_relations_by_entity_id(self, entity_id: str, limit: Optional[int] = None,
                                           time_point: Optional[datetime] = None,
                                           max_version_absolute_id: Optional[str] = None) -> List[Relation]:
        """通过 entity_id 获取实体的所有关系（包含所有版本）。"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return []
        abs_ids = self._get_all_absolute_ids_for_entity(entity_id)
        if not abs_ids:
            return []
        if max_version_absolute_id:
            # 获取从最早到 max_version 的所有 absolute_id
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity {entity_id: $eid})
                    WHERE e.processed_time <= (
                        MATCH (e2:Entity {uuid: $max_abs}) RETURN e2.processed_time
                    )
                    RETURN e.uuid AS uuid
                    ORDER BY e.processed_time ASC
                    """,
                    eid=entity_id,
                    max_abs=max_version_absolute_id,
                )
                abs_ids = [r["uuid"] for r in result]

        if not abs_ids:
            return []

        with self._driver.session() as session:
            if time_point:
                query = """
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id IN $abs_ids OR r.entity2_absolute_id IN $abs_ids)
                    AND r.event_time <= datetime($tp)
                    WITH r.relation_id AS rid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH rid, r ORDER BY r.processed_time DESC
                    WITH rid, HEAD(COLLECT(r)) AS r
                    RETURN r.uuid AS uuid, r.relation_id AS relation_id,
                           r.entity1_absolute_id AS entity1_absolute_id,
                           r.entity2_absolute_id AS entity2_absolute_id,
                           r.content AS content, r.event_time AS event_time,
                           r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                           r.source_document AS source_document
                    ORDER BY r.processed_time DESC
                """
                params = {"abs_ids": abs_ids, "tp": time_point.isoformat()}
            else:
                query = """
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id IN $abs_ids OR r.entity2_absolute_id IN $abs_ids)
                    WITH r.relation_id AS rid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH rid, r ORDER BY r.processed_time DESC
                    WITH rid, HEAD(COLLECT(r)) AS r
                    RETURN r.uuid AS uuid, r.relation_id AS relation_id,
                           r.entity1_absolute_id AS entity1_absolute_id,
                           r.entity2_absolute_id AS entity2_absolute_id,
                           r.content AS content, r.event_time AS event_time,
                           r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                           r.source_document AS source_document
                    ORDER BY r.processed_time DESC
                """
                params = {"abs_ids": abs_ids}

            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = session.run(query, **params)
            return [_neo4j_record_to_relation(r) for r in result]

    def get_relations_by_entity_absolute_ids(self, entity_absolute_ids: List[str],
                                              limit: Optional[int] = None) -> List[Relation]:
        """根据 absolute_id 列表获取关系。"""
        if not entity_absolute_ids:
            return []
        with self._driver.session() as session:
            query = """
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id IN $abs_ids OR r.entity2_absolute_id IN $abs_ids)
                WITH r.relation_id AS rid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH rid, r ORDER BY r.processed_time DESC
                WITH rid, HEAD(COLLECT(r)) AS r
                RETURN r.uuid AS uuid, r.relation_id AS relation_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                       r.source_document AS source_document
                ORDER BY r.processed_time DESC
            """
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = session.run(query, abs_ids=entity_absolute_ids)
            return [_neo4j_record_to_relation(r) for r in result]

    def get_entity_absolute_ids_up_to_version(self, entity_id: str, max_absolute_id: str) -> List[str]:
        """获取从最早版本到指定版本的所有 absolute_id。"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return []
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {entity_id: $eid})
                WHERE e.processed_time <= (
                    MATCH (e2:Entity {uuid: $max_abs}) RETURN e2.processed_time
                )
                RETURN e.uuid AS uuid
                ORDER BY e.processed_time ASC
                """,
                eid=entity_id,
                max_abs=max_absolute_id,
            )
            return [r["uuid"] for r in result]

    def get_all_relations(self, limit: Optional[int] = None, offset: Optional[int] = None,
                           exclude_embedding: bool = False) -> List[Relation]:
        """获取所有关系的最新版本。"""
        with self._driver.session() as session:
            query = """
                MATCH (r:Relation)
                WITH r.relation_id AS rid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH rid, r ORDER BY r.processed_time DESC
                WITH rid, HEAD(COLLECT(r)) AS r
                RETURN r.uuid AS uuid, r.relation_id AS relation_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                       r.source_document AS source_document
                ORDER BY r.processed_time DESC
            """
            if offset is not None and offset > 0:
                query += f" SKIP {int(offset)}"
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = session.run(query)
            relations = []
            for record in result:
                relation = _neo4j_record_to_relation(record)
                if not exclude_embedding:
                    emb = self._vector_store.get("relation_vectors", relation.absolute_id)
                    if emb:
                        relation.embedding = np.array(emb, dtype=np.float32).tobytes()
                relations.append(relation)
            return relations

    def _get_relations_with_embeddings(self) -> List[tuple]:
        """获取所有关系的最新版本及其 embedding。"""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (r:Relation)
                WITH r.relation_id AS rid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH rid, r ORDER BY r.processed_time DESC
                WITH rid, HEAD(COLLECT(r)) AS r
                RETURN r.uuid AS uuid, r.relation_id AS relation_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                       r.source_document AS source_document
                ORDER BY r.processed_time DESC
                """
            )
            relations = []
            for record in result:
                relation = _neo4j_record_to_relation(record)
                emb_list = self._vector_store.get("relation_vectors", relation.absolute_id)
                emb_array = np.array(emb_list, dtype=np.float32) if emb_list else None
                relations.append((relation, emb_array))
            return relations

    def search_relations_by_similarity(self, query_text: str,
                                       threshold: float = 0.3,
                                       max_results: int = 10) -> List[Relation]:
        """根据相似度搜索关系。"""
        relations_with_embeddings = self._get_relations_with_embeddings()
        if not relations_with_embeddings:
            return []

        if self.embedding_client and self.embedding_client.is_available():
            return self._search_relations_with_embedding(
                query_text, relations_with_embeddings, threshold, max_results
            )
        else:
            return self._search_relations_with_text_similarity(
                query_text, [r for r, _ in relations_with_embeddings], threshold, max_results
            )

    def _search_relations_with_embedding(self, query_text: str,
                                          relations_with_embeddings: List[tuple],
                                          threshold: float,
                                          max_results: int) -> List[Relation]:
        """使用 embedding 进行关系搜索。"""
        query_embedding = self.embedding_client.encode(query_text)
        if query_embedding is None:
            return []

        query_emb_array = np.array(
            query_embedding[0] if isinstance(query_embedding, (list, np.ndarray)) else query_embedding,
            dtype=np.float32
        )

        similarities = []
        for relation, stored_emb in relations_with_embeddings:
            if stored_emb is None:
                continue
            dot = np.dot(query_emb_array, stored_emb)
            norm_q = np.linalg.norm(query_emb_array)
            norm_s = np.linalg.norm(stored_emb)
            sim = float(dot / (norm_q * norm_s + 1e-9))
            if sim >= threshold:
                similarities.append((relation, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        relations = []
        seen = set()
        for relation, _ in similarities:
            if relation.relation_id not in seen:
                relations.append(relation)
                seen.add(relation.relation_id)
                if len(relations) >= max_results:
                    break
        return relations

    def _search_relations_with_text_similarity(self, query_text: str,
                                                all_relations: List[Relation],
                                                threshold: float,
                                                max_results: int) -> List[Relation]:
        """使用文本相似度进行关系搜索。"""
        scored = []
        for relation in all_relations:
            sim = difflib.SequenceMatcher(None, query_text.lower(), relation.content.lower()).ratio()
            if sim >= threshold:
                scored.append((relation, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        relations = []
        seen = set()
        for relation, _ in scored:
            if relation.relation_id not in seen:
                relations.append(relation)
                seen.add(relation.relation_id)
                if len(relations) >= max_results:
                    break
        return relations

    # ------------------------------------------------------------------
    # 文档操作
    # ------------------------------------------------------------------

    def get_doc_hash_by_cache_id(self, cache_id: str) -> Optional[str]:
        """根据 cache_id 获取 doc_hash。"""
        return self._id_to_doc_hash.get(cache_id)

    def get_memory_cache_text(self, cache_id: str) -> Optional[str]:
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
                except Exception:
                    pass
        # 回退旧结构
        metadata_path = self.cache_json_dir / f"{cache_id}.json"
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                return meta.get("text", "")
            except Exception:
                pass
        return None

    def get_doc_dir(self, doc_hash: str) -> Optional[Path]:
        """获取文档目录。"""
        return self._get_cache_dir_by_doc_hash(doc_hash)

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
        except Exception:
            return None

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
            except Exception:
                continue
        return results

    # ------------------------------------------------------------------
    # 图遍历操作（Neo4j 原生优势）
    # ------------------------------------------------------------------

    def find_related_entities_by_embedding(self, similarity_threshold: float = 0.7,
                                            max_candidates: int = 50,
                                            use_mixed_search: bool = False,
                                            content_snippet_length: int = 50,
                                            progress_callback: Optional[callable] = None) -> Dict[str, set]:
        """通过 embedding 相似度查找相关实体。"""
        all_entities = self.get_all_entities(exclude_embedding=True)
        if not all_entities:
            return {}

        if not self.embedding_client or not self.embedding_client.is_available():
            return {}

        # 编码所有实体
        texts = [f"{e.name} {e.content[:content_snippet_length]}" for e in all_entities]
        embeddings = self.embedding_client.encode(texts)
        if embeddings is None:
            return {}

        # 计算相似度
        entity_pairs: Dict[str, set] = {}
        n = len(all_entities)
        for i in range(n):
            if progress_callback:
                progress_callback(i, n)
            for j in range(i + 1, n):
                emb_i = np.array(embeddings[i], dtype=np.float32)
                emb_j = np.array(embeddings[j], dtype=np.float32)
                dot = np.dot(emb_i, emb_j)
                norm_i = np.linalg.norm(emb_i)
                norm_j = np.linalg.norm(emb_j)
                sim = float(dot / (norm_i * norm_j + 1e-9))
                if sim >= similarity_threshold:
                    eid_i = all_entities[i].entity_id
                    eid_j = all_entities[j].entity_id
                    entity_pairs.setdefault(eid_i, set()).add(eid_j)
                    entity_pairs.setdefault(eid_j, set()).add(eid_i)

        return entity_pairs

    def get_entities_grouped_by_similarity(self, similarity_threshold: float = 0.6) -> List[List[Entity]]:
        """按相似度分组实体（Union-Find）。"""
        all_entities = self.get_all_entities(exclude_embedding=True)
        if not all_entities:
            return []

        if not self.embedding_client or not self.embedding_client.is_available():
            return []

        texts = [f"{e.name} {e.content[:self.entity_content_snippet_length]}" for e in all_entities]
        embeddings = self.embedding_client.encode(texts)
        if embeddings is None:
            return []

        n = len(all_entities)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                emb_i = np.array(embeddings[i], dtype=np.float32)
                emb_j = np.array(embeddings[j], dtype=np.float32)
                dot = np.dot(emb_i, emb_j)
                norm_i = np.linalg.norm(emb_i)
                norm_j = np.linalg.norm(emb_j)
                sim = float(dot / (norm_i * norm_j + 1e-9))
                if sim >= similarity_threshold:
                    union(i, j)

        groups: Dict[int, List[Entity]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(all_entities[i])

        return [group for group in groups.values() if len(group) > 1]

    def merge_entity_ids(self, target_entity_id: str, source_entity_ids: List[str]) -> Dict[str, Any]:
        """合并多个 entity_id 到目标 entity_id。"""
        target_entity_id = self.resolve_entity_id(target_entity_id)
        if not target_entity_id or not source_entity_ids:
            return {"entities_updated": 0, "relations_updated": 0}

        with self._write_lock:
            with self._driver.session() as session:
                entities_updated = 0
                canonical_source_ids: List[str] = []

                for source_id in source_entity_ids:
                    source_id = self._resolve_entity_id_in_session(session, source_id)
                    if not source_id or source_id == target_entity_id or source_id in canonical_source_ids:
                        continue
                    canonical_source_ids.append(source_id)

                    # 更新所有使用 source_id 的实体节点
                    result = session.run(
                        """
                        MATCH (e:Entity {entity_id: $sid})
                        SET e.entity_id = $tid
                        RETURN COUNT(e) AS cnt
                        """,
                        sid=source_id,
                        tid=target_entity_id,
                    )
                    record = result.single()
                    if record:
                        entities_updated += record["cnt"]

                    # 创建 redirect
                    now_iso = datetime.now().isoformat()
                    session.run(
                        """
                        MERGE (red:EntityRedirect {source_id: $sid})
                        SET red.target_id = $tid, red.updated_at = $now
                        """,
                        sid=source_id,
                        tid=target_entity_id,
                        now=now_iso,
                    )

                return {
                    "entities_updated": entities_updated,
                    "relations_updated": 0,
                    "target_entity_id": target_entity_id,
                    "merged_source_ids": canonical_source_ids,
                }

    def find_shortest_paths(self, source_entity_id: str, target_entity_id: str,
                             max_depth: int = 6, max_paths: int = 10) -> Dict[str, Any]:
        """使用 Neo4j Cypher 查找最短路径。"""
        result_empty = {
            "source_entity": None,
            "target_entity": None,
            "path_length": -1,
            "total_shortest_paths": 0,
            "paths": [],
        }

        source_entity = self.get_entity_by_entity_id(source_entity_id)
        target_entity = self.get_entity_by_entity_id(target_entity_id)

        if not source_entity or not target_entity:
            result_empty["source_entity"] = source_entity
            result_empty["target_entity"] = target_entity
            return result_empty

        if source_entity_id == target_entity_id:
            return {
                "source_entity": source_entity,
                "target_entity": target_entity,
                "path_length": 0,
                "total_shortest_paths": 1,
                "paths": [{
                    "entities": [source_entity],
                    "relations": [],
                    "length": 0,
                }],
            }

        # 使用 Cypher allShortestPaths
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (source:Entity {entity_id: $sid}),
                      (target:Entity {entity_id: $tid})
                MATCH path = allShortestPaths((source)-[:RELATES_TO*1..""" + str(max_depth) + """]-(target))
                UNWIND [n IN nodes(path) | n.uuid] AS abs_ids
                UNWIND [r IN relationships(path) | r.relation_uuid] AS rel_uuids
                WITH path, COLLECT(DISTINCT abs_ids) AS abs_id_set, COLLECT(DISTINCT rel_uuids) AS rel_id_set
                RETURN abs_id_set, rel_id_set
                LIMIT $max_paths
                """,
                sid=source_entity_id,
                tid=target_entity_id,
                max_paths=max_paths,
            )

            paths_result = []
            abs_to_eid: Dict[str, str] = {}
            needed_abs_ids: Set[str] = set()
            needed_rel_ids: Set[str] = set()

            for record in result:
                abs_ids = record["abs_id_set"]
                rel_ids = record["rel_id_set"]
                needed_abs_ids.update(abs_ids)
                needed_rel_ids.update(rel_ids)

            # 批量获取实体和关系
            abs_entity_map: Dict[str, Entity] = {}
            if needed_abs_ids:
                res = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.uuid IN $uuids
                    RETURN e.uuid AS uuid, e.entity_id AS entity_id, e.name AS name,
                           e.content AS content, e.event_time AS event_time,
                           e.processed_time AS processed_time, e.memory_cache_id AS memory_cache_id,
                           e.source_document AS source_document
                    """,
                    uuids=list(needed_abs_ids),
                )
                for r in res:
                    entity = _neo4j_record_to_entity(r)
                    abs_entity_map[entity.absolute_id] = entity
                    abs_to_eid[entity.absolute_id] = entity.entity_id

            rel_map: Dict[str, Relation] = {}
            if needed_rel_ids:
                res = session.run(
                    """
                    MATCH (r:Relation)
                    WHERE r.uuid IN $uuids
                    RETURN r.uuid AS uuid, r.relation_id AS relation_id,
                           r.entity1_absolute_id AS entity1_absolute_id,
                           r.entity2_absolute_id AS entity2_absolute_id,
                           r.content AS content, r.event_time AS event_time,
                           r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                           r.source_document AS source_document
                    """,
                    uuids=list(needed_rel_ids),
                )
                for r in res:
                    relation = _neo4j_record_to_relation(r)
                    rel_map[relation.absolute_id] = relation

            # 构建路径结果
            result2 = session.run(
                """
                MATCH (source:Entity {entity_id: $sid}),
                      (target:Entity {entity_id: $tid})
                MATCH path = allShortestPaths((source)-[:RELATES_TO*1..""" + str(max_depth) + """]-(target))
                UNWIND nodes(path) AS n
                UNWIND relationships(path) AS r
                WITH path, COLLECT(DISTINCT {uuid: n.uuid, entity_id: n.entity_id}) AS nodes,
                     COLLECT(DISTINCT {uuid: r.relation_uuid}) AS rels
                RETURN nodes, rels
                LIMIT $max_paths
                """,
                sid=source_entity_id,
                tid=target_entity_id,
                max_paths=max_paths,
            )

            for record in result2:
                path_entities = []
                seen_abs: Set[str] = set()
                for node_info in record["nodes"]:
                    abs_id = node_info["uuid"]
                    if abs_id not in seen_abs and abs_id in abs_entity_map:
                        path_entities.append(abs_entity_map[abs_id])
                        seen_abs.add(abs_id)

                path_relations = []
                for rel_info in record["rels"]:
                    rel_id = rel_info["uuid"]
                    if rel_id in rel_map:
                        path_relations.append(rel_map[rel_id])

                paths_result.append({
                    "entities": path_entities,
                    "relations": path_relations,
                    "length": len(path_entities) - 1,
                })

            path_length = paths_result[0]["length"] if paths_result else -1

            return {
                "source_entity": source_entity,
                "target_entity": target_entity,
                "path_length": path_length,
                "total_shortest_paths": len(paths_result),
                "paths": paths_result,
            }

    # ------------------------------------------------------------------
    # Neo4j 特有操作（新增能力）
    # ------------------------------------------------------------------

    def find_shortest_path_cypher(self, source_entity_id: str, target_entity_id: str,
                                   max_depth: int = 6) -> List[List[str]]:
        """使用 Cypher shortestPath 查找单条最短路径（性能更优）。

        Returns:
            路径列表，每条路径为实体名称列表。
        """
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (a:Entity {entity_id: $sid}), (b:Entity {entity_id: $tid})
                MATCH path = shortestPath((a)-[:RELATES_TO*1..""" + str(max_depth) + """]-(b))
                RETURN [n IN nodes(path) | n.name] AS names
                """,
                sid=source_entity_id,
                tid=target_entity_id,
            )
            records = list(result)
            if not records:
                return []
            return [record["names"] for record in records]

    def get_entity_neighbors(self, entity_uuid: str, depth: int = 1) -> Dict:
        """获取实体的邻居图，返回完整的 nodes + edges 结构。"""
        with self._driver.session() as session:
            # 先获取中心节点
            center = session.run(
                "MATCH (e:Entity {uuid: $uuid}) RETURN e.uuid AS uuid, e.name AS name, e.entity_id AS entity_id",
                uuid=entity_uuid,
            )
            center_records = list(center)
            center_node = None
            if center_records:
                r = center_records[0]
                center_node = {"uuid": r["uuid"], "name": r["name"], "entity_id": r["entity_id"]}

            # 获取所有邻居节点和边
            result = session.run(
                f"""
                MATCH (e:Entity {{uuid: $uuid}})-[r:RELATES_TO*1..{depth}]-(neighbor:Entity)
                WITH neighbor, r LIMIT 500
                RETURN neighbor.uuid AS uuid, neighbor.name AS name, neighbor.entity_id AS entity_id,
                       startNode(r).uuid AS source_uuid, endNode(r).uuid AS target_uuid,
                       r.relation_uuid AS relation_uuid, r.fact AS fact
                """,
                uuid=entity_uuid,
            )
            neighbors = {
                "entity": center_node,
                "nodes": [],
                "edges": [],
            }
            seen = set()
            seen_edges = set()
            for record in result:
                uuid_val = record["uuid"]
                if uuid_val and uuid_val not in seen:
                    neighbors["nodes"].append({
                        "uuid": uuid_val,
                        "name": record["name"],
                        "entity_id": record["entity_id"],
                    })
                    seen.add(uuid_val)
                edge_key = (record.get("source_uuid"), record.get("target_uuid"))
                if edge_key[0] and edge_key[1] and edge_key not in seen_edges:
                    neighbors["edges"].append({
                        "source_uuid": edge_key[0],
                        "target_uuid": edge_key[1],
                        "content": record["fact"],
                        "relation_uuid": record.get("relation_uuid"),
                    })
                    seen_edges.add(edge_key)
            return neighbors

    # ------------------------------------------------------------------
    # 属性访问兼容
    # ------------------------------------------------------------------

    @property
    def is_neo4j(self) -> bool:
        """标识当前为 Neo4j 后端。"""
        return True

    @property
    def db_path(self) -> Path:
        """兼容 StorageManager.db_path 属性。"""
        return self.storage_path / "graph.db"

    # ------------------------------------------------------------------
    # Episode 管理
    # ------------------------------------------------------------------

    def list_episodes(self, limit: int = 20, offset: int = 0) -> List[Dict]:
        """分页查询 Episode 节点，按 created_at DESC。"""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (ep:Episode) RETURN ep.uuid AS uuid, ep.content AS content, "
                "ep.source_document AS source_document, ep.event_time AS event_time, "
                "ep.memory_cache_id AS memory_cache_id, "
                "ep.created_at AS created_at "
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
                    "memory_cache_id": r["memory_cache_id"] or "",
                    "created_at": _fmt_dt(r["created_at"]),
                }
                episodes.append(ep)
            return episodes

    def count_episodes(self) -> int:
        """统计 Episode 节点总数。"""
        with self._driver.session() as session:
            result = session.run("MATCH (ep:Episode) RETURN COUNT(ep) AS cnt")
            record = result.single()
            return record["cnt"] if record else 0

    def count_communities(self) -> int:
        """统计社区数量（DISTINCT community_id）。"""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                "RETURN count(DISTINCT e.community_id) AS cnt"
            )
            record = result.single()
            return record["cnt"] if record else 0

    def get_episode(self, uuid: str) -> Optional[Dict]:
        """获取单个 Episode 详情。"""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (ep:Episode {uuid: $uuid}) RETURN ep.uuid AS uuid, ep.content AS content, "
                "ep.source_document AS source_document, ep.event_time AS event_time, "
                "ep.memory_cache_id AS memory_cache_id, ep.created_at AS created_at",
                uuid=uuid,
            )
            record = result.single()
            if not record:
                return None
            return {
                "uuid": record["uuid"],
                "content": record["content"] or "",
                "source_document": record["source_document"] or "",
                "event_time": _fmt_dt(record["event_time"]),
                "memory_cache_id": record["memory_cache_id"] or "",
                "created_at": _fmt_dt(record["created_at"]),
            }

    def search_episodes(self, query: str, limit: int = 20) -> List[Dict]:
        """通过 content LIKE 搜索 Episode。"""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (ep:Episode) WHERE ep.content CONTAINS $query "
                "RETURN ep.uuid AS uuid, ep.content AS content, "
                "ep.source_document AS source_document, ep.event_time AS event_time, "
                "ep.memory_cache_id AS memory_cache_id, ep.created_at AS created_at "
                "ORDER BY ep.created_at DESC LIMIT $limit",
                query=query, limit=limit,
            )
            episodes = []
            for r in result:
                episodes.append({
                    "uuid": r["uuid"],
                    "content": r["content"] or "",
                    "source_document": r["source_document"] or "",
                    "event_time": _fmt_dt(r["event_time"]),
                    "memory_cache_id": r["memory_cache_id"] or "",
                    "created_at": _fmt_dt(r["created_at"]),
                })
            return episodes

    def get_episode_entities(self, uuid: str) -> List[Dict]:
        """通过 memory_cache_id 关联查出 Episode 下的实体。"""
        with self._driver.session() as session:
            episode = session.run(
                "MATCH (ep:Episode {uuid: $uuid}) RETURN ep.memory_cache_id AS mcid",
                uuid=uuid,
            ).single()
            if not episode or not episode["mcid"]:
                return []
            mcid = episode["mcid"]
            result = session.run(
                "MATCH (e:Entity {memory_cache_id: $mcid}) "
                "RETURN e.uuid AS uuid, e.entity_id AS entity_id, e.name AS name, "
                "e.content AS content, e.event_time AS event_time "
                "ORDER BY e.processed_time DESC LIMIT 200",
                mcid=mcid,
            )
            entities = []
            for r in result:
                entities.append({
                    "uuid": r["uuid"],
                    "entity_id": r["entity_id"],
                    "name": r["name"],
                    "content": r["content"] or "",
                    "event_time": r["event_time"].isoformat() if r["event_time"] else None,
                })
            return entities

    def delete_episode(self, uuid: str) -> bool:
        """删除 Episode 节点。"""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (ep:Episode {uuid: $uuid}) DELETE ep RETURN COUNT(ep) AS deleted",
                uuid=uuid,
            )
            record = result.single()
            return record["deleted"] > 0 if record else False

    # ------------------------------------------------------------------
    # 社区检测
    # ------------------------------------------------------------------

    def detect_communities(self, algorithm: str = 'louvain', resolution: float = 1.0) -> Dict:
        """从 Neo4j 加载图 → networkx Louvain → 写回 community_id。"""
        import time
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        t0 = time.time()

        # 加载所有 Entity + RELATES_TO 边
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) RETURN e.uuid AS uuid, e.entity_id AS eid, e.name AS name"
            )
            entity_map = {}  # uuid -> eid
            for r in result:
                entity_map[r["uuid"]] = r["eid"]

            result = session.run(
                "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                "RETURN a.uuid AS src, b.uuid AS tgt"
            )
            edges = [(r["src"], r["tgt"]) for r in result]

        # 构建 networkx Graph
        G = nx.Graph()
        for uuid_val in entity_map:
            G.add_node(uuid_val)
        for src, tgt in edges:
            if src in G and tgt in G:
                G.add_edge(src, tgt)

        # Louvain 社区检测
        communities = louvain_communities(G, resolution=resolution, seed=42)

        # 构建 assignment: uuid -> community_id
        assignment = {}
        for cid, community_set in enumerate(communities):
            for uuid_val in community_set:
                assignment[uuid_val] = cid

        # 写回 Neo4j
        self._write_community_labels(assignment)

        elapsed = time.time() - t0
        community_sizes = [len(c) for c in communities]
        return {
            "total_communities": len(communities),
            "community_sizes": sorted(community_sizes, reverse=True),
            "elapsed_seconds": round(elapsed, 3),
        }

    def _write_community_labels(self, assignment: Dict[str, int]):
        """批量 UNWIND SET community_id。"""
        # Neo4j 参数列表
        items = [{"uuid": uuid_val, "cid": cid} for uuid_val, cid in assignment.items()]
        if not items:
            return
        batch_size = 5000
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            with self._driver.session() as session:
                session.run(
                    "UNWIND $items AS item "
                    "MATCH (e:Entity {uuid: item.uuid}) "
                    "SET e.community_id = item.cid",
                    items=batch,
                )

    def get_communities(self, limit: int = 50, min_size: int = 3) -> List[Dict]:
        """按社区分组，返回 members 列表。"""
        cache_key = f"communities:{limit}:{min_size}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                "WITH e.community_id AS cid, collect(e) AS members "
                "WHERE size(members) >= $min_size "
                "RETURN cid, size(members) AS size, "
                "[m IN members | {uuid: m.uuid, entity_id: m.entity_id, name: m.name}] AS members "
                "ORDER BY size DESC LIMIT $limit",
                min_size=min_size, limit=limit,
            )
            communities = []
            for r in result:
                communities.append({
                    "community_id": r["cid"],
                    "size": r["size"],
                    "members": r["members"],
                })
            self._cache.set(cache_key, communities, ttl=120)
            return communities

    def get_community(self, cid: int) -> Optional[Dict]:
        """单社区详情 + 社区内关系（合并单条 Cypher，LIMIT 500）。"""
        cache_key = f"community:{cid}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.community_id = $cid "
                "OPTIONAL MATCH (e)-[r:RELATES_TO]-(other:Entity) "
                "WHERE other.community_id = $cid "
                "WITH e, collect(DISTINCT {uuid: other.uuid, name: other.name, fact: r.fact, "
                "ruuid: r.relation_uuid}) AS rels "
                "RETURN e.uuid AS uuid, e.entity_id AS entity_id, e.name AS name, "
                "e.content AS content, rels "
                "ORDER BY size(rels) DESC LIMIT 500",
                cid=cid,
            )
            members = []
            all_relations = []
            seen_rels = set()
            for r in result:
                members.append({
                    "uuid": r["uuid"],
                    "entity_id": r["entity_id"],
                    "name": r["name"],
                    "content": r["content"] or "",
                })
                for rel in (r["rels"] or []):
                    if not rel.get("uuid"):
                        continue
                    rel_key = tuple(sorted([r["uuid"], rel["uuid"]]))
                    if rel_key not in seen_rels:
                        seen_rels.add(rel_key)
                        all_relations.append({
                            "source_uuid": r["uuid"],
                            "source_name": r["name"],
                            "target_uuid": rel["uuid"],
                            "target_name": rel["name"],
                            "content": rel["fact"] or "",
                            "relation_uuid": rel["ruuid"],
                        })
            if not members:
                return None

            result = {
                "community_id": cid,
                "size": len(members),
                "members": members,
                "relations": all_relations,
            }
            self._cache.set(cache_key, result, ttl=120)
            return result

    def get_community_graph(self, cid: int) -> Dict:
        """社区子图的 nodes + edges（供 vis-network 渲染），LIMIT 300 节点。"""
        cache_key = f"community_graph:{cid}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.community_id = $cid "
                "WITH e LIMIT 300 "
                "OPTIONAL MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                "WHERE (a.uuid = e.uuid OR b.uuid = e.uuid) "
                "WITH e, collect(DISTINCT {src: a.uuid, tgt: b.uuid, fact: r.fact}) AS rels "
                "RETURN e.uuid AS uuid, e.entity_id AS entity_id, e.name AS name, rels",
                cid=cid,
            )
            nodes = []
            seen_edges = set()
            edges = []
            for r in result:
                nodes.append({
                    "uuid": r["uuid"],
                    "entity_id": r["entity_id"],
                    "name": r["name"],
                })
                for rel in (r["rels"] or []):
                    if rel.get("src") and rel.get("tgt"):
                        edge_key = (rel["src"], rel["tgt"])
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            edges.append({
                                "source_uuid": rel["src"],
                                "target_uuid": rel["tgt"],
                                "content": rel["fact"] or "",
                            })

            result = {"nodes": nodes, "edges": edges}
            self._cache.set(cache_key, result, ttl=120)
            return result

    def clear_communities(self) -> int:
        """清除所有 community_id 属性，返回清除数量。"""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                "REMOVE e.community_id RETURN COUNT(e) AS cleared"
            )
            record = result.single()
            count = record["cleared"] if record else 0
            self._cache.invalidate(pattern="communities")
            return count

    # ------------------------------------------------------------------
    # 时间旅行（Time Travel）功能
    # ------------------------------------------------------------------

    def get_snapshot(self, time_point: datetime, limit: Optional[int] = None) -> Dict[str, Any]:
        """获取指定时间点的实体/关系快照"""
        with self._driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE (e.valid_at IS NULL OR e.valid_at <= $time)
                  AND (e.invalid_at IS NULL OR e.invalid_at > $time)
                RETURN e.uuid AS id, e.entity_id AS entity_id, e.name AS name,
                       e.content AS content, e.event_time AS event_time,
                       e.processed_time AS processed_time, e.memory_cache_id AS memory_cache_id,
                       e.source_document AS source_document
                ORDER BY e.event_time DESC
                LIMIT $limit
            """, time=time_point.isoformat(), limit=limit or 10000)
            entities = [_neo4j_record_to_entity(r) for r in result]

            result = session.run("""
                MATCH (r:Relation)
                WHERE (r.valid_at IS NULL OR r.valid_at <= $time)
                  AND (r.invalid_at IS NULL OR r.invalid_at > $time)
                RETURN r.uuid AS id, r.relation_id AS relation_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                       r.source_document AS source_document, r.valid_at AS valid_at,
                       r.invalid_at AS invalid_at
                ORDER BY r.event_time DESC
                LIMIT $limit
            """, time=time_point.isoformat(), limit=limit or 10000)
            relations = [_neo4j_record_to_relation(r) for r in result]

        return {"entities": entities, "relations": relations}

    def get_changes(self, since: datetime, until: Optional[datetime] = None) -> Dict[str, Any]:
        """获取时间范围内的变更"""
        if until is None:
            until = datetime.now(timezone.utc)
        with self._driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.event_time >= $since AND e.event_time <= $until
                RETURN e.uuid AS id, e.entity_id AS entity_id, e.name AS name,
                       e.content AS content, e.event_time AS event_time,
                       e.processed_time AS processed_time, e.memory_cache_id AS memory_cache_id,
                       e.source_document AS source_document
                ORDER BY e.event_time DESC
            """, since=since.isoformat(), until=until.isoformat())
            entities = [_neo4j_record_to_entity(r) for r in result]

            result = session.run("""
                MATCH (r:Relation)
                WHERE r.event_time >= $since AND r.event_time <= $until
                RETURN r.uuid AS id, r.relation_id AS relation_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                       r.source_document AS source_document, r.valid_at AS valid_at,
                       r.invalid_at AS invalid_at
                ORDER BY r.event_time DESC
            """, since=since.isoformat(), until=until.isoformat())
            relations = [_neo4j_record_to_relation(r) for r in result]

        return {"entities": entities, "relations": relations}

    def invalidate_relation(self, relation_id: str, reason: str = "") -> int:
        """标记关系为失效"""
        now = datetime.now(timezone.utc).isoformat()
        with self._driver.session() as session:
            result = session.run("""
                MATCH (r:Relation {relation_id: $relation_id})
                WHERE r.invalid_at IS NULL
                SET r.invalid_at = $now
                RETURN count(r) AS cnt
            """, relation_id=relation_id, now=now)
            record = result.single()
            return record["cnt"] if record else 0

    def get_invalidated_relations(self, limit: int = 100) -> List[Relation]:
        """列出已失效的关系"""
        with self._driver.session() as session:
            result = session.run("""
                MATCH (r:Relation)
                WHERE r.invalid_at IS NOT NULL
                RETURN r.uuid AS id, r.relation_id AS relation_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                       r.source_document AS source_document, r.valid_at AS valid_at,
                       r.invalid_at AS invalid_at
                ORDER BY r.invalid_at DESC
                LIMIT $limit
            """, limit=limit)
            return [_neo4j_record_to_relation(r) for r in result]

    # ------------------------------------------------------------------
    # Phase A/C/D/E: 新增方法
    # ------------------------------------------------------------------

    def update_entity_summary(self, entity_id: str, summary: str):
        """更新实体摘要。"""
        resolved = self.resolve_entity_id(entity_id)
        if not resolved:
            return
        with self._driver.session() as session:
            session.run("""
                MATCH (e:Entity {entity_id: $eid})
                WHERE e.invalid_at IS NULL
                SET e.summary = $summary
            """, eid=resolved, summary=summary)
        self._cache.invalidate()

    def update_entity_attributes(self, entity_id: str, attributes: str):
        """更新实体结构化属性。"""
        resolved = self.resolve_entity_id(entity_id)
        if not resolved:
            return
        with self._driver.session() as session:
            session.run("""
                MATCH (e:Entity {entity_id: $eid})
                WHERE e.invalid_at IS NULL
                SET e.attributes = $attributes
            """, eid=resolved, attributes=attributes)
        self._cache.invalidate()

    def update_entity_confidence(self, entity_id: str, confidence: float):
        """更新实体置信度。"""
        resolved = self.resolve_entity_id(entity_id)
        if not resolved:
            return
        with self._driver.session() as session:
            session.run("""
                MATCH (e:Entity {entity_id: $eid})
                WHERE e.invalid_at IS NULL
                SET e.confidence = $confidence
            """, eid=resolved, confidence=confidence)
        self._cache.invalidate()

    def compute_entity_confidence(self, entity_id: str) -> float:
        """计算实体置信度（基于被提及次数和更新新鲜度）。"""
        try:
            provenance = self.get_entity_provenance(entity_id)
            mention_count = len(provenance)
            entity = self.get_entity_by_entity_id(entity_id)
            if not entity:
                return 0.0
            base = 0.3
            mention_score = min(mention_count * 0.1, 0.4)
            freshness_score = 0.3
            if entity.processed_time:
                from datetime import timezone
                age_days = (datetime.now(timezone.utc) - entity.processed_time).days
                freshness_score = max(0.0, 0.3 - age_days * 0.01)
            return round(min(base + mention_score + freshness_score, 1.0), 3)
        except Exception:
            return 0.5

    def get_relations_by_entity_ids(self, entity_ids: List[str], limit: int = 100) -> List[Relation]:
        """获取指定实体 ID 列表相关的所有关系。"""
        if not entity_ids:
            return []
        with self._driver.session() as session:
            abs_ids = []
            for eid in entity_ids:
                resolved = self.resolve_entity_id(eid)
                if resolved:
                    entity = self.get_entity_by_entity_id(resolved)
                    if entity:
                        abs_ids.append(entity.absolute_id)
            if not abs_ids:
                return []
            result = session.run("""
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id IN $abs_ids OR r.entity2_absolute_id IN $abs_ids)
                  AND r.invalid_at IS NULL
                RETURN r.uuid AS id, r.relation_id AS relation_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.memory_cache_id AS memory_cache_id,
                       r.source_document AS source_document, r.valid_at AS valid_at,
                       r.invalid_at AS invalid_at, r.summary AS summary,
                       r.attributes AS attributes, r.confidence AS confidence,
                       r.provenance AS provenance
                LIMIT $limit
            """, abs_ids=abs_ids, limit=limit)
            return [_neo4j_record_to_relation(r) for r in result]

    def get_entity_degree(self, entity_id: str) -> int:
        """获取实体的度（连接数）。"""
        return len(self.get_relations_by_entity_ids([entity_id]))

    def save_episode_mentions(self, episode_id: str, entity_absolute_ids: List[str], context: str = ""):
        """记录 Episode 提及的实体。"""
        with self._write_lock:
            with self._driver.session() as session:
                session.run("""
                    MERGE (ep:Episode {uuid: $ep_id})
                """, ep_id=episode_id)
                for abs_id in entity_absolute_ids:
                    session.run("""
                        MATCH (ep:Episode {uuid: $ep_id})
                        MATCH (e:Entity {uuid: $e_abs})
                        MERGE (ep)-[m:MENTIONS {context: $ctx}]->(e)
                    """, ep_id=episode_id, e_abs=abs_id, ctx=context)

    def get_entity_provenance(self, entity_id: str) -> List[dict]:
        """获取提及该实体的所有 Episode。"""
        entity = self.get_entity_by_entity_id(entity_id)
        if not entity:
            return []
        with self._driver.session() as session:
            result = session.run("""
                MATCH (ep:Episode)-[m:MENTIONS]->(e:Entity {uuid: $abs_id})
                RETURN ep.uuid AS episode_id, m.context AS context
            """, abs_id=entity.absolute_id)
            return [{"episode_id": r["episode_id"], "context": r.get("context", "")} for r in result]

    def get_episode_entities(self, episode_id: str) -> List[dict]:
        """获取 Episode 关联的所有实体。"""
        with self._driver.session() as session:
            result = session.run("""
                MATCH (ep:Episode {uuid: $ep_id})-[m:MENTIONS]->(e:Entity)
                RETURN e.uuid AS entity_absolute_id, e.entity_id AS entity_id,
                       e.name AS name, m.context AS context
            """, ep_id=episode_id)
            return [
                {
                    "entity_absolute_id": r["entity_absolute_id"],
                    "entity_id": r.get("entity_id", ""),
                    "name": r.get("name", ""),
                    "context": r.get("context", ""),
                }
                for r in result
            ]

    def delete_episode_mentions(self, episode_id: str):
        """删除 Episode 的所有 MENTIONS 边。"""
        with self._driver.session() as session:
            session.run("""
                MATCH (ep:Episode {uuid: $ep_id})-[m:MENTIONS]->()
                DELETE m
            """, ep_id=episode_id)

    def update_relation_provenance(self, relation_id: str, provenance: str):
        """更新关系的事实溯源信息。"""
        with self._driver.session() as session:
            session.run("""
                MATCH (r:Relation {relation_id: $rid})
                WHERE r.invalid_at IS NULL
                SET r.provenance = $provenance
            """, rid=relation_id, provenance=provenance)
        self._cache.invalidate()

    def save_dream_log(self, report):
        """保存梦境日志。"""
        import json as _json
        with self._driver.session() as session:
            session.run("""
                MERGE (d:DreamLog {cycle_id: $cycle_id})
                SET d.graph_id = $graph_id,
                    d.start_time = datetime($start_time),
                    d.end_time = datetime($end_time),
                    d.status = $status,
                    d.narrative = $narrative,
                    d.insights = $insights,
                    d.connections = $connections,
                    d.consolidations = $consolidations
            """,
                cycle_id=report.cycle_id,
                graph_id=report.graph_id,
                start_time=report.start_time.isoformat(),
                end_time=(report.end_time or datetime.now()).isoformat(),
                status=report.status,
                narrative=report.narrative,
                insights=_json.dumps(report.insights, ensure_ascii=False),
                connections=_json.dumps(report.new_connections, ensure_ascii=False),
                consolidations=_json.dumps(report.consolidations, ensure_ascii=False),
            )

    def list_dream_logs(self, graph_id: str = "default", limit: int = 20) -> List[dict]:
        """列出梦境日志。"""
        import json as _json
        with self._driver.session() as session:
            result = session.run("""
                MATCH (d:DreamLog)
                WHERE d.graph_id = $graph_id
                RETURN d.cycle_id AS cycle_id, d.graph_id AS graph_id,
                       d.start_time AS start_time, d.end_time AS end_time,
                       d.status AS status, d.narrative AS narrative,
                       d.insights AS insights, d.connections AS connections,
                       d.consolidations AS consolidations
                ORDER BY d.start_time DESC
                LIMIT $limit
            """, graph_id=graph_id, limit=limit)
            logs = []
            for r in result:
                logs.append({
                    "cycle_id": r["cycle_id"],
                    "graph_id": r["graph_id"],
                    "start_time": str(r.get("start_time", "")),
                    "end_time": str(r.get("end_time", "")),
                    "status": r.get("status", ""),
                    "narrative": r.get("narrative", ""),
                    "insights": _json.loads(r["insights"]) if r.get("insights") else [],
                    "connections": _json.loads(r["connections"]) if r.get("connections") else [],
                    "consolidations": _json.loads(r["consolidations"]) if r.get("consolidations") else [],
                })
            return logs

    def get_dream_log(self, cycle_id: str) -> Optional[dict]:
        """获取单条梦境日志。"""
        import json as _json
        with self._driver.session() as session:
            result = session.run("""
                MATCH (d:DreamLog {cycle_id: $cycle_id})
                RETURN d.cycle_id AS cycle_id, d.graph_id AS graph_id,
                       d.start_time AS start_time, d.end_time AS end_time,
                       d.status AS status, d.narrative AS narrative,
                       d.insights AS insights, d.connections AS connections,
                       d.consolidations AS consolidations
            """, cycle_id=cycle_id)
            r = result.single()
            if not r:
                return None
            return {
                "cycle_id": r["cycle_id"],
                "graph_id": r["graph_id"],
                "start_time": str(r.get("start_time", "")),
                "end_time": str(r.get("end_time", "")),
                "status": r.get("status", ""),
                "narrative": r.get("narrative", ""),
                "insights": _json.loads(r["insights"]) if r.get("insights") else [],
                "connections": _json.loads(r["connections"]) if r.get("connections") else [],
                "consolidations": _json.loads(r["consolidations"]) if r.get("consolidations") else [],
            }
