"""
Neo4jStorageManager: Neo4j + sqlite-vec 混合存储后端。

借鉴 Graphiti (Zep) 的分层节点架构：
    Neo4j       → 图结构存储（Entity / Relation / Episode 节点及边）
    sqlite-vec  → embedding 向量存储与 KNN 搜索

与 StorageManager 保持完全相同的公共接口，可作为 drop-in replacement。
"""


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

from ..models import ContentPatch, Episode, Entity, Relation
from ..utils import clean_markdown_code_blocks, wprint
from ..perf import _perf_timer
from .cache import QueryCache
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cypher RETURN 子句片段 — 所有 Entity 查询共用
# ---------------------------------------------------------------------------
_ENTITY_RETURN_FIELDS = """\
e.uuid AS uuid, e.family_id AS family_id, e.name AS name,
e.content AS content, e.summary AS summary,
e.attributes AS attributes, e.confidence AS confidence,
e.content_format AS content_format, e.community_id AS community_id,
e.valid_at AS valid_at, e.invalid_at AS invalid_at,
e.event_time AS event_time, e.processed_time AS processed_time,
e.episode_id AS episode_id, e.source_document AS source_document"""

# ---------------------------------------------------------------------------
# Neo4j 节点 / 边 属性 → Entity / Relation 转换
# ---------------------------------------------------------------------------

def _neo4j_record_to_entity(record) -> Entity:
    """将 Neo4j 查询返回的单条记录转为 Entity dataclass。"""
    return Entity(
        absolute_id=record["uuid"],
        family_id=record["family_id"],
        name=record.get("name", ""),
        content=record.get("content", ""),
        event_time=_parse_dt(record.get("event_time")),
        processed_time=_parse_dt(record.get("processed_time")),
        episode_id=record.get("episode_id", ""),
        source_document=record.get("source_document", "") or "",
        embedding=record.get("embedding"),
        valid_at=_parse_dt(record.get("valid_at")) if record.get("valid_at") is not None else None,
        invalid_at=_parse_dt(record.get("invalid_at")) if record.get("invalid_at") is not None else None,
        summary=record.get("summary"),
        attributes=record.get("attributes"),
        confidence=float(record["confidence"]) if record.get("confidence") is not None else None,
        content_format=record.get("content_format", "plain"),
        community_id=record.get("community_id"),
    )


def _neo4j_record_to_relation(record) -> Relation:
    """将 Neo4j 查询返回的单条记录转为 Relation dataclass。"""
    return Relation(
        absolute_id=record["uuid"],
        family_id=record["family_id"],
        entity1_absolute_id=record.get("entity1_absolute_id", ""),
        entity2_absolute_id=record.get("entity2_absolute_id", ""),
        content=record.get("content", ""),
        event_time=_parse_dt(record.get("event_time")),
        processed_time=_parse_dt(record.get("processed_time")),
        episode_id=record.get("episode_id", ""),
        source_document=record.get("source_document", "") or "",
        embedding=record.get("embedding"),
        valid_at=_parse_dt(record.get("valid_at")) if record.get("valid_at") is not None else None,
        invalid_at=_parse_dt(record.get("invalid_at")) if record.get("invalid_at") is not None else None,
        summary=record.get("summary"),
        attributes=record.get("attributes"),
        confidence=float(record["confidence"]) if record.get("confidence") is not None else None,
        provenance=record.get("provenance"),
        content_format=record.get("content_format", "plain"),
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
        graph_id: str = "default",
        **_kwargs,
    ):
        import neo4j

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Neo4j 多数据库隔离：每个 graph_id 对应独立数据库
        self._graph_id = graph_id
        # Neo4j 数据库名规则：小写字母、数字、下划线、点；将连字符替换为下划线
        self._database = f"deepdream_{graph_id.lower().replace('-', '_')}"

        # Neo4j 驱动
        self._neo4j_uri = neo4j_uri
        self._neo4j_auth = neo4j_auth
        self._driver = neo4j.GraphDatabase.driver(
            neo4j_uri, auth=neo4j_auth,
            max_connection_pool_size=50,
            connection_acquisition_timeout=30.0,
            max_transaction_retry_time=15.0,
            notifications_disabled_categories=["UNRECOGNIZED"],
        )
        self._driver.verify_connectivity()

        # 确保数据库存在
        self._ensure_database()

        # 文档目录（与 StorageManager 相同的文件存储结构）
        self.docs_dir = self.storage_path / "docs"
        self.docs_dir.mkdir(exist_ok=True)
        self.cache_dir = self.storage_path / "episodes"
        self.cache_json_dir = self.cache_dir / "json"
        self.cache_md_dir = self.cache_dir / "md"

        # 缓存 cache_id → doc_hash 映射
        self._id_to_doc_hash: Dict[str, str] = {}

        # 写锁（按资源类型拆分，提升并发）
        self._entity_write_lock = threading.Lock()
        self._relation_write_lock = threading.Lock()
        self._episode_write_lock = threading.Lock()
        # 兼容旧代码
        self._write_lock = self._entity_write_lock

        # 查询缓存
        self._cache = QueryCache(default_ttl=30)

        # Embedding 客户端
        self.embedding_client = embedding_client
        self.entity_content_snippet_length = entity_content_snippet_length
        self.relation_content_snippet_length = relation_content_snippet_length

        # 全量 embedding 缓存（短 TTL，避免同一 remember() 调用中重复全量加载）
        self._entity_emb_cache: Optional[List[tuple]] = None
        self._entity_emb_cache_ts: float = 0.0
        self._relation_emb_cache: Optional[List[tuple]] = None
        self._relation_emb_cache_ts: float = 0.0
        self._emb_cache_ttl: float = 5.0

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

    def _ensure_database(self):
        """确保 graph_id 对应的 Neo4j 数据库存在。

        策略:
          - graph_id="default" → 直接使用原 "neo4j" 数据库（老数据零迁移）
          - 其他 graph_id → 尝试 CREATE DATABASE（Enterprise Edition）
          - Community Edition 不支持多数据库时回退到 "neo4j"
        """
        # default 图谱直接使用原 neo4j 数据库，无需迁移
        if self._graph_id == "default":
            self._database = "neo4j"
            logger.info("graph_id='default' → using original 'neo4j' database (no migration needed)")
            return

        db_name = self._database
        try:
            with self._driver.session(database="system") as session:
                session.run(f"CREATE DATABASE `{db_name}` IF NOT EXISTS")
            # 等待数据库上线
            for _ in range(30):
                with self._driver.session(database="system") as session:
                    result = session.run("SHOW DATABASES")
                    for record in result:
                        if record["name"] == db_name and record.get("currentStatus") == "online":
                            logger.info("Neo4j database '%s' is online", db_name)
                            return
                time.sleep(0.5)
            logger.warning("Neo4j database '%s' may not be online yet, proceeding anyway", db_name)
        except Exception as e:
            # Community Edition 不支持 CREATE DATABASE，回退到默认数据库
            logger.warning(
                "Could not create database '%s' (falling back to default 'neo4j'): %s",
                db_name, e,
            )
            self._database = "neo4j"

    def _session(self):
        """创建指向当前图谱数据库的 session。"""
        return self._driver.session(database=self._database)

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
            # ContentPatch 唯一性约束
            "CREATE CONSTRAINT content_patch_uuid IF NOT EXISTS FOR (cp:ContentPatch) REQUIRE cp.uuid IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX entity_family_id IF NOT EXISTS FOR (e:Entity) ON (e.family_id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_processed_time IF NOT EXISTS FOR (e:Entity) ON (e.processed_time)",
            "CREATE INDEX entity_event_time IF NOT EXISTS FOR (e:Entity) ON (e.event_time)",
            "CREATE INDEX entity_cache_id IF NOT EXISTS FOR (e:Entity) ON (e.episode_id)",
            "CREATE INDEX relation_family_id IF NOT EXISTS FOR (r:Relation) ON (r.family_id)",
            "CREATE INDEX relation_processed_time IF NOT EXISTS FOR (r:Relation) ON (r.processed_time)",
            "CREATE INDEX relation_entities IF NOT EXISTS FOR (r:Relation) ON (r.entity1_absolute_id, r.entity2_absolute_id)",
            "CREATE INDEX redirect_target IF NOT EXISTS FOR (red:EntityRedirect) ON (red.target_id)",
            # ContentPatch 索引
            "CREATE INDEX content_patch_target IF NOT EXISTS FOR (cp:ContentPatch) ON (cp.target_absolute_id)",
            "CREATE INDEX content_patch_family IF NOT EXISTS FOR (cp:ContentPatch) ON (cp.target_family_id)",
        ]
        with self._session() as session:
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
                "CREATE INDEX episode_episode_type IF NOT EXISTS FOR (ep:Episode) ON (ep.episode_type)",
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
                except Exception as e:
                    logger.debug("Failed to read doc meta.json: %s", e)

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ContentPatch（Section 级变更记录）
    # ------------------------------------------------------------------

    def save_content_patches(self, patches: list):
        """批量保存 ContentPatch 节点到 Neo4j。"""

        if not patches:
            return
        with self._entity_write_lock:
            with self._session() as session:
                for p in patches:
                    session.run(
                        """
                        CREATE (cp:ContentPatch {
                            uuid: $uuid,
                            target_type: $target_type,
                            target_absolute_id: $target_abs_id,
                            target_family_id: $target_family_id,
                            section_key: $section_key,
                            change_type: $change_type,
                            old_hash: $old_hash,
                            new_hash: $new_hash,
                            diff_summary: $diff_summary,
                            source_document: $source,
                            event_time: datetime($event_time)
                        })
                        WITH cp, $target_abs_id AS abs_id
                        MATCH (t) WHERE t.uuid = abs_id
                        MERGE (cp)-[:PATCHES]->(t)
                        """,
                        uuid=p.uuid,
                        target_type=p.target_type,
                        target_abs_id=p.target_absolute_id,
                        target_family_id=p.target_family_id,
                        section_key=p.section_key,
                        change_type=p.change_type,
                        old_hash=p.old_hash,
                        new_hash=p.new_hash,
                        diff_summary=p.diff_summary,
                        source=p.source_document,
                        event_time=p.event_time.isoformat() if p.event_time else datetime.now().isoformat(),
                    )

    def get_content_patches(self, family_id: str, section_key: str = None) -> list:
        """查询指定 family_id 的 ContentPatch 记录。"""

        with self._session() as session:
            if section_key:
                result = session.run(
                    """
                    MATCH (cp:ContentPatch {target_family_id: $fid, section_key: $sk})
                    RETURN cp ORDER BY cp.event_time DESC
                    """,
                    fid=family_id, sk=section_key,
                )
            else:
                result = session.run(
                    """
                    MATCH (cp:ContentPatch {target_family_id: $fid})
                    RETURN cp ORDER BY cp.event_time DESC
                    """,
                    fid=family_id,
                )
            patches = []
            for record in result:
                cp = record["cp"]
                patches.append(ContentPatch(
                    uuid=cp["uuid"],
                    target_type=cp["target_type"],
                    target_absolute_id=cp["target_absolute_id"],
                    target_family_id=cp["target_family_id"],
                    section_key=cp["section_key"],
                    change_type=cp["change_type"],
                    old_hash=cp.get("old_hash", ""),
                    new_hash=cp.get("new_hash", ""),
                    diff_summary=cp.get("diff_summary", ""),
                    source_document=cp.get("source_document", ""),
                    event_time=_parse_dt(cp.get("event_time")),
                ))
            return patches

    def get_section_history(self, family_id: str, section_key: str) -> list:
        """获取单个 section 的全版本变更历史。"""
        return self.get_content_patches(family_id, section_key=section_key)

    def get_version_diff(self, family_id: str, v1: str, v2: str) -> dict:
        """获取两个版本之间的 section 级 diff。

        v1, v2 是两个 absolute_id（版本 uuid）。
        返回 {section_key: {"v1": content_or_None, "v2": content_or_None, "changed": bool}}
        """
        from ..content_schema import parse_markdown_sections, compute_section_diff
        with self._session() as session:
            v1_content = ""
            v2_content = ""
            result = session.run(
                """
                MATCH (e:Entity) WHERE e.uuid = $v1 OR e.uuid = $v2
                RETURN e.uuid AS uid, e.content AS content
                """,
                v1=v1, v2=v2,
            )
            for record in result:
                if record["uid"] == v1:
                    v1_content = record["content"] or ""
                elif record["uid"] == v2:
                    v2_content = record["content"] or ""
            s1 = parse_markdown_sections(v1_content)
            s2 = parse_markdown_sections(v2_content)
            return compute_section_diff(s1, s2)

    def close(self):
        """关闭 Neo4j 驱动和向量存储连接。"""
        try:
            self._driver.close()
        except Exception as e:
            logger.warning("Error closing Neo4j driver: %s", e)
        self._vector_store.close()

    # ------------------------------------------------------------------
    # Entity Redirect（实体 ID 解析）
    # ------------------------------------------------------------------

    def _resolve_family_id_in_session(self, session, family_id: str) -> str:
        """沿 EntityRedirect 链解析到 canonical family_id。"""
        current_id = (family_id or "").strip()
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

    def resolve_family_id(self, family_id: str) -> str:
        """解析 family_id 到 canonical id。"""
        cache_key = f"resolve:{family_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with _perf_timer("resolve_family_id"):
            with self._session() as session:
                resolved = self._resolve_family_id_in_session(session, family_id)
        self._cache.set(cache_key, resolved, ttl=120)
        return resolved

    def resolve_family_ids(self, family_ids: List[str]) -> Dict[str, str]:
        """批量解析 family_id 到 canonical id。利用缓存 + 一次 Cypher 查询未缓存项。

        Returns:
            {原始 family_id: canonical family_id} 映射
        """
        if not family_ids:
            return {}
        unique_ids = list(set(fid.strip() for fid in family_ids if fid and fid.strip()))
        if not unique_ids:
            return {}

        # 第一步：从缓存获取
        result: Dict[str, str] = {}
        uncached: List[str] = []
        for fid in unique_ids:
            cache_key = f"resolve:{fid}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                result[fid] = cached
            else:
                uncached.append(fid)

        # 第二步：批量查询未缓存项
        if uncached:
            with _perf_timer("resolve_family_ids_batch"):
                with self._session() as session:
                    cypher = """
                    UNWIND $ids AS sid
                    MATCH (red:EntityRedirect {source_id: sid})
                    RETURN red.source_id AS source, red.target_id AS target
                    """
                    records = session.run(cypher, ids=uncached)
                    redirect_map = {r["source"]: r["target"] for r in records}

            # 解析第二跳（链式重定向）
            second_hop_sources = [t for t in redirect_map.values() if t not in result and t in uncached]
            if second_hop_sources:
                with self._session() as session:
                    cypher2 = """
                    UNWIND $ids AS sid
                    MATCH (red:EntityRedirect {source_id: sid})
                    RETURN red.source_id AS source, red.target_id AS target
                    """
                    records2 = session.run(cypher2, ids=second_hop_sources)
                    redirect_map.update({r["source"]: r["target"] for r in records2})

            for fid in uncached:
                target = redirect_map.get(fid, fid)
                # 沿链解析（通常只需 1 跳）
                final = redirect_map.get(target, target) if target != fid else fid
                result[fid] = final
                cache_key = f"resolve:{fid}"
                self._cache.set(cache_key, final, ttl=120)

        # 构建输出映射（处理可能有重复的 family_ids）
        output: Dict[str, str] = {}
        for fid in family_ids:
            key = fid.strip() if fid else ""
            output[fid] = result.get(key, key)
        return output

    def register_entity_redirect(self, source_family_id: str, target_family_id: str) -> str:
        """登记旧 family_id → canonical family_id 映射。"""
        source_id = (source_family_id or "").strip()
        target_id = (target_family_id or "").strip()
        if not source_id or not target_id:
            return target_id
        with self._write_lock:
            with self._session() as session:
                canonical_target = self._resolve_family_id_in_session(session, target_id)
                if not canonical_target:
                    canonical_target = target_id
                canonical_source = self._resolve_family_id_in_session(session, source_id)
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

    def register_entity_redirects(self, target_family_id: str, source_family_ids: List[str]) -> str:
        """批量登记多个旧 family_id 指向同一 canonical id。"""
        canonical_target = (target_family_id or "").strip()
        if not canonical_target:
            return canonical_target
        for source_id in source_family_ids:
            if source_id and source_id != canonical_target:
                canonical_target = self.register_entity_redirect(source_id, canonical_target)
        return canonical_target

    # ------------------------------------------------------------------
    # Episode 操作（文件存储，与 StorageManager 相同逻辑）
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
            session.run(
                """
                MERGE (ep:Episode {uuid: $uuid})
                SET ep.content = $content,
                    ep.source_document = $source,
                    ep.event_time = $event_time,
                    ep.episode_type = $episode_type,
                    ep.activity_type = $activity_type,
                    ep.created_at = datetime()
                """,
                uuid=cache.absolute_id,
                content=cache.content,
                source=cache.source_document,
                event_time=cache.event_time.isoformat(),
                episode_type=cache.episode_type,
                activity_type=cache.activity_type,
            )

        return doc_hash

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
            })
        with self._session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (ep:Episode {uuid: row.uuid})
                SET ep.content = row.content,
                    ep.source_document = row.source,
                    ep.event_time = row.event_time,
                    ep.episode_type = row.episode_type,
                    ep.activity_type = row.activity_type,
                    ep.created_at = datetime()
                """,
                rows=rows,
            )
        return len(rows)

    def load_episode(self, cache_id: str) -> Optional[Episode]:
        """从 Neo4j 或文件系统加载 Episode。"""
        # 优先从 Neo4j 加载
        with self._session() as session:
            result = session.run(
                "MATCH (ep:Episode {uuid: $uuid}) RETURN ep.content AS content, "
                "ep.event_time AS event_time, ep.source_document AS source_document",
                uuid=cache_id,
            )
            record = result.single()
            if record:
                return Episode(
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
                    return Episode(
                        absolute_id=cache_id,
                        content=(doc_dir / "cache.md").read_text(encoding="utf-8") if (doc_dir / "cache.md").exists() else "",
                        event_time=datetime.fromisoformat(meta["event_time"]) if meta.get("event_time") else datetime.now(),
                        source_document=meta.get("source_document", ""),
                        activity_type=meta.get("activity_type"),
                    )
                except Exception as e:
                    logger.debug("Failed to load episode %s from file: %s", cache_id, e)
        return None

    def _iter_cache_meta_files(self) -> List[Path]:
        """遍历 docs/ 目录下所有 meta.json 文件。"""
        if not self.docs_dir.is_dir():
            return []
        return sorted(self.docs_dir.glob("*/meta.json"))

    def get_latest_episode(self, activity_type: Optional[str] = None) -> Optional[Episode]:
        """获取最新的记忆缓存。"""
        with self._session() as session:
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
                return Episode(
                    absolute_id=record["uuid"],
                    content=record["content"] or "",
                    event_time=_parse_dt(record["event_time"]),
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
            result = session.run(query, **params)
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
                        source_document=meta.get("source_document", ""),
                        activity_type=meta.get("activity_type"),
                    )
            except Exception as e:
                logger.debug("Skipping meta file during doc_hash lookup: %s", e)
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

    def _compute_entity_embedding(self, entity: Entity) -> Optional[bytes]:
        """计算实体的 embedding 向量（L2 归一化后存储）。"""
        if not self.embedding_client or not self.embedding_client.is_available():
            return None
        n = self.entity_content_snippet_length
        text = f"{entity.name} {entity.content[:n]}"
        embedding = self.embedding_client.encode(text)
        if embedding is None or len(embedding) == 0:
            return None
        emb_array = np.array(embedding[0] if isinstance(embedding, list) else embedding, dtype=np.float32)
        norm = np.linalg.norm(emb_array)
        if norm > 0:
            emb_array = emb_array / norm
        return emb_array.tobytes()

    def _compute_relation_embedding(self, relation: Relation) -> Optional[bytes]:
        """计算关系的 embedding 向量（L2 归一化后存储）。"""
        if not self.embedding_client or not self.embedding_client.is_available():
            return None
        n = self.relation_content_snippet_length
        text = relation.content if n is None or n <= 0 else relation.content[:n]
        embedding = self.embedding_client.encode(text)
        if embedding is None or len(embedding) == 0:
            return None
        emb_array = np.array(embedding[0] if isinstance(embedding, list) else embedding, dtype=np.float32)
        norm = np.linalg.norm(emb_array)
        if norm > 0:
            emb_array = emb_array / norm
        return emb_array.tobytes()

    # ------------------------------------------------------------------
    # Entity 操作
    # ------------------------------------------------------------------

    def save_entity(self, entity: Entity):
        """保存实体到 Neo4j + sqlite-vec（合并为单条 Cypher）。"""
        self._invalidate_emb_cache()
        with _perf_timer("save_entity"):
            embedding_blob = self._compute_entity_embedding(entity)
            entity.embedding = embedding_blob

            valid_at = (entity.valid_at or entity.event_time).isoformat()

            with self._write_lock:
                with self._session() as session:
                    session.run(
                        """
                        MERGE (e:Entity {uuid: $uuid})
                        SET e.family_id = $family_id,
                            e.name = $name,
                            e.content = $content,
                            e.event_time = datetime($event_time),
                            e.processed_time = datetime($processed_time),
                            e.episode_id = $cache_id,
                            e.source_document = $source,
                            e.summary = $summary,
                            e.attributes = $attributes,
                            e.confidence = $confidence,
                            e.content_format = $content_format,
                            e.valid_at = datetime($valid_at)
                        WITH $uuid AS abs_id, $family_id AS fid, $event_time AS et
                        MATCH (e:Entity {family_id: fid})
                        WHERE e.uuid <> abs_id AND e.invalid_at IS NULL
                        SET e.invalid_at = datetime(et)
                        """,
                        uuid=entity.absolute_id,
                        family_id=entity.family_id,
                        name=entity.name,
                        content=entity.content,
                        event_time=entity.event_time.isoformat(),
                        processed_time=entity.processed_time.isoformat(),
                    cache_id=entity.episode_id,
                    source=entity.source_document,
                    summary=entity.summary,
                    attributes=entity.attributes,
                    confidence=entity.confidence,
                    content_format=getattr(entity, "content_format", "plain"),
                    valid_at=valid_at,
                )

            # 存储向量
            if embedding_blob:
                emb_list = list(np.frombuffer(embedding_blob, dtype=np.float32))
                self._vector_store.upsert("entity_vectors", entity.absolute_id, emb_list)

        self._cache.invalidate("entity:")
        self._cache.invalidate("resolve:")
        self._cache.invalidate("sim_search:")

    def bulk_save_entities(self, entities: List[Entity]):
        """批量保存实体（UNWIND 批量写入）。"""
        if not entities:
            return
        self._invalidate_emb_cache()

        with _perf_timer(f"bulk_save_entities | count={len(entities)}"):
            # 批量计算 embedding
            embeddings = None
            if self.embedding_client and self.embedding_client.is_available():
                texts = [
                    f"{e.name} {e.content[:self.entity_content_snippet_length]}"
                    for e in entities
                ]
                embeddings = self.embedding_client.encode(texts)

            with self._write_lock:
                vec_items = []
                rows = []
                for idx, entity in enumerate(entities):
                    embedding_blob = None
                    if embeddings is not None:
                        try:
                            emb_arr = np.array(embeddings[idx], dtype=np.float32)
                            norm = np.linalg.norm(emb_arr)
                            if norm > 0:
                                emb_arr = emb_arr / norm
                            embedding_blob = emb_arr.tobytes()
                        except Exception as e:
                            logger.debug("Embedding decode failed for entity index %d: %s", idx, e)
                            embedding_blob = None
                    entity.embedding = embedding_blob

                    rows.append({
                        "uuid": entity.absolute_id,
                        "family_id": entity.family_id,
                        "name": entity.name,
                        "content": entity.content,
                        "event_time": entity.event_time.isoformat(),
                        "processed_time": entity.processed_time.isoformat(),
                        "cache_id": entity.episode_id,
                        "source": entity.source_document,
                        "summary": entity.summary,
                        "attributes": entity.attributes,
                        "confidence": entity.confidence,
                        "valid_at": (entity.valid_at or entity.event_time).isoformat(),
                    })

                    if embedding_blob:
                        emb_list = list(np.frombuffer(embedding_blob, dtype=np.float32))
                        vec_items.append((entity.absolute_id, emb_list))

                # 一次 UNWIND 替代 N 次 session.run
                with self._session() as session:
                    session.run(
                        """
                        UNWIND $rows AS row
                        MERGE (e:Entity {uuid: row.uuid})
                        SET e.family_id = row.family_id,
                            e.name = row.name,
                            e.content = row.content,
                            e.event_time = datetime(row.event_time),
                            e.processed_time = datetime(row.processed_time),
                            e.episode_id = row.cache_id,
                            e.source_document = row.source,
                            e.summary = row.summary,
                            e.attributes = row.attributes,
                            e.confidence = row.confidence,
                            e.valid_at = datetime(row.valid_at)
                        WITH row
                        MATCH (e:Entity {family_id: row.family_id})
                        WHERE e.uuid <> row.uuid AND e.invalid_at IS NULL
                        SET e.invalid_at = datetime(row.event_time)
                        """,
                        rows=rows,
                    )

                if vec_items:
                    self._vector_store.upsert_batch("entity_vectors", vec_items)

            self._cache.invalidate("entity:")
            self._cache.invalidate("resolve:")
            self._cache.invalidate("sim_search:")

    def get_entity_by_family_id(self, family_id: str) -> Optional[Entity]:
        """根据 family_id 获取最新版本的实体。"""
        cache_key = f"entity:by_fid:{family_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with _perf_timer("get_entity_by_family_id"):
            family_id = self.resolve_family_id(family_id)
            if not family_id:
                return None
            with self._session() as session:
                result = session.run(
                    f"""
                    MATCH (e:Entity {{family_id: $fid}})
                    RETURN {_ENTITY_RETURN_FIELDS}
                    ORDER BY e.processed_time DESC LIMIT 1
                    """,
                    fid=family_id,
                )
                record = result.single()
                if not record:
                    return None
                entity = _neo4j_record_to_entity(record)
                # 从 sqlite-vec 获取 embedding
                emb = self._vector_store.get("entity_vectors", entity.absolute_id)
                if emb:
                    entity.embedding = np.array(emb, dtype=np.float32).tobytes()
                self._cache.set(cache_key, entity, ttl=60)
                return entity

    def get_entity_by_absolute_id(self, absolute_id: str) -> Optional[Entity]:
        """根据 absolute_id 获取实体。"""
        cache_key = f"entity:by_abs:{absolute_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with self._session() as session:
            result = session.run(
                f"""
                MATCH (e:Entity {{uuid: $uuid}})
                RETURN {_ENTITY_RETURN_FIELDS}
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
            self._cache.set(cache_key, entity, ttl=60)
            return entity

    def get_entity_names_by_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, str]:
        """批量根据 absolute_id 查询实体名称。"""
        if not absolute_ids:
            return {}
        with self._session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.uuid IN $uuids
                RETURN e.uuid AS uuid, e.name AS name
                """,
                uuids=absolute_ids,
            )
            return {record["uuid"]: record["name"] for record in result}

    def get_entities_by_absolute_ids(self, absolute_ids: List[str]) -> List[Entity]:
        """批量根据 absolute_id 获取实体。"""
        if not absolute_ids:
            return []
        with self._session() as session:
            result = session.run(
                f"""
                MATCH (e:Entity)
                WHERE e.uuid IN $uuids
                RETURN {_ENTITY_RETURN_FIELDS}
                """,
                uuids=absolute_ids,
            )
            return [_neo4j_record_to_entity(r) for r in result]

    def get_entity_version_at_time(self, family_id: str, time_point: datetime) -> Optional[Entity]:
        """获取实体在指定时间点的版本。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return None
        with self._session() as session:
            result = session.run(
                f"""
                MATCH (e:Entity {{family_id: $fid}})
                WHERE e.event_time <= datetime($tp)
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time DESC LIMIT 1
                """,
                fid=family_id,
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

    def get_entity_versions(self, family_id: str) -> List[Entity]:
        """获取实体的所有版本。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        with self._session() as session:
            result = session.run(
                f"""
                MATCH (e:Entity {{family_id: $fid}})
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time ASC
                """,
                fid=family_id,
            )
            entities = []
            for record in result:
                entities.append(_neo4j_record_to_entity(record))
            return entities

    def _invalidate_emb_cache(self):
        """清除 embedding 缓存（在实体/关系写入时调用）。"""
        self._entity_emb_cache = None
        self._entity_emb_cache_ts = 0.0
        self._relation_emb_cache = None
        self._relation_emb_cache_ts = 0.0

    def _get_entities_with_embeddings(self) -> List[tuple]:
        """获取所有实体的最新版本及其 embedding（带短 TTL 缓存）。"""
        import time as _time
        now = _time.time()
        if self._entity_emb_cache is not None and (now - self._entity_emb_cache_ts) < self._emb_cache_ttl:
            return self._entity_emb_cache
        with _perf_timer("_get_entities_with_embeddings"):
            result = self._get_entities_with_embeddings_impl()
        self._entity_emb_cache = result
        self._entity_emb_cache_ts = _time.time()
        return result

    def _get_entities_with_embeddings_impl(self) -> List[tuple]:
        """获取所有实体的最新版本及其 embedding（实际实现）。"""
        with self._session() as session:
            result = session.run(
                f"""
                MATCH (e:Entity)
                WITH e.family_id AS fid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(COLLECT(e)) AS e
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time DESC
                """
            )
            records = list(result)

        if not records:
            return []

        # 批量获取所有 embedding（1 次 sqlite 查询 vs N 次）
        uuids = [record["uuid"] for record in records]
        emb_map = self._vector_store.get_batch("entity_vectors", uuids)

        entities = []
        for record in records:
            entity = _neo4j_record_to_entity(record)
            emb_list = emb_map.get(entity.absolute_id)
            emb_array = np.array(emb_list, dtype=np.float32) if emb_list else None
            entities.append((entity, emb_array))
        return entities

    def get_latest_entities_projection(self, content_snippet_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取最新实体投影。"""
        snippet_length = content_snippet_length or self.entity_content_snippet_length
        entities_with_emb = self._get_entities_with_embeddings()
        version_counts = self.get_entity_version_counts([
            e.family_id for e, _ in entities_with_emb
        ])
        results: List[Dict[str, Any]] = []
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

    def get_all_entities(self, limit: Optional[int] = None, offset: Optional[int] = None, exclude_embedding: bool = False) -> List[Entity]:
        """获取所有实体的最新版本。"""
        with self._session() as session:
            query = f"""
                MATCH (e:Entity)
                WITH e.family_id AS fid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(COLLECT(e)) AS e
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time DESC
            """
            if offset is not None and offset > 0:
                query += f" SKIP {int(offset)}"
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = session.run(query)
            records = list(result)

        entities = [_neo4j_record_to_entity(r) for r in records]

        if not exclude_embedding and entities:
            uuids = [e.absolute_id for e in entities]
            emb_map = self._vector_store.get_batch("entity_vectors", uuids)
            for entity in entities:
                emb_list = emb_map.get(entity.absolute_id)
                if emb_list:
                    entity.embedding = np.array(emb_list, dtype=np.float32).tobytes()

        return entities

    def count_unique_entities(self) -> int:
        """统计有效实体中不重复的 family_id 数量。"""
        with self._session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.invalid_at IS NULL RETURN COUNT(DISTINCT e.family_id) AS cnt"
            )
            record = result.single()
            return record["cnt"] if record else 0

    def count_unique_relations(self) -> int:
        """统计有效关系中不重复的 family_id 数量。"""
        with self._session() as session:
            result = session.run(
                "MATCH (r:Relation) WHERE r.invalid_at IS NULL RETURN COUNT(DISTINCT r.family_id) AS cnt"
            )
            record = result.single()
            return record["cnt"] if record else 0

    def get_all_entities_before_time(self, time_point: datetime, limit: Optional[int] = None,
                                      exclude_embedding: bool = False) -> List[Entity]:
        """获取指定时间点之前的所有实体最新版本。"""
        with self._session() as session:
            query = f"""
                MATCH (e:Entity)
                WHERE e.event_time <= datetime($tp)
                WITH e.family_id AS fid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(COLLECT(e)) AS e
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time DESC
            """
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = session.run(query, tp=time_point.isoformat())
            records = list(result)

        entities = [_neo4j_record_to_entity(r) for r in records]

        if not exclude_embedding and entities:
            uuids = [e.absolute_id for e in entities]
            emb_map = self._vector_store.get_batch("entity_vectors", uuids)
            for entity in entities:
                emb_list = emb_map.get(entity.absolute_id)
                if emb_list:
                    entity.embedding = np.array(emb_list, dtype=np.float32).tobytes()

        return entities

    def get_entity_version_count(self, family_id: str) -> int:
        """获取指定 family_id 的版本数量。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        with self._session() as session:
            result = session.run(
                "MATCH (e:Entity {family_id: $fid}) RETURN COUNT(e) AS cnt",
                fid=family_id,
            )
            record = result.single()
            return record["cnt"] if record else 0

    def get_entity_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
        """批量获取多个 family_id 的版本数量。"""
        if not family_ids:
            return {}
        # 批量解析重定向
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = list(set(r for r in resolved_map.values() if r))
        if not canonical_ids:
            return {}
        with self._session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.family_id IN $fids
                RETURN e.family_id AS family_id, COUNT(e) AS cnt
                """,
                fids=canonical_ids,
            )
            return {record["family_id"]: record["cnt"] for record in result}

    def batch_get_entity_profiles(self, family_ids: List[str]) -> List[Dict[str, Any]]:
        """批量获取实体档案（entity + relations + version_count），一次查询。

        替代对每个 family_id 分别调用 get_entity_by_family_id +
        get_entity_relations_by_family_id + get_entity_version_count 的 N+1 模式。

        Returns:
            [{"family_id", "entity", "relations", "version_count"}, ...]
        """
        if not family_ids:
            return []

        # 去重 + 解析 canonical IDs
        canonical_map: Dict[str, str] = {}  # original -> canonical
        canonical_set: List[str] = []
        for fid in family_ids:
            resolved = self.resolve_family_id(fid)
            if resolved and resolved not in canonical_map.values():
                canonical_map[fid] = resolved
                canonical_set.append(resolved)

        if not canonical_set:
            return [{"family_id": fid, "entity": None, "relations": [], "version_count": 0} for fid in family_ids]

        # Session 1: 批量获取实体 + 版本数
        with self._session() as session:
            result = session.run(
                f"""
                MATCH (e:Entity)
                WHERE e.family_id IN $fids AND e.invalid_at IS NULL
                WITH e.family_id AS fid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(COLLECT(e)) AS latest, COUNT(e) AS vcnt
                RETURN {_ENTITY_RETURN_FIELDS}, vcnt
                ORDER BY e.processed_time DESC
                """,
                fids=canonical_set,
            )
            records = list(result)

        entity_map: Dict[str, tuple] = {}  # family_id -> (entity, version_count)
        for record in records:
            entity = _neo4j_record_to_entity(record)
            vc = record.get("vcnt", 1)
            entity_map[entity.family_id] = (entity, vc)

        # Session 2: 批量获取所有相关关系
        all_aids = set()
        for entity, _ in entity_map.values():
            # 获取每个实体的所有 absolute_id
            all_aids.add(entity.absolute_id)

        # 还需要每个实体的所有版本 ID 才能找到关系
        with self._session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.family_id IN $fids AND e.invalid_at IS NULL
                RETURN e.family_id AS fid, e.uuid AS uuid
                """,
                fids=canonical_set,
            )
            fid_to_aids: Dict[str, List[str]] = {}
            for record in result:
                fid_to_aids.setdefault(record["fid"], []).append(record["uuid"])

        all_aids = set()
        for aids in fid_to_aids.values():
            all_aids.update(aids)

        relations_map: Dict[str, List] = {fid: [] for fid in canonical_set}
        if all_aids:
            with self._session() as session:
                result = session.run(
                    """
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id IN $aids OR r.entity2_absolute_id IN $aids)
                      AND r.invalid_at IS NULL
                    RETURN r.uuid AS uuid, r.family_id AS family_id,
                           r.entity1_absolute_id AS entity1_absolute_id,
                           r.entity2_absolute_id AS entity2_absolute_id,
                           r.content AS content, r.event_time AS event_time,
                           r.processed_time AS processed_time, r.episode_id AS episode_id,
                           r.source_document AS source_document
                    """,
                    aids=list(all_aids),
                )
                all_rels = [_neo4j_record_to_relation(rec) for rec in result]

            # 分配关系到对应的 family_id
            for rel in all_rels:
                for fid, aids in fid_to_aids.items():
                    if rel.entity1_absolute_id in aids or rel.entity2_absolute_id in aids:
                        relations_map[fid].append(rel)

        # 组装结果
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

    def get_graph_statistics(self) -> Dict[str, Any]:
        """返回图谱结构统计数据（仅统计有效版本，排除已失效的旧版本节点）"""
        cached = self._cache.get("graph_stats")
        if cached is not None:
            return cached
        with self._session() as session:
            # 有效实体数（去重 family_id，排除 invalidated）
            r = session.run("""
                MATCH (e:Entity)
                WHERE e.invalid_at IS NULL
                RETURN count(DISTINCT e.family_id) AS cnt
            """)
            entity_count = r.single()["cnt"]

            # 总节点数（含已失效旧版本）
            r = session.run("MATCH (e:Entity) RETURN count(e) AS cnt")
            total_entity_versions = r.single()["cnt"]

            # 有效关系数（去重 family_id）
            r = session.run("""
                MATCH (r:Relation)
                WHERE r.invalid_at IS NULL
                RETURN count(DISTINCT r.family_id) AS cnt
            """)
            relation_count = r.single()["cnt"]

            r = session.run("MATCH (r:Relation) RETURN count(r) AS cnt")
            total_relation_versions = r.single()["cnt"]

            stats = {
                "entity_count": entity_count,
                "relation_count": relation_count,
                "total_entity_versions": total_entity_versions,
                "total_relation_versions": total_relation_versions,
            }

            if entity_count > 0:
                # 收集所有有效 Relation 引用的 entity uuid
                r = session.run("""
                    MATCH (rel:Relation) WHERE rel.invalid_at IS NULL
                    WITH collect(DISTINCT rel.entity1_absolute_id)
                       + collect(DISTINCT rel.entity2_absolute_id) AS aids
                    UNWIND aids AS aid
                    RETURN collect(DISTINCT aid) AS connected_uuids
                """)
                row = r.single()
                connected_uuids = set(row["connected_uuids"]) if row and row["connected_uuids"] else set()

                # 计算每个 family_id 的度数（= 有多少个 uuid 被 Relation 引用）
                r = session.run("""
                    MATCH (e:Entity)
                    WHERE e.invalid_at IS NULL AND e.family_id IS NOT NULL
                    RETURN e.uuid AS uid, e.family_id AS fid
                """)
                fid_degrees = {}
                for rec in r:
                    uid = rec["uid"]
                    fid = rec["fid"]
                    deg = fid_degrees.get(fid, 0)
                    if uid in connected_uuids:
                        deg += 1
                    fid_degrees[fid] = deg

                if fid_degrees:
                    degrees = list(fid_degrees.values())
                    stats["avg_relations_per_entity"] = round(sum(degrees) / len(degrees), 2)
                    stats["max_relations_per_entity"] = max(degrees)
                else:
                    stats["avg_relations_per_entity"] = 0
                    stats["max_relations_per_entity"] = 0

                # 孤立实体（有效实体中不被任何有效 Relation 引用的）
                isolated_fids = [fid for fid, deg in fid_degrees.items() if deg == 0]
                stats["isolated_entities"] = len(isolated_fids)

                # 图密度
                if entity_count > 1:
                    max_possible = entity_count * (entity_count - 1) / 2
                    stats["graph_density"] = round(relation_count / max_possible, 4)
                else:
                    stats["graph_density"] = 0.0
            else:
                stats.update({
                    "avg_relations_per_entity": 0,
                    "max_relations_per_entity": 0,
                    "isolated_entities": 0,
                    "graph_density": 0.0,
                })

            # 时间趋势（基于有效实体）
            r = session.run("""
                MATCH (e:Entity)
                WHERE e.invalid_at IS NULL AND e.event_time IS NOT NULL
                WITH date(e.event_time) AS d, e.family_id AS fid
                RETURN d AS date, count(DISTINCT fid) AS cnt
                ORDER BY d
                LIMIT 30
            """)
            stats["entity_count_over_time"] = [{"date": str(r["date"]), "count": r["cnt"]} for r in r]

            r = session.run("""
                MATCH (r:Relation)
                WHERE r.invalid_at IS NULL AND r.event_time IS NOT NULL
                WITH date(r.event_time) AS d, r.family_id AS fid
                RETURN d AS date, count(DISTINCT fid) AS cnt
                ORDER BY d
                LIMIT 30
            """)
            stats["relation_count_over_time"] = [{"date": str(r["date"]), "count": r["cnt"]} for r in r]

        self._cache.set("graph_stats", stats, ttl=60)
        return stats

    # ------------------------------------------------------------------
    # 管理类方法：孤立实体、数据质量报告、版本清理
    # ------------------------------------------------------------------

    def get_isolated_entities(self, limit: int = 100, offset: int = 0) -> List[Entity]:
        """获取所有孤立实体（有效实体中不被任何有效 Relation 引用的）。"""
        with self._session() as session:
            # 先收集所有被有效 Relation 引用的 entity uuid
            r = session.run("""
                MATCH (rel:Relation) WHERE rel.invalid_at IS NULL
                WITH collect(DISTINCT rel.entity1_absolute_id)
                   + collect(DISTINCT rel.entity2_absolute_id) AS aids
                UNWIND aids AS aid
                RETURN collect(DISTINCT aid) AS connected
            """)
            row = r.single()
            connected = set(row["connected"]) if row and row["connected"] else set()

            # 获取每个 family_id 的最新有效版本
            r = session.run(f"""
                MATCH (e:Entity)
                WHERE e.invalid_at IS NULL AND e.family_id IS NOT NULL
                WITH e.family_id AS fid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(COLLECT(e)) AS e
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time DESC
            """)
            isolated = []
            for rec in r:
                uid = rec.get("uuid")
                if uid and uid not in connected:
                    isolated.append(_neo4j_record_to_entity(rec))
            return isolated[offset:offset + limit]

    def count_isolated_entities(self) -> int:
        """统计孤立实体数量。"""
        with self._session() as session:
            r = session.run("""
                MATCH (rel:Relation) WHERE rel.invalid_at IS NULL
                WITH collect(DISTINCT rel.entity1_absolute_id)
                   + collect(DISTINCT rel.entity2_absolute_id) AS aids
                UNWIND aids AS aid
                WITH collect(DISTINCT aid) AS connected
                MATCH (e:Entity)
                WHERE e.invalid_at IS NULL AND e.family_id IS NOT NULL
                  AND NOT e.uuid IN connected
                RETURN count(DISTINCT e.family_id) AS cnt
            """)
            row = r.single()
            return row["cnt"] if row else 0

    def get_data_quality_report(self) -> Dict[str, Any]:
        """返回数据质量报告。"""
        # Session 1: 所有计数查询（实体 + 关系）
        with self._session() as session:
            # 有效实体
            r = session.run("""
                MATCH (e:Entity) WHERE e.invalid_at IS NULL AND e.family_id IS NOT NULL
                RETURN count(DISTINCT e.family_id) AS valid_families, count(e) AS valid_nodes
            """)
            row = r.single()
            valid_families = row["valid_families"]
            valid_nodes = row["valid_nodes"]

            # 失效版本
            r = session.run("MATCH (e:Entity) WHERE e.invalid_at IS NOT NULL RETURN count(e) AS cnt")
            invalidated_entity_versions = r.single()["cnt"]

            # 无 family_id 的实体
            r = session.run("MATCH (e:Entity) WHERE e.family_id IS NULL RETURN count(e) AS cnt")
            no_family_id = r.single()["cnt"]

            # 有效关系
            r = session.run("""
                MATCH (rel:Relation) WHERE rel.invalid_at IS NULL
                RETURN count(DISTINCT rel.family_id) AS valid_families, count(rel) AS valid_nodes
            """)
            row = r.single()
            valid_relation_families = row["valid_families"]
            valid_relation_nodes = row["valid_nodes"]

            # 失效关系版本
            r = session.run("MATCH (rel:Relation) WHERE rel.invalid_at IS NOT NULL RETURN count(rel) AS cnt")
            invalidated_relation_versions = r.single()["cnt"]

        # 孤立实体
        isolated_count = self.count_isolated_entities()

        # Session 2: 悬空引用检测
        with self._session() as session:
            r = session.run("""
                MATCH (rel:Relation) WHERE rel.invalid_at IS NULL
                RETURN collect(DISTINCT rel.entity1_absolute_id) AS e1_ids,
                       collect(DISTINCT rel.entity2_absolute_id) AS e2_ids
            """)
            row = r.single()
            rel_aids = set()
            if row:
                if row.get("e1_ids"):
                    rel_aids.update(row["e1_ids"])
                if row.get("e2_ids"):
                    rel_aids.update(row["e2_ids"])

            r = session.run("MATCH (e:Entity) WHERE e.invalid_at IS NULL RETURN collect(DISTINCT e.uuid) AS uuids")
            row = r.single()
            valid_uuids = set(row["uuids"]) if row and row["uuids"] else set()

        dangling_refs = len(rel_aids - valid_uuids)

        return {
            "entities": {
                "valid_unique": valid_families,
                "valid_versions": valid_nodes,
                "invalidated_versions": invalidated_entity_versions,
                "no_family_id": no_family_id,
                "isolated": isolated_count,
            },
            "relations": {
                "valid_unique": valid_relation_families,
                "valid_versions": valid_relation_nodes,
                "invalidated_versions": invalidated_relation_versions,
                "dangling_entity_refs": dangling_refs,
            },
            "total_nodes": valid_nodes + invalidated_entity_versions + valid_relation_nodes + invalidated_relation_versions + no_family_id,
        }

    def cleanup_invalidated_versions(self, before_date: str = None, dry_run: bool = False) -> Dict[str, Any]:
        """清理已失效的旧版本节点。"""
        with self._session() as session:
            date_filter = ""
            params = {}
            if before_date:
                date_filter = " AND e.invalid_at < datetime($before_date)"
                params["before_date"] = before_date

            # Count entities to remove
            r = session.run(f"""
                MATCH (e:Entity) WHERE e.invalid_at IS NOT NULL {date_filter}
                RETURN count(e) AS cnt
            """, **params)
            entity_count = r.single()["cnt"]

            r = session.run(f"""
                MATCH (r:Relation) WHERE r.invalid_at IS NOT NULL {date_filter}
                RETURN count(r) AS cnt
            """, **params)
            relation_count = r.single()["cnt"]

            if dry_run:
                return {
                    "dry_run": True,
                    "entities_to_remove": entity_count,
                    "relations_to_remove": relation_count,
                    "message": f"预览：将删除 {entity_count} 个已失效实体版本和 {relation_count} 个已失效关系版本",
                }

            # Actually delete
            r = session.run(f"""
                MATCH (e:Entity) WHERE e.invalid_at IS NOT NULL {date_filter}
                DELETE e
                RETURN count(*) AS cnt
            """, **params)
            deleted_entities = r.single()["cnt"]

            r = session.run(f"""
                MATCH (r:Relation) WHERE r.invalid_at IS NOT NULL {date_filter}
                DELETE r
                RETURN count(*) AS cnt
            """, **params)
            deleted_relations = r.single()["cnt"]

            return {
                "dry_run": False,
                "deleted_entity_versions": deleted_entities,
                "deleted_relation_versions": deleted_relations,
                "message": f"已删除 {deleted_entities} 个已失效实体版本和 {deleted_relations} 个已失效关系版本",
            }

    def entity_has_any_relation(self, family_id: str) -> bool:
        """检查实体是否有关系。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return False
        with self._session() as session:
            result = session.run(
                """
                MATCH (r:Relation)
                WHERE r.entity1_absolute_id IN (
                    MATCH (e:Entity {family_id: $fid}) RETURN e.uuid
                ) OR r.entity2_absolute_id IN (
                    MATCH (e:Entity {family_id: $fid}) RETURN e.uuid
                )
                RETURN COUNT(r) AS cnt LIMIT 1
                """,
                fid=family_id,
            )
            record = result.single()
            return (record["cnt"] or 0) > 0

    def delete_orphan_entities(self, candidate_family_ids: list) -> list:
        """批量检查并删除没有关系的实体。"""
        if not candidate_family_ids:
            return []
        with self._write_lock:
            with self._session() as session:
                # 找出候选中无关系的 family_id
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.family_id IN $fids
                    AND NOT (
                        (e)-[:RELATES_TO]-(:Entity)
                    )
                    RETURN DISTINCT e.family_id AS fid
                    """,
                    fids=candidate_family_ids,
                )
                orphan_ids = [record["fid"] for record in result]
                if orphan_ids:
                    session.run(
                        """
                        MATCH (e:Entity)
                        WHERE e.family_id IN $fids
                        DETACH DELETE e
                        """,
                        fids=orphan_ids,
                    )
                    # 同步删除向量
                    self._vector_store.delete_batch("entity_vectors", orphan_ids)
                return orphan_ids

    def delete_entity_by_id(self, family_id: str) -> int:
        """删除实体的所有版本。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        with self._write_lock:
            with self._session() as session:
                # 获取所有 absolute_id
                result = session.run(
                    "MATCH (e:Entity {family_id: $fid}) RETURN e.uuid AS uuid",
                    fid=family_id,
                )
                uuids = [record["uuid"] for record in result]
                count = len(uuids)
                if uuids:
                    session.run(
                        """
                        MATCH (e:Entity {family_id: $fid})
                        DETACH DELETE e
                        """,
                        fid=family_id,
                    )
                    self._vector_store.delete_batch("entity_vectors", uuids)
                self._cache.invalidate("entity:")
                self._cache.invalidate("resolve:")
                self._cache.invalidate("sim_search:")
                self._cache.invalidate("graph_stats")
                return count

    def delete_relation_by_id(self, family_id: str) -> int:
        """删除关系的所有版本。返回删除的行数。"""
        with self._relation_write_lock:
            with self._session() as session:
                # 删除关系节点
                result = session.run(
                    "MATCH (r:Relation {family_id: $fid}) DETACH DELETE r RETURN count(r) AS cnt",
                    fid=family_id,
                )
                record = result.single()
                count = record["cnt"] if record else 0
                # 清理向量存储
                try:
                    self._vector_store.delete_batch("relation_vectors",
                        [r.absolute_id for r in self.get_relation_versions(family_id)])
                except Exception as e:
                    logger.warning("Failed to clean up relation vectors for %s: %s", family_id, e)
                    pass
                self._cache.invalidate("relation:")
                self._cache.invalidate("graph_stats")
                return count

    def delete_entity_all_versions(self, family_id: str) -> int:
        """删除实体的所有版本（含关系边）。返回删除的行数。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        with self._write_lock:
            with self._session() as session:
                # 删除相关关系
                session.run(
                    """MATCH (e:Entity {family_id: $fid})-[r:RELATES_TO]-()
                       DETACH DELETE r""",
                    fid=family_id,
                )
                # 删除实体节点
                result = session.run(
                    "MATCH (e:Entity {family_id: $fid}) DETACH DELETE e RETURN count(e) AS cnt",
                    fid=family_id,
                )
                record = result.single()
                count = record["cnt"] if record else 0
                # 清理向量存储
                try:
                    self._vector_store.delete_batch("entity_vectors",
                        [e.absolute_id for e in self.get_entity_versions(family_id)])
                except Exception as e:
                    logger.warning("Failed to clean up entity vectors for %s: %s", family_id, e)
                self._cache.invalidate("entity:")
                self._cache.invalidate("resolve:")
                self._cache.invalidate("sim_search:")
                self._cache.invalidate("graph_stats")
                return count

    def delete_relation_all_versions(self, family_id: str) -> int:
        """删除关系的所有版本。返回删除的行数。"""
        return self.delete_relation_by_id(family_id)

    def batch_delete_entities(self, family_ids: List[str]) -> int:
        """批量删除实体 — 单次事务，替代 N 次 DETACH DELETE。"""
        resolved_map = self.resolve_family_ids(family_ids)
        resolved = list(set(r for r in resolved_map.values() if r))
        if not resolved:
            return 0
        with self._write_lock:
            with self._session() as session:
                result = session.run(
                    "UNWIND $fids AS fid MATCH (e:Entity {family_id: fid}) DETACH DELETE e RETURN count(e) AS cnt",
                    fids=resolved,
                )
                record = result.single()
                count = record["cnt"] if record else 0
            self._cache.invalidate("entity:")
            self._cache.invalidate("resolve:")
            self._cache.invalidate("sim_search:")
            self._cache.invalidate("graph_stats")
            return count

    def batch_delete_relations(self, family_ids: List[str]) -> int:
        """批量删除关系 — 单次事务，替代 N 次删除。"""
        if not family_ids:
            return 0
        with self._write_lock:
            with self._session() as session:
                result = session.run(
                    "UNWIND $fids AS fid MATCH (r:Relation {family_id: fid}) DELETE r RETURN count(r) AS cnt",
                    fids=family_ids,
                )
                record = result.single()
                count = record["cnt"] if record else 0
            self._cache.invalidate("relation:")
            self._cache.invalidate("graph_stats")
            return count

    def get_family_ids_by_names(self, names: list) -> dict:
        """按名称批量查询 family_id。"""
        if not names:
            return {}
        with self._session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.name IN $names
                RETURN e.name AS name, e.family_id AS family_id
                ORDER BY e.processed_time DESC
                """,
                names=names,
            )
            output = {}
            for record in result:
                name = record["name"]
                if name not in output:
                    output[name] = self.resolve_family_id(record["family_id"])
            return output

    def find_entity_by_name_prefix(self, prefix: str, limit: int = 5) -> list:
        """查找名称以 prefix 开头的实体（处理消歧括号场景）。
        例如 prefix="Go语言" 可匹配 "Go语言（Golang）"。
        返回 Entity 对象列表，按 processed_time 倒序。
        """
        if not prefix:
            return []
        try:
            with self._session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE (e.name STARTS WITH $prefix OR e.name = $prefix)
                      AND e.invalid_at IS NULL
                    RETURN e.uuid AS uuid, e.family_id AS family_id,
                           e.name AS name, e.content AS content,
                           e.summary AS summary,
                           e.attributes AS attributes, e.confidence AS confidence,
                           e.source_document AS source_document, e.episode_id AS episode_id,
                           e.doc_name AS doc_name,
                           e.processed_time AS processed_time, e.event_time AS event_time,
                           e.content_format AS content_format
                    ORDER BY e.processed_time DESC
                    LIMIT $limit
                    """,
                    prefix=prefix,
                    limit=limit,
                )
                entities = []
                seen_fids = set()
                for record in result:
                    fid = record.get("family_id")
                    if fid and fid not in seen_fids:
                        seen_fids.add(fid)
                        entities.append(_neo4j_record_to_entity(record))
                return entities
        except Exception as e:
            logger.debug("find_entity_by_name_prefix failed for '%s': %s", prefix, e)
            return []

    # ------------------------------------------------------------------
    # BM25 Full-Text Search
    # ------------------------------------------------------------------

    def search_entities_by_bm25(self, query: str, limit: int = 20) -> List[Entity]:
        """BM25 全文搜索实体（Neo4j 5.x 全文索引），去重 family_id 只保留最高分版本。"""
        if not query:
            return []
        try:
            with self._session() as session:
                # 多取一些再在 Python 层去重，确保 limit 个 unique family_id
                raw_limit = min(limit * 5, 500)
                result = session.run(
                    """CALL db.index.fulltext.queryNodes('entityFulltext', $search_query)
                       YIELD node, score
                       WHERE node.invalid_at IS NULL
                       RETURN node.uuid AS uuid, node.family_id AS family_id,
                              node.name AS name, node.content AS content,
                              node.summary AS summary,
                              node.attributes AS attributes, node.confidence AS confidence,
                              node.content_format AS content_format, node.community_id AS community_id,
                              node.valid_at AS valid_at, node.invalid_at AS invalid_at,
                              node.event_time AS event_time,
                              node.processed_time AS processed_time,
                              node.episode_id AS episode_id,
                              node.source_document AS source_document,
                              score AS bm25_score
                       ORDER BY score DESC
                       LIMIT $raw_limit""",
                    search_query=query, raw_limit=raw_limit,
                )
                seen_fids = set()
                entities = []
                for record in result:
                    fid = record.get("family_id")
                    if fid and fid in seen_fids:
                        continue
                    if fid:
                        seen_fids.add(fid)
                    entities.append(_neo4j_record_to_entity(record))
                    if len(entities) >= limit:
                        break
                return entities
        except Exception as e:
            logger.warning("BM25 search failed, falling back to empty: %s", e)
            return []

    def search_relations_by_bm25(self, query: str, limit: int = 20) -> List[Relation]:
        """BM25 全文搜索关系（Neo4j 5.x 全文索引），去重 family_id 只保留最高分版本。"""
        if not query:
            return []
        try:
            with self._session() as session:
                raw_limit = min(limit * 5, 500)
                result = session.run(
                    """CALL db.index.fulltext.queryNodes('relationFulltext', $search_query)
                       YIELD node, score
                       WHERE node.invalid_at IS NULL
                       RETURN node.uuid AS uuid, node.family_id AS family_id,
                              node.entity1_absolute_id AS entity1_absolute_id,
                              node.entity2_absolute_id AS entity2_absolute_id,
                              node.content AS content,
                              node.event_time AS event_time,
                              node.processed_time AS processed_time,
                              node.episode_id AS episode_id,
                              node.source_document AS source_document,
                              score AS bm25_score
                       ORDER BY score DESC
                       LIMIT $raw_limit""",
                    search_query=query, raw_limit=raw_limit,
                )
                seen_fids = set()
                relations = []
                for record in result:
                    fid = record.get("family_id")
                    if fid and fid in seen_fids:
                        continue
                    if fid:
                        seen_fids.add(fid)
                    relations.append(_neo4j_record_to_relation(record))
                    if len(relations) >= limit:
                        break
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
        # 结果缓存
        cache_key = f"sim_search:{hash(query_name)}:{hash(query_content or '')}:{threshold}:{max_results}:{text_mode}:{similarity_method}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        with _perf_timer("search_entities_by_similarity"):
            if text_mode == "name_only":
                query_text = query_name
                use_content = False
            elif text_mode == "content_only":
                if not query_content:
                    self._cache.set(cache_key, [], ttl=30)
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
                # KNN 路径：不需要加载全量数据
                result = self._search_with_embedding(
                    query_text, [], threshold,
                    use_content, max_results, content_snippet_length, text_mode
                )
            else:
                # BM25 替代 Jaccard 全量扫描，走 Neo4j 全文索引
                result = self.search_entities_by_bm25(query_text, limit=max_results * 3)
                result = result[:max_results]
            self._cache.set(cache_key, result, ttl=30)
            return result

    def _search_with_embedding(self, query_text: str, entities_with_embeddings: List[tuple],
                                threshold: float, use_content: bool = False,
                                max_results: int = 10, content_snippet_length: int = 50,
                                text_mode: str = "name_and_content") -> List[Entity]:
        """使用 sqlite-vec KNN 进行实体相似度搜索。"""
        # 1. Encode + 归一化 query
        query_embedding = self.embedding_client.encode(query_text)
        if query_embedding is None:
            return self.search_entities_by_bm25(query_text, limit=max_results * 3)[:max_results]

        query_emb = np.array(
            query_embedding[0] if isinstance(query_embedding, (list, np.ndarray)) else query_embedding,
            dtype=np.float32
        )
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        # 2. KNN top-k（多取几倍候选，因为同一 family_id 可能有多个版本）
        knn_limit = max_results * 5
        knn_results = self._vector_store.search(
            "entity_vectors", query_emb.tolist(), limit=knn_limit
        )
        # knn_results: [(uuid, l2_dist_sq), ...]

        if not knn_results:
            return []

        # 3. L2 距离转余弦相似度: cos_sim = 1 - l2_dist_sq / 2
        uuid_dist = {uuid: dist for uuid, dist in knn_results}

        # 4. 批量获取实体（一次 Neo4j 查询）
        uuids = [uuid for uuid, _ in knn_results]
        entities = self.get_entities_by_absolute_ids(uuids)

        # 5. 过滤 threshold + 去重（同 family_id 取最新，即 KNN 中距离最小的）
        seen = set()
        results = []
        for entity in entities:
            if entity is None:
                continue
            l2_dist_sq = uuid_dist.get(entity.absolute_id)
            if l2_dist_sq is None:
                continue
            cos_sim = 1.0 - l2_dist_sq / 2.0
            if cos_sim >= threshold and entity.family_id not in seen:
                results.append(entity)
                seen.add(entity.family_id)
                if len(results) >= max_results:
                    break
        return results

    # ------------------------------------------------------------------
    # Relation 操作
    # ------------------------------------------------------------------

    def save_relation(self, relation: Relation):
        """保存关系到 Neo4j + sqlite-vec（合并为单条 Cypher）。"""
        self._invalidate_emb_cache()
        with _perf_timer("save_relation"):
            self._save_relation_impl(relation)

    def _save_relation_impl(self, relation: Relation):
        """保存关系的实际实现。"""
        embedding_blob = self._compute_relation_embedding(relation)
        relation.embedding = embedding_blob

        valid_at = (relation.valid_at or relation.event_time).isoformat()

        with self._relation_write_lock:
            with self._session() as session:
                session.run(
                    """
                    MERGE (r:Relation {uuid: $uuid})
                    SET r.family_id = $family_id,
                        r.entity1_absolute_id = $e1_abs,
                        r.entity2_absolute_id = $e2_abs,
                        r.content = $content,
                        r.event_time = datetime($event_time),
                        r.processed_time = datetime($processed_time),
                        r.episode_id = $cache_id,
                        r.source_document = $source,
                        r.summary = $summary,
                        r.attributes = $attributes,
                        r.confidence = $confidence,
                        r.provenance = $provenance,
                        r.content_format = $content_format,
                        r.valid_at = datetime($valid_at)
                    WITH $uuid AS abs_id, $family_id AS fid, $event_time AS et
                    MATCH (r:Relation {family_id: fid})
                    WHERE r.uuid <> abs_id AND r.invalid_at IS NULL
                    SET r.invalid_at = datetime(et)
                    WITH $uuid AS rel_uuid, $e1_abs AS e1, $e2_abs AS e2, $content AS fact
                    MATCH (n1:Entity {uuid: e1})
                    MATCH (n2:Entity {uuid: e2})
                    MERGE (n1)-[rel:RELATES_TO {relation_uuid: rel_uuid}]->(n2)
                    SET rel.fact = fact
                    """,
                    uuid=relation.absolute_id,
                    family_id=relation.family_id,
                    e1_abs=relation.entity1_absolute_id,
                    e2_abs=relation.entity2_absolute_id,
                    content=relation.content,
                    event_time=relation.event_time.isoformat(),
                    processed_time=relation.processed_time.isoformat(),
                    cache_id=relation.episode_id,
                    source=relation.source_document,
                    summary=relation.summary,
                    attributes=relation.attributes,
                    confidence=relation.confidence,
                    provenance=relation.provenance,
                    content_format=getattr(relation, "content_format", "plain"),
                    valid_at=valid_at,
                )

            if embedding_blob:
                emb_list = list(np.frombuffer(embedding_blob, dtype=np.float32))
                self._vector_store.upsert("relation_vectors", relation.absolute_id, emb_list)

        self._cache.invalidate("relation:")
        self._cache.invalidate("graph_stats")

    def bulk_save_relations(self, relations: List[Relation]):
        """批量保存关系（UNWIND 批量写入）。"""
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

        with self._relation_write_lock:
            vec_items = []
            rows = []
            for idx, relation in enumerate(relations):
                embedding_blob = None
                if embeddings is not None:
                    try:
                        emb_arr = np.array(embeddings[idx], dtype=np.float32)
                        norm = np.linalg.norm(emb_arr)
                        if norm > 0:
                            emb_arr = emb_arr / norm
                        embedding_blob = emb_arr.tobytes()
                    except Exception as e:
                        logger.debug("Embedding decode failed for relation index %d: %s", idx, e)
                        embedding_blob = None
                relation.embedding = embedding_blob

                rows.append({
                    "uuid": relation.absolute_id,
                    "family_id": relation.family_id,
                    "e1_abs": relation.entity1_absolute_id,
                    "e2_abs": relation.entity2_absolute_id,
                    "content": relation.content,
                    "event_time": relation.event_time.isoformat(),
                    "processed_time": relation.processed_time.isoformat(),
                    "cache_id": relation.episode_id,
                    "source": relation.source_document,
                })

                if embedding_blob:
                    emb_list = list(np.frombuffer(embedding_blob, dtype=np.float32))
                    vec_items.append((relation.absolute_id, emb_list))

            # 一次 UNWIND 替代 N 次 session.run
            with self._session() as session:
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (r:Relation {uuid: row.uuid})
                    SET r.family_id = row.family_id,
                        r.entity1_absolute_id = row.e1_abs,
                        r.entity2_absolute_id = row.e2_abs,
                        r.content = row.content,
                        r.event_time = datetime(row.event_time),
                        r.processed_time = datetime(row.processed_time),
                        r.episode_id = row.cache_id,
                        r.source_document = row.source
                    WITH row
                    MATCH (n1:Entity {uuid: row.e1_abs})
                    MATCH (n2:Entity {uuid: row.e2_abs})
                    MERGE (n1)-[rel:RELATES_TO {relation_uuid: row.uuid}]->(n2)
                    SET rel.fact = row.content
                    """,
                    rows=rows,
                )

            if vec_items:
                self._vector_store.upsert_batch("relation_vectors", vec_items)

        self._cache.invalidate("relation:")
        self._cache.invalidate("graph_stats")

    def get_relation_by_absolute_id(self, relation_absolute_id: str) -> Optional[Relation]:
        """根据 absolute_id 获取关系。"""
        with self._session() as session:
            result = session.run(
                """
                MATCH (r:Relation {uuid: $uuid})
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.episode_id AS episode_id,
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

    def get_relations_by_absolute_ids(self, absolute_ids: List[str]) -> List[Relation]:
        """批量根据 absolute_id 获取关系。"""
        if not absolute_ids:
            return []
        with self._session() as session:
            result = session.run(
                """
                MATCH (r:Relation)
                WHERE r.uuid IN $uuids
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.episode_id AS episode_id,
                       r.source_document AS source_document
                """,
                uuids=absolute_ids,
            )
            return [_neo4j_record_to_relation(r) for r in result]

    def get_relation_by_family_id(self, family_id: str) -> Optional[Relation]:
        with self._session() as session:
            result = session.run(
                """
                MATCH (r:Relation {family_id: $fid})
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.episode_id AS episode_id,
                       r.source_document AS source_document
                ORDER BY r.processed_time DESC LIMIT 1
                """,
                fid=family_id,
            )
            record = result.single()
            if not record:
                return None
            relation = _neo4j_record_to_relation(record)
            emb = self._vector_store.get("relation_vectors", relation.absolute_id)
            if emb:
                relation.embedding = np.array(emb, dtype=np.float32).tobytes()
            return relation

    def get_relations_by_entities(self, from_family_id: str, to_family_id: str) -> List[Relation]:
        """根据两个 family_id 获取所有关系（合并为 2 次 session 查询）。"""
        with _perf_timer("get_relations_by_entities"):
            return self._get_relations_by_entities_impl(from_family_id, to_family_id)

    def _get_relations_by_entities_impl(self, from_family_id: str, to_family_id: str) -> List[Relation]:
        """根据两个 family_id 获取所有关系（实际实现）。"""
        from_family_id = self.resolve_family_id(from_family_id)
        to_family_id = self.resolve_family_id(to_family_id)
        if not from_family_id or not to_family_id:
            return []

        with self._session() as session:
            # Step 1: 批量获取两个 family_id 的所有 absolute_id（合并 2 次 resolve + 2 次 _get_all_absolute_ids）
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.family_id IN [$fid1, $fid2]
                WITH e.family_id AS fid, collect(e.uuid) AS abs_ids
                RETURN fid, abs_ids
                """,
                fid1=from_family_id,
                fid2=to_family_id,
            )
            fid_to_abs: Dict[str, List[str]] = {}
            for record in result:
                fid_to_abs[record["fid"]] = record["abs_ids"]

            from_ids = fid_to_abs.get(from_family_id, [])
            to_ids = fid_to_abs.get(to_family_id, [])
            if not from_ids or not to_ids:
                return []

            # Step 2: 查询关系
            result = session.run(
                """
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id IN $from_ids AND r.entity2_absolute_id IN $to_ids)
                   OR (r.entity1_absolute_id IN $to_ids AND r.entity2_absolute_id IN $from_ids)
                WITH r.family_id AS fid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH fid, r ORDER BY r.processed_time DESC
                WITH fid, HEAD(COLLECT(r)) AS r
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.episode_id AS episode_id,
                       r.source_document AS source_document
                ORDER BY r.processed_time DESC
                """,
                from_ids=from_ids,
                to_ids=to_ids,
            )
            return [_neo4j_record_to_relation(r) for r in result]

    def get_relations_by_entity_pairs(self, entity_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], List[Relation]]:
        """批量获取多个实体对的关系。"""
        if not entity_pairs:
            return {}

        # 收集所有唯一的 family_id
        all_family_ids = set()
        for e1, e2 in entity_pairs:
            all_family_ids.add(e1)
            all_family_ids.add(e2)

        # 单次查询获取所有相关的绝对 ID
        with self._session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.family_id IN $fids AND e.invalid_at IS NULL RETURN e.family_id AS fid, e.uuid AS uuid",
                fids=list(all_family_ids),
            )
            fid_to_aids: Dict[str, List[str]] = {}
            for record in result:
                fid_to_aids.setdefault(record["fid"], []).append(record["uuid"])

        # 构建所有绝对 ID 集合
        all_aids = set()
        for aids in fid_to_aids.values():
            all_aids.update(aids)

        if not all_aids:
            return {tuple(sorted((e1, e2))): [] for e1, e2 in entity_pairs}

        # 单次查询获取所有相关关系
        with self._session() as session:
            result = session.run(
                """
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id IN $aids OR r.entity2_absolute_id IN $aids)
                  AND r.invalid_at IS NULL
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.episode_id AS episode_id,
                       r.source_document AS source_document
                """,
                aids=list(all_aids),
            )
            all_relations = [_neo4j_record_to_relation(rec) for rec in result]

        # 按 family_id 对的绝对 ID 组合进行分组
        results: Dict[Tuple[str, str], List[Relation]] = {}
        for e1_fid, e2_fid in entity_pairs:
            pair_key = tuple(sorted((e1_fid, e2_fid)))
            if pair_key in results:
                continue
            e1_aids = set(fid_to_aids.get(e1_fid, []))
            e2_aids = set(fid_to_aids.get(e2_fid, []))
            pair_rels = [
                rel for rel in all_relations
                if (rel.entity1_absolute_id in e1_aids and rel.entity2_absolute_id in e2_aids)
                or (rel.entity1_absolute_id in e2_aids and rel.entity2_absolute_id in e1_aids)
            ]
            results[pair_key] = pair_rels

        return results

    def _get_all_absolute_ids_for_entity(self, family_id: str) -> List[str]:
        """获取实体的所有版本的 absolute_id。"""
        with self._session() as session:
            result = session.run(
                "MATCH (e:Entity {family_id: $fid}) RETURN e.uuid AS uuid",
                fid=family_id,
            )
            return [record["uuid"] for record in result]


    def get_relation_versions(self, family_id: str) -> List[Relation]:
        """获取关系的所有版本。"""
        with self._session() as session:
            result = session.run(
                """
                MATCH (r:Relation {family_id: $fid})
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.episode_id AS episode_id,
                       r.source_document AS source_document
                ORDER BY r.processed_time ASC
                """,
                fid=family_id,
            )
            return [_neo4j_record_to_relation(r) for r in result]

    def update_relation_episode_id(self, family_id: str, episode_id: str):
        """更新关系的 episode_id。"""
        with self._relation_write_lock:
            with self._session() as session:
                session.run(
                    """
                    MATCH (r:Relation {family_id: $fid})
                    SET r.episode_id = $cache_id
                    """,
                    fid=family_id,
                    cache_id=episode_id,
                )

    def get_self_referential_relations(self) -> Dict[str, List[Dict]]:
        """获取自引用关系。"""
        with self._session() as session:
            result = session.run(
                """
                MATCH (r:Relation)
                WHERE r.entity1_absolute_id = r.entity2_absolute_id
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.content AS content
                """
            )
            grouped: Dict[str, List[Dict]] = {}
            for record in result:
                eid = record["entity1_absolute_id"]
                grouped.setdefault(eid, []).append({
                    "uuid": record["uuid"],
                    "family_id": record["family_id"],
                    "content": record["content"],
                })
            return grouped

    def delete_self_referential_relations(self) -> int:
        """删除所有自引用关系。"""
        with self._relation_write_lock:
            with self._session() as session:
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

    def get_self_referential_relations_for_entity(self, family_id: str) -> List[Dict]:
        """获取指定实体的自引用关系（包括合并后两端指向同一 family 的关系）。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        abs_ids = self._get_all_absolute_ids_for_entity(family_id)
        if not abs_ids:
            return []
        with self._session() as session:
            result = session.run(
                """
                MATCH (r:Relation)
                WHERE r.entity1_absolute_id IN $abs_ids AND r.entity2_absolute_id IN $abs_ids
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.content AS content
                """,
                abs_ids=abs_ids,
            )
            return [
                {
                    "uuid": record["uuid"],
                    "family_id": record["family_id"],
                    "entity1_absolute_id": record["entity1_absolute_id"],
                    "content": record["content"],
                }
                for record in result
            ]

    def delete_self_referential_relations_for_entity(self, family_id: str) -> int:
        """删除指定实体的自引用关系。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        abs_ids = self._get_all_absolute_ids_for_entity(family_id)
        if not abs_ids:
            return 0
        with self._relation_write_lock:
            with self._session() as session:
                result = session.run(
                    """
                    MATCH (r:Relation)
                    WHERE r.entity1_absolute_id IN $abs_ids AND r.entity2_absolute_id IN $abs_ids
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
        with self._session() as session:
            if time_point:
                query = """
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id = $abs_id OR r.entity2_absolute_id = $abs_id)
                    AND r.event_time <= datetime($tp)
                    WITH r.family_id AS fid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH fid, r ORDER BY r.processed_time DESC
                    WITH fid, HEAD(COLLECT(r)) AS r
                    RETURN r.uuid AS uuid, r.family_id AS family_id,
                           r.entity1_absolute_id AS entity1_absolute_id,
                           r.entity2_absolute_id AS entity2_absolute_id,
                           r.content AS content, r.event_time AS event_time,
                           r.processed_time AS processed_time, r.episode_id AS episode_id,
                           r.source_document AS source_document
                    ORDER BY r.processed_time DESC
                """
                params = {"abs_id": entity_absolute_id, "tp": time_point.isoformat()}
            else:
                query = """
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id = $abs_id OR r.entity2_absolute_id = $abs_id)
                    WITH r.family_id AS fid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH fid, r ORDER BY r.processed_time DESC
                    WITH fid, HEAD(COLLECT(r)) AS r
                    RETURN r.uuid AS uuid, r.family_id AS family_id,
                           r.entity1_absolute_id AS entity1_absolute_id,
                           r.entity2_absolute_id AS entity2_absolute_id,
                           r.content AS content, r.event_time AS event_time,
                           r.processed_time AS processed_time, r.episode_id AS episode_id,
                           r.source_document AS source_document
                    ORDER BY r.processed_time DESC
                """
                params = {"abs_id": entity_absolute_id}

            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = session.run(query, **params)
            return [_neo4j_record_to_relation(r) for r in result]

    def get_entity_relations_by_family_id(self, family_id: str, limit: Optional[int] = None,
                                           time_point: Optional[datetime] = None,
                                           max_version_absolute_id: Optional[str] = None) -> List[Relation]:
        """通过 family_id 获取实体的所有关系（包含所有版本）。"""
        with _perf_timer("get_entity_relations_by_family_id"):
            return self._get_entity_relations_by_family_id_impl(family_id, limit, time_point, max_version_absolute_id)

    def _get_entity_relations_by_family_id_impl(self, family_id: str, limit: Optional[int] = None,
                                                 time_point: Optional[datetime] = None,
                                                 max_version_absolute_id: Optional[str] = None) -> List[Relation]:
        """通过 family_id 获取实体的所有关系（实际实现）。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        abs_ids = self._get_all_absolute_ids_for_entity(family_id)
        if not abs_ids:
            return []
        if max_version_absolute_id:
            # 获取从最早到 max_version 的所有 absolute_id（拆分为两次查询，避免嵌套 MATCH 语法错误）
            with self._session() as session:
                result = session.run(
                    "MATCH (e2:Entity {uuid: $max_abs}) RETURN e2.processed_time AS max_pt",
                    max_abs=max_version_absolute_id,
                )
                record = result.single()
                if record and record["max_pt"]:
                    max_pt = record["max_pt"]
                    result = session.run(
                        """
                        MATCH (e:Entity {family_id: $fid})
                        WHERE e.processed_time <= $max_pt
                        RETURN e.uuid AS uuid
                        ORDER BY e.processed_time ASC
                        """,
                        fid=family_id,
                        max_pt=max_pt,
                    )
                    abs_ids = [r["uuid"] for r in result]

        if not abs_ids:
            return []

        with self._session() as session:
            if time_point:
                query = """
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id IN $abs_ids OR r.entity2_absolute_id IN $abs_ids)
                    AND r.event_time <= datetime($tp)
                    WITH r.family_id AS fid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH fid, r ORDER BY r.processed_time DESC
                    WITH fid, HEAD(COLLECT(r)) AS r
                    RETURN r.uuid AS uuid, r.family_id AS family_id,
                           r.entity1_absolute_id AS entity1_absolute_id,
                           r.entity2_absolute_id AS entity2_absolute_id,
                           r.content AS content, r.event_time AS event_time,
                           r.processed_time AS processed_time, r.episode_id AS episode_id,
                           r.source_document AS source_document
                    ORDER BY r.processed_time DESC
                """
                params = {"abs_ids": abs_ids, "tp": time_point.isoformat()}
            else:
                query = """
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id IN $abs_ids OR r.entity2_absolute_id IN $abs_ids)
                    WITH r.family_id AS fid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH fid, r ORDER BY r.processed_time DESC
                    WITH fid, HEAD(COLLECT(r)) AS r
                    RETURN r.uuid AS uuid, r.family_id AS family_id,
                           r.entity1_absolute_id AS entity1_absolute_id,
                           r.entity2_absolute_id AS entity2_absolute_id,
                           r.content AS content, r.event_time AS event_time,
                           r.processed_time AS processed_time, r.episode_id AS episode_id,
                           r.source_document AS source_document
                    ORDER BY r.processed_time DESC
                """
                params = {"abs_ids": abs_ids}

            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = session.run(query, **params)
            return [_neo4j_record_to_relation(r) for r in result]

    def get_entity_relations_timeline(self, family_id: str, version_abs_ids: List[str]) -> List[Dict]:
        """批量获取实体在各版本时间点的关系（消除 N+1 查询）。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id or not version_abs_ids:
            return []
        abs_ids = self._get_all_absolute_ids_for_entity(family_id)
        if not abs_ids:
            return []

        with self._session() as session:
            # 获取各版本的 processed_time
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.uuid IN $version_abs_ids
                RETURN e.uuid AS uuid, e.processed_time AS pt
                ORDER BY e.pt ASC
                """,
                version_abs_ids=version_abs_ids,
            )
            version_times = [(record["uuid"], record["pt"]) for record in result]
            if not version_times:
                return []

            # 获取所有相关关系（一次查询）
            result = session.run(
                """
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id IN $abs_ids OR r.entity2_absolute_id IN $abs_ids)
                WITH r.family_id AS fid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH fid, r ORDER BY r.processed_time DESC
                WITH fid, HEAD(COLLECT(r)) AS r
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time
                ORDER BY r.processed_time ASC
                """,
                abs_ids=abs_ids,
            )

            # 按 processed_time 过滤：只返回在每个版本时间点之前出现的关系
            timeline = []
            seen = set()
            for record in result:
                if record["uuid"] in seen:
                    continue
                rel_pt = record["processed_time"]
                for v_uuid, v_pt in version_times:
                    if rel_pt and v_pt and rel_pt <= v_pt:
                        seen.add(record["uuid"])
                        timeline.append({
                            "family_id": record["family_id"],
                            "content": record["content"],
                            "event_time": record["event_time"].isoformat() if record["event_time"] else None,
                            "absolute_id": record["uuid"],
                        })
                        break
            return timeline

    def get_relations_by_entity_absolute_ids(self, entity_absolute_ids: List[str],
                                              limit: Optional[int] = None) -> List[Relation]:
        """根据 absolute_id 列表获取关系。"""
        if not entity_absolute_ids:
            return []
        with self._session() as session:
            query = """
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id IN $abs_ids OR r.entity2_absolute_id IN $abs_ids)
                WITH r.family_id AS fid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH fid, r ORDER BY r.processed_time DESC
                WITH fid, HEAD(COLLECT(r)) AS r
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.episode_id AS episode_id,
                       r.source_document AS source_document
                ORDER BY r.processed_time DESC
            """
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = session.run(query, abs_ids=entity_absolute_ids)
            return [_neo4j_record_to_relation(r) for r in result]

    def get_entity_absolute_ids_up_to_version(self, family_id: str, max_absolute_id: str) -> List[str]:
        """获取从最早版本到指定版本的所有 absolute_id。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        with self._session() as session:
            result = session.run(
                """
                MATCH (e:Entity {family_id: $fid})
                WHERE e.processed_time <= (
                    MATCH (e2:Entity {uuid: $max_abs}) RETURN e2.processed_time
                )
                RETURN e.uuid AS uuid
                ORDER BY e.processed_time ASC
                """,
                fid=family_id,
                max_abs=max_absolute_id,
            )
            return [r["uuid"] for r in result]

    def get_all_relations(self, limit: Optional[int] = None, offset: Optional[int] = None,
                           exclude_embedding: bool = False) -> List[Relation]:
        """获取所有关系的最新版本。"""
        with self._session() as session:
            query = """
                MATCH (r:Relation)
                WITH r.family_id AS fid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH fid, r ORDER BY r.processed_time DESC
                WITH fid, HEAD(COLLECT(r)) AS r
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.episode_id AS episode_id,
                       r.source_document AS source_document
                ORDER BY r.processed_time DESC
            """
            if offset is not None and offset > 0:
                query += f" SKIP {int(offset)}"
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = session.run(query)
            records = list(result)

        relations = [_neo4j_record_to_relation(r) for r in records]

        if not exclude_embedding and relations:
            uuids = [rel.absolute_id for rel in relations]
            emb_map = self._vector_store.get_batch("relation_vectors", uuids)
            for rel in relations:
                emb_list = emb_map.get(rel.absolute_id)
                if emb_list:
                    rel.embedding = np.array(emb_list, dtype=np.float32).tobytes()

        return relations

    def _get_relations_with_embeddings(self) -> List[tuple]:
        """获取所有关系的最新版本及其 embedding（带短 TTL 缓存）。"""
        import time as _time
        now = _time.time()
        if self._relation_emb_cache is not None and (now - self._relation_emb_cache_ts) < self._emb_cache_ttl:
            return self._relation_emb_cache
        with _perf_timer("_get_relations_with_embeddings"):
            result = self._get_relations_with_embeddings_impl()
        self._relation_emb_cache = result
        self._relation_emb_cache_ts = _time.time()
        return result

    def _get_relations_with_embeddings_impl(self) -> List[tuple]:
        """获取所有关系的最新版本及其 embedding（实际实现）。"""
        with self._session() as session:
            result = session.run(
                """
                MATCH (r:Relation)
                WITH r.family_id AS fid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH fid, r ORDER BY r.processed_time DESC
                WITH fid, HEAD(COLLECT(r)) AS r
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.episode_id AS episode_id,
                       r.source_document AS source_document
                ORDER BY r.processed_time DESC
                """
            )
            records = list(result)

        if not records:
            return []

        # 批量获取所有 embedding
        uuids = [record["uuid"] for record in records]
        emb_map = self._vector_store.get_batch("relation_vectors", uuids)

        relations = []
        for record in records:
            relation = _neo4j_record_to_relation(record)
            emb_list = emb_map.get(relation.absolute_id)
            emb_array = np.array(emb_list, dtype=np.float32) if emb_list else None
            relations.append((relation, emb_array))
        return relations

    def search_relations_by_similarity(self, query_text: str,
                                       threshold: float = 0.3,
                                       max_results: int = 10) -> List[Relation]:
        """根据相似度搜索关系。"""
        if self.embedding_client and self.embedding_client.is_available():
            # KNN 路径：不需要加载全量数据
            return self._search_relations_with_embedding(
                query_text, [], threshold, max_results
            )
        else:
            # BM25 替代文本全量扫描
            return self.search_relations_by_bm25(query_text, limit=max_results)

    def _search_relations_with_embedding(self, query_text: str,
                                          relations_with_embeddings: List[tuple],
                                          threshold: float,
                                          max_results: int) -> List[Relation]:
        """使用 sqlite-vec KNN 进行关系相似度搜索。"""
        # 1. Encode + 归一化 query
        query_embedding = self.embedding_client.encode(query_text)
        if query_embedding is None:
            return []

        query_emb = np.array(
            query_embedding[0] if isinstance(query_embedding, (list, np.ndarray)) else query_embedding,
            dtype=np.float32
        )
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        # 2. KNN top-k（多取几倍候选，因为同一 family_id 可能有多个版本）
        knn_limit = max_results * 5
        knn_results = self._vector_store.search(
            "relation_vectors", query_emb.tolist(), limit=knn_limit
        )

        if not knn_results:
            return []

        # 3. L2 距离转余弦相似度
        uuid_dist = {uuid: dist for uuid, dist in knn_results}

        # 4. 批量获取关系（一次 Neo4j 查询）
        uuids = [uuid for uuid, _ in knn_results]
        relations = self.get_relations_by_absolute_ids(uuids)

        # 5. 过滤 threshold + 去重（同 family_id 取最新）
        seen = set()
        results = []
        for relation in relations:
            if relation is None:
                continue
            l2_dist_sq = uuid_dist.get(relation.absolute_id)
            if l2_dist_sq is None:
                continue
            cos_sim = 1.0 - l2_dist_sq / 2.0
            if cos_sim >= threshold and relation.family_id not in seen:
                results.append(relation)
                seen.add(relation.family_id)
                if len(results) >= max_results:
                    break
        return results

    # ------------------------------------------------------------------
    # 文档操作
    # ------------------------------------------------------------------

    def get_doc_hash_by_cache_id(self, cache_id: str) -> Optional[str]:
        """根据 cache_id 获取 doc_hash。"""
        return self._id_to_doc_hash.get(cache_id)

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
        except Exception as e:
            logger.debug("Failed to read doc content for '%s': %s", filename, e)
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
            except Exception as e:
                logger.debug("Skipping meta file during list_docs: %s", e)
                continue
        return results

    # ------------------------------------------------------------------
    # 图遍历操作（Neo4j 原生优势）
    # ------------------------------------------------------------------

    def merge_entity_families(self, target_family_id: str, source_family_ids: List[str],
                              skip_name_check: bool = False) -> Dict[str, Any]:
        """合并多个 family_id 到目标 family_id。

        Args:
            target_family_id: 目标实体的 family_id。
            source_family_ids: 要合并的源实体 family_id 列表。
            skip_name_check: 如果为 True，跳过名称安全检查（用于已确认的合并）。
        """
        target_family_id = self.resolve_family_id(target_family_id)
        if not target_family_id or not source_family_ids:
            return {"entities_updated": 0, "relations_updated": 0}

        # 名称安全检查：拒绝名称完全不相关的合并
        if not skip_name_check:
            target_entity = self.get_entity_by_family_id(target_family_id)
            target_name = target_entity.name if target_entity else ""
            for source_id in source_family_ids:
                resolved_source = self.resolve_family_id(source_id)
                if not resolved_source:
                    continue
                source_entity = self.get_entity_by_family_id(resolved_source)
                if not source_entity:
                    continue
                source_name = source_entity.name
                if target_name and source_name:
                    shared = len(set(source_name) & set(target_name))
                    total = len(set(source_name) | set(target_name))
                    overlap = shared / total if total > 0 else 0
                    if overlap < 0.2:
                        logging.getLogger(__name__).warning(
                            f"拒绝合并：名称差异过大 — "
                            f"target={target_name}({target_family_id}) "
                            f"source={source_name}({resolved_source}) "
                            f"overlap={overlap:.2f}"
                        )
                        # 从列表中移除此 source
                        source_family_ids = [
                            s for s in source_family_ids
                            if self.resolve_family_id(s) != resolved_source
                        ]

        if not source_family_ids:
            return {"entities_updated": 0, "relations_updated": 0, "rejected": True}

        with self._write_lock:
            with self._session() as session:
                entities_updated = 0
                canonical_source_ids: List[str] = []

                for source_id in source_family_ids:
                    source_id = self._resolve_family_id_in_session(session, source_id)
                    if not source_id or source_id == target_family_id or source_id in canonical_source_ids:
                        continue
                    canonical_source_ids.append(source_id)

                    # 更新所有使用 source_id 的实体节点
                    result = session.run(
                        """
                        MATCH (e:Entity {family_id: $sid})
                        SET e.family_id = $tid
                        RETURN COUNT(e) AS cnt
                        """,
                        sid=source_id,
                        tid=target_family_id,
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
                        tid=target_family_id,
                        now=now_iso,
                    )

                return {
                    "entities_updated": entities_updated,
                    "relations_updated": 0,
                    "target_family_id": target_family_id,
                    "merged_source_ids": canonical_source_ids,
                }

    def find_shortest_paths(self, source_family_id: str, target_family_id: str,
                             max_depth: int = 6, max_paths: int = 10) -> Dict[str, Any]:
        """使用 Neo4j Cypher 查找最短路径。"""
        result_empty = {
            "source_entity": None,
            "target_entity": None,
            "path_length": -1,
            "total_shortest_paths": 0,
            "paths": [],
        }

        source_entity = self.get_entity_by_family_id(source_family_id)
        target_entity = self.get_entity_by_family_id(target_family_id)

        if not source_entity or not target_entity:
            result_empty["source_entity"] = source_entity
            result_empty["target_entity"] = target_entity
            return result_empty

        if source_family_id == target_family_id:
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
        with self._session() as session:
            result = session.run(
                """
                MATCH (source:Entity {family_id: $sid}),
                      (target:Entity {family_id: $tid})
                MATCH path = allShortestPaths((source)-[:RELATES_TO*1..""" + str(max_depth) + """]-(target))
                UNWIND [n IN nodes(path) | n.uuid] AS abs_ids
                UNWIND [r IN relationships(path) | r.relation_uuid] AS rel_uuids
                WITH path, COLLECT(DISTINCT abs_ids) AS abs_id_set, COLLECT(DISTINCT rel_uuids) AS rel_id_set
                RETURN abs_id_set, rel_id_set
                LIMIT $max_paths
                """,
                sid=source_family_id,
                tid=target_family_id,
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
                    f"""
                    MATCH (e:Entity)
                    WHERE e.uuid IN $uuids
                    RETURN {_ENTITY_RETURN_FIELDS}
                    """,
                    uuids=list(needed_abs_ids),
                )
                for r in res:
                    entity = _neo4j_record_to_entity(r)
                    abs_entity_map[entity.absolute_id] = entity
                    abs_to_eid[entity.absolute_id] = entity.family_id

            rel_map: Dict[str, Relation] = {}
            if needed_rel_ids:
                res = session.run(
                    """
                    MATCH (r:Relation)
                    WHERE r.uuid IN $uuids
                    RETURN r.uuid AS uuid, r.family_id AS family_id,
                           r.entity1_absolute_id AS entity1_absolute_id,
                           r.entity2_absolute_id AS entity2_absolute_id,
                           r.content AS content, r.event_time AS event_time,
                           r.processed_time AS processed_time, r.episode_id AS episode_id,
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
                MATCH (source:Entity {family_id: $sid}),
                      (target:Entity {family_id: $tid})
                MATCH path = allShortestPaths((source)-[:RELATES_TO*1..""" + str(max_depth) + """]-(target))
                UNWIND nodes(path) AS n
                UNWIND relationships(path) AS r
                WITH path, COLLECT(DISTINCT {uuid: n.uuid, family_id: n.family_id}) AS nodes,
                     COLLECT(DISTINCT {uuid: r.relation_uuid}) AS rels
                RETURN nodes, rels
                LIMIT $max_paths
                """,
                sid=source_family_id,
                tid=target_family_id,
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

    def find_shortest_path_cypher(self, source_family_id: str, target_family_id: str,
                                   max_depth: int = 6) -> List[List[str]]:
        """使用 Cypher shortestPath 查找单条最短路径（性能更优）。

        Returns:
            路径列表，每条路径为实体名称列表。
        """
        with self._session() as session:
            result = session.run(
                """
                MATCH (a:Entity {family_id: $sid}), (b:Entity {family_id: $tid})
                MATCH path = shortestPath((a)-[:RELATES_TO*1..""" + str(max_depth) + """]-(b))
                RETURN [n IN nodes(path) | n.name] AS names
                """,
                sid=source_family_id,
                tid=target_family_id,
            )
            records = list(result)
            if not records:
                return []
            return [record["names"] for record in records]

    def get_entity_neighbors(self, entity_uuid: str, depth: int = 1) -> Dict:
        """获取实体的邻居图，返回完整的 nodes + edges 结构。"""
        with self._session() as session:
            # 先获取中心节点
            center = session.run(
                "MATCH (e:Entity {uuid: $uuid}) RETURN e.uuid AS uuid, e.name AS name, e.family_id AS family_id",
                uuid=entity_uuid,
            )
            center_records = list(center)
            center_node = None
            if center_records:
                r = center_records[0]
                center_node = {"uuid": r["uuid"], "name": r["name"], "family_id": r["family_id"]}

            # 获取所有邻居节点和边
            result = session.run(
                f"""
                MATCH (e:Entity {{uuid: $uuid}})-[r:RELATES_TO*1..{depth}]-(neighbor:Entity)
                UNWIND r AS rel
                WITH DISTINCT neighbor, rel LIMIT 500
                RETURN neighbor.uuid AS uuid, neighbor.name AS name, neighbor.family_id AS family_id,
                       startNode(rel).uuid AS source_uuid, endNode(rel).uuid AS target_uuid,
                       rel.relation_uuid AS relation_uuid, rel.fact AS fact
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
                        "family_id": record["family_id"],
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
        with self._session() as session:
            result = session.run(
                "MATCH (ep:Episode) RETURN ep.uuid AS uuid, ep.content AS content, "
                "ep.source_document AS source_document, ep.event_time AS event_time, "
                "ep.episode_id AS episode_id, "
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
                    "episode_id": r["episode_id"] or "",
                    "created_at": _fmt_dt(r["created_at"]),
                }
                episodes.append(ep)
            return episodes

    def count_episodes(self) -> int:
        """统计 Episode 节点总数。"""
        with self._session() as session:
            result = session.run("MATCH (ep:Episode) RETURN COUNT(ep) AS cnt")
            record = result.single()
            return record["cnt"] if record else 0

    def count_communities(self) -> int:
        """统计社区数量（DISTINCT community_id）。"""
        with self._session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                "RETURN count(DISTINCT e.community_id) AS cnt"
            )
            record = result.single()
            return record["cnt"] if record else 0

    def get_episode(self, uuid: str) -> Optional[Dict]:
        """获取单个 Episode 详情。"""
        with self._session() as session:
            result = session.run(
                "MATCH (ep:Episode {uuid: $uuid}) RETURN ep.uuid AS uuid, ep.content AS content, "
                "ep.source_document AS source_document, ep.event_time AS event_time, "
                "ep.episode_id AS episode_id, ep.created_at AS created_at",
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
                "episode_id": record["episode_id"] or "",
                "created_at": _fmt_dt(record["created_at"]),
            }

    def search_episodes(self, query: str, limit: int = 20) -> List[Dict]:
        """通过 content LIKE 搜索 Episode。"""
        with self._session() as session:
            result = session.run(
                "MATCH (ep:Episode) WHERE ep.content CONTAINS $search_query "
                "RETURN ep.uuid AS uuid, ep.content AS content, "
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
                    "source_document": r["source_document"] or "",
                    "event_time": _fmt_dt(r["event_time"]),
                    "episode_id": r["episode_id"] or "",
                    "created_at": _fmt_dt(r["created_at"]),
                })
            return episodes

    def get_episode_entities(self, uuid: str) -> List[Dict]:
        """通过 episode_id 关联查出 Episode 下的实体。"""
        with self._session() as session:
            episode = session.run(
                "MATCH (ep:Episode {uuid: $uuid}) RETURN ep.episode_id AS mcid",
                uuid=uuid,
            ).single()
            if not episode or not episode["mcid"]:
                return []
            mcid = episode["mcid"]
            result = session.run(
                "MATCH (e:Entity {episode_id: $mcid}) "
                "RETURN e.uuid AS uuid, e.family_id AS family_id, e.name AS name, "
                "e.content AS content, e.event_time AS event_time "
                "ORDER BY e.processed_time DESC LIMIT 200",
                mcid=mcid,
            )
            entities = []
            for r in result:
                entities.append({
                    "uuid": r["uuid"],
                    "family_id": r["family_id"],
                    "name": r["name"],
                    "content": r["content"] or "",
                    "event_time": r["event_time"].isoformat() if r["event_time"] else None,
                })
            return entities

    # ------------------------------------------------------------------
    # 社区检测
    # ------------------------------------------------------------------

    def detect_communities(self, algorithm: str = 'louvain', resolution: float = 1.0) -> Dict:
        """从 Neo4j 加载图 → networkx Louvain → 写回 community_id。"""
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        t0 = time.time()

        # 加载所有 Entity + RELATES_TO 边
        with self._session() as session:
            result = session.run(
                "MATCH (e:Entity) RETURN e.uuid AS uuid, e.family_id AS fid, e.name AS name"
            )
            entity_map = {}  # uuid -> fid
            for r in result:
                entity_map[r["uuid"]] = r["fid"]

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
            with self._session() as session:
                session.run(
                    "UNWIND $items AS item "
                    "MATCH (e:Entity {uuid: item.uuid}) "
                    "SET e.community_id = item.cid",
                    items=batch,
                )

    def get_communities(self, limit: int = 50, min_size: int = 3, offset: int = 0) -> Tuple[List[Dict], int]:
        """按社区分组，返回 members 列表 + 总数。"""
        with self._session() as session:
            # 先获取数据
            result = session.run(
                "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                "WITH e.community_id AS cid, collect(e) AS members "
                "WHERE size(members) >= $min_size "
                "RETURN cid, size(members) AS size, "
                "[m IN members | {uuid: m.uuid, family_id: m.family_id, name: m.name}] AS members "
                "ORDER BY size DESC SKIP $offset LIMIT $limit",
                min_size=min_size, offset=offset, limit=limit,
            )
            communities = []
            for r in result:
                communities.append({
                    "community_id": r["cid"],
                    "size": r["size"],
                    "members": r["members"],
                })

            # 再 count 总数（用独立 session 避免流冲突）
            with self._session() as count_session:
                count_result = count_session.run(
                    "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                    "WITH e.community_id AS cid, collect(e) AS members "
                    "WHERE size(members) >= $min_size "
                    "RETURN count(cid) AS total",
                    min_size=min_size,
                )
                count_record = count_result.single()
                total = count_record["total"] if count_record else len(communities)

            return communities, total

    def get_community(self, cid: int) -> Optional[Dict]:
        """单社区详情 + 社区内关系（合并单条 Cypher，LIMIT 500）。"""
        cache_key = f"community:{cid}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        with self._session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.community_id = $cid "
                "OPTIONAL MATCH (e)-[r:RELATES_TO]-(other:Entity) "
                "WHERE other.community_id = $cid "
                "WITH e, collect(DISTINCT {uuid: other.uuid, name: other.name, fact: r.fact, "
                "ruuid: r.relation_uuid}) AS rels "
                "RETURN e.uuid AS uuid, e.family_id AS family_id, e.name AS name, "
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
                    "family_id": r["family_id"],
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

        with self._session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.community_id = $cid "
                "WITH e LIMIT 300 "
                "OPTIONAL MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                "WHERE (a.uuid = e.uuid OR b.uuid = e.uuid) "
                "WITH e, collect(DISTINCT {src: a.uuid, tgt: b.uuid, fact: r.fact}) AS rels "
                "RETURN e.uuid AS uuid, e.family_id AS family_id, e.name AS name, rels",
                cid=cid,
            )
            nodes = []
            seen_edges = set()
            edges = []
            for r in result:
                nodes.append({
                    "uuid": r["uuid"],
                    "family_id": r["family_id"],
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
        with self._session() as session:
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
        with self._session() as session:
            result = session.run(f"""
                MATCH (e:Entity)
                WHERE (e.valid_at IS NULL OR e.valid_at <= $time)
                  AND (e.invalid_at IS NULL OR e.invalid_at > $time)
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.event_time DESC
                LIMIT $limit
            """, time=time_point.isoformat(), limit=limit or 10000)
            entities = [_neo4j_record_to_entity(r) for r in result]

            result = session.run("""
                MATCH (r:Relation)
                WHERE (r.valid_at IS NULL OR r.valid_at <= $time)
                  AND (r.invalid_at IS NULL OR r.invalid_at > $time)
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.episode_id AS episode_id,
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
        with self._session() as session:
            result = session.run(f"""
                MATCH (e:Entity)
                WHERE e.event_time >= $since AND e.event_time <= $until
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.event_time DESC
            """, since=since.isoformat(), until=until.isoformat())
            entities = [_neo4j_record_to_entity(r) for r in result]

            result = session.run("""
                MATCH (r:Relation)
                WHERE r.event_time >= $since AND r.event_time <= $until
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.episode_id AS episode_id,
                       r.source_document AS source_document, r.valid_at AS valid_at,
                       r.invalid_at AS invalid_at
                ORDER BY r.event_time DESC
            """, since=since.isoformat(), until=until.isoformat())
            relations = [_neo4j_record_to_relation(r) for r in result]

        return {"entities": entities, "relations": relations}

    def invalidate_relation(self, family_id: str, reason: str = "") -> int:
        """标记关系为失效"""
        now = datetime.now(timezone.utc).isoformat()
        with self._session() as session:
            result = session.run("""
                MATCH (r:Relation {family_id: $family_id})
                WHERE r.invalid_at IS NULL
                SET r.invalid_at = $now
                RETURN count(r) AS cnt
            """, family_id=family_id, now=now)
            record = result.single()
            return record["cnt"] if record else 0

    def get_invalidated_relations(self, limit: int = 100) -> List[Relation]:
        """列出已失效的关系"""
        with self._session() as session:
            result = session.run("""
                MATCH (r:Relation)
                WHERE r.invalid_at IS NOT NULL
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.episode_id AS episode_id,
                       r.source_document AS source_document, r.valid_at AS valid_at,
                       r.invalid_at AS invalid_at
                ORDER BY r.invalid_at DESC
                LIMIT $limit
            """, limit=limit)
            return [_neo4j_record_to_relation(r) for r in result]

    # ------------------------------------------------------------------
    # Phase A/C/D/E: 新增方法
    # ------------------------------------------------------------------

    def update_entity_summary(self, family_id: str, summary: str):
        """更新实体摘要。"""
        resolved = self.resolve_family_id(family_id)
        if not resolved:
            return
        with self._session() as session:
            session.run("""
                MATCH (e:Entity {family_id: $fid})
                WHERE e.invalid_at IS NULL
                SET e.summary = $summary
            """, fid=resolved, summary=summary)
        self._cache.invalidate("entity:")

    def update_entity_attributes(self, family_id: str, attributes: str):
        """更新实体结构化属性。"""
        resolved = self.resolve_family_id(family_id)
        if not resolved:
            return
        with self._session() as session:
            session.run("""
                MATCH (e:Entity {family_id: $fid})
                WHERE e.invalid_at IS NULL
                SET e.attributes = $attributes
            """, fid=resolved, attributes=attributes)
        self._cache.invalidate("entity:")

    def get_relations_by_family_ids(self, family_ids: List[str], limit: int = 100) -> List[Relation]:
        """获取指定实体 ID 列表相关的所有关系。

        使用单次 Cypher 查询完成 family_id→absolute_id 解析 + 关系检索，
        避免逐个 family_id 调用 resolve_family_id + get_entity_by_family_id 的 N+1 问题。
        """
        if not family_ids:
            return []
        with self._session() as session:
            # 单次查询：解析 family_id → 最新 absolute_id，再查找关联关系
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.family_id IN $family_ids AND e.invalid_at IS NULL
                WITH collect(DISTINCT e.uuid) AS abs_ids
                UNWIND abs_ids AS aid
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id = aid OR r.entity2_absolute_id = aid)
                  AND r.invalid_at IS NULL
                RETURN DISTINCT r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time, r.episode_id AS episode_id,
                       r.source_document AS source_document, r.valid_at AS valid_at,
                       r.invalid_at AS invalid_at, r.summary AS summary,
                       r.attributes AS attributes, r.confidence AS confidence,
                       r.provenance AS provenance
                LIMIT $limit
            """, family_ids=family_ids, limit=limit)
            return [_neo4j_record_to_relation(r) for r in result]

    def get_entity_degree(self, family_id: str) -> int:
        """获取实体的度（连接数）— 轻量 COUNT 查询，不物化 Relation 对象。"""
        with self._session() as session:
            result = session.run("""
                MATCH (e:Entity) WHERE e.family_id = $fid AND e.invalid_at IS NULL
                WITH collect(DISTINCT e.uuid) AS abs_ids
                UNWIND abs_ids AS aid
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id = aid OR r.entity2_absolute_id = aid)
                  AND r.invalid_at IS NULL
                RETURN count(DISTINCT r) AS cnt
            """, fid=family_id)
            record = result.single()
            return record["cnt"] if record else 0

    def batch_get_entity_degrees(self, family_ids: List[str]) -> Dict[str, int]:
        """批量获取实体度数 — 单次 Cypher 查询替代 N 次 get_entity_degree。"""
        if not family_ids:
            return {}
        with self._session() as session:
            result = session.run("""
                UNWIND $fids AS fid
                MATCH (e:Entity) WHERE e.family_id = fid AND e.invalid_at IS NULL
                WITH fid, collect(DISTINCT e.uuid) AS abs_ids
                UNWIND abs_ids AS aid
                OPTIONAL MATCH (r:Relation)
                WHERE (r.entity1_absolute_id = aid OR r.entity2_absolute_id = aid)
                  AND r.invalid_at IS NULL
                RETURN fid, count(DISTINCT r) AS cnt
            """, fids=family_ids)
            degree_map = {}
            for record in result:
                degree_map[record["fid"]] = record["cnt"]
        for fid in family_ids:
            degree_map.setdefault(fid, 0)
        return degree_map

    def batch_bfs_traverse(self, seed_family_ids: List[str], max_depth: int = 2, max_nodes: int = 50) -> Tuple[List[Entity], List[Relation], Dict[str, int]]:
        """批量 BFS 遍历：从种子实体出发，单次 Cypher 查询完成多跳扩展。

        Args:
            seed_family_ids: 种子实体的 family_id 列表
            max_depth: 最大扩展深度
            max_nodes: 最多返回的节点数

        Returns:
            (entities, relations, hop_map) 其中 hop_map[family_id] = hop 距离
        """
        if not seed_family_ids:
            return [], [], {}

        with self._session() as session:
            # 第一步：解析种子 family_id → absolute_id
            seed_result = session.run("""
                MATCH (e:Entity)
                WHERE e.family_id IN $family_ids AND e.invalid_at IS NULL
                RETURN e.family_id AS family_id, e.uuid AS absolute_id
            """, family_ids=seed_family_ids)

            seed_abs_to_fid = {}
            seed_fids = []
            for rec in seed_result:
                fid = rec["family_id"]
                aid = rec["absolute_id"]
                seed_abs_to_fid[aid] = fid
                seed_fids.append(fid)

            if not seed_fids:
                return [], [], {}

            # 第二步：Cypher BFS — 从种子 absolute_id 出发，沿关系边扩展
            # 通过 Entity 节点的 RELATES_TO 边进行遍历（已有边类型）
            result = session.run("""
                MATCH (seed:Entity)
                WHERE seed.family_id IN $seed_fids AND seed.invalid_at IS NULL
                WITH collect(seed) AS seeds
                UNWIND seeds AS s
                MATCH path = (s)-[:RELATES_TO*1..%d]-(neighbor:Entity)
                WHERE neighbor.invalid_at IS NULL
                WITH DISTINCT neighbor, length(path) AS dist
                ORDER BY dist ASC
                LIMIT $max_nodes
                RETURN neighbor.uuid AS absolute_id,
                       neighbor.family_id AS family_id,
                       neighbor.name AS name,
                       neighbor.content AS content,
                       neighbor.event_time AS event_time,
                       neighbor.processed_time AS processed_time,
                       neighbor.episode_id AS episode_id,
                       neighbor.source_document AS source_document,
                       neighbor.summary AS summary,
                       neighbor.confidence AS confidence,
                       neighbor.attributes AS attributes,
                       neighbor.community_id AS community_id,
                       dist AS dist
            """ % max_depth, seed_fids=seed_fids, max_nodes=max_nodes)

            entities = []
            hop_map = {}
            for rec in result:
                ent = _neo4j_record_to_entity(rec)
                if ent and ent.family_id not in hop_map:
                    entities.append(ent)
                    hop_map[ent.family_id] = rec["dist"]

            # 种子实体也加入（如果 BFS 没有返回）
            for fid in seed_fids:
                if fid not in hop_map:
                    entity = self.get_entity_by_family_id(fid)
                    if entity and entity not in entities:
                        entities.insert(0, entity)
                        hop_map[fid] = 0

            # 第三步：批量获取这些实体之间的关系
            discovered_fids = list(hop_map.keys())
            relations = self.get_relations_by_family_ids(discovered_fids, limit=max_nodes * 3) if discovered_fids else []

            return entities, relations, hop_map

    def save_episode_mentions(self, episode_id: str, entity_absolute_ids: List[str], context: str = ""):
        """记录 Episode 提及的实体（单次 UNWIND 批量写入）。"""
        if not entity_absolute_ids:
            return
        with self._episode_write_lock:
            with self._session() as session:
                session.run("""
                    MERGE (ep:Episode {uuid: $ep_id})
                    WITH ep
                    UNWIND $items AS item
                    MATCH (e:Entity {uuid: item.abs_id})
                    MERGE (ep)-[m:MENTIONS {context: item.ctx}]->(e)
                """, ep_id=episode_id,
                     items=[{"abs_id": aid, "ctx": context} for aid in entity_absolute_ids])

    def get_entity_provenance(self, family_id: str) -> List[dict]:
        """获取提及该实体的所有 Episode。"""
        entity = self.get_entity_by_family_id(family_id)
        if not entity:
            return []
        with self._session() as session:
            result = session.run("""
                MATCH (ep:Episode)-[m:MENTIONS]->(e:Entity {uuid: $abs_id})
                RETURN ep.uuid AS episode_id, m.context AS context
            """, abs_id=entity.absolute_id)
            return [{"episode_id": r["episode_id"], "context": r.get("context", "")} for r in result]

    def get_episode_entities(self, episode_id: str) -> List[dict]:
        """获取 Episode 关联的所有实体。"""
        with self._session() as session:
            result = session.run("""
                MATCH (ep:Episode {uuid: $ep_id})-[m:MENTIONS]->(e:Entity)
                RETURN e.uuid AS entity_absolute_id, e.family_id AS family_id,
                       e.name AS name, m.context AS context
            """, ep_id=episode_id)
            return [
                {
                    "entity_absolute_id": r["entity_absolute_id"],
                    "family_id": r.get("family_id", ""),
                    "name": r.get("name", ""),
                    "context": r.get("context", ""),
                }
                for r in result
            ]

    def delete_episode_mentions(self, episode_id: str):
        """删除 Episode 的所有 MENTIONS 边。"""
        with self._session() as session:
            session.run("""
                MATCH (ep:Episode {uuid: $ep_id})-[m:MENTIONS]->()
                DELETE m
            """, ep_id=episode_id)

    def save_dream_log(self, report):
        """保存梦境日志。"""
        import json as _json
        with self._session() as session:
            session.run("""
                MERGE (d:DreamLog {cycle_id: $cycle_id})
                SET d.graph_id = $graph_id,
                    d.start_time = datetime($start_time),
                    d.end_time = datetime($end_time),
                    d.status = $status,
                    d.narrative = $narrative,
                    d.insights = $insights,
                    d.connections = $connections,
                    d.consolidations = $consolidations,
                    d.strategy = $strategy,
                    d.entities_examined = $entities_examined,
                    d.relations_created = $relations_created,
                    d.episode_ids = $episode_ids
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
                strategy=getattr(report, 'strategy', ''),
                entities_examined=getattr(report, 'entities_examined', 0),
                relations_created=getattr(report, 'relations_created', 0),
                episode_ids=_json.dumps(getattr(report, 'episode_ids', []), ensure_ascii=False),
            )

    def list_dream_logs(self, graph_id: str = "default", limit: int = 20) -> List[dict]:
        """列出梦境日志。"""
        import json as _json
        with self._session() as session:
            result = session.run("""
                MATCH (d:DreamLog)
                WHERE d.graph_id = $graph_id
                RETURN d.cycle_id AS cycle_id, d.graph_id AS graph_id,
                       d.start_time AS start_time, d.end_time AS end_time,
                       d.status AS status, d.narrative AS narrative,
                       d.insights AS insights, d.connections AS connections,
                       d.consolidations AS consolidations,
                       d.strategy AS strategy, d.entities_examined AS entities_examined,
                       d.relations_created AS relations_created,
                       d.episode_ids AS episode_ids
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
                    "strategy": r.get("strategy", ""),
                    "entities_examined": r.get("entities_examined", 0),
                    "relations_created": r.get("relations_created", 0),
                    "episode_ids": _json.loads(r["episode_ids"]) if r.get("episode_ids") else [],
                })
            return logs

    def get_dream_log(self, cycle_id: str) -> Optional[dict]:
        """获取单条梦境日志。"""
        import json as _json
        with self._session() as session:
            result = session.run("""
                MATCH (d:DreamLog {cycle_id: $cycle_id})
                RETURN d.cycle_id AS cycle_id, d.graph_id AS graph_id,
                       d.start_time AS start_time, d.end_time AS end_time,
                       d.status AS status, d.narrative AS narrative,
                       d.insights AS insights, d.connections AS connections,
                       d.consolidations AS consolidations,
                       d.strategy AS strategy, d.entities_examined AS entities_examined,
                       d.relations_created AS relations_created,
                       d.episode_ids AS episode_ids
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
                "strategy": r.get("strategy", ""),
                "entities_examined": r.get("entities_examined", 0),
                "relations_created": r.get("relations_created", 0),
                "episode_ids": _json.loads(r["episode_ids"]) if r.get("episode_ids") else [],
            }

    # =========================================================
    # DeepDream Agent API — 存储层
    # =========================================================

    def get_dream_seeds(self, strategy: str = "random", count: int = 10,
                        exclude_ids: Optional[List[str]] = None,
                        community_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """按策略选取梦境种子实体。

        strategy: random | orphan | hub | time_gap | cross_community | low_confidence
        """
        import uuid as _uuid

        exclude_uuids = set()
        if exclude_ids:
            for eid in exclude_ids:
                resolved = self.resolve_family_id(eid)
                if resolved:
                    entity = self.get_entity_by_family_id(resolved)
                    if entity:
                        exclude_uuids.add(entity.absolute_id)

        strategies = {
            "random": self._dream_seeds_random,
            "orphan": self._dream_seeds_orphan,
            "hub": self._dream_seeds_hub,
            "time_gap": self._dream_seeds_time_gap,
            "low_confidence": self._dream_seeds_low_confidence,
            "cross_community": self._dream_seeds_cross_community,
        }

        handler = strategies.get(strategy)
        if not handler:
            raise ValueError(f"未知的种子策略: {strategy}，可选: {', '.join(strategies.keys())}")

        seeds = handler(count, exclude_uuids, community_id)

        # 添加 reason 字段
        reason_map = {
            "random": "随机选取",
            "orphan": "孤立实体：无任何关系连接",
            "hub": "高连接度实体",
            "time_gap": "长时间未更新的实体",
            "low_confidence": "低置信度实体",
            "cross_community": "跨社区桥接候选",
        }
        for s in seeds:
            s["reason"] = reason_map.get(strategy, "")

        return seeds

    def _dream_seeds_random(self, count, exclude_uuids, community_id):
        with self._session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE NOT e.uuid IN $exclude_uuids
                  AND ($cid IS NULL OR e.community_id = $cid)
                WITH e ORDER BY rand() LIMIT $count
                RETURN e.uuid AS uuid, e.family_id AS family_id,
                       e.name AS name, e.content AS content,
                       e.confidence AS confidence, e.event_time AS event_time,
                       e.community_id AS community_id,
                       size([(e)--() | 1]) AS degree
            """, exclude_uuids=list(exclude_uuids), cid=community_id, count=count)
            return [dict(r) for r in result]

    def _dream_seeds_orphan(self, count, exclude_uuids, community_id):
        with self._session() as session:
            result = session.run("""
                MATCH (e:Entity) WHERE NOT (e)--()
                  AND NOT e.uuid IN $exclude_uuids
                  AND ($cid IS NULL OR e.community_id = $cid)
                RETURN e.uuid AS uuid, e.family_id AS family_id,
                       e.name AS name, e.content AS content,
                       e.confidence AS confidence, e.event_time AS event_time,
                       e.community_id AS community_id, 0 AS degree
                LIMIT $count
            """, exclude_uuids=list(exclude_uuids), cid=community_id, count=count)
            return [dict(r) for r in result]

    def _dream_seeds_hub(self, count, exclude_uuids, community_id):
        with self._session() as session:
            result = session.run("""
                MATCH (e:Entity)-[r]-(rel)
                WITH e, count(DISTINCT rel) AS degree
                WHERE NOT e.uuid IN $exclude_uuids
                  AND ($cid IS NULL OR e.community_id = $cid)
                ORDER BY degree DESC LIMIT $count
                RETURN e.uuid AS uuid, e.family_id AS family_id,
                       e.name AS name, e.content AS content,
                       e.confidence AS confidence, e.event_time AS event_time,
                       e.community_id AS community_id, degree
            """, exclude_uuids=list(exclude_uuids), cid=community_id, count=count)
            return [dict(r) for r in result]

    def _dream_seeds_time_gap(self, count, exclude_uuids, community_id):
        with self._session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.processed_time IS NOT NULL
                  AND NOT e.uuid IN $exclude_uuids
                  AND duration.between(e.processed_time, datetime()).days > 30
                  AND ($cid IS NULL OR e.community_id = $cid)
                ORDER BY e.processed_time ASC LIMIT $count
                RETURN e.uuid AS uuid, e.family_id AS family_id,
                       e.name AS name, e.content AS content,
                       e.confidence AS confidence, e.event_time AS event_time,
                       e.community_id AS community_id,
                       size([(e)--() | 1]) AS degree
            """, exclude_uuids=list(exclude_uuids), cid=community_id, count=count)
            return [dict(r) for r in result]

    def _dream_seeds_low_confidence(self, count, exclude_uuids, community_id):
        with self._session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.confidence IS NOT NULL AND e.confidence < 0.5
                  AND NOT e.uuid IN $exclude_uuids
                  AND ($cid IS NULL OR e.community_id = $cid)
                ORDER BY e.confidence ASC LIMIT $count
                RETURN e.uuid AS uuid, e.family_id AS family_id,
                       e.name AS name, e.content AS content,
                       e.confidence AS confidence, e.event_time AS event_time,
                       e.community_id AS community_id,
                       size([(e)--() | 1]) AS degree
            """, exclude_uuids=list(exclude_uuids), cid=community_id, count=count)
            return [dict(r) for r in result]

    def _dream_seeds_cross_community(self, count, exclude_uuids, community_id):
        """从不同社区各取 1 个随机实体，组合返回跨社区实体对。

        注意：cross_community 策略忽略 community_id 参数，因为它需要跨社区采样。
        """
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
                # 从每个社区取 1 个非排除的随机成员
                import random as _random
                c1_valid = [m for m in c1_members if m["uuid"] not in exclude_uuids]
                c2_valid = [m for m in c2_members if m["uuid"] not in exclude_uuids]
                if c1_valid and c2_valid:
                    e1 = _random.choice(c1_valid)
                    e2 = _random.choice(c2_valid)
                    pairs.extend([e1, e2])
            if len(pairs) >= count * 2:
                break

        return pairs[:count * 2]

    def save_dream_relation(self, entity1_id: str, entity2_id: str,
                            content: str, confidence: float, reasoning: str,
                            dream_cycle_id: Optional[str] = None,
                            episode_id: Optional[str] = None) -> Dict[str, Any]:
        """创建或合并梦境发现的关系。

        Returns: {"family_id": "...", "entity1_family_id": "...", "entity2_family_id": "...", "action": "created"|"merged"}
        Raises: ValueError 如果实体不存在
        """
        import uuid as _uuid
        import json as _json
        from processor.models import Relation

        # 解析实体
        resolved1 = self.resolve_family_id(entity1_id)
        resolved2 = self.resolve_family_id(entity2_id)
        if not resolved1:
            raise ValueError(f"实体不存在: {entity1_id}")
        if not resolved2:
            raise ValueError(f"实体不存在: {entity2_id}")

        entity1 = self.get_entity_by_family_id(resolved1)
        entity2 = self.get_entity_by_family_id(resolved2)
        if not entity1:
            raise ValueError(f"实体不存在: {entity1_id}")
        if not entity2:
            raise ValueError(f"实体不存在: {entity2_id}")

        # 检查是否已存在关系 — 合并而非报错
        existing = self.get_relations_by_entities(resolved1, resolved2)
        if existing:
            latest = existing[0]
            # 合并：取较高 confidence，追加 reasoning
            new_confidence = max(latest.confidence or 0, confidence)
            # 构建新的 provenance entry
            new_prov_entry = {
                "source": "dream",
                "dream_cycle_id": dream_cycle_id,
                "confidence": confidence,
                "reasoning": reasoning,
            }
            try:
                old_prov = _json.loads(latest.provenance) if latest.provenance else []
            except Exception:
                old_prov = []
            old_prov.append(new_prov_entry)

            # 创建新版本（保留同一 family_id）
            now = datetime.now()
            record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}"
            source_doc = f"dream:{dream_cycle_id}" if dream_cycle_id else "dream"
            merged_content = f"{latest.content}\n[Dream update] {content}" if content != latest.content else latest.content

            relation = Relation(
                absolute_id=record_id,
                family_id=latest.family_id,
                entity1_absolute_id=latest.entity1_absolute_id,
                entity2_absolute_id=latest.entity2_absolute_id,
                content=merged_content,
                event_time=now,
                processed_time=now,
                episode_id=episode_id or latest.episode_id or "",
                source_document=source_doc,
                confidence=new_confidence,
                provenance=_json.dumps(old_prov, ensure_ascii=False),
            )
            self.save_relation(relation)
            return {
                "family_id": latest.family_id,
                "entity1_family_id": resolved1,
                "entity2_family_id": resolved2,
                "entity1_name": entity1.name,
                "entity2_name": entity2.name,
                "action": "merged",
            }

        # 排序确保 (A,B) 和 (B,A) 视为同一关系
        if entity1.name <= entity2.name:
            e1_abs, e2_abs = entity1.absolute_id, entity2.absolute_id
        else:
            e1_abs, e2_abs = entity2.absolute_id, entity1.absolute_id

        now = datetime.now()
        family_id = f"rel_{_uuid.uuid4().hex[:12]}"
        record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}"

        source_doc = f"dream:{dream_cycle_id}" if dream_cycle_id else "dream"
        provenance_data = {
            "source": "dream",
            "dream_cycle_id": dream_cycle_id,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        relation = Relation(
            absolute_id=record_id,
            family_id=family_id,
            entity1_absolute_id=e1_abs,
            entity2_absolute_id=e2_abs,
            content=content,
            event_time=now,
            processed_time=now,
            episode_id=episode_id or "",
            source_document=source_doc,
            confidence=confidence,
            provenance=_json.dumps([provenance_data], ensure_ascii=False),
        )

        self.save_relation(relation)

        return {
            "family_id": family_id,
            "entity1_family_id": resolved1,
            "entity2_family_id": resolved2,
            "entity1_name": entity1.name,
            "entity2_name": entity2.name,
            "action": "created",
        }

    def save_dream_episode(self, content: str,
                           entities_examined: Optional[List[str]] = None,
                           relations_created: Optional[List[Dict]] = None,
                           strategy_used: str = "",
                           dream_cycle_id: Optional[str] = None,
                           **kwargs) -> Dict[str, Any]:
        """保存梦境 episode，同时创建 DreamLog 节点以便 dream_status/dream_logs 查询。

        Returns: {"episode_id": "...", "episode_type": "dream", "cycle_id": "..."}
        """
        import uuid as _uuid
        import json as _json
        from types import SimpleNamespace
        from processor.models import Episode

        now = datetime.now()
        episode_id = f"episode_dream_{now.strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}"

        # 自动生成 cycle_id（如果未提供）
        if not dream_cycle_id:
            dream_cycle_id = f"dream_{now.strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:6]}"

        # 构建结构化内容
        _explicit_rel_count = kwargs.get("relations_created_count")
        _explicit_ent_count = kwargs.get("entities_examined_count")
        ent_count = _explicit_ent_count if _explicit_ent_count is not None else (len(entities_examined) if entities_examined else 0)
        rel_count = _explicit_rel_count if _explicit_rel_count is not None else (len(relations_created) if relations_created else 0)
        structured = {
            "narrative": content,
            "strategy": strategy_used,
            "entities_examined_count": ent_count,
            "relations_created_count": rel_count,
        }
        if relations_created:
            structured["relations_created"] = relations_created

        full_content = content
        if rel_count > 0 or ent_count > 0:
            full_content += "\n\n---\n" + _json.dumps(structured, ensure_ascii=False, indent=2)

        cache = Episode(
            absolute_id=episode_id,
            content=full_content,
            event_time=now,
            source_document=f"dream:{dream_cycle_id}" if dream_cycle_id else "dream",
            episode_type="dream",
        )

        self.save_episode(cache)

        # 记录提及的实体
        if entities_examined:
            abs_ids = []
            for eid in entities_examined:
                resolved = self.resolve_family_id(eid)
                if resolved:
                    entity = self.get_entity_by_family_id(resolved)
                    if entity:
                        abs_ids.append(entity.absolute_id)
            if abs_ids:
                self.save_episode_mentions(episode_id, abs_ids, context=f"dream:{strategy_used}")

        # 同时创建 DreamLog 节点，以便 dream_status/dream_logs 可以查询
        report = SimpleNamespace(
            cycle_id=dream_cycle_id,
            graph_id=self._graph_id,
            start_time=now,
            end_time=now,
            status="completed",
            narrative=content[:2000],
            insights=[],
            new_connections=relations_created or [],
            consolidations=[],
            strategy=strategy_used,
            entities_examined=ent_count,
            relations_created=rel_count,
            episode_ids=[episode_id],
        )
        self.save_dream_log(report)

        return {
            "episode_id": episode_id,
            "episode_type": "dream",
            "cycle_id": dream_cycle_id,
        }

    # ========== Version-Level CRUD (Phase 2) ==========

    def update_entity_by_absolute_id(self, absolute_id: str, **fields) -> Optional[Entity]:
        """根据 absolute_id 更新指定字段，返回更新后的 Entity 或 None。"""
        valid_keys = {"name", "content", "summary", "attributes", "confidence"}
        filtered = {k: v for k, v in fields.items() if k in valid_keys and v is not None}
        if not filtered:
            return self.get_entity_by_absolute_id(absolute_id)

        with self._write_lock:
            with self._session() as session:
                set_clauses = ", ".join(f"e.{k} = ${k}" for k in filtered)
                params = dict(filtered, aid=absolute_id)
                cypher = (
                    f"MATCH (e:Entity {{uuid: $aid}}) "
                    f"SET {set_clauses} "
                    f"RETURN {_ENTITY_RETURN_FIELDS}"
                )
                result = session.run(cypher, **params)
                record = result.single()
                if not record:
                    return None
                entity = _neo4j_record_to_entity(record)
            self._cache.invalidate("entity:")
            return entity

    def delete_entity_by_absolute_id(self, absolute_id: str) -> bool:
        """根据 absolute_id 删除实体及其所有关系，返回是否成功删除。"""
        with self._write_lock:
            with self._session() as session:
                result = session.run(
                    "MATCH (e:Entity {uuid: $aid}) DETACH DELETE e RETURN count(e) AS cnt",
                    aid=absolute_id,
                )
                record = result.single()
                deleted = record is not None and record["cnt"] > 0
            if deleted:
                self._vector_store.delete_batch("entity_vectors", [absolute_id])
            self._cache.invalidate("entity:")
            self._cache.invalidate("resolve:")
            self._cache.invalidate("sim_search:")
            self._cache.invalidate("graph_stats")
            return deleted

    def update_relation_by_absolute_id(self, absolute_id: str, **fields) -> Optional[Relation]:
        """根据 absolute_id 更新指定字段，返回更新后的 Relation 或 None。"""
        valid_keys = {"content", "summary", "attributes", "confidence"}
        filtered = {k: v for k, v in fields.items() if k in valid_keys and v is not None}
        if not filtered:
            return None

        with self._relation_write_lock:
            with self._session() as session:
                set_clauses = ", ".join(f"r.{k} = ${k}" for k in filtered)
                params = dict(filtered, aid=absolute_id)
                cypher = (
                    f"MATCH (r:Relation {{uuid: $aid}}) "
                    f"SET {set_clauses} "
                    f"RETURN r.uuid AS uuid, r.family_id AS family_id, "
                    f"r.entity1_absolute_id AS entity1_absolute_id, "
                    f"r.entity2_absolute_id AS entity2_absolute_id, "
                    f"r.content AS content, r.event_time AS event_time, "
                    f"r.processed_time AS processed_time, r.episode_id AS episode_id, "
                    f"r.source_document AS source_document"
                )
                result = session.run(cypher, **params)
                record = result.single()
                if not record:
                    return None
                relation = _neo4j_record_to_relation(record)
            self._cache.invalidate("relation:")
            return relation

    def delete_relation_by_absolute_id(self, absolute_id: str) -> bool:
        """根据 absolute_id 删除关系，返回是否成功删除。"""
        with self._relation_write_lock:
            with self._session() as session:
                result = session.run(
                    "MATCH (r:Relation {uuid: $aid}) DETACH DELETE r RETURN count(r) AS cnt",
                    aid=absolute_id,
                )
                record = result.single()
                deleted = record is not None and record["cnt"] > 0
            if deleted:
                self._vector_store.delete_batch("relation_vectors", [absolute_id])
            self._cache.invalidate("relation:")
            self._cache.invalidate("graph_stats")
            return deleted

    def batch_delete_relation_versions_by_absolute_ids(self, absolute_ids: List[str]) -> int:
        """批量删除指定关系版本，返回成功删除的数量。"""
        if not absolute_ids:
            return 0
        with self._relation_write_lock:
            with self._session() as session:
                result = session.run(
                    """
                    MATCH (r:Relation) WHERE r.uuid IN $aids
                    DETACH DELETE r
                    RETURN count(r) AS deleted
                    """,
                    aids=absolute_ids,
                )
                record = result.single()
                deleted = record["deleted"] if record else 0
            if deleted > 0:
                self._vector_store.delete_batch("relation_vectors", absolute_ids)
            self._cache.invalidate("relation:")
            self._cache.invalidate("graph_stats")
            return deleted

    def get_relations_referencing_absolute_id(self, absolute_id: str) -> List[Relation]:
        """获取所有引用了指定 absolute_id 的关系。"""
        with self._session() as session:
            result = session.run(
                """
                MATCH (r:Relation)
                WHERE r.entity1_absolute_id = $aid OR r.entity2_absolute_id = $aid
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time,
                       r.episode_id AS episode_id,
                       r.source_document AS source_document
                """,
                aid=absolute_id,
            )
            return [_neo4j_record_to_relation(r) for r in result]

    def batch_get_relations_referencing_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, List[Relation]]:
        """批量获取引用指定实体绝对ID的关系（消除 N+1 查询）。"""
        if not absolute_ids:
            return {}
        with self._session() as session:
            result = session.run(
                """
                MATCH (r:Relation)
                WHERE r.entity1_absolute_id IN $aids OR r.entity2_absolute_id IN $aids
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.entity1_absolute_id AS entity1_absolute_id,
                       r.entity2_absolute_id AS entity2_absolute_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time,
                       r.episode_id AS episode_id,
                       r.source_document AS source_document
                """,
                aids=absolute_ids,
            )
            result_map: Dict[str, List[Relation]] = {aid: [] for aid in absolute_ids}
            for record in result:
                rel = _neo4j_record_to_relation(record)
                if rel.entity1_absolute_id in result_map:
                    result_map[rel.entity1_absolute_id].append(rel)
                if rel.entity2_absolute_id in result_map:
                    result_map[rel.entity2_absolute_id].append(rel)
            return result_map

    def batch_delete_entity_versions_by_absolute_ids(self, absolute_ids: List[str]) -> int:
        """批量删除指定实体版本，返回成功删除的数量。"""
        if not absolute_ids:
            return 0
        with self._write_lock:
            with self._session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity) WHERE e.uuid IN $aids
                    DETACH DELETE e
                    RETURN count(e) AS deleted
                    """,
                    aids=absolute_ids,
                )
                record = result.single()
                return record["deleted"] if record else 0

    def split_entity_version(self, absolute_id: str, new_family_id: str = "") -> Optional[Entity]:
        """将实体拆分到新的 family_id，返回更新后的 Entity。"""
        import uuid as _uuid

        if not new_family_id:
            new_family_id = f"ent_{_uuid.uuid4().hex[:12]}"

        with self._write_lock:
            with self._session() as session:
                result = session.run(
                    f"""
                    MATCH (e:Entity {{uuid: $aid}})
                    SET e.family_id = $new_fid
                    RETURN {_ENTITY_RETURN_FIELDS}
                    """,
                    aid=absolute_id,
                    new_fid=new_family_id,
                )
                record = result.single()
                if not record:
                    return None
                entity = _neo4j_record_to_entity(record)
            self._cache.invalidate("entity:")
            self._cache.invalidate("resolve:")
            return entity

    def redirect_relation(self, family_id: str, side: str, new_family_id: str) -> int:
        """将指定 family_id 的所有关系在 side 侧重定向到 new_family_id。

        Args:
            family_id: 要重定向的关系的 family_id。
            side: "entity1" 或 "entity2"。
            new_family_id: 新目标实体的 family_id。

        Returns:
            更新的关系数量。
        """
        if side not in ("entity1", "entity2"):
            raise ValueError(f"side must be 'entity1' or 'entity2', got '{side}'")

        side_field = f"{side}_absolute_id"

        with self._relation_write_lock:
            with self._session() as session:
                # 1. 获取 new_family_id 对应的最新实体 absolute_id
                target_result = session.run(
                    """
                    MATCH (e:Entity {family_id: $fid})
                    RETURN e.uuid AS uuid
                    ORDER BY e.processed_time DESC LIMIT 1
                    """,
                    fid=new_family_id,
                )
                target_record = target_result.single()
                if not target_record:
                    return 0
                new_abs_id = target_record["uuid"]

                # 2. 更新所有匹配的关系
                update_result = session.run(
                    f"MATCH (r:Relation {{family_id: $fid}}) "
                    f"SET r.{side_field} = $new_abs_id "
                    f"RETURN count(r) AS cnt",
                    fid=family_id,
                    new_abs_id=new_abs_id,
                )
                update_record = update_result.single()
                count = update_record["cnt"] if update_record else 0
            self._cache.invalidate("relation:")
            return count
