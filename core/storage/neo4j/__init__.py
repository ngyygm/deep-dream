"""Neo4jStorageManager: 基于 Neo4j 的图存储后端。

所有数据（图结构 + embedding 向量 + 向量索引）统一存储在 Neo4j 中。
使用 Neo4j 5.11+ 原生 HNSW 向量索引进行语义搜索。
"""


import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..cache import QueryCache

# Shared helpers (constants + pure functions used by mixins)
from ._helpers import (  # noqa: F401 — re-exported for backward compat
    _ENTITY_RETURN_FIELDS,
    _RELATION_RETURN_FIELDS,
    _expand_cypher,
    _fmt_dt,
    _inject_graph_id_filter,
    _neo4j_record_to_entity,
    _neo4j_record_to_relation,
    _neo4j_types_to_native,
    _parse_dt,
    _q,
)
# Mixins
from ._base import Neo4jBaseMixin
from ._community import CommunityMixin
from ._concepts import ConceptMixin
from ._dream import DreamMixin
from ._entities import EntityStoreMixin
from ._episodes import EpisodeStoreMixin
from ._graph import GraphTraversalMixin
from ._relations import RelationStoreMixin
from ._search import SearchMixin
from ._stats import StatsMixin

logger = logging.getLogger(__name__)


class Neo4jStorageManager(Neo4jBaseMixin, EntityStoreMixin, RelationStoreMixin, EpisodeStoreMixin, SearchMixin, StatsMixin, GraphTraversalMixin, CommunityMixin, DreamMixin, ConceptMixin):

    """Neo4j 存储管理器。

    所有数据统一存储在 Neo4j 中，包括 embedding 向量和 HNSW 向量索引。

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

        # Entity name cache: avoid per-relation Neo4j sessions for embedding text
        self._entity_name_cache: Dict[str, str] = {}

        # Reverse map: doc_hash suffix → full dir name (for O(1) lookup in _get_cache_dir_by_doc_hash)
        self._doc_hash_to_dirname: Dict[str, str] = {}

        # Embedding 客户端
        self.embedding_client = embedding_client
        self.entity_content_snippet_length = entity_content_snippet_length
        self.relation_content_snippet_length = relation_content_snippet_length

        # 全量 embedding 缓存（短 TTL，避免同一 remember() 调用中重复全量加载）
        # 2026-04-26: Added max size to prevent unbounded memory growth for large graphs
        self._entity_emb_cache: Optional[List[tuple]] = None
        self._entity_emb_cache_ts: float = 0.0
        self._relation_emb_cache: Optional[List[tuple]] = None
        self._relation_emb_cache_ts: float = 0.0
        self._emb_cache_ttl: float = 5.0
        self._emb_cache_max_size: int = 10000  # Max entities to cache in memory

        # 向量维度（用于创建 HNSW 索引）
        self._vector_dim = vector_dim

        # 初始化 Neo4j 约束和索引
        self._init_schema()

        # 一次性迁移：清理无 graph_id 的遗留数据
        self._migrate_graph_id()

        # 为已有节点添加 :Concept 标签和 role 属性（幂等）
        self.migrate_to_concepts()

        # 构建缓存映射
        self._build_doc_hash_cache()


    def _ensure_database(self):
        """Community Edition: 所有图谱共享 'neo4j' 数据库。

        通过属性级别的 graph_id 过滤实现图谱隔离（而非数据库级别隔离），
        由 _run() 方法自动注入 graph_id WHERE 子句。
        """
        self._database = "neo4j"
        logger.info(
            "graph_id='%s' → using shared 'neo4j' database (Community Edition, property-level isolation)",
            self._graph_id,
        )


    def _session(self):
        """创建指向当前图谱数据库的 session。"""
        return self._driver.session(database=self._database)


    def _run(self, session, cypher: str, graph_id_safe: bool = True, **kwargs):
        """执行 Cypher 查询，自动注入 graph_id 隔离过滤。

        Args:
            session: 活跃的 neo4j session（来自 self._session()）
            cypher: Cypher 查询字符串
            graph_id_safe: True（默认）时自动为 Entity/Relation/Episode 的 MATCH
                模式注入 WHERE graph_id = $graph_id 过滤。设为 False 跳过注入
                （用于 schema 操作、EntityRedirect、ContentPatch 等全局数据）。
            **kwargs: 传递给 session.run() 的参数
        """
        if graph_id_safe:
            cypher = _inject_graph_id_filter(cypher)
            kwargs["graph_id"] = self._graph_id
        return session.run(cypher, **kwargs)


    def _init_schema(self):
        """创建 Neo4j 约束和索引（幂等）。

        Optimisation: on subsequent starts, probe one known constraint to skip
        the entire ~35-statement schema bootstrap, since all statements use
        IF NOT EXISTS (idempotent but each is a round-trip).
        """
        # Quick probe: if the entity_uuid constraint exists, assume all schema is current
        try:
            with self._session() as session:
                result = session.run("SHOW CONSTRAINTS YIELD name WHERE name = 'entity_uuid' RETURN count(*) AS cnt")
                for row in result:
                    if row["cnt"] > 0:
                        logger.debug("_init_schema: constraints already exist, skipping bootstrap")
                        return
                    break
        except Exception:
            pass  # Probe failed (e.g. older Neo4j) — fall through to full bootstrap
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
            # graph_id 复合索引（Community Edition 属性级图谱隔离）
            "CREATE INDEX entity_graph_family IF NOT EXISTS FOR (e:Entity) ON (e.graph_id, e.family_id)",
            "CREATE INDEX entity_graph_uuid IF NOT EXISTS FOR (e:Entity) ON (e.graph_id, e.uuid)",
            "CREATE INDEX relation_graph_family IF NOT EXISTS FOR (r:Relation) ON (r.graph_id, r.family_id)",
            "CREATE INDEX relation_graph_uuid IF NOT EXISTS FOR (r:Relation) ON (r.graph_id, r.uuid)",
            "CREATE INDEX episode_graph_uuid IF NOT EXISTS FOR (ep:Episode) ON (ep.graph_id, ep.uuid)",
            # Episode doc_hash index (used by find_cache_by_doc_hash, called per window)
            "CREATE INDEX episode_doc_hash IF NOT EXISTS FOR (ep:Episode) ON (ep.doc_hash)",
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
            # BM25 全文搜索索引（Neo4j 4.x/5.x 兼容语法）
            fulltext_indexes = [
                ("entityFulltext", "CREATE FULLTEXT INDEX entityFulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.content]"),
                ("relationFulltext", "CREATE FULLTEXT INDEX relationFulltext IF NOT EXISTS FOR (r:Relation) ON EACH [r.content]"),
                ("conceptFulltext", "CREATE FULLTEXT INDEX conceptFulltext IF NOT EXISTS FOR (c:Concept) ON EACH [c.name, c.content]"),
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
                # Version validity indexes — most queries filter on invalid_at IS NULL
                "CREATE INDEX entity_invalid_at IF NOT EXISTS FOR (e:Entity) ON (e.invalid_at)",
                "CREATE INDEX relation_invalid_at IF NOT EXISTS FOR (r:Relation) ON (r.invalid_at)",
                "CREATE INDEX entity_valid_at IF NOT EXISTS FOR (e:Entity) ON (e.valid_at)",
                "CREATE INDEX relation_valid_at IF NOT EXISTS FOR (r:Relation) ON (r.valid_at)",
                # Composite: graph_id + invalid_at for graph-scoped latest entity queries
                "CREATE INDEX entity_graph_invalid IF NOT EXISTS FOR (e:Entity) ON (e.graph_id, e.invalid_at)",
                # Relation: graph_id + invalid_at for graph-scoped latest relation queries
                "CREATE INDEX relation_graph_invalid IF NOT EXISTS FOR (r:Relation) ON (r.graph_id, r.invalid_at)",
                # Dream seed queries: confidence filter + community_id filter
                "CREATE INDEX entity_confidence IF NOT EXISTS FOR (e:Entity) ON (e.confidence)",
                "CREATE INDEX entity_community IF NOT EXISTS FOR (e:Entity) ON (e.community_id)",
                # 2026-04-26: Performance optimization - composite index for family_id lookup within graph
                "CREATE INDEX entity_graph_family_invalid IF NOT EXISTS FOR (e:Entity) ON (e.graph_id, e.family_id, e.invalid_at)",
                "CREATE INDEX relation_graph_family_invalid IF NOT EXISTS FOR (r:Relation) ON (r.graph_id, r.family_id, r.invalid_at)",
                # 2026-04-26: Relation endpoint indexes for graph-scoped queries filtering by entity1/entity2
                "CREATE INDEX relation_graph_entity1_invalid IF NOT EXISTS FOR (r:Relation) ON (r.graph_id, r.entity1_absolute_id, r.invalid_at)",
                "CREATE INDEX relation_graph_entity2_invalid IF NOT EXISTS FOR (r:Relation) ON (r.graph_id, r.entity2_absolute_id, r.invalid_at)",
            ]
            for idx in perf_indexes:
                try:
                    session.run(idx)
                except Exception as e:
                    logger.debug("Performance index creation skipped: %s", e)

            # 向量索引 — HNSW (Neo4j 5.11+)
            vector_dim = self._vector_dim
            vector_indexes = [
                ("entity_embedding",
                 "CREATE VECTOR INDEX entity_embedding IF NOT EXISTS "
                 "FOR (e:Entity) ON (e.embedding) "
                 "OPTIONS {indexConfig: {'vector.dimensions': $dim, 'vector.similarity_function': 'cosine'}}"),
                ("relation_embedding",
                 "CREATE VECTOR INDEX relation_embedding IF NOT EXISTS "
                 "FOR (r:Relation) ON (r.embedding) "
                 "OPTIONS {indexConfig: {'vector.dimensions': $dim, 'vector.similarity_function': 'cosine'}}"),
            ]
            for idx_name, idx_cypher in vector_indexes:
                try:
                    session.run(idx_cypher, dim=vector_dim)
                    logger.info("Vector index %s created (dim=%d)", idx_name, vector_dim)
                except Exception as e:
                    logger.warning("Vector index %s creation failed: %s", idx_name, e)


    def migrate_to_concepts(self):
        """Add :Concept label and role property to all existing nodes (idempotent).

        Runs once on startup. Skips if any :Concept nodes already exist,
        which means the migration was previously applied.
        """
        try:
            with self._session() as session:
                result = session.run("MATCH (c:Concept) RETURN count(c) AS cnt LIMIT 1")
                cnt = result.single()["cnt"]
                if cnt > 0:
                    logger.debug("migrate_to_concepts: skipped (%d Concept nodes already exist)", cnt)
                    return
                session.run("MATCH (e:Entity) SET e:Concept, e.role = 'entity'")
                session.run("MATCH (r:Relation) SET r:Concept, r.role = 'relation'")
                session.run("MATCH (ep:Episode) SET ep:Concept, ep.role = 'observation'")
                logger.info("migrate_to_concepts: applied :Concept label and role to all nodes")
        except Exception as e:
            logger.warning("migrate_to_concepts failed (non-fatal): %s", e)


    def _migrate_graph_id(self):
        """一次性迁移：删除无 graph_id 属性的遗留数据。

        在 Community Edition 属性级隔离上线后，旧数据没有 graph_id 属性。
        为确保图谱隔离的干净性，首次启动时检测并清理这些遗留数据。
        """
        try:
            with self._session() as session:
                result = session.run(
                    "MATCH (e:Entity) WHERE e.graph_id IS NULL RETURN count(e) AS cnt LIMIT 1"
                )
                cnt = result.single()["cnt"]
                if cnt > 0:
                    logger.warning(
                        "Found %d legacy Entity nodes without graph_id (pre-isolation data). "
                        "Deleting all legacy data for clean start.", cnt,
                    )
                    session.run("MATCH (e:Entity) WHERE e.graph_id IS NULL DETACH DELETE e")
                    session.run("MATCH (r:Relation) WHERE r.graph_id IS NULL DETACH DELETE r")
                    session.run("MATCH (ep:Episode) WHERE ep.graph_id IS NULL DETACH DELETE ep")
                    session.run("MATCH (cp:ContentPatch) DETACH DELETE cp")
                    logger.info("Legacy data cleanup complete (%d entities removed)", cnt)
                else:
                    logger.debug("migrate_graph_id: no legacy data found")
        except Exception as e:
            logger.warning("migrate_graph_id failed (non-fatal): %s", e)


    def _build_doc_hash_cache(self):
        """从 docs/ 目录构建 cache_id → doc_hash 映射 + doc_hash → dirname 反向映射。"""
        if not self.docs_dir.is_dir():
            return
        for doc_dir in self.docs_dir.iterdir():
            if not doc_dir.is_dir():
                continue
            dirname = doc_dir.name
            # Build reverse map: extract doc_hash from dirname
            if "_" in dirname:
                _hash_part = dirname.rpartition("_")[2]
            else:
                _hash_part = dirname
            if _hash_part and len(_hash_part) >= 8:
                self._doc_hash_to_dirname[_hash_part] = dirname
            # Build cache_id → doc_hash from meta.json
            meta_path = doc_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    cache_id = meta.get("absolute_id")
                    doc_hash = meta.get("doc_hash")
                    if cache_id and doc_hash:
                        self._id_to_doc_hash[cache_id] = doc_hash
                except Exception:
                    pass

    def close(self):
        """关闭 Neo4j 驱动。"""
        try:
            self._driver.close()
        except Exception as e:
            logger.warning("Error closing Neo4j driver: %s", e)
