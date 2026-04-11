"""存储层：SQLite数据库 + Markdown文件存储

StorageManager now inherits from four mixin modules:
  - EntityStoreMixin: entity CRUD, search, merge, version management
  - RelationStoreMixin: relation CRUD, search, timeline queries
  - EpisodeStoreMixin: episode management, migration, search
  - ConceptStoreMixin: unified concept queries, dual-write, migration
"""
import sqlite3
import threading
import json
import time
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Literal, Tuple, Set
from pathlib import Path

logger = logging.getLogger(__name__)
import hashlib
import numpy as np
import difflib

from ..models import ContentPatch, Episode, Entity, Relation
from ..utils import clean_markdown_code_blocks, wprint, calculate_jaccard_similarity
from .mixins.entity_store import EntityStoreMixin
from .mixins.relation_store import RelationStoreMixin
from .mixins.episode_store import EpisodeStoreMixin
from .mixins.concept_store import ConceptStoreMixin


class StorageManager(EntityStoreMixin, RelationStoreMixin, EpisodeStoreMixin, ConceptStoreMixin):
    """存储管理器 — composed from four domain-specific mixins.

    Core responsibilities kept here:
      - SQLite connection management (_get_conn, _ensure_tables)
      - Row conversion helpers (_row_to_entity, _row_to_relation)
      - Family ID resolution (resolve_family_id, resolve_family_ids)
      - Cross-cutting queries (get_graph_statistics, find_shortest_paths)
      - Embedding cache invalidation
      - Entity/relation similarity search helpers
    """

    # SELECT column lists — single source of truth for Entity/Relation reads
    _ENTITY_SELECT = (
        "id, family_id, name, content, event_time, processed_time, "
        "episode_id, source_document, embedding, summary, attributes, "
        "confidence, valid_at, invalid_at"
    )
    _RELATION_SELECT = (
        "id, family_id, entity1_absolute_id, entity2_absolute_id, "
        "content, event_time, processed_time, episode_id, source_document, "
        "embedding, summary, attributes, confidence, valid_at, invalid_at"
    )

    def _row_to_entity(self, row) -> Entity:
        """Convert a SELECT row tuple to an Entity object using _ENTITY_SELECT column order."""
        return Entity(
            absolute_id=row[0],
            family_id=row[1],
            name=row[2],
            content=row[3],
            event_time=self._safe_parse_datetime(row[4]),
            processed_time=self._safe_parse_datetime(row[5]),
            episode_id=row[6],
            source_document=row[7] if len(row) > 7 else '',
            embedding=row[8] if len(row) > 8 else None,
            summary=row[9] if len(row) > 9 else None,
            attributes=row[10] if len(row) > 10 else None,
            confidence=row[11] if len(row) > 11 else None,
            valid_at=self._safe_parse_datetime(row[12]) if len(row) > 12 and row[12] else None,
            invalid_at=self._safe_parse_datetime(row[13]) if len(row) > 13 and row[13] else None,
        )

    def _row_to_relation(self, row) -> Relation:
        """Convert a SELECT row tuple to a Relation object using _RELATION_SELECT column order."""
        return Relation(
            absolute_id=row[0],
            family_id=row[1],
            entity1_absolute_id=row[2],
            entity2_absolute_id=row[3],
            content=row[4],
            event_time=self._safe_parse_datetime(row[5]),
            processed_time=self._safe_parse_datetime(row[6]),
            episode_id=row[7],
            source_document=row[8] if len(row) > 8 else '',
            embedding=row[9] if len(row) > 9 else None,
            summary=row[10] if len(row) > 10 else None,
            attributes=row[11] if len(row) > 11 else None,
            confidence=row[12] if len(row) > 12 else None,
            valid_at=self._safe_parse_datetime(row[13]) if len(row) > 13 and row[13] else None,
            invalid_at=self._safe_parse_datetime(row[14]) if len(row) > 14 and row[14] else None,
        )

    def __init__(self, storage_path: str, embedding_client=None,
                 entity_content_snippet_length: int = 50,
                 relation_content_snippet_length: int = 50):
        """
        初始化存储管理器

        Args:
            storage_path: 存储路径
            embedding_client: Embedding客户端（可选）
            entity_content_snippet_length: 实体embedding计算时的content截取长度
            relation_content_snippet_length: 关系embedding计算时的content截取长度
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 新目录结构
        self.db_path = self.storage_path / "graph.db"
        self.docs_dir = self.storage_path / "docs"
        self.docs_dir.mkdir(exist_ok=True)

        # 保留旧目录引用（用于迁移和向后兼容读取）
        self.cache_dir = self.storage_path / "episodes"
        self.cache_json_dir = self.cache_dir / "json"
        self.cache_md_dir = self.cache_dir / "md"

        # 缓存 cache_id → doc_hash 映射（用于从 cache_id 反查文档目录）
        self._id_to_doc_hash: Dict[str, str] = {}

        # 线程局部连接（每个线程复用同一个连接）
        self._local = threading.local()

        # 写锁：序列化所有 SQLite 写操作，防止多线程并发写入导致 "database is locked"
        self._write_lock = threading.Lock()

        # Embedding客户端
        self.embedding_client = embedding_client
        self.entity_content_snippet_length = entity_content_snippet_length
        self.relation_content_snippet_length = relation_content_snippet_length

        # 全量 embedding 缓存（短 TTL，避免同一 remember() 调用中重复全表扫描）
        self._entity_emb_cache: Optional[List[tuple]] = None
        self._entity_emb_cache_ts: float = 0.0
        self._relation_emb_cache: Optional[List[tuple]] = None
        self._relation_emb_cache_ts: float = 0.0
        self._concept_emb_cache: Optional[List[tuple]] = None
        self._concept_emb_cache_ts: float = 0.0
        self._emb_cache_ttl: float = 5.0  # 秒
        self._emb_cache_lock = threading.Lock()

        # 初始化数据库
        self._init_database()

        # 自动迁移旧目录结构
        self._migrate_storage()

        # Phase 1: 将已有 Episode 文件元数据迁移到 SQLite
        self._migrate_episodes_from_files()

        # Phase 2: 将已有 entities/relations/episodes 迁移到 concepts 统一表
        self.migrate_to_concepts()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_column(cursor, table: str, column: str, col_type: str):
        """幂等地为已有表添加缺失列（旧库迁移）。"""
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        except sqlite3.OperationalError:
            pass  # 列已存在

    def _ensure_tables(self, conn):
        """在已有连接上确保表结构存在（数据库文件被删除后重建场景）。
        仅执行 CREATE TABLE IF NOT EXISTS，不重复做迁移。"""
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                family_id TEXT NOT NULL,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                event_time TEXT NOT NULL,
                processed_time TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                source_document TEXT DEFAULT '',
                embedding BLOB
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                family_id TEXT NOT NULL,
                entity1_absolute_id TEXT NOT NULL,
                entity2_absolute_id TEXT NOT NULL,
                content TEXT NOT NULL,
                event_time TEXT NOT NULL,
                processed_time TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                source_document TEXT DEFAULT '',
                embedding BLOB
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS entity_redirects (
                source_family_id TEXT PRIMARY KEY,
                target_family_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_family_id ON entities(family_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_entity_event_time ON entities(event_time)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_entity_processed_time ON entities(processed_time)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_relation_id ON relations(family_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_relation_entities ON relations(entity1_absolute_id, entity2_absolute_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_relation_event_time ON relations(event_time)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_relation_processed_time ON relations(processed_time)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_entity_redirect_target ON entity_redirects(target_family_id)")
        # 唯一索引：防止并行创建时产生重复版本
        try:
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_unique ON entities(family_id, processed_time)")
        except sqlite3.OperationalError:
            pass  # 索引已存在或存在重复数据，忽略
        try:
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_relation_unique ON relations(family_id, processed_time)")
        except sqlite3.OperationalError:
            pass
        # 为旧库自动添加缺失列（幂等）
        self._ensure_column(c, "entities", "source_document", "TEXT DEFAULT ''")
        self._ensure_column(c, "relations", "source_document", "TEXT DEFAULT ''")
        self._ensure_column(c, "entities", "valid_at", "TEXT")
        self._ensure_column(c, "entities", "invalid_at", "TEXT")
        self._ensure_column(c, "relations", "valid_at", "TEXT")
        self._ensure_column(c, "relations", "invalid_at", "TEXT")
        self._ensure_column(c, "entities", "summary", "TEXT")
        self._ensure_column(c, "entities", "attributes", "TEXT")
        self._ensure_column(c, "entities", "confidence", "REAL")
        self._ensure_column(c, "relations", "summary", "TEXT")
        self._ensure_column(c, "relations", "attributes", "TEXT")
        self._ensure_column(c, "relations", "confidence", "REAL")
        self._ensure_column(c, "relations", "provenance", "TEXT")
        self._ensure_column(c, "entities", "content_format", "TEXT DEFAULT 'plain'")
        self._ensure_column(c, "relations", "content_format", "TEXT DEFAULT 'plain'")
        # ContentPatch: section 级变更记录
        c.execute("""
            CREATE TABLE IF NOT EXISTS content_patches (
                uuid TEXT PRIMARY KEY,
                target_type TEXT NOT NULL,
                target_absolute_id TEXT NOT NULL,
                target_family_id TEXT NOT NULL,
                section_key TEXT NOT NULL,
                change_type TEXT NOT NULL,
                old_hash TEXT DEFAULT '',
                new_hash TEXT DEFAULT '',
                diff_summary TEXT DEFAULT '',
                source_document TEXT DEFAULT '',
                event_time TEXT NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_cp_target_abs ON content_patches(target_absolute_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_cp_family_id ON content_patches(target_family_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_cp_section ON content_patches(target_family_id, section_key)")
        # Episodes (Phase 1: Episode 入库，支持图查询)
        c.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                family_id TEXT NOT NULL,
                content TEXT NOT NULL,
                event_time TEXT NOT NULL,
                processed_time TEXT NOT NULL,
                source_document TEXT DEFAULT '',
                activity_type TEXT DEFAULT '',
                episode_type TEXT DEFAULT '',
                doc_hash TEXT DEFAULT '',
                embedding BLOB
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_episodes_family ON episodes(family_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_episodes_event_time ON episodes(event_time)")
        # Episode mentions (Phase 1: 扩展为 entity + relation)
        c.execute("""
            CREATE TABLE IF NOT EXISTS episode_mentions (
                episode_id TEXT NOT NULL,
                target_absolute_id TEXT NOT NULL,
                target_type TEXT NOT NULL DEFAULT 'entity',
                mention_context TEXT DEFAULT '',
                PRIMARY KEY (episode_id, target_absolute_id, target_type)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_episode_mentions_target ON episode_mentions(target_absolute_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_episode_mentions_episode ON episode_mentions(episode_id)")
        # 兼容旧表：如果存在旧 episode_mentions 表只有 (episode_id, entity_absolute_id) 列
        self._migrate_episode_mentions(c)
        # Dream logs
        c.execute("""
            CREATE TABLE IF NOT EXISTS dream_logs (
                cycle_id TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                status TEXT DEFAULT 'running',
                narrative TEXT DEFAULT '',
                insights_json TEXT DEFAULT '[]',
                connections_json TEXT DEFAULT '[]',
                consolidations_json TEXT DEFAULT '[]',
                config_json TEXT DEFAULT '{}'
            )
        """)
        # BM25 全文搜索虚拟表
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS entity_fts USING fts5(name, content, family_id UNINDEXED)
        """)
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS relation_fts USING fts5(content, family_id UNINDEXED)
        """)
        # Phase 2: 统一 concepts 表（entity + relation + observation 三合一）
        c.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                family_id TEXT NOT NULL,
                role TEXT NOT NULL,
                name TEXT DEFAULT '',
                content TEXT NOT NULL,
                event_time TEXT NOT NULL,
                processed_time TEXT NOT NULL,
                source_document TEXT DEFAULT '',
                episode_id TEXT DEFAULT '',
                embedding BLOB,
                connects TEXT DEFAULT '',
                activity_type TEXT DEFAULT '',
                episode_type TEXT DEFAULT '',
                valid_at TEXT,
                invalid_at TEXT,
                summary TEXT,
                attributes TEXT,
                confidence REAL,
                content_format TEXT DEFAULT 'plain',
                provenance TEXT DEFAULT ''
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_concepts_family ON concepts(family_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_concepts_role ON concepts(role)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(name)")
        try:
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_concepts_unique ON concepts(family_id, processed_time)")
        except sqlite3.OperationalError:
            pass  # 索引已存在或存在重复数据，忽略
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS concept_fts USING fts5(name, content, content='concepts', content_rowid='rowid')
        """)
        conn.commit()

    def _ensure_dirs(self):
        """确保关键目录存在（运行中目录被删除时自动恢复）。"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(exist_ok=True)

    def _get_conn(self):
        """获取当前线程的 SQLite 连接（线程局部复用，启用 WAL 模式）。
        如果连接失效或目录/数据库丢失，自动重建表结构。"""
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            try:
                conn.execute("SELECT 1 FROM entities LIMIT 0")
                return conn
            except Exception as _conn_err:
                logger.debug("连接健康检查失败，重建连接: %s", _conn_err)
                try:
                    conn.close()
                except Exception as _close_err:
                    logger.debug("关闭失效连接失败: %s", _close_err)
                self._local.conn = None
        # 确保目录存在
        self._ensure_dirs()
        max_retries = 3
        last_err = None
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=30000")
                conn.execute("PRAGMA foreign_keys=ON")
                # 确保表结构存在（数据库文件被删除后重建场景）
                self._ensure_tables(conn)
                self._local.conn = conn
                return conn
            except sqlite3.OperationalError as e:
                last_err = e
                logger.warning("_get_conn: 第 %d 次连接失败 (%s), 路径=%s, 重试中...", attempt + 1, e, self.db_path)
                # 连接失败时清理可能残留的半开连接
                self._local.conn = None
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    self._ensure_dirs()
        raise last_err  # type: ignore[misc]

    # ========== ContentPatch 操作 ==========

    def save_content_patches(self, patches: list):
        """批量保存 ContentPatch 记录到 SQLite。"""

        if not patches:
            return
        conn = self._get_conn()
        with self._write_lock:
            cursor = conn.cursor()
            for p in patches:
                cursor.execute("""
                    INSERT OR REPLACE INTO content_patches
                    (uuid, target_type, target_absolute_id, target_family_id,
                     section_key, change_type, old_hash, new_hash, diff_summary,
                     source_document, event_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    p.uuid, p.target_type, p.target_absolute_id, p.target_family_id,
                    p.section_key, p.change_type, p.old_hash, p.new_hash,
                    p.diff_summary, p.source_document,
                    p.event_time.isoformat() if p.event_time else datetime.now().isoformat(),
                ))
            conn.commit()

    def get_content_patches(self, family_id: str, section_key: str = None) -> list:
        """查询指定 family_id 的 ContentPatch 记录。"""

        conn = self._get_conn()
        cursor = conn.cursor()
        if section_key:
            cursor.execute(
                "SELECT * FROM content_patches WHERE target_family_id = ? AND section_key = ? ORDER BY event_time DESC",
                (family_id, section_key),
            )
        else:
            cursor.execute(
                "SELECT * FROM content_patches WHERE target_family_id = ? ORDER BY event_time DESC",
                (family_id,),
            )
        patches = []
        for row in cursor.fetchall():
            patches.append(ContentPatch(
                uuid=row[0],
                target_type=row[1],
                target_absolute_id=row[2],
                target_family_id=row[3],
                section_key=row[4],
                change_type=row[5],
                old_hash=row[6] or "",
                new_hash=row[7] or "",
                diff_summary=row[8] or "",
                source_document=row[9] or "",
                event_time=datetime.fromisoformat(row[10]) if row[10] else datetime.now(),
            ))
        return patches

    def get_section_history(self, family_id: str, section_key: str) -> list:
        """获取单个 section 的全版本变更历史。"""
        return self.get_content_patches(family_id, section_key=section_key)

    def get_version_diff(self, family_id: str, v1: str, v2: str) -> dict:
        """获取两个版本之间的 section 级 diff。"""
        from ..content_schema import parse_markdown_sections, compute_section_diff
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, content FROM entities WHERE id IN (?, ?)",
            (v1, v2),
        )
        rows = {row[0]: row[1] for row in cursor.fetchall()}
        s1 = parse_markdown_sections(rows.get(v1, "") or "")
        s2 = parse_markdown_sections(rows.get(v2, "") or "")
        return compute_section_diff(s1, s2)

    def close(self):
        """关闭当前线程的数据库连接。"""
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _migrate_storage(self):
        """启动时自动将旧目录结构迁移到新结构（幂等，旧目录保留不删）。"""
        # 1. 迁移 remember_journal/ → tasks/
        old_journal = self.storage_path / "remember_journal"
        new_journal = self.storage_path / "tasks"
        if old_journal.is_dir() and not new_journal.exists():
            try:
                old_journal.rename(new_journal)
                wprint(f"[迁移] {old_journal} → {new_journal}")
            except OSError as e:
                wprint(f"[迁移警告] remember_journal 重命名失败: {e}")

        # 1.5 迁移旧的任务独立 JSON 文件 → queue.jsonl（仅保留未完成的任务）
        tasks_dir = self.storage_path / "tasks"
        queue_file = tasks_dir / "queue.jsonl"
        if tasks_dir.is_dir() and not queue_file.exists():
            old_json_files = list(tasks_dir.glob("*.json"))
            if old_json_files:
                try:
                    migrated = 0
                    lines: list = []
                    for jf in old_json_files:
                        if jf.name.endswith(".tmp") or jf.name.endswith(".bad.json"):
                            continue
                        try:
                            rec = json.loads(jf.read_text(encoding="utf-8"))
                            st = rec.get("status")
                            if st in ("queued", "running"):
                                lines.append(json.dumps(rec, ensure_ascii=False))
                                migrated += 1
                        except Exception as exc:
                            logger.debug("task json parse failed: %s", exc)
                    if lines:
                        queue_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
                        wprint(f"[迁移] {len(old_json_files)} 个旧任务文件 → queue.jsonl（{migrated} 个未完成）")
                    # 清理旧的独立 JSON 文件
                    for jf in old_json_files:
                        try:
                            jf.unlink()
                        except Exception as exc:
                            logger.debug("task file unlink failed: %s", exc)
                except Exception as e:
                    wprint(f"[迁移警告] 任务文件迁移失败: {e}")

        # 2. 迁移 memory_caches/ → docs/
        old_cache_json = self.storage_path / "memory_caches" / "json"
        if old_cache_json.is_dir():
            for json_file in old_cache_json.glob("*.json"):
                try:
                    meta = json.loads(json_file.read_text(encoding="utf-8"))
                    text = meta.get("text", "")
                    if not text:
                        continue
                    doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
                    doc_dir = self.docs_dir / doc_hash
                    doc_dir.mkdir(parents=True, exist_ok=True)

                    # 迁移原始文本
                    original_path = doc_dir / "original.txt"
                    if not original_path.exists():
                        original_path.write_text(text, encoding="utf-8")

                    # 迁移元数据
                    new_meta = {
                        "absolute_id": meta.get("id"),
                        "event_time": meta.get("event_time"),
                        "activity_type": meta.get("activity_type"),
                        "source_document": meta.get("source_document") or meta.get("doc_name", ""),
                        "text": text,
                        "document_path": meta.get("document_path", ""),
                        "doc_hash": doc_hash,
                    }
                    (doc_dir / "meta.json").write_text(
                        json.dumps(new_meta, ensure_ascii=False, indent=2), encoding="utf-8"
                    )

                    # 迁移 cache.md
                    cache_id = meta.get("id", "")
                    md_file = (self.storage_path / "memory_caches" / "md" / f"{cache_id}.md")
                    if md_file.exists():
                        (doc_dir / "cache.md").write_text(md_file.read_text(encoding="utf-8"), encoding="utf-8")

                    # 更新缓存映射
                    if cache_id:
                        self._id_to_doc_hash[cache_id] = doc_hash
                except Exception as e:
                    wprint(f"[迁移警告] 跳过文件 {json_file}: {e}")

        # 3. 迁移 originals/ 中独立保存的文件（未被 memory_caches 引用的）
        old_originals = self.storage_path / "originals"
        if old_originals.is_dir():
            for txt_file in old_originals.glob("*.txt"):
                try:
                    text = txt_file.read_text(encoding="utf-8")
                    doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
                    doc_dir = self.docs_dir / doc_hash
                    doc_dir.mkdir(parents=True, exist_ok=True)
                    original_path = doc_dir / "original.txt"
                    if not original_path.exists():
                        original_path.write_text(text, encoding="utf-8")
                except Exception as e:
                    wprint(f"[迁移警告] 跳过文件 {txt_file}: {e}")

        # 4. 构建新结构中已有的 id→doc_hash 映射
        if self.docs_dir.is_dir():
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
                    except Exception as exc:
                        logger.debug("directory rename failed: %s", exc)

    @staticmethod
    def _safe_parse_datetime(value: Any, default: Optional[datetime] = None) -> Optional[datetime]:
        """安全解析 ISO 格式时间字符串，解析失败返回 default。"""
        if value is None:
            return default
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                return default
        return default

    def _normalize_datetime_for_compare(self, t: Optional[Any]) -> datetime:
        """将时间归一为可比较的 naive datetime，供版本按时间排序（处理 None / 字符串 / 时区）。"""
        if t is None:
            return datetime.min
        if isinstance(t, str):
            t = datetime.fromisoformat(t.replace('Z', '+00:00'))
        if getattr(t, 'tzinfo', None) is not None and t.tzinfo is not None:
            return t.astimezone(timezone.utc).replace(tzinfo=None)
        return t

    def _init_database(self):
        """初始化SQLite数据库（使用独立连接，此时线程池尚未启用）。"""
        conn = sqlite3.connect(str(self.db_path))
        self._ensure_tables(conn)
        conn.close()

    def _resolve_family_id_with_cursor(self, cursor, family_id: str) -> str:
        """沿 redirect 链解析到当前 canonical family_id。"""
        current_id = (family_id or "").strip()
        if not current_id:
            return ""
        seen: Set[str] = set()
        while current_id and current_id not in seen:
            seen.add(current_id)
            cursor.execute(
                "SELECT target_family_id FROM entity_redirects WHERE source_family_id = ?",
                (current_id,),
            )
            row = cursor.fetchone()
            if not row or not row[0] or row[0] == current_id:
                break
            current_id = row[0]
        return current_id

    def resolve_family_ids(self, family_ids: List[str]) -> Dict[str, str]:
        """批量解析 family_id 到 canonical id。一次 SQL 查询获取所有重定向。

        Returns:
            {原始 family_id: canonical family_id} 映射
        """
        if not family_ids:
            return {}
        unique_ids = list(set(fid.strip() for fid in family_ids if fid and fid.strip()))
        if not unique_ids:
            return {}

        conn = self._get_conn()
        cursor = conn.cursor()

        # 一次查询获取所有重定向
        placeholders = ",".join("?" * len(unique_ids))
        cursor.execute(
            f"SELECT source_family_id, target_family_id FROM entity_redirects WHERE source_family_id IN ({placeholders})",
            unique_ids,
        )
        redirect_map: Dict[str, str] = {}
        for row in cursor.fetchall():
            redirect_map[row[0]] = row[1]

        # 沿重定向链解析（通常链长 <= 2，几乎不需要再次查库）
        result: Dict[str, str] = {}
        needs_second_hop: Dict[str, str] = {}  # source -> intermediate target

        for fid in unique_ids:
            target = redirect_map.get(fid)
            if target and target != fid:
                needs_second_hop[fid] = target
            else:
                result[fid] = fid

        # 第二跳：检查中间 target 是否也有重定向
        if needs_second_hop:
            intermediate_ids = list(set(needs_second_hop.values()))
            placeholders2 = ",".join("?" * len(intermediate_ids))
            cursor.execute(
                f"SELECT source_family_id, target_family_id FROM entity_redirects WHERE source_family_id IN ({placeholders2})",
                intermediate_ids,
            )
            second_hop = {row[0]: row[1] for row in cursor.fetchall()}

            for fid, intermediate in needs_second_hop.items():
                final = second_hop.get(intermediate, intermediate)
                result[fid] = final

        # 为未在 unique_ids 中的原始 family_ids 生成映射
        output: Dict[str, str] = {}
        for fid in family_ids:
            key = fid.strip() if fid else ""
            output[fid] = result.get(key, key)

        return output

    def resolve_family_id(self, family_id: str) -> str:
        """解析 family_id 到当前 canonical id；不存在映射时原样返回。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        return self._resolve_family_id_with_cursor(cursor, family_id)

    def register_entity_redirect(self, source_family_id: str, target_family_id: str) -> str:
        """登记旧 family_id 到 canonical family_id 的映射，支持链式合并。"""
        source_id = (source_family_id or "").strip()
        target_id = (target_family_id or "").strip()
        if not source_id or not target_id:
            return target_id
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            canonical_target = self._resolve_family_id_with_cursor(cursor, target_id)
            if not canonical_target:
                canonical_target = target_id
            canonical_source = self._resolve_family_id_with_cursor(cursor, source_id)
            if canonical_source == canonical_target:
                return canonical_target
            cursor.execute(
                """
                INSERT INTO entity_redirects (source_family_id, target_family_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(source_family_id) DO UPDATE SET
                    target_family_id = excluded.target_family_id,
                    updated_at = excluded.updated_at
                """,
                (source_id, canonical_target, datetime.now().isoformat()),
            )
            conn.commit()
            return canonical_target


    # ========== Episode 操作 ==========


    def _invalidate_emb_cache(self):
        """清除 embedding 缓存（在实体/关系写入时调用）。"""
        with self._emb_cache_lock:
            self._entity_emb_cache = None
            self._entity_emb_cache_ts = 0.0
            self._relation_emb_cache = None
            self._relation_emb_cache_ts = 0.0
            self._concept_emb_cache = None
            self._concept_emb_cache_ts = 0.0


    def _search_with_embedding(self, query_text: str, entities_with_embeddings: List[tuple], 
                               threshold: float, use_content: bool = False, 
                               max_results: int = 10, content_snippet_length: int = 50,
                               text_mode: Literal["name_only", "content_only", "name_and_content"] = "name_and_content") -> List[Entity]:
        """使用embedding向量进行相似度搜索（优先使用已存储的embedding）"""
        # 编码查询文本
        query_embedding = self.embedding_client.encode(query_text)
        if query_embedding is None:
            # 如果编码失败，回退到文本相似度
            all_entities = [e for e, _ in entities_with_embeddings]
            return self._search_with_text_similarity(
                query_text, all_entities, threshold, use_content, max_results, content_snippet_length, text_mode, "text"
            )
        
        query_embedding_array = np.array(query_embedding[0] if isinstance(query_embedding, (list, np.ndarray)) else query_embedding, dtype=np.float32)
        
        # 收集已存储的embedding和需要重新计算的实体
        stored_embeddings = []
        entities_to_encode = []
        entity_indices = []
        
        for idx, (entity, stored_embedding) in enumerate(entities_with_embeddings):
            if stored_embedding is not None:
                stored_embeddings.append((idx, stored_embedding))
            else:
                entities_to_encode.append(entity)
                entity_indices.append(idx)
        
        # 如果有需要重新计算的实体，进行编码
        if entities_to_encode:
            # 根据text_mode构建实体文本
            entity_texts = []
            for entity in entities_to_encode:
                if text_mode == "name_only":
                    entity_texts.append(entity.name)
                elif text_mode == "content_only":
                    entity_texts.append(entity.content[:content_snippet_length])
                else:  # name_and_content
                    if use_content:
                        entity_texts.append(f"{entity.name} {entity.content[:content_snippet_length]}")
                    else:
                        entity_texts.append(entity.name)
            
            new_embeddings = self.embedding_client.encode(entity_texts)
            if new_embeddings is not None:
                # 将新计算的embedding添加到存储列表中
                for i, entity in enumerate(entities_to_encode):
                    embedding_array = np.array(new_embeddings[i] if isinstance(new_embeddings, (list, np.ndarray)) else new_embeddings, dtype=np.float32)
                    stored_embeddings.append((entity_indices[i], embedding_array))
        
        if not stored_embeddings:
            # 如果没有可用的embedding，回退到文本相似度
            all_entities = [e for e, _ in entities_with_embeddings]
            return self._search_with_text_similarity(
                query_text, all_entities, threshold, use_content, max_results, content_snippet_length, text_mode, "text"
            )
        
        # 计算相似度
        similarities = []
        for idx, stored_embedding in stored_embeddings:
            # 计算余弦相似度
            dot_product = np.dot(query_embedding_array, stored_embedding)
            norm_query = np.linalg.norm(query_embedding_array)
            norm_stored = np.linalg.norm(stored_embedding)
            similarity = dot_product / (norm_query * norm_stored + 1e-9)
            entity = entities_with_embeddings[idx][0]
            similarities.append((entity, float(similarity)))
        
        # 筛选和排序
        scored_entities = [(entity, sim) for entity, sim in similarities if sim >= threshold]
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回实体列表（去重，每个family_id只保留一个，并限制最大数量）
        entities = []
        seen_ids = set()
        for entity, _ in scored_entities:
            if entity.family_id not in seen_ids:
                entities.append(entity)
                seen_ids.add(entity.family_id)
                # 达到最大数量后停止
                if len(entities) >= max_results:
                    break
        
        return entities
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """计算Jaccard相似度（委托给共享工具函数）"""
        return calculate_jaccard_similarity(text1, text2)
    
    def _calculate_bleu_similarity(self, text1: str, text2: str) -> float:
        """计算BLEU相似度（基于字符n-gram）"""
        # 简化的BLEU计算：使用字符级别的1-gram和2-gram
        def get_char_ngrams(text, n):
            return [text[i:i+n] for i in range(len(text)-n+1)]
        
        ngrams1_1 = set(get_char_ngrams(text1.lower(), 1))
        ngrams2_1 = set(get_char_ngrams(text2.lower(), 1))
        ngrams1_2 = set(get_char_ngrams(text1.lower(), 2))
        ngrams2_2 = set(get_char_ngrams(text2.lower(), 2))
        
        # 计算1-gram和2-gram的精确匹配率
        precision_1 = len(ngrams1_1 & ngrams2_1) / max(len(ngrams1_1), 1)
        precision_2 = len(ngrams1_2 & ngrams2_2) / max(len(ngrams1_2), 1)
        
        # 简化的BLEU分数（几何平均）
        if precision_1 == 0 or precision_2 == 0:
            return 0.0
        return (precision_1 * precision_2) ** 0.5
    
    def _search_with_text_similarity(self, query_text: str, all_entities: List[Entity],
                                     threshold: float, use_content: bool = False,
                                     max_results: int = 10, content_snippet_length: int = 50,
                                     text_mode: Literal["name_only", "content_only", "name_and_content"] = "name_and_content",
                                     similarity_method: Literal["text", "jaccard", "bleu"] = "text") -> List[Entity]:
        """使用文本相似度进行搜索"""
        # 计算相似度并筛选
        scored_entities = []
        for entity in all_entities:
            # 根据text_mode构建实体文本
            if text_mode == "name_only":
                entity_text = entity.name
            elif text_mode == "content_only":
                entity_text = entity.content[:content_snippet_length]
            else:  # name_and_content
                if use_content:
                    entity_text = f"{entity.name} {entity.content[:content_snippet_length]}"
                else:
                    entity_text = entity.name
            
            # 根据similarity_method计算相似度
            if similarity_method == "jaccard":
                similarity = self._calculate_jaccard_similarity(query_text, entity_text)
            elif similarity_method == "bleu":
                similarity = self._calculate_bleu_similarity(query_text, entity_text)
            else:  # text (SequenceMatcher)
                similarity = difflib.SequenceMatcher(
                    None, 
                    query_text.lower(), 
                    entity_text.lower()
                ).ratio()
            
            if similarity >= threshold:
                scored_entities.append((entity, similarity))
        
        # 按相似度排序
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回实体列表（去重，每个family_id只保留一个，并限制最大数量）
        entities = []
        seen_ids = set()
        for entity, _ in scored_entities:
            if entity.family_id not in seen_ids:
                entities.append(entity)
                seen_ids.add(entity.family_id)
                # 达到最大数量后停止
                if len(entities) >= max_results:
                    break
        
        return entities
    
    # ========== Relation 操作 ==========
    

    def get_graph_statistics(self) -> Dict[str, Any]:
        """返回图谱结构统计数据"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # 基础计数
        cursor.execute("SELECT COUNT(*) FROM entities")
        entity_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM relations")
        relation_count = cursor.fetchone()[0]

        stats = {
            "entity_count": entity_count,
            "relation_count": relation_count,
        }

        # 平均关系数 / 实体
        if entity_count > 0:
            cursor.execute("""
                SELECT AVG(cnt) FROM (
                    SELECT COUNT(*) as cnt FROM (
                        SELECT entity1_absolute_id AS abs_id FROM relations
                        UNION ALL
                        SELECT entity2_absolute_id AS abs_id FROM relations
                    ) GROUP BY abs_id
                )
            """)
            row = cursor.fetchone()
            stats["avg_relations_per_entity"] = round(row[0], 2) if row and row[0] else 0

            # 最大关系数
            cursor.execute("""
                SELECT MAX(cnt) FROM (
                    SELECT COUNT(*) as cnt FROM (
                        SELECT entity1_absolute_id AS abs_id FROM relations
                        UNION ALL
                        SELECT entity2_absolute_id AS abs_id FROM relations
                    ) GROUP BY abs_id
                )
            """)
            row = cursor.fetchone()
            stats["max_relations_per_entity"] = row[0] if row and row[0] else 0

            # 孤立实体数
            cursor.execute("""
                SELECT COUNT(*) FROM entities e
                WHERE e.id NOT IN (
                    SELECT entity1_absolute_id FROM relations
                    UNION
                    SELECT entity2_absolute_id FROM relations
                )
            """)
            stats["isolated_entities"] = cursor.fetchone()[0]

            # 图密度 (实际边数 / 最大可能边数)
            cursor.execute("SELECT COUNT(DISTINCT family_id) FROM entities")
            unique_entities = cursor.fetchone()[0]
            if unique_entities > 1:
                max_possible = unique_entities * (unique_entities - 1) / 2
                cursor.execute("SELECT COUNT(DISTINCT family_id) FROM relations")
                unique_relations = cursor.fetchone()[0]
                stats["graph_density"] = round(unique_relations / max_possible, 4)
            else:
                stats["graph_density"] = 0.0
        else:
            stats["avg_relations_per_entity"] = 0
            stats["max_relations_per_entity"] = 0
            stats["isolated_entities"] = entity_count
            stats["graph_density"] = 0.0

        # 时间趋势
        cursor.execute("""
            SELECT DATE(event_time) as d, COUNT(*) as cnt
            FROM entities
            GROUP BY d
            ORDER BY d
            LIMIT 30
        """)
        stats["entity_count_over_time"] = [{"date": r[0], "count": r[1]} for r in cursor.fetchall()]

        cursor.execute("""
            SELECT DATE(event_time) as d, COUNT(*) as cnt
            FROM relations
            GROUP BY d
            ORDER BY d
            LIMIT 30
        """)
        stats["relation_count_over_time"] = [{"date": r[0], "count": r[1]} for r in cursor.fetchall()]

        return stats


    def find_shortest_paths(self, source_family_id: str, target_family_id: str,
                            max_depth: int = 6, max_paths: int = 10) -> Dict[str, Any]:
        """使用 BFS 查找两个实体之间的所有最短路径。

        在 family_id 级别的无向图上执行 BFS，找到所有等长的最短路径，
        然后重构路径中每对相邻实体之间的连接关系。

        使用 bounded neighborhood-expansion BFS，每次只查询当前 frontier 的邻居，
        避免加载全量关系到内存中。

        Args:
            source_family_id: 起始实体的 family_id
            target_family_id: 目标实体的 family_id
            max_depth: 最大搜索深度（默认6）
            max_paths: 最多返回的路径数量（默认10）

        Returns:
            {
                "source_entity": Entity | None,
                "target_entity": Entity | None,
                "path_length": int,   # -1=不可达, 0=同一实体
                "total_shortest_paths": int,
                "paths": [{
                    "entities": [Entity, ...],
                    "relations": [Relation, ...],
                    "length": int,
                }, ...]
            }
        """
        result_empty = {
            "source_entity": None,
            "target_entity": None,
            "path_length": -1,
            "total_shortest_paths": 0,
            "paths": [],
        }

        # 1. 验证实体存在
        source_entity = self.get_entity_by_family_id(source_family_id)
        target_entity = self.get_entity_by_family_id(target_family_id)

        if not source_entity or not target_entity:
            result_empty["source_entity"] = source_entity
            result_empty["target_entity"] = target_entity
            return result_empty

        # 同一实体
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

        # 2. Bounded BFS: expand neighbors level-by-level using SQL queries
        # instead of loading all relations into memory.
        conn = self._get_conn()
        cursor = conn.cursor()

        # visited: family_id → distance from source
        # parents: family_id → list of parent family_ids on shortest paths
        visited: Dict[str, int] = {source_family_id: 0}
        parents: Dict[str, List[str]] = {source_family_id: []}
        # Cache: family_id → set of absolute_ids for each visited family_id
        family_abs_ids: Dict[str, List[str]] = {}
        # Cache: sorted_pair → [Relation] for explored edges
        pair_relations: Dict[Tuple[str, str], List[Relation]] = {}

        # Pre-load source entity's absolute_ids
        cursor.execute("SELECT id FROM entities WHERE family_id = ?", (source_family_id,))
        family_abs_ids[source_family_id] = [r[0] for r in cursor.fetchall()]

        queue = [source_family_id]
        found_depth = None

        for _depth in range(max_depth):
            if not queue or found_depth is not None:
                break

            # Collect all absolute_ids in current frontier
            frontier_aids: List[str] = []
            for fid in queue:
                frontier_aids.extend(family_abs_ids.get(fid, []))

            if not frontier_aids:
                break

            # Single query: get all relations touching frontier entities
            # Returns (entity1_absolute_id, entity2_absolute_id) + relation columns
            placeholders = ",".join("?" * len(frontier_aids))
            cursor.execute(f"""
                SELECT {self._RELATION_SELECT}
                FROM relations
                WHERE (entity1_absolute_id IN ({placeholders})
                    OR entity2_absolute_id IN ({placeholders}))
                  AND invalid_at IS NULL
            """, tuple(frontier_aids) * 2)

            frontier_rels = cursor.fetchall()

            # Resolve absolute_ids → family_ids for all relation endpoints
            all_rel_aids: Set[str] = set()
            for row in frontier_rels:
                all_rel_aids.add(row[2])  # entity1_absolute_id
                all_rel_aids.add(row[3])  # entity2_absolute_id

            # Batch resolve only the new absolute_ids we haven't seen
            unresolved = all_rel_aids - set(aid for aids in family_abs_ids.values() for aid in aids)
            aid_to_fid: Dict[str, str] = {}
            if unresolved:
                ph = ",".join("?" * len(unresolved))
                cursor.execute(f"SELECT id, family_id FROM entities WHERE id IN ({ph})", list(unresolved))
                for r in cursor.fetchall():
                    aid_to_fid[r[0]] = r[1]

            # Add already-known mappings
            for fid, aids in family_abs_ids.items():
                for aid in aids:
                    aid_to_fid[aid] = fid

            # Build adjacency for this frontier level + collect pair_relations
            next_queue_set: Set[str] = set()
            new_fids_to_load: Set[str] = set()

            for row in frontier_rels:
                rel = self._row_to_relation(row)
                fid1 = aid_to_fid.get(rel.entity1_absolute_id)
                fid2 = aid_to_fid.get(rel.entity2_absolute_id)
                if not fid1 or not fid2 or fid1 == fid2:
                    continue

                pair_key = tuple(sorted((fid1, fid2)))
                pair_relations.setdefault(pair_key, []).append(rel)

                # If one end is in the frontier and the other is new/at same depth
                for near_fid, far_fid in [(fid1, fid2), (fid2, fid1)]:
                    if near_fid in visited and visited[near_fid] == _depth:
                        if far_fid not in visited:
                            visited[far_fid] = _depth + 1
                            parents[far_fid] = [near_fid]
                            next_queue_set.add(far_fid)
                            new_fids_to_load.add(far_fid)
                            if far_fid == target_family_id:
                                found_depth = _depth + 1
                        elif visited.get(far_fid) == _depth + 1 and near_fid not in parents.get(far_fid, []):
                            parents.setdefault(far_fid, []).append(near_fid)

            # Batch load absolute_ids for newly discovered family_ids
            if new_fids_to_load:
                ph = ",".join("?" * len(new_fids_to_load))
                cursor.execute(f"SELECT family_id, id FROM entities WHERE family_id IN ({ph})", list(new_fids_to_load))
                for r in cursor.fetchall():
                    family_abs_ids.setdefault(r[0], []).append(r[1])

            queue = list(next_queue_set)

        # 未到达目标
        if target_family_id not in visited:
            result_empty["source_entity"] = source_entity
            result_empty["target_entity"] = target_entity
            return result_empty

        # 3. 回溯重构所有最短路径（DFS on parents）
        all_paths_eid: List[List[str]] = []

        def backtrack(node: str, path: List[str]):
            if len(all_paths_eid) >= max_paths * 10:
                return  # 防止爆炸
            if node == source_family_id:
                all_paths_eid.append(list(reversed(path)))
                return
            for parent in parents.get(node, []):
                path.append(parent)
                backtrack(parent, path)
                path.pop()

        backtrack(target_family_id, [target_family_id])
        all_paths_eid.sort()  # 稳定排序
        total_shortest_paths = len(all_paths_eid)
        all_paths_eid = all_paths_eid[:max_paths]

        # 4. 构建返回结果
        needed_abs_ids: Set[str] = set()
        for path_eids in all_paths_eid:
            for i in range(len(path_eids) - 1):
                pair_key = tuple(sorted((path_eids[i], path_eids[i + 1])))
                for rel in pair_relations.get(pair_key, []):
                    needed_abs_ids.add(rel.entity1_absolute_id)
                    needed_abs_ids.add(rel.entity2_absolute_id)

        # 批量查询 absolute_id → Entity
        abs_entity_map: Dict[str, Entity] = {}
        if needed_abs_ids:
            placeholders = ','.join('?' * len(needed_abs_ids))
            cursor.execute(f"""
                SELECT {self._ENTITY_SELECT}
                FROM entities WHERE id IN ({placeholders})
            """, list(needed_abs_ids))
            for row in cursor.fetchall():
                abs_entity_map[row[0]] = self._row_to_entity(row)

        # Build abs_to_eid only for needed entities
        abs_to_eid: Dict[str, str] = {}
        for fid in set(fid for path in all_paths_eid for fid in path):
            for aid in family_abs_ids.get(fid, []):
                abs_to_eid[aid] = fid

        paths_result = []
        for path_eids in all_paths_eid:
            path_entities = []
            path_relations = []
            seen_abs: Set[str] = set()

            for i in range(len(path_eids) - 1):
                pair_key = tuple(sorted((path_eids[i], path_eids[i + 1])))
                rels = pair_relations.get(pair_key, [])
                if rels:
                    rel = rels[0]
                    path_relations.append(rel)
                    e1_eid = abs_to_eid.get(rel.entity1_absolute_id)
                    first_abs = rel.entity1_absolute_id if e1_eid == path_eids[i] else rel.entity2_absolute_id
                    second_abs = rel.entity2_absolute_id if first_abs == rel.entity1_absolute_id else rel.entity1_absolute_id

                    for abs_id in [first_abs, second_abs]:
                        if abs_id not in seen_abs and abs_id in abs_entity_map:
                            path_entities.append(abs_entity_map[abs_id])
                            seen_abs.add(abs_id)

            paths_result.append({
                "entities": path_entities,
                "relations": path_relations,
                "length": len(path_eids) - 1,
            })

        return {
            "source_entity": source_entity,
            "target_entity": target_entity,
            "path_length": found_depth,
            "total_shortest_paths": total_shortest_paths,
            "paths": paths_result,
        }

    # ------------------------------------------------------------------
    # 时间旅行（Time Travel）功能
    # ------------------------------------------------------------------


    def get_snapshot(self, time_point: datetime, limit: Optional[int] = None) -> Dict[str, Any]:
        """获取指定时间点的实体/关系快照"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM entities
            WHERE (valid_at IS NULL OR valid_at <= ?)
              AND (invalid_at IS NULL OR invalid_at > ?)
            ORDER BY event_time DESC
            LIMIT ?
        """, (time_point.isoformat(), time_point.isoformat(), limit or 10000))

        entities = [self._row_to_entity(row) for row in cursor.fetchall()]

        cursor.execute(f"""
            SELECT {self._RELATION_SELECT}
            FROM relations
            WHERE (valid_at IS NULL OR valid_at <= ?)
              AND (invalid_at IS NULL OR invalid_at > ?)
            ORDER BY event_time DESC
            LIMIT ?
        """, (time_point.isoformat(), time_point.isoformat(), limit or 10000))

        relations = [self._row_to_relation(row) for row in cursor.fetchall()]

        return {"entities": entities, "relations": relations}

    def get_changes(self, since: datetime, until: Optional[datetime] = None) -> Dict[str, Any]:
        """获取时间范围内的变更记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        until_str = until.isoformat() if until else datetime.now(timezone.utc).isoformat()

        # 新增/修改的实体
        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM entities
            WHERE event_time >= ? AND event_time <= ?
            ORDER BY event_time DESC
        """, (since.isoformat(), until_str))

        entities = [self._row_to_entity(row) for row in cursor.fetchall()]

        # 新增/修改/失效的关系
        cursor.execute(f"""
            SELECT {self._RELATION_SELECT}
            FROM relations
            WHERE event_time >= ? AND event_time <= ?
            ORDER BY event_time DESC
        """, (since.isoformat(), until_str))

        relations = [self._row_to_relation(row) for row in cursor.fetchall()]

        return {"entities": entities, "relations": relations}

