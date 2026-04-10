"""
存储层：SQLite数据库 + Markdown文件存储
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
from ..utils import clean_markdown_code_blocks, wprint


class StorageManager:
    """存储管理器"""

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
        self._emb_cache_ttl: float = 5.0  # 秒

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

    def save_episode(self, cache: Episode, text: str = "", document_path: str = "", doc_hash: str = "") -> str:
        """保存记忆缓存到 docs/{timestamp}_{doc_hash}/ 目录

        Args:
            cache: 记忆缓存对象
            text: 当前处理的文本内容（可选，用于生成 doc_hash）
            document_path: 当前处理的文档完整路径（可选，用于断点续传定位）
            doc_hash: 文档 hash（可选，不传则从 text 自动计算）
        """
        if not doc_hash and text:
            doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        if not doc_hash:
            doc_hash = "unknown"

        # 目录命名：时间戳前缀 + hash 后缀，按文件名自然排序即时间排序
        ts_prefix = cache.event_time.strftime("%Y%m%d_%H%M%S") if cache.event_time else datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{ts_prefix}_{doc_hash}"
        doc_dir = self.docs_dir / dir_name
        self._ensure_dirs()
        doc_dir.mkdir(parents=True, exist_ok=True)

        # 保存原始文本（去重：已存在则跳过）
        if text:
            original_path = doc_dir / "original.txt"
            if not original_path.exists():
                original_path.write_text(text, encoding="utf-8")

        # 保存 LLM 摘要
        content = clean_markdown_code_blocks(cache.content)
        (doc_dir / "cache.md").write_text(content, encoding="utf-8")

        # 保存元数据
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

        # 更新缓存映射（用目录名而非纯 hash，以支持新命名）
        self._id_to_doc_hash[cache.absolute_id] = dir_name

        # 同步写入 SQLite episodes 表（Phase 1）
        self._save_episode_to_db(cache, doc_hash=dir_name)

        return cache.absolute_id
    
    def load_episode(self, cache_id: str) -> Optional[Episode]:
        """加载记忆缓存（优先从 docs/ 新结构读取，兼容旧结构）"""
        metadata = None
        md_content = None

        # 1. 尝试从 docs/ 新结构加载（通过缓存映射）
        doc_hash = self._id_to_doc_hash.get(cache_id)
        if doc_hash:
            doc_dir = self.docs_dir / doc_hash
            meta_path = doc_dir / "meta.json"
            cache_md_path = doc_dir / "cache.md"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                if cache_md_path.exists():
                    with open(cache_md_path, "r", encoding="utf-8") as f:
                        md_content = f.read()

        # 2. 回退到旧结构 memory_caches/json/
        if metadata is None:
            metadata_path = self.cache_json_dir / f"{cache_id}.json"
            if not metadata_path.exists():
                metadata_path = self.cache_dir / f"{cache_id}.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                filename = metadata.get("filename", f"{cache_id}.md")
                filepath = self.cache_md_dir / filename
                if not filepath.exists():
                    filepath = self.cache_dir / filename
                if filepath.exists():
                    with open(filepath, "r", encoding="utf-8") as f:
                        md_content = f.read()

        if metadata is None or md_content is None:
            return None

        # 清理 markdown 代码块标识符
        md_content = clean_markdown_code_blocks(md_content)

        return Episode(
            absolute_id=metadata.get("absolute_id") or metadata.get("id"),
            content=md_content,
            event_time=self._safe_parse_datetime(metadata.get("event_time"), datetime.now()),
            source_document=metadata.get("source_document") or metadata.get("doc_name", ""),
            activity_type=metadata.get("activity_type"),
        )

    def delete_episode(self, cache_id: str) -> int:
        """删除记忆缓存，返回删除的文件数。0 表示未找到。"""
        import shutil

        # 1. 尝试 docs/ 新结构
        doc_hash = self._id_to_doc_hash.get(cache_id)
        if doc_hash:
            doc_dir = self.docs_dir / doc_hash
            if doc_dir.is_dir():
                shutil.rmtree(doc_dir, ignore_errors=True)
                self._id_to_doc_hash.pop(cache_id, None)
                return 1

        # 2. 回退到旧结构
        for base_dir in (self.cache_json_dir, self.cache_dir):
            meta_path = base_dir / f"{cache_id}.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                meta_path.unlink(missing_ok=True)
                # 尝试删除对应的 .md 文件
                filename = metadata.get("filename", f"{cache_id}.md")
                for md_dir in (self.cache_md_dir, self.cache_dir):
                    filepath = md_dir / filename
                    if filepath.exists():
                        filepath.unlink()
                        return 1
        return 0
    
    def _iter_cache_meta_files(self) -> List[Path]:
        """迭代所有 cache 元数据文件（优先 docs/ 子目录，回退旧结构）"""
        files = []
        if self.docs_dir.is_dir():
            # 只匹配子目录中的 meta.json，排除扁平的 .txt 文件
            files = sorted([
                p for p in self.docs_dir.glob("*/meta.json")
                if p.parent.is_dir()
            ])
        if not files:
            files = list(self.cache_json_dir.glob("*.json"))
            if not files:
                files = list(self.cache_dir.glob("*.json"))
        return files

    def get_latest_episode(self, activity_type: Optional[str] = None) -> Optional[Episode]:
        """获取最新的记忆缓存"""
        cache_files = self._iter_cache_meta_files()
        if not cache_files:
            return None

        latest_cache = None
        latest_time = None
        latest_cache_id = None

        for cache_file in cache_files:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as exc:
                logger.debug("episode load failed: %s", exc)
                continue

            if activity_type and metadata.get("activity_type") != activity_type:
                continue

            cache_id = metadata.get("absolute_id") or metadata.get("id")
            if not cache_id:
                continue

            cache_time = self._safe_parse_datetime(metadata.get("event_time"), datetime.now())
            if latest_time is None or cache_time > latest_time:
                latest_time = cache_time
                latest_cache_id = cache_id

        if latest_cache_id:
            latest_cache = self.load_episode(latest_cache_id)

        return latest_cache

    def get_latest_episode_metadata(self, activity_type: Optional[str] = None) -> Optional[Dict]:
        """获取最新的记忆缓存元数据（用于断点续传）

        Returns:
            包含以下字段的字典：
            - absolute_id: 缓存ID
            - event_time: 事件发生时间
            - activity_type: 活动类型
            - text: 当前处理的文本内容
            - document_path: 当前处理的文档完整路径
            - doc_hash: 文档 hash
        """
        cache_files = self._iter_cache_meta_files()
        if not cache_files:
            return None

        latest_metadata = None
        latest_time = None

        for cache_file in cache_files:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as exc:
                logger.debug("episode load failed: %s", exc)
                continue

            if activity_type and metadata.get("activity_type") != activity_type:
                continue

            cache_time = self._safe_parse_datetime(metadata.get("event_time"), datetime.now())
            if latest_time is None or cache_time > latest_time:
                latest_time = cache_time
                latest_metadata = metadata

        return latest_metadata

    def search_episodes_by_bm25(self, query: str, limit: int = 20) -> List[Episode]:
        """搜索 Episode（优先使用 SQLite，回退到文件遍历）。

        当 episodes 表有数据时，先在 SQL 层用 LIKE 过滤匹配，
        再用 Python 评分排序，只加载 top-N 的完整 Episode 对象。
        """
        if not query:
            return []
        # 优先使用 SQLite episodes 表
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM episodes")
        if cursor.fetchone()[0] > 0:
            query_lower = query.lower()
            escaped = query_lower.replace('"', '""')
            # SQL LIKE 过滤候选，再在 Python 中精确评分
            cursor.execute(
                'SELECT id, content, source_document, event_time, activity_type '
                'FROM episodes WHERE LOWER(content) LIKE ? '
                'ORDER BY event_time DESC',
                (f'%{escaped}%',)
            )
            scored = []
            for row in cursor.fetchall():
                content = row[1] or ""
                score = content.lower().count(query_lower)
                scored.append((score, row))
            scored.sort(key=lambda x: x[0], reverse=True)
            # 只对 top-N 加载完整 Episode（可能从文件系统）
            results = []
            for score, row in scored[:limit]:
                cache = self.load_episode(row[0])
                if cache:
                    results.append(cache)
                else:
                    results.append(Episode(
                        absolute_id=row[0],
                        content=row[1] or "",
                        source_document=row[2] or "",
                        event_time=datetime.fromisoformat(row[3]) if row[3] else datetime.now(),
                        activity_type=row[4] or "",
                    ))
            return results
        # 回退：文件遍历（旧逻辑，episodes 表为空时）
        query_lower = query.lower()
        results = []
        for cache_file in self._iter_cache_meta_files():
            try:
                cache = self.load_episode(
                    cache_file.stem if cache_file.suffix == ".json" else
                    (cache_file.parent.name if cache_file.name == "meta.json" else cache_file.stem)
                )
            except Exception as exc:
                logger.debug("episode load failed: %s", exc)
                continue
            if cache is None:
                continue
            content_lower = (cache.content or "").lower()
            if query_lower in content_lower:
                score = content_lower.count(query_lower)
                results.append((score, cache))
        results.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in results[:limit]]

    def find_cache_by_doc_hash(self, doc_hash: str, document_path: str = "") -> Optional[Episode]:
        """通过 doc_hash 查找已存在的缓存（断点续传复用）。

        Args:
            doc_hash: 12位 MD5 hash
            document_path: 可选，用于精确匹配同一文档的缓存

        Returns:
            找到的 Episode，未找到返回 None
        """
        if not doc_hash or not self.docs_dir.is_dir():
            return None
        matches = list(self.docs_dir.glob(f"*_{doc_hash}/meta.json"))
        for meta_path in matches:
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as exc:
                logger.debug("episode load failed: %s", exc)
                continue
            if document_path and metadata.get("document_path") != document_path:
                continue
            cache_id = metadata.get("absolute_id")
            if cache_id:
                return self.load_episode(cache_id)
        return None

    def _get_cache_dir_by_doc_hash(self, doc_hash: str, document_path: str = "") -> Optional[Path]:
        """根据 doc_hash 找到缓存目录路径（不加载缓存内容）。"""
        if not doc_hash or not self.docs_dir.is_dir():
            return None
        matches = list(self.docs_dir.glob(f"*_{doc_hash}/meta.json"))
        for meta_path in matches:
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as exc:
                logger.debug("episode load failed: %s", exc)
                continue
            if document_path and metadata.get("document_path") != document_path:
                continue
            return meta_path.parent
        return None

    def save_extraction_result(self, doc_hash: str, entities: list, relations: list,
                               document_path: str = "") -> bool:
        """保存步骤2-5的抽取结果到缓存目录（断点续传复用）。

        Returns:
            True 保存成功，False 失败
        """
        cache_dir = self._get_cache_dir_by_doc_hash(doc_hash, document_path)
        if not cache_dir:
            return False
        try:
            data = {"entities": entities, "relations": relations}
            (cache_dir / "extraction.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return True
        except Exception as exc:
            logger.debug("extraction save failed: %s", exc)
            return False

    def load_extraction_result(self, doc_hash: str,
                               document_path: str = "") -> Optional[tuple]:
        """加载步骤2-5的抽取结果。

        Returns:
            (entities, relations) 元组，未找到返回 None
        """
        cache_dir = self._get_cache_dir_by_doc_hash(doc_hash, document_path)
        if not cache_dir:
            return None
        try:
            raw = (cache_dir / "extraction.json").read_text(encoding="utf-8")
            data = json.loads(raw)
            entities = data.get("entities", [])
            relations = data.get("relations", [])
            if isinstance(entities, list) and isinstance(relations, list):
                return (entities, relations)
        except Exception as exc:
            logger.debug("extraction result save failed: %s", exc)
        return None

    # ========== Entity 操作 ==========
    
    def _compute_entity_embedding(self, entity: Entity) -> Optional[bytes]:
        """计算实体的embedding向量并转换为BLOB"""
        if not self.embedding_client or not self.embedding_client.is_available():
            return None
        
        # 构建文本：name + content[:snippet_length]
        text = f"{entity.name} {entity.content[:self.entity_content_snippet_length]}"
        embedding = self.embedding_client.encode(text)
        
        if embedding is None or len(embedding) == 0:
            return None
        
        # 转换为numpy数组并序列化为BLOB
        embedding_array = np.array(embedding[0] if isinstance(embedding, list) else embedding, dtype=np.float32)
        return embedding_array.tobytes()
    
    def save_entity(self, entity: Entity):
        """保存实体（包含预计算的embedding向量）"""
        self._invalidate_emb_cache()
        # 计算embedding（无需锁，纯计算）
        embedding_blob = self._compute_entity_embedding(entity)
        entity.embedding = embedding_blob

        with self._write_lock:
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO entities (id, family_id, name, content, event_time, processed_time, episode_id, source_document, embedding, valid_at, summary, attributes, confidence, content_format)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.absolute_id,
                    entity.family_id,
                    entity.name,
                    entity.content,
                    entity.event_time.isoformat(),
                    entity.processed_time.isoformat(),
                    entity.episode_id,
                    entity.source_document,
                    embedding_blob,
                    (entity.valid_at or entity.event_time).isoformat(),
                    getattr(entity, 'summary', None),
                    getattr(entity, 'attributes', None),
                    getattr(entity, 'confidence', None),
                    getattr(entity, 'content_format', 'plain'),
                ))
                # 同步写入 FTS 表（使用整数 rowid，非文本 id）
                try:
                    cursor.execute("""
                        INSERT INTO entity_fts(rowid, name, content, family_id)
                        VALUES (?, ?, ?, ?)
                    """, (cursor.lastrowid, entity.name, entity.content, entity.family_id))
                except Exception as exc:
                    logger.warning("FTS entity write failed: %s", exc)
                # 设置旧版本 invalid_at
                try:
                    cursor.execute("""
                        UPDATE entities SET invalid_at = ?
                        WHERE family_id = ? AND id != ? AND invalid_at IS NULL
                    """, (entity.event_time.isoformat(), entity.family_id, entity.absolute_id))
                except Exception as exc:
                    logger.warning("FTS invalid_at update failed: %s", exc)
                # Phase 2: dual-write to concepts
                self._write_concept_from_entity(entity, cursor)
                # 单次 commit 包含所有写操作
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def bulk_save_entities(self, entities: List[Entity]):
        """批量保存实体，使用批量 embedding 与单事务写入。"""
        if not entities:
            return
        self._invalidate_emb_cache()

        # 批量计算 embedding（无需锁）
        embeddings = None
        if self.embedding_client and self.embedding_client.is_available():
            texts = [
                f"{entity.name} {entity.content[:self.entity_content_snippet_length]}"
                for entity in entities
            ]
            embeddings = self.embedding_client.encode(texts)

        rows = []
        for idx, entity in enumerate(entities):
            embedding_blob = None
            if embeddings is not None:
                try:
                    embedding_blob = np.array(embeddings[idx], dtype=np.float32).tobytes()
                except Exception as exc:
                    logger.debug("embedding encode failed: %s", exc)
                    embedding_blob = None
            entity.embedding = embedding_blob
            rows.append((
                entity.absolute_id,
                entity.family_id,
                entity.name,
                entity.content,
                entity.event_time.isoformat(),
                entity.processed_time.isoformat(),
                entity.episode_id,
                entity.source_document,
                embedding_blob,
                (entity.valid_at or entity.event_time).isoformat(),
                getattr(entity, 'summary', None),
                getattr(entity, 'attributes', None),
                getattr(entity, 'confidence', None),
                getattr(entity, 'content_format', 'plain'),
            ))

        with self._write_lock:
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR IGNORE INTO entities (id, family_id, name, content, event_time, processed_time, episode_id, source_document, embedding, valid_at, summary, attributes, confidence, content_format)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
                # 同步写入 FTS 表（使用整数 rowid）
                try:
                    # 查询刚插入的实体的整数 rowid
                    aids = [e.absolute_id for e in entities]
                    placeholders = ",".join("?" * len(aids))
                    cursor.execute(f"SELECT rowid, id FROM entities WHERE id IN ({placeholders})", aids)
                    id_to_rowid = {r[1]: r[0] for r in cursor.fetchall()}
                    fts_rows = [(id_to_rowid.get(e.absolute_id), e.name, e.content, e.family_id) for e in entities if id_to_rowid.get(e.absolute_id)]
                    if fts_rows:
                        cursor.executemany("""
                            INSERT OR REPLACE INTO entity_fts(rowid, name, content, family_id)
                            VALUES (?, ?, ?, ?)
                        """, fts_rows)
                except Exception as exc:
                    logger.debug("FTS bulk entity write failed: %s", exc)
                # 设置旧版本 invalid_at + Phase 2 dual-write
                for entity in entities:
                    try:
                        cursor.execute("""
                            UPDATE entities SET invalid_at = ?
                            WHERE family_id = ? AND id != ? AND invalid_at IS NULL
                        """, (entity.event_time.isoformat(), entity.family_id, entity.absolute_id))
                    except Exception as exc:
                        logger.debug("bulk invalid_at update failed: %s", exc)
                    self._write_concept_from_entity(entity, cursor)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def get_entity_by_family_id(self, family_id: str) -> Optional[Entity]:
        """根据family_id获取最新版本的实体"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return None
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM entities
            WHERE family_id = ?
            ORDER BY processed_time DESC
            LIMIT 1
        """, (family_id,))

        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_entity(row)

    def get_entities_by_family_ids(self, family_ids: List[str]) -> Dict[str, Entity]:
        """批量根据 family_id 获取最新版本实体，返回 {family_id: Entity}。"""
        if not family_ids:
            return {}
        # 先 resolve 所有 family_id
        resolved_map = self.resolve_family_ids(list(family_ids))
        valid_fids = set(resolved_map.keys()) | set(resolved_map.values())
        if not valid_fids:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(valid_fids))
        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM entities
            WHERE family_id IN ({placeholders})
            ORDER BY processed_time DESC
        """, list(valid_fids))
        seen = set()
        result: Dict[str, Entity] = {}
        for row in cursor.fetchall():
            fid = row[1]
            if fid in seen:
                continue
            seen.add(fid)
            entity = self._row_to_entity(row)
            result[fid] = entity
            # 如果原始 family_id != resolved，也映射原始 ID
            for orig_fid, resolved_fid in resolved_map.items():
                if resolved_fid == fid:
                    result[orig_fid] = entity
        return result

    def get_entity_by_absolute_id(self, absolute_id: str) -> Optional[Entity]:
        """根据绝对ID获取实体"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM entities
            WHERE id = ?
        """, (absolute_id,))

        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_entity(row)

    def get_entities_by_absolute_ids(self, absolute_ids: List[str]) -> List[Entity]:
        """根据绝对ID列表批量获取实体。"""
        if not absolute_ids:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(absolute_ids))
        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM entities
            WHERE id IN ({placeholders})
        """, tuple(absolute_ids))
        rows = cursor.fetchall()
        return [self._row_to_entity(row) for row in rows]

    def get_relation_by_absolute_id(self, relation_absolute_id: str) -> Optional[Relation]:
        """根据关系行的主键 id（绝对ID）获取单条关系"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {self._RELATION_SELECT} FROM relations WHERE id = ?",
            (relation_absolute_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_relation(row)
    
    def get_relation_by_family_id(self, family_id: str) -> Optional[Relation]:
        """根据family_id获取最新版本的关系"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {self._RELATION_SELECT} FROM relations WHERE family_id = ? ORDER BY processed_time DESC LIMIT 1",
            (family_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_relation(row)

    def get_entity_names_by_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, str]:
        """批量根据 absolute_id 查询实体名称"""
        if not absolute_ids:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        unique_ids = list(set(absolute_ids))
        placeholders = ','.join('?' * len(unique_ids))
        cursor.execute(f"SELECT id, name FROM entities WHERE id IN ({placeholders})", unique_ids)
        return {row[0]: row[1] or '' for row in cursor.fetchall()}

    def get_entity_version_at_time(self, family_id: str, time_point: datetime) -> Optional[Entity]:
        """获取实体在指定时间点的版本（该时间点之前或等于该时间点的最新版本）"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return None
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM entities
            WHERE family_id = ? AND event_time <= ?
            ORDER BY processed_time DESC
            LIMIT 1
        """, (family_id, time_point.isoformat()))

        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_entity(row)
    
    def get_entity_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        """获取实体embedding向量的前N个值"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT embedding
            FROM entities
            WHERE id = ?
        """, (absolute_id,))
        
        row = cursor.fetchone()
        
        if row is None or row[0] is None:
            return None
        
        try:
            embedding_array = np.frombuffer(row[0], dtype=np.float32)
            return embedding_array[:num_values].tolist()
        except Exception as exc:
            logger.debug("entity embedding preview failed: %s", exc)
            return None

    def get_relation_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        """获取关系embedding向量的前N个值"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT embedding
            FROM relations
            WHERE id = ?
        """, (absolute_id,))
        
        row = cursor.fetchone()
        
        if row is None or row[0] is None:
            return None
        
        try:
            embedding_array = np.frombuffer(row[0], dtype=np.float32)
            return embedding_array[:num_values].tolist()
        except Exception as exc:
            logger.debug("relation embedding preview failed: %s", exc)
            return None

    def get_entity_versions(self, family_id: str) -> List[Entity]:
        """获取实体的所有版本"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute(
            f"SELECT {self._ENTITY_SELECT} FROM entities WHERE family_id = ? ORDER BY processed_time DESC",
            (family_id,),
        )

        rows = cursor.fetchall()

        return [self._row_to_entity(row) for row in rows]
    
    def _invalidate_emb_cache(self):
        """清除 embedding 缓存（在实体/关系写入时调用）。"""
        self._entity_emb_cache = None
        self._entity_emb_cache_ts = 0.0
        self._relation_emb_cache = None
        self._relation_emb_cache_ts = 0.0

    def _get_entities_with_embeddings(self) -> List[tuple]:
        """
        获取所有实体的最新版本及其embedding（带短 TTL 缓存）。

        Returns:
            List of (Entity, embedding_array) tuples, embedding_array为None表示没有embedding
        """
        now = time.time()
        if self._entity_emb_cache is not None and (now - self._entity_emb_cache_ts) < self._emb_cache_ttl:
            return self._entity_emb_cache

        conn = self._get_conn()
        cursor = conn.cursor()

        # 使用窗口函数获取每个 family_id 的最新版本（O(N) 替代 O(N^2) 子查询）
        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY family_id ORDER BY processed_time DESC) AS rn
                FROM entities
            )
            WHERE rn = 1
        """)

        results = []
        for row in cursor.fetchall():
            embedding_array = None
            if len(row) > 8 and row[8] is not None:
                try:
                    embedding_array = np.frombuffer(row[8], dtype=np.float32)
                except (ValueError, TypeError):
                    embedding_array = None
            entity = self._row_to_entity(row)
            results.append((entity, embedding_array))

        self._entity_emb_cache = results
        self._entity_emb_cache_ts = time.time()
        return results

    def get_latest_entities_projection(self, content_snippet_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取最新实体投影，供窗口级批量候选生成使用。"""
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
                "content_snippet": entity.content[:snippet_length],
                "version_count": version_counts.get(entity.family_id, 1),
                "embedding_array": embedding_array,
            })
        return results

    def search_entities_by_bm25(self, query: str, limit: int = 20) -> List[Entity]:
        """BM25 全文搜索实体。"""
        if not query:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        # FTS5 BM25 搜索，按 bm25 排序
        ent_cols = ", e.".join(self._ENTITY_SELECT.split(", "))
        cursor.execute(f"""
            SELECT e.{ent_cols},
                   fts.rank AS bm25_score
            FROM entity_fts AS fts
            JOIN entities AS e ON e.rowid = fts.rowid
            WHERE entity_fts MATCH ?
            ORDER BY fts.rank
            LIMIT ?
        """, (query, limit))

        rows = cursor.fetchall()
        raw_fids = [row[1] for row in rows]
        resolved_map = self.resolve_family_ids(raw_fids) if raw_fids else {}

        entities = []
        for row in rows:
            ent = self._row_to_entity(row)
            if resolved_map.get(row[1], row[1]) != row[1]:
                ent = Entity(
                    absolute_id=ent.absolute_id,
                    family_id=resolved_map.get(row[1], row[1]),
                    name=ent.name, content=ent.content,
                    event_time=ent.event_time, processed_time=ent.processed_time,
                    episode_id=ent.episode_id, source_document=ent.source_document,
                    embedding=ent.embedding, summary=ent.summary,
                    attributes=ent.attributes, confidence=ent.confidence,
                    valid_at=ent.valid_at, invalid_at=ent.invalid_at,
                )
            entities.append(ent)
        return entities

    def search_relations_by_bm25(self, query: str, limit: int = 20) -> List[Relation]:
        """BM25 全文搜索关系。"""
        if not query:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        rel_cols = ", r.".join(self._RELATION_SELECT.split(", "))
        cursor.execute(f"""
            SELECT r.{rel_cols},
                   fts.rank AS bm25_score
            FROM relation_fts AS fts
            JOIN relations AS r ON r.rowid = fts.rowid
            WHERE relation_fts MATCH ?
            ORDER BY fts.rank
            LIMIT ?
        """, (query, limit))

        relations = []
        for row in cursor.fetchall():
            relations.append(self._row_to_relation(row))
        return relations

    def search_entities_by_similarity(self, query_name: str, query_content: Optional[str] = None,
                                     threshold: float = 0.7, max_results: int = 10,
                                     content_snippet_length: int = 50,
                                     text_mode: Literal["name_only", "content_only", "name_and_content"] = "name_and_content",
                                     similarity_method: Literal["embedding", "text", "jaccard", "bleu"] = "embedding",
                                     _preloaded_entities: Optional[List[tuple]] = None) -> List[Entity]:
        """
        根据名称相似度搜索实体
        
        支持多种检索模式：
        - text_mode: 使用哪些字段进行检索（name_only/content_only/name_and_content）
        - similarity_method: 使用哪种相似度计算方法（embedding/text/jaccard/bleu）
        
        Args:
            query_name: 查询的实体名称
            query_content: 查询的实体内容（可选）
            threshold: 相似度阈值
            max_results: 返回的最大相似实体数量（默认10）
            content_snippet_length: 用于相似度搜索的content截取长度（默认50字符）
            text_mode: 文本模式
                - "name_only": 只使用name进行检索
                - "content_only": 只使用content进行检索
                - "name_and_content": 使用name + content进行检索
            similarity_method: 相似度计算方法
                - "embedding": 使用embedding向量相似度（优先使用已存储的embedding）
                - "text": 使用文本序列相似度（SequenceMatcher）
                - "jaccard": 使用Jaccard相似度
                - "bleu": 使用BLEU相似度
        """
        # 获取所有实体及其embedding（优先使用预加载数据，避免N+1）
        entities_with_embeddings = _preloaded_entities if _preloaded_entities is not None else self._get_entities_with_embeddings()
        
        if not entities_with_embeddings:
            return []
        
        all_entities = [e for e, _ in entities_with_embeddings]
        
        # 根据text_mode构建查询文本
        if text_mode == "name_only":
            query_text = query_name
            use_content = False
        elif text_mode == "content_only":
            if not query_content:
                return []  # 如果没有content，无法检索
            query_text = query_content[:content_snippet_length]
            use_content = True
        else:  # name_and_content
            if query_content:
                query_text = f"{query_name} {query_content[:content_snippet_length]}"
            else:
                query_text = query_name
            use_content = query_content is not None
        
        # 根据similarity_method选择检索方式
        if similarity_method == "embedding" and self.embedding_client and self.embedding_client.is_available():
            return self._search_with_embedding(
                query_text, entities_with_embeddings, threshold, 
                use_content, max_results, content_snippet_length, text_mode
            )
        else:
            # 使用文本相似度（text/jaccard/bleu）
            return self._search_with_text_similarity(
                query_text, all_entities, threshold, 
                use_content, max_results, content_snippet_length, 
                text_mode, similarity_method
            )
    
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
        """计算Jaccard相似度（基于bigram集合，比字符集更精确）"""
        s1 = (text1 or "").lower().strip()
        s2 = (text2 or "").lower().strip()
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0
        set1 = {s1[i:i+2] for i in range(len(s1) - 1)}
        set2 = {s2[i:i+2] for i in range(len(s2) - 1)}
        if not set1 or not set2:
            cs1, cs2 = set(s1), set(s2)
            union = len(cs1 | cs2)
            return len(cs1 & cs2) / union if union else 0.0
        union = len(set1 | set2)
        return len(set1 & set2) / union
    
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
    
    def _compute_relation_embedding(self, relation: Relation) -> Optional[bytes]:
        """计算关系的embedding向量并转换为BLOB"""
        if not self.embedding_client or not self.embedding_client.is_available():
            return None
        
        # 构建文本：content[:snippet_length]；snippet<=0 时用全文（与关系抽取「仅列实体名」配置一致，避免空串 embedding）
        n = self.relation_content_snippet_length
        text = relation.content if n is None or n <= 0 else relation.content[:n]
        embedding = self.embedding_client.encode(text)
        
        if embedding is None or len(embedding) == 0:
            return None
        
        # 转换为numpy数组并序列化为BLOB
        embedding_array = np.array(embedding[0] if isinstance(embedding, list) else embedding, dtype=np.float32)
        return embedding_array.tobytes()
    
    def save_relation(self, relation: Relation):
        """保存关系（包含预计算的embedding向量）"""
        self._invalidate_emb_cache()
        # 计算embedding（无需锁）
        embedding_blob = self._compute_relation_embedding(relation)
        relation.embedding = embedding_blob

        with self._write_lock:
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO relations (id, family_id, entity1_absolute_id, entity2_absolute_id, content, event_time, processed_time, episode_id, source_document, embedding, valid_at, summary, attributes, confidence, provenance, content_format)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    relation.absolute_id,
                    relation.family_id,
                    relation.entity1_absolute_id,
                    relation.entity2_absolute_id,
                    relation.content,
                    relation.event_time.isoformat(),
                    relation.processed_time.isoformat(),
                    relation.episode_id,
                    relation.source_document,
                    embedding_blob,
                    (relation.valid_at or relation.event_time).isoformat(),
                    getattr(relation, 'summary', None),
                    getattr(relation, 'attributes', None),
                    getattr(relation, 'confidence', None),
                    getattr(relation, 'provenance', None),
                    getattr(relation, 'content_format', 'plain'),
                ))
                # 同步写入 FTS 表（使用整数 rowid）
                try:
                    cursor.execute("""
                        INSERT INTO relation_fts(rowid, content, family_id)
                        VALUES (?, ?, ?)
                    """, (cursor.lastrowid, relation.content, relation.family_id))
                except Exception as exc:
                    logger.warning("FTS relation write failed: %s", exc)
                # 设置旧版本 invalid_at
                try:
                    cursor.execute("""
                        UPDATE relations SET invalid_at = ?
                        WHERE family_id = ? AND id != ? AND invalid_at IS NULL
                    """, (relation.event_time.isoformat(), relation.family_id, relation.absolute_id))
                except Exception as exc:
                    logger.warning("FTS relation invalid_at update failed: %s", exc)
                # Phase 2: dual-write to concepts
                self._write_concept_from_relation(relation, cursor)
                # 单次 commit 包含所有写操作
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def bulk_save_relations(self, relations: List[Relation]):
        """批量保存关系，使用批量 embedding 与单事务写入。"""
        if not relations:
            return

        # 批量计算 embedding（无需锁）
        embeddings = None
        if self.embedding_client and self.embedding_client.is_available():
            _n = self.relation_content_snippet_length
            texts = [
                relation.content if _n is None or _n <= 0 else relation.content[:_n]
                for relation in relations
            ]
            embeddings = self.embedding_client.encode(texts)

        rows = []
        for idx, relation in enumerate(relations):
            embedding_blob = None
            if embeddings is not None:
                try:
                    embedding_blob = np.array(embeddings[idx], dtype=np.float32).tobytes()
                except Exception as exc:
                    logger.debug("embedding encode failed: %s", exc)
                    embedding_blob = None
            relation.embedding = embedding_blob
            rows.append((
                relation.absolute_id,
                relation.family_id,
                relation.entity1_absolute_id,
                relation.entity2_absolute_id,
                relation.content,
                relation.event_time.isoformat(),
                relation.processed_time.isoformat(),
                relation.episode_id,
                relation.source_document,
                embedding_blob,
                (relation.valid_at or relation.event_time).isoformat(),
                getattr(relation, 'summary', None),
                getattr(relation, 'attributes', None),
                getattr(relation, 'confidence', None),
                getattr(relation, 'provenance', None),
                getattr(relation, 'content_format', 'plain'),
            ))

        with self._write_lock:
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO relations (id, family_id, entity1_absolute_id, entity2_absolute_id, content, event_time, processed_time, episode_id, source_document, embedding, valid_at, summary, attributes, confidence, provenance, content_format)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
                # 同步写入 FTS 表（使用整数 rowid）
                try:
                    aids = [r.absolute_id for r in relations]
                    placeholders = ",".join("?" * len(aids))
                    cursor.execute(f"SELECT rowid, id FROM relations WHERE id IN ({placeholders})", aids)
                    id_to_rowid = {r[1]: r[0] for r in cursor.fetchall()}
                    fts_rows = [(id_to_rowid.get(r.absolute_id), r.content, r.family_id) for r in relations if id_to_rowid.get(r.absolute_id)]
                    if fts_rows:
                        cursor.executemany("""
                            INSERT OR REPLACE INTO relation_fts(rowid, content, family_id)
                            VALUES (?, ?, ?)
                        """, fts_rows)
                except Exception as exc:
                    logger.debug("FTS bulk relation write failed: %s", exc)
                # 设置旧版本 invalid_at + Phase 2 dual-write
                for relation in relations:
                    try:
                        cursor.execute("""
                            UPDATE relations SET invalid_at = ?
                            WHERE family_id = ? AND id != ? AND invalid_at IS NULL
                        """, (relation.event_time.isoformat(), relation.family_id, relation.absolute_id))
                    except Exception as exc:
                        logger.debug("bulk relation invalid_at update failed: %s", exc)
                    self._write_concept_from_relation(relation, cursor)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    
    def get_relations_by_entities(self, from_family_id: str, to_family_id: str) -> List[Relation]:
        """根据两个实体ID获取所有关系（通过family_id查找，内部转换为绝对ID查询）

        每个 family_id 只返回最新版本（与 get_entity_relations 保持一致的去重逻辑）。
        """
        from_family_id = self.resolve_family_id(from_family_id)
        to_family_id = self.resolve_family_id(to_family_id)
        if not from_family_id or not to_family_id:
            return []
        # 先通过family_id获取最新版本的绝对ID
        from_entity = self.get_entity_by_family_id(from_family_id)
        to_entity = self.get_entity_by_family_id(to_family_id)

        if not from_entity or not to_entity:
            return []

        # 获取所有具有相同family_id的实体的绝对ID
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id FROM entities WHERE family_id = ?
        """, (from_family_id,))
        from_absolute_ids = [row[0] for row in cursor.fetchall()]

        cursor.execute("""
            SELECT id FROM entities WHERE family_id = ?
        """, (to_family_id,))
        to_absolute_ids = [row[0] for row in cursor.fetchall()]

        if not from_absolute_ids or not to_absolute_ids:
            return []

        # 查询关系（无向关系，考虑两个方向）
        # 每个 family_id 只返回最新版本（INNER JOIN MAX(processed_time) 去重）
        placeholders_from = ','.join(['?'] * len(from_absolute_ids))
        placeholders_to = ','.join(['?'] * len(to_absolute_ids))

        r1_cols = ", r1.".join(self._RELATION_SELECT.split(", "))
        cursor.execute(f"""
            SELECT r1.{r1_cols}
            FROM relations r1
            INNER JOIN (
                SELECT family_id, MAX(processed_time) as max_time
                FROM relations
                WHERE (entity1_absolute_id IN ({placeholders_from}) AND entity2_absolute_id IN ({placeholders_to}))
                   OR (entity1_absolute_id IN ({placeholders_to}) AND entity2_absolute_id IN ({placeholders_from}))
                GROUP BY family_id
            ) r2 ON r1.family_id = r2.family_id
                AND r1.processed_time = r2.max_time
                AND ((r1.entity1_absolute_id IN ({placeholders_from}) AND r1.entity2_absolute_id IN ({placeholders_to}))
                  OR (r1.entity1_absolute_id IN ({placeholders_to}) AND r1.entity2_absolute_id IN ({placeholders_from})))
            ORDER BY r1.processed_time DESC
        """, (from_absolute_ids + to_absolute_ids + to_absolute_ids + from_absolute_ids)
              + (from_absolute_ids + to_absolute_ids + to_absolute_ids + from_absolute_ids))

        rows = cursor.fetchall()

        return [self._row_to_relation(row) for row in rows]

    def get_relations_by_entity_pairs(self, entity_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], List[Relation]]:
        """批量获取多个实体对的关系，按无向 pair 返回。

        优化：批量解析所有 family_id → absolute_id，单次关系查询，按 pair 分组。
        """
        if not entity_pairs:
            return {}

        # 去重 pair keys
        unique_pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for e1, e2 in entity_pairs:
            pair_key = tuple(sorted((e1, e2)))
            if pair_key not in unique_pairs:
                unique_pairs[pair_key] = (e1, e2)

        # 批量解析所有 family_id → canonical
        all_raw_fids = set()
        for e1, e2 in unique_pairs.values():
            all_raw_fids.add(e1)
            all_raw_fids.add(e2)

        canonical_map: Dict[str, Optional[str]] = self.resolve_family_ids(all_raw_fids)

        # 过滤有效的 canonical pairs
        valid_canonical_fids = set()
        canonical_pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for pair_key, (e1, e2) in unique_pairs.items():
            c1 = canonical_map.get(e1)
            c2 = canonical_map.get(e2)
            if c1 and c2:
                canonical_pair = tuple(sorted((c1, c2)))
                if canonical_pair not in canonical_pairs:
                    canonical_pairs[canonical_pair] = pair_key
                    valid_canonical_fids.add(c1)
                    valid_canonical_fids.add(c2)

        if not valid_canonical_fids:
            return {pk: [] for pk in unique_pairs}

        # 批量获取所有 canonical family_id 的 absolute_ids
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(valid_canonical_fids))
        cursor.execute(
            f"SELECT family_id, id FROM entities WHERE family_id IN ({placeholders})",
            tuple(valid_canonical_fids),
        )
        fid_to_aids: Dict[str, List[str]] = {}
        all_aids: List[str] = []
        for row in cursor.fetchall():
            fid_to_aids.setdefault(row[0], []).append(row[1])
            all_aids.append(row[1])

        if not all_aids:
            return {pk: [] for pk in unique_pairs}

        # 单次查询所有相关关系
        aid_placeholders = ",".join("?" * len(all_aids))
        cursor.execute(f"""
            SELECT r.id, r.family_id, r.entity1_absolute_id, r.entity2_absolute_id,
                   r.content, r.event_time, r.processed_time, r.episode_id, r.source_document,
                   r.summary, r.attributes, r.confidence, r.valid_at, r.invalid_at
            FROM relations r
            WHERE (r.entity1_absolute_id IN ({aid_placeholders})
                OR r.entity2_absolute_id IN ({aid_placeholders}))
              AND r.invalid_at IS NULL
        """, tuple(all_aids) * 2)

        all_rels = []
        for row in cursor.fetchall():
            all_rels.append(self._row_to_relation(row))

        # 按 canonical pair 分组 — O(R) 用 reverse lookup dict 替代 O(R*F) 嵌套循环
        aid_to_fid: Dict[str, str] = {}
        for fid, aids in fid_to_aids.items():
            for aid in aids:
                aid_to_fid[aid] = fid

        canonical_rels: Dict[Tuple[str, str], List[Relation]] = {cp: [] for cp in canonical_pairs}
        for rel in all_rels:
            e1_fid = aid_to_fid.get(rel.entity1_absolute_id)
            e2_fid = aid_to_fid.get(rel.entity2_absolute_id)
            if e1_fid and e2_fid:
                pair = tuple(sorted((e1_fid, e2_fid)))
                if pair in canonical_rels:
                    canonical_rels[pair].append(rel)

        # 映射回原始 pair keys
        results: Dict[Tuple[str, str], List[Relation]] = {}
        for canonical_pair, original_key in canonical_pairs.items():
            results[original_key] = canonical_rels.get(canonical_pair, [])

        return results

    def get_relation_versions(self, family_id: str) -> List[Relation]:
        """获取关系的所有版本"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT {self._RELATION_SELECT} FROM relations WHERE family_id = ? ORDER BY processed_time DESC",
            (family_id,),
        )

        rows = cursor.fetchall()

        return [self._row_to_relation(row) for row in rows]
    
    
    def get_all_entities(self, limit: Optional[int] = None, offset: Optional[int] = None, exclude_embedding: bool = False) -> List[Entity]:
        """获取所有实体的最新版本

        Args:
            limit: 限制返回的实体数量（按时间倒序），None表示不限制
            offset: 跳过前N条记录，None表示不跳过
            exclude_embedding: 是否排除 embedding 字段（前端展示等不需要 embedding 的场景应设为 True）
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        e1_cols = ", e1.".join(self._ENTITY_SELECT.split(", "))
        if exclude_embedding:
            e1_cols = e1_cols.replace("e1.embedding", "NULL AS e1__embedding")
        query = f"""
            SELECT e1.{e1_cols}
            FROM entities e1
            INNER JOIN (
                SELECT family_id, MAX(processed_time) as max_time
                FROM entities
                GROUP BY family_id
            ) e2 ON e1.family_id = e2.family_id AND e1.processed_time = e2.max_time
            ORDER BY e1.processed_time DESC
        """

        if limit is not None and offset is not None and offset > 0:
            query += f" LIMIT {int(limit)} OFFSET {int(offset)}"
        elif limit is not None:
            query += f" LIMIT {int(limit)}"

        cursor.execute(query)

        rows = cursor.fetchall()

        return [self._row_to_entity(row) for row in rows]

    def count_unique_entities(self) -> int:
        """轻量统计：返回不重复的 family_id 数量（不加载任何实体数据）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT family_id) FROM entities")
        return cursor.fetchone()[0]

    def count_unique_relations(self) -> int:
        """轻量统计：返回不重复的 family_id 数量（不加载任何关系数据）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT family_id) FROM relations")
        return cursor.fetchone()[0]

    def get_all_entities_before_time(self, time_point: datetime, limit: Optional[int] = None,
                                     exclude_embedding: bool = False) -> List[Entity]:
        """获取指定时间点之前或等于该时间点的所有实体的最新版本

        Args:
            time_point: 时间点
            limit: 限制返回的实体数量（按时间倒序），None表示不限制
            exclude_embedding: 是否排除 embedding 字段
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        e1_cols = ", e1.".join(self._ENTITY_SELECT.split(", "))
        if exclude_embedding:
            e1_cols = e1_cols.replace("e1.embedding", "NULL AS e1__embedding")
        query = f"""
            SELECT e1.{e1_cols}
            FROM entities e1
            INNER JOIN (
                SELECT family_id, MAX(processed_time) as max_time
                FROM entities
                WHERE event_time <= ?
                GROUP BY family_id
            ) e2 ON e1.family_id = e2.family_id AND e1.processed_time = e2.max_time
            ORDER BY e1.processed_time DESC
        """

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        cursor.execute(query, (time_point.isoformat(),))

        rows = cursor.fetchall()

        return [self._row_to_entity(row) for row in rows]
    
    def get_entity_relations(self, entity_absolute_id: str, limit: Optional[int] = None, time_point: Optional[datetime] = None) -> List[Relation]:
        """获取与指定实体相关的所有关系（作为起点或终点）
        
        Args:
            entity_absolute_id: 实体的绝对ID
            limit: 限制返回的关系数量（按时间倒序），None表示不限制
            time_point: 时间点（可选），如果提供，只返回该时间点之前或等于该时间点的关系，且每个family_id只返回最新版本
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        r1_cols = ", r1.".join(self._RELATION_SELECT.split(", "))
        if time_point:
            # 获取每个family_id在该时间点之前或等于该时间点的最新版本
            query = f"""
                SELECT r1.{r1_cols}
                FROM relations r1
                INNER JOIN (
                    SELECT family_id, MAX(processed_time) as max_time
                    FROM relations
                    WHERE (entity1_absolute_id = ? OR entity2_absolute_id = ?)
                    AND event_time <= ?
                    GROUP BY family_id
                ) r2 ON r1.family_id = r2.family_id
                    AND r1.processed_time = r2.max_time
                    AND (r1.entity1_absolute_id = ? OR r1.entity2_absolute_id = ?)
                ORDER BY r1.processed_time DESC
            """
            params = (entity_absolute_id, entity_absolute_id, time_point.isoformat(), entity_absolute_id, entity_absolute_id)
        else:
            # 获取每个family_id的最新版本
            query = f"""
                SELECT r1.{r1_cols}
                FROM relations r1
                INNER JOIN (
                    SELECT family_id, MAX(processed_time) as max_time
                    FROM relations
                    WHERE entity1_absolute_id = ? OR entity2_absolute_id = ?
                    GROUP BY family_id
                ) r2 ON r1.family_id = r2.family_id
                    AND r1.processed_time = r2.max_time
                    AND (r1.entity1_absolute_id = ? OR r1.entity2_absolute_id = ?)
                ORDER BY r1.processed_time DESC
            """
            params = (entity_absolute_id, entity_absolute_id, entity_absolute_id, entity_absolute_id)

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        cursor.execute(query, params)

        rows = cursor.fetchall()

        return [self._row_to_relation(row) for row in rows]
    
    def get_entity_relations_by_family_id(self, family_id: str, limit: Optional[int] = None, time_point: Optional[datetime] = None, max_version_absolute_id: Optional[str] = None) -> List[Relation]:
        """获取与指定实体相关的所有关系（通过family_id查找，包含该实体的所有版本）

        这个方法会查找该实体的所有版本（从最早版本开始）的所有关系，
        然后按family_id去重，保留每个family_id的最新版本。

        Args:
            family_id: 实体的family_id（不是absolute_id）
            limit: 限制返回的关系数量（按时间倒序），None表示不限制
            time_point: 时间点（可选），如果提供，只返回该时间点之前或等于该时间点的关系，且每个family_id只返回最新版本
            max_version_absolute_id: 最大版本absolute_id（可选），如果提供，只查询从最早版本到该版本的所有关系
        """
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        # 轻量查询：只取 id 和 processed_time，避免加载 content/embedding BLOB
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, processed_time FROM entities WHERE family_id = ? ORDER BY processed_time",
            (family_id,),
        )
        version_rows = cursor.fetchall()
        if not version_rows:
            return []

        # 如果指定了max_version_absolute_id，只取到该版本为止的所有版本
        if max_version_absolute_id:
            # 按时间排序，找到max_version_absolute_id对应的版本
            max_version = None
            for vid, pt in version_rows:
                if vid == max_version_absolute_id:
                    max_version = (vid, pt)
                    break

            if max_version:
                t_max = self._normalize_datetime_for_compare(
                    self._safe_parse_datetime(max_version[1])
                )
                # 只取到该版本（包含）为止的所有版本
                entity_absolute_ids = [
                    vid for vid, pt in version_rows
                    if self._normalize_datetime_for_compare(
                        self._safe_parse_datetime(pt)
                    ) <= t_max
                ]
                # 同时设置time_point为该版本的时间点
                if not time_point:
                    time_point = self._safe_parse_datetime(max_version[1])
                else:
                    # 如果已经设置了time_point，取较小值（避免 naive/aware 无法比较）
                    nt = self._normalize_datetime_for_compare(time_point)
                    if nt <= t_max:
                        pass  # 保持 time_point
                    else:
                        time_point = self._safe_parse_datetime(max_version[1])
            else:
                # 如果找不到指定的版本，使用所有版本
                entity_absolute_ids = [vid for vid, _ in version_rows]
        else:
            # 收集所有版本的absolute_id
            entity_absolute_ids = [vid for vid, _ in version_rows]
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 构建查询：查找所有版本的关系，按family_id去重
        placeholders = ','.join(['?'] * len(entity_absolute_ids))
        
        if time_point:
            # 获取每个family_id在该时间点之前或等于该时间点的最新版本
            r1_cols = ", r1.".join(self._RELATION_SELECT.split(", "))
            query = f"""
                SELECT r1.{r1_cols}
                FROM relations r1
                INNER JOIN (
                    SELECT family_id, MAX(processed_time) as max_time
                    FROM relations
                    WHERE (entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders}))
                    AND event_time <= ?
                    GROUP BY family_id
                ) r2 ON r1.family_id = r2.family_id 
                    AND r1.processed_time = r2.max_time
                    AND (r1.entity1_absolute_id IN ({placeholders}) OR r1.entity2_absolute_id IN ({placeholders}))
                ORDER BY r1.processed_time DESC
            """
            params = tuple(entity_absolute_ids * 2 + [time_point.isoformat()] + entity_absolute_ids * 2)
        else:
            # 获取每个family_id的最新版本
            r1_cols = ", r1.".join(self._RELATION_SELECT.split(", "))
            query = f"""
            SELECT r1.{r1_cols}
            FROM relations r1
            INNER JOIN (
                SELECT family_id, MAX(processed_time) as max_time
                FROM relations
                WHERE entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})
                GROUP BY family_id
            ) r2 ON r1.family_id = r2.family_id 
                AND r1.processed_time = r2.max_time
                AND (r1.entity1_absolute_id IN ({placeholders}) OR r1.entity2_absolute_id IN ({placeholders}))
            ORDER BY r1.processed_time DESC
            """
            params = tuple(entity_absolute_ids * 4)
        
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        
        cursor.execute(query, params)
        
        rows = cursor.fetchall()
        
        return [self._row_to_relation(row) for row in rows]

    def get_entity_relations_timeline(self, family_id: str, version_abs_ids: List[str]) -> List[Dict]:
        """批量获取实体在各版本时间点的关系（消除 N+1 查询）。

        与 Neo4j 后端保持一致的接口：获取该实体所有版本关联的关系，
        按每个版本的 processed_time 过滤，只返回在该版本时间点之前出现的关系。
        """
        family_id = self.resolve_family_id(family_id)
        if not family_id or not version_abs_ids:
            return []

        conn = self._get_conn()
        cursor = conn.cursor()

        # 获取各版本的 processed_time
        placeholders = ",".join("?" * len(version_abs_ids))
        cursor.execute(
            f"SELECT id, processed_time FROM entities WHERE id IN ({placeholders})",
            tuple(version_abs_ids),
        )
        version_times = []
        for row in cursor.fetchall():
            pt = row[1]
            if pt:
                pt = datetime.fromisoformat(pt) if isinstance(pt, str) else pt
            version_times.append((row[0], pt))
        if not version_times:
            return []

        # 获取该实体的所有 absolute_id
        all_abs_ids = self.get_entity_absolute_ids_up_to_version(
            family_id, max(version_abs_ids)
        )
        if not all_abs_ids:
            all_abs_ids = version_abs_ids[:]

        # 一次查询获取所有相关关系（按 family_id 去重，保留最新版本）
        placeholders = ",".join("?" * len(all_abs_ids))
        cursor.execute(
            f"""
            SELECT r.id, r.family_id, r.content, r.event_time, r.processed_time
            FROM relations r
            INNER JOIN (
                SELECT family_id, MAX(processed_time) AS max_pt
                FROM relations
                WHERE (entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders}))
                  AND invalid_at IS NULL
                GROUP BY family_id
            ) latest ON r.family_id = latest.family_id AND r.processed_time = latest.max_pt
            WHERE (r.entity1_absolute_id IN ({placeholders}) OR r.entity2_absolute_id IN ({placeholders}))
              AND r.invalid_at IS NULL
            ORDER BY r.processed_time ASC
            """,
            tuple(all_abs_ids) * 4,
        )

        timeline = []
        seen = set()
        for row in cursor.fetchall():
            rel_uuid = row[0]
            if rel_uuid in seen:
                continue
            rel_pt = row[4]
            if rel_pt:
                rel_pt = datetime.fromisoformat(rel_pt) if isinstance(rel_pt, str) else rel_pt
            # 检查关系是否出现在某个版本之前
            for _, v_pt in version_times:
                if rel_pt and v_pt and rel_pt <= v_pt:
                    seen.add(rel_uuid)
                    timeline.append({
                        "family_id": row[1],
                        "content": row[2],
                        "event_time": row[3],
                        "absolute_id": rel_uuid,
                    })
                    break
        return timeline

    def get_relations_by_entity_absolute_ids(self, entity_absolute_ids: List[str], limit: Optional[int] = None) -> List[Relation]:
        """获取与指定实体版本列表直接关联的所有关系（通过entity_absolute_id直接匹配）
        
        这个方法根据关系边中的 entity1_absolute_id 或 entity2_absolute_id 直接匹配，
        不使用时间过滤，只返回直接引用这些实体版本的关系边。
        按 family_id 去重，每个 family_id 只返回一条记录（保留最新的）。
        
        Args:
            entity_absolute_ids: 实体版本的absolute_id列表
            limit: 限制返回的关系数量，None表示不限制
        
        Returns:
            直接与这些实体版本关联的关系列表
        """
        if not entity_absolute_ids:
            return []
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 构建查询：查找直接引用这些 entity_absolute_id 的关系边
        placeholders = ','.join(['?'] * len(entity_absolute_ids))
        query = f"""
            SELECT {self._RELATION_SELECT}
            FROM relations
            WHERE entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})
            ORDER BY processed_time DESC
        """
        
        params = tuple(entity_absolute_ids * 2)
        cursor.execute(query, params)
        
        rows = cursor.fetchall()
        
        # 按 family_id 去重，保留第一个（最新的）
        seen_family_ids = set()
        result = []
        for row in rows:
            family_id_val = row[1]
            if family_id_val not in seen_family_ids:
                seen_family_ids.add(family_id_val)
                result.append(self._row_to_relation(row))
                if limit is not None and len(result) >= limit:
                    break
        
        return result
    
    def get_entity_absolute_ids_up_to_version(self, family_id: str, max_absolute_id: str) -> List[str]:
        """获取指定实体从最早版本到指定版本的所有 absolute_id 列表

        Args:
            family_id: 实体的 family_id
            max_absolute_id: 最大版本的 absolute_id（包含）
        
        Returns:
            从最早版本到指定版本的所有 absolute_id 列表（按时间顺序）
        """
        versions = self.get_entity_versions(family_id)
        if not versions:
            return []
        
        # 按时间排序（统一 naive/aware，避免混排报错）
        versions_sorted = sorted(
            versions,
            key=lambda v: self._normalize_datetime_for_compare(v.processed_time),
        )
        
        # 找到 max_absolute_id 对应的版本
        max_version = None
        for v in versions_sorted:
            if v.absolute_id == max_absolute_id:
                max_version = v
                break

        if not max_version:
            # 如果找不到指定的版本，返回空列表
            return []

        # 返回从最早版本到该版本（包含）的所有 absolute_id
        result = []
        for v in versions_sorted:
            result.append(v.absolute_id)
            if v.absolute_id == max_absolute_id:
                break
        
        return result
    
    def get_all_relations(self, limit: Optional[int] = None, offset: Optional[int] = None,
                          exclude_embedding: bool = False) -> List[Relation]:
        """获取所有关系的最新版本

        Args:
            limit: SQL 层限制返回条数（避免全量读取后在 Python 中截断）
            offset: SQL 层偏移量
            exclude_embedding: 是否排除 embedding 字段
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        r1_cols = ", r1.".join(self._RELATION_SELECT.split(", "))
        if exclude_embedding:
            r1_cols = r1_cols.replace("r1.embedding", "NULL AS r1__embedding")
        query = f"""
            SELECT r1.{r1_cols}
            FROM relations r1
            INNER JOIN (
                SELECT family_id, MAX(processed_time) as max_time
                FROM relations
                GROUP BY family_id
            ) r2 ON r1.family_id = r2.family_id AND r1.processed_time = r2.max_time
            ORDER BY r1.processed_time DESC
        """

        if offset is not None and offset > 0:
            query += f" OFFSET {int(offset)}"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        cursor.execute(query)

        rows = cursor.fetchall()

        return [self._row_to_relation(row) for row in rows]
    
    def _get_relations_with_embeddings(self) -> List[tuple]:
        """
        获取所有关系的最新版本及其embedding（带短 TTL 缓存）。

        Returns:
            List of (Relation, embedding_array) tuples, embedding_array为None表示没有embedding
        """
        now = time.time()
        if self._relation_emb_cache is not None and (now - self._relation_emb_cache_ts) < self._emb_cache_ttl:
            return self._relation_emb_cache

        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 使用窗口函数获取每个 family_id 的最新版本（O(N) 替代 O(N^2) 子查询）
        cursor.execute(f"""
            SELECT {self._RELATION_SELECT}
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY family_id ORDER BY processed_time DESC) AS rn
                FROM relations
            )
            WHERE rn = 1
        """)

        results = []
        for row in cursor.fetchall():
            # 解析embedding
            embedding_array = None
            if len(row) > 9 and row[9] is not None:
                try:
                    embedding_array = np.frombuffer(row[9], dtype=np.float32)
                except (ValueError, TypeError):
                    embedding_array = None
            relation = self._row_to_relation(row)
            results.append((relation, embedding_array))

        self._relation_emb_cache = results
        self._relation_emb_cache_ts = time.time()
        return results

    def search_relations_by_similarity(self, query_text: str,
                                      threshold: float = 0.3,
                                      max_results: int = 10) -> List[Relation]:
        """
        根据embedding相似度搜索关系
        
        Args:
            query_text: 查询文本
            threshold: 相似度阈值
            max_results: 返回的最大关系数量
            
        Returns:
            匹配的关系列表（按相似度排序）
        """
        # 获取所有关系及其embedding
        relations_with_embeddings = self._get_relations_with_embeddings()
        
        if not relations_with_embeddings:
            return []
        
        # 使用embedding相似度（如果可用）
        if self.embedding_client and self.embedding_client.is_available():
            return self._search_relations_with_embedding(
                query_text, relations_with_embeddings, threshold, max_results
            )
        else:
            # 使用文本相似度
            return self._search_relations_with_text_similarity(
                query_text, [r for r, _ in relations_with_embeddings], threshold, max_results
            )
    
    def _search_relations_with_embedding(self, query_text: str, 
                                         relations_with_embeddings: List[tuple],
                                         threshold: float, 
                                         max_results: int) -> List[Relation]:
        """使用embedding向量进行关系相似度搜索"""
        # 编码查询文本
        query_embedding = self.embedding_client.encode(query_text)
        if query_embedding is None:
            return []
        
        query_embedding_array = np.array(query_embedding[0] if isinstance(query_embedding, (list, np.ndarray)) else query_embedding, dtype=np.float32)
        
        # 计算相似度
        similarities = []
        for relation, stored_embedding in relations_with_embeddings:
            if stored_embedding is None:
                continue
            
            # 计算余弦相似度
            dot_product = np.dot(query_embedding_array, stored_embedding)
            norm_query = np.linalg.norm(query_embedding_array)
            norm_stored = np.linalg.norm(stored_embedding)
            similarity = dot_product / (norm_query * norm_stored + 1e-9)
            
            if similarity >= threshold:
                similarities.append((relation, float(similarity)))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回关系列表（去重，每个family_id只保留一个，并限制最大数量）
        relations = []
        seen_ids = set()
        for relation, _ in similarities:
            if relation.family_id not in seen_ids:
                relations.append(relation)
                seen_ids.add(relation.family_id)
                if len(relations) >= max_results:
                    break
        
        return relations
    
    def _search_relations_with_text_similarity(self, query_text: str, 
                                               all_relations: List[Relation],
                                               threshold: float,
                                               max_results: int) -> List[Relation]:
        """使用文本相似度进行关系搜索"""

        # 计算相似度并筛选
        scored_relations = []
        for relation in all_relations:
            relation_text = relation.content.lower()
            similarity = difflib.SequenceMatcher(
                None,
                query_text.lower(),
                relation_text
            ).ratio()
            
            if similarity >= threshold:
                scored_relations.append((relation, similarity))
        
        # 按相似度排序
        scored_relations.sort(key=lambda x: x[1], reverse=True)
        
        # 返回关系列表（去重，每个family_id只保留一个，并限制最大数量）
        relations = []
        seen_ids = set()
        for relation, _ in scored_relations:
            if relation.family_id not in seen_ids:
                relations.append(relation)
                seen_ids.add(relation.family_id)
                if len(relations) >= max_results:
                    break
        
        return relations
    
    # ========== 知识图谱整理操作 ==========
    
    def get_doc_hash_by_cache_id(self, cache_id: str) -> Optional[str]:
        """根据 episode_id 获取对应的文档目录名（doc_hash）。"""
        return self._id_to_doc_hash.get(cache_id)

    def get_episode_text(self, cache_id: str) -> Optional[str]:
        """获取记忆缓存对应的原始文本内容（优先从 docs/ 读取，回退旧结构）"""
        # 1. 尝试从 docs/ 新结构读取
        doc_hash = self._id_to_doc_hash.get(cache_id)
        if doc_hash:
            doc_dir = self.docs_dir / doc_hash
            # 优先从 original.txt 读取
            original_path = doc_dir / "original.txt"
            if original_path.exists():
                return original_path.read_text(encoding="utf-8")
            # 回退到 meta.json 中的 text 字段
            meta_path = doc_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    return meta.get("text")
                except Exception as exc:
                    logger.debug("meta.json text read failed: %s", exc)

        # 2. 回退到旧结构
        metadata_path = self.cache_json_dir / f"{cache_id}.json"
        if not metadata_path.exists():
            metadata_path = self.cache_dir / f"{cache_id}.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                return metadata.get("text")
            except Exception as exc:
                logger.debug("old cache json read failed: %s", exc)

        return None

    def get_doc_dir(self, doc_hash: str) -> Optional[Path]:
        """获取文档目录路径，不存在则返回 None。支持 hash 或时间戳+hash 格式。"""
        doc_dir = self.docs_dir / doc_hash
        if doc_dir.is_dir():
            return doc_dir
        # 回退：可能是旧纯 hash 格式，搜索匹配的目录
        if self.docs_dir.is_dir():
            for d in self.docs_dir.iterdir():
                if d.is_dir() and d.name.endswith(f"_{doc_hash}"):
                    return d
        return None

    def get_doc_content(self, filename: str) -> Optional[Dict[str, Any]]:
        """获取文档的原始文本和缓存摘要。

        Args:
            filename: 文档目录名（如 20260328_181737_f51dfac3186b）或 doc_hash。

        Returns:
            包含 original, cache, meta 的字典，找不到返回 None。
        """
        doc_dir = self.get_doc_dir(filename)
        if not doc_dir:
            return None

        result = {"meta": None, "original": None, "cache": None}

        # 读取 meta.json
        meta_path = doc_dir / "meta.json"
        if meta_path.exists():
            try:
                result["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.debug("meta.json read failed: %s", exc)

        # 读取 original.txt
        original_path = doc_dir / "original.txt"
        if original_path.exists():
            try:
                result["original"] = original_path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.debug("original.txt read failed: %s", exc)

        # 读取 cache.md
        cache_path = doc_dir / "cache.md"
        if cache_path.exists():
            try:
                result["cache"] = cache_path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.debug("cache.md read failed: %s", exc)

        return result

    def list_docs(self) -> List[Dict[str, Any]]:
        """列出所有文档的元数据摘要。

        文件格式：docs/{YYYYMMDD_HHMMSS}_{hash}/ 目录（按目录名自然排序即时间排序）。
        每个目录包含 original.txt、cache.md、meta.json。
        """
        docs = []
        if not self.docs_dir.is_dir():
            return docs

        for doc_dir in sorted(self.docs_dir.iterdir()):
            if not doc_dir.is_dir():
                continue
            meta_path = doc_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.debug("list_docs meta.json parse failed: %s", exc)
                continue

            source_name = meta.get("source_document") or ""
            event_time_str = meta.get("event_time", "")
            doc_hash = meta.get("doc_hash", "")

            # 从目录名提取 hash（格式: YYYYMMDD_HHMMSS_{hash}）
            if not doc_hash:
                parts = doc_dir.name.split("_", 2)
                if len(parts) >= 3:
                    doc_hash = parts[2]

            activity_type = meta.get("activity_type", "")

            # 读取 original.txt 大小作为文本长度，同时获取文件系统时间作为 processed_time
            text_len = 0
            processed_time_str = None
            original_path = doc_dir / "original.txt"
            if original_path.exists():
                try:
                    st = original_path.stat()
                    text_len = st.st_size
                    processed_time_str = datetime.fromtimestamp(st.st_mtime).isoformat()
                except Exception as exc:
                    logger.debug("original.txt stat failed: %s", exc)

            docs.append({
                "source_name": source_name,
                "source_document": source_name,
                "doc_name": source_name,
                "doc_hash": doc_hash,
                "activity_type": activity_type,
                "event_time": event_time_str or None,
                "processed_time": processed_time_str,
                "text_length": text_len,
                "filename": doc_dir.name,
            })
        return docs

    def merge_entity_families(self, target_family_id: str, source_family_ids: List[str]) -> Dict[str, Any]:
        """
        将多个source_family_id的记录合并到target_family_id

        Args:
            target_family_id: 目标实体ID（保留的ID）
            source_family_ids: 要合并的源实体ID列表
            
        Returns:
            合并结果统计，包含更新的实体数量和关系数量
        """
        target_family_id = self.resolve_family_id(target_family_id)
        if not target_family_id or not source_family_ids:
            return {"entities_updated": 0, "relations_updated": 0}

        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            entities_updated = 0
            relations_updated = 0

            try:
                # 1. 先获取所有源实体的版本数量（在更新之前，用于验证）
                source_version_counts = {}
                canonical_source_ids: List[str] = []
                for source_id in source_family_ids:
                    source_id = self._resolve_family_id_with_cursor(cursor, source_id)
                    if not source_id or source_id == target_family_id or source_id in canonical_source_ids:
                        continue
                    canonical_source_ids.append(source_id)
                    cursor.execute("""
                        SELECT COUNT(*) FROM entities
                        WHERE family_id = ?
                    """, (source_id,))
                    count = cursor.fetchone()[0]
                    source_version_counts[source_id] = count

                # 2. 更新entities表中的所有family_id记录
                # 这会更新所有使用source_family_id的记录，包括所有版本
                for source_id in canonical_source_ids:
                    cursor.execute("""
                        UPDATE entities
                        SET family_id = ?
                        WHERE family_id = ?
                    """, (target_family_id, source_id))
                    entities_updated += cursor.rowcount

                # 2.5. 验证：确保所有源实体的版本都被更新了
                # 检查是否还有任何源family_id的记录残留
                for source_id in canonical_source_ids:
                    cursor.execute("""
                        SELECT COUNT(*) FROM entities
                        WHERE family_id = ?
                    """, (source_id,))
                    remaining_count = cursor.fetchone()[0]
                    if remaining_count > 0:
                        # 如果还有残留记录，说明更新失败，回滚事务
                        conn.rollback()
                        raise ValueError(
                            f"合并失败：源实体 {source_id} 仍有 {remaining_count} 条记录未被更新 "
                            f"（预期应更新 {source_version_counts[source_id]} 条记录，实际更新了 {source_version_counts[source_id] - remaining_count} 条）"
                        )

                # 3. 获取target_family_id的最新版本的绝对ID
                cursor.execute("""
                    SELECT id FROM entities
                    WHERE family_id = ?
                    ORDER BY processed_time DESC
                    LIMIT 1
                """, (target_family_id,))
                target_absolute_id_row = cursor.fetchone()
                if not target_absolute_id_row:
                    conn.rollback()
                    return {"entities_updated": 0, "relations_updated": 0, "error": "目标实体不存在"}

                target_absolute_id = target_absolute_id_row[0]

                # 注意：关系边中的绝对ID保持不变，因为它们指向的是特定版本
                # 合并实体后，这些关系仍然有效，只是family_id变了
                # 不需要更新relations表，因为：
                # - 关系表只存储absolute_id，不存储family_id
                # - 通过absolute_id查询实体时，会得到更新后的family_id
                # - 所有使用family_id查询的地方（如get_relations_by_entities）都会自动使用新的family_id

                now_iso = datetime.now().isoformat()
                for source_id in canonical_source_ids:
                    cursor.execute(
                        """
                        INSERT INTO entity_redirects (source_family_id, target_family_id, updated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(source_family_id) DO UPDATE SET
                            target_family_id = excluded.target_family_id,
                            updated_at = excluded.updated_at
                        """,
                        (source_id, target_family_id, now_iso),
                    )

                conn.commit()

            except Exception as e:
                conn.rollback()
                raise e

            return {
                "entities_updated": entities_updated,
                "relations_updated": relations_updated,
                "target_family_id": target_family_id,
                "merged_source_ids": canonical_source_ids
            }
    
    def get_entity_version_count(self, family_id: str) -> int:
        """获取指定family_id的版本数量

        Args:
            family_id: 实体ID

        Returns:
            版本数量
        """
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM entities WHERE family_id = ?
        """, (family_id,))
        
        count = cursor.fetchone()[0]
        
        return count
    
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

    def get_entity_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
        """批量获取多个family_id的版本数量

        Args:
            family_ids: 实体ID列表

        Returns:
            Dict[family_id, version_count]
        """
        if not family_ids:
            return {}
        # 批量解析重定向
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = []
        seen_canonical = set()
        for family_id in family_ids:
            canonical_id = resolved_map.get(family_id, family_id)
            if canonical_id and canonical_id not in seen_canonical:
                seen_canonical.add(canonical_id)
                canonical_ids.append(canonical_id)
        if not canonical_ids:
            return {}
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 使用IN子句批量查询
        placeholders = ','.join(['?'] * len(canonical_ids))
        cursor.execute(f"""
            SELECT family_id, COUNT(*) as version_count
            FROM entities
            WHERE family_id IN ({placeholders})
            GROUP BY family_id
        """, canonical_ids)
        
        rows = cursor.fetchall()
        
        return {row[0]: row[1] for row in rows}

    def delete_entity_by_id(self, family_id: str) -> int:
        """删除实体的所有版本。返回删除的行数。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM entities WHERE family_id = ?", (family_id,))
            # 清理 FTS 表
            try:
                cursor.execute("DELETE FROM entity_fts WHERE family_id = ?", (family_id,))
            except Exception as exc:
                logger.warning("FTS delete failed: %s", exc)
            # Phase 2: cleanup concepts
            self._delete_concepts_by_family(family_id, cursor)
            return cursor.rowcount

    def delete_relation_by_id(self, family_id: str) -> int:
        """删除关系的所有版本。返回删除的行数。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM relations WHERE family_id = ?", (family_id,))
            count = cursor.rowcount
            # 清理 FTS 表
            try:
                cursor.execute("DELETE FROM relation_fts WHERE family_id = ?", (family_id,))
            except Exception as exc:
                logger.warning("FTS delete failed: %s", exc)
            # Phase 2: cleanup concepts
            self._delete_concepts_by_family(family_id, cursor)
            conn.commit()
            return count

    def delete_entity_all_versions(self, family_id: str) -> int:
        """删除实体的所有版本（含重定向解析）。返回删除的行数。"""
        return self.delete_entity_by_id(family_id)

    def delete_relation_all_versions(self, family_id: str) -> int:
        """删除关系的所有版本。返回删除的行数。"""
        return self.delete_relation_by_id(family_id)

    def batch_delete_entities(self, family_ids: List[str]) -> int:
        """批量删除实体 — 单次事务，替代 N 次单独删除。"""
        resolved_map = self.resolve_family_ids(family_ids)
        resolved = list(set(r for r in resolved_map.values() if r))
        if not resolved:
            return 0
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(resolved))
            cursor.execute(f"DELETE FROM entities WHERE family_id IN ({placeholders})", tuple(resolved))
            count = cursor.rowcount
            try:
                cursor.execute(f"DELETE FROM entity_fts WHERE family_id IN ({placeholders})", tuple(resolved))
            except Exception as exc:
                logger.warning("FTS delete failed: %s", exc)
            conn.commit()
            return count

    def batch_delete_relations(self, family_ids: List[str]) -> int:
        """批量删除关系 — 单次事务，替代 N 次单独删除。"""
        if not family_ids:
            return 0
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(family_ids))
            cursor.execute(f"DELETE FROM relations WHERE family_id IN ({placeholders})", tuple(family_ids))
            count = cursor.rowcount
            try:
                cursor.execute(f"DELETE FROM relation_fts WHERE family_id IN ({placeholders})", tuple(family_ids))
            except Exception as exc:
                logger.warning("FTS delete failed: %s", exc)
            conn.commit()
            return count

    def get_family_ids_by_names(self, names: list) -> dict:
        """按名称批量查询实 family_id（每个 name 取最新版本）。

        Returns:
            {name: family_id} 仅包含能找到的名称。
        """
        if not names:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ','.join(['?'] * len(names))
        cursor.execute(f"""
            SELECT name, family_id FROM entities
            WHERE name IN ({placeholders})
            ORDER BY processed_time DESC
        """, names)
        result = {}
        for name, eid in cursor.fetchall():
            if name not in result:
                result[name] = self.resolve_family_id(eid)
        return result

    def find_shortest_paths(self, source_family_id: str, target_family_id: str,
                            max_depth: int = 6, max_paths: int = 10) -> Dict[str, Any]:
        """使用 BFS 查找两个实体之间的所有最短路径。

        在 family_id 级别的无向图上执行 BFS，找到所有等长的最短路径，
        然后重构路径中每对相邻实体之间的连接关系。        Args:
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

        # 2. 加载全量最新关系（路径查找不需要 embedding）
        all_relations = self.get_all_relations(exclude_embedding=True)
        if not all_relations:
            result_empty["source_entity"] = source_entity
            result_empty["target_entity"] = target_entity
            return result_empty

        # 3. 构建 absolute_id → family_id 映射
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT id, family_id FROM entities")
        abs_to_eid = {row[0]: row[1] for row in cursor.fetchall()}

        # 4. 构建 family_id 级邻接表 和 entity_pair → [Relation] 映射
        adjacency: Dict[str, Set[str]] = {}
        pair_relations: Dict[Tuple[str, str], List[Relation]] = {}

        for rel in all_relations:
            eid1 = abs_to_eid.get(rel.entity1_absolute_id)
            eid2 = abs_to_eid.get(rel.entity2_absolute_id)
            if not eid1 or not eid2 or eid1 == eid2:
                continue  # 跳过无法解析或自指向的关系

            # 无向邻接
            adjacency.setdefault(eid1, set()).add(eid2)
            adjacency.setdefault(eid2, set()).add(eid1)

            # 有序 pair → relations 映射
            pair_key = tuple(sorted((eid1, eid2)))
            pair_relations.setdefault(pair_key, []).append(rel)

        # 5. 改进 BFS：记录所有最短路径父节点
        # visited: family_id → distance from source
        # parents: family_id → list of parent family_ids on shortest paths
        visited: Dict[str, int] = {source_family_id: 0}
        parents: Dict[str, List[str]] = {source_family_id: []}
        queue = [source_family_id]
        found_depth = None

        while queue and found_depth is None:
            next_queue = []
            for current in queue:
                current_dist = visited[current]
                if current_dist >= max_depth:
                    continue
                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = current_dist + 1
                        parents[neighbor] = [current]
                        next_queue.append(neighbor)
                        if neighbor == target_family_id:
                            found_depth = current_dist + 1
                    elif visited[neighbor] == current_dist + 1:
                        # 另一条等长路径
                        parents[neighbor].append(current)
            queue = next_queue

        # 未到达目标
        if target_family_id not in visited:
            result_empty["source_entity"] = source_entity
            result_empty["target_entity"] = target_entity
            return result_empty

        # 6. 回溯重构所有最短路径（DFS on parents）
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

        # 7. 构建返回结果
        # 使用关系实际引用的 absolute_id 查找对应版本的实体，
        # 确保前端 buildEdges() 的 nodeIds 过滤不会因版本不匹配而丢弃边。
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
                    # 按路径方向确定实体顺序
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

    def invalidate_relation(self, family_id: str, reason: str = "") -> int:
        """标记关系为失效（不删除数据，保留历史记录）"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE relations SET invalid_at = ?
                WHERE family_id = ? AND invalid_at IS NULL
            """, (datetime.now(timezone.utc).isoformat(), family_id))
            conn.commit()
            return cursor.rowcount

    def get_invalidated_relations(self, limit: int = 100) -> List[Relation]:
        """列出所有已失效的关系"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT {self._RELATION_SELECT}
            FROM relations
            WHERE invalid_at IS NOT NULL
            ORDER BY invalid_at DESC
            LIMIT ?
        """, (limit,))
        return [self._row_to_relation(row) for row in cursor.fetchall()]

    # ========== Phase A: 实体智能 ==========

    def update_entity_summary(self, family_id: str, summary: str):
        """更新实体的摘要（最新版本）。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE entities SET summary = ?
                WHERE id = (
                    SELECT id FROM entities
                    WHERE family_id = ?
                    ORDER BY processed_time DESC LIMIT 1
                )
            """, (summary, family_id))
            conn.commit()

    def update_entity_attributes(self, family_id: str, attributes: str):
        """更新实体的属性字典（JSON 字符串）。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE entities SET attributes = ?
                WHERE id = (
                    SELECT id FROM entities
                    WHERE family_id = ?
                    ORDER BY processed_time DESC LIMIT 1
                )
            """, (attributes, family_id))
            conn.commit()

    # ========== Phase B: 图遍历辅助 ==========

    def get_relations_by_family_ids(self, family_ids: List[str], limit: Optional[int] = None) -> List[Relation]:
        """获取与指定实体 ID 列表相关的所有关系。"""
        if not family_ids:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(family_ids))
        cursor.execute(f"""
            SELECT family_id, MAX(processed_time), id
            FROM entities
            WHERE family_id IN ({placeholders})
            GROUP BY family_id
        """, family_ids)
        abs_id_map = {row[0]: row[2] for row in cursor.fetchall()}
        abs_ids = list(abs_id_map.values())
        if not abs_ids:
            return []
        abs_placeholders = ",".join("?" * len(abs_ids))
        query = f"""
            SELECT {self._RELATION_SELECT}
            FROM relations
            WHERE entity1_absolute_id IN ({abs_placeholders})
               OR entity2_absolute_id IN ({abs_placeholders})
            ORDER BY processed_time DESC
        """
        params = abs_ids + abs_ids
        if limit:
            query += " LIMIT ?"
            params.append(int(limit))
        cursor.execute(query, params)
        return [self._row_to_relation(row) for row in cursor.fetchall()]

    def batch_get_entity_degrees(self, family_ids: List[str]) -> Dict[str, int]:
        """批量获取实体度数 — 单次查询替代 N 次 get_entity_degree。"""
        if not family_ids:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(family_ids))
        cursor.execute(
            f"SELECT e.family_id, COUNT(r.id) "
            f"FROM entities e "
            f"LEFT JOIN relations r ON ("
            f"(r.entity1_absolute_id = e.id OR r.entity2_absolute_id = e.id) "
            f"AND r.invalid_at IS NULL) "
            f"WHERE e.family_id IN ({placeholders}) "
            f"GROUP BY e.family_id",
            tuple(family_ids),
        )
        result = {row[0]: row[1] for row in cursor.fetchall()}
        # 补零：未出现在结果中的 family_id 度数为 0
        for fid in family_ids:
            result.setdefault(fid, 0)
        return result

    def batch_get_entity_profiles(self, family_ids: List[str]) -> List[Dict[str, Any]]:
        """批量获取实体档案（entity + relations + version_count），消除 N+1。

        替代对每个 family_id 分别调用 get_entity_by_family_id +
        get_entity_relations_by_family_id + get_entity_version_count 的 N+1 模式。

        Returns:
            [{"family_id", "entity", "relations", "version_count"}, ...]
        """
        if not family_ids:
            return []

        # Step 1: 解析 canonical family_ids
        canonical_map: Dict[str, str] = {}  # original -> canonical
        canonical_set: List[str] = []
        for fid in family_ids:
            resolved = self.resolve_family_id(fid)
            canonical = resolved if resolved else fid
            canonical_map[fid] = canonical
            if canonical not in canonical_map.values() or canonical == canonical_map.get(fid):
                if canonical not in canonical_set:
                    canonical_set.append(canonical)

        if not canonical_set:
            return [{"family_id": fid, "entity": None, "relations": [], "version_count": 0} for fid in family_ids]

        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(canonical_set))

        # Step 2: 批量获取最新实体（窗口函数取每组最新一条）
        cursor.execute(f"""
            SELECT id, family_id, name, content, event_time, processed_time,
                   episode_id, source_document, embedding, summary, attributes, confidence
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY family_id ORDER BY processed_time DESC) AS rn
                FROM entities WHERE family_id IN ({placeholders})
            ) WHERE rn = 1
        """, tuple(canonical_set))

        entity_map: Dict[str, Entity] = {}
        for row in cursor.fetchall():
            fid = row[1]
            entity_map[fid] = self._row_to_entity(row)

        # Step 3: 批量获取版本数
        version_counts = self.get_entity_version_counts(canonical_set)

        # Step 4: 批量获取所有相关 absolute_ids，再批量查关系
        cursor.execute(f"""
            SELECT family_id, id FROM entities
            WHERE family_id IN ({placeholders}) AND invalid_at IS NULL
        """, tuple(canonical_set))

        fid_to_aids: Dict[str, List[str]] = {}
        all_aids: List[str] = []
        for row in cursor.fetchall():
            fid_to_aids.setdefault(row[0], []).append(row[1])
            all_aids.append(row[1])

        relations_map: Dict[str, List[Relation]] = {fid: [] for fid in canonical_set}
        if all_aids:
            aid_placeholders = ",".join("?" * len(all_aids))
            cursor.execute(f"""
                SELECT {self._RELATION_SELECT}
                FROM relations
                WHERE (entity1_absolute_id IN ({aid_placeholders})
                    OR entity2_absolute_id IN ({aid_placeholders}))
                  AND invalid_at IS NULL
            """, tuple(all_aids) * 2)

            all_rels = [self._row_to_relation(row) for row in cursor.fetchall()]

            # 分配关系到对应的 family_id — O(R) 用 reverse lookup 替代 O(R*F)
            aid_to_fid: Dict[str, str] = {}
            for fid, aids in fid_to_aids.items():
                for aid in aids:
                    aid_to_fid[aid] = fid

            for rel in all_rels:
                fid1 = aid_to_fid.get(rel.entity1_absolute_id)
                fid2 = aid_to_fid.get(rel.entity2_absolute_id)
                if fid1:
                    relations_map[fid1].append(rel)
                if fid2 and fid2 != fid1:
                    relations_map[fid2].append(rel)

        # Step 5: 组装结果
        results = []
        seen_fids = set()
        for fid in family_ids:
            canonical = canonical_map.get(fid, fid)
            if canonical in seen_fids:
                results.append({"family_id": fid, "entity": None, "relations": [], "version_count": 0})
                continue
            seen_fids.add(canonical)
            entity = entity_map.get(canonical)
            if entity:
                results.append({
                    "family_id": canonical,
                    "entity": entity,
                    "relations": relations_map.get(canonical, []),
                    "version_count": version_counts.get(canonical, 1),
                })
            else:
                results.append({"family_id": canonical, "entity": None, "relations": [], "version_count": 0})

        return results

    # ========== Phase C: Episode MENTIONS ==========

    def _migrate_episode_mentions(self, cursor):
        """迁移旧 episode_mentions 表到新 schema（幂等）。

        旧表: (episode_id, entity_absolute_id, mention_context)
        新表: (episode_id, target_absolute_id, target_type, mention_context)

        SQLite 的 CREATE TABLE IF NOT EXISTS 不会修改已存在表的列。
        因此需要：检测旧 schema → rename → 创建新表 → 迁移数据 → drop 旧表。
        """
        try:
            cursor.execute("PRAGMA table_info(episode_mentions)")
            columns = {row[1] for row in cursor.fetchall()}
            if "entity_absolute_id" in columns and "target_type" not in columns:
                # 旧表结构：rename → 创建新表 → 迁移 → 删除旧表
                cursor.execute(
                    "ALTER TABLE episode_mentions RENAME TO episode_mentions_old"
                )
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS episode_mentions (
                        episode_id TEXT NOT NULL,
                        target_absolute_id TEXT NOT NULL,
                        target_type TEXT NOT NULL DEFAULT 'entity',
                        mention_context TEXT DEFAULT '',
                        PRIMARY KEY (episode_id, target_absolute_id, target_type)
                    )
                """)
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_episode_mentions_target "
                    "ON episode_mentions(target_absolute_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_episode_mentions_episode "
                    "ON episode_mentions(episode_id)"
                )
                cursor.execute("""
                    INSERT OR IGNORE INTO episode_mentions
                        (episode_id, target_absolute_id, target_type, mention_context)
                    SELECT episode_id, entity_absolute_id, 'entity', mention_context
                    FROM episode_mentions_old
                """)
                cursor.execute("DROP TABLE episode_mentions_old")
                logger.info("episode_mentions: 旧表迁移到新 schema 完成")
        except Exception as exc:
            logger.debug("episode_mentions migration skipped: %s", exc)

    def _save_episode_to_db(self, cache: Episode, doc_hash: str = ""):
        """将 Episode 元数据写入 SQLite episodes 表。"""
        try:
            conn = self._get_conn()
            with self._write_lock:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO episodes
                    (id, family_id, content, event_time, processed_time,
                     source_document, activity_type, episode_type, doc_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache.absolute_id,
                    cache.absolute_id,  # family_id = absolute_id（Episode 当前不可版本化）
                    cache.content,
                    cache.event_time.isoformat(),
                    datetime.now().isoformat(),
                    cache.source_document,
                    cache.activity_type or "",
                    getattr(cache, 'episode_type', '') or "",
                    doc_hash,
                ))
                # Phase 2: dual-write to concepts
                self._write_concept_from_episode(cache, doc_hash, cursor)
                conn.commit()
        except Exception as exc:
            logger.warning("episode SQLite write failed: %s", exc)

    def _migrate_episodes_from_files(self):
        """启动时将 docs/ 目录中的 Episode 元数据迁移到 SQLite（幂等）。"""
        if not self.docs_dir.is_dir():
            return
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM episodes")
        existing = cursor.fetchone()[0]
        if existing > 0:
            return  # 已迁移过
        migrated = 0
        for doc_dir in self.docs_dir.iterdir():
            if not doc_dir.is_dir():
                continue
            meta_path = doc_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                cache_id = meta.get("absolute_id") or meta.get("id")
                if not cache_id:
                    continue
                cache_md_path = doc_dir / "cache.md"
                content = ""
                if cache_md_path.exists():
                    content = cache_md_path.read_text(encoding="utf-8")
                cursor.execute("""
                    INSERT OR IGNORE INTO episodes
                    (id, family_id, content, event_time, processed_time,
                     source_document, activity_type, episode_type, doc_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_id,
                    cache_id,
                    content,
                    meta.get("event_time", ""),
                    meta.get("event_time", ""),
                    meta.get("source_document", ""),
                    meta.get("activity_type", ""),
                    "",
                    doc_dir.name,
                ))
                migrated += 1
            except Exception as exc:
                logger.debug("episode migration skip: %s", exc)
        if migrated:
            conn.commit()
            logger.info("episodes 表迁移完成: %d 条", migrated)

    def save_episode_mentions(self, episode_id: str, target_absolute_ids: List[str],
                              context: str = "", target_type: str = "entity"):
        """记录 Episode 中提及的概念（entity / relation）。

        Args:
            episode_id: Episode 的 absolute_id
            target_absolute_ids: 被提及概念的 absolute_id 列表
            context: 提及上下文
            target_type: "entity" | "relation"
        """
        if not target_absolute_ids:
            return
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            for abs_id in target_absolute_ids:
                cursor.execute("""
                    INSERT OR REPLACE INTO episode_mentions
                    (episode_id, target_absolute_id, target_type, mention_context)
                    VALUES (?, ?, ?, ?)
                """, (episode_id, abs_id, target_type, context))
            conn.commit()

    def get_entity_provenance(self, family_id: str) -> List[dict]:
        """获取提及该实体的所有 Episode。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM entities WHERE family_id = ?", (family_id,))
        abs_ids = [row[0] for row in cursor.fetchall()]
        if not abs_ids:
            return []
        placeholders = ",".join("?" * len(abs_ids))
        cursor.execute(f"""
            SELECT episode_id, target_absolute_id, target_type, mention_context
            FROM episode_mentions
            WHERE target_absolute_id IN ({placeholders})
        """, abs_ids)
        return [
            {"episode_id": row[0], "entity_absolute_id": row[1],
             "target_type": row[2], "mention_context": row[3] or ""}
            for row in cursor.fetchall()
        ]

    def get_episode_entities(self, episode_id: str) -> List[dict]:
        """获取 Episode 中提及的所有实体和关系。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        # Entity targets
        cursor.execute("""
            SELECT em.target_absolute_id, em.target_type, em.mention_context,
                   e.name, e.family_id
            FROM episode_mentions em
            LEFT JOIN entities e ON em.target_absolute_id = e.id
            WHERE em.episode_id = ? AND em.target_type = 'entity'
        """, (episode_id,))
        results = [
            {"absolute_id": row[0], "target_type": row[1] or "entity",
             "mention_context": row[2] or "", "name": row[3] or "",
             "family_id": row[4] or ""}
            for row in cursor.fetchall()
        ]
        # Relation targets
        cursor.execute("""
            SELECT em.target_absolute_id, em.target_type, em.mention_context,
                   r.family_id
            FROM episode_mentions em
            LEFT JOIN relations r ON em.target_absolute_id = r.id
            WHERE em.episode_id = ? AND em.target_type = 'relation'
        """, (episode_id,))
        for row in cursor.fetchall():
            results.append({
                "absolute_id": row[0], "target_type": "relation",
                "mention_context": row[2] or "", "name": row[3] or "",
                "family_id": row[3] or "",
            })
        return results

    def delete_episode_mentions(self, episode_id: str):
        """删除 Episode 的所有提及记录。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM episode_mentions WHERE episode_id = ?", (episode_id,))
            conn.commit()

    def get_episode_from_db(self, episode_id: str) -> Optional[Dict]:
        """从 SQLite episodes 表获取 Episode 元数据。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        row = cursor.fetchone()
        if not row:
            return None
        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))

    def list_episodes_from_db(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """从 SQLite episodes 表列出 Episode（按时间倒序）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM episodes ORDER BY event_time DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def list_episodes(self, limit: int = 20, offset: int = 0) -> List[Dict]:
        """分页列出 Episode（兼容 Neo4j 接口）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, content, source_document, event_time, activity_type "
            "FROM episodes ORDER BY event_time DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        episodes = []
        for row in cursor.fetchall():
            episodes.append({
                "uuid": row[0],
                "content": row[1] or "",
                "source_document": row[2] or "",
                "event_time": row[3] or "",
                "activity_type": row[4] or "",
                "created_at": row[3] or "",
            })
        return episodes

    def count_episodes(self) -> int:
        """统计 Episode 总数。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM episodes")
        return cursor.fetchone()[0]

    def get_episode(self, episode_id: str) -> Optional[Dict]:
        """获取单个 Episode 详情（兼容 Neo4j 接口）。"""
        return self.get_episode_from_db(episode_id)

    # ========== Phase D: 关系溯源 ==========

    # ========== Phase E: Dream Logs ==========

    def save_dream_log(self, report):
        """保存梦境报告。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO dream_logs
                (cycle_id, graph_id, start_time, end_time, status, narrative,
                 insights_json, connections_json, consolidations_json, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.cycle_id, report.graph_id,
                report.start_time.isoformat(),
                report.end_time.isoformat() if report.end_time else None,
                report.status, report.narrative,
                json.dumps(report.insights, ensure_ascii=False),
                json.dumps(report.new_connections, ensure_ascii=False),
                json.dumps(report.consolidations, ensure_ascii=False),
                json.dumps({}, ensure_ascii=False),
            ))
            conn.commit()

    def list_dream_logs(self, graph_id: str = "default", limit: int = 20) -> List[dict]:
        """列出梦境日志。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT cycle_id, graph_id, start_time, end_time, status, narrative
            FROM dream_logs WHERE graph_id = ?
            ORDER BY start_time DESC LIMIT ?
        """, (graph_id, limit))
        return [
            {"cycle_id": row[0], "graph_id": row[1], "start_time": row[2],
             "end_time": row[3], "status": row[4], "narrative": (row[5] or "")[:200]}
            for row in cursor.fetchall()
        ]

    def get_dream_log(self, cycle_id: str) -> Optional[dict]:
        """获取单次梦境日志详情。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT cycle_id, graph_id, start_time, end_time, status, narrative,
                   insights_json, connections_json, consolidations_json
            FROM dream_logs WHERE cycle_id = ?
        """, (cycle_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "cycle_id": row[0], "graph_id": row[1], "start_time": row[2],
            "end_time": row[3], "status": row[4], "narrative": row[5] or "",
            "insights": json.loads(row[6]) if row[6] else [],
            "connections": json.loads(row[7]) if row[7] else [],
            "consolidations": json.loads(row[8]) if row[8] else [],
        }

    # ========== Version-Level CRUD (Phase 2) ==========

    def update_entity_by_absolute_id(self, absolute_id: str, **fields) -> Optional[Entity]:
        """根据绝对ID更新实体字段（name, content, summary, attributes, confidence）。"""
        allowed = {"name", "content", "summary", "attributes", "confidence"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_entity_by_absolute_id(absolute_id)

        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            for field, value in updates.items():
                cursor.execute(
                    f"UPDATE entities SET {field} = ? WHERE id = ?",
                    (value, absolute_id),
                )
            conn.commit()

        return self.get_entity_by_absolute_id(absolute_id)

    def delete_entity_by_absolute_id(self, absolute_id: str) -> bool:
        """根据绝对ID删除实体，同时清理 FTS 索引。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            # Check existence
            cursor.execute(
                "SELECT 1 FROM entities WHERE id = ?",
                (absolute_id,),
            )
            if cursor.fetchone() is None:
                return False
            # Delete FTS entry (using integer rowid)
            cursor.execute(
                "DELETE FROM entity_fts WHERE rowid = (SELECT rowid FROM entities WHERE id = ?)",
                (absolute_id,),
            )
            # Delete entity version
            cursor.execute(
                "DELETE FROM entities WHERE id = ?",
                (absolute_id,),
            )
            affected = cursor.rowcount
            # Phase 2: cleanup concepts
            self._delete_concept_by_id(absolute_id, cursor)
            conn.commit()
            return affected > 0

    def update_relation_by_absolute_id(self, absolute_id: str, **fields) -> Optional[Relation]:
        """根据绝对ID更新关系字段（content, summary, attributes, confidence）。"""
        allowed = {"content", "summary", "attributes", "confidence"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_relation_by_absolute_id(absolute_id)

        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            for field, value in updates.items():
                cursor.execute(
                    f"UPDATE relations SET {field} = ? WHERE id = ?",
                    (value, absolute_id),
                )
            conn.commit()

        return self.get_relation_by_absolute_id(absolute_id)

    def delete_relation_by_absolute_id(self, absolute_id: str) -> bool:
        """根据绝对ID删除关系，同时清理 FTS 索引。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            # Check existence
            cursor.execute(
                "SELECT 1 FROM relations WHERE id = ?",
                (absolute_id,),
            )
            if cursor.fetchone() is None:
                return False
            # Delete FTS entry (using integer rowid)
            cursor.execute(
                "DELETE FROM relation_fts WHERE rowid = (SELECT rowid FROM relations WHERE id = ?)",
                (absolute_id,),
            )
            # Delete relation version
            cursor.execute(
                "DELETE FROM relations WHERE id = ?",
                (absolute_id,),
            )
            affected = cursor.rowcount
            # Phase 2: cleanup concepts
            self._delete_concept_by_id(absolute_id, cursor)
            conn.commit()
            return affected > 0

    def batch_delete_relation_versions_by_absolute_ids(self, absolute_ids: List[str]) -> int:
        """批量删除指定关系版本（带 FTS 清理），返回成功删除的数量。"""
        if not absolute_ids:
            return 0
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(absolute_ids))
            # Delete FTS entries (using integer rowids)
            cursor.execute(
                f"DELETE FROM relation_fts WHERE rowid IN (SELECT rowid FROM relations WHERE id IN ({placeholders}))",
                list(absolute_ids),
            )
            # Delete relation versions
            cursor.execute(
                f"DELETE FROM relations WHERE id IN ({placeholders})",
                list(absolute_ids),
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted

    def get_relations_referencing_absolute_id(self, absolute_id: str) -> List[Relation]:
        """获取所有引用指定实体绝对ID的关系。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {self._RELATION_SELECT} FROM relations WHERE entity1_absolute_id = ? OR entity2_absolute_id = ?",
            (absolute_id, absolute_id),
        )
        return [self._row_to_relation(row) for row in cursor.fetchall()]

    def batch_get_relations_referencing_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, List[Relation]]:
        """批量获取引用指定实体绝对ID的关系（消除 N+1 查询）。"""
        if not absolute_ids:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(absolute_ids))
        params = list(absolute_ids) + list(absolute_ids)
        cursor.execute(f"""
            SELECT {self._RELATION_SELECT}
            FROM relations
            WHERE entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})
        """, params)
        result_map: Dict[str, List[Relation]] = {aid: [] for aid in absolute_ids}
        for row in cursor.fetchall():
            rel = self._row_to_relation(row)
            if rel.entity1_absolute_id in result_map:
                result_map[rel.entity1_absolute_id].append(rel)
            if rel.entity2_absolute_id in result_map:
                result_map[rel.entity2_absolute_id].append(rel)
        return result_map

    def batch_delete_entity_versions_by_absolute_ids(self, absolute_ids: List[str]) -> int:
        """批量删除指定实体版本（带 FTS 清理），返回成功删除的数量。"""
        if not absolute_ids:
            return 0
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(absolute_ids))
            # Delete FTS entries (using integer rowids)
            cursor.execute(
                f"DELETE FROM entity_fts WHERE rowid IN (SELECT rowid FROM entities WHERE id IN ({placeholders}))",
                list(absolute_ids),
            )
            # Delete entity versions
            cursor.execute(
                f"DELETE FROM entities WHERE id IN ({placeholders})",
                list(absolute_ids),
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted

    def split_entity_version(self, absolute_id: str, new_family_id: str = "") -> Optional[Entity]:
        """将实体版本从当前 family 拆分到新 family。"""
        import uuid as _uuid
        if not new_family_id:
            new_family_id = f"ent_{_uuid.uuid4().hex[:12]}"

        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            # Check existence and get old family_id
            cursor.execute(
                "SELECT family_id FROM entities WHERE id = ?",
                (absolute_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            # Update family_id
            cursor.execute(
                "UPDATE entities SET family_id = ? WHERE id = ?",
                (new_family_id, absolute_id),
            )
            conn.commit()

        return self.get_entity_by_absolute_id(absolute_id)

    def redirect_relation(self, family_id: str, side: str, new_family_id: str) -> int:
        """将指定 family 下所有关系的某一端重定向到新 family 的最新实体。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            # Resolve new_family_id through redirect chain
            resolved_family = self._resolve_family_id_with_cursor(cursor, new_family_id)
            # Get latest entity absolute_id for the new family
            cursor.execute(
                "SELECT id FROM entities WHERE family_id = ? ORDER BY processed_time DESC LIMIT 1",
                (resolved_family,),
            )
            entity_row = cursor.fetchone()
            if entity_row is None:
                return 0
            new_absolute_id = entity_row[0]
            # Update the specified side
            if side == "entity1":
                cursor.execute(
                    "UPDATE relations SET entity1_absolute_id = ? WHERE family_id = ?",
                    (new_absolute_id, family_id),
                )
            elif side == "entity2":
                cursor.execute(
                    "UPDATE relations SET entity2_absolute_id = ? WHERE family_id = ?",
                    (new_absolute_id, family_id),
                )
            else:
                return 0
            affected = cursor.rowcount
            conn.commit()
            return affected

    # ========== Phase 2: concepts 统一表双写 ==========

    def _write_concept_from_entity(self, entity: Entity, cursor):
        """Dual-write: write Entity to concepts table (called within existing write transaction)."""
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO concepts
                (id, family_id, role, name, content, event_time, processed_time,
                 source_document, episode_id, embedding, valid_at, invalid_at,
                 summary, attributes, confidence, content_format, provenance)
                VALUES (?, ?, 'entity', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '')
            """, (
                entity.absolute_id,
                entity.family_id,
                entity.name,
                entity.content,
                entity.event_time.isoformat(),
                entity.processed_time.isoformat(),
                entity.source_document or '',
                entity.episode_id or '',
                entity.embedding,
                (entity.valid_at or entity.event_time).isoformat(),
                getattr(entity, 'invalid_at', None),
                getattr(entity, 'summary', None),
                json.dumps(getattr(entity, 'attributes', None)) if isinstance(getattr(entity, 'attributes', None), dict) else getattr(entity, 'attributes', None),
                getattr(entity, 'confidence', None),
                getattr(entity, 'content_format', 'plain'),
            ))
            # FTS (content-sync: use integer rowid from concepts table)
            try:
                cursor.execute(
                    "SELECT rowid FROM concepts WHERE id = ?", (entity.absolute_id,)
                )
                _row = cursor.fetchone()
                if _row:
                    cursor.execute("""
                        INSERT INTO concept_fts(rowid, name, content)
                        VALUES (?, ?, ?)
                    """, (_row[0], entity.name, entity.content))
            except Exception as exc:
                logger.debug("concept_fts entity write failed: %s", exc)
        except Exception as exc:
            logger.debug("concept entity dual-write failed: %s", exc)

    def _write_concept_from_relation(self, relation: Relation, cursor):
        """Dual-write: write Relation to concepts table (called within existing write transaction)."""
        try:
            connects = json.dumps([relation.entity1_absolute_id, relation.entity2_absolute_id])
            cursor.execute("""
                INSERT OR IGNORE INTO concepts
                (id, family_id, role, name, content, event_time, processed_time,
                 source_document, episode_id, embedding, valid_at, invalid_at,
                 summary, attributes, confidence, content_format, provenance, connects)
                VALUES (?, ?, 'relation', '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                relation.absolute_id,
                relation.family_id,
                relation.content,
                relation.event_time.isoformat(),
                relation.processed_time.isoformat(),
                relation.source_document or '',
                relation.episode_id or '',
                relation.embedding,
                (relation.valid_at or relation.event_time).isoformat(),
                getattr(relation, 'invalid_at', None),
                getattr(relation, 'summary', None),
                json.dumps(getattr(relation, 'attributes', None)) if isinstance(getattr(relation, 'attributes', None), dict) else getattr(relation, 'attributes', None),
                getattr(relation, 'confidence', None),
                getattr(relation, 'content_format', 'plain'),
                getattr(relation, 'provenance', '') or '',
                connects,
            ))
            # FTS (content-sync: use integer rowid from concepts table)
            try:
                cursor.execute(
                    "SELECT rowid FROM concepts WHERE id = ?", (relation.absolute_id,)
                )
                _row = cursor.fetchone()
                if _row:
                    cursor.execute("""
                        INSERT INTO concept_fts(rowid, name, content)
                        VALUES (?, '', ?)
                    """, (_row[0], relation.content))
            except Exception as exc:
                logger.debug("concept_fts relation write failed: %s", exc)
        except Exception as exc:
            logger.debug("concept relation dual-write failed: %s", exc)

    def _write_concept_from_episode(self, cache: Episode, doc_hash: str, cursor):
        """Dual-write: write Episode to concepts table as 'observation' (called within existing write transaction)."""
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO concepts
                (id, family_id, role, name, content, event_time, processed_time,
                 source_document, activity_type, episode_type, provenance)
                VALUES (?, ?, 'observation', '', ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache.absolute_id,
                cache.absolute_id,  # family_id = absolute_id for episodes
                cache.content,
                cache.event_time.isoformat(),
                datetime.now().isoformat(),
                cache.source_document or '',
                cache.activity_type or '',
                getattr(cache, 'episode_type', '') or '',
                json.dumps({"doc_hash": doc_hash}) if doc_hash else '',
            ))
            # FTS (content-sync: use integer rowid from concepts table)
            try:
                cursor.execute(
                    "SELECT rowid FROM concepts WHERE id = ?", (cache.absolute_id,)
                )
                _row = cursor.fetchone()
                if _row:
                    cursor.execute("""
                        INSERT INTO concept_fts(rowid, name, content)
                        VALUES (?, '', ?)
                    """, (_row[0], cache.content))
            except Exception as exc:
                logger.debug("concept_fts episode write failed: %s", exc)
        except Exception as exc:
            logger.debug("concept episode dual-write failed: %s", exc)

    def _delete_concept_by_id(self, absolute_id: str, cursor):
        """Delete a concept version by absolute_id (called within existing write transaction)."""
        try:
            # Get integer rowid before delete for FTS cleanup
            cursor.execute("SELECT rowid FROM concepts WHERE id = ?", (absolute_id,))
            _row = cursor.fetchone()
            if _row:
                try:
                    cursor.execute("DELETE FROM concept_fts WHERE rowid = ?", (_row[0],))
                except Exception as exc:
                    logger.debug("concept_fts delete by id failed: %s", exc)
            cursor.execute("DELETE FROM concepts WHERE id = ?", (absolute_id,))
        except Exception as exc:
            logger.debug("concept delete by id failed: %s", exc)

    def _delete_concepts_by_family(self, family_id: str, cursor):
        """Delete all concept versions by family_id (called within existing write transaction)."""
        try:
            # Collect integer rowids before delete for FTS cleanup
            cursor.execute("SELECT rowid FROM concepts WHERE family_id = ?", (family_id,))
            _rowids = [r[0] for r in cursor.fetchall()]
            if _rowids:
                try:
                    placeholders = ",".join("?" * len(_rowids))
                    cursor.execute(
                        f"DELETE FROM concept_fts WHERE rowid IN ({placeholders})",
                        _rowids,
                    )
                except Exception as exc:
                    logger.debug("concept_fts delete by family failed: %s", exc)
            cursor.execute("DELETE FROM concepts WHERE family_id = ?", (family_id,))
        except Exception as exc:
            logger.debug("concept delete by family failed: %s", exc)

    def migrate_to_concepts(self):
        """Migrate existing entities + relations + episodes to concepts table (idempotent)."""
        conn = self._get_conn()
        with self._write_lock:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM concepts")
                if cursor.fetchone()[0] > 0:
                    return  # already migrated
                # entities -> concepts
                cursor.execute("""
                    INSERT OR IGNORE INTO concepts
                    (id, family_id, role, name, content, event_time, processed_time,
                     source_document, episode_id, embedding, valid_at, invalid_at,
                     summary, attributes, confidence, content_format)
                    SELECT id, family_id, 'entity', name, content, event_time, processed_time,
                           source_document, episode_id, embedding, valid_at, invalid_at,
                           summary, attributes, confidence, content_format
                    FROM entities
                """)
                # relations -> concepts
                cursor.execute("""
                    INSERT OR IGNORE INTO concepts
                    (id, family_id, role, content, event_time, processed_time,
                     source_document, episode_id, embedding, valid_at, invalid_at,
                     summary, attributes, confidence, content_format, connects)
                    SELECT id, family_id, 'relation', content, event_time, processed_time,
                           source_document, episode_id, embedding, valid_at, invalid_at,
                           summary, attributes, confidence, content_format,
                           json_array(entity1_absolute_id, entity2_absolute_id)
                    FROM relations
                """)
                # episodes -> concepts
                cursor.execute("""
                    INSERT OR IGNORE INTO concepts
                    (id, family_id, role, content, event_time, processed_time,
                     source_document, activity_type, episode_type)
                    SELECT id, family_id, 'observation', content, event_time, processed_time,
                           source_document, activity_type, episode_type
                    FROM episodes
                """)
                # Rebuild concept_fts（content-sync FTS5 使用 rebuild 命令）
                cursor.execute("INSERT INTO concept_fts(concept_fts) VALUES('rebuild')")
                conn.commit()
                logger.info("concepts 表迁移完成")
            except Exception as exc:
                logger.warning("concepts 迁移失败: %s", exc)
                conn.rollback()

    # ========== Phase 3: 统一概念查询接口 ==========

    def get_concept_by_family_id(self, family_id: str) -> Optional[dict]:
        """获取任意 role 的概念最新版本。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM concepts WHERE family_id = ? ORDER BY processed_time DESC LIMIT 1",
            (family_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))

    def get_concepts_by_family_ids(self, family_ids: List[str]) -> Dict[str, dict]:
        """批量获取概念最新版本。"""
        if not family_ids:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ','.join('?' * len(family_ids))
        # Get latest version of each family_id using a subquery
        cursor.execute(f"""
            SELECT c.* FROM concepts c
            INNER JOIN (
                SELECT family_id, MAX(processed_time) as max_pt
                FROM concepts WHERE family_id IN ({placeholders})
                GROUP BY family_id
            ) latest ON c.family_id = latest.family_id AND c.processed_time = latest.max_pt
        """, family_ids)
        cols = [desc[0] for desc in cursor.description]
        result = {}
        for row in cursor.fetchall():
            d = dict(zip(cols, row))
            result[d['family_id']] = d
        return result

    def search_concepts_by_bm25(self, query: str, role: str = None, limit: int = 20) -> List[dict]:
        """BM25 搜索概念，可选按 role 过滤。"""
        if not query:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            if role:
                cursor.execute("""
                    SELECT c.* FROM concepts c
                    JOIN concept_fts f ON c.rowid = f.rowid
                    WHERE concept_fts MATCH ? AND c.role = ?
                    ORDER BY f.rank
                    LIMIT ?
                """, (query, role, limit))
            else:
                cursor.execute("""
                    SELECT c.* FROM concepts c
                    JOIN concept_fts f ON c.rowid = f.rowid
                    WHERE concept_fts MATCH ?
                    ORDER BY f.rank
                    LIMIT ?
                """, (query, limit))
            cols = [desc[0] for desc in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception:
            # FTS match syntax error -- fallback to LIKE
            query_lower = query.lower()
            q = f"%{query_lower}%"
            if role:
                cursor.execute(
                    "SELECT * FROM concepts WHERE LOWER(content) LIKE ? AND role = ? "
                    "ORDER BY processed_time DESC LIMIT ?",
                    (q, role, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM concepts WHERE LOWER(content) LIKE ? "
                    "ORDER BY processed_time DESC LIMIT ?",
                    (q, limit)
                )
            cols = [desc[0] for desc in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def search_concepts_by_similarity(self, query_text: str, role: str = None,
                                       threshold: float = 0.5, max_results: int = 10) -> List[dict]:
        """语义相似度搜索（需要 embedding 支撑）。

        当前回退到 BM25；后续接入 embedding 向量后替换为余弦相似度搜索。
        """
        return self.search_concepts_by_bm25(query_text, role=role, limit=max_results)

    def get_concept_neighbors(self, family_id: str, max_depth: int = 1) -> List[dict]:
        """获取概念的邻居（无论 role）。

        - entity: 返回关联的 relation（通过 connects 字段包含该 entity family_id 的任意版本的 absolute_id）
        - relation: 返回它连接的 entity
        - observation: 返回它 MENTIONS 的所有 concept
        """
        concept = self.get_concept_by_family_id(family_id)
        if not concept:
            return []
        role = concept.get('role', 'entity')
        results = []

        if role == 'entity':
            # Find all absolute_ids for this entity family
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM entities WHERE family_id = ?", (family_id,)
            )
            abs_ids = [row[0] for row in cursor.fetchall()]
            if abs_ids:
                placeholders = ','.join('?' * len(abs_ids))
                # Find relations that connect to this entity
                cursor.execute(f"""
                    SELECT DISTINCT c.family_id FROM concepts c, json_each(c.connects) je
                    WHERE c.role = 'relation' AND je.value IN ({placeholders})
                """, abs_ids)
                rel_fids = list(set(r[0] for r in cursor.fetchall()))
                if rel_fids:
                    rel_concepts = self.get_concepts_by_family_ids(rel_fids)
                    results.extend(rel_concepts.values())

        elif role == 'relation':
            connects = concept.get('connects', '')
            if connects:
                try:
                    abs_ids = json.loads(connects) if isinstance(connects, str) else connects
                    # Resolve absolute_ids to family_ids
                    entity_fids = set()
                    conn = self._get_conn()
                    cursor = conn.cursor()
                    for aid in abs_ids:
                        cursor.execute(
                            "SELECT family_id FROM concepts WHERE id = ?", (aid,)
                        )
                        row = cursor.fetchone()
                        if row:
                            entity_fids.add(row[0])
                    if entity_fids:
                        ent_concepts = self.get_concepts_by_family_ids(list(entity_fids))
                        results.extend(ent_concepts.values())
                except (json.JSONDecodeError, Exception):
                    pass

        elif role == 'observation':
            # Get concepts mentioned by this episode
            abs_id = concept['id']
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT target_absolute_id FROM episode_mentions
                WHERE episode_id = ?
            """, (abs_id,))
            target_abs_ids = list(set(r[0] for r in cursor.fetchall()))
            # Resolve to family_ids
            if target_abs_ids:
                placeholders = ','.join('?' * len(target_abs_ids))
                cursor.execute(f"""
                    SELECT family_id, id FROM concepts WHERE id IN ({placeholders})
                """, target_abs_ids)
                fid_set = set()
                for row in cursor.fetchall():
                    fid_set.add(row[0])
                if fid_set:
                    mentioned = self.get_concepts_by_family_ids(list(fid_set))
                    results.extend(mentioned.values())

        return results

    def get_concept_provenance(self, family_id: str) -> List[dict]:
        """溯源：返回所有提及此概念的 observation。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM entities WHERE family_id = ?", (family_id,)
        )
        abs_ids = [row[0] for row in cursor.fetchall()]
        if not abs_ids:
            return []
        placeholders = ','.join('?' * len(abs_ids))
        cursor.execute(f"""
            SELECT DISTINCT ep.id, ep.content, ep.event_time, ep.source_document
            FROM episodes ep
            JOIN episode_mentions em ON ep.id = em.episode_id
            WHERE em.target_absolute_id IN ({placeholders})
            ORDER BY ep.event_time DESC
        """, abs_ids)
        return [
            {"episode_id": row[0], "content": row[1] or "",
             "event_time": row[2] or "", "source_document": row[3] or ""}
            for row in cursor.fetchall()
        ]

    def get_concept_mentions(self, family_id: str) -> List[dict]:
        """获取提及此概念的所有 Episode。"""
        return self.get_concept_provenance(family_id)

    def get_episode_concepts(self, episode_id: str) -> List[dict]:
        """获取 Episode 提及的所有概念（entity + relation）。"""
        return self.get_episode_entities(episode_id)

    def traverse_concepts(self, start_family_ids: List[str], max_depth: int = 2) -> dict:
        """BFS 遍历概念图。"""
        visited = set()
        queue = list(start_family_ids)
        all_concepts = {}
        all_relations_info = []

        for _ in range(max_depth):
            next_queue = []
            for fid in queue:
                if fid in visited:
                    continue
                visited.add(fid)
                concept = self.get_concept_by_family_id(fid)
                if not concept:
                    continue
                all_concepts[fid] = concept
                neighbors = self.get_concept_neighbors(fid)
                for n in neighbors:
                    nfid = n.get('family_id', '')
                    if nfid and nfid not in visited:
                        next_queue.append(nfid)
                        # Track relation connections
                        if n.get('role') == 'relation' and n.get('connects'):
                            all_relations_info.append({
                                "family_id": nfid,
                                "connects": n.get('connects', ''),
                                "content": n.get('content', ''),
                            })
            queue = next_queue

        return {
            "concepts": all_concepts,
            "relations": all_relations_info,
            "visited_count": len(visited),
        }

    def list_concepts(self, role: str = None, limit: int = 50, offset: int = 0) -> List[dict]:
        """列出概念（分页 + 可选 role 过滤）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        if role:
            cursor.execute(
                "SELECT * FROM concepts WHERE role = ? ORDER BY processed_time DESC LIMIT ? OFFSET ?",
                (role, limit, offset)
            )
        else:
            cursor.execute(
                "SELECT * FROM concepts ORDER BY processed_time DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def count_concepts(self, role: str = None) -> int:
        """统计概念数量。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        if role:
            cursor.execute("SELECT COUNT(*) FROM concepts WHERE role = ?", (role,))
        else:
            cursor.execute("SELECT COUNT(*) FROM concepts")
        return cursor.fetchone()[0]
