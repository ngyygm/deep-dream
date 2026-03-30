"""
存储层：SQLite数据库 + Markdown文件存储
"""
import sqlite3
import threading
import os
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

from ..models import MemoryCache, Entity, Relation
from ..utils import clean_markdown_code_blocks, wprint


class StorageManager:
    """存储管理器"""

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
        self.cache_dir = self.storage_path / "memory_caches"
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

        # 初始化数据库
        self._init_database()

        # 自动迁移旧目录结构
        self._migrate_storage()

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
                entity_id TEXT NOT NULL,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                event_time TEXT NOT NULL,
                processed_time TEXT NOT NULL,
                memory_cache_id TEXT NOT NULL,
                source_document TEXT DEFAULT '',
                embedding BLOB
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                relation_id TEXT NOT NULL,
                entity1_absolute_id TEXT NOT NULL,
                entity2_absolute_id TEXT NOT NULL,
                content TEXT NOT NULL,
                event_time TEXT NOT NULL,
                processed_time TEXT NOT NULL,
                memory_cache_id TEXT NOT NULL,
                source_document TEXT DEFAULT '',
                embedding BLOB
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS entity_redirects (
                source_entity_id TEXT PRIMARY KEY,
                target_entity_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_entity_id ON entities(entity_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_entity_event_time ON entities(event_time)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_entity_processed_time ON entities(processed_time)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_relation_id ON relations(relation_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_relation_entities ON relations(entity1_absolute_id, entity2_absolute_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_relation_event_time ON relations(event_time)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_relation_processed_time ON relations(processed_time)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_entity_redirect_target ON entity_redirects(target_entity_id)")
        # 为旧库自动添加缺失的 source_document 列（幂等）
        self._ensure_column(c, "entities", "source_document", "TEXT DEFAULT ''")
        self._ensure_column(c, "relations", "source_document", "TEXT DEFAULT ''")
        self._ensure_column(c, "entities", "summary", "TEXT")
        self._ensure_column(c, "entities", "attributes", "TEXT")
        self._ensure_column(c, "entities", "confidence", "REAL")
        self._ensure_column(c, "relations", "summary", "TEXT")
        self._ensure_column(c, "relations", "attributes", "TEXT")
        self._ensure_column(c, "relations", "confidence", "REAL")
        self._ensure_column(c, "relations", "provenance", "TEXT")
        # BM25 全文搜索虚拟表
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS entity_fts USING fts5(name, content, entity_id UNINDEXED)
        """)
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS relation_fts USING fts5(content, relation_id UNINDEXED)
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
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
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
                        except Exception:
                            pass
                    if lines:
                        queue_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
                        wprint(f"[迁移] {len(old_json_files)} 个旧任务文件 → queue.jsonl（{migrated} 个未完成）")
                    # 清理旧的独立 JSON 文件
                    for jf in old_json_files:
                        try:
                            jf.unlink()
                        except Exception:
                            pass
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
                    except Exception:
                        pass

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

    def _row_to_entity(self, row) -> Entity:
        """从数据库行构造 Entity 对象，自动适配新旧 schema。"""
        return Entity(
            absolute_id=row[0],
            entity_id=row[1],
            name=row[2],
            content=row[3],
            event_time=self._safe_parse_datetime(row[4]),
            processed_time=self._safe_parse_datetime(row[5]),
            memory_cache_id=row[6],
            source_document=row[7] if len(row) > 7 else '',
            embedding=row[8] if len(row) > 8 else None,
            summary=row[9] if len(row) > 9 else None,
            attributes=row[10] if len(row) > 10 else None,
            confidence=float(row[11]) if len(row) > 11 and row[11] is not None else None,
        )

    def _row_to_relation(self, row) -> Relation:
        """从数据库行构造 Relation 对象，自动适配新旧 schema。"""
        return Relation(
            absolute_id=row[0],
            relation_id=row[1],
            entity1_absolute_id=row[2] or "",
            entity2_absolute_id=row[3] or "",
            content=row[4],
            event_time=self._safe_parse_datetime(row[5]),
            processed_time=self._safe_parse_datetime(row[6]),
            memory_cache_id=row[7],
            source_document=row[8] if len(row) > 8 else '',
            embedding=row[9] if len(row) > 9 else None,
            summary=row[10] if len(row) > 10 else None,
            attributes=row[11] if len(row) > 11 else None,
            confidence=float(row[12]) if len(row) > 12 and row[12] is not None else None,
            provenance=row[13] if len(row) > 13 else None,
        )

    def _init_database(self):
        """初始化SQLite数据库（使用独立连接，此时线程池尚未启用）。"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 创建实体表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                event_time TEXT NOT NULL,
                processed_time TEXT NOT NULL,
                memory_cache_id TEXT NOT NULL,
                source_document TEXT DEFAULT '',
                embedding BLOB
            )
        """)

        # 创建关系表（只使用绝对ID，无向关系）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                relation_id TEXT NOT NULL,
                entity1_absolute_id TEXT NOT NULL,
                entity2_absolute_id TEXT NOT NULL,
                content TEXT NOT NULL,
                event_time TEXT NOT NULL,
                processed_time TEXT NOT NULL,
                memory_cache_id TEXT NOT NULL,
                source_document TEXT DEFAULT '',
                embedding BLOB
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_redirects (
                source_entity_id TEXT PRIMARY KEY,
                target_entity_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_id ON entities(entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_event_time ON entities(event_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_processed_time ON entities(processed_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relation_id ON relations(relation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relation_entities ON relations(entity1_absolute_id, entity2_absolute_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relation_event_time ON relations(event_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relation_processed_time ON relations(processed_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_redirect_target ON entity_redirects(target_entity_id)")

        # 唯一索引：防止并行创建时产生重复版本
        try:
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_unique ON entities(entity_id, processed_time)")
        except sqlite3.OperationalError:
            pass  # 索引已存在或存在重复数据，忽略
        try:
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_relation_unique ON relations(relation_id, processed_time)")
        except sqlite3.OperationalError:
            pass

        # 旧库迁移：自动添加 source_document 列（幂等）
        self._ensure_column(cursor, "entities", "source_document", "TEXT DEFAULT ''")
        self._ensure_column(cursor, "relations", "source_document", "TEXT DEFAULT ''")

        # 旧库迁移：为 Phase 3 添加 valid_at / invalid_at 列（幂等）
        self._ensure_column(cursor, "entities", "valid_at", "TEXT")
        self._ensure_column(cursor, "entities", "invalid_at", "TEXT")
        self._ensure_column(cursor, "relations", "valid_at", "TEXT")
        self._ensure_column(cursor, "relations", "invalid_at", "TEXT")

        # Phase A: 摘要、属性、置信度
        self._ensure_column(cursor, "entities", "summary", "TEXT")
        self._ensure_column(cursor, "entities", "attributes", "TEXT")
        self._ensure_column(cursor, "entities", "confidence", "REAL")
        self._ensure_column(cursor, "relations", "summary", "TEXT")
        self._ensure_column(cursor, "relations", "attributes", "TEXT")
        self._ensure_column(cursor, "relations", "confidence", "REAL")
        self._ensure_column(cursor, "relations", "provenance", "TEXT")

        # Phase C: Episode mentions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episode_mentions (
                episode_id TEXT NOT NULL,
                entity_absolute_id TEXT NOT NULL,
                mention_context TEXT DEFAULT '',
                PRIMARY KEY (episode_id, entity_absolute_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episode_mentions_entity ON episode_mentions(entity_absolute_id)")

        # Phase E: Dream logs
        cursor.execute("""
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
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS entity_fts USING fts5(name, content, entity_id UNINDEXED)
        """)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS relation_fts USING fts5(content, relation_id UNINDEXED)
        """)

        conn.commit()
        conn.close()

    def _resolve_entity_id_with_cursor(self, cursor, entity_id: str) -> str:
        """沿 redirect 链解析到当前 canonical entity_id。"""
        current_id = (entity_id or "").strip()
        if not current_id:
            return ""
        seen: Set[str] = set()
        while current_id and current_id not in seen:
            seen.add(current_id)
            cursor.execute(
                "SELECT target_entity_id FROM entity_redirects WHERE source_entity_id = ?",
                (current_id,),
            )
            row = cursor.fetchone()
            if not row or not row[0] or row[0] == current_id:
                break
            current_id = row[0]
        return current_id

    def resolve_entity_id(self, entity_id: str) -> str:
        """解析 entity_id 到当前 canonical id；不存在映射时原样返回。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        return self._resolve_entity_id_with_cursor(cursor, entity_id)

    def register_entity_redirect(self, source_entity_id: str, target_entity_id: str) -> str:
        """登记旧 entity_id 到 canonical entity_id 的映射，支持链式合并。"""
        source_id = (source_entity_id or "").strip()
        target_id = (target_entity_id or "").strip()
        if not source_id or not target_id:
            return target_id
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            canonical_target = self._resolve_entity_id_with_cursor(cursor, target_id)
            if not canonical_target:
                canonical_target = target_id
            canonical_source = self._resolve_entity_id_with_cursor(cursor, source_id)
            if canonical_source == canonical_target:
                return canonical_target
            cursor.execute(
                """
                INSERT INTO entity_redirects (source_entity_id, target_entity_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(source_entity_id) DO UPDATE SET
                    target_entity_id = excluded.target_entity_id,
                    updated_at = excluded.updated_at
                """,
                (source_id, canonical_target, datetime.now().isoformat()),
            )
            conn.commit()
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

    # ========== MemoryCache 操作 ==========
    
    def save_memory_cache(self, cache: MemoryCache, text: str = "", document_path: str = "", doc_hash: str = "") -> str:
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

        return cache.absolute_id
    
    def load_memory_cache(self, cache_id: str) -> Optional[MemoryCache]:
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

        return MemoryCache(
            absolute_id=metadata.get("absolute_id") or metadata.get("id"),
            content=md_content,
            event_time=self._safe_parse_datetime(metadata.get("event_time"), datetime.now()),
            source_document=metadata.get("source_document") or metadata.get("doc_name", ""),
            activity_type=metadata.get("activity_type"),
        )

    def get_entity_count(self) -> int:
        """返回实体总数（去重 entity_id）。"""
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(DISTINCT entity_id) AS cnt FROM entities").fetchone()
            return row["cnt"] if row else 0

    def get_relation_count(self) -> int:
        """返回关系总数（去重 relation_id）。"""
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(DISTINCT relation_id) AS cnt FROM relations").fetchone()
            return row["cnt"] if row else 0

    def delete_memory_cache(self, cache_id: str) -> int:
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

    def get_latest_memory_cache(self, activity_type: Optional[str] = None) -> Optional[MemoryCache]:
        """获取最新的记忆缓存"""
        cache_files = self._iter_cache_meta_files()
        if not cache_files:
            return None

        latest_cache = None
        latest_time = None

        for cache_file in cache_files:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                continue

            if activity_type and metadata.get("activity_type") != activity_type:
                continue

            cache_id = metadata.get("absolute_id") or metadata.get("id")
            if not cache_id:
                continue

            cache_time = self._safe_parse_datetime(metadata.get("event_time"), datetime.now())
            if latest_time is None or cache_time > latest_time:
                latest_time = cache_time
                latest_cache = self.load_memory_cache(cache_id)

        return latest_cache

    def get_latest_memory_cache_metadata(self, activity_type: Optional[str] = None) -> Optional[Dict]:
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
            except Exception:
                continue

            if activity_type and metadata.get("activity_type") != activity_type:
                continue

            cache_time = self._safe_parse_datetime(metadata.get("event_time"), datetime.now())
            if latest_time is None or cache_time > latest_time:
                latest_time = cache_time
                latest_metadata = metadata

        return latest_metadata

    def search_memory_caches_by_bm25(self, query: str, limit: int = 20) -> List[MemoryCache]:
        """简单文本搜索记忆缓存（遍历所有缓存，按内容匹配排序）。

        注意：这不是真正的 BM25，因为记忆缓存使用文件存储而非 SQLite FTS。
        对于生产环境的大规模数据，应使用向量搜索或专用全文索引。
        """
        if not query:
            return []
        query_lower = query.lower()
        results = []
        for cache_file in self._iter_cache_meta_files():
            try:
                cache = self.load_memory_cache(
                    cache_file.stem if cache_file.suffix == ".json" else
                    (cache_file.parent.name if cache_file.name == "meta.json" else cache_file.stem)
                )
            except Exception:
                continue
            if cache is None:
                continue
            # 简单的子串匹配评分
            content_lower = (cache.content or "").lower()
            if query_lower in content_lower:
                score = content_lower.count(query_lower)
                results.append((score, cache))
        results.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in results[:limit]]

    def find_cache_by_doc_hash(self, doc_hash: str, document_path: str = "") -> Optional[MemoryCache]:
        """通过 doc_hash 查找已存在的缓存（断点续传复用）。

        Args:
            doc_hash: 12位 MD5 hash
            document_path: 可选，用于精确匹配同一文档的缓存

        Returns:
            找到的 MemoryCache，未找到返回 None
        """
        if not doc_hash or not self.docs_dir.is_dir():
            return None
        matches = list(self.docs_dir.glob(f"*_{doc_hash}/meta.json"))
        for meta_path in matches:
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                continue
            if document_path and metadata.get("document_path") != document_path:
                continue
            cache_id = metadata.get("absolute_id")
            if cache_id:
                return self.load_memory_cache(cache_id)
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
            except Exception:
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
        except Exception:
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
        except Exception:
            pass
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
        # 计算embedding（无需锁，纯计算）
        embedding_blob = self._compute_entity_embedding(entity)
        entity.embedding = embedding_blob

        with self._write_lock:
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO entities (id, entity_id, name, content, event_time, processed_time, memory_cache_id, source_document, embedding, valid_at, summary, attributes, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.absolute_id,
                    entity.entity_id,
                    entity.name,
                    entity.content,
                    entity.event_time.isoformat(),
                    entity.processed_time.isoformat(),
                    entity.memory_cache_id,
                    entity.source_document,
                    embedding_blob,
                    (entity.valid_at or entity.event_time).isoformat(),
                    getattr(entity, 'summary', None),
                    getattr(entity, 'attributes', None),
                    getattr(entity, 'confidence', None),
                ))
                conn.commit()
                # 同步写入 FTS 表
                try:
                    cursor.execute("""
                        INSERT INTO entity_fts(rowid, name, content, entity_id)
                        VALUES (?, ?, ?, ?)
                    """, (entity.absolute_id, entity.name, entity.content, entity.entity_id))
                    conn.commit()
                except Exception:
                    pass  # FTS 写入失败不影响主流程
                # 设置旧版本 invalid_at
                try:
                    cursor.execute("""
                        UPDATE entities SET invalid_at = ?
                        WHERE entity_id = ? AND id != ? AND invalid_at IS NULL
                    """, (entity.event_time.isoformat(), entity.entity_id, entity.absolute_id))
                    conn.commit()
                except Exception:
                    pass
            except Exception:
                conn.rollback()
                raise

    def bulk_save_entities(self, entities: List[Entity]):
        """批量保存实体，使用批量 embedding 与单事务写入。"""
        if not entities:
            return

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
                except Exception:
                    embedding_blob = None
            entity.embedding = embedding_blob
            rows.append((
                entity.absolute_id,
                entity.entity_id,
                entity.name,
                entity.content,
                entity.event_time.isoformat(),
                entity.processed_time.isoformat(),
                entity.memory_cache_id,
                entity.source_document,
                embedding_blob,
                (entity.valid_at or entity.event_time).isoformat(),
            ))

        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR IGNORE INTO entities (id, entity_id, name, content, event_time, processed_time, memory_cache_id, source_document, embedding, valid_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            # 同步写入 FTS 表
            try:
                fts_rows = [(e.absolute_id, e.name, e.content, e.entity_id) for e in entities]
                cursor.executemany("""
                    INSERT OR REPLACE INTO entity_fts(rowid, name, content, entity_id)
                    VALUES (?, ?, ?, ?)
                """, fts_rows)
                conn.commit()
            except Exception:
                pass

    def get_entity_by_entity_id(self, entity_id: str) -> Optional[Entity]:
        """根据entity_id获取最新版本的实体"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return None
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, entity_id, name, content, event_time, processed_time, memory_cache_id, source_document, embedding, summary, attributes, confidence
            FROM entities
            WHERE entity_id = ?
            ORDER BY processed_time DESC
            LIMIT 1
        """, (entity_id,))

        row = cursor.fetchone()

        if row is None:
            return None

        return Entity(
            absolute_id=row[0],
            entity_id=row[1],
            name=row[2],
            content=row[3],
            event_time=self._safe_parse_datetime(row[4]),
            processed_time=self._safe_parse_datetime(row[5]),
            memory_cache_id=row[6],
            source_document=row[7] if len(row) > 7 else '',
            embedding=row[8] if len(row) > 8 else None,
            summary=row[9] if len(row) > 9 else None,
            attributes=row[10] if len(row) > 10 else None,
            confidence=row[11] if len(row) > 11 else None,
        )
    
    # get_entity_by_id 是 get_entity_by_entity_id 的别名，兼容 pipeline 层调用
    get_entity_by_id = get_entity_by_entity_id

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """根据entity_id获取实体（别名，等同于get_entity_by_entity_id）"""
        return self.get_entity_by_entity_id(entity_id)

    def get_entity_by_absolute_id(self, absolute_id: str) -> Optional[Entity]:
        """根据绝对ID获取实体"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, entity_id, name, content, event_time, processed_time, memory_cache_id, source_document, embedding
            FROM entities
            WHERE id = ?
        """, (absolute_id,))

        row = cursor.fetchone()

        if row is None:
            return None

        return Entity(
            absolute_id=row[0],
            entity_id=row[1],
            name=row[2],
            content=row[3],
            event_time=self._safe_parse_datetime(row[4]),
            processed_time=self._safe_parse_datetime(row[5]),
            memory_cache_id=row[6],
            source_document=row[7] if len(row) > 7 else '',
            embedding=row[8] if len(row) > 8 else None
        )

    def get_relation_by_absolute_id(self, relation_absolute_id: str) -> Optional[Relation]:
        """根据关系行的主键 id（绝对ID）获取单条关系"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, relation_id, entity1_absolute_id, entity2_absolute_id, content, event_time, processed_time, memory_cache_id, source_document, embedding
            FROM relations
            WHERE id = ?
        """, (relation_absolute_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return Relation(
            absolute_id=row[0],
            relation_id=row[1],
            entity1_absolute_id=row[2] or "",
            entity2_absolute_id=row[3] or "",
            content=row[4],
            event_time=self._safe_parse_datetime(row[5]),
            processed_time=self._safe_parse_datetime(row[6]),
            memory_cache_id=row[7],
            source_document=row[8] if len(row) > 8 else '',
            embedding=row[9] if len(row) > 9 else None
        )
    
    def get_relation_by_relation_id(self, relation_id: str) -> Optional[Relation]:
        """根据relation_id获取最新版本的关系"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, relation_id, entity1_absolute_id, entity2_absolute_id, content, event_time, processed_time, memory_cache_id, source_document, embedding
            FROM relations
            WHERE relation_id = ?
            ORDER BY processed_time DESC
            LIMIT 1
        """, (relation_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return Relation(
            absolute_id=row[0],
            relation_id=row[1],
            entity1_absolute_id=row[2],
            entity2_absolute_id=row[3],
            content=row[4],
            event_time=self._safe_parse_datetime(row[5]),
            processed_time=self._safe_parse_datetime(row[6]),
            memory_cache_id=row[7],
            source_document=row[8] if len(row) > 8 else '',
            embedding=row[9] if len(row) > 9 else None
        )

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

    def get_entity_version_at_time(self, entity_id: str, time_point: datetime) -> Optional[Entity]:
        """获取实体在指定时间点的版本（该时间点之前或等于该时间点的最新版本）"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return None
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, entity_id, name, content, event_time, processed_time, memory_cache_id, source_document, embedding, valid_at, invalid_at
            FROM entities
            WHERE entity_id = ? AND event_time <= ?
            ORDER BY processed_time DESC
            LIMIT 1
        """, (entity_id, time_point.isoformat()))

        row = cursor.fetchone()

        if row is None:
            return None

        return Entity(
            absolute_id=row[0],
            entity_id=row[1],
            name=row[2],
            content=row[3],
            event_time=self._safe_parse_datetime(row[4]),
            processed_time=self._safe_parse_datetime(row[5]),
            memory_cache_id=row[6],
            source_document=row[7] if len(row) > 7 else '',
            embedding=row[8] if len(row) > 8 else None,
            valid_at=self._safe_parse_datetime(row[9]) if len(row) > 9 else None,
            invalid_at=self._safe_parse_datetime(row[10]) if len(row) > 10 else None
        )
    
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
        except Exception:
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
        except Exception:
            return None
    
    def get_entity_versions(self, entity_id: str) -> List[Entity]:
        """获取实体的所有版本"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, entity_id, name, content, event_time, processed_time, memory_cache_id, source_document, embedding
            FROM entities
            WHERE entity_id = ?
            ORDER BY processed_time DESC
        """, (entity_id,))

        rows = cursor.fetchall()

        return [
            Entity(
                absolute_id=row[0],
                entity_id=row[1],
                name=row[2],
                content=row[3],
                event_time=datetime.fromisoformat(row[4]),
                processed_time=datetime.fromisoformat(row[5]),
                memory_cache_id=row[6],
                source_document=row[7] if len(row) > 7 else '',
                embedding=row[8] if len(row) > 8 else None
            )
            for row in rows
        ]
    
    def _get_entities_with_embeddings(self) -> List[tuple]:
        """
        获取所有实体的最新版本及其embedding
        
        Returns:
            List of (Entity, embedding_array) tuples, embedding_array为None表示没有embedding
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 获取每个entity_id的最新版本及其embedding
        cursor.execute("""
            SELECT id, entity_id, name, content, event_time, processed_time, memory_cache_id, source_document, embedding
            FROM entities e1
            WHERE e1.processed_time = (
                SELECT MAX(e2.processed_time)
                FROM entities e2
                WHERE e2.entity_id = e1.entity_id
            )
        """)

        results = []
        for row in cursor.fetchall():
            # 解析embedding
            embedding_array = None
            if len(row) > 8 and row[8] is not None:
                try:
                    embedding_array = np.frombuffer(row[8], dtype=np.float32)
                except (ValueError, TypeError):
                    embedding_array = None
            entity = Entity(
                absolute_id=row[0],
                entity_id=row[1],
                name=row[2],
                content=row[3],
                event_time=datetime.fromisoformat(row[4]),
                processed_time=datetime.fromisoformat(row[5]),
                memory_cache_id=row[6],
                source_document=row[7] if len(row) > 7 else '',
                embedding=row[8] if len(row) > 8 else None
            )
            results.append((entity, embedding_array))
        
        return results

    def get_latest_entities_projection(self, content_snippet_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取最新实体投影，供窗口级批量候选生成使用。"""
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

    def search_entities_by_bm25(self, query: str, limit: int = 20) -> List[Entity]:
        """BM25 全文搜索实体。"""
        if not query:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        # FTS5 BM25 搜索，按 bm25 排序
        cursor.execute("""
            SELECT e.id, e.entity_id, e.name, e.content, e.event_time, e.processed_time,
                   e.memory_cache_id, e.source_document, e.embedding,
                   fts.rank AS bm25_score
            FROM entity_fts AS fts
            JOIN entities AS e ON e.id = fts.rowid
            WHERE entity_fts MATCH ?
            ORDER BY fts.rank
            LIMIT ?
        """, (query, limit))

        entities = []
        for row in cursor.fetchall():
            entities.append(Entity(
                absolute_id=row[0],
                entity_id=self.resolve_entity_id(row[1]),
                name=row[2],
                content=row[3],
                event_time=self._safe_parse_datetime(row[4]),
                processed_time=self._safe_parse_datetime(row[5]),
                memory_cache_id=row[6],
                source_document=row[7] or '',
                embedding=row[8],
            ))
        return entities

    def search_relations_by_bm25(self, query: str, limit: int = 20) -> List[Relation]:
        """BM25 全文搜索关系。"""
        if not query:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT r.id, r.relation_id, r.entity1_absolute_id, r.entity2_absolute_id,
                   r.content, r.event_time, r.processed_time,
                   r.memory_cache_id, r.source_document, r.embedding,
                   fts.rank AS bm25_score
            FROM relation_fts AS fts
            JOIN relations AS r ON r.id = fts.rowid
            WHERE relation_fts MATCH ?
            ORDER BY fts.rank
            LIMIT ?
        """, (query, limit))

        relations = []
        for row in cursor.fetchall():
            relations.append(Relation(
                absolute_id=row[0],
                relation_id=row[1],
                entity1_absolute_id=row[2],
                entity2_absolute_id=row[3],
                content=row[4],
                event_time=self._safe_parse_datetime(row[5]),
                processed_time=self._safe_parse_datetime(row[6]),
                memory_cache_id=row[7],
                source_document=row[8] or '',
                embedding=row[9],
            ))
        return relations

    def search_entities_by_similarity(self, query_name: str, query_content: Optional[str] = None,
                                     threshold: float = 0.7, max_results: int = 10, 
                                     content_snippet_length: int = 50,
                                     text_mode: Literal["name_only", "content_only", "name_and_content"] = "name_and_content",
                                     similarity_method: Literal["embedding", "text", "jaccard", "bleu"] = "embedding") -> List[Entity]:
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
        # 获取所有实体及其embedding
        entities_with_embeddings = self._get_entities_with_embeddings()
        
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
        
        # 返回实体列表（去重，每个entity_id只保留一个，并限制最大数量）
        entities = []
        seen_ids = set()
        for entity, _ in scored_entities:
            if entity.entity_id not in seen_ids:
                entities.append(entity)
                seen_ids.add(entity.entity_id)
                # 达到最大数量后停止
                if len(entities) >= max_results:
                    break
        
        return entities
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """计算Jaccard相似度（基于字符集合）"""
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        if union == 0:
            return 0.0
        return intersection / union
    
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
        
        # 返回实体列表（去重，每个entity_id只保留一个，并限制最大数量）
        entities = []
        seen_ids = set()
        for entity, _ in scored_entities:
            if entity.entity_id not in seen_ids:
                entities.append(entity)
                seen_ids.add(entity.entity_id)
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
        # 计算embedding（无需锁）
        embedding_blob = self._compute_relation_embedding(relation)
        relation.embedding = embedding_blob

        with self._write_lock:
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO relations (id, relation_id, entity1_absolute_id, entity2_absolute_id, content, event_time, processed_time, memory_cache_id, source_document, embedding, valid_at, summary, attributes, confidence, provenance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    relation.absolute_id,
                    relation.relation_id,
                    relation.entity1_absolute_id,
                    relation.entity2_absolute_id,
                    relation.content,
                    relation.event_time.isoformat(),
                    relation.processed_time.isoformat(),
                    relation.memory_cache_id,
                    relation.source_document,
                    embedding_blob,
                    (relation.valid_at or relation.event_time).isoformat(),
                    getattr(relation, 'summary', None),
                    getattr(relation, 'attributes', None),
                    getattr(relation, 'confidence', None),
                    getattr(relation, 'provenance', None),
                ))
                conn.commit()
                # 同步写入 FTS 表
                try:
                    cursor.execute("""
                        INSERT INTO relation_fts(rowid, content, relation_id)
                        VALUES (?, ?, ?)
                    """, (relation.absolute_id, relation.content, relation.relation_id))
                    conn.commit()
                except Exception:
                    pass  # FTS 写入失败不影响主流程
                # 设置旧版本 invalid_at
                try:
                    cursor.execute("""
                        UPDATE relations SET invalid_at = ?
                        WHERE relation_id = ? AND id != ? AND invalid_at IS NULL
                    """, (relation.event_time.isoformat(), relation.relation_id, relation.absolute_id))
                    conn.commit()
                except Exception:
                    pass
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
                except Exception:
                    embedding_blob = None
            relation.embedding = embedding_blob
            rows.append((
                relation.absolute_id,
                relation.relation_id,
                relation.entity1_absolute_id,
                relation.entity2_absolute_id,
                relation.content,
                relation.event_time.isoformat(),
                relation.processed_time.isoformat(),
                relation.memory_cache_id,
                relation.source_document,
                embedding_blob,
                (relation.valid_at or relation.event_time).isoformat(),
            ))

        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO relations (id, relation_id, entity1_absolute_id, entity2_absolute_id, content, event_time, processed_time, memory_cache_id, source_document, embedding, valid_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
    
    def get_relations_by_entities(self, from_entity_id: str, to_entity_id: str) -> List[Relation]:
        """根据两个实体ID获取所有关系（通过entity_id查找，内部转换为绝对ID查询）

        每个 relation_id 只返回最新版本（与 get_entity_relations 保持一致的去重逻辑）。
        """
        from_entity_id = self.resolve_entity_id(from_entity_id)
        to_entity_id = self.resolve_entity_id(to_entity_id)
        if not from_entity_id or not to_entity_id:
            return []
        # 先通过entity_id获取最新版本的绝对ID
        from_entity = self.get_entity_by_entity_id(from_entity_id)
        to_entity = self.get_entity_by_entity_id(to_entity_id)

        if not from_entity or not to_entity:
            return []

        # 获取所有具有相同entity_id的实体的绝对ID
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id FROM entities WHERE entity_id = ?
        """, (from_entity_id,))
        from_absolute_ids = [row[0] for row in cursor.fetchall()]

        cursor.execute("""
            SELECT id FROM entities WHERE entity_id = ?
        """, (to_entity_id,))
        to_absolute_ids = [row[0] for row in cursor.fetchall()]

        if not from_absolute_ids or not to_absolute_ids:
            return []

        # 查询关系（无向关系，考虑两个方向）
        # 每个 relation_id 只返回最新版本（INNER JOIN MAX(processed_time) 去重）
        placeholders_from = ','.join(['?'] * len(from_absolute_ids))
        placeholders_to = ','.join(['?'] * len(to_absolute_ids))

        cursor.execute(f"""
            SELECT r1.id, r1.relation_id, r1.entity1_absolute_id, r1.entity2_absolute_id,
                   r1.content, r1.event_time, r1.processed_time, r1.memory_cache_id, r1.source_document, r1.embedding
            FROM relations r1
            INNER JOIN (
                SELECT relation_id, MAX(processed_time) as max_time
                FROM relations
                WHERE (entity1_absolute_id IN ({placeholders_from}) AND entity2_absolute_id IN ({placeholders_to}))
                   OR (entity1_absolute_id IN ({placeholders_to}) AND entity2_absolute_id IN ({placeholders_from}))
                GROUP BY relation_id
            ) r2 ON r1.relation_id = r2.relation_id
                AND r1.processed_time = r2.max_time
                AND ((r1.entity1_absolute_id IN ({placeholders_from}) AND r1.entity2_absolute_id IN ({placeholders_to}))
                  OR (r1.entity1_absolute_id IN ({placeholders_to}) AND r1.entity2_absolute_id IN ({placeholders_from})))
            ORDER BY r1.processed_time DESC
        """, (from_absolute_ids + to_absolute_ids + to_absolute_ids + from_absolute_ids)
              + (from_absolute_ids + to_absolute_ids + to_absolute_ids + from_absolute_ids))

        rows = cursor.fetchall()

        return [
            Relation(
                absolute_id=row[0],
                relation_id=row[1],
                entity1_absolute_id=row[2] or "",
                entity2_absolute_id=row[3] or "",
                content=row[4],
                event_time=datetime.fromisoformat(row[5]),
                processed_time=datetime.fromisoformat(row[6]),
                memory_cache_id=row[7],
                source_document=row[8] if len(row) > 8 else '',
                embedding=row[9] if len(row) > 9 else None
            )
            for row in rows
        ]

    def get_relations_by_entity_pairs(self, entity_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], List[Relation]]:
        """批量获取多个实体对的关系，按无向 pair 返回。"""
        results: Dict[Tuple[str, str], List[Relation]] = {}
        for entity1_id, entity2_id in entity_pairs:
            pair_key = tuple(sorted((entity1_id, entity2_id)))
            if pair_key in results:
                continue
            results[pair_key] = self.get_relations_by_entities(pair_key[0], pair_key[1])
        return results

    def get_latest_relations_projection(self, content_snippet_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取最新关系投影，供关系批量 upsert 使用。"""
        snippet_length = (
            self.relation_content_snippet_length
            if content_snippet_length is None
            else content_snippet_length
        )
        results: List[Dict[str, Any]] = []
        for relation, embedding_array in self._get_relations_with_embeddings():
            if snippet_length is None or snippet_length <= 0:
                _csnip = relation.content
            else:
                _csnip = relation.content[:snippet_length]
            results.append({
                "relation": relation,
                "relation_id": relation.relation_id,
                "pair": tuple(sorted((relation.entity1_absolute_id, relation.entity2_absolute_id))),
                "content": relation.content,
                "content_snippet": _csnip,
                "embedding_array": embedding_array,
            })
        return results
    
    def get_relation_versions(self, relation_id: str) -> List[Relation]:
        """获取关系的所有版本"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, relation_id, entity1_absolute_id, entity2_absolute_id, content, event_time, processed_time, memory_cache_id, source_document, embedding
            FROM relations
            WHERE relation_id = ?
            ORDER BY processed_time DESC
        """, (relation_id,))

        rows = cursor.fetchall()

        return [
            Relation(
                absolute_id=row[0],
                relation_id=row[1],
                entity1_absolute_id=row[2] or "",
                entity2_absolute_id=row[3] or "",
                content=row[4],
                event_time=datetime.fromisoformat(row[5]),
                processed_time=datetime.fromisoformat(row[6]),
                memory_cache_id=row[7],
                source_document=row[8] if len(row) > 8 else '',
                embedding=row[9] if len(row) > 9 else None
            )
            for row in rows
        ]
    
    def update_relation_memory_cache_id(self, relation_id: str, memory_cache_id: str):
        """更新关系最新版本的 memory_cache_id"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            # 获取最新版本的id
            cursor.execute("""
                SELECT id FROM relations
                WHERE relation_id = ?
                ORDER BY processed_time DESC
                LIMIT 1
            """, (relation_id,))

            row = cursor.fetchone()
            if row:
                latest_id = row[0]
                cursor.execute("""
                    UPDATE relations
                    SET memory_cache_id = ?
                    WHERE id = ?
                """, (memory_cache_id, latest_id))
                conn.commit()
        
    
    def get_self_referential_relations(self) -> Dict[str, List[Dict]]:
        """获取所有自指向的关系（两端指向同一个entity_id），按entity_id分组
        
        自指向关系的定义：关系的两端实体具有相同的entity_id
        这包括两种情况：
        1. entity1_absolute_id == entity2_absolute_id（指向完全相同的版本）
        2. entity1_absolute_id 和 entity2_absolute_id 不同，但它们对应的entity_id相同
        
        Returns:
            字典，key为entity_id，value为该实体的所有自指向关系列表
            每个关系包含：id, relation_id, content, processed_time
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 查找所有自指向的关系（两端entity_id相同）
        cursor.execute("""
            SELECT r.id, r.relation_id, r.content, r.processed_time, e1.entity_id
            FROM relations r
            JOIN entities e1 ON r.entity1_absolute_id = e1.id
            JOIN entities e2 ON r.entity2_absolute_id = e2.id
            WHERE e1.entity_id = e2.entity_id
            ORDER BY e1.entity_id, r.processed_time
        """)
        
        rows = cursor.fetchall()
        
        # 按entity_id分组
        result = {}
        for row in rows:
            relation_id, relation_id_str, content, processed_time, entity_id = row
            if entity_id not in result:
                result[entity_id] = []
            result[entity_id].append({
                'id': relation_id,
                'relation_id': relation_id_str,
                'content': content,
                'processed_time': processed_time
            })
        
        return result
    
    def delete_self_referential_relations(self) -> int:
        """删除所有自指向的关系（两端指向同一个entity_id）

        Returns:
            删除的关系数量
        """
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            # 查找所有自指向的关系（两端entity_id相同）
            cursor.execute("""
                SELECT r.id FROM relations r
                JOIN entities e1 ON r.entity1_absolute_id = e1.id
                JOIN entities e2 ON r.entity2_absolute_id = e2.id
                WHERE e1.entity_id = e2.entity_id
            """)

            rows = cursor.fetchall()
            deleted_count = 0

            if rows:
                relation_ids = [row[0] for row in rows]
                placeholders = ','.join(['?' for _ in relation_ids])

                cursor.execute(f"""
                    DELETE FROM relations
                    WHERE id IN ({placeholders})
                """, relation_ids)
                deleted_count = cursor.rowcount
                conn.commit()

            return deleted_count
    
    def get_self_referential_relations_for_entity(self, entity_id: str) -> List[Dict]:
        """获取指定entity_id的自指向关系
        
        Args:
            entity_id: 实体ID
        
        Returns:
            自指向关系列表，每个关系包含：id, relation_id, content, processed_time
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 查找该entity_id的自指向关系
        cursor.execute("""
            SELECT r.id, r.relation_id, r.content, r.processed_time
            FROM relations r
            JOIN entities e1 ON r.entity1_absolute_id = e1.id
            JOIN entities e2 ON r.entity2_absolute_id = e2.id
            WHERE e1.entity_id = ? AND e2.entity_id = ?
            ORDER BY r.processed_time
        """, (entity_id, entity_id))
        
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            relation_id, relation_id_str, content, processed_time = row
            result.append({
                'id': relation_id,
                'relation_id': relation_id_str,
                'content': content,
                'processed_time': processed_time
            })
        
        return result
    
    def delete_self_referential_relations_for_entity(self, entity_id: str) -> int:
        """删除指定entity_id的自指向关系

        Args:
            entity_id: 实体ID

        Returns:
            删除的关系数量
        """
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            # 查找该entity_id的自指向关系
            cursor.execute("""
                SELECT r.id FROM relations r
                JOIN entities e1 ON r.entity1_absolute_id = e1.id
                JOIN entities e2 ON r.entity2_absolute_id = e2.id
                WHERE e1.entity_id = ? AND e2.entity_id = ?
            """, (entity_id, entity_id))

            rows = cursor.fetchall()
            deleted_count = 0

            if rows:
                relation_ids = [row[0] for row in rows]
                placeholders = ','.join(['?' for _ in relation_ids])

                cursor.execute(f"""
                    DELETE FROM relations
                    WHERE id IN ({placeholders})
                """, relation_ids)
                deleted_count = cursor.rowcount
                conn.commit()

            return deleted_count
    
    def get_all_entities(self, limit: Optional[int] = None, exclude_embedding: bool = False) -> List[Entity]:
        """获取所有实体的最新版本

        Args:
            limit: 限制返回的实体数量（按时间倒序），None表示不限制
            exclude_embedding: 是否排除 embedding 字段（前端展示等不需要 embedding 的场景应设为 True）
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        emb_col = ", e1.embedding" if not exclude_embedding else ""
        # 获取每个 entity_id 的最新版本
        query = f"""
            SELECT e1.id, e1.entity_id, e1.name, e1.content, e1.event_time, e1.processed_time, e1.memory_cache_id, e1.source_document{emb_col}
            FROM entities e1
            INNER JOIN (
                SELECT entity_id, MAX(processed_time) as max_time
                FROM entities
                GROUP BY entity_id
            ) e2 ON e1.entity_id = e2.entity_id AND e1.processed_time = e2.max_time
            ORDER BY e1.processed_time DESC
        """

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        cursor.execute(query)

        rows = cursor.fetchall()

        return [
            Entity(
                absolute_id=row[0],
                entity_id=row[1],
                name=row[2],
                content=row[3],
                event_time=datetime.fromisoformat(row[4]),
                processed_time=datetime.fromisoformat(row[5]),
                memory_cache_id=row[6],
                source_document=row[7] if len(row) > 7 and row[7] is not None else "",
                embedding=row[8] if not exclude_embedding and len(row) > 8 else None
            )
            for row in rows
        ]

    def count_unique_entities(self) -> int:
        """轻量统计：返回不重复的 entity_id 数量（不加载任何实体数据）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT entity_id) FROM entities")
        return cursor.fetchone()[0]

    def count_unique_relations(self) -> int:
        """轻量统计：返回不重复的 relation_id 数量（不加载任何关系数据）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT relation_id) FROM relations")
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

        emb_col = ", e1.embedding" if not exclude_embedding else ""
        # 获取每个 entity_id 在指定时间点之前或等于该时间点的最新版本
        query = f"""
            SELECT e1.id, e1.entity_id, e1.name, e1.content, e1.event_time, e1.processed_time, e1.memory_cache_id, e1.source_document{emb_col}
            FROM entities e1
            INNER JOIN (
                SELECT entity_id, MAX(processed_time) as max_time
                FROM entities
                WHERE event_time <= ?
                GROUP BY entity_id
            ) e2 ON e1.entity_id = e2.entity_id AND e1.processed_time = e2.max_time
            ORDER BY e1.processed_time DESC
        """

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        cursor.execute(query, (time_point.isoformat(),))

        rows = cursor.fetchall()

        return [
            Entity(
                absolute_id=row[0],
                entity_id=row[1],
                name=row[2],
                content=row[3],
                event_time=datetime.fromisoformat(row[4]),
                processed_time=datetime.fromisoformat(row[5]),
                memory_cache_id=row[6],
                source_document=row[7] if len(row) > 7 and row[7] is not None else "",
                embedding=row[8] if not exclude_embedding and len(row) > 8 else None
            )
            for row in rows
        ]
    
    def get_entity_relations(self, entity_absolute_id: str, limit: Optional[int] = None, time_point: Optional[datetime] = None) -> List[Relation]:
        """获取与指定实体相关的所有关系（作为起点或终点）
        
        Args:
            entity_absolute_id: 实体的绝对ID
            limit: 限制返回的关系数量（按时间倒序），None表示不限制
            time_point: 时间点（可选），如果提供，只返回该时间点之前或等于该时间点的关系，且每个relation_id只返回最新版本
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        if time_point:
            # 获取每个relation_id在该时间点之前或等于该时间点的最新版本
            query = """
                SELECT r1.id, r1.relation_id, r1.entity1_absolute_id, r1.entity2_absolute_id, 
                       r1.content, r1.event_time, r1.processed_time, r1.memory_cache_id, r1.source_document, r1.embedding
                FROM relations r1
                INNER JOIN (
                    SELECT relation_id, MAX(processed_time) as max_time
                    FROM relations
                    WHERE (entity1_absolute_id = ? OR entity2_absolute_id = ?)
                    AND event_time <= ?
                    GROUP BY relation_id
                ) r2 ON r1.relation_id = r2.relation_id 
                    AND r1.processed_time = r2.max_time
                    AND (r1.entity1_absolute_id = ? OR r1.entity2_absolute_id = ?)
                ORDER BY r1.processed_time DESC
            """
            params = (entity_absolute_id, entity_absolute_id, time_point.isoformat(), entity_absolute_id, entity_absolute_id)
        else:
            # 获取每个relation_id的最新版本
            query = """
                SELECT r1.id, r1.relation_id, r1.entity1_absolute_id, r1.entity2_absolute_id, 
                       r1.content, r1.event_time, r1.processed_time, r1.memory_cache_id, r1.source_document, r1.embedding
                FROM relations r1
                INNER JOIN (
                    SELECT relation_id, MAX(processed_time) as max_time
                    FROM relations
                    WHERE entity1_absolute_id = ? OR entity2_absolute_id = ?
                    GROUP BY relation_id
                ) r2 ON r1.relation_id = r2.relation_id 
                    AND r1.processed_time = r2.max_time
                    AND (r1.entity1_absolute_id = ? OR r1.entity2_absolute_id = ?)
                ORDER BY r1.processed_time DESC
            """
            params = (entity_absolute_id, entity_absolute_id, entity_absolute_id, entity_absolute_id)
        
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        
        cursor.execute(query, params)
        
        rows = cursor.fetchall()
        
        return [
            Relation(
                absolute_id=row[0],
                relation_id=row[1],
                entity1_absolute_id=row[2] or "",
                entity2_absolute_id=row[3] or "",
                content=row[4],
                event_time=datetime.fromisoformat(row[5]),
                processed_time=datetime.fromisoformat(row[6]),
                memory_cache_id=row[7],
                source_document=row[8] if len(row) > 8 and row[8] is not None else "",  # 向后兼容
                embedding=row[9] if len(row) > 9 else None
            )
            for row in rows
        ]
    
    def get_entity_relations_by_entity_id(self, entity_id: str, limit: Optional[int] = None, time_point: Optional[datetime] = None, max_version_absolute_id: Optional[str] = None) -> List[Relation]:
        """获取与指定实体相关的所有关系（通过entity_id查找，包含该实体的所有版本）
        
        这个方法会查找该实体的所有版本（从最早版本开始）的所有关系，
        然后按relation_id去重，保留每个relation_id的最新版本。
        
        Args:
            entity_id: 实体的entity_id（不是absolute_id）
            limit: 限制返回的关系数量（按时间倒序），None表示不限制
            time_point: 时间点（可选），如果提供，只返回该时间点之前或等于该时间点的关系，且每个relation_id只返回最新版本
            max_version_absolute_id: 最大版本absolute_id（可选），如果提供，只查询从最早版本到该版本的所有关系
        """
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return []
        # 先获取该实体的所有版本的absolute_id
        versions = self.get_entity_versions(entity_id)
        if not versions:
            return []
        
        # 如果指定了max_version_absolute_id，只取到该版本为止的所有版本
        if max_version_absolute_id:
            # 按时间排序，找到max_version_absolute_id对应的版本
            versions_sorted = sorted(
                versions,
                key=lambda v: self._normalize_datetime_for_compare(v.processed_time),
            )
            max_version = None
            for v in versions_sorted:
                if v.absolute_id == max_version_absolute_id:
                    max_version = v
                    break
            
            if max_version:
                t_max = self._normalize_datetime_for_compare(max_version.processed_time)
                # 只取到该版本（包含）为止的所有版本
                entity_absolute_ids = [
                    v.absolute_id for v in versions_sorted
                    if self._normalize_datetime_for_compare(v.processed_time) <= t_max
                ]
                # 同时设置time_point为该版本的时间点
                if not time_point:
                    time_point = max_version.processed_time
                else:
                    # 如果已经设置了time_point，取较小值（避免 naive/aware 无法比较）
                    nt = self._normalize_datetime_for_compare(time_point)
                    if nt <= t_max:
                        pass  # 保持 time_point
                    else:
                        time_point = max_version.processed_time
            else:
                # 如果找不到指定的版本，使用所有版本
                entity_absolute_ids = [v.absolute_id for v in versions]
        else:
            # 收集所有版本的absolute_id
            entity_absolute_ids = [v.absolute_id for v in versions]
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 构建查询：查找所有版本的关系，按relation_id去重
        placeholders = ','.join(['?'] * len(entity_absolute_ids))
        
        if time_point:
            # 获取每个relation_id在该时间点之前或等于该时间点的最新版本
            query = f"""
                SELECT r1.id, r1.relation_id, r1.entity1_absolute_id, r1.entity2_absolute_id, 
                       r1.content, r1.event_time, r1.processed_time, r1.memory_cache_id, r1.source_document, r1.embedding
                FROM relations r1
                INNER JOIN (
                    SELECT relation_id, MAX(processed_time) as max_time
                    FROM relations
                    WHERE (entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders}))
                    AND event_time <= ?
                    GROUP BY relation_id
                ) r2 ON r1.relation_id = r2.relation_id 
                    AND r1.processed_time = r2.max_time
                    AND (r1.entity1_absolute_id IN ({placeholders}) OR r1.entity2_absolute_id IN ({placeholders}))
                ORDER BY r1.processed_time DESC
            """
            params = tuple(entity_absolute_ids * 2 + [time_point.isoformat()] + entity_absolute_ids * 2)
        else:
            # 获取每个relation_id的最新版本
            query = f"""
            SELECT r1.id, r1.relation_id, r1.entity1_absolute_id, r1.entity2_absolute_id, 
                   r1.content, r1.event_time, r1.processed_time, r1.memory_cache_id, r1.source_document, r1.embedding
            FROM relations r1
            INNER JOIN (
                SELECT relation_id, MAX(processed_time) as max_time
                FROM relations
                WHERE entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})
                GROUP BY relation_id
            ) r2 ON r1.relation_id = r2.relation_id 
                AND r1.processed_time = r2.max_time
                AND (r1.entity1_absolute_id IN ({placeholders}) OR r1.entity2_absolute_id IN ({placeholders}))
            ORDER BY r1.processed_time DESC
            """
            params = tuple(entity_absolute_ids * 4)
        
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        
        cursor.execute(query, params)
        
        rows = cursor.fetchall()
        
        return [
            Relation(
                absolute_id=row[0],
                relation_id=row[1],
                entity1_absolute_id=row[2] or "",
                entity2_absolute_id=row[3] or "",
                content=row[4],
                event_time=datetime.fromisoformat(row[5]),
                processed_time=datetime.fromisoformat(row[6]),
                memory_cache_id=row[7],
                source_document=row[8] if len(row) > 8 and row[8] is not None else "",  # 向后兼容
                embedding=row[9] if len(row) > 9 else None
            )
            for row in rows
        ]
    
    def get_relations_by_entity_absolute_ids(self, entity_absolute_ids: List[str], limit: Optional[int] = None) -> List[Relation]:
        """获取与指定实体版本列表直接关联的所有关系（通过entity_absolute_id直接匹配）
        
        这个方法根据关系边中的 entity1_absolute_id 或 entity2_absolute_id 直接匹配，
        不使用时间过滤，只返回直接引用这些实体版本的关系边。
        按 relation_id 去重，每个 relation_id 只返回一条记录（保留最新的）。
        
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
            SELECT id, relation_id, entity1_absolute_id, entity2_absolute_id, 
                   content, event_time, processed_time, memory_cache_id, source_document, embedding
            FROM relations
            WHERE entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})
            ORDER BY processed_time DESC
        """
        
        params = tuple(entity_absolute_ids * 2)
        cursor.execute(query, params)
        
        rows = cursor.fetchall()
        
        # 按 relation_id 去重，保留第一个（最新的）
        seen_relation_ids = set()
        result = []
        for row in rows:
            relation_id = row[1]
            if relation_id not in seen_relation_ids:
                seen_relation_ids.add(relation_id)
                result.append(
                    Relation(
                        absolute_id=row[0],
                        relation_id=row[1],
                        entity1_absolute_id=row[2] or "",
                        entity2_absolute_id=row[3] or "",
                        content=row[4],
                        event_time=datetime.fromisoformat(row[5]),
                        processed_time=datetime.fromisoformat(row[6]),
                        memory_cache_id=row[7],
                        source_document=row[8] if len(row) > 8 and row[8] is not None else "",  # 向后兼容
                        embedding=row[9] if len(row) > 9 else None
                    )
                )
                if limit is not None and len(result) >= limit:
                    break
        
        return result
    
    def get_entity_absolute_ids_up_to_version(self, entity_id: str, max_absolute_id: str) -> List[str]:
        """获取指定实体从最早版本到指定版本的所有 absolute_id 列表
        
        Args:
            entity_id: 实体的 entity_id
            max_absolute_id: 最大版本的 absolute_id（包含）
        
        Returns:
            从最早版本到指定版本的所有 absolute_id 列表（按时间顺序）
        """
        versions = self.get_entity_versions(entity_id)
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

        emb_col = ", r1.embedding" if not exclude_embedding else ""
        # 获取每个 relation_id 的最新版本
        query = f"""
            SELECT r1.id, r1.relation_id, r1.entity1_absolute_id, r1.entity2_absolute_id,
                   r1.content, r1.event_time, r1.processed_time, r1.memory_cache_id, r1.source_document{emb_col}
            FROM relations r1
            INNER JOIN (
                SELECT relation_id, MAX(processed_time) as max_time
                FROM relations
                GROUP BY relation_id
            ) r2 ON r1.relation_id = r2.relation_id AND r1.processed_time = r2.max_time
            ORDER BY r1.processed_time DESC
        """

        if offset is not None and offset > 0:
            query += f" OFFSET {int(offset)}"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        cursor.execute(query)

        rows = cursor.fetchall()

        return [
            Relation(
                absolute_id=row[0],
                relation_id=row[1],
                entity1_absolute_id=row[2] or "",
                entity2_absolute_id=row[3] or "",
                content=row[4],
                event_time=datetime.fromisoformat(row[5]),
                processed_time=datetime.fromisoformat(row[6]),
                memory_cache_id=row[7],
                source_document=row[8] if len(row) > 8 and row[8] is not None else "",
                embedding=row[9] if not exclude_embedding and len(row) > 9 else None
            )
            for row in rows
        ]
    
    def _get_relations_with_embeddings(self) -> List[tuple]:
        """
        获取所有关系的最新版本及其embedding
        
        Returns:
            List of (Relation, embedding_array) tuples, embedding_array为None表示没有embedding
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 获取每个relation_id的最新版本及其embedding
        cursor.execute("""
            SELECT id, relation_id, entity1_absolute_id, entity2_absolute_id,
                   content, event_time, processed_time, memory_cache_id, source_document, embedding
            FROM relations r1
            WHERE r1.processed_time = (
                SELECT MAX(r2.processed_time)
                FROM relations r2
                WHERE r2.relation_id = r1.relation_id
            )
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
            relation = Relation(
                absolute_id=row[0],
                relation_id=row[1],
                entity1_absolute_id=row[2] or "",
                entity2_absolute_id=row[3] or "",
                content=row[4],
                event_time=datetime.fromisoformat(row[5]),
                processed_time=datetime.fromisoformat(row[6]),
                memory_cache_id=row[7],
                source_document=row[8] if len(row) > 8 else '',
                embedding=row[9] if len(row) > 9 else None
            )
            results.append((relation, embedding_array))
        
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
        
        # 返回关系列表（去重，每个relation_id只保留一个，并限制最大数量）
        relations = []
        seen_ids = set()
        for relation, _ in similarities:
            if relation.relation_id not in seen_ids:
                relations.append(relation)
                seen_ids.add(relation.relation_id)
                if len(relations) >= max_results:
                    break
        
        return relations
    
    def _search_relations_with_text_similarity(self, query_text: str, 
                                               all_relations: List[Relation],
                                               threshold: float,
                                               max_results: int) -> List[Relation]:
        """使用文本相似度进行关系搜索"""
        import difflib
        
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
        
        # 返回关系列表（去重，每个relation_id只保留一个，并限制最大数量）
        relations = []
        seen_ids = set()
        for relation, _ in scored_relations:
            if relation.relation_id not in seen_ids:
                relations.append(relation)
                seen_ids.add(relation.relation_id)
                if len(relations) >= max_results:
                    break
        
        return relations
    
    # ========== 知识图谱整理操作 ==========
    
    def get_doc_hash_by_cache_id(self, cache_id: str) -> Optional[str]:
        """根据 memory_cache_id 获取对应的文档目录名（doc_hash）。"""
        return self._id_to_doc_hash.get(cache_id)

    def get_memory_cache_text(self, cache_id: str) -> Optional[str]:
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
                except Exception:
                    pass

        # 2. 回退到旧结构
        metadata_path = self.cache_json_dir / f"{cache_id}.json"
        if not metadata_path.exists():
            metadata_path = self.cache_dir / f"{cache_id}.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                return metadata.get("text")
            except Exception:
                pass

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
            except Exception:
                pass

        # 读取 original.txt
        original_path = doc_dir / "original.txt"
        if original_path.exists():
            try:
                result["original"] = original_path.read_text(encoding="utf-8")
            except Exception:
                pass

        # 读取 cache.md
        cache_path = doc_dir / "cache.md"
        if cache_path.exists():
            try:
                result["cache"] = cache_path.read_text(encoding="utf-8")
            except Exception:
                pass

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
            except Exception:
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
                    from datetime import datetime as _dt
                    processed_time_str = _dt.fromtimestamp(st.st_mtime).isoformat()
                except Exception:
                    pass

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

    def find_related_entities_by_embedding(self, similarity_threshold: float = 0.7,
                                           max_candidates: int = 5,
                                           use_mixed_search: bool = True,
                                           content_snippet_length: int = 50,
                                           progress_callback: Optional[callable] = None) -> Dict[str, set]:
        """
        使用混合检索方式找到每个实体的关联实体
        
        每个 entity_id 只使用最新版本（processed_time 最新的）的实体来计算相似度
        
        Args:
            similarity_threshold: 相似度阈值
            max_candidates: 每个实体返回的最大候选实体数
            use_mixed_search: 是否使用混合检索（多种模式和方法）
            content_snippet_length: 用于检索的content截取长度
            
        Returns:
            Dict[entity_id, set of candidate_entity_ids]
        """
        # 获取所有实体及其embedding（已经按entity_id去重，每个entity_id只返回最新版本）
        entities_with_embeddings = self._get_entities_with_embeddings()
        
        if not entities_with_embeddings:
            return {}
        
        result = {}
        
        if use_mixed_search:
            # 使用混合检索方式：为每个实体使用多种模式和方法检索，然后合并结果
            total_entities = len(entities_with_embeddings)
            for idx, (entity, _) in enumerate(entities_with_embeddings, 1):
                entity_id = entity.entity_id
                candidate_ids = set()
                
                # 显示进度
                if progress_callback:
                    progress_callback(idx, total_entities, entity.name)
                
                # 计算每种模式应该返回的候选数量（对半分）
                half_candidates = max(1, max_candidates // 2)
                
                # 模式1：只用name检索（使用embedding）
                candidates_name_embedding = self.search_entities_by_similarity(
                    query_name=entity.name,
                    query_content=None,
                    threshold=similarity_threshold,
                    max_results=half_candidates,
                    content_snippet_length=content_snippet_length,
                    text_mode="name_only",
                    similarity_method="embedding"
                )
                
                # 模式2：使用name+content检索（使用embedding）
                candidates_full_embedding = self.search_entities_by_similarity(
                    query_name=entity.name,
                    query_content=entity.content,
                    threshold=similarity_threshold,
                    max_results=half_candidates,
                    content_snippet_length=content_snippet_length,
                    text_mode="name_and_content",
                    similarity_method="embedding"
                )
                
                # 模式3：只用name检索（使用文本相似度，作为补充）
                if len(candidate_ids) < max_candidates:
                    candidates_name_text = self.search_entities_by_similarity(
                        query_name=entity.name,
                        query_content=None,
                        threshold=similarity_threshold,
                        max_results=half_candidates,
                        content_snippet_length=content_snippet_length,
                        text_mode="name_only",
                        similarity_method="text"
                    )
                    for candidate in candidates_name_text:
                        if candidate.entity_id != entity_id:
                            candidate_ids.add(candidate.entity_id)
                
                # 合并所有候选实体
                for candidate in candidates_name_embedding + candidates_full_embedding:
                    if candidate.entity_id != entity_id:
                        candidate_ids.add(candidate.entity_id)
                
                # 如果候选数量不足，尝试使用Jaccard相似度补充
                if len(candidate_ids) < max_candidates:
                    candidates_name_jaccard = self.search_entities_by_similarity(
                        query_name=entity.name,
                        query_content=None,
                        threshold=similarity_threshold,
                        max_results=half_candidates,
                        content_snippet_length=content_snippet_length,
                        text_mode="name_only",
                        similarity_method="jaccard"
                    )
                    for candidate in candidates_name_jaccard:
                        if candidate.entity_id != entity_id and len(candidate_ids) < max_candidates:
                            candidate_ids.add(candidate.entity_id)
                
                result[entity_id] = candidate_ids
        else:
            # 使用原来的批量embedding计算方式（更高效，但只使用embedding）
            # 构建实体索引（确保每个entity_id只保留一个，使用最新版本）
            # entity_index: entity_id -> (Entity, embedding_array, embedding_index_in_all_embeddings)
            entity_index = {}
            all_embeddings = []
            entity_id_list = []  # 与all_embeddings对应的entity_id列表，用于快速查找
            
            for entity, embedding_array in entities_with_embeddings:
                if embedding_array is not None:
                    # 确保每个entity_id只保留一个（如果已存在，跳过，因为_get_entities_with_embeddings已经返回最新版本）
                    if entity.entity_id not in entity_index:
                        embedding_idx = len(all_embeddings)
                        entity_index[entity.entity_id] = (entity, embedding_array, embedding_idx)
                        all_embeddings.append(embedding_array)
                        entity_id_list.append(entity.entity_id)
            
            if not all_embeddings:
                return {}
            
            # 转换为numpy数组以便批量计算
            all_embeddings_array = np.array(all_embeddings)
            
            # 计算每个实体与其他实体的相似度
            total_entities = len(entity_index)
            for idx, (entity_id, (entity, embedding, embedding_idx)) in enumerate(entity_index.items(), 1):
                candidate_ids = set()
                
                # 显示进度
                if progress_callback:
                    progress_callback(idx, total_entities, entity.name)
                
                # 计算与所有其他实体的余弦相似度
                dot_products = np.dot(all_embeddings_array, embedding)
                norms = np.linalg.norm(all_embeddings_array, axis=1)
                norm_entity = np.linalg.norm(embedding)
                similarities = dot_products / (norms * norm_entity + 1e-9)
                
                # 找到相似度高于阈值且不是自己的实体
                similar_indices = np.where((similarities >= similarity_threshold) & 
                                          (np.arange(len(similarities)) != embedding_idx))[0]
                
                # 按相似度排序
                if len(similar_indices) > 0:
                    similar_scores = similarities[similar_indices]
                    sorted_order = np.argsort(similar_scores)[::-1]  # 降序
                    similar_indices = similar_indices[sorted_order]
                    
                    # 取前 max_candidates 个
                    for i in similar_indices[:max_candidates]:
                        candidate_entity_id = entity_id_list[i]
                        if candidate_entity_id != entity_id:
                            candidate_ids.add(candidate_entity_id)
                
                result[entity_id] = candidate_ids
        
        return result
    
    def get_entities_grouped_by_similarity(self, similarity_threshold: float = 0.6) -> List[List[Entity]]:
        """
        获取按名称相似度分组的实体
        
        使用embedding向量计算实体之间的相似度，将相似的实体分组
        
        Args:
            similarity_threshold: 相似度阈值，高于此值的实体会被分到同一组
            
        Returns:
            实体分组列表，每组包含相似的实体
        """
        # 获取所有实体及其embedding
        entities_with_embeddings = self._get_entities_with_embeddings()
        
        if not entities_with_embeddings:
            return []
        
        # 构建相似度矩阵
        n = len(entities_with_embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            entity_i, embedding_i = entities_with_embeddings[i]
            if embedding_i is None:
                continue
            
            for j in range(i + 1, n):
                entity_j, embedding_j = entities_with_embeddings[j]
                if embedding_j is None:
                    continue
                
                # 计算余弦相似度
                dot_product = np.dot(embedding_i, embedding_j)
                norm_i = np.linalg.norm(embedding_i)
                norm_j = np.linalg.norm(embedding_j)
                similarity = dot_product / (norm_i * norm_j + 1e-9)
                
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        # 使用并查集进行分组
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 根据相似度阈值合并
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] >= similarity_threshold:
                    union(i, j)
        
        # 构建分组
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(entities_with_embeddings[i][0])  # 只添加Entity，不需要embedding
        
        # 只返回包含多个实体的组（单个实体不需要整理）
        result = [group for group in groups.values() if len(group) > 1]
        
        return result
    
    def merge_entity_ids(self, target_entity_id: str, source_entity_ids: List[str]) -> Dict[str, Any]:
        """
        将多个source_entity_id的记录合并到target_entity_id
        
        Args:
            target_entity_id: 目标实体ID（保留的ID）
            source_entity_ids: 要合并的源实体ID列表
            
        Returns:
            合并结果统计，包含更新的实体数量和关系数量
        """
        target_entity_id = self.resolve_entity_id(target_entity_id)
        if not target_entity_id or not source_entity_ids:
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
                for source_id in source_entity_ids:
                    source_id = self._resolve_entity_id_with_cursor(cursor, source_id)
                    if not source_id or source_id == target_entity_id or source_id in canonical_source_ids:
                        continue
                    canonical_source_ids.append(source_id)
                    cursor.execute("""
                        SELECT COUNT(*) FROM entities
                        WHERE entity_id = ?
                    """, (source_id,))
                    count = cursor.fetchone()[0]
                    source_version_counts[source_id] = count

                # 2. 更新entities表中的所有entity_id记录
                # 这会更新所有使用source_entity_id的记录，包括所有版本
                for source_id in canonical_source_ids:
                    cursor.execute("""
                        UPDATE entities
                        SET entity_id = ?
                        WHERE entity_id = ?
                    """, (target_entity_id, source_id))
                    entities_updated += cursor.rowcount

                # 2.5. 验证：确保所有源实体的版本都被更新了
                # 检查是否还有任何源entity_id的记录残留
                for source_id in canonical_source_ids:
                    cursor.execute("""
                        SELECT COUNT(*) FROM entities
                        WHERE entity_id = ?
                    """, (source_id,))
                    remaining_count = cursor.fetchone()[0]
                    if remaining_count > 0:
                        # 如果还有残留记录，说明更新失败，回滚事务
                        conn.rollback()
                        raise ValueError(
                            f"合并失败：源实体 {source_id} 仍有 {remaining_count} 条记录未被更新 "
                            f"（预期应更新 {source_version_counts[source_id]} 条记录，实际更新了 {source_version_counts[source_id] - remaining_count} 条）"
                        )

                # 3. 获取target_entity_id的最新版本的绝对ID
                cursor.execute("""
                    SELECT id FROM entities
                    WHERE entity_id = ?
                    ORDER BY processed_time DESC
                    LIMIT 1
                """, (target_entity_id,))

                target_absolute_id_row = cursor.fetchone()
                if not target_absolute_id_row:
                    conn.rollback()
                    return {"entities_updated": 0, "relations_updated": 0, "error": "目标实体不存在"}

                target_absolute_id = target_absolute_id_row[0]

                # 注意：关系边中的绝对ID保持不变，因为它们指向的是特定版本
                # 合并实体后，这些关系仍然有效，只是entity_id变了
                # 不需要更新relations表，因为：
                # - 关系表只存储absolute_id，不存储entity_id
                # - 通过absolute_id查询实体时，会得到更新后的entity_id
                # - 所有使用entity_id查询的地方（如get_relations_by_entities）都会自动使用新的entity_id

                now_iso = datetime.now().isoformat()
                for source_id in canonical_source_ids:
                    cursor.execute(
                        """
                        INSERT INTO entity_redirects (source_entity_id, target_entity_id, updated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(source_entity_id) DO UPDATE SET
                            target_entity_id = excluded.target_entity_id,
                            updated_at = excluded.updated_at
                        """,
                        (source_id, target_entity_id, now_iso),
                    )

                conn.commit()

            except Exception as e:
                conn.rollback()
                raise e

            return {
                "entities_updated": entities_updated,
                "relations_updated": relations_updated,
                "target_entity_id": target_entity_id,
                "merged_source_ids": canonical_source_ids
            }
    
    def get_entity_version_count(self, entity_id: str) -> int:
        """获取指定entity_id的版本数量
        
        Args:
            entity_id: 实体ID
            
        Returns:
            版本数量
        """
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return 0
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM entities WHERE entity_id = ?
        """, (entity_id,))
        
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
            cursor.execute("SELECT COUNT(DISTINCT entity_id) FROM entities")
            unique_entities = cursor.fetchone()[0]
            if unique_entities > 1:
                max_possible = unique_entities * (unique_entities - 1) / 2
                cursor.execute("SELECT COUNT(DISTINCT relation_id) FROM relations")
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

    def get_entity_version_counts(self, entity_ids: List[str]) -> Dict[str, int]:
        """批量获取多个entity_id的版本数量

        Args:
            entity_ids: 实体ID列表

        Returns:
            Dict[entity_id, version_count]
        """
        if not entity_ids:
            return {}
        canonical_ids = []
        for entity_id in entity_ids:
            canonical_id = self.resolve_entity_id(entity_id)
            if canonical_id and canonical_id not in canonical_ids:
                canonical_ids.append(canonical_id)
        if not canonical_ids:
            return {}
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 使用IN子句批量查询
        placeholders = ','.join(['?'] * len(canonical_ids))
        cursor.execute(f"""
            SELECT entity_id, COUNT(*) as version_count
            FROM entities
            WHERE entity_id IN ({placeholders})
            GROUP BY entity_id
        """, canonical_ids)
        
        rows = cursor.fetchall()
        
        return {row[0]: row[1] for row in rows}

    def entity_has_any_relation(self, entity_id: str) -> bool:
        """检查实体是否在关系表中作为任一端出现（轻量查询，只查 COUNT）。"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return False
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM relations
            WHERE entity1_absolute_id IN (SELECT id FROM entities WHERE entity_id = ?)
               OR entity2_absolute_id IN (SELECT id FROM entities WHERE entity_id = ?)
        """, (entity_id, entity_id))
        return cursor.fetchone()[0] > 0

    def delete_orphan_entities(self, candidate_entity_ids: list) -> list:
        """批量检查并删除没有关系的实体。

        一条 SQL 找出候选列表中确实无关系的 entity_id，然后删除。

        Args:
            candidate_entity_ids: 待检查的 entity_id 列表

        Returns:
            被删除的 entity_id 列表
        """
        if not candidate_entity_ids:
            return []
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            placeholders = ','.join(['?'] * len(candidate_entity_ids))
            # 找出候选中没有任何关系的 entity_id
            cursor.execute(f"""
                SELECT e.entity_id FROM entities e
                WHERE e.entity_id IN ({placeholders})
                  AND e.id NOT IN (
                      SELECT entity1_absolute_id FROM relations
                      UNION
                      SELECT entity2_absolute_id FROM relations
                  )
                GROUP BY e.entity_id
            """, candidate_entity_ids)
            orphan_ids = [row[0] for row in cursor.fetchall()]
            if orphan_ids:
                ph2 = ','.join(['?'] * len(orphan_ids))
                cursor.execute(f"DELETE FROM entities WHERE entity_id IN ({ph2})", orphan_ids)
            return orphan_ids

    def delete_entity_by_id(self, entity_id: str) -> int:
        """删除实体的所有版本。返回删除的行数。"""
        entity_id = self.resolve_entity_id(entity_id)
        if not entity_id:
            return 0
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM entities WHERE entity_id = ?", (entity_id,))
            # 清理 FTS 表
            try:
                cursor.execute("DELETE FROM entity_fts WHERE entity_id = ?", (entity_id,))
            except Exception:
                pass
            return cursor.rowcount

    def delete_relation_by_id(self, relation_id: str) -> int:
        """删除关系的所有版本。返回删除的行数。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM relations WHERE relation_id = ?", (relation_id,))
            count = cursor.rowcount
            # 清理 FTS 表
            try:
                cursor.execute("DELETE FROM relation_fts WHERE relation_id = ?", (relation_id,))
            except Exception:
                pass
            conn.commit()
            return count

    def delete_entity_all_versions(self, entity_id: str) -> int:
        """删除实体的所有版本（含重定向解析）。返回删除的行数。"""
        return self.delete_entity_by_id(entity_id)

    def delete_relation_all_versions(self, relation_id: str) -> int:
        """删除关系的所有版本。返回删除的行数。"""
        return self.delete_relation_by_id(relation_id)

    def get_entity_ids_by_names(self, names: list) -> dict:
        """按名称批量查询实 entity_id（每个 name 取最新版本）。

        Returns:
            {name: entity_id} 仅包含能找到的名称。
        """
        if not names:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ','.join(['?'] * len(names))
        cursor.execute(f"""
            SELECT name, entity_id FROM entities
            WHERE name IN ({placeholders})
            ORDER BY processed_time DESC
        """, names)
        result = {}
        for name, eid in cursor.fetchall():
            if name not in result:
                result[name] = self.resolve_entity_id(eid)
        return result

    def get_total_entity_count(self) -> int:
        """获取数据库中的实体总数（去重 entity_id 后的数量）。"""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT entity_id) FROM entities")
            return cursor.fetchone()[0]
        except Exception:
            return 0

    def find_shortest_paths(self, source_entity_id: str, target_entity_id: str,
                            max_depth: int = 6, max_paths: int = 10) -> Dict[str, Any]:
        """使用 BFS 查找两个实体之间的所有最短路径。

        在 entity_id 级别的无向图上执行 BFS，找到所有等长的最短路径，
        然后重构路径中每对相邻实体之间的连接关系。

        Args:
            source_entity_id: 起始实体的 entity_id
            target_entity_id: 目标实体的 entity_id
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
        source_entity = self.get_entity_by_entity_id(source_entity_id)
        target_entity = self.get_entity_by_entity_id(target_entity_id)

        if not source_entity or not target_entity:
            result_empty["source_entity"] = source_entity
            result_empty["target_entity"] = target_entity
            return result_empty

        # 同一实体
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

        # 2. 加载全量最新关系（路径查找不需要 embedding）
        all_relations = self.get_all_relations(exclude_embedding=True)
        if not all_relations:
            result_empty["source_entity"] = source_entity
            result_empty["target_entity"] = target_entity
            return result_empty

        # 3. 构建 absolute_id → entity_id 映射
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT id, entity_id FROM entities")
        abs_to_eid = {row[0]: row[1] for row in cursor.fetchall()}

        # 4. 构建 entity_id 级邻接表 和 entity_pair → [Relation] 映射
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
        # visited: entity_id → distance from source
        # parents: entity_id → list of parent entity_ids on shortest paths
        visited: Dict[str, int] = {source_entity_id: 0}
        parents: Dict[str, List[str]] = {source_entity_id: []}
        queue = [source_entity_id]
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
                        if neighbor == target_entity_id:
                            found_depth = current_dist + 1
                    elif visited[neighbor] == current_dist + 1:
                        # 另一条等长路径
                        parents[neighbor].append(current)
            queue = next_queue

        # 未到达目标
        if target_entity_id not in visited:
            result_empty["source_entity"] = source_entity
            result_empty["target_entity"] = target_entity
            return result_empty

        # 6. 回溯重构所有最短路径（DFS on parents）
        all_paths_eid: List[List[str]] = []

        def backtrack(node: str, path: List[str]):
            if len(all_paths_eid) >= max_paths * 10:
                return  # 防止爆炸
            if node == source_entity_id:
                all_paths_eid.append(list(reversed(path)))
                return
            for parent in parents.get(node, []):
                path.append(parent)
                backtrack(parent, path)
                path.pop()

        backtrack(target_entity_id, [target_entity_id])
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
                SELECT id, entity_id, name, content, event_time, processed_time,
                       memory_cache_id, source_document
                FROM entities WHERE id IN ({placeholders})
            """, list(needed_abs_ids))
            for row in cursor.fetchall():
                abs_entity_map[row[0]] = Entity(
                    absolute_id=row[0],
                    entity_id=row[1],
                    name=row[2],
                    content=row[3],
                    event_time=self._safe_parse_datetime(row[4]),
                    processed_time=self._safe_parse_datetime(row[5]),
                    memory_cache_id=row[6],
                    source_document=row[7] if len(row) > 7 else '',
                )

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

        cursor.execute("""
            SELECT id, entity_id, name, content, event_time, processed_time, memory_cache_id, source_document, valid_at, invalid_at
            FROM entities
            WHERE (valid_at IS NULL OR valid_at <= ?)
              AND (invalid_at IS NULL OR invalid_at > ?)
            ORDER BY event_time DESC
            LIMIT ?
        """, (time_point.isoformat(), time_point.isoformat(), limit or 10000))

        entities = []
        for row in cursor.fetchall():
            entities.append(Entity(
                absolute_id=row[0], entity_id=row[1], name=row[2], content=row[3],
                event_time=self._safe_parse_datetime(row[4]),
                processed_time=self._safe_parse_datetime(row[5]),
                memory_cache_id=row[6], source_document=row[7] if len(row) > 7 else '',
                valid_at=self._safe_parse_datetime(row[8]) if len(row) > 8 else None,
                invalid_at=self._safe_parse_datetime(row[9]) if len(row) > 9 else None,
            ))

        cursor.execute("""
            SELECT id, relation_id, entity1_absolute_id, entity2_absolute_id, content, event_time, processed_time, memory_cache_id, source_document, valid_at, invalid_at
            FROM relations
            WHERE (valid_at IS NULL OR valid_at <= ?)
              AND (invalid_at IS NULL OR invalid_at > ?)
            ORDER BY event_time DESC
            LIMIT ?
        """, (time_point.isoformat(), time_point.isoformat(), limit or 10000))

        relations = []
        for row in cursor.fetchall():
            relations.append(Relation(
                absolute_id=row[0], relation_id=row[1],
                entity1_absolute_id=row[2], entity2_absolute_id=row[3],
                content=row[4], event_time=self._safe_parse_datetime(row[5]),
                processed_time=self._safe_parse_datetime(row[6]),
                memory_cache_id=row[7], source_document=row[8] if len(row) > 8 else '',
                valid_at=self._safe_parse_datetime(row[9]) if len(row) > 9 else None,
                invalid_at=self._safe_parse_datetime(row[10]) if len(row) > 10 else None,
            ))

        return {"entities": entities, "relations": relations}

    def get_changes(self, since: datetime, until: Optional[datetime] = None) -> Dict[str, Any]:
        """获取时间范围内的变更记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        until_str = until.isoformat() if until else datetime.now(timezone.utc).isoformat()

        # 新增/修改的实体
        cursor.execute("""
            SELECT id, entity_id, name, content, event_time, processed_time, memory_cache_id, source_document, valid_at, invalid_at
            FROM entities
            WHERE event_time >= ? AND event_time <= ?
            ORDER BY event_time DESC
        """, (since.isoformat(), until_str))

        entities = []
        for row in cursor.fetchall():
            entities.append(Entity(
                absolute_id=row[0], entity_id=row[1], name=row[2], content=row[3],
                event_time=self._safe_parse_datetime(row[4]),
                processed_time=self._safe_parse_datetime(row[5]),
                memory_cache_id=row[6], source_document=row[7] if len(row) > 7 else '',
                valid_at=self._safe_parse_datetime(row[8]) if len(row) > 8 else None,
                invalid_at=self._safe_parse_datetime(row[9]) if len(row) > 9 else None,
            ))

        # 新增/修改/失效的关系
        cursor.execute("""
            SELECT id, relation_id, entity1_absolute_id, entity2_absolute_id, content, event_time, processed_time, memory_cache_id, source_document, valid_at, invalid_at
            FROM relations
            WHERE event_time >= ? AND event_time <= ?
            ORDER BY event_time DESC
        """, (since.isoformat(), until_str))

        relations = []
        for row in cursor.fetchall():
            relations.append(Relation(
                absolute_id=row[0], relation_id=row[1],
                entity1_absolute_id=row[2], entity2_absolute_id=row[3],
                content=row[4], event_time=self._safe_parse_datetime(row[5]),
                processed_time=self._safe_parse_datetime(row[6]),
                memory_cache_id=row[7], source_document=row[8] if len(row) > 8 else '',
                valid_at=self._safe_parse_datetime(row[9]) if len(row) > 9 else None,
                invalid_at=self._safe_parse_datetime(row[10]) if len(row) > 10 else None,
            ))

        return {"entities": entities, "relations": relations}

    def invalidate_relation(self, relation_id: str, reason: str = "") -> int:
        """标记关系为失效（不删除数据，保留历史记录）"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE relations SET invalid_at = ?
                WHERE relation_id = ? AND invalid_at IS NULL
            """, (datetime.now(timezone.utc).isoformat(), relation_id))
            conn.commit()
            return cursor.rowcount

    def get_invalidated_relations(self, limit: int = 100) -> List[Relation]:
        """列出所有已失效的关系"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, relation_id, entity1_absolute_id, entity2_absolute_id, content, event_time, processed_time, memory_cache_id, source_document, valid_at, invalid_at
            FROM relations
            WHERE invalid_at IS NOT NULL
            ORDER BY invalid_at DESC
            LIMIT ?
        """, (limit,))
        relations = []
        for row in cursor.fetchall():
            relations.append(Relation(
                absolute_id=row[0], relation_id=row[1],
                entity1_absolute_id=row[2], entity2_absolute_id=row[3],
                content=row[4], event_time=self._safe_parse_datetime(row[5]),
                processed_time=self._safe_parse_datetime(row[6]),
                memory_cache_id=row[7], source_document=row[8] if len(row) > 8 else '',
                valid_at=self._safe_parse_datetime(row[9]) if len(row) > 9 else None,
                invalid_at=self._safe_parse_datetime(row[10]) if len(row) > 10 else None,
            ))
        return relations

    # ========== Phase A: 实体智能 ==========

    def update_entity_summary(self, entity_id: str, summary: str):
        """更新实体的摘要（最新版本）。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE entities SET summary = ?
                WHERE id = (
                    SELECT id FROM entities
                    WHERE entity_id = ?
                    ORDER BY processed_time DESC LIMIT 1
                )
            """, (summary, entity_id))
            conn.commit()

    def update_entity_attributes(self, entity_id: str, attributes: str):
        """更新实体的属性字典（JSON 字符串）。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE entities SET attributes = ?
                WHERE id = (
                    SELECT id FROM entities
                    WHERE entity_id = ?
                    ORDER BY processed_time DESC LIMIT 1
                )
            """, (attributes, entity_id))
            conn.commit()

    def update_entity_confidence(self, entity_id: str, confidence: float):
        """更新实体的置信度评分。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE entities SET confidence = ?
                WHERE id = (
                    SELECT id FROM entities
                    WHERE entity_id = ?
                    ORDER BY processed_time DESC LIMIT 1
                )
            """, (confidence, entity_id))
            conn.commit()

    def compute_entity_confidence(self, entity_id: str) -> float:
        """自动计算实体的置信度评分。"""
        versions = self.get_entity_versions(entity_id)
        if not versions:
            return 0.0
        score = min(len(versions) * 0.1, 0.3)
        latest = versions[0]
        if latest.processed_time:
            days_since = (datetime.now(timezone.utc).replace(tzinfo=None) - latest.processed_time).days
            if days_since <= 30:
                score += 0.3
            elif days_since <= 90:
                score += 0.2
            else:
                score += 0.1
        mentions = self.get_entity_provenance(entity_id)
        if mentions:
            score += min(len(mentions) * 0.05, 0.3)
        content_len = len(latest.content)
        if content_len > 200:
            score += 0.1
        return min(score, 1.0)

    # ========== Phase B: 图遍历辅助 ==========

    def get_relations_by_entity_ids(self, entity_ids: List[str], limit: Optional[int] = None) -> List[Relation]:
        """获取与指定实体 ID 列表相关的所有关系。"""
        if not entity_ids:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(entity_ids))
        cursor.execute(f"""
            SELECT entity_id, MAX(processed_time), id
            FROM entities
            WHERE entity_id IN ({placeholders})
            GROUP BY entity_id
        """, entity_ids)
        abs_id_map = {row[0]: row[2] for row in cursor.fetchall()}
        abs_ids = list(abs_id_map.values())
        if not abs_ids:
            return []
        abs_placeholders = ",".join("?" * len(abs_ids))
        query = f"""
            SELECT id, relation_id, entity1_absolute_id, entity2_absolute_id,
                   content, event_time, processed_time, memory_cache_id, source_document, embedding
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
        relations = []
        for row in cursor.fetchall():
            relations.append(Relation(
                absolute_id=row[0], relation_id=row[1],
                entity1_absolute_id=row[2] or "", entity2_absolute_id=row[3] or "",
                content=row[4], event_time=self._safe_parse_datetime(row[5]),
                processed_time=self._safe_parse_datetime(row[6]),
                memory_cache_id=row[7], source_document=row[8] if len(row) > 8 else '',
                embedding=row[9] if len(row) > 9 else None,
            ))
        return relations

    def get_entity_degree(self, entity_id: str) -> int:
        """获取实体的度（连接数）。"""
        return len(self.get_relations_by_entity_ids([entity_id]))

    # ========== Phase C: Episode MENTIONS ==========

    def save_episode_mentions(self, episode_id: str, entity_absolute_ids: List[str], context: str = ""):
        """记录 Episode 中提及的实体。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            for abs_id in entity_absolute_ids:
                cursor.execute("""
                    INSERT OR REPLACE INTO episode_mentions (episode_id, entity_absolute_id, mention_context)
                    VALUES (?, ?, ?)
                """, (episode_id, abs_id, context))
            conn.commit()

    def get_entity_provenance(self, entity_id: str) -> List[dict]:
        """获取提及该实体的所有 Episode。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM entities WHERE entity_id = ?", (entity_id,))
        abs_ids = [row[0] for row in cursor.fetchall()]
        if not abs_ids:
            return []
        placeholders = ",".join("?" * len(abs_ids))
        cursor.execute(f"""
            SELECT episode_id, entity_absolute_id, mention_context
            FROM episode_mentions
            WHERE entity_absolute_id IN ({placeholders})
        """, abs_ids)
        return [
            {"episode_id": row[0], "entity_absolute_id": row[1], "mention_context": row[2] or ""}
            for row in cursor.fetchall()
        ]

    def get_episode_entities(self, episode_id: str) -> List[dict]:
        """获取 Episode 中提及的所有实体。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT em.entity_absolute_id, em.mention_context, e.name, e.entity_id
            FROM episode_mentions em
            LEFT JOIN entities e ON em.entity_absolute_id = e.id
            WHERE em.episode_id = ?
        """, (episode_id,))
        return [
            {"absolute_id": row[0], "mention_context": row[1] or "", "name": row[2] or "", "entity_id": row[3] or ""}
            for row in cursor.fetchall()
        ]

    def delete_episode_mentions(self, episode_id: str):
        """删除 Episode 的所有提及记录。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM episode_mentions WHERE episode_id = ?", (episode_id,))
            conn.commit()

    # ========== Phase D: 关系溯源 ==========

    def update_relation_provenance(self, relation_id: str, provenance: str):
        """更新关系的溯源信息（JSON 字符串）。"""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE relations SET provenance = ?
                WHERE id = (
                    SELECT id FROM relations
                    WHERE relation_id = ?
                    ORDER BY processed_time DESC LIMIT 1
                )
            """, (provenance, relation_id))
            conn.commit()

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