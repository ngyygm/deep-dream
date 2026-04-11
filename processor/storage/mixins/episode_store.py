"""
EpisodeStoreMixin — Episode 相关的存储操作。

抽取自 StorageManager，通过 Mixin 方式混入宿主类。
依赖宿主提供:
  - self._get_conn()
  - self._write_lock
  - self.storage_path, self.cache_dir, self.cache_json_dir, self.cache_md_dir, self.docs_dir
  - self._id_to_doc_hash
  - self._safe_parse_datetime()
  - self._ensure_dirs()
  - self._write_concept_from_episode()
"""
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from ...models import Episode
from ...utils import clean_markdown_code_blocks

logger = logging.getLogger(__name__)


class EpisodeStoreMixin:
    """Episode 存储 Mixin — 无 __init__，依赖宿主类属性。"""

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

    # ========== 知识图谱整理操作 (Episode 部分) ==========

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
        """将 Episode 元数据写入 SQLite episodes 表。

        Vision 原则「Observation 也会演化」：同一 source_document 再次处理时，
        复用已有 family_id 作为新版本（而非独立概念）。
        """
        try:
            conn = self._get_conn()
            with self._write_lock:
                cursor = conn.cursor()

                # Resolve family_id: reuse from existing episode with same source_document
                resolved_family_id = cache.absolute_id  # default: new standalone
                source_doc = getattr(cache, 'source_document', '') or ''
                if source_doc:
                    cursor.execute(
                        "SELECT family_id FROM episodes WHERE source_document = ? "
                        "ORDER BY processed_time DESC LIMIT 1",
                        (source_doc,)
                    )
                    existing = cursor.fetchone()
                    if existing and existing[0]:
                        resolved_family_id = existing[0]

                cursor.execute("""
                    INSERT OR REPLACE INTO episodes
                    (id, family_id, content, event_time, processed_time,
                     source_document, activity_type, episode_type, doc_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache.absolute_id,
                    resolved_family_id,
                    cache.content,
                    cache.event_time.isoformat(),
                    datetime.now().isoformat(),
                    cache.source_document,
                    cache.activity_type or "",
                    getattr(cache, 'episode_type', '') or "",
                    doc_hash,
                ))
                # Phase 2: dual-write to concepts (use resolved family_id)
                self._write_concept_from_episode(cache, doc_hash, cursor,
                                                  family_id=resolved_family_id)
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
            cursor.executemany(
                "INSERT OR REPLACE INTO episode_mentions "
                "(episode_id, target_absolute_id, target_type, mention_context) "
                "VALUES (?, ?, ?, ?)",
                [(episode_id, abs_id, target_type, context) for abs_id in target_absolute_ids],
            )
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
