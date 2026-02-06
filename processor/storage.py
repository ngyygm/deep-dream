"""
存储层：SQLite数据库 + Markdown文件存储
"""
import sqlite3
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path
import hashlib
import numpy as np
import difflib

from .models import MemoryCache, Entity, Relation


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
        
        # 创建子目录
        self.db_path = self.storage_path / "graph.db"
        self.cache_dir = self.storage_path / "memory_caches"
        self.cache_dir.mkdir(exist_ok=True)
        
        # 创建json和md子文件夹
        self.cache_json_dir = self.cache_dir / "json"
        self.cache_md_dir = self.cache_dir / "md"
        self.cache_json_dir.mkdir(exist_ok=True)
        self.cache_md_dir.mkdir(exist_ok=True)
        
        # Embedding客户端
        self.embedding_client = embedding_client
        self.entity_content_snippet_length = entity_content_snippet_length
        self.relation_content_snippet_length = relation_content_snippet_length
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建实体表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                physical_time TEXT NOT NULL,
                memory_cache_id TEXT NOT NULL,
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
                physical_time TEXT NOT NULL,
                memory_cache_id TEXT NOT NULL,
                embedding BLOB
            )
        """)
        
        # 数据迁移：从旧字段名迁移到新字段名
        self._migrate_relation_schema(cursor)
        
        # 添加embedding字段（如果不存在）
        try:
            cursor.execute("ALTER TABLE entities ADD COLUMN embedding BLOB")
        except sqlite3.OperationalError:
            pass  # 字段已存在
        
        try:
            cursor.execute("ALTER TABLE relations ADD COLUMN embedding BLOB")
        except sqlite3.OperationalError:
            pass  # 字段已存在
        
        # 添加doc_name字段（如果不存在）
        try:
            cursor.execute("ALTER TABLE entities ADD COLUMN doc_name TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # 字段已存在
        
        try:
            cursor.execute("ALTER TABLE relations ADD COLUMN doc_name TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # 字段已存在
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_id ON entities(entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_time ON entities(physical_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relation_id ON relations(relation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relation_entities ON relations(entity1_absolute_id, entity2_absolute_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relation_time ON relations(physical_time)")
        
        conn.commit()
        conn.close()
    
    def _migrate_relation_schema(self, cursor):
        """
        迁移关系表 schema：从 from_entity_absolute_id/to_entity_absolute_id 迁移到 entity1_absolute_id/entity2_absolute_id
        同时标准化所有关系中的实体对顺序（确保 entity1 < entity2）
        """
        # 检查是否存在旧字段
        cursor.execute("PRAGMA table_info(relations)")
        columns = [row[1] for row in cursor.fetchall()]
        
        has_old_fields = 'from_entity_absolute_id' in columns and 'to_entity_absolute_id' in columns
        has_new_fields = 'entity1_absolute_id' in columns and 'entity2_absolute_id' in columns
        
        if has_old_fields and not has_new_fields:
            # 需要迁移：添加新字段
            try:
                cursor.execute("ALTER TABLE relations ADD COLUMN entity1_absolute_id TEXT")
                cursor.execute("ALTER TABLE relations ADD COLUMN entity2_absolute_id TEXT")
            except sqlite3.OperationalError:
                pass  # 字段可能已存在
            
            # 迁移数据：复制数据并标准化实体对顺序
            cursor.execute("""
                SELECT id, relation_id, from_entity_absolute_id, to_entity_absolute_id, 
                       content, physical_time, memory_cache_id, embedding
                FROM relations
                WHERE entity1_absolute_id IS NULL OR entity2_absolute_id IS NULL
            """)
            
            rows = cursor.fetchall()
            
            for row in rows:
                rel_id, relation_id, from_id, to_id, content, physical_time, memory_cache_id, embedding = row
                
                # 标准化实体对（按绝对ID的字符串顺序排序）
                # 注意：理想情况下应该获取实体名称来排序，但为了迁移简单，先按ID排序
                if from_id and to_id:
                    if from_id <= to_id:
                        entity1_id = from_id
                        entity2_id = to_id
                    else:
                        entity1_id = to_id
                        entity2_id = from_id
                    
                    # 更新记录
                    cursor.execute("""
                        UPDATE relations 
                        SET entity1_absolute_id = ?, entity2_absolute_id = ?
                        WHERE id = ?
                    """, (entity1_id, entity2_id, rel_id))
            
            # 删除旧索引（如果存在）
            try:
                cursor.execute("DROP INDEX IF EXISTS idx_relation_entities")
            except sqlite3.OperationalError:
                pass
            
            # 创建新索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relation_entities ON relations(entity1_absolute_id, entity2_absolute_id)")
            
            # 注意：不删除旧字段，保留以支持向后兼容
            # SQLite 不支持直接删除列，需要重建表，这里暂时保留旧字段
    
    def _clean_markdown_code_blocks(self, text: str) -> str:
        """
        清理文本中的 markdown 代码块标识符
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本（移除 ```markdown 和 ``` 等代码块标识符）
        """
        import re
        # 移除开头的 ```markdown 或 ``` 标识符
        text = re.sub(r'^```\s*markdown\s*\n?', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^```\s*\n?', '', text, flags=re.MULTILINE)
        # 移除结尾的 ``` 标识符
        text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
        # 移除首尾空白
        text = text.strip()
        return text
    
    # ========== MemoryCache 操作 ==========
    
    def save_memory_cache(self, cache: MemoryCache, text: str = "", document_path: str = "") -> str:
        """保存记忆缓存到Markdown文件
        
        Args:
            cache: 记忆缓存对象
            text: 当前处理的文本内容（可选）
            document_path: 当前处理的文档完整路径（可选，用于断点续传定位）
        """
        # 使用cache.id作为文件名基础，确保JSON和MD文件名一一对应
        # cache.id格式：cache_{timestamp}_{uuid}
        filename = f"{cache.id}.md"
        filepath = self.cache_md_dir / filename
        
        # 清理 markdown 代码块标识符（双重保险）
        content = self._clean_markdown_code_blocks(cache.content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 保存元数据到JSON文件
        metadata = {
            'id': cache.id,
            'physical_time': cache.physical_time.isoformat(),
            'activity_type': cache.activity_type,
            'doc_name': cache.doc_name,  # 文档名称
            'filename': filename,
            'text': text,  # 当前处理的文本内容
            'document_path': document_path  # 当前处理的文档完整路径
        }
        metadata_path = self.cache_json_dir / f"{cache.id}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return cache.id
    
    def load_memory_cache(self, cache_id: str) -> Optional[MemoryCache]:
        """加载记忆缓存"""
        # 先尝试从新的json文件夹加载
        metadata_path = self.cache_json_dir / f"{cache_id}.json"
        # 如果不存在，尝试从旧路径加载（向后兼容）
        if not metadata_path.exists():
            metadata_path = self.cache_dir / f"{cache_id}.json"
            if not metadata_path.exists():
                return None
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        filename = metadata['filename']
        # 先尝试从新的md文件夹加载
        filepath = self.cache_md_dir / filename
        # 如果不存在，尝试从旧路径加载（向后兼容）
        if not filepath.exists():
            filepath = self.cache_dir / filename
            if not filepath.exists():
                return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 清理 markdown 代码块标识符（处理旧缓存文件）
        content = self._clean_markdown_code_blocks(content)
        
        return MemoryCache(
            id=metadata['id'],
            content=content,
            physical_time=datetime.fromisoformat(metadata['physical_time']),
            doc_name=metadata.get('doc_name', ''),  # 向后兼容：如果旧数据没有doc_name，使用空字符串
            activity_type=metadata.get('activity_type')
        )
    
    def get_latest_memory_cache(self, activity_type: Optional[str] = None) -> Optional[MemoryCache]:
        """获取最新的记忆缓存"""
        # 先尝试从新的json文件夹查找
        cache_files = list(self.cache_json_dir.glob("*.json"))
        # 如果新文件夹为空，尝试从旧路径查找（向后兼容）
        if not cache_files:
            cache_files = list(self.cache_dir.glob("*.json"))
        if not cache_files:
            return None
        
        latest_cache = None
        latest_time = None
        
        for cache_file in cache_files:
            with open(cache_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            if activity_type and metadata.get('activity_type') != activity_type:
                continue
            
            cache_time = datetime.fromisoformat(metadata['physical_time'])
            if latest_time is None or cache_time > latest_time:
                latest_time = cache_time
                latest_cache = self.load_memory_cache(metadata['id'])
        
        return latest_cache
    
    def get_latest_memory_cache_metadata(self, activity_type: Optional[str] = None) -> Optional[Dict]:
        """获取最新的记忆缓存元数据（用于断点续传）
        
        Returns:
            包含以下字段的字典：
            - id: 缓存ID
            - physical_time: 物理时间
            - activity_type: 活动类型
            - filename: MD文件名
            - text: 当前处理的文本内容
            - document_path: 当前处理的文档完整路径
        """
        # 先尝试从新的json文件夹查找
        cache_files = list(self.cache_json_dir.glob("*.json"))
        # 如果新文件夹为空，尝试从旧路径查找（向后兼容）
        if not cache_files:
            cache_files = list(self.cache_dir.glob("*.json"))
        if not cache_files:
            return None
        
        latest_metadata = None
        latest_time = None
        
        for cache_file in cache_files:
            with open(cache_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            if activity_type and metadata.get('activity_type') != activity_type:
                continue
            
            cache_time = datetime.fromisoformat(metadata['physical_time'])
            if latest_time is None or cache_time > latest_time:
                latest_time = cache_time
                latest_metadata = metadata
        
        return latest_metadata
    
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 计算embedding
        embedding_blob = self._compute_entity_embedding(entity)
        
        # 更新Entity对象的embedding字段，保持一致性
        entity.embedding = embedding_blob
        
        cursor.execute("""
            INSERT INTO entities (id, entity_id, name, content, physical_time, memory_cache_id, doc_name, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entity.id,
            entity.entity_id,
            entity.name,
            entity.content,
            entity.physical_time.isoformat(),
            entity.memory_cache_id,
            entity.doc_name,
            embedding_blob
        ))
        
        conn.commit()
        conn.close()
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """根据entity_id获取最新版本的实体"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, entity_id, name, content, physical_time, memory_cache_id, doc_name, embedding
            FROM entities
            WHERE entity_id = ?
            ORDER BY physical_time DESC
            LIMIT 1
        """, (entity_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return Entity(
            id=row[0],
            entity_id=row[1],
            name=row[2],
            content=row[3],
            physical_time=datetime.fromisoformat(row[4]),
            memory_cache_id=row[5],
            doc_name=row[6] if len(row) > 6 and row[6] is not None else "",  # 向后兼容
            embedding=row[7] if len(row) > 7 else None
        )
    
    def get_entity_by_absolute_id(self, absolute_id: str) -> Optional[Entity]:
        """根据绝对ID获取实体"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, entity_id, name, content, physical_time, memory_cache_id, doc_name, embedding
            FROM entities
            WHERE id = ?
        """, (absolute_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return Entity(
            id=row[0],
            entity_id=row[1],
            name=row[2],
            content=row[3],
            physical_time=datetime.fromisoformat(row[4]),
            memory_cache_id=row[5],
            doc_name=row[6] if len(row) > 6 and row[6] is not None else "",  # 向后兼容
            embedding=row[7] if len(row) > 7 else None
        )
    
    def get_entity_version_at_time(self, entity_id: str, time_point: datetime) -> Optional[Entity]:
        """获取实体在指定时间点的版本（该时间点之前或等于该时间点的最新版本）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, entity_id, name, content, physical_time, memory_cache_id, doc_name, embedding
            FROM entities
            WHERE entity_id = ? AND physical_time <= ?
            ORDER BY physical_time DESC
            LIMIT 1
        """, (entity_id, time_point.isoformat()))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return Entity(
            id=row[0],
            entity_id=row[1],
            name=row[2],
            content=row[3],
            physical_time=datetime.fromisoformat(row[4]),
            memory_cache_id=row[5],
            doc_name=row[6] if len(row) > 6 and row[6] is not None else "",  # 向后兼容
            embedding=row[7] if len(row) > 7 else None
        )
    
    def get_entity_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        """获取实体embedding向量的前N个值"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT embedding
            FROM entities
            WHERE id = ?
        """, (absolute_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None or row[0] is None:
            return None
        
        try:
            embedding_array = np.frombuffer(row[0], dtype=np.float32)
            return embedding_array[:num_values].tolist()
        except Exception:
            return None
    
    def get_relation_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        """获取关系embedding向量的前N个值"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT embedding
            FROM relations
            WHERE id = ?
        """, (absolute_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None or row[0] is None:
            return None
        
        try:
            embedding_array = np.frombuffer(row[0], dtype=np.float32)
            return embedding_array[:num_values].tolist()
        except Exception:
            return None
    
    def get_entity_versions(self, entity_id: str) -> List[Entity]:
        """获取实体的所有版本"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, entity_id, name, content, physical_time, memory_cache_id, doc_name, embedding
            FROM entities
            WHERE entity_id = ?
            ORDER BY physical_time DESC
        """, (entity_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Entity(
                id=row[0],
                entity_id=row[1],
                name=row[2],
                content=row[3],
                physical_time=datetime.fromisoformat(row[4]),
                memory_cache_id=row[5],
                doc_name=row[6] if len(row) > 6 and row[6] is not None else "",  # 向后兼容
                embedding=row[7] if len(row) > 7 else None
            )
            for row in rows
        ]
    
    def _get_entities_with_embeddings(self) -> List[tuple]:
        """
        获取所有实体的最新版本及其embedding
        
        Returns:
            List of (Entity, embedding_array) tuples, embedding_array为None表示没有embedding
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取每个entity_id的最新版本及其embedding
        cursor.execute("""
            SELECT id, entity_id, name, content, physical_time, memory_cache_id, doc_name, embedding
            FROM entities e1
            WHERE e1.physical_time = (
                SELECT MAX(e2.physical_time)
                FROM entities e2
                WHERE e2.entity_id = e1.entity_id
            )
        """)
        
        results = []
        for row in cursor.fetchall():
            # 解析embedding
            embedding_array = None
            if len(row) > 7 and row[7] is not None:
                try:
                    embedding_array = np.frombuffer(row[7], dtype=np.float32)
                except:
                    embedding_array = None
            entity = Entity(
                id=row[0],
                entity_id=row[1],
                name=row[2],
                content=row[3],
                physical_time=datetime.fromisoformat(row[4]),
                memory_cache_id=row[5],
                doc_name=row[6] if len(row) > 6 and row[6] is not None else "",  # 向后兼容
                embedding=row[7] if len(row) > 7 else None
            )
            results.append((entity, embedding_array))
        
        conn.close()
        return results
    
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
        
        query_embedding_array = np.array(query_embedding[0] if isinstance(query_embedding, list) else query_embedding, dtype=np.float32)
        
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
                    embedding_array = np.array(new_embeddings[i] if isinstance(new_embeddings, list) else new_embeddings, dtype=np.float32)
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
        
        # 构建文本：content[:snippet_length]
        text = relation.content[:self.relation_content_snippet_length]
        embedding = self.embedding_client.encode(text)
        
        if embedding is None or len(embedding) == 0:
            return None
        
        # 转换为numpy数组并序列化为BLOB
        embedding_array = np.array(embedding[0] if isinstance(embedding, list) else embedding, dtype=np.float32)
        return embedding_array.tobytes()
    
    def save_relation(self, relation: Relation):
        """保存关系（包含预计算的embedding向量）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 计算embedding
        embedding_blob = self._compute_relation_embedding(relation)
        
        # 更新Relation对象的embedding字段，保持一致性
        relation.embedding = embedding_blob
        
        cursor.execute("""
            INSERT INTO relations (id, relation_id, entity1_absolute_id, entity2_absolute_id, content, physical_time, memory_cache_id, doc_name, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            relation.id,
            relation.relation_id,
            relation.entity1_absolute_id,
            relation.entity2_absolute_id,
            relation.content,
            relation.physical_time.isoformat(),
            relation.memory_cache_id,
            relation.doc_name,
            embedding_blob
        ))
        
        conn.commit()
        conn.close()
    
    def get_relations_by_entities(self, from_entity_id: str, to_entity_id: str) -> List[Relation]:
        """根据两个实体ID获取所有关系（通过entity_id查找，内部转换为绝对ID查询）"""
        # 先通过entity_id获取最新版本的绝对ID
        from_entity = self.get_entity_by_id(from_entity_id)
        to_entity = self.get_entity_by_id(to_entity_id)
        
        if not from_entity or not to_entity:
            return []
        
        # 通过绝对ID查询关系（查询所有包含这些实体的关系，不管版本）
        # 需要查询所有以这些entity_id开头的绝对ID的关系
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取所有具有相同entity_id的实体的绝对ID
        cursor.execute("""
            SELECT id FROM entities WHERE entity_id = ?
        """, (from_entity_id,))
        from_absolute_ids = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("""
            SELECT id FROM entities WHERE entity_id = ?
        """, (to_entity_id,))
        to_absolute_ids = [row[0] for row in cursor.fetchall()]
        
        if not from_absolute_ids or not to_absolute_ids:
            conn.close()
            return []
        
        # 查询关系（无向关系，考虑两个方向）
        # 关系是无向的，需要查询 (entity1, entity2) 和 (entity2, entity1) 两种情况
        placeholders_from = ','.join(['?'] * len(from_absolute_ids))
        placeholders_to = ','.join(['?'] * len(to_absolute_ids))
        
        cursor.execute(f"""
            SELECT id, relation_id, entity1_absolute_id, entity2_absolute_id, content, physical_time, memory_cache_id, doc_name, embedding
            FROM relations
            WHERE (entity1_absolute_id IN ({placeholders_from}) AND entity2_absolute_id IN ({placeholders_to}))
               OR (entity1_absolute_id IN ({placeholders_to}) AND entity2_absolute_id IN ({placeholders_from}))
            ORDER BY physical_time DESC
        """, from_absolute_ids + to_absolute_ids + to_absolute_ids + from_absolute_ids)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Relation(
                id=row[0],
                relation_id=row[1],
                entity1_absolute_id=row[2] or "",
                entity2_absolute_id=row[3] or "",
                content=row[4],
                physical_time=datetime.fromisoformat(row[5]),
                memory_cache_id=row[6],
                doc_name=row[7] if len(row) > 7 and row[7] is not None else "",  # 向后兼容
                embedding=row[8] if len(row) > 8 else None
            )
            for row in rows
        ]
    
    def get_relation_versions(self, relation_id: str) -> List[Relation]:
        """获取关系的所有版本"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, relation_id, entity1_absolute_id, entity2_absolute_id, content, physical_time, memory_cache_id, doc_name, embedding
            FROM relations
            WHERE relation_id = ?
            ORDER BY physical_time DESC
        """, (relation_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Relation(
                id=row[0],
                relation_id=row[1],
                entity1_absolute_id=row[2] or "",
                entity2_absolute_id=row[3] or "",
                content=row[4],
                physical_time=datetime.fromisoformat(row[5]),
                memory_cache_id=row[6],
                doc_name=row[7] if len(row) > 7 and row[7] is not None else "",  # 向后兼容
                embedding=row[8] if len(row) > 8 else None
            )
            for row in rows
        ]
    
    def update_relation_memory_cache_id(self, relation_id: str, memory_cache_id: str):
        """更新关系最新版本的 memory_cache_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取最新版本的id
        cursor.execute("""
            SELECT id FROM relations
            WHERE relation_id = ?
            ORDER BY physical_time DESC
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
        
        conn.close()
    
    def get_self_referential_relations(self) -> Dict[str, List[Dict]]:
        """获取所有自指向的关系（两端指向同一个entity_id），按entity_id分组
        
        自指向关系的定义：关系的两端实体具有相同的entity_id
        这包括两种情况：
        1. entity1_absolute_id == entity2_absolute_id（指向完全相同的版本）
        2. entity1_absolute_id 和 entity2_absolute_id 不同，但它们对应的entity_id相同
        
        Returns:
            字典，key为entity_id，value为该实体的所有自指向关系列表
            每个关系包含：id, relation_id, content, physical_time
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 查找所有自指向的关系（两端entity_id相同）
        cursor.execute("""
            SELECT r.id, r.relation_id, r.content, r.physical_time, e1.entity_id
            FROM relations r
            JOIN entities e1 ON r.entity1_absolute_id = e1.id
            JOIN entities e2 ON r.entity2_absolute_id = e2.id
            WHERE e1.entity_id = e2.entity_id
            ORDER BY e1.entity_id, r.physical_time
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        # 按entity_id分组
        result = {}
        for row in rows:
            relation_id, relation_id_str, content, physical_time, entity_id = row
            if entity_id not in result:
                result[entity_id] = []
            result[entity_id].append({
                'id': relation_id,
                'relation_id': relation_id_str,
                'content': content,
                'physical_time': physical_time
            })
        
        return result
    
    def delete_self_referential_relations(self) -> int:
        """删除所有自指向的关系（两端指向同一个entity_id）
        
        Returns:
            删除的关系数量
        """
        conn = sqlite3.connect(self.db_path)
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
            # 获取所有需要删除的关系ID
            relation_ids = [row[0] for row in rows]
            placeholders = ','.join(['?' for _ in relation_ids])
            
            # 删除这些关系
            cursor.execute(f"""
                DELETE FROM relations
                WHERE id IN ({placeholders})
            """, relation_ids)
            deleted_count = cursor.rowcount
            conn.commit()
        
        conn.close()
        return deleted_count
    
    def get_self_referential_relations_for_entity(self, entity_id: str) -> List[Dict]:
        """获取指定entity_id的自指向关系
        
        Args:
            entity_id: 实体ID
        
        Returns:
            自指向关系列表，每个关系包含：id, relation_id, content, physical_time
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 查找该entity_id的自指向关系
        cursor.execute("""
            SELECT r.id, r.relation_id, r.content, r.physical_time
            FROM relations r
            JOIN entities e1 ON r.entity1_absolute_id = e1.id
            JOIN entities e2 ON r.entity2_absolute_id = e2.id
            WHERE e1.entity_id = ? AND e2.entity_id = ?
            ORDER BY r.physical_time
        """, (entity_id, entity_id))
        
        rows = cursor.fetchall()
        conn.close()
        
        result = []
        for row in rows:
            relation_id, relation_id_str, content, physical_time = row
            result.append({
                'id': relation_id,
                'relation_id': relation_id_str,
                'content': content,
                'physical_time': physical_time
            })
        
        return result
    
    def delete_self_referential_relations_for_entity(self, entity_id: str) -> int:
        """删除指定entity_id的自指向关系
        
        Args:
            entity_id: 实体ID
        
        Returns:
            删除的关系数量
        """
        conn = sqlite3.connect(self.db_path)
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
        
        conn.close()
        return deleted_count
    
    def get_all_entities(self, limit: Optional[int] = None) -> List[Entity]:
        """获取所有实体的最新版本
        
        Args:
            limit: 限制返回的实体数量（按时间倒序），None表示不限制
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取每个 entity_id 的最新版本
        query = """
            SELECT e1.id, e1.entity_id, e1.name, e1.content, e1.physical_time, e1.memory_cache_id, e1.embedding
            FROM entities e1
            INNER JOIN (
                SELECT entity_id, MAX(physical_time) as max_time
                FROM entities
                GROUP BY entity_id
            ) e2 ON e1.entity_id = e2.entity_id AND e1.physical_time = e2.max_time
            ORDER BY e1.physical_time DESC
        """
        
        if limit is not None:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Entity(
                id=row[0],
                entity_id=row[1],
                name=row[2],
                content=row[3],
                physical_time=datetime.fromisoformat(row[4]),
                memory_cache_id=row[5],
                embedding=row[6]
            )
            for row in rows
        ]
    
    def get_all_entities_before_time(self, time_point: datetime, limit: Optional[int] = None) -> List[Entity]:
        """获取指定时间点之前或等于该时间点的所有实体的最新版本
        
        Args:
            time_point: 时间点
            limit: 限制返回的实体数量（按时间倒序），None表示不限制
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取每个 entity_id 在指定时间点之前或等于该时间点的最新版本
        query = """
            SELECT e1.id, e1.entity_id, e1.name, e1.content, e1.physical_time, e1.memory_cache_id, e1.embedding
            FROM entities e1
            INNER JOIN (
                SELECT entity_id, MAX(physical_time) as max_time
                FROM entities
                WHERE physical_time <= ?
                GROUP BY entity_id
            ) e2 ON e1.entity_id = e2.entity_id AND e1.physical_time = e2.max_time
            ORDER BY e1.physical_time DESC
        """
        
        if limit is not None:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (time_point.isoformat(),))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Entity(
                id=row[0],
                entity_id=row[1],
                name=row[2],
                content=row[3],
                physical_time=datetime.fromisoformat(row[4]),
                memory_cache_id=row[5],
                embedding=row[6]
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if time_point:
            # 获取每个relation_id在该时间点之前或等于该时间点的最新版本
            query = """
                SELECT r1.id, r1.relation_id, r1.entity1_absolute_id, r1.entity2_absolute_id, 
                       r1.content, r1.physical_time, r1.memory_cache_id, r1.doc_name, r1.embedding
                FROM relations r1
                INNER JOIN (
                    SELECT relation_id, MAX(physical_time) as max_time
                    FROM relations
                    WHERE (entity1_absolute_id = ? OR entity2_absolute_id = ?)
                    AND physical_time <= ?
                    GROUP BY relation_id
                ) r2 ON r1.relation_id = r2.relation_id 
                    AND r1.physical_time = r2.max_time
                    AND (r1.entity1_absolute_id = ? OR r1.entity2_absolute_id = ?)
                ORDER BY r1.physical_time DESC
            """
            params = (entity_absolute_id, entity_absolute_id, time_point.isoformat(), entity_absolute_id, entity_absolute_id)
        else:
            # 获取每个relation_id的最新版本
            query = """
                SELECT r1.id, r1.relation_id, r1.entity1_absolute_id, r1.entity2_absolute_id, 
                       r1.content, r1.physical_time, r1.memory_cache_id, r1.doc_name, r1.embedding
                FROM relations r1
                INNER JOIN (
                    SELECT relation_id, MAX(physical_time) as max_time
                    FROM relations
                    WHERE entity1_absolute_id = ? OR entity2_absolute_id = ?
                    GROUP BY relation_id
                ) r2 ON r1.relation_id = r2.relation_id 
                    AND r1.physical_time = r2.max_time
                    AND (r1.entity1_absolute_id = ? OR r1.entity2_absolute_id = ?)
                ORDER BY r1.physical_time DESC
            """
            params = (entity_absolute_id, entity_absolute_id, entity_absolute_id, entity_absolute_id)
        
        if limit is not None:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Relation(
                id=row[0],
                relation_id=row[1],
                entity1_absolute_id=row[2] or "",
                entity2_absolute_id=row[3] or "",
                content=row[4],
                physical_time=datetime.fromisoformat(row[5]),
                memory_cache_id=row[6],
                doc_name=row[7] if len(row) > 7 and row[7] is not None else "",  # 向后兼容
                embedding=row[8] if len(row) > 8 else None
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
        # 先获取该实体的所有版本的absolute_id
        versions = self.get_entity_versions(entity_id)
        if not versions:
            return []
        
        # 如果指定了max_version_absolute_id，只取到该版本为止的所有版本
        if max_version_absolute_id:
            # 按时间排序，找到max_version_absolute_id对应的版本
            versions_sorted = sorted(versions, key=lambda v: v.physical_time)
            max_version = None
            for v in versions_sorted:
                if v.id == max_version_absolute_id:
                    max_version = v
                    break
            
            if max_version:
                # 只取到该版本（包含）为止的所有版本
                entity_absolute_ids = [v.id for v in versions_sorted if v.physical_time <= max_version.physical_time]
                # 同时设置time_point为该版本的时间点
                if not time_point:
                    time_point = max_version.physical_time
                else:
                    # 如果已经设置了time_point，取较小值
                    time_point = min(time_point, max_version.physical_time)
            else:
                # 如果找不到指定的版本，使用所有版本
                entity_absolute_ids = [v.id for v in versions]
        else:
            # 收集所有版本的absolute_id
            entity_absolute_ids = [v.id for v in versions]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构建查询：查找所有版本的关系，按relation_id去重
        placeholders = ','.join(['?'] * len(entity_absolute_ids))
        
        if time_point:
            # 获取每个relation_id在该时间点之前或等于该时间点的最新版本
            query = f"""
                SELECT r1.id, r1.relation_id, r1.entity1_absolute_id, r1.entity2_absolute_id, 
                       r1.content, r1.physical_time, r1.memory_cache_id, r1.doc_name, r1.embedding
                FROM relations r1
                INNER JOIN (
                    SELECT relation_id, MAX(physical_time) as max_time
                    FROM relations
                    WHERE (entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders}))
                    AND physical_time <= ?
                    GROUP BY relation_id
                ) r2 ON r1.relation_id = r2.relation_id 
                    AND r1.physical_time = r2.max_time
                    AND (r1.entity1_absolute_id IN ({placeholders}) OR r1.entity2_absolute_id IN ({placeholders}))
                ORDER BY r1.physical_time DESC
            """
            params = tuple(entity_absolute_ids * 2 + [time_point.isoformat()] + entity_absolute_ids * 2)
        else:
            # 获取每个relation_id的最新版本
            query = f"""
            SELECT r1.id, r1.relation_id, r1.entity1_absolute_id, r1.entity2_absolute_id, 
                   r1.content, r1.physical_time, r1.memory_cache_id, r1.doc_name, r1.embedding
            FROM relations r1
            INNER JOIN (
                SELECT relation_id, MAX(physical_time) as max_time
                FROM relations
                WHERE entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})
                GROUP BY relation_id
            ) r2 ON r1.relation_id = r2.relation_id 
                AND r1.physical_time = r2.max_time
                AND (r1.entity1_absolute_id IN ({placeholders}) OR r1.entity2_absolute_id IN ({placeholders}))
            ORDER BY r1.physical_time DESC
            """
            params = tuple(entity_absolute_ids * 4)
        
        if limit is not None:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Relation(
                id=row[0],
                relation_id=row[1],
                entity1_absolute_id=row[2] or "",
                entity2_absolute_id=row[3] or "",
                content=row[4],
                physical_time=datetime.fromisoformat(row[5]),
                memory_cache_id=row[6],
                doc_name=row[7] if len(row) > 7 and row[7] is not None else "",  # 向后兼容
                embedding=row[8] if len(row) > 8 else None
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
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构建查询：查找直接引用这些 entity_absolute_id 的关系边
        placeholders = ','.join(['?'] * len(entity_absolute_ids))
        query = f"""
            SELECT id, relation_id, entity1_absolute_id, entity2_absolute_id, 
                   content, physical_time, memory_cache_id, doc_name, embedding
            FROM relations
            WHERE entity1_absolute_id IN ({placeholders}) OR entity2_absolute_id IN ({placeholders})
            ORDER BY physical_time DESC
        """
        
        params = tuple(entity_absolute_ids * 2)
        cursor.execute(query, params)
        
        rows = cursor.fetchall()
        conn.close()
        
        # 按 relation_id 去重，保留第一个（最新的）
        seen_relation_ids = set()
        result = []
        for row in rows:
            relation_id = row[1]
            if relation_id not in seen_relation_ids:
                seen_relation_ids.add(relation_id)
                result.append(
                    Relation(
                        id=row[0],
                        relation_id=row[1],
                        entity1_absolute_id=row[2] or "",
                        entity2_absolute_id=row[3] or "",
                        content=row[4],
                        physical_time=datetime.fromisoformat(row[5]),
                        memory_cache_id=row[6],
                        doc_name=row[7] if len(row) > 7 and row[7] is not None else "",  # 向后兼容
                        embedding=row[8] if len(row) > 8 else None
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
        
        # 按时间排序
        versions_sorted = sorted(versions, key=lambda v: v.physical_time)
        
        # 找到 max_absolute_id 对应的版本
        max_version = None
        for v in versions_sorted:
            if v.id == max_absolute_id:
                max_version = v
                break
        
        if not max_version:
            # 如果找不到指定的版本，返回空列表
            return []
        
        # 返回从最早版本到该版本（包含）的所有 absolute_id
        result = []
        for v in versions_sorted:
            result.append(v.id)
            if v.id == max_absolute_id:
                break
        
        return result
    
    def get_all_relations(self) -> List[Relation]:
        """获取所有关系的最新版本"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取每个 relation_id 的最新版本
        cursor.execute("""
            SELECT r1.id, r1.relation_id, r1.entity1_absolute_id, r1.entity2_absolute_id,
                   r1.content, r1.physical_time, r1.memory_cache_id, r1.doc_name, r1.embedding
            FROM relations r1
            INNER JOIN (
                SELECT relation_id, MAX(physical_time) as max_time
                FROM relations
                GROUP BY relation_id
            ) r2 ON r1.relation_id = r2.relation_id AND r1.physical_time = r2.max_time
            ORDER BY r1.physical_time DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Relation(
                id=row[0],
                relation_id=row[1],
                entity1_absolute_id=row[2] or "",
                entity2_absolute_id=row[3] or "",
                content=row[4],
                physical_time=datetime.fromisoformat(row[5]),
                memory_cache_id=row[6],
                doc_name=row[7] if len(row) > 7 and row[7] is not None else "",  # 向后兼容
                embedding=row[8] if len(row) > 8 else None
            )
            for row in rows
        ]
    
    def _get_relations_with_embeddings(self) -> List[tuple]:
        """
        获取所有关系的最新版本及其embedding
        
        Returns:
            List of (Relation, embedding_array) tuples, embedding_array为None表示没有embedding
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取每个relation_id的最新版本及其embedding
        cursor.execute("""
            SELECT id, relation_id, entity1_absolute_id, entity2_absolute_id,
                   content, physical_time, memory_cache_id, doc_name, embedding
            FROM relations r1
            WHERE r1.physical_time = (
                SELECT MAX(r2.physical_time)
                FROM relations r2
                WHERE r2.relation_id = r1.relation_id
            )
        """)
        
        results = []
        for row in cursor.fetchall():
            # 解析embedding
            embedding_array = None
            if len(row) > 8 and row[8] is not None:
                try:
                    embedding_array = np.frombuffer(row[8], dtype=np.float32)
                except:
                    embedding_array = None
            relation = Relation(
                id=row[0],
                relation_id=row[1],
                entity1_absolute_id=row[2] or "",
                entity2_absolute_id=row[3] or "",
                content=row[4],
                physical_time=datetime.fromisoformat(row[5]),
                memory_cache_id=row[6],
                doc_name=row[7] if len(row) > 7 and row[7] is not None else "",  # 向后兼容
                embedding=row[8] if len(row) > 8 else None
            )
            results.append((relation, embedding_array))
        
        conn.close()
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
        
        query_embedding_array = np.array(query_embedding[0] if isinstance(query_embedding, list) else query_embedding, dtype=np.float32)
        
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
    
    def get_memory_cache_text(self, cache_id: str) -> Optional[str]:
        """获取记忆缓存对应的原始文本内容
        
        Args:
            cache_id: 缓存记忆的ID
            
        Returns:
            原始文本内容，如果不存在则返回None
        """
        # 先尝试从新的json文件夹加载
        metadata_path = self.cache_json_dir / f"{cache_id}.json"
        # 如果不存在，尝试从旧路径加载（向后兼容）
        if not metadata_path.exists():
            metadata_path = self.cache_dir / f"{cache_id}.json"
            if not metadata_path.exists():
                return None
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return metadata.get('text', None)
    
    def find_related_entities_by_embedding(self, similarity_threshold: float = 0.7, 
                                           max_candidates: int = 5,
                                           use_mixed_search: bool = True,
                                           content_snippet_length: int = 50,
                                           progress_callback: Optional[callable] = None) -> Dict[str, set]:
        """
        使用混合检索方式找到每个实体的关联实体
        
        每个 entity_id 只使用最新版本（physical_time 最新的）的实体来计算相似度
        
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
        if not source_entity_ids:
            return {"entities_updated": 0, "relations_updated": 0}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        entities_updated = 0
        relations_updated = 0
        
        try:
            # 1. 先获取所有源实体的版本数量（在更新之前，用于验证）
            source_version_counts = {}
            for source_id in source_entity_ids:
                cursor.execute("""
                    SELECT COUNT(*) FROM entities
                    WHERE entity_id = ?
                """, (source_id,))
                count = cursor.fetchone()[0]
                source_version_counts[source_id] = count
            
            # 2. 更新entities表中的所有entity_id记录
            # 这会更新所有使用source_entity_id的记录，包括所有版本
            for source_id in source_entity_ids:
                cursor.execute("""
                    UPDATE entities
                    SET entity_id = ?
                    WHERE entity_id = ?
                """, (target_entity_id, source_id))
                entities_updated += cursor.rowcount
            
            # 2.5. 验证：确保所有源实体的版本都被更新了
            # 检查是否还有任何源entity_id的记录残留
            for source_id in source_entity_ids:
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
                ORDER BY physical_time DESC
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
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
        
        return {
            "entities_updated": entities_updated,
            "relations_updated": relations_updated,
            "target_entity_id": target_entity_id,
            "merged_source_ids": source_entity_ids
        }
    
    def get_entity_version_count(self, entity_id: str) -> int:
        """获取指定entity_id的版本数量
        
        Args:
            entity_id: 实体ID
            
        Returns:
            版本数量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM entities WHERE entity_id = ?
        """, (entity_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def get_entity_version_counts(self, entity_ids: List[str]) -> Dict[str, int]:
        """批量获取多个entity_id的版本数量
        
        Args:
            entity_ids: 实体ID列表
            
        Returns:
            Dict[entity_id, version_count]
        """
        if not entity_ids:
            return {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 使用IN子句批量查询
        placeholders = ','.join(['?'] * len(entity_ids))
        cursor.execute(f"""
            SELECT entity_id, COUNT(*) as version_count
            FROM entities
            WHERE entity_id IN ({placeholders})
            GROUP BY entity_id
        """, entity_ids)
        
        rows = cursor.fetchall()
        conn.close()
        
        return {row[0]: row[1] for row in rows}