"""
VectorStore: 基于 sqlite-vec 的向量存储与 ANN 搜索后端。

借鉴 Graphiti (Zep) 的设计思路，将 embedding 向量存储与图结构存储分离。
图结构由 Neo4j 管理，向量搜索由 sqlite-vec 完成。

架构：
    Neo4j  → 存储 Entity/Relation/Episode 节点及其属性
    sqlite-vec → 存储 embedding 向量，提供 KNN 搜索

sqlite-vec 使用 vec0 虚拟表，支持 L2 距离搜索（当前版本为暴力搜索，
后续版本将支持 HNSW/IVF 索引）。
"""

import logging
import os
import sqlite3
import struct
import threading
from typing import Dict, List, Optional, Tuple

import sqlite_vec

logger = logging.getLogger(__name__)

# sqlite-vec 的 KNN 搜索使用 L2 距离。
# 余弦相似度可以通过 L2 归一化向量后用 L2 距离近似：
#   cosine_sim(a, b) = 1 - ||a-b||^2 / 2  (when ||a||=||b||=1)
# 因此要求存入的向量已经 L2 归一化。

TABLES = ("entity_vectors", "relation_vectors", "episode_vectors")


def _floats_to_bytes(vec: List[float]) -> bytes:
    """将 float 列表序列化为 sqlite-vec 要求的 bytes 格式。"""
    return struct.pack(f"{len(vec)}f", *vec)


def _bytes_to_floats(data: bytes) -> List[float]:
    """将 sqlite-vec 返回的 bytes 反序列化为 float 列表。"""
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


class VectorStore:
    """基于 sqlite-vec 的向量存储与 KNN 搜索。

    线程安全：内部使用 threading.local 管理每个线程的连接，
    与 StorageManager 的连接管理模式保持一致。

    Usage:
        vs = VectorStore("graph/vectors.db", dim=1024)
        vs.upsert("entity_vectors", "uuid-123", [0.1, 0.2, ...])
        results = vs.search("entity_vectors", [0.1, 0.2, ...], limit=10)
        # results: [("uuid-123", 0.05), ...]
    """

    def __init__(self, db_path: str, dim: int = 1024):
        self.db_path = str(db_path)
        self.dim = dim
        self._local = threading.local()
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_tables()

    def _get_conn(self) -> sqlite3.Connection:
        """获取当前线程的 sqlite-vec 连接（惰性创建）。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.execute("SELECT 1")
                return conn
            except Exception:
                conn.close()
                self._local.conn = None

        _conn = sqlite3.connect(self.db_path)
        _conn.enable_load_extension(True)
        sqlite_vec.load(_conn)
        _conn.enable_load_extension(False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA busy_timeout=30000")

        # 确保虚拟表存在
        for table in TABLES:
            _conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} "
                f"USING vec0(uuid TEXT PRIMARY KEY, embedding float[{self.dim}])"
            )

        self._local.conn = _conn
        return _conn

    def _init_tables(self):
        """初始化阶段确保虚拟表存在（连接线程安全由 _get_conn 保证）。"""
        conn = self._get_conn()
        conn.commit()

    def upsert(self, table: str, uuid: str, embedding: List[float]) -> None:
        """插入或更新一条向量。

        Args:
            table: 目标虚拟表名（entity_vectors / relation_vectors / episode_vectors）
            uuid: 向量的唯一标识（与 Neo4j 节点 uuid 对应）
            embedding: float 列表，应已 L2 归一化
        """
        conn = self._get_conn()
        emb_bytes = _floats_to_bytes(embedding)
        # vec0 虚拟表不支持 INSERT OR REPLACE，需先 DELETE 再 INSERT
        conn.execute(f"DELETE FROM {table} WHERE uuid = ?", (uuid,))
        conn.execute(
            f"INSERT INTO {table}(uuid, embedding) VALUES(?, ?)",
            (uuid, emb_bytes),
        )
        conn.commit()

    def upsert_batch(self, table: str, items: List[Tuple[str, List[float]]]) -> None:
        """批量插入或更新向量。"""
        if not items:
            return
        conn = self._get_conn()
        # vec0 虚拟表不支持 INSERT OR REPLACE，需先批量删除再插入
        uuids = [(uuid,) for uuid, _ in items]
        conn.executemany(f"DELETE FROM {table} WHERE uuid = ?", uuids)
        rows = [(uuid, _floats_to_bytes(emb)) for uuid, emb in items]
        conn.executemany(
            f"INSERT INTO {table}(uuid, embedding) VALUES(?, ?)",
            rows,
        )
        conn.commit()

    def search(
        self, table: str, query_embedding: List[float], limit: int = 10
    ) -> List[Tuple[str, float]]:
        """KNN 搜索：返回与查询向量最相似的 top-k 条目。

        Args:
            table: 目标虚拟表名
            query_embedding: 查询向量（应已 L2 归一化）
            limit: 返回的最大条目数

        Returns:
            [(uuid, distance), ...] 按 distance 升序排列。
            distance 为 L2 距离的平方，越小越相似。
        """
        conn = self._get_conn()
        query_bytes = _floats_to_bytes(query_embedding)
        cursor = conn.execute(
            f"SELECT uuid, distance FROM {table} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (query_bytes, limit),
        )
        results = cursor.fetchall()
        return [(row[0], row[1]) for row in results]

    def delete(self, table: str, uuid: str) -> None:
        """删除指定 uuid 的向量。"""
        conn = self._get_conn()
        conn.execute(f"DELETE FROM {table} WHERE uuid = ?", (uuid,))
        conn.commit()

    def delete_batch(self, table: str, uuids: List[str]) -> None:
        """批量删除向量。"""
        if not uuids:
            return
        conn = self._get_conn()
        conn.executemany(f"DELETE FROM {table} WHERE uuid = ?", [(u,) for u in uuids])
        conn.commit()

    def get(self, table: str, uuid: str) -> Optional[List[float]]:
        """获取指定 uuid 的向量。"""
        conn = self._get_conn()
        cursor = conn.execute(
            f"SELECT embedding FROM {table} WHERE uuid = ?", (uuid,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _bytes_to_floats(row[0])

    def get_batch(self, table: str, uuids: List[str]) -> Dict[str, List[float]]:
        """批量获取向量。"""
        if not uuids:
            return {}
        conn = self._get_conn()
        placeholders = ",".join("?" * len(uuids))
        cursor = conn.execute(
            f"SELECT uuid, embedding FROM {table} WHERE uuid IN ({placeholders})",
            uuids,
        )
        return {row[0]: _bytes_to_floats(row[1]) for row in cursor}

    def close(self) -> None:
        """关闭当前线程的连接。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception as _e:
                logger.debug("关闭向量数据库连接失败: %s", _e)
            self._local.conn = None