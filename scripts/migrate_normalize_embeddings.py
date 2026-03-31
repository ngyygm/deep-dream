#!/usr/bin/env python3
"""
迁移脚本：将已有的 embedding 向量做 L2 归一化。

KNN 搜索要求存入的向量已 L2 归一化（cosine_sim = 1 - l2_dist²/2）。
此脚本遍历 sqlite-vec 中所有 entity_vectors 和 relation_vectors，对每个向量
做 L2 归一化后回写。

Usage:
    python scripts/migrate_normalize_embeddings.py
    python scripts/migrate_normalize_embeddings.py --dry-run   # 只统计不写入
"""

import argparse
import json
import os
import sqlite3
import struct
import sys
import time
from pathlib import Path

import sqlite_vec
import numpy as np

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_vector_db_path() -> str:
    """从 service_config.json 获取 vector store 路径。

    搜索顺序：
    1. {storage_path}/{default_graph_id}/vectors.db
    2. {storage_path}/vectors.db
    """
    config_path = Path(__file__).resolve().parent.parent / "service_config.json"
    with open(config_path) as f:
        config = json.load(f)
    storage_path = Path(config["storage_path"])

    # 尝试常见的子目录
    for subdir in ["default"]:
        candidate = storage_path / subdir / "vectors.db"
        if candidate.exists():
            return str(candidate)

    return str(storage_path / "vectors.db")


def get_dim(db_path: str) -> int:
    """通过读取一条记录推断向量维度。"""
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    try:
        cursor = conn.execute("SELECT embedding FROM entity_vectors LIMIT 1")
        row = cursor.fetchone()
        if row:
            return len(row[0]) // 4
    except Exception:
        pass
    conn.close()
    return 1024  # 默认


def normalize_table(conn, table: str, dim: int, dry_run: bool = False) -> dict:
    """归一化指定表中的所有向量。"""
    cursor = conn.execute(f"SELECT uuid, embedding FROM {table}")
    rows = cursor.fetchall()

    stats = {"total": len(rows), "normalized": 0, "already_normalized": 0, "zero_vectors": 0}

    if dry_run:
        # 只统计
        for uuid, emb_bytes in rows:
            arr = np.frombuffer(emb_bytes, dtype=np.float32).copy()
            norm = np.linalg.norm(arr)
            if norm == 0:
                stats["zero_vectors"] += 1
            elif abs(norm - 1.0) < 1e-4:
                stats["already_normalized"] += 1
            else:
                stats["normalized"] += 1
        return stats

    # 批量归一化 + 回写
    batch = []
    batch_size = 500
    for uuid, emb_bytes in rows:
        arr = np.frombuffer(emb_bytes, dtype=np.float32).copy()
        norm = np.linalg.norm(arr)
        if norm == 0:
            stats["zero_vectors"] += 1
            continue
        if abs(norm - 1.0) < 1e-4:
            stats["already_normalized"] += 1
            continue

        arr = arr / norm
        emb_bytes_new = struct.pack(f"{len(arr)}f", *arr)
        batch.append((emb_bytes_new, uuid))
        stats["normalized"] += 1

        if len(batch) >= batch_size:
            conn.executemany(
                f"UPDATE {table} SET embedding = ? WHERE uuid = ?",
                batch,
            )
            conn.commit()
            batch = []

    if batch:
        conn.executemany(
            f"UPDATE {table} SET embedding = ? WHERE uuid = ?",
            batch,
        )
        conn.commit()

    return stats


def main():
    parser = argparse.ArgumentParser(description="归一化 sqlite-vec 中的 embedding 向量")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写入")
    parser.add_argument("--db-path", default=None, help="vectors.db 路径（默认从 service_config.json 读取）")
    args = parser.parse_args()

    db_path = args.db_path or get_vector_db_path()
    if not os.path.exists(db_path):
        print(f"错误: 向量数据库不存在: {db_path}")
        sys.exit(1)

    print(f"向量数据库: {db_path}")

    dim = get_dim(db_path)
    print(f"向量维度: {dim}")

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute("PRAGMA journal_mode=WAL")

    tables = ["entity_vectors", "relation_vectors"]
    total_start = time.perf_counter()

    for table in tables:
        print(f"\n{'='*50}")
        action = "统计" if args.dry_run else "归一化"
        print(f"  {action}: {table}")
        print(f"{'='*50}")

        t0 = time.perf_counter()
        stats = normalize_table(conn, table, dim, dry_run=args.dry_run)
        elapsed = time.perf_counter() - t0

        print(f"  总数: {stats['total']}")
        print(f"  需要归一化: {stats['normalized']}")
        print(f"  已归一化: {stats['already_normalized']}")
        print(f"  零向量: {stats['zero_vectors']}")
        print(f"  耗时: {elapsed:.2f}s")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n总耗时: {total_elapsed:.2f}s")

    conn.close()
    print("完成!")


if __name__ == "__main__":
    main()
