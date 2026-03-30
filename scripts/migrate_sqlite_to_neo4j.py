#!/usr/bin/env python3
"""
SQLite → Neo4j + sqlite-vec 数据迁移脚本。

将现有 SQLite 存储的数据迁移到 Neo4j 图数据库 + sqlite-vec 向量存储。

Usage:
    python scripts/migrate_sqlite_to_neo4j.py --sqlite-path ./graph/default/graph.db \
        --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password tmg2024secure \
        --vector-path ./graph/default/vectors.db

    # 使用环境变量
    export NEO4J_PASSWORD=tmg2024secure
    python scripts/migrate_sqlite_to_neo4j.py --sqlite-path ./graph/default/graph.db

    # 只迁移向量（Neo4j 已有数据）
    python scripts/migrate_sqlite_to_neo4j.py --sqlite-path ./graph/default/graph.db --vectors-only

    # 试运行（不实际写入）
    python scripts/migrate_sqlite_to_neo4j.py --sqlite-path ./graph/default/graph.db --dry-run
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="SQLite → Neo4j + sqlite-vec 数据迁移")
    parser.add_argument("--sqlite-path", required=True, help="SQLite 数据库路径 (graph.db)")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j 用户名")
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", "tmg2024secure"),
                        help="Neo4j 密码")
    parser.add_argument("--vector-path", default=None, help="sqlite-vec 向量库路径（默认同目录 vectors.db）")
    parser.add_argument("--vector-dim", type=int, default=1024, help="向量维度")
    parser.add_argument("--batch-size", type=int, default=500, help="每批处理数量")
    parser.add_argument("--vectors-only", action="store_true", help="只迁移向量数据")
    parser.add_argument("--dry-run", action="store_true", help="试运行，不实际写入")
    return parser.parse_args()


def count_sqlite_data(conn):
    """统计 SQLite 数据量。"""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM entities")
    entity_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(DISTINCT entity_id) FROM entities")
    unique_entity_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM relations")
    relation_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(DISTINCT relation_id) FROM relations")
    unique_relation_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM entity_redirects")
    redirect_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM entities WHERE embedding IS NOT NULL")
    entity_emb_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM relations WHERE embedding IS NOT NULL")
    relation_emb_count = cursor.fetchone()[0]
    return {
        "entities": entity_count,
        "unique_entities": unique_entity_count,
        "relations": relation_count,
        "unique_relations": unique_relation_count,
        "redirects": redirect_count,
        "entity_embeddings": entity_emb_count,
        "relation_embeddings": relation_emb_count,
    }


def migrate_entities(conn, driver, batch_size=500, dry_run=False):
    """迁移实体数据到 Neo4j。"""
    import neo4j

    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, entity_id, name, content, event_time, processed_time,
               memory_cache_id, source_document
        FROM entities
        ORDER BY processed_time ASC
    """)
    rows = cursor.fetchall()
    total = len(rows)

    print(f"  迁移 {total} 条实体记录...")

    with driver.session() as session:
        for i in range(0, total, batch_size):
            batch = rows[i:i + batch_size]
            if dry_run:
                print(f"    [DRY-RUN] 跳过 batch {i // batch_size + 1} ({len(batch)} 条)")
                continue

            tx = session.begin_transaction()
            try:
                for row in batch:
                    tx.run(
                        """
                        MERGE (e:Entity {uuid: $uuid})
                        SET e.entity_id = $entity_id,
                            e.name = $name,
                            e.content = $content,
                            e.event_time = datetime($event_time),
                            e.processed_time = datetime($processed_time),
                            e.memory_cache_id = $cache_id,
                            e.source_document = $source
                        """,
                        uuid=row[0],
                        entity_id=row[1],
                        name=row[2],
                        content=row[3],
                        event_time=row[4],
                        processed_time=row[5],
                        cache_id=row[6],
                        source=row[7] or "",
                    )
                tx.commit()
                print(f"    batch {i // batch_size + 1}: {min(i + batch_size, total)}/{total}")
            except Exception as e:
                tx.rollback()
                print(f"    batch {i // batch_size + 1} 失败: {e}")
                raise

    print(f"  实体迁移完成: {total} 条")


def migrate_relations(conn, driver, batch_size=500, dry_run=False):
    """迁移关系数据到 Neo4j。"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, relation_id, entity1_absolute_id, entity2_absolute_id,
               content, event_time, processed_time, memory_cache_id, source_document
        FROM relations
        ORDER BY processed_time ASC
    """)
    rows = cursor.fetchall()
    total = len(rows)

    print(f"  迁移 {total} 条关系记录...")

    with driver.session() as session:
        for i in range(0, total, batch_size):
            batch = rows[i:i + batch_size]
            if dry_run:
                print(f"    [DRY-RUN] 跳过 batch {i // batch_size + 1} ({len(batch)} 条)")
                continue

            tx = session.begin_transaction()
            try:
                for row in batch:
                    tx.run(
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
                        uuid=row[0],
                        relation_id=row[1],
                        e1_abs=row[2] or "",
                        e2_abs=row[3] or "",
                        content=row[4],
                        event_time=row[5],
                        processed_time=row[6],
                        cache_id=row[7],
                        source=row[8] or "",
                    )
                    # 创建 RELATES_TO 边
                    if row[2] and row[3]:
                        tx.run(
                            """
                            MATCH (e1:Entity {uuid: $e1_abs})
                            MATCH (e2:Entity {uuid: $e2_abs})
                            MERGE (e1)-[rel:RELATES_TO {relation_uuid: $uuid}]->(e2)
                            SET rel.fact = $content
                            """,
                            e1_abs=row[2],
                            e2_abs=row[3],
                            uuid=row[0],
                            content=row[4],
                        )
                tx.commit()
                print(f"    batch {i // batch_size + 1}: {min(i + batch_size, total)}/{total}")
            except Exception as e:
                tx.rollback()
                print(f"    batch {i // batch_size + 1} 失败: {e}")
                raise

    print(f"  关系迁移完成: {total} 条")


def migrate_redirects(conn, driver, dry_run=False):
    """迁移实体重定向数据到 Neo4j。"""
    cursor = conn.cursor()
    cursor.execute("SELECT source_entity_id, target_entity_id, updated_at FROM entity_redirects")
    rows = cursor.fetchall()
    total = len(rows)

    print(f"  迁移 {total} 条重定向记录...")

    if dry_run:
        print(f"    [DRY-RUN] 跳过 {total} 条重定向")
        return

    with driver.session() as session:
        for row in rows:
            session.run(
                """
                MERGE (red:EntityRedirect {source_id: $sid})
                SET red.target_id = $tid, red.updated_at = $now
                """,
                sid=row[0],
                tid=row[1],
                now=row[2],
            )

    print(f"  重定向迁移完成: {total} 条")


def migrate_vectors(conn, vector_path, dim=1024, batch_size=500, dry_run=False):
    """迁移 embedding 向量到 sqlite-vec。"""
    import sqlite_vec
    import struct

    print(f"  迁移向量到 {vector_path} (dim={dim})...")

    if dry_run:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entities WHERE embedding IS NOT NULL")
        entity_emb_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM relations WHERE embedding IS NOT NULL")
        relation_emb_count = cursor.fetchone()[0]
        print(f"    [DRY-RUN] 跳过 {entity_emb_count} 条实体向量, {relation_emb_count} 条关系向量")
        return

    os.makedirs(os.path.dirname(vector_path) or ".", exist_ok=True)
    vconn = sqlite3.connect(vector_path)
    vconn.enable_load_extension(True)
    sqlite_vec.load(vconn)
    vconn.enable_load_extension(False)
    vconn.execute("PRAGMA journal_mode=WAL")

    for table in ("entity_vectors", "relation_vectors", "episode_vectors"):
        vconn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} "
            f"USING vec0(uuid TEXT PRIMARY KEY, embedding float[{dim}])"
        )

    # 迁移实体向量
    cursor = conn.cursor()
    cursor.execute("SELECT id, embedding FROM entities WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    print(f"    迁移 {len(rows)} 条实体向量...")
    batch = []
    for uuid, emb_blob in rows:
        if emb_blob:
            try:
                emb_list = list(struct.unpack(f"{len(emb_blob) // 4}f", emb_blob))
                batch.append((uuid, emb_list))
                if len(batch) >= batch_size:
                    vconn.executemany(
                        f"INSERT OR REPLACE INTO entity_vectors(uuid, embedding) VALUES(?, ?)",
                        [(u, struct.pack(f"{len(e)}f", *e)) for u, e in batch],
                    )
                    vconn.commit()
                    batch = []
            except Exception as e:
                print(f"      跳过实体 {uuid}: {e}")
    if batch:
        vconn.executemany(
            f"INSERT OR REPLACE INTO entity_vectors(uuid, embedding) VALUES(?, ?)",
            [(u, struct.pack(f"{len(e)}f", *e)) for u, e in batch],
        )
        vconn.commit()

    # 迁移关系向量
    cursor.execute("SELECT id, embedding FROM relations WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    print(f"    迁移 {len(rows)} 条关系向量...")
    batch = []
    for uuid, emb_blob in rows:
        if emb_blob:
            try:
                emb_list = list(struct.unpack(f"{len(emb_blob) // 4}f", emb_blob))
                batch.append((uuid, emb_list))
                if len(batch) >= batch_size:
                    vconn.executemany(
                        f"INSERT OR REPLACE INTO relation_vectors(uuid, embedding) VALUES(?, ?)",
                        [(u, struct.pack(f"{len(e)}f", *e)) for u, e in batch],
                    )
                    vconn.commit()
                    batch = []
            except Exception as e:
                print(f"      跳过关系 {uuid}: {e}")
    if batch:
        vconn.executemany(
            f"INSERT OR REPLACE INTO relation_vectors(uuid, embedding) VALUES(?, ?)",
            [(u, struct.pack(f"{len(e)}f", *e)) for u, e in batch],
        )
        vconn.commit()

    vconn.close()
    print(f"  向量迁移完成")


def main():
    args = parse_args()

    sqlite_path = Path(args.sqlite_path)
    if not sqlite_path.exists():
        print(f"错误: SQLite 数据库不存在: {sqlite_path}")
        sys.exit(1)

    # 连接 SQLite
    print(f"连接 SQLite: {sqlite_path}")
    conn = sqlite3.connect(str(sqlite_path))

    # 统计数据
    stats = count_sqlite_data(conn)
    print(f"SQLite 数据统计:")
    print(f"  实体: {stats['entities']} 条 ({stats['unique_entities']} 个唯一 entity_id)")
    print(f"  关系: {stats['relations']} 条 ({stats['unique_relations']} 个唯一 relation_id)")
    print(f"  重定向: {stats['redirects']} 条")
    print(f"  实体向量: {stats['entity_embeddings']} 条")
    print(f"  关系向量: {stats['relation_embeddings']} 条")
    print()

    if args.dry_run:
        print("[DRY-RUN 模式 - 不实际写入任何数据]")
        print()

    vector_path = args.vector_path or str(sqlite_path.parent / "vectors.db")

    if not args.vectors_only:
        # 连接 Neo4j
        print(f"连接 Neo4j: {args.neo4j_uri}")
        import neo4j
        driver = neo4j.GraphDatabase.driver(
            args.neo4j_uri,
            auth=(args.neo4j_user, args.neo4j_password),
        )
        driver.verify_connectivity()
        print("  Neo4j 连接成功")
        print()

        # 创建约束和索引
        print("创建 Neo4j 约束和索引...")
        constraints = [
            "CREATE CONSTRAINT entity_uuid IF NOT EXISTS FOR (e:Entity) REQUIRE e.uuid IS UNIQUE",
            "CREATE CONSTRAINT relation_uuid IF NOT EXISTS FOR (r:Relation) REQUIRE r.uuid IS UNIQUE",
            "CREATE CONSTRAINT redirect_source IF NOT EXISTS FOR (red:EntityRedirect) REQUIRE red.source_id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX entity_entity_id IF NOT EXISTS FOR (e:Entity) ON (e.entity_id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_processed_time IF NOT EXISTS FOR (e:Entity) ON (e.processed_time)",
            "CREATE INDEX relation_relation_id IF NOT EXISTS FOR (r:Relation) ON (r.relation_id)",
            "CREATE INDEX relation_processed_time IF NOT EXISTS FOR (r:Relation) ON (r.processed_time)",
        ]
        with driver.session() as session:
            for c in constraints:
                try:
                    session.run(c)
                except Exception as e:
                    print(f"    跳过约束: {e}")
            for idx in indexes:
                try:
                    session.run(idx)
                except Exception as e:
                    print(f"    跳过索引: {e}")
        print("  约束和索引创建完成")
        print()

        # 迁移数据
        start = time.time()
        migrate_entities(conn, driver, batch_size=args.batch_size, dry_run=args.dry_run)
        print()
        migrate_relations(conn, driver, batch_size=args.batch_size, dry_run=args.dry_run)
        print()
        migrate_redirects(conn, driver, dry_run=args.dry_run)
        elapsed = time.time() - start
        print(f"\nNeo4j 数据迁移耗时: {elapsed:.1f}s")
        driver.close()

    # 迁移向量
    print()
    start = time.time()
    migrate_vectors(conn, vector_path, dim=args.vector_dim, batch_size=args.batch_size, dry_run=args.dry_run)
    elapsed = time.time() - start
    print(f"向量迁移耗时: {elapsed:.1f}s")

    conn.close()
    print("\n迁移完成!")


if __name__ == "__main__":
    main()
