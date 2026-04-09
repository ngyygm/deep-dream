#!/usr/bin/env python3
"""
Neo4j 属性重命名迁移：entity_id → family_id, relation_id → family_id

在 Neo4j 数据库中重命名节点属性和索引，配合代码层面的全局重命名。

Usage:
    python scripts/migrate_rename_family_id.py --config service_config.json

    # 试运行（不实际写入）
    python scripts/migrate_rename_family_id.py --config service_config.json --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="Neo4j 属性重命名：entity_id/relation_id → family_id")
    parser.add_argument("--config", required=True, help="service_config.json 路径")
    parser.add_argument("--neo4j-uri", default=None, help="覆盖 Neo4j URI")
    parser.add_argument("--neo4j-user", default=None, help="覆盖 Neo4j 用户名")
    parser.add_argument("--neo4j-password", default=None, help="覆盖 Neo4j 密码")
    parser.add_argument("--dry-run", action="store_true", help="试运行，只输出将要执行的语句")
    return parser.parse_args()


def get_neo4j_driver(uri, user, password):
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver


def run_migration(driver, dry_run=False):
    """执行属性重命名和索引重建。"""

    # Step 1: 重命名 Entity 节点属性
    entity_rename_cypher = """
    MATCH (e:Entity)
    WHERE e.entity_id IS NOT NULL AND e.family_id IS NULL
    SET e.family_id = e.entity_id
    REMOVE e.entity_id
    RETURN count(e) AS renamed
    """

    # Step 2: 重命名 Relation 节点属性
    relation_rename_cypher = """
    MATCH (r:Relation)
    WHERE r.relation_id IS NOT NULL AND r.family_id IS NULL
    SET r.family_id = r.relation_id
    REMOVE r.relation_id
    RETURN count(r) AS renamed
    """

    # Step 3: 重命名 ContentPatch 节点属性
    patch_rename_cypher = """
    MATCH (cp:ContentPatch)
    WHERE cp.target_entity_id IS NOT NULL AND cp.target_family_id IS NULL
    SET cp.target_family_id = cp.target_entity_id
    REMOVE cp.target_entity_id
    RETURN count(cp) AS renamed
    """

    # Step 4: 重建索引
    index_statements = [
        # 删除旧索引
        "DROP INDEX entity_entity_id IF EXISTS",
        "DROP INDEX relation_relation_id IF EXISTS",
        "DROP INDEX content_patch_entity IF EXISTS",
        # 创建新索引
        "CREATE INDEX entity_family_id IF NOT EXISTS FOR (e:Entity) ON (e.family_id)",
        "CREATE INDEX relation_family_id IF NOT EXISTS FOR (r:Relation) ON (r.family_id)",
        "CREATE INDEX content_patch_family IF NOT EXISTS FOR (cp:ContentPatch) ON (cp.target_family_id)",
    ]

    migrations = [
        ("Entity: entity_id → family_id", entity_rename_cypher),
        ("Relation: relation_id → family_id", relation_rename_cypher),
        ("ContentPatch: target_entity_id → target_family_id", patch_rename_cypher),
    ]

    with driver.session() as session:
        # 执行属性重命名
        for label, cypher in migrations:
            if dry_run:
                print(f"[DRY RUN] {label}")
                print(f"  Cypher: {cypher.strip()}")
                continue
            result = session.run(cypher)
            record = result.single()
            count = record["renamed"] if record else 0
            print(f"[OK] {label}: {count} nodes updated")

        # 重建索引
        for stmt in index_statements:
            if dry_run:
                print(f"[DRY RUN] {stmt}")
                continue
            try:
                session.run(stmt)
                print(f"[OK] {stmt}")
            except Exception as e:
                print(f"[WARN] {stmt}: {e}")

    print("\n迁移完成！" if not dry_run else "\n试运行完成（未实际写入）。")


def main():
    args = parse_args()

    # 读取配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误：配置文件不存在: {config_path}")
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    storage = config.get("storage", {})
    if storage.get("backend") != "neo4j":
        print("当前存储后端不是 Neo4j，跳过迁移。")
        return 0

    neo4j_cfg = storage.get("neo4j", {})
    uri = args.neo4j_uri or neo4j_cfg.get("uri", "bolt://localhost:7687")
    user = args.neo4j_user or neo4j_cfg.get("user", "neo4j")
    password = args.neo4j_password or neo4j_cfg.get("password", "")

    if not password:
        password = os.environ.get("NEO4J_PASSWORD", "")
    if not password:
        print("错误：未提供 Neo4j 密码。使用 --neo4j-password 或设置 NEO4J_PASSWORD 环境变量。")
        return 1

    try:
        driver = get_neo4j_driver(uri, user, password)
    except Exception as e:
        print(f"错误：无法连接 Neo4j: {e}")
        return 1

    try:
        run_migration(driver, dry_run=args.dry_run)
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
