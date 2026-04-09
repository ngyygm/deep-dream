#!/usr/bin/env python3
"""
构建 Episode 节点和 MENTIONS 边的迁移脚本。

从 docs/ 目录的 meta.json 创建 Neo4j :Episode 节点，
从 SQLite entities.episode_id 创建 [:MENTIONS] 边到关联 Entity。

Usage:
    python scripts/migrate_build_episodes.py
    python scripts/migrate_build_episodes.py --dry-run
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import neo4j


def parse_args():
    parser = argparse.ArgumentParser(description="构建 Episode 节点和 MENTIONS 边")
    parser.add_argument("--sqlite-path", default="graph/default/graph.db")
    parser.add_argument("--docs-dir", default="graph/default/docs")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", "tmg2024secure"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    docs_dir = Path(args.sqlite_path).parent / "docs" if not args.docs_dir else Path(args.docs_dir)
    sqlite_path = Path(args.sqlite_path)

    if not docs_dir.is_dir():
        print(f"ERROR: docs directory not found: {docs_dir}")
        sys.exit(1)
    if not sqlite_path.is_file():
        print(f"ERROR: SQLite database not found: {sqlite_path}")
        sys.exit(1)

    # --- Phase 1: 从 docs/ 构建 Episode 节点 ---
    print("=" * 60)
    print("Phase 1: Building Episode nodes from docs/")
    print("=" * 60)

    meta_files = sorted(docs_dir.glob("*/meta.json"))
    print(f"Found {len(meta_files)} doc directories")

    episodes = []
    for mf in meta_files:
        try:
            meta = json.loads(mf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  WARN: failed to read {mf}: {e}")
            continue

        cache_id = meta.get("absolute_id") or meta.get("id")
        if not cache_id:
            continue

        cache_md = mf.parent / "cache.md"
        content = ""
        if cache_md.exists():
            try:
                content = cache_md.read_text(encoding="utf-8")[:8000]
            except Exception:
                pass

        episodes.append({
            "uuid": cache_id,
            "content": content,
            "event_time": meta.get("event_time", ""),
            "source_document": meta.get("source_document", meta.get("doc_name", "")),
            "doc_hash": meta.get("doc_hash", ""),
        })

    print(f"Parsed {len(episodes)} episodes")

    # --- Phase 2: 从 SQLite 构建 MENTIONS 映射 ---
    print()
    print("=" * 60)
    print("Phase 2: Building MENTIONS edges from SQLite")
    print("=" * 60)

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # cache_id -> [entity_absolute_id, ...]
    mentions_map: dict[str, list[str]] = {}
    cursor.execute("SELECT DISTINCT id, entity_id, episode_id FROM entities")
    rows = cursor.fetchall()
    print(f"Found {len(rows)} entity rows")

    for row in rows:
        cache_id = row["episode_id"]
        entity_abs_id = row["id"]
        if cache_id and entity_abs_id:
            mentions_map.setdefault(cache_id, []).append(entity_abs_id)

    print(f"Built mentions for {len(mentions_map)} unique cache_ids")
    conn.close()

    # --- Phase 3: 写入 Neo4j ---
    print()
    print("=" * 60)
    print("Phase 3: Writing to Neo4j")
    print("=" * 60)

    if args.dry_run:
        print("DRY RUN - skipping Neo4j writes")
        print(f"  Would create {len(episodes)} Episode nodes")
        total_mentions = sum(len(v) for v in mentions_map.values())
        print(f"  Would create ~{total_mentions} MENTIONS edges")
        return

    driver = neo4j.GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    driver.verify_connectivity()

    with driver.session() as session:
        # Check existing Episode count
        result = session.run("MATCH (ep:Episode) RETURN count(ep) AS cnt")
        existing = result.single()["cnt"]
        print(f"Existing Episode nodes: {existing}")

        # Create Episode nodes in batches
        batch = []
        for i, ep in enumerate(episodes):
            batch.append(ep)
            if len(batch) >= args.batch_size or i == len(episodes) - 1:
                session.execute_write(_create_episodes_batch, batch)
                print(f"  Created episodes {i - len(batch) + 1}-{i + 1}/{len(episodes)}")
                batch = []

        # Create MENTIONS edges in batches
        print()
        mentions_items = list(mentions_map.items())
        total_mentions = sum(len(v) for _, v in mentions_items)
        print(f"Creating {total_mentions} MENTIONS edges for {len(mentions_items)} episodes...")

        edge_batch = []
        edge_count = 0
        for cache_id, entity_ids in mentions_items:
            for eid in entity_ids:
                edge_batch.append({"cache_id": cache_id, "entity_id": eid})
                if len(edge_batch) >= args.batch_size:
                    session.execute_write(_create_mentions_batch, edge_batch)
                    edge_count += len(edge_batch)
                    print(f"  Created MENTIONS edges: {edge_count}/{total_mentions}")
                    edge_batch = []

        if edge_batch:
            session.execute_write(_create_mentions_batch, edge_batch)
            edge_count += len(edge_batch)
            print(f"  Created MENTIONS edges: {edge_count}/{total_mentions}")

    driver.close()

    print()
    print("=" * 60)
    print("Migration complete!")
    print(f"  Episode nodes created: {len(episodes)}")
    print(f"  MENTIONS edges created: {edge_count}")
    print("=" * 60)


def _create_episodes_batch(tx, episodes):
    for ep in episodes:
        tx.run(
            "MERGE (ep:Episode {uuid: $uuid}) "
            "SET ep.content = $content, ep.event_time = $event_time, "
            "ep.source_document = $source_document, ep.created_at = datetime()",
            uuid=ep["uuid"],
            content=ep["content"],
            event_time=ep["event_time"],
            source_document=ep["source_document"],
        )


def _create_mentions_batch(tx, edges):
    for e in edges:
        tx.run(
            "MATCH (ep:Episode {uuid: $cache_id}) "
            "MATCH (ent:Entity {uuid: $entity_id}) "
            "MERGE (ep)-[:MENTIONS]->(ent)",
            cache_id=e["cache_id"],
            entity_id=e["entity_id"],
        )


if __name__ == "__main__":
    main()
