#!/usr/bin/env python3
"""
Migrate data from Neo4j to SQLite storage backend.

Transfers all 8 data types: entities, relations, episodes, relates_to edges,
mentions, entity redirects, content patches, and dream logs.

Usage:
    python scripts/migrate_neo4j_to_sqlite.py \
        --neo4j-uri bolt://localhost:7687 \
        --neo4j-user neo4j \
        --neo4j-password password \
        --sqlite-path ./graph \
        [--graph-id mygraph] \
        [--batch-size 500]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def migrate_graph(neo4j_mgr, sqlite_mgr, graph_id, batch_size=500):
    """Migrate all data for a single graph_id."""
    timings = {}
    counts = {}

    # 1. Entities
    print(f"\n[1/8] Migrating entities for graph_id={graph_id}...")
    t0 = time.time()
    entity_count = 0
    entities = neo4j_mgr.get_all_entities(exclude_embedding=False)
    if entities:
        from core.models import Entity
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            sqlite_mgr.bulk_save_entities_with_embedding(batch)
            entity_count += len(batch)
    timings["entities"] = time.time() - t0
    counts["entities"] = entity_count
    print(f"  Migrated {entity_count} entities in {timings['entities']:.2f}s")

    # 2. Relations
    print(f"\n[2/8] Migrating relations for graph_id={graph_id}...")
    t0 = time.time()
    relation_count = 0
    relations = neo4j_mgr.get_all_relations(exclude_embedding=False)
    if relations:
        for i in range(0, len(relations), batch_size):
            batch = relations[i:i + batch_size]
            sqlite_mgr.bulk_save_relations_with_embedding(batch)
            relation_count += len(batch)
    timings["relations"] = time.time() - t0
    counts["relations"] = relation_count
    print(f"  Migrated {relation_count} relations in {timings['relations']:.2f}s")

    # 3. Episodes
    print(f"\n[3/8] Migrating episodes for graph_id={graph_id}...")
    t0 = time.time()
    episode_count = 0
    try:
        with neo4j_mgr._session() as session:
            result = session.run(
                "MATCH (ep:Episode) WHERE ep.graph_id = $gid RETURN ep",
                gid=graph_id,
            )
            from core.models import Episode
            from datetime import datetime as _dt
            for record in result:
                ep_data = dict(record["ep"])
                raw_et = ep_data.get("event_time")
                if isinstance(raw_et, str):
                    try:
                        raw_et = _dt.fromisoformat(raw_et)
                    except (ValueError, TypeError):
                        raw_et = _dt.now()
                elif not isinstance(raw_et, _dt):
                    raw_et = _dt.now()
                raw_pt = ep_data.get("processed_time")
                if isinstance(raw_pt, str):
                    try:
                        raw_pt = _dt.fromisoformat(raw_pt)
                    except (ValueError, TypeError):
                        raw_pt = None
                elif not isinstance(raw_pt, _dt):
                    raw_pt = None
                ep = Episode(
                    absolute_id=ep_data.get("uuid") or ep_data.get("episode_id", ""),
                    content=ep_data.get("content", ""),
                    event_time=raw_et,
                    source_document=ep_data.get("source_document", ""),
                    processed_time=raw_pt,
                    episode_type=ep_data.get("episode_type"),
                    activity_type=ep_data.get("activity_type"),
                )
                sqlite_mgr.save_episode(ep)
                episode_count += 1
    except Exception as e:
        print(f"  Warning: Episode migration partial: {e}")
    timings["episodes"] = time.time() - t0
    counts["episodes"] = episode_count
    print(f"  Migrated {episode_count} episodes in {timings['episodes']:.2f}s")

    # 4. Relates_to edges
    print(f"\n[4/8] Migrating RELATES_TO edges...")
    t0 = time.time()
    relates_count = 0
    try:
        conn = sqlite_mgr._connect()
        # Fetch from Neo4j via session
        with neo4j_mgr._session() as session:
            result = session.run(
                "MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity) "
                "WHERE e1.graph_id = $gid "
                "RETURN e1.uuid AS e1_uuid, e2.uuid AS e2_uuid, "
                "r.fact AS fact, r.relation_uuid AS rel_uuid",
                gid=graph_id,
            )
            rows = []
            for record in result:
                rows.append((
                    record["e1_uuid"], record["e2_uuid"],
                    record.get("rel_uuid"), record.get("fact", ""),
                    graph_id,
                ))
            if rows:
                for i in range(0, len(rows), batch_size):
                    batch = rows[i:i + batch_size]
                    conn.executemany(
                        "INSERT OR REPLACE INTO relates_to "
                        "(entity1_uuid, entity2_uuid, relation_uuid, fact, graph_id) "
                        "VALUES (?, ?, ?, ?, ?)",
                        batch,
                    )
                    relates_count += len(batch)
                conn.commit()
    except Exception as e:
        print(f"  Warning: Relates_to migration partial: {e}")
    timings["relates_to"] = time.time() - t0
    counts["relates_to"] = relates_count
    print(f"  Migrated {relates_count} RELATES_TO edges in {timings['relates_to']:.2f}s")

    # 5. Mentions edges
    print(f"\n[5/8] Migrating MENTIONS edges...")
    t0 = time.time()
    mentions_count = 0
    try:
        conn = sqlite_mgr._connect()
        with neo4j_mgr._session() as session:
            result = session.run(
                "MATCH (ep:Episode)-[m:MENTIONS]->(target) "
                "WHERE ep.graph_id = $gid "
                "RETURN ep.uuid AS ep_uuid, target.uuid AS target_uuid, "
                "labels(target)[0] AS target_label, m.context AS context, "
                "m.entity_absolute_id AS entity_abs_id",
                gid=graph_id,
            )
            rows = []
            for record in result:
                target_type = "relation" if record.get("target_label") == "Relation" else "entity"
                rows.append((
                    record["ep_uuid"], record["target_uuid"],
                    target_type, record.get("context", ""),
                    record.get("entity_abs_id"), graph_id,
                ))
            if rows:
                for i in range(0, len(rows), batch_size):
                    batch = rows[i:i + batch_size]
                    conn.executemany(
                        "INSERT OR REPLACE INTO mentions "
                        "(episode_uuid, target_uuid, target_type, context, entity_absolute_id, graph_id) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    mentions_count += len(batch)
                conn.commit()
    except Exception as e:
        print(f"  Warning: Mentions migration partial: {e}")
    timings["mentions"] = time.time() - t0
    counts["mentions"] = mentions_count
    print(f"  Migrated {mentions_count} MENTIONS edges in {timings['mentions']:.2f}s")

    # 6. Entity redirects
    print(f"\n[6/8] Migrating entity redirects...")
    t0 = time.time()
    redirect_count = 0
    try:
        conn = sqlite_mgr._connect()
        with neo4j_mgr._session() as session:
            result = session.run(
                "MATCH (s:Entity)-[r:REDIRECTS_TO]->(t:Entity) "
                "WHERE s.graph_id = $gid "
                "RETURN s.uuid AS source_id, t.uuid AS target_id, r.updated_at AS updated_at",
                gid=graph_id,
            )
            rows = []
            for record in result:
                rows.append((record["source_id"], record["target_id"], record.get("updated_at")))
            if rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO entity_redirect (source_id, target_id, updated_at) "
                    "VALUES (?, ?, ?)",
                    rows,
                )
                redirect_count = len(rows)
                conn.commit()
    except Exception as e:
        print(f"  Warning: Redirect migration partial: {e}")
    timings["redirects"] = time.time() - t0
    counts["redirects"] = redirect_count
    print(f"  Migrated {redirect_count} redirects in {timings['redirects']:.2f}s")

    # 7. Content patches
    print(f"\n[7/8] Migrating content patches...")
    t0 = time.time()
    patch_count = 0
    try:
        conn = sqlite_mgr._connect()
        with neo4j_mgr._session() as session:
            result = session.run(
                "MATCH (p:ContentPatch) WHERE p.graph_id = $gid RETURN p",
                gid=graph_id,
            )
            rows = []
            for record in result:
                p = dict(record["p"])
                rows.append((
                    p.get("uuid"), p.get("target_type"),
                    p.get("target_absolute_id"), p.get("target_family_id"),
                    p.get("section_key"), p.get("change_type"),
                    p.get("old_hash"), p.get("new_hash"),
                    p.get("diff_summary"), p.get("source_document"),
                    p.get("event_time"),
                ))
            if rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO content_patch "
                    "(uuid, target_type, target_absolute_id, target_family_id, "
                    "section_key, change_type, old_hash, new_hash, diff_summary, "
                    "source_document, event_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                patch_count = len(rows)
                conn.commit()
    except Exception as e:
        print(f"  Warning: Content patch migration partial: {e}")
    timings["patches"] = time.time() - t0
    counts["patches"] = patch_count
    print(f"  Migrated {patch_count} content patches in {timings['patches']:.2f}s")

    # 8. Dream logs
    print(f"\n[8/8] Migrating dream logs...")
    t0 = time.time()
    dream_count = 0
    try:
        conn = sqlite_mgr._connect()
        with neo4j_mgr._session() as session:
            result = session.run(
                "MATCH (d:DreamLog) WHERE d.graph_id = $gid RETURN d",
                gid=graph_id,
            )
            rows = []
            for record in result:
                d = dict(record["d"])
                rows.append((
                    d.get("cycle_id"), d.get("graph_id", graph_id),
                    str(d.get("start_time", "")) if d.get("start_time") else None,
                    str(d.get("end_time", "")) if d.get("end_time") else None,
                    d.get("status"), d.get("narrative"),
                    d.get("insights"), d.get("connections"),
                    d.get("consolidations"), d.get("strategy"),
                    int(d.get("entities_examined", 0) or 0),
                    int(d.get("relations_created", 0) or 0),
                    d.get("episode_ids"),
                ))
            if rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO dream_log "
                    "(cycle_id, graph_id, start_time, end_time, status, narrative, "
                    "insights, connections, consolidations, strategy, "
                    "entities_examined, relations_created, episode_ids) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                dream_count = len(rows)
                conn.commit()
    except Exception as e:
        print(f"  Warning: Dream log migration partial: {e}")
    timings["dream_logs"] = time.time() - t0
    counts["dream_logs"] = dream_count
    print(f"  Migrated {dream_count} dream logs in {timings['dream_logs']:.2f}s")

    # Rebuild FTS5 index
    print(f"\n[FTS5] Rebuilding full-text search indexes...")
    t0 = time.time()
    conn = sqlite_mgr._connect()
    # Clear stale FTS data then populate from source tables
    conn.execute("DELETE FROM entity_fts")
    conn.execute("DELETE FROM relation_fts")
    conn.execute("""
        INSERT INTO entity_fts(rowid, name, content, graph_id)
        SELECT rowid, name, content, graph_id FROM entity
    """)
    conn.execute("""
        INSERT INTO relation_fts(rowid, content, graph_id)
        SELECT rowid, content, graph_id FROM relation
    """)
    conn.commit()
    timings["fts5_rebuild"] = time.time() - t0
    print(f"  Rebuilt FTS5 indexes in {timings['fts5_rebuild']:.2f}s")

    return counts, timings


def verify_counts(neo4j_mgr, sqlite_mgr, graph_id, expected_counts):
    """Verify migration counts match between source and target."""
    print(f"\n[VERIFY] Checking counts...")
    stats = sqlite_mgr.get_stats()
    issues = []

    actual_entities = stats.get("entities", 0)
    if actual_entities < expected_counts.get("entities", 0):
        issues.append(f"Entities: expected >= {expected_counts['entities']}, got {actual_entities}")

    actual_relations = stats.get("relations", 0)
    if actual_relations < expected_counts.get("relations", 0):
        issues.append(f"Relations: expected >= {expected_counts['relations']}, got {actual_relations}")

    if issues:
        print("  WARNING: Count mismatches detected:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  All counts verified OK")

    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Migrate Neo4j data to SQLite")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", default="tmg2024secure", help="Neo4j password")
    parser.add_argument("--sqlite-path", default="./graph", help="SQLite storage path")
    parser.add_argument("--graph-id", default=None, help="Specific graph_id (default: migrate all)")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for inserts")
    parser.add_argument("--vector-dim", type=int, default=1024, help="Vector dimension")
    args = parser.parse_args()

    from core.storage.neo4j_store import Neo4jStorageManager
    from core.storage.sqlite.manager import SQLiteGraphStorageManager

    # Connect to Neo4j
    print(f"Connecting to Neo4j at {args.neo4j_uri}...")
    neo4j_mgr = Neo4jStorageManager(
        storage_path=args.sqlite_path,
        neo4j_uri=args.neo4j_uri,
        neo4j_auth=(args.neo4j_user, args.neo4j_password),
        vector_dim=args.vector_dim,
        graph_id=args.graph_id or "default",
    )

    # Determine which graph_ids to migrate
    graph_ids = [args.graph_id] if args.graph_id else ["default"]
    if not args.graph_id:
        try:
            with neo4j_mgr._session() as session:
                result = session.run(
                    "MATCH (e:Entity) RETURN DISTINCT e.graph_id AS gid"
                )
                gids = [r["gid"] for r in result if r["gid"]]
                if gids:
                    graph_ids = list(set(gids))
        except Exception:
            pass

    print(f"Will migrate graph_ids: {graph_ids}")

    total_start = time.time()

    for gid in graph_ids:
        print(f"\n{'='*60}")
        print(f"Migrating graph_id: {gid}")
        print(f"{'='*60}")

        sqlite_mgr = SQLiteGraphStorageManager(
            storage_path=f"{args.sqlite_path}/{gid}",
            vector_dim=args.vector_dim,
            graph_id=gid,
        )

        try:
            neo4j_mgr._graph_id = gid  # Temporarily switch graph_id
            counts, timings = migrate_graph(neo4j_mgr, sqlite_mgr, gid, args.batch_size)

            # Verify
            verify_counts(neo4j_mgr, sqlite_mgr, gid, counts)

            # Summary
            print(f"\n--- Summary for {gid} ---")
            for k, v in counts.items():
                print(f"  {k}: {v} ({timings.get(k, 0):.2f}s)")
            total_items = sum(counts.values())
            total_time = sum(timings.values())
            if total_time > 0:
                print(f"  Total: {total_items} items in {total_time:.2f}s ({total_items/total_time:.0f} items/s)")
        finally:
            sqlite_mgr.close()

    print(f"\nTotal migration time: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()
