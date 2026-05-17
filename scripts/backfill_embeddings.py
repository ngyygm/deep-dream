#!/usr/bin/env python3
"""
Backfill missing embeddings for entities and relations in SQLite storage.

Uses the same EmbeddingClient as the main application to compute and save
embeddings for records that have none.

Usage:
    python scripts/backfill_embeddings.py \
        --sqlite-path ./graph_migrated3 \
        --graph-id default \
        --model Qwen/Qwen3-Embedding-0.6B \
        --batch-size 64

    # Backfill all graphs:
    python scripts/backfill_embeddings.py \
        --sqlite-path ./graph_migrated3 \
        --model Qwen/Qwen3-Embedding-0.6B
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

_EMB_CONTENT_MAX = 500


def _get_missing(conn, table, graph_id):
    """Get rows missing embeddings from a table."""
    if table == "entity":
        rows = conn.execute(
            "SELECT uuid, name, content FROM entity "
            "WHERE embedding IS NULL AND graph_id = ? "
            "ORDER BY processed_time DESC",
            (graph_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT uuid, content FROM relation "
            "WHERE embedding IS NULL AND graph_id = ? "
            "ORDER BY processed_time DESC",
            (graph_id,),
        ).fetchall()
    return rows


def _build_texts(table, rows):
    """Build embedding input texts from rows."""
    texts = []
    if table == "entity":
        for uuid, name, content in rows:
            text = f"# {name}\n{(content or '')[:_EMB_CONTENT_MAX]}"
            texts.append(text)
    else:
        for uuid, content in rows:
            text = (content or "")[:_EMB_CONTENT_MAX]
            texts.append(text)
    return texts


def backfill_graph(sqlite_path, graph_id, embedding_client, batch_size):
    """Backfill embeddings for a single graph."""
    db_path = Path(sqlite_path) / graph_id / "graph.db"
    if not db_path.exists():
        print(f"  Skipping {graph_id}: database not found at {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    for table in ["entity", "relation"]:
        rows = _get_missing(conn, table, graph_id)
        if not rows:
            print(f"  {table}: all embeddings present ✓")
            continue

        print(f"  {table}: {len(rows)} missing embeddings, computing...")

        total_done = 0
        t0 = time.time()
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            texts = _build_texts(table, batch)

            try:
                embeddings = embedding_client.encode(texts, batch_size=len(texts))
            except Exception as e:
                print(f"    Batch {i}: encode error: {e}")
                continue

            if embeddings is None:
                print(f"    Batch {i}: encode returned None, skipping")
                continue

            update_rows = []
            for j, row in enumerate(batch):
                uuid = row[0]
                try:
                    emb_array = np.array(embeddings[j], dtype=np.float32)
                    norm = np.linalg.norm(emb_array)
                    if norm > 0:
                        emb_array = emb_array / norm
                    update_rows.append((emb_array.tobytes(), uuid))
                except Exception:
                    continue

            if update_rows:
                conn.executemany(
                    f"UPDATE {table} SET embedding = ? WHERE uuid = ?",
                    update_rows,
                )
                conn.commit()

            total_done += len(batch)
            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            pct = total_done / len(rows) * 100
            print(f"    {total_done}/{len(rows)} ({pct:.0f}%) - {rate:.0f}/s")

        elapsed = time.time() - t0
        print(f"  {table}: {total_done} embeddings computed in {elapsed:.1f}s")

    # Verify
    for table in ["entity", "relation"]:
        total = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE graph_id = ?", (graph_id,)).fetchone()[0]
        with_emb = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL AND graph_id = ?", (graph_id,)).fetchone()[0]
        pct = with_emb / total * 100 if total > 0 else 100
        print(f"  {table}: {with_emb}/{total} ({pct:.0f}%) now have embeddings")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill missing embeddings in SQLite")
    parser.add_argument("--sqlite-path", default="./graph_migrated3", help="SQLite storage root path")
    parser.add_argument("--graph-id", default=None, help="Specific graph_id (default: all)")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="Embedding model name or path")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for encoding")
    args = parser.parse_args()

    from core.storage.embedding import EmbeddingClient

    print(f"Loading embedding model: {args.model}...")
    client = EmbeddingClient(model_name=args.model, device=args.device)
    if client.model is None:
        print("ERROR: Failed to load embedding model")
        sys.exit(1)
    print("Model loaded ✓")

    # Determine graph_ids
    if args.graph_id:
        graph_ids = [args.graph_id]
    else:
        base = Path(args.sqlite_path)
        graph_ids = [d.name for d in base.iterdir() if d.is_dir() and (d / "graph.db").exists()]
        graph_ids.sort()

    print(f"Graph IDs: {graph_ids}")

    total_start = time.time()
    for gid in graph_ids:
        print(f"\n{'='*50}")
        print(f"Backfilling: {gid}")
        print(f"{'='*50}")
        backfill_graph(args.sqlite_path, gid, client, args.batch_size)

    print(f"\nTotal time: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
