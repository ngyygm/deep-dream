"""
Backfill missing embeddings for all types using the configured embedding model.

Types:
  1. entity_obs (content)     — missing 12,193
  2. relation_assert (content) — missing 1,227
  3. entity_family (canonical_text) — missing 51,674
  4. episode (memory_text)    — missing 1,041
  5. episode (source_text)    — missing 1,041

Usage: python tmp/backfill_embeddings.py [--batch-size 64] [--types entity_obs,relation_assert,entity_family,episode_mem,episode_src]
"""
import argparse
import hashlib
import sqlite3
import sys
import time
import os
import uuid

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.storage.embedding import EmbeddingClient
from core.storage.sqlite.repositories import embeddings as emb_repo
import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "library", "library.db")

# Config from service_config.json
EMBEDDING_MODEL_PATH = r"D:\Project\Data\Model\Embedding\Qwen3-Embedding-0.6B"
EMBEDDING_DEVICE = "cuda:0"
VECTOR_DIM = 1024


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def encode_batch(client, texts, batch_size=32):
    """Encode texts in sub-batches, return list of (bytes, ndarray) or None."""
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        emb = client.encode(chunk)
        if emb is None:
            results.extend([None] * len(chunk))
            continue
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        for j in range(len(chunk)):
            arr = np.array(emb[j], dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            results.append(arr.tobytes())
    return results


def find_missing(conn, owner_type, text_kind, id_col, text_query, extra_where=""):
    """Find rows that don't have embeddings yet."""
    sql = f"""
        SELECT t.{id_col} as id, ({text_query}) as text
        FROM ({_table_for(owner_type)}) t
        WHERE t.{_status_col(owner_type)} = 'active'
          {extra_where}
          AND t.{id_col} NOT IN (
              SELECT e.owner_id FROM embeddings e
              WHERE e.owner_type = ? AND e.text_kind = ?
          )
        ORDER BY t.{id_col}
    """
    rows = conn.execute(sql, (owner_type, text_kind)).fetchall()
    return [(r["id"], r["text"]) for r in rows if r["text"]]


def _table_for(owner_type):
    tables = {
        "entity_obs": "entity_observations",
        "relation_assert": "relation_assertions",
        "entity_family": "entity_families",
        "episode": "episodes",
    }
    return tables[owner_type]


def _status_col(owner_type):
    cols = {
        "entity_obs": "status",
        "relation_assert": "status",
        "entity_family": "entity_family_id",  # no status, always use id check
        "episode": "status",
    }
    if owner_type == "entity_family":
        return "entity_family_id"  # will be wrapped in a condition that always passes
    return "status"


def backfill_type(conn, client, owner_type, text_kind, items, batch_size, model_name):
    """Backfill embeddings for a list of (id, text) items."""
    total = len(items)
    if total == 0:
        print(f"  ✅ {owner_type}/{text_kind}: no missing embeddings")
        return 0

    print(f"  🔄 {owner_type}/{text_kind}: encoding {total} items in batches of {batch_size}...")

    done = 0
    t_start = time.time()
    for i in range(0, total, batch_size):
        chunk = items[i:i + batch_size]
        ids = [c[0] for c in chunk]
        texts = [c[1] for c in chunk]

        blobs = encode_batch(client, texts, batch_size=min(32, len(texts)))

        for j, (oid, blob) in enumerate(zip(ids, blobs)):
            if blob is None:
                continue
            th = text_hash(texts[j])
            emb_id = f"emb_{owner_type}_{oid}_{text_kind[:8]}"
            emb_repo.insert_embedding(
                conn, emb_id, owner_type, oid, text_kind,
                th, model_name, VECTOR_DIM, blob,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )

        conn.commit()
        done += len(chunk)
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"    {done}/{total} ({done*100//total}%) — {rate:.0f}/s — ETA {eta:.0f}s")

    elapsed = time.time() - t_start
    print(f"  ✅ {owner_type}/{text_kind}: {done} embeddings in {elapsed:.1f}s")
    return done


def main():
    parser = argparse.ArgumentParser(description="Backfill missing embeddings")
    parser.add_argument("--batch-size", type=int, default=64, help="DB write batch size")
    parser.add_argument("--encode-batch", type=int, default=32, help="Encoding batch size")
    parser.add_argument("--types", default="all",
                        help="Comma-separated types: entity_obs,relation_assert,entity_family,episode_mem,episode_src")
    args = parser.parse_args()

    if args.types == "all":
        types = ["entity_obs", "relation_assert", "entity_family", "episode_mem", "episode_src"]
    else:
        types = [t.strip() for t in args.types.split(",")]

    print("=" * 60)
    print("Embedding Backfill")
    print(f"Model: {EMBEDDING_MODEL_PATH}")
    print(f"Device: {EMBEDDING_DEVICE}")
    print(f"Types: {types}")
    print("=" * 60)

    # Initialize embedding client
    print("\n📦 Loading embedding model...")
    client = EmbeddingClient(
        model_path=EMBEDDING_MODEL_PATH,
        device=EMBEDDING_DEVICE,
        cache_max_size=16384,
        cache_ttl=7200,
        max_concurrency=1,
    )
    if not client.is_available():
        print("❌ Embedding model failed to load!")
        sys.exit(1)
    print("✅ Model loaded")

    model_name = EMBEDDING_MODEL_PATH
    conn = get_conn()
    total_written = 0

    try:
        # --- entity_obs ---
        if "entity_obs" in types:
            print("\n📊 [1/5] entity_obs (content)")
            rows = conn.execute("""
                SELECT eo.entity_id as id,
                       CASE WHEN eo.content != '' THEN eo.name || ': ' || eo.content
                            ELSE eo.name END as text
                FROM entity_observations eo
                WHERE eo.status = 'active'
                  AND eo.entity_id NOT IN (
                      SELECT e.owner_id FROM embeddings e WHERE e.owner_type = 'entity_obs'
                  )
                ORDER BY eo.entity_id
            """).fetchall()
            items = [(r["id"], r["text"]) for r in rows if r["text"]]
            print(f"  Missing: {len(items)}")
            total_written += backfill_type(conn, client, "entity_obs", "content", items, args.batch_size, model_name)

        # --- relation_assert ---
        if "relation_assert" in types:
            print("\n📊 [2/5] relation_assert (content)")
            rows = conn.execute("""
                SELECT ra.relation_id as id, ra.content as text
                FROM relation_assertions ra
                WHERE ra.status = 'active'
                  AND ra.relation_id NOT IN (
                      SELECT e.owner_id FROM embeddings e WHERE e.owner_type = 'relation_assert'
                  )
                  AND ra.content != ''
                ORDER BY ra.relation_id
            """).fetchall()
            items = [(r["id"], r["text"]) for r in rows if r["text"]]
            print(f"  Missing: {len(items)}")
            total_written += backfill_type(conn, client, "relation_assert", "content", items, args.batch_size, model_name)

        # --- entity_family ---
        if "entity_family" in types:
            print("\n📊 [3/5] entity_family (canonical_text)")
            rows = conn.execute("""
                SELECT ef.entity_family_id as id,
                       CASE WHEN ef.canonical_content != ''
                            THEN ef.canonical_name || ': ' || ef.canonical_content
                            ELSE ef.canonical_name END as text
                FROM entity_families ef
                WHERE ef.entity_family_id NOT IN (
                    SELECT e.owner_id FROM embeddings e WHERE e.owner_type = 'entity_family'
                )
                ORDER BY ef.entity_family_id
            """).fetchall()
            items = [(r["id"], r["text"]) for r in rows if r["text"]]
            print(f"  Missing: {len(items)}")
            total_written += backfill_type(conn, client, "entity_family", "canonical_text", items, args.batch_size, model_name)

        # --- episode memory_text ---
        if "episode_mem" in types:
            print("\n📊 [4/5] episode (memory_text)")
            rows = conn.execute("""
                SELECT ep.episode_id as id, ep.memory_text as text
                FROM episodes ep
                WHERE ep.status = 'active'
                  AND ep.episode_id NOT IN (
                      SELECT e.owner_id FROM embeddings e
                      WHERE e.owner_type = 'episode' AND e.text_kind = 'memory_text'
                  )
                  AND ep.memory_text != ''
                ORDER BY ep.episode_id
            """).fetchall()
            items = [(r["id"], r["text"]) for r in rows if r["text"]]
            print(f"  Missing: {len(items)}")
            total_written += backfill_type(conn, client, "episode", "memory_text", items, args.batch_size, model_name)

        # --- episode source_text ---
        if "episode_src" in types:
            print("\n📊 [5/5] episode (source_text)")
            rows = conn.execute("""
                SELECT ep.episode_id as id, ep.source_text as text
                FROM episodes ep
                WHERE ep.status = 'active'
                  AND ep.episode_id NOT IN (
                      SELECT e.owner_id FROM embeddings e
                      WHERE e.owner_type = 'episode' AND e.text_kind = 'source_text'
                  )
                  AND ep.source_text != ''
                ORDER BY ep.episode_id
            """).fetchall()
            items = [(r["id"], r["text"]) for r in rows if r["text"]]
            print(f"  Missing: {len(items)}")
            total_written += backfill_type(conn, client, "episode", "source_text", items, args.batch_size, model_name)

    finally:
        conn.close()

    print("\n" + "=" * 60)
    print(f"✅ Done! Total embeddings written: {total_written}")
    print("=" * 60)


if __name__ == "__main__":
    main()
