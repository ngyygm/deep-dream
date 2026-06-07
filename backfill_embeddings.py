"""Backfill embeddings — OPTIMIZED version.

Pre-loads all mappings into dicts, then batch-encodes and bulk-inserts.
Should sustain ~300 texts/sec on GPU.
"""
import json
import time
import sqlite3
import hashlib
import numpy as np
from datetime import datetime, timezone

BATCH_SIZE = 128
DB_PATH = "library/library.db"


def _now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def main():
    with open("service_config.json") as f:
        config = json.load(f)
    emb_cfg = config.get("embedding", {})
    model_name = emb_cfg.get("model", "unknown")

    from core.storage.embedding import EmbeddingClient
    client = EmbeddingClient(model_name=model_name, device=emb_cfg.get("device", "cpu"))
    assert client.is_available(), "Embedding client not available"
    client.encode("warm up")
    print(f"Client ready. Model: {model_name}", flush=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")

    t_start = time.time()

    # ── Phase 1: Entity embeddings ──────────────────────────────────────────
    print("\n=== Phase 1: Entity embeddings ===", flush=True)

    # Pre-load ALL (family_id -> active_obs_id) mappings in ONE query
    obs_map = {}
    for row in conn.execute(
        "SELECT entity_family_id, entity_id FROM entity_observations "
        "WHERE status = 'active' "
        "GROUP BY entity_family_id HAVING rowid = MAX(rowid)"
    ).fetchall():
        obs_map[row[0]] = row[1]

    # Pre-load existing embeddings (set of obs_ids that already have one)
    existing_emb_obs = set()
    for row in conn.execute(
        "SELECT owner_id FROM embeddings WHERE owner_type = 'entity_obs'"
    ).fetchall():
        existing_emb_obs.add(row[0])

    # Load entity families
    rows = conn.execute(
        "SELECT entity_family_id, canonical_name, canonical_content FROM entity_families"
    ).fetchall()

    # Filter to families that have an obs_id and no embedding yet
    todo = []
    for r in rows:
        fid, name, content = r[0], r[1], r[2] or ""
        obs_id = obs_map.get(fid)
        if obs_id and obs_id not in existing_emb_obs:
            text = f"{name}: {content}" if content else name
            todo.append((fid, obs_id, text[:512]))

    print(f"Entities to backfill: {len(todo)}", flush=True)
    e_stored = 0
    e_skip = 0

    for offset in range(0, len(todo), BATCH_SIZE):
        batch = todo[offset:offset + BATCH_SIZE]
        texts = [item[2] for item in batch]

        try:
            vecs = client.encode(texts)
        except Exception as exc:
            print(f"\nEncode error at {offset}: {exc}", flush=True)
            e_skip += len(batch)
            continue

        rows_data = []
        for i, (fid, obs_id, text) in enumerate(batch):
            vec = vecs[i]
            blob = vec.astype(np.float32).tobytes()
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            emb_id = f"emb_{obs_id}"
            dim = len(blob) // 4
            rows_data.append((emb_id, obs_id, text_hash, dim, blob))

        conn.executemany(
            "INSERT OR REPLACE INTO embeddings "
            "(embedding_id, owner_type, owner_id, text_kind, text_hash, "
            " embedding_model, dimensions, vector, created_at) "
            "VALUES (?, 'entity_obs', ?, 'content', ?, ?, ?, ?, ?)",
            [(r[0], r[1], r[2], model_name, r[3], r[4], _now_str()) for r in rows_data],
        )
        conn.commit()
        e_stored += len(batch)

        done = min(offset + BATCH_SIZE, len(todo))
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(todo) - done) / rate / 60 if rate > 0 else 0
        print(f"\r  Entity: {done}/{len(todo)} ({done*100//len(todo)}%) "
              f"rate={rate:.0f}/s ETA={eta:.1f}min", end="", flush=True)

    print(f"\n  Entity done: {e_stored} stored, {e_skip} skipped", flush=True)

    # ── Phase 2: Relation embeddings ───────────────────────────────────────
    print("\n=== Phase 2: Relation embeddings ===", flush=True)

    # Pre-load (relation_family_id -> active_assertion_id) mappings
    assert_map = {}
    for row in conn.execute(
        "SELECT relation_family_id, relation_id FROM relation_assertions "
        "WHERE status = 'active' "
        "GROUP BY relation_family_id HAVING rowid = MAX(rowid)"
    ).fetchall():
        assert_map[row[0]] = row[1]

    existing_emb_rel = set()
    for row in conn.execute(
        "SELECT owner_id FROM embeddings WHERE owner_type = 'relation_assert'"
    ).fetchall():
        existing_emb_rel.add(row[0])

    rel_rows = conn.execute(
        "SELECT relation_family_id, canonical_content, "
        "       subject_entity_family_id, object_entity_family_id "
        "FROM relation_families"
    ).fetchall()

    rel_todo = []
    for r in rel_rows:
        fid, content, sub_fid, obj_fid = r[0], r[1] or "", r[2], r[3]
        assert_id = assert_map.get(fid)
        if assert_id and assert_id not in existing_emb_rel:
            text = content or f"relation_{sub_fid[:8]}_{obj_fid[:8]}"
            rel_todo.append((fid, assert_id, text[:512]))

    print(f"Relations to backfill: {len(rel_todo)}", flush=True)
    r_stored = 0
    r_skip = 0

    for offset in range(0, len(rel_todo), BATCH_SIZE):
        batch = rel_todo[offset:offset + BATCH_SIZE]
        texts = [item[2] for item in batch]

        try:
            vecs = client.encode(texts)
        except Exception as exc:
            print(f"\nEncode error at {offset}: {exc}", flush=True)
            r_skip += len(batch)
            continue

        rows_data = []
        for i, (fid, assert_id, text) in enumerate(batch):
            vec = vecs[i]
            blob = vec.astype(np.float32).tobytes()
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            emb_id = f"emb_{assert_id}"
            dim = len(blob) // 4
            rows_data.append((emb_id, assert_id, text_hash, dim, blob))

        conn.executemany(
            "INSERT OR REPLACE INTO embeddings "
            "(embedding_id, owner_type, owner_id, text_kind, text_hash, "
            " embedding_model, dimensions, vector, created_at) "
            "VALUES (?, 'relation_assert', ?, 'content', ?, ?, ?, ?, ?)",
            [(r[0], r[1], r[2], model_name, r[3], r[4], _now_str()) for r in rows_data],
        )
        conn.commit()
        r_stored += len(batch)

        done = min(offset + BATCH_SIZE, len(rel_todo))
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(rel_todo) - done) / rate / 60 if rate > 0 else 0
        print(f"\r  Relation: {done}/{len(rel_todo)} ({done*100//len(rel_todo)}%) "
              f"rate={rate:.0f}/s ETA={eta:.1f}min", end="", flush=True)

    print(f"\n  Relation done: {r_stored} stored, {r_skip} skipped", flush=True)

    # ── Summary ─────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    total_emb = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    print(f"\n{'='*60}")
    print(f"BACKFILL COMPLETE in {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Entity embeddings:   {e_stored}")
    print(f"  Relation embeddings: {r_stored}")
    print(f"  Total in DB:         {total_emb}")
    print(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()
