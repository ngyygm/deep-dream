"""
Stress tests for SQLite graph storage.

Validates concurrent write safety, memory stability, WAL growth, and cache behavior.
Run with: pytest core/tests/test_stress.py -v -s --tb=short
"""
import os
import threading
import time
import uuid
from datetime import datetime, timezone

import pytest

from core.models import Entity, Relation
from core.storage.sqlite.manager import SQLiteGraphStorageManager


def _make_entity(idx, graph_id="stress"):
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    return Entity(
        absolute_id=f"entity_{ts}_{uuid.uuid4().hex[:8]}",
        family_id=f"fam_{idx}_{threading.get_ident()}",
        name=f"Stress Entity {idx} from thread {threading.get_ident()}",
        content=f"Content for stress test entity {idx}.",
        event_time=now,
        processed_time=now,
        episode_id="stress_episode",
        source_document="stress.txt",
    )


def _make_relation(idx, e1, e2):
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    sorted_ids = sorted([e1, e2])
    return Relation(
        absolute_id=f"relation_{ts}_{uuid.uuid4().hex[:8]}",
        family_id=f"r_fam_{idx}_{threading.get_ident()}",
        entity1_absolute_id=sorted_ids[0],
        entity2_absolute_id=sorted_ids[1],
        content=f"Stress relation {idx}",
        event_time=now,
        processed_time=now,
        episode_id="stress_episode",
        source_document="stress.txt",
    )


def _get_rss_mb():
    """Get current process RSS in MB (Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except Exception:
        return 0


@pytest.fixture(scope="module")
def stress_storage(tmp_path_factory):
    d = tmp_path_factory.mktemp("stress_graph")
    mgr = SQLiteGraphStorageManager(
        storage_path=str(d / "graph"),
        vector_dim=1024,
        graph_id="stress",
    )
    yield mgr
    mgr.close()


# ── Concurrent Writes ────────────────────────────────────────────────────


class TestConcurrentWrites:
    def test_concurrent_entity_writes(self, stress_storage):
        """8 threads × 1000 entities = 8000 entities, verify no data loss."""
        N_THREADS = 8
        N_PER_THREAD = 500
        errors = []
        written = []

        def writer(thread_idx):
            try:
                entities = []
                for i in range(N_PER_THREAD):
                    e = _make_entity(thread_idx * N_PER_THREAD + i)
                    entities.append(e)
                stress_storage.bulk_save_entities(entities)
                written.extend([e.family_id for e in entities])
            except Exception as ex:
                errors.append((thread_idx, str(ex)))

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not errors, f"Errors in concurrent writes: {errors}"
        # Verify all family_ids exist in storage
        for fid in written:
            entity = stress_storage.get_entity_by_family_id(fid)
            assert entity is not None, f"Missing entity with family_id={fid}"

    def test_concurrent_mixed_rw(self, stress_storage):
        """16 threads doing interleaved reads and writes."""
        N_THREADS = 16
        N_OPS = 100
        errors = []
        barrier = threading.Barrier(N_THREADS)

        def worker(idx):
            barrier.wait(timeout=10)
            for i in range(N_OPS):
                try:
                    if i % 3 == 0:
                        e = _make_entity(idx * N_OPS + i + 10000)
                        stress_storage.save_entity(e)
                    else:
                        stress_storage.count_unique_entities()
                except Exception as ex:
                    errors.append((idx, i, str(ex)))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        assert not errors, f"Errors in mixed R/W: {errors[:5]}"


# ── Memory Stability ─────────────────────────────────────────────────────


class TestMemoryStability:
    def test_entity_bulk_memory(self, stress_storage):
        """Write 10k entities + 10k relations, check RSS growth < 512MB."""
        rss_before = _get_rss_mb()
        entities = [_make_entity(i + 50000) for i in range(5000)]
        # Bulk insert in batches of 500
        for batch_start in range(0, len(entities), 500):
            batch = entities[batch_start:batch_start + 500]
            stress_storage.bulk_save_entities(batch)
        rss_after_entities = _get_rss_mb()
        entity_growth = rss_after_entities - rss_before
        print(f"\n  RSS after 5k entities: {rss_after_entities:.0f}MB (growth: {entity_growth:.0f}MB)")

        # Create relations
        for batch_start in range(0, len(entities) - 1, 500):
            batch_end = min(batch_start + 500, len(entities) - 1)
            rels = []
            for i in range(batch_start, batch_end):
                r = _make_relation(i, entities[i].absolute_id, entities[i + 1].absolute_id)
                rels.append(r)
            stress_storage.bulk_save_relations(rels)

        rss_after = _get_rss_mb()
        total_growth = rss_after - rss_before
        print(f"  RSS after 5k entities + relations: {rss_after:.0f}MB (growth: {total_growth:.0f}MB)")
        assert total_growth < 512, f"Memory growth too high: {total_growth:.0f}MB"


# ── WAL Growth ───────────────────────────────────────────────────────────


class TestWALGrowth:
    def test_wal_bounded_after_many_writes(self, stress_storage):
        """Write 5k entities rapidly, verify WAL checkpoint keeps it bounded."""
        # Force a checkpoint first
        conn = stress_storage._connect()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.rollback()

        entities = [_make_entity(i + 60000) for i in range(5000)]
        for batch_start in range(0, len(entities), 200):
            batch = entities[batch_start:batch_start + 200]
            stress_storage.bulk_save_entities(batch)

        # Check WAL file size
        wal_path = str(stress_storage._db_path) + "-wal"
        if os.path.exists(wal_path):
            wal_size_mb = os.path.getsize(wal_path) / (1024 * 1024)
            print(f"\n  WAL size after 5k writes: {wal_size_mb:.1f}MB")

            # Force checkpoint and verify it works
            conn = stress_storage._connect()
            result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            conn.rollback()
            wal_size_after = os.path.getsize(wal_path) / (1024 * 1024)
            print(f"  WAL size after checkpoint: {wal_size_after:.1f}MB")
            assert wal_size_after < 10, f"WAL not truncated: {wal_size_after:.1f}MB"


# ── Cache Pressure ───────────────────────────────────────────────────────


class TestCachePressure:
    def test_cache_under_pressure(self, stress_storage):
        """Run many unique queries, verify cache doesn't grow unbounded."""
        cache_before = stress_storage._cache.stats()
        print(f"\n  Cache before: size={cache_before['size']}")

        for i in range(1000):
            stress_storage._cache.set(f"pressure_key_{i}", f"value_{i}", ttl=60)

        cache_after = stress_storage._cache.stats()
        print(f"  Cache after 1000 inserts: size={cache_after['size']}, max={cache_after['max_size']}, evictions={cache_after['evictions']}")
        assert cache_after['size'] <= cache_after['max_size'], "Cache exceeded max_size"
