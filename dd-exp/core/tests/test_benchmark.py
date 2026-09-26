"""
Benchmark suite for SQLite graph storage operations.

Reports mean/p95/p99 latencies for critical operations.
Run with: pytest core/tests/test_benchmark.py -v -s --tb=short
"""
import time
import uuid
import statistics
from datetime import datetime, timezone

import pytest

from core.models import Entity, Relation, Episode
from core.storage.sqlite.manager import SQLiteGraphStorageManager


def _make_entity(idx, family_id=None):
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    fid = family_id or f"fam_{idx}"
    return Entity(
        absolute_id=f"entity_{ts}_{uuid.uuid4().hex[:8]}",
        family_id=fid,
        name=f"Benchmark Entity {idx}",
        content=f"Content for benchmark entity {idx}. " * 10,
        event_time=now,
        processed_time=now,
        episode_id="bench_episode",
        source_document="benchmark.txt",
    )


def _make_relation(idx, e1_abs, e2_abs, family_id=None):
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    sorted_ids = sorted([e1_abs, e2_abs])
    fid = family_id or f"r_fam_{idx}"
    return Relation(
        absolute_id=f"relation_{ts}_{uuid.uuid4().hex[:8]}",
        family_id=fid,
        entity1_absolute_id=sorted_ids[0],
        entity2_absolute_id=sorted_ids[1],
        content=f"Benchmark relation {idx}: connects entities",
        event_time=now,
        processed_time=now,
        episode_id="bench_episode",
        source_document="benchmark.txt",
    )


def _measure(func, iterations=50):
    """Run func N times, return (mean, p95, p99) in ms."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "mean": statistics.mean(times),
        "p95": times[int(len(times) * 0.95)],
        "p99": times[int(len(times) * 0.99)],
        "min": times[0],
        "max": times[-1],
    }


@pytest.fixture(scope="module")
def bench_storage(tmp_path_factory):
    """Create a SQLite storage manager for benchmarking."""
    d = tmp_path_factory.mktemp("bench_graph")
    mgr = SQLiteGraphStorageManager(
        storage_path=str(d / "graph"),
        vector_dim=1024,
        graph_id="bench",
    )
    yield mgr
    mgr.close()


@pytest.fixture(scope="module")
def populated_storage(bench_storage):
    """Populate storage with 1000 entities + relations for search benchmarks."""
    mgr = bench_storage
    entities = [_make_entity(i) for i in range(1000)]
    mgr.bulk_save_entities(entities)
    # Create relations between consecutive entities
    relations = []
    for i in range(999):
        rel = _make_relation(i, entities[i].absolute_id, entities[i + 1].absolute_id)
        relations.append(rel)
    mgr.bulk_save_relations(relations)
    return mgr


# ── Write Benchmarks ─────────────────────────────────────────────────────


class TestWriteBenchmarks:
    def test_save_entity_single(self, bench_storage):
        """Benchmark single entity save."""
        idx = [0]
        def save_one():
            e = _make_entity(idx[0])
            idx[0] += 1
            bench_storage.save_entity(e)

        stats = _measure(save_one, 100)
        print(f"\n  save_entity (single): {stats}")
        assert stats["mean"] < 50, f"save_entity too slow: {stats['mean']:.1f}ms"

    def test_bulk_save_entities_100(self, bench_storage):
        """Benchmark bulk save of 100 entities."""
        def bulk_100():
            entities = [_make_entity(i) for i in range(100)]
            bench_storage.bulk_save_entities(entities)

        stats = _measure(bulk_100, 20)
        print(f"\n  bulk_save_entities (100): {stats}")
        assert stats["mean"] < 500, f"bulk_save_entities too slow: {stats['mean']:.1f}ms"

    def test_save_relation_single(self, populated_storage):
        """Benchmark single relation save."""
        entities = list(populated_storage.get_all_entities(limit=2))
        if len(entities) < 2:
            pytest.skip("Not enough entities")
        idx = [0]
        def save_one():
            r = _make_relation(idx[0], entities[0].absolute_id, entities[1].absolute_id)
            idx[0] += 1
            populated_storage.save_relation(r)

        stats = _measure(save_one, 100)
        print(f"\n  save_relation (single): {stats}")
        assert stats["mean"] < 50, f"save_relation too slow: {stats['mean']:.1f}ms"

    def test_bulk_save_relations_100(self, populated_storage):
        """Benchmark bulk save of 100 relations."""
        entities = list(populated_storage.get_all_entities(limit=102))
        if len(entities) < 102:
            pytest.skip("Not enough entities")

        def bulk_100():
            rels = []
            for i in range(100):
                r = _make_relation(i, entities[i].absolute_id, entities[i + 1].absolute_id)
                rels.append(r)
            populated_storage.bulk_save_relations(rels)

        stats = _measure(bulk_100, 20)
        print(f"\n  bulk_save_relations (100): {stats}")
        assert stats["mean"] < 500, f"bulk_save_relations too slow: {stats['mean']:.1f}ms"


# ── Read Benchmarks ──────────────────────────────────────────────────────


class TestReadBenchmarks:
    def test_get_entity_by_family_id(self, populated_storage):
        """Benchmark entity lookup by family_id."""
        entities = list(populated_storage.get_all_entities(limit=1))
        if not entities:
            pytest.skip("No entities")
        fid = entities[0].family_id

        stats = _measure(lambda: populated_storage.get_entity_by_family_id(fid), 200)
        print(f"\n  get_entity_by_family_id: {stats}")
        assert stats["mean"] < 10, f"get_entity too slow: {stats['mean']:.1f}ms"

    def test_get_all_entities(self, populated_storage):
        """Benchmark listing entities."""
        stats = _measure(lambda: populated_storage.get_all_entities(limit=100), 50)
        print(f"\n  get_all_entities (100): {stats}")
        assert stats["mean"] < 100, f"get_all_entities too slow: {stats['mean']:.1f}ms"

    def test_search_entities_bm25(self, populated_storage):
        """Benchmark BM25 entity search."""
        stats = _measure(
            lambda: populated_storage.search_entities_by_bm25("benchmark entity", limit=20),
            100,
        )
        print(f"\n  search_entities_bm25: {stats}")
        assert stats["mean"] < 50, f"BM25 search too slow: {stats['mean']:.1f}ms"

    def test_search_relations_bm25(self, populated_storage):
        """Benchmark BM25 relation search."""
        stats = _measure(
            lambda: populated_storage.search_relations_by_bm25("benchmark relation", limit=20),
            100,
        )
        print(f"\n  search_relations_bm25: {stats}")
        assert stats["mean"] < 50, f"Relation BM25 search too slow: {stats['mean']:.1f}ms"

    def test_count_entities(self, populated_storage):
        """Benchmark entity counting."""
        stats = _measure(lambda: populated_storage.count_unique_entities(), 200)
        print(f"\n  count_unique_entities: {stats}")
        assert stats["mean"] < 20, f"count_unique_entities too slow: {stats['mean']:.1f}ms"

    def test_get_stats(self, populated_storage):
        """Benchmark get_stats (combined counts)."""
        stats = _measure(lambda: populated_storage.get_stats(), 200)
        print(f"\n  get_stats: {stats}")
        assert stats["mean"] < 50, f"get_stats too slow: {stats['mean']:.1f}ms"


# ── Episode Benchmarks ───────────────────────────────────────────────────


class TestEpisodeBenchmarks:
    def test_save_and_get_episode(self, bench_storage):
        """Benchmark episode save + retrieve cycle."""
        idx = [0]

        def cycle():
            now = datetime.now(timezone.utc)
            ep = Episode(
                absolute_id=f"episode_{uuid.uuid4().hex[:12]}",
                content=f"Benchmark episode content {idx[0]}",
                event_time=now,
                source_document="benchmark.txt",
                processed_time=now,
                episode_type="fact",
            )
            bench_storage.save_episode(ep)
            idx[0] += 1

        stats = _measure(cycle, 100)
        print(f"\n  save_episode: {stats}")
        assert stats["mean"] < 30, f"save_episode too slow: {stats['mean']:.1f}ms"
