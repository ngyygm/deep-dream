#!/usr/bin/env python3
"""
DeepDream 性能基准测试：对比优化前后的真实耗时。

用法：
    # 1. 先 stash 优化代码，跑基准
    git stash push -m "perf-opt" -- processor/storage/neo4j_store.py processor/storage/vector_store.py processor/storage/embedding.py server/api.py processor/perf.py

    # 2. 跑基准测试
    python bench_perf.py --label "BEFORE"

    # 3. 恢复优化代码
    git stash pop

    # 4. 跑优化后测试
    python bench_perf.py --label "AFTER"

    # 5. 查看对比
    python bench_perf.py --compare bench_before.json bench_after.json
"""
import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from processor.storage.neo4j_store import Neo4jStorageManager
from processor.storage.cache import QueryCache
from processor.storage.vector_store import VectorStore
from processor.storage.embedding import EmbeddingClient

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def create_storage():
    """创建 storage 实例（连接真实 Neo4j + sqlite-vec）。"""
    config = json.load(open("service_config.json"))
    neo4j_cfg = config["storage"]["neo4j"]
    vector_dim = config["storage"].get("vector_dim", 1024)
    emb_cfg = config.get("embedding", {})
    emb_client = EmbeddingClient(
        model_path=emb_cfg.get("model"),
        device=emb_cfg.get("device", "cpu"),
    )

    # 检查 Neo4jStorageManager 的构造函数签名来适配
    import inspect
    sig = inspect.signature(Neo4jStorageManager.__init__)
    params = list(sig.parameters.keys())

    if "neo4j_auth" in params:
        # 优化后版本: (storage_path, neo4j_uri, neo4j_auth, ...)
        storage = Neo4jStorageManager(
            storage_path="graph",
            neo4j_uri=neo4j_cfg["uri"],
            neo4j_auth=(neo4j_cfg["user"], neo4j_cfg["password"]),
            embedding_client=emb_client,
            vector_dim=vector_dim,
        )
    else:
        # 原始版本: 可能不同参数
        storage = Neo4jStorageManager(
            storage_path="graph",
            neo4j_uri=neo4j_cfg["uri"],
            neo4j_auth=(neo4j_cfg["user"], neo4j_cfg["password"]),
            embedding_client=emb_client,
            vector_dim=vector_dim,
        )
    return storage


def warmup(storage):
    """预热：跑一遍让连接池和缓存就绪。"""
    print("  预热中...")
    storage.get_entity_by_entity_id("test_warmup")
    storage.resolve_entity_id("test_warmup")


def bench_single(func, label, rounds=5, warmup_rounds=1, clear_cache_fn=None):
    """对单个函数计时，返回统计数据。

    Args:
        func: 被测函数
        label: 标签名
        rounds: 测试轮数
        warmup_rounds: 预热轮数
        clear_cache_fn: 每轮调用前执行的缓存清除函数
    """
    # warmup
    for _ in range(warmup_rounds):
        try:
            if clear_cache_fn:
                clear_cache_fn()
            func()
        except Exception:
            pass

    times = []
    errors = 0
    for i in range(rounds):
        if clear_cache_fn:
            clear_cache_fn()
        t0 = time.perf_counter()
        try:
            result = func()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        except Exception as e:
            errors += 1
            print(f"    第 {i+1} 轮出错: {e}")

    if not times:
        return {"label": label, "error": "全部失败", "times_ms": []}

    return {
        "label": label,
        "rounds": rounds,
        "errors": errors,
        "times_ms": [round(t, 1) for t in times],
        "mean_ms": round(statistics.mean(times), 1),
        "median_ms": round(statistics.median(times), 1),
        "min_ms": round(min(times), 1),
        "max_ms": round(max(times), 1),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0], 1),
    }


def run_benchmarks(storage, label: str, rounds: int = 5):
    """运行所有 benchmark。同时跑冷查询（清缓存）和热查询（缓存命中）。"""
    results = {}

    # 缓存清除函数
    def clear_cache():
        if hasattr(storage, '_cache'):
            storage._cache.invalidate()

    # 先获取一些真实 entity_id 用于测试
    print("  获取测试数据...")
    sample_entities = storage.get_all_entities(limit=10, exclude_embedding=True)
    if not sample_entities:
        print("  错误: 数据库中无实体，无法跑 benchmark")
        return results

    sample_eid = sample_entities[0].entity_id
    sample_abs_id = sample_entities[0].absolute_id

    # 找一对有关系的实体
    sample_rel = None
    for e in sample_entities:
        all_rels = storage.get_entity_relations_by_family_id(e.entity_id, limit=1)
        if all_rels:
            sample_rel = all_rels[0]
            break

    if sample_rel:
        partner_eid = None
        for e in sample_entities:
            if e.absolute_id in (sample_rel.entity1_absolute_id, sample_rel.entity2_absolute_id):
                partner_eid = e.entity_id
                break
    else:
        partner_eid = sample_eid

    print(f"  测试实体: entity_id={sample_eid}")
    print(f"  关系对: {sample_eid} <-> {partner_eid}")
    print()

    # ===== 冷查询（每轮清缓存）=====
    print("  ── 冷查询（每轮清缓存）──")
    clear_cache()

    # --- 1. resolve_entity_id ---
    print(f"  [1/8] resolve_entity_id ({rounds} 轮)...", end="", flush=True)
    results["resolve_entity_id"] = bench_single(
        lambda: storage.resolve_entity_id(sample_eid),
        "resolve_entity_id", rounds=rounds, clear_cache_fn=clear_cache
    )
    print(f"  {results['resolve_entity_id']['median_ms']:.1f}ms")

    # --- 2. get_entity_by_entity_id ---
    print(f"  [2/8] get_entity_by_entity_id ({rounds} 轮)...", end="", flush=True)
    results["get_entity_by_entity_id"] = bench_single(
        lambda: storage.get_entity_by_entity_id(sample_eid),
        "get_entity_by_entity_id", rounds=rounds, clear_cache_fn=clear_cache
    )
    print(f"  {results['get_entity_by_entity_id']['median_ms']:.1f}ms")

    # --- 3. get_entity_by_absolute_id ---
    print(f"  [3/8] get_entity_by_absolute_id ({rounds} 轮)...", end="", flush=True)
    results["get_entity_by_absolute_id"] = bench_single(
        lambda: storage.get_entity_by_absolute_id(sample_abs_id),
        "get_entity_by_absolute_id", rounds=rounds, clear_cache_fn=clear_cache
    )
    print(f"  {results['get_entity_by_absolute_id']['median_ms']:.1f}ms")

    # --- 4. get_relations_by_entities ---
    print(f"  [4/8] get_relations_by_entities ({rounds} 轮)...", end="", flush=True)
    results["get_relations_by_entities"] = bench_single(
        lambda: storage.get_relations_by_entities(sample_eid, partner_eid),
        "get_relations_by_entities", rounds=rounds, clear_cache_fn=clear_cache
    )
    print(f"  {results['get_relations_by_entities']['median_ms']:.1f}ms")

    # --- 5. get_entity_relations_by_family_id ---
    print(f"  [5/8] get_entity_relations_by_family_id ({rounds} 轮)...", end="", flush=True)
    results["get_entity_relations_by_family_id"] = bench_single(
        lambda: storage.get_entity_relations_by_family_id(sample_eid, limit=20),
        "get_entity_relations_by_family_id", rounds=rounds, clear_cache_fn=clear_cache
    )
    print(f"  {results['get_entity_relations_by_family_id']['median_ms']:.1f}ms")

    # --- 6. search_entities_by_similarity (只跑 3 轮，因为全量加载) ---
    sim_rounds = 3
    print(f"  [6/8] search_entities_by_similarity ({sim_rounds} 轮)...", end="", flush=True)
    results["search_entities_by_similarity"] = bench_single(
        lambda: storage.search_entities_by_similarity(
            query_name=sample_entities[0].name,
            threshold=0.3, max_results=10
        ),
        "search_entities_by_similarity", rounds=sim_rounds, clear_cache_fn=clear_cache
    )
    print(f"  {results['search_entities_by_similarity']['median_ms']:.1f}ms")

    # --- 7. get_entities_by_absolute_ids (批量, 50个) ---
    batch_ids = [e.absolute_id for e in sample_entities[:10]] * 5  # 50 IDs
    if hasattr(storage, 'get_entities_by_absolute_ids'):
        print(f"  [7/8] get_entities_by_absolute_ids x50 ({rounds} 轮)...", end="", flush=True)
        results["get_entities_by_absolute_ids_x50"] = bench_single(
            lambda: storage.get_entities_by_absolute_ids(batch_ids),
            "get_entities_by_absolute_ids_x50", rounds=rounds, clear_cache_fn=clear_cache
        )
        print(f"  {results['get_entities_by_absolute_ids_x50']['median_ms']:.1f}ms")
    else:
        print(f"  [7/8] get_entity_by_absolute_id x50 loop ({rounds} 轮)...", end="", flush=True)
        results["get_entities_by_absolute_ids_x50"] = bench_single(
            lambda: [storage.get_entity_by_absolute_id(aid) for aid in batch_ids],
            "get_entities_by_absolute_ids_x50", rounds=rounds, clear_cache_fn=clear_cache
        )
        print(f"  {results['get_entities_by_absolute_ids_x50']['median_ms']:.1f}ms")

    # --- 8. _get_entities_with_embeddings (全量加载) ---
    print(f"  [8/8] _get_entities_with_embeddings ({sim_rounds} 轮)...", end="", flush=True)
    results["_get_entities_with_embeddings"] = bench_single(
        lambda: storage._get_entities_with_embeddings(),
        "_get_entities_with_embeddings", rounds=sim_rounds
    )
    print(f"  {results['_get_entities_with_embeddings']['median_ms']:.1f}ms")

    return results


def print_results(results: dict, label: str):
    """打印结果表格。"""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  {'测试项':<40} {'中位数':>8} {'均值':>8} {'最小':>8} {'最大':>8} {'P95':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for key, r in results.items():
        if "error" in r:
            print(f"  {r['label']:<40} {'ERROR':>8}")
        else:
            print(f"  {r['label']:<40} {r['median_ms']:>7.1f}ms {r['mean_ms']:>7.1f}ms {r['min_ms']:>7.1f}ms {r['max_ms']:>7.1f}ms {r['p95_ms']:>7.1f}ms")
    print(f"{'='*70}\n")


def compare_results(before_file: str, after_file: str):
    """对比两次 benchmark 结果。"""
    with open(before_file) as f:
        before = json.load(f)
    with open(after_file) as f:
        after = json.load(f)

    before_data = before["results"]
    after_data = after["results"]

    print(f"\n{'='*80}")
    print(f"  性能对比: {before['label']} vs {after['label']}")
    print(f"  数据量: Entity={before.get('entity_count','?')}, Relation={before.get('relation_count','?')}")
    print(f"{'='*80}")
    print(f"  {'测试项':<40} {'优化前':>10} {'优化后':>10} {'加速比':>8} {'变化':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    for key in before_data:
        b = before_data[key]
        a = after_data.get(key, {})
        if "error" in b or "error" in a:
            print(f"  {b['label']:<40} {'ERROR':>10}")
            continue
        b_med = b["median_ms"]
        a_med = a.get("median_ms", 0)
        if b_med > 0:
            speedup = b_med / a_med if a_med > 0 else float("inf")
            change = ((a_med - b_med) / b_med) * 100
            arrow = "↓" if change < 0 else "↑"
            print(f"  {b['label']:<40} {b_med:>9.1f}ms {a_med:>9.1f}ms {speedup:>7.2f}x {arrow}{abs(change):>8.1f}%")
        else:
            print(f"  {b['label']:<40} {b_med:>9.1f}ms {a_med:>9.1f}ms {'N/A':>8}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="DeepDream 性能基准测试")
    parser.add_argument("--label", default="TEST", help="测试标签 (BEFORE/AFTER)")
    parser.add_argument("--rounds", type=int, default=5, help="每项测试轮数")
    parser.add_argument("--output", "-o", default=None, help="输出 JSON 文件")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE_JSON", "AFTER_JSON"), help="对比两个 JSON 结果")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    print(f"\n>>> DeepDream Performance Benchmark: {args.label}")
    print(f">>> Rounds: {args.rounds}")

    storage = create_storage()
    print(">>> 连接成功")

    # 统计数据量
    try:
        import neo4j
        with storage._driver.session() as session:
            ec = session.run("MATCH (e:Entity) RETURN count(e)").single()[0]
            rc = session.run("MATCH (r:Relation) RETURN count(r)").single()[0]
        print(f">>> 数据量: Entity={ec}, Relation={rc}")
    except Exception as e:
        ec, rc = "unknown", "unknown"
        print(f">>> 数据量统计失败: {e}")

    warmup(storage)
    print()
    results = run_benchmarks(storage, args.label, rounds=args.rounds)

    print_results(results, args.label)

    output = {
        "label": args.label,
        "entity_count": ec,
        "relation_count": rc,
        "results": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  结果已保存: {args.output}")
    else:
        default_file = f"bench_{args.label.lower()}.json"
        with open(default_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  结果已保存: {default_file}")

    storage.close()


if __name__ == "__main__":
    main()
