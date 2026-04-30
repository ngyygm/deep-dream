"""
Entity Content Batch Writing Experiment (Real DB Data)
======================================================
Loads a real episode from Neo4j with its actual entities,
tests different batch sizes (10, 20, 30, 50) for LLM content writing.

Usage:
    python -m tests.test_entity_content_batch
"""

import json
import statistics
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm.client import LLMClient

BASE_URL = "http://localhost:16200/api/v1"
EP_ID = "cache_20260428_030440_ab50ab3a"


def load_real_data():
    """Load source text and entity names from DB."""
    ep_resp = requests.get(f"{BASE_URL}/episodes/{EP_ID}", params={"graph_id": "default"})
    ep_resp.raise_for_status()
    source_text = ep_resp.json()["data"]["source_text"]

    ents_resp = requests.get(f"{BASE_URL}/episodes/{EP_ID}/entities", params={"graph_id": "default"})
    ents_resp.raise_for_status()
    items = ents_resp.json()["data"]["entities"]
    entity_names = [e["name"] for e in items if e.get("target_type") == "entity"]

    return source_text, entity_names


def create_llm_client(config: dict) -> LLMClient:
    llm_cfg = config.get("llm", {})
    ext = llm_cfg.get("extraction", llm_cfg)
    return LLMClient(
        api_key=ext.get("api_key", "ollama"),
        model_name=ext.get("model", "gemma4-26b-32k"),
        base_url=ext.get("base_url", "http://localhost:11434/v1"),
        think_mode=False,
        max_tokens=ext.get("max_tokens", 16000),
        context_window_tokens=ext.get("context_window_tokens", 32000),
        timeout_seconds=llm_cfg.get("timeout_seconds", 300),
    )


def run_batch_test(client: LLMClient, entities: list, text: str, batch_size: int) -> dict:
    t0 = time.time()
    results = client.batch_write_entity_content(entities, text, chunk_size=batch_size)
    elapsed = time.time() - t0

    covered = len(results)
    total = len(entities)
    missing = [e for e in entities if e not in results]

    lengths = [len(v) for v in results.values()]
    return {
        "batch_size": batch_size,
        "elapsed": elapsed,
        "total_entities": total,
        "covered": covered,
        "missing": missing,
        "coverage_rate": covered / total if total else 0,
        "avg_len": statistics.mean(lengths) if lengths else 0,
        "median_len": statistics.median(lengths) if lengths else 0,
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "stdev_len": statistics.stdev(lengths) if len(lengths) > 1 else 0,
        "lengths": lengths,
        "results": results,
    }


def print_length_distribution(lengths: list):
    buckets = {"<30": 0, "30-50": 0, "50-80": 0, "80-100": 0, ">100": 0}
    for l in lengths:
        if l < 30:
            buckets["<30"] += 1
        elif l < 50:
            buckets["30-50"] += 1
        elif l < 80:
            buckets["50-80"] += 1
        elif l <= 100:
            buckets["80-100"] += 1
        else:
            buckets[">100"] += 1
    total = len(lengths)
    for label, cnt in buckets.items():
        pct = cnt / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"    {label:>6}: {cnt:>3} ({pct:>5.1f}%) {bar}")


def print_comparison(all_results: list):
    print("\n" + "=" * 90)
    print("Entity Content Batch Writing — Real DB Data Comparison")
    print("=" * 90)

    # Summary table
    print(f"\n{'Batch':>6} | {'Time':>7} | {'Cover':>6} | {'Rate':>5} | "
          f"{'Avg':>5} | {'Med':>5} | {'Min':>4} | {'Max':>4} | {'StdDev':>6} | {'Missing'}")
    print("-" * 90)

    for r in all_results:
        missing_str = ", ".join(r["missing"][:4])
        if len(r["missing"]) > 4:
            missing_str += f"... ({len(r['missing'])} total)"
        print(f"{r['batch_size']:>6} | {r['elapsed']:>6.1f}s | {r['covered']:>3}/{r['total_entities']:<3} | "
              f"{r['coverage_rate']:>4.0%} | "
              f"{r['avg_len']:>5.0f} | {r['median_len']:>5.0f} | {r['min_len']:>4} | {r['max_len']:>4} | "
              f"{r['stdev_len']:>6.1f} | {missing_str}")

    # Length distribution per batch
    print("\n" + "=" * 90)
    print("Content Length Distribution")
    print("=" * 90)
    for r in all_results:
        print(f"\n  batch_size={r['batch_size']}:")
        print_length_distribution(r["lengths"])

    # Quality samples: 10 entities
    print("\n" + "=" * 90)
    print("Quality Sample — 10 Entities (side-by-side)")
    print("=" * 90)

    # Pick 10 evenly-spaced entities
    entity_names = all_results[0]["results"].keys() if all_results else []
    if not entity_names:
        # Fallback: use entities from the test
        return

    all_entity_list = sorted(set(all_results[0]["results"].keys()))
    for r in all_results[1:]:
        all_entity_list = sorted(set(all_entity_list) | set(r["results"].keys()))

    # Pick indices for sampling
    total = len(all_entity_list)
    sample_indices = [0, total // 10, 2 * total // 10, 3 * total // 10,
                      4 * total // 10, 5 * total // 10, 6 * total // 10,
                      7 * total // 10, 8 * total // 10, total - 1]
    sample_entities = [all_entity_list[i] for i in sample_indices if i < total]

    # Print header
    batch_sizes = [r["batch_size"] for r in all_results]
    header = f"{'Entity':<20}"
    for bs in batch_sizes:
        header += f" | batch={bs:<6}"
    print(f"\n{header}")
    print("-" * 90)

    for entity in sample_entities:
        row = f"{entity:<20}"
        for r in all_results:
            content = r["results"].get(entity, "(MISSING)")
            row += f" | {content[:30]:<14}"
        print(row)

    # Full content for detailed review
    print("\n" + "=" * 90)
    print("Full Quality Sample Content (10 entities × batch sizes)")
    print("=" * 90)

    for entity in sample_entities:
        print(f"\n--- {entity} ---")
        for r in all_results:
            content = r["results"].get(entity, "(MISSING)")
            print(f"  batch={r['batch_size']:>3} [{len(content):>3}字]: {content}")


def main():
    config_path = Path(__file__).resolve().parent.parent / "service_config.json"
    with open(config_path) as f:
        config = json.load(f)

    print("Loading real data from database...")
    source_text, entity_names = load_real_data()
    print(f"  Episode: {EP_ID}")
    print(f"  Source text: {len(source_text)} chars")
    print(f"  Entities: {len(entity_names)}")
    print(f"  Model: {config['llm']['extraction']['model']}")

    batch_sizes = [10, 20, 30, 50]
    all_results = []

    for bs in batch_sizes:
        print(f"\n{'='*40}")
        print(f"Testing batch_size={bs} ({len(entity_names)} entities → {len(entity_names)//bs + (1 if len(entity_names)%bs else 0)} LLM calls)")
        print(f"{'='*40}")
        client = create_llm_client(config)
        r = run_batch_test(client, entity_names, source_text, bs)
        all_results.append(r)
        print(f"  Time: {r['elapsed']:.1f}s | Covered: {r['covered']}/{r['total_entities']} | "
              f"Avg: {r['avg_len']:.0f} chars | Median: {r['median_len']:.0f}")

    print_comparison(all_results)

    # Save results
    output_dir = Path(__file__).resolve().parent / "result"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for r in all_results:
        s = {k: v for k, v in r.items() if k not in ("results", "lengths")}
        summary.append(s)
        batch_path = output_dir / f"entity_content_batch_{r['batch_size']}.json"
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(r["results"], f, ensure_ascii=False, indent=2)
        print(f"Saved batch {r['batch_size']} → {batch_path}")

    summary_path = output_dir / "entity_content_batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary → {summary_path}")


if __name__ == "__main__":
    main()
