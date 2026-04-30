"""
Relation Content Batch Writing Experiment
==========================================
Loads real relation pairs from source text (via LLM extraction),
tests different batch sizes (10, 15, 20, 35) for LLM relation content writing.

Usage:
    python -m tests.test_relation_content_batch
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm.client import LLMClient

SOURCE_FILE = Path(__file__).resolve().parent.parent / "core" / "tests" / "data" / "红楼梦_节选.txt"


def load_test_data():
    """Load source text and split into windows."""
    with open(SOURCE_FILE, encoding="utf-8") as f:
        text = f.read()

    window_size = 800
    overlap = 100
    windows = []
    start = 0
    while start < len(text):
        end = min(start + window_size, len(text))
        windows.append(text[start:end])
        start += window_size - overlap
    return text, windows


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


def extract_relation_pairs_from_windows(client: LLMClient, windows: list) -> list:
    """Use LLM to extract relation pairs from multiple windows.

    Returns list of (entity_a, entity_b) tuples.
    """
    all_pairs = []

    # Use a simple extraction prompt for relation pairs
    system = "你是关系提取专家。从文本中提取所有明确提及的实体对及其关系。只输出JSON数组。"
    user_tpl = """从以下文本中提取实体之间的关系对。

要求：
1. 每对关系包含两个实体名和简短关系描述
2. 只提取文本中明确提及或可直接推断的关系
3. 实体名必须与文本中出现的完全一致

文本：
{text}

只输出```json```数组：
```json
[{{"entity1": "实体A", "entity2": "实体B", "relation": "关系描述"}}]
```"""

    seen = set()
    for i, window in enumerate(windows[:4]):
        print(f"  Extracting pairs from window {i+1}/{min(4, len(windows))}...")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_tpl.format(text=window)},
        ]

        try:
            result, _ = client.call_llm_until_json_parses(
                messages, parse_fn=_parse_pairs, json_parse_retries=2,
            )
            if isinstance(result, list):
                for a, b in result:
                    key = tuple(sorted([a, b]))
                    if key not in seen:
                        seen.add(key)
                        all_pairs.append((a, b))
                print(f"    Got {len(result)} pairs, total unique: {len(all_pairs)}")
        except Exception as e:
            print(f"    Error: {e}")

    return all_pairs


def _parse_pairs(response: str) -> list:
    """Parse relation pair extraction response."""
    from core.llm.extraction import ExtractionLLM
    data = ExtractionLLM._parse_json_response(None, response)
    pairs = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("relations") or data.get("data") or data.get("pairs") or []
        if not isinstance(items, list):
            items = []
    else:
        return pairs

    for item in items:
        if isinstance(item, dict):
            a = item.get("entity1", "").strip()
            b = item.get("entity2", "").strip()
            if a and b:
                pairs.append((a, b))
    return pairs


def run_batch_test(client: LLMClient, pairs: list, text: str, chunk_size: int) -> dict:
    t0 = time.time()
    results = client.batch_write_relation_content(pairs, text, chunk_size=chunk_size)
    elapsed = time.time() - t0

    covered = len(results)
    total = len(pairs)
    missing = [(a, b) for a, b in pairs if (a, b) not in results and (b, a) not in results]

    lengths = [len(v) for v in results.values()]
    return {
        "chunk_size": chunk_size,
        "elapsed": elapsed,
        "total_pairs": total,
        "covered": covered,
        "missing": missing[:20],
        "coverage_rate": covered / total if total else 0,
        "avg_len": statistics.mean(lengths) if lengths else 0,
        "median_len": statistics.median(lengths) if lengths else 0,
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "stdev_len": statistics.stdev(lengths) if len(lengths) > 1 else 0,
        "lengths": lengths,
        "results": {f"{a}|{b}": v for (a, b), v in results.items()},
    }


def print_length_distribution(lengths: list):
    buckets = {"<20": 0, "20-30": 0, "30-50": 0, "50-80": 0, ">80": 0}
    for l in lengths:
        if l < 20:
            buckets["<20"] += 1
        elif l < 30:
            buckets["20-30"] += 1
        elif l < 50:
            buckets["30-50"] += 1
        elif l < 80:
            buckets["50-80"] += 1
        else:
            buckets[">80"] += 1
    total = len(lengths)
    for label, cnt in buckets.items():
        pct = cnt / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"    {label:>8}: {cnt:>3} ({pct:>5.1f}%) {bar}")


def print_comparison(all_results: list, sample_pairs: list):
    print("\n" + "=" * 100)
    print("Relation Content Batch Writing — Real Data Comparison")
    print("=" * 100)

    print(f"\n{'Chunk':>6} | {'Time':>7} | {'Cover':>6} | {'Rate':>5} | "
          f"{'Avg':>5} | {'Med':>5} | {'Min':>4} | {'Max':>4} | {'StdDev':>6} | {'Missing'}")
    print("-" * 100)

    for r in all_results:
        missing_str = f"{len(r['missing'])} pairs"
        print(f"{r['chunk_size']:>6} | {r['elapsed']:>6.1f}s | {r['covered']:>3}/{r['total_pairs']:<3} | "
              f"{r['coverage_rate']:>4.0%} | "
              f"{r['avg_len']:>5.0f} | {r['median_len']:>5.0f} | {r['min_len']:>4} | {r['max_len']:>4} | "
              f"{r['stdev_len']:>6.1f} | {missing_str}")

    # Length distribution
    print("\n" + "=" * 100)
    print("Content Length Distribution")
    print("=" * 100)
    for r in all_results:
        print(f"\n  chunk_size={r['chunk_size']}:")
        print_length_distribution(r["lengths"])

    # Quality samples
    print("\n" + "=" * 100)
    print("Quality Sample — 10 Pairs (side-by-side)")
    print("=" * 100)

    chunk_sizes = [r["chunk_size"] for r in all_results]
    header = f"{'Pair':<30}"
    for cs in chunk_sizes:
        header += f" | chunk={cs:<6}"
    print(f"\n{header}")
    print("-" * 100)

    for a, b in sample_pairs:
        pair_key = f"{a} ↔ {b}"
        row = f"{pair_key:<30}"
        for r in all_results:
            content = r["results"].get(f"{a}|{b}", r["results"].get(f"{b}|{a}", "(MISSING)"))
            row += f" | {content[:25]:<14}"
        print(row)

    # Full content for review
    print("\n" + "=" * 100)
    print("Full Quality Sample Content (10 pairs × chunk sizes)")
    print("=" * 100)

    for a, b in sample_pairs:
        print(f"\n--- {a} ↔ {b} ---")
        for r in all_results:
            content = r["results"].get(f"{a}|{b}", r["results"].get(f"{b}|{a}", "(MISSING)"))
            print(f"  chunk={r['chunk_size']:>3} [{len(content):>3}字]: {content}")


def main():
    config_path = Path(__file__).resolve().parent.parent / "service_config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    print("Loading source text...")
    full_text, windows = load_test_data()
    print(f"  Source: {len(full_text)} chars, {len(windows)} windows")

    # Step 1: Extract real relation pairs from source text
    print("\nExtracting relation pairs from source text...")
    client = create_llm_client(config)
    pairs = extract_relation_pairs_from_windows(client, windows)

    if len(pairs) < 10:
        print(f"WARNING: Only got {len(pairs)} pairs. Adding some manual pairs for testing.")
        # Add known pairs from 红楼梦 to ensure enough data
        manual_pairs = [
            ("曹雪芹", "红楼梦"), ("甄士隐", "贾雨村"), ("通灵宝玉", "贾宝玉"),
            ("大荒山无稽崖", "青埂峰"), ("一僧一道", "顽石"), ("甄士隐", "葫芦庙"),
            ("封氏", "甄士隐"), ("英莲", "甄士隐"), ("贾雨村", "娇杏"),
            ("茫茫大士", "渺渺真人"), ("空空道人", "石头记"), ("孔梅溪", "风月宝鉴"),
            ("甄英莲", "葫芦僧"), ("霍启", "甄士隐"), ("严老爷", "贾雨村"),
            ("十里街", "仁清巷"), ("葫芦庙", "姑苏"), ("真事隐去", "假语村言"),
            ("荣华富贵", "打动凡心"), ("茅椽蓬牖", "锦衣纨绔"),
        ]
        seen = set(tuple(sorted(p)) for p in pairs)
        for a, b in manual_pairs:
            key = tuple(sorted([a, b]))
            if key not in seen:
                seen.add(key)
                pairs.append((a, b))

    print(f"\nTotal relation pairs: {len(pairs)}")
    for a, b in pairs[:20]:
        print(f"  {a} ↔ {b}")
    if len(pairs) > 20:
        print(f"  ... and {len(pairs)-20} more")

    # Use first window text for the batch test
    test_text = windows[0] + windows[1] + windows[2] + windows[3]
    print(f"\nTest text: {len(test_text)} chars (4 windows concatenated)")

    # Step 2: Run batch experiments
    chunk_sizes = [10, 15, 20, 35]
    all_results = []

    for cs in chunk_sizes:
        n_calls = len(pairs) // cs + (1 if len(pairs) % cs else 0)
        print(f"\n{'='*50}")
        print(f"Testing chunk_size={cs} ({len(pairs)} pairs → {n_calls} LLM calls)")
        print(f"{'='*50}")
        client = create_llm_client(config)
        r = run_batch_test(client, pairs, test_text, cs)
        all_results.append(r)
        print(f"  Time: {r['elapsed']:.1f}s | Covered: {r['covered']}/{r['total_pairs']} | "
              f"Avg: {r['avg_len']:.0f} chars | Median: {r['median_len']:.0f}")

    # Step 3: Pick sample pairs for quality review
    total = len(pairs)
    sample_indices = [0, total // 10, 2 * total // 10, 3 * total // 10,
                      4 * total // 10, 5 * total // 10, 6 * total // 10,
                      7 * total // 10, 8 * total // 10, total - 1]
    sample_pairs = [pairs[i] for i in sample_indices if i < total]

    print_comparison(all_results, sample_pairs)

    # Save results
    output_dir = Path(__file__).resolve().parent / "result"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for r in all_results:
        s = {k: v for k, v in r.items() if k not in ("results", "lengths")}
        summary.append(s)
        batch_path = output_dir / f"relation_content_batch_{r['chunk_size']}.json"
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(r["results"], f, ensure_ascii=False, indent=2)
        print(f"Saved chunk {r['chunk_size']} → {batch_path}")

    summary_path = output_dir / "relation_content_batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary → {summary_path}")


if __name__ == "__main__":
    main()
