"""Run cache continuity experiment: sequential processing across multiple windows.

Tests how well the memory cache maintains information when processing
sequential text segments, and how it handles cross-document / cross-type transitions.

Usage:
    python -m experiments.prompt_optimization.run_cache_continuity
    python -m experiments.prompt_optimization.run_cache_continuity --scenarios 3
"""
import argparse
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    DISTILL_DIR, RESULTS_DIR, RESOURCES_DIR, STEP_DIRS,
    NOVELS, RANDOM_SEED, LLM_CONFIG,
)
from .data_loader import (
    load_window,
    extract_input_text,
    extract_tag_content,
    find_resource_windows,
    load_resource_window,
)
from .llm_runner import ExperimentLLMRunner
from .prompt_variants import get_variant, get_variant_names
from .evaluators import aggregate_results


# ============================================================
# Scenario Definitions
# ============================================================

def build_scenarios(
    distill_dir: str = DISTILL_DIR,
    resources_dir: str = RESOURCES_DIR,
    seed: int = RANDOM_SEED,
) -> List[Dict[str, Any]]:
    """Build test scenarios for cache continuity.

    Each scenario has:
        - name: descriptive name
        - windows: list of 3 window specs
          Each spec: {"type": "distill"|"resource", "key": "...", "step1_dir": "..."}
    """
    random.seed(seed)

    # Gather available windows per novel
    cache_dir = os.path.join(distill_dir, STEP_DIRS[1])
    novel_windows: Dict[str, List[str]] = {}
    if os.path.isdir(cache_dir):
        for fn in os.listdir(cache_dir):
            if not fn.endswith(".jsonl"):
                continue
            parts = fn.rsplit("_", 1)
            key = parts[0]
            novel = key.split(".")[0]
            novel_windows.setdefault(novel, []).append(key)

    # Gather available resources
    resource_keys = []
    if os.path.isdir(resources_dir):
        for cat in sorted(os.listdir(resources_dir)):
            cat_path = os.path.join(resources_dir, cat)
            if not os.path.isdir(cat_path):
                continue
            for fn in sorted(os.listdir(cat_path)):
                if fn.endswith(".txt"):
                    resource_keys.append(f"{cat}/{fn[:-4]}")

    scenarios = []

    # Scenario 1: Same document sequential (novel)
    for novel in NOVELS[:2]:  # 红楼梦, 三国演义
        pool = novel_windows.get(novel, [])
        if len(pool) >= 3:
            chosen = random.sample(pool, 3)
            scenarios.append({
                "name": f"同文档连续（{novel}）",
                "type": "same_doc",
                "windows": [
                    {"type": "distill", "key": k}
                    for k in chosen
                ],
            })

    # Scenario 2: Cross-document within same type (novel → novel → novel)
    cross_novel_keys = []
    for novel in NOVELS:
        pool = novel_windows.get(novel, [])
        if pool:
            cross_novel_keys.append(random.choice(pool))
    if len(cross_novel_keys) >= 3:
        scenarios.append({
            "name": "跨文档切换（小说→小说→小说）",
            "type": "cross_doc",
            "windows": [
                {"type": "distill", "key": k}
                for k in cross_novel_keys[:3]
            ],
        })

    # Scenario 3: Cross-type (novel → code/log → prose)
    if novel_windows.get("红楼梦") and len(resource_keys) >= 2:
        # Pick resources from different categories
        categories = {}
        for rk in resource_keys:
            cat = rk.split("/")[0]
            categories.setdefault(cat, []).append(rk)

        diverse_resources = []
        for cat in ["code", "logs", "prose", "mixed", "chat"]:
            if cat in categories:
                diverse_resources.append(random.choice(categories[cat]))

        if len(diverse_resources) >= 2:
            scenarios.append({
                "name": "跨类型切换（小说→技术→日志）",
                "type": "cross_type",
                "windows": [
                    {"type": "distill", "key": random.choice(novel_windows["红楼梦"])},
                    {"type": "resource", "key": diverse_resources[0]},
                    {"type": "resource", "key": diverse_resources[1]},
                ],
            })

    # Scenario 4: Resource-only sequential (code → prose → project)
    if len(resource_keys) >= 3:
        # Pick from different categories
        cats_needed = ["code", "prose", "project"]
        res_picks = []
        for cat in cats_needed:
            cat_keys = [k for k in resource_keys if k.startswith(f"{cat}/")]
            if cat_keys:
                res_picks.append(random.choice(cat_keys))
        if len(res_picks) >= 3:
            scenarios.append({
                "name": "资源连续（代码→随笔→项目书）",
                "type": "resource_seq",
                "windows": [
                    {"type": "resource", "key": k}
                    for k in res_picks[:3]
                ],
            })

    # Scenario 5: Chat → project → chat
    chat_keys = [k for k in resource_keys if k.startswith("chat/")]
    project_keys = [k for k in resource_keys if k.startswith("project/")]
    if len(chat_keys) >= 2 and len(project_keys) >= 1:
        scenarios.append({
            "name": "聊天→项目→聊天",
            "type": "chat_project",
            "windows": [
                {"type": "resource", "key": chat_keys[0]},
                {"type": "resource", "key": project_keys[0]},
                {"type": "resource", "key": chat_keys[1] if len(chat_keys) > 1 else chat_keys[0]},
            ],
        })

    print(f"Built {len(scenarios)} cache continuity scenarios")
    for s in scenarios:
        print(f"  {s['name']}: {len(s['windows'])} windows")
    return scenarios


# ============================================================
# Sequential Processing
# ============================================================

def extract_input_for_window(
    window_spec: Dict,
    distill_dir: str,
    resources_dir: str,
) -> str:
    """Extract input text from a window spec."""
    if window_spec["type"] == "distill":
        data = load_window(distill_dir, STEP_DIRS[1], window_spec["key"])
        if data is None:
            return ""
        messages = data if isinstance(data, list) else data.get("messages", [])
        return extract_input_text(messages)
    elif window_spec["type"] == "resource":
        res = load_resource_window(resources_dir, window_spec["key"])
        return res["content"] if res else ""
    return ""


def build_cache_user_prompt(cache_content: str, input_text: str) -> str:
    """Build user prompt for cache update step."""
    return f"""<记忆缓存>
{cache_content}
</记忆缓存>

<输入文本>
{input_text}
</输入文本>"""


def extract_entities_from_text(text: str) -> set:
    """Extract 2-4 char Chinese names as proxy for entities."""
    return set(re.findall(r"[\u4e00-\u9fff]{2,4}", text[:500]))


# ============================================================
# Evaluation
# ============================================================

def evaluate_continuity(
    scenario: Dict,
    cache_outputs: List[str],
    input_texts: List[str],
) -> Dict[str, Any]:
    """Evaluate cache continuity for one scenario run."""
    m: Dict[str, Any] = {"parse_success": 1.0}

    if len(cache_outputs) < 3 or len(input_texts) < 3:
        m["error"] = "insufficient outputs"
        return m

    # Entity extraction from inputs
    entities_w1 = extract_entities_from_text(input_texts[0])
    entities_w2 = extract_entities_from_text(input_texts[1])
    entities_w3 = extract_entities_from_text(input_texts[2])

    # All unique entities across windows
    all_entities = entities_w1 | entities_w2 | entities_w3

    # --- Entity retention rate ---
    # How many W1 entities still appear in W3 cache?
    w3_cache_entities = extract_entities_from_text(cache_outputs[2])
    retained_w1 = entities_w1 & w3_cache_entities
    m["entity_retention_rate"] = len(retained_w1) / max(1, len(entities_w1))

    # --- New info coverage ---
    # How many W3 input entities appear in W3 cache?
    covered_w3 = entities_w3 & w3_cache_entities
    m["new_info_coverage"] = len(covered_w3) / max(1, len(entities_w3))

    # --- Cache length growth ---
    lengths = [len(c) for c in cache_outputs]
    m["cache_length_initial"] = lengths[0]
    m["cache_length_final"] = lengths[2]
    if lengths[0] > 0:
        m["cache_length_growth"] = lengths[2] / lengths[0]
    else:
        m["cache_length_growth"] = 0.0

    # --- Cross-doc purge rate (only for cross_doc / cross_type) ---
    if scenario["type"] in ("cross_doc", "cross_type", "chat_project"):
        # After switching away from W1's document, how much of W1 entities
        # are still in the final cache?
        m["cross_doc_retention"] = len(retained_w1) / max(1, len(entities_w1))

    # --- Verbatim ratio (should not just copy input) ---
    for i in range(3):
        if input_texts[i] and cache_outputs[i]:
            set_cache = set(cache_outputs[i][:300])
            set_input = set(input_texts[i][:300])
            if set_cache and set_input:
                ratio = len(set_cache & set_input) / max(1, len(set_cache | set_input))
                m[f"verbatim_ratio_w{i+1}"] = ratio
            else:
                m[f"verbatim_ratio_w{i+1}"] = 0.0

    return m


# ============================================================
# Main Runner
# ============================================================

def run_cache_experiment(
    scenarios: Optional[List[Dict]] = None,
    round_num: int = 2,
    max_scenarios: int = 10,
) -> Dict[str, Any]:
    """Run cache continuity experiment across all variants."""
    distill_dir = DISTILL_DIR
    resources_dir = RESOURCES_DIR

    if scenarios is None:
        scenarios = build_scenarios(distill_dir, resources_dir)
    scenarios = scenarios[:max_scenarios]

    variant_names = get_variant_names(1, round_num=round_num)
    runner = ExperimentLLMRunner()

    results: Dict[str, List[Dict]] = {name: [] for name in variant_names}

    print(f"\n{'='*60}")
    print("Cache Continuity Experiment")
    print(f"Round: {round_num}")
    print(f"Variants: {variant_names}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"{'='*60}")

    for si, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {si+1}/{len(scenarios)}: {scenario['name']} ---")

        # Load input texts for all windows
        input_texts = []
        for ws in scenario["windows"]:
            text = extract_input_for_window(ws, distill_dir, resources_dir)
            if not text:
                print(f"  WARNING: empty input for {ws['key']}")
            input_texts.append(text)

        if any(not t for t in input_texts):
            print(f"  SKIP: missing input text")
            continue

        for vname in variant_names:
            system_prompt, desc = get_variant(1, vname, round_num=round_num)

            cache_outputs = []
            try:
                for wi in range(3):
                    if wi == 0:
                        cache_content = "（系统初始化，暂无缓存）"
                    else:
                        cache_content = cache_outputs[wi - 1]

                    user_prompt = build_cache_user_prompt(cache_content, input_texts[wi])
                    response = runner.call(system_prompt, user_prompt)
                    cache_outputs.append(response)

                    # Extract just the cache content (between section headers)
                    # The response IS the new cache

            except Exception as e:
                print(f"  {vname}: ERROR {e}")
                results[vname].append({"parse_success": 0.0, "error": str(e)})
                continue

            # Evaluate
            metrics = evaluate_continuity(scenario, cache_outputs, input_texts)
            metrics["scenario"] = scenario["name"]
            metrics["scenario_type"] = scenario["type"]
            results[vname].append(metrics)

            retention = metrics.get("entity_retention_rate", 0)
            coverage = metrics.get("new_info_coverage", 0)
            growth = metrics.get("cache_length_growth", 0)
            print(f"  {vname}: retention={retention:.2f} coverage={coverage:.2f} growth={growth:.1f}x")

        time.sleep(0.5)

    # Aggregate
    aggregated = {}
    for vname in variant_names:
        aggregated[vname] = aggregate_results(results[vname])
        aggregated[vname]["_scenario_count"] = len(results[vname])

    # Save raw results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_path = os.path.join(RESULTS_DIR, "cache_continuity_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nRaw results saved to {raw_path}")

    # Save aggregated
    agg_path = os.path.join(RESULTS_DIR, "cache_continuity_aggregated.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"Aggregated saved to {agg_path}")

    # Print comparison
    print_cache_comparison(aggregated)

    return aggregated


def print_cache_comparison(aggregated: Dict[str, Any]):
    """Print comparison table for cache continuity."""
    variant_names = [k for k in aggregated if not k.startswith("_")]

    mean_keys = sorted(k for k in set().union(*[aggregated[v].keys() for v in variant_names])
                       if k.endswith("_mean"))

    print(f"\n{'='*70}")
    print("Cache Continuity Comparison")
    print(f"{'='*70}")

    header = f"{'Metric':<30}"
    for vname in variant_names:
        header += f"{vname:>18}"
    print(header)
    print("-" * len(header))

    for key in mean_keys:
        label = key.replace("_mean", "")
        row = f"{label:<30}"
        for vname in variant_names:
            val = aggregated[vname].get(key)
            if val is not None:
                row += f"{val:>18.3f}"
            else:
                row += f"{'N/A':>18}"
        print(row)

    row = f"{'scenarios':<30}"
    for vname in variant_names:
        row += f"{aggregated[vname].get('_scenario_count', 0):>18}"
    print(row)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cache continuity experiment")
    parser.add_argument("--round", type=int, default=2, help="Experiment round")
    parser.add_argument("--scenarios", type=int, default=10, help="Max scenarios to run")
    args = parser.parse_args()

    run_cache_experiment(round_num=args.round, max_scenarios=args.scenarios)
