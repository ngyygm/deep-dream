"""Run a single pipeline step experiment across all prompt variants.

Usage:
    python -m experiments.prompt_optimization.run_step --step 2
    python -m experiments.prompt_optimization.run_step --step 3 --windows 10
"""
import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from .config import STEP_DIRS, RESULTS_DIR
from .data_loader import (
    find_common_windows,
    load_window,
    extract_input_text,
    extract_memory_cache,
    extract_entity_list,
    extract_system_prompt,
    extract_assistant_response,
    parse_json_from_response,
)
from .evaluators import aggregate_results
from .llm_runner import ExperimentLLMRunner
from .prompt_variants import STEP_VARIANTS, ROUND2_VARIANTS, get_variant, get_variant_names
from .data_loader import find_resource_windows, load_resource_window, build_resource_user_prompt
from .data_loader import load_all_alignment_entries

# Map step -> evaluator function (lazy import to avoid circular)
EVALUATOR_MAP = {
    1: "evaluate_step1",
    2: "evaluate_step2",
    3: "evaluate_step3",
    4: "evaluate_step4",
    5: "evaluate_step5",
    6: "evaluate_step6_batch",
    7: "evaluate_step7_batch",
}


def build_user_prompt(step: int, messages: List[Dict], step2_response: str = "") -> str:
    """Reconstruct user prompt from distillation messages."""
    # Find the user message content
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def run_step_experiment(
    step: int,
    windows: Optional[List[str]] = None,
    distill_dir: Optional[str] = None,
    max_windows: int = 20,
    round_num: int = 2,
    use_resources: bool = False,
) -> Dict[str, Any]:
    """Run experiment for one pipeline step.

    Returns dict with variant_name -> aggregated metrics.
    """
    from .config import DISTILL_DIR as _DISTILL_DIR, RESOURCES_DIR as _RES_DIR
    distill_dir = distill_dir or _DISTILL_DIR

    if windows is None:
        if use_resources:
            windows = find_resource_windows(resources_dir=_RES_DIR, total=max_windows)
        else:
            windows = find_common_windows(distill_dir=distill_dir, total=max_windows)

    step_name = STEP_DIRS[step]
    variant_names = get_variant_names(step, round_num=round_num)
    runner = ExperimentLLMRunner()

    # Import evaluator
    from . import evaluators as eval_mod
    evaluator_fn = getattr(eval_mod, EVALUATOR_MAP[step])

    results: Dict[str, List[Dict]] = {name: [] for name in variant_names}

    print(f"\n{'='*60}")
    print(f"Step {step}: {step_name}")
    print(f"Round: {round_num}")
    print(f"Variants: {variant_names}")
    print(f"Mode: {'resource' if use_resources else 'distill'}")
    print(f"Windows: {len(windows)}")
    print(f"{'='*60}")

    for wi, window_key in enumerate(windows):
        if use_resources:
            # Load from resource library
            resource = load_resource_window(_RES_DIR, window_key)
            if resource is None:
                print(f"  [{wi+1}/{len(windows)}] {window_key}: NO DATA, skipping")
                continue
            user_prompt = build_resource_user_prompt(step, resource)
            input_text = resource["content"]
            messages = []  # No distillation messages for resource mode
        else:
            # Load from distillation data
            window_data = load_window(distill_dir, step_name, window_key)
            if window_data is None:
                print(f"  [{wi+1}/{len(windows)}] {window_key}: NO DATA, skipping")
                continue
            messages = window_data if isinstance(window_data, list) else window_data.get("messages", [])
            user_prompt = build_user_prompt(step, messages)
            input_text = extract_input_text(messages)

        if not user_prompt:
            print(f"  [{wi+1}/{len(windows)}] {window_key}: NO USER PROMPT, skipping")
            continue

        print(f"  [{wi+1}/{len(windows)}] {window_key} (input={len(input_text)} chars)")

        for vname in variant_names:
            system_prompt, desc = get_variant(step, vname, round_num=round_num)
            prompt_tokens = runner.estimate_tokens(system_prompt + user_prompt)

            try:
                response = runner.call(system_prompt, user_prompt)
            except Exception as e:
                print(f"    {vname}: ERROR {e}")
                results[vname].append({
                    "parse_success": 0.0,
                    "error": str(e),
                    "prompt_tokens": prompt_tokens,
                })
                continue

            # Build eval kwargs
            eval_kwargs = {"input_text": input_text, "prompt_tokens": prompt_tokens}

            # Step-specific context
            if step == 3:
                if use_resources:
                    entity_names = set()  # No entity list in resource mode
                else:
                    entity_list_str = extract_entity_list(messages)
                    entity_names = set()
                    for line in entity_list_str.split("\n"):
                        line = line.strip().lstrip("- ").strip()
                        if line:
                            entity_names.add(line.split("（")[0].split("(")[0].strip())
                eval_kwargs["entity_names"] = entity_names if entity_names else None
            elif step == 4:
                # Extract requested names from user prompt
                requested = set()
                for line in user_prompt.split("\n"):
                    line = line.strip().lstrip("- ").strip()
                    if line and len(line) < 30:
                        requested.add(line)
                eval_kwargs["requested_names"] = requested
            elif step == 5:
                eval_kwargs["original_content"] = input_text

            try:
                metrics = evaluator_fn(response, **eval_kwargs)
            except Exception as e:
                print(f"    {vname}: EVAL ERROR {e}")
                metrics = {"parse_success": 0.0, "eval_error": str(e), "prompt_tokens": prompt_tokens}

            metrics["response_length"] = len(response)
            results[vname].append(metrics)

            # Brief status
            status = "OK" if metrics.get("parse_success", 0) == 1.0 else "PARSE_FAIL"
            count = metrics.get("entity_count", metrics.get("relation_count", ""))
            print(f"    {vname}: {status} | resp={len(response)} | count={count}")

        # Small delay between windows to respect rate limits
        time.sleep(0.5)

    # Aggregate
    aggregated = {}
    for vname in variant_names:
        aggregated[vname] = aggregate_results(results[vname])
        aggregated[vname]["_window_count"] = len(results[vname])

    # Save raw results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_path = os.path.join(RESULTS_DIR, f"step{step}_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Raw results saved to {raw_path}")

    # Save aggregated
    agg_path = os.path.join(RESULTS_DIR, f"step{step}_aggregated.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"  Aggregated results saved to {agg_path}")

    return aggregated


def print_step_comparison(step: int, aggregated: Dict[str, Any]):
    """Print a comparison table for one step."""
    variant_names = [k for k in aggregated.keys() if not k.startswith("_")]

    # Collect all metric keys
    all_keys = set()
    for vname in variant_names:
        all_keys.update(k for k in aggregated[vname] if not k.startswith("_"))

    # Filter to mean values
    mean_keys = sorted(k for k in all_keys if k.endswith("_mean"))

    print(f"\n{'='*70}")
    print(f"Step {step} Comparison")
    print(f"{'='*70}")

    # Header
    header = f"{'Metric':<35}"
    for vname in variant_names:
        header += f"{vname:>15}"
    print(header)
    print("-" * len(header))

    for key in mean_keys:
        label = key.replace("_mean", "")
        row = f"{label:<35}"
        for vname in variant_names:
            val = aggregated[vname].get(key, None)
            if val is not None:
                row += f"{val:>15.3f}"
            else:
                row += f"{'N/A':>15}"
        print(row)

    # Window count
    row = f"{'windows':<35}"
    for vname in variant_names:
        row += f"{aggregated[vname].get('_window_count', 0):>15}"
    print(row)
    print()


def run_alignment_experiment(
    step: int,
    max_entries: int = 60,
    round_num: int = 2,
) -> Dict[str, Any]:
    """Run alignment experiment for Steps 6/7.

    Uses multi-entry JSONL data with ground truth for correctness evaluation.
    """
    from .config import DISTILL_DIR as _DISTILL_DIR

    variant_names = get_variant_names(step, round_num=round_num)
    runner = ExperimentLLMRunner()

    from . import evaluators as eval_mod
    evaluator_fn = getattr(eval_mod, EVALUATOR_MAP[step])

    # Load and sample alignment entries
    entries = load_all_alignment_entries(
        distill_dir=_DISTILL_DIR, step=step, max_entries=max_entries, seed=42
    )
    if not entries:
        print("  No entries loaded, aborting.")
        return {}

    results: Dict[str, List[Dict]] = {name: [] for name in variant_names}

    print(f"\n{'='*60}")
    print(f"Step {step}: {STEP_DIRS[step]} (Alignment Mode)")
    print(f"Round: {round_num}")
    print(f"Variants: {variant_names}")
    print(f"Entries: {len(entries)}")
    print(f"{'='*60}")

    for ei, entry in enumerate(entries):
        user_prompt = entry.get("user_prompt", "")
        gt_parsed = entry.get("ground_truth_parsed", {})
        gt_response = entry.get("ground_truth_response", "")
        gt_action = entry.get("ground_truth_action", "")

        if not user_prompt:
            print(f"  [{ei+1}/{len(entries)}] EMPTY PROMPT, skipping")
            continue

        print(f"  [{ei+1}/{len(entries)}] action={gt_action} | prompt={len(user_prompt)} chars")

        for vname in variant_names:
            system_prompt, desc = get_variant(step, vname, round_num=round_num)
            prompt_tokens = runner.estimate_tokens(system_prompt + user_prompt)

            try:
                response = runner.call(system_prompt, user_prompt)
            except Exception as e:
                print(f"    {vname}: ERROR {e}")
                results[vname].append({
                    "parse_success": 0.0,
                    "error": str(e),
                    "prompt_tokens": prompt_tokens,
                })
                continue

            try:
                metrics = evaluator_fn(
                    response,
                    ground_truth=gt_parsed,
                    prompt_tokens=prompt_tokens,
                )
            except Exception as e:
                print(f"    {vname}: EVAL ERROR {e}")
                metrics = {"parse_success": 0.0, "eval_error": str(e), "prompt_tokens": prompt_tokens}

            metrics["response_length"] = len(response)
            results[vname].append(metrics)

            status = "OK" if metrics.get("parse_success", 0) == 1.0 else "PARSE_FAIL"
            pred = metrics.get("predicted_action", "")
            conf = metrics.get("confidence", 0.0)
            acc = metrics.get(
                "decision_match", metrics.get("action_accuracy", "")
            )
            print(f"    {vname}: {status} | pred={pred} | gt={gt_action} | acc={acc} | conf={conf:.2f}")

        time.sleep(0.3)

    # Aggregate
    aggregated = {}
    for vname in variant_names:
        aggregated[vname] = aggregate_results(results[vname])
        aggregated[vname]["_entry_count"] = len(results[vname])

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_path = os.path.join(RESULTS_DIR, f"step{step}_alignment_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Raw results saved to {raw_path}")

    agg_path = os.path.join(RESULTS_DIR, f"step{step}_alignment_aggregated.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"  Aggregated results saved to {agg_path}")

    return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prompt experiment for one step")
    parser.add_argument("--step", type=int, required=True, choices=range(1, 8))
    parser.add_argument("--windows", type=int, default=20, help="Number of test windows")
    parser.add_argument("--round", type=int, default=2, help="Experiment round (1 or 2)")
    parser.add_argument("--resource", action="store_true", help="Use resource library instead of distill data")
    args = parser.parse_args()

    if args.step >= 6:
        # Steps 6/7 use alignment experiment with multi-entry JSONL
        agg = run_alignment_experiment(
            step=args.step,
            max_entries=args.windows,
            round_num=args.round,
        )
    else:
        agg = run_step_experiment(
            step=args.step,
            max_windows=args.windows,
            round_num=args.round,
            use_resources=args.resource,
        )
    print_step_comparison(step=args.step, aggregated=agg)
