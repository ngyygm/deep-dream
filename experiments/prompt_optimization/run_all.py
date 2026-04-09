"""Run all prompt optimization experiments and generate comparison report.

Usage:
    python -m experiments.prompt_optimization.run_all
    python -m experiments.prompt_optimization.run_all --steps 2 3
    python -m experiments.prompt_optimization.run_all --steps 2 3 --windows 10
"""
import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from .config import RESULTS_DIR
from .data_loader import find_common_windows
from .run_step import run_step_experiment, print_step_comparison


# Priority order: most impactful steps first
DEFAULT_STEP_ORDER = [2, 3, 1, 5, 6, 7, 4]

# Key metrics for each step (for report highlighting)
KEY_METRICS = {
    1: ["summary_verbatim_ratio_mean", "thinking_specificity_mean", "response_length_mean"],
    2: ["noise_entity_ratio_mean", "avg_content_length_mean", "parse_success_mean", "entity_count_mean"],
    3: ["mention_pattern_ratio_mean", "avg_content_length_mean", "endpoint_validity_mean", "parse_success_mean"],
    4: ["content_depth_mean", "name_match_rate_mean", "parse_success_mean"],
    5: ["content_expansion_ratio_mean", "novelty_ratio_mean", "enhanced_length_mean"],
    6: ["parse_success_mean"],
    7: ["parse_success_mean", "confidence_mean"],
}

# Lower is better for these metrics
LOWER_IS_BETTER = {
    "summary_verbatim_ratio_mean",
    "noise_entity_ratio_mean",
    "mention_pattern_ratio_mean",
    "generic_pattern_ratio_mean",
}


def pick_winner(step: int, aggregated: Dict[str, Any]) -> str:
    """Pick the best variant for a step based on key metrics."""
    variant_names = [k for k in aggregated.keys() if not k.startswith("_")]
    if len(variant_names) <= 1:
        return variant_names[0] if variant_names else "N/A"

    key_metrics = KEY_METRICS.get(step, ["parse_success_mean"])

    # Score each variant: count how many key metrics it wins
    scores = {v: 0 for v in variant_names}
    for metric in key_metrics:
        lower_better = metric in LOWER_IS_BETTER
        best_val = None
        best_var = None
        for vname in variant_names:
            val = aggregated[vname].get(metric)
            if val is None:
                continue
            if best_val is None or (val < best_val if lower_better else val > best_val):
                best_val = val
                best_var = vname
        if best_var:
            scores[best_var] += 1

    # Tiebreak: prefer non-baseline (B or C) variants with parse_success >= 0.9
    for vname in variant_names:
        if aggregated[vname].get("parse_success_mean", 0) < 0.9:
            scores[vname] -= 100  # Penalize low parse success

    winner = max(scores, key=scores.get)
    return winner


def generate_report(all_results: Dict[int, Dict], windows: List[str]) -> str:
    """Generate markdown comparison report."""
    lines = []
    lines.append("# Prompt Optimization Experiment Report")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Test windows: {len(windows)}")
    lines.append("")

    for step in sorted(all_results.keys()):
        aggregated = all_results[step]
        variant_names = [k for k in aggregated.keys() if not k.startswith("_")]

        lines.append(f"## Step {step}")
        lines.append("")

        # Comparison table
        all_mean_keys = set()
        for vname in variant_names:
            all_mean_keys.update(k for k in aggregated[vname] if k.endswith("_mean") and not k.startswith("_"))
        mean_keys = sorted(all_mean_keys)

        # Header
        header = "| Metric | " + " | ".join(vname for vname in variant_names) + " |"
        sep = "|--------|" + "|".join(["--------" for _ in variant_names]) + "|"
        lines.append(header)
        lines.append(sep)

        for key in mean_keys:
            label = key.replace("_mean", "")
            row = f"| {label} |"
            for vname in variant_names:
                val = aggregated[vname].get(key)
                if val is not None:
                    row += f" {val:.3f} |"
                else:
                    row += " N/A |"
            lines.append(row)

        # Window count
        row = "| windows |"
        for vname in variant_names:
            row += f" {aggregated[vname].get('_window_count', 0)} |"
        lines.append(row)
        lines.append("")

        # Winner
        winner = pick_winner(step, aggregated)
        lines.append(f"**Winner**: `{winner}`")
        lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Step | Winner |")
    lines.append("|------|--------|")
    for step in sorted(all_results.keys()):
        winner = pick_winner(step, all_results[step])
        lines.append(f"| {step} | `{winner}` |")
    lines.append("")

    return "\n".join(lines)


def run_all(
    steps: Optional[List[int]] = None,
    max_windows: int = 20,
    report_only: bool = False,
):
    """Run experiments for all steps (or specified steps)."""
    steps = steps or DEFAULT_STEP_ORDER

    # Find common windows once
    print("Finding common test windows...")
    windows = find_common_windows(total=max_windows)
    if not windows:
        print("ERROR: No common windows found. Check distill_pipeline/ directory.")
        return

    all_results: Dict[int, Dict] = {}

    # Load existing results if report_only
    if report_only:
        for step in steps:
            agg_path = os.path.join(RESULTS_DIR, f"step{step}_aggregated.json")
            if os.path.exists(agg_path):
                with open(agg_path, "r", encoding="utf-8") as f:
                    all_results[step] = json.load(f)
                print(f"  Loaded existing results for step {step}")
    else:
        for step in steps:
            print(f"\n{'#'*60}")
            print(f"# Running Step {step}")
            print(f"{'#'*60}")

            try:
                agg = run_step_experiment(step=step, windows=windows)
                all_results[step] = agg
                print_step_comparison(step, agg)
            except Exception as e:
                print(f"ERROR on step {step}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Delay between steps
            time.sleep(2)

    # Generate report
    if all_results:
        report = generate_report(all_results, windows)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        report_path = os.path.join(RESULTS_DIR, "comparison_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")
        print("\n" + report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all prompt optimization experiments")
    parser.add_argument("--steps", type=int, nargs="+", default=None, help="Steps to run (default: all)")
    parser.add_argument("--windows", type=int, default=20, help="Number of test windows")
    parser.add_argument("--report-only", action="store_true", help="Generate report from existing results")
    args = parser.parse_args()

    run_all(steps=args.steps, max_windows=args.windows, report_only=args.report_only)
