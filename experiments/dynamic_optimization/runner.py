"""Run a single prompt variant on test data and save results.

Usage:
    python -m experiments.dynamic_optimization.runner --step 2
    python -m experiments.dynamic_optimization.runner --step 2 --prompt-file variant.txt
    python -m experiments.dynamic_optimization.runner --step 2 --output results/step2_run1/
    python -m experiments.dynamic_optimization.runner --step 6 --entries 30
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# Ensure parent packages are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.prompt_optimization.data_loader import (
    find_common_windows,
    load_window,
    extract_input_text,
    extract_memory_cache,
    extract_entity_list,
    extract_system_prompt,
    extract_assistant_response,
    parse_json_from_response,
    find_resource_windows,
    load_resource_window,
    build_resource_user_prompt,
    load_all_alignment_entries,
)
from experiments.prompt_optimization.llm_runner import ExperimentLLMRunner
from experiments.prompt_optimization.config import (
    DISTILL_DIR,
    RESOURCES_DIR,
    STEP_DIRS,
    NOVELS,
    NOVEL_WEIGHTS,
)

from .config import (
    RESULTS_DIR,
    FULL_SAMPLE_N,
    ALIGNMENT_ENTRIES_TOTAL,
    LLM_CONFIG,
)
from .prompts_source import get_production_prompt


def _build_user_prompt_from_messages(messages: List[Dict]) -> str:
    """Extract user prompt from distillation messages."""
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _build_step2_user_prompt(messages: List[Dict]) -> str:
    """Build user prompt for step 2 from distillation data."""
    cache = extract_memory_cache(messages)
    input_text = extract_input_text(messages)
    parts = []
    if cache:
        parts.append(f"<记忆缓存>\n{cache}\n</记忆缓存>")
    parts.append(f"<输入文本>\n{input_text}\n</输入文本>")
    parts.append("请从以上文本中抽取概念实体。")
    return "\n\n".join(parts)


def _build_step3_user_prompt(messages: List[Dict]) -> str:
    """Build user prompt for step 3 from distillation data."""
    cache = extract_memory_cache(messages)
    input_text = extract_input_text(messages)
    entity_list = extract_entity_list(messages)
    parts = []
    if cache:
        parts.append(f"<记忆缓存>\n{cache}\n</记忆缓存>")
    parts.append(f"<输入文本>\n{input_text}\n</输入文本>")
    if entity_list:
        parts.append(f"<概念实体列表>\n{entity_list}\n</概念实体列表>")
    parts.append("请从以上文本中的概念实体间抽取概念关系。")
    return "\n\n".join(parts)


def _build_step1_user_prompt(messages: List[Dict]) -> str:
    """Build user prompt for step 1 from distillation data."""
    cache = extract_memory_cache(messages)
    input_text = extract_input_text(messages)
    parts = []
    if cache:
        parts.append(f"<记忆缓存>\n{cache}\n</记忆缓存>")
    else:
        parts.append("<记忆缓存>\n（系统初始化，暂无缓存）\n</记忆缓存>")
    parts.append(f"<输入文本>\n{input_text}\n</输入文本>")
    return "\n\n".join(parts)


def _build_step4_user_prompt(messages: List[Dict]) -> str:
    """Build user prompt for step 4 (supplement entities) from distillation data."""
    # Step 4 user messages contain the requested entity names
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _build_step5_user_prompt(messages: List[Dict]) -> str:
    """Build user prompt for step 5 (entity enhancement) from distillation data."""
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


# Step -> user prompt builder
_USER_PROMPT_BUILDERS = {
    1: _build_step1_user_prompt,
    2: _build_step2_user_prompt,
    3: _build_step3_user_prompt,
    4: _build_step4_user_prompt,
    5: _build_step5_user_prompt,
}


def run_step(
    step: int,
    system_prompt: str,
    output_dir: str,
    max_samples: int = FULL_SAMPLE_N,
    distill_dir: str = DISTILL_DIR,
) -> Dict[str, Any]:
    """Run a prompt variant on test data for steps 1-5.

    Returns metadata dict.
    """
    runner = ExperimentLLMRunner()
    builder = _USER_PROMPT_BUILDERS.get(step, _build_user_prompt_from_messages)

    # Load test windows
    windows = find_common_windows(distill_dir=distill_dir, total=max_samples)
    step_name = STEP_DIRS[step]

    results = []
    errors = 0
    total_prompt_tokens = 0

    for wi, window_key in enumerate(windows):
        window_data = load_window(distill_dir, step_name, window_key)
        if window_data is None:
            print(f"  [{wi+1}/{len(windows)}] {window_key}: NO DATA, skipping")
            continue

        messages = window_data if isinstance(window_data, list) else window_data.get("messages", [])
        user_prompt = builder(messages)
        input_text = extract_input_text(messages) if messages else ""

        if not user_prompt:
            print(f"  [{wi+1}/{len(windows)}] {window_key}: NO USER PROMPT, skipping")
            continue

        prompt_tokens = runner.estimate_tokens(system_prompt + user_prompt)
        total_prompt_tokens += prompt_tokens

        print(f"  [{wi+1}/{len(windows)}] {window_key} (input={len(input_text)} chars)", end=" ")

        try:
            response = runner.call(system_prompt, user_prompt)
            parse_ok = True
        except Exception as e:
            print(f"ERROR: {e}")
            response = ""
            parse_ok = False
            errors += 1

        # Extract ground truth if available
        gt_response = ""
        gt_parsed = None
        if messages:
            gt_response = extract_assistant_response(messages)
            if gt_response:
                try:
                    gt_parsed = parse_json_from_response(gt_response)
                except Exception:
                    gt_parsed = None

        record = {
            "sample_id": window_key,
            "input_text": input_text[:2000],  # Truncate for storage
            "user_prompt": user_prompt,
            "response": response,
            "ground_truth_response": gt_response[:2000] if gt_response else "",
            "ground_truth_parsed": gt_parsed,
            "metadata": {
                "window_idx": wi,
                "prompt_tokens": prompt_tokens,
                "response_length": len(response),
                "parse_ok": parse_ok,
            },
        }
        results.append(record)

        # Extract novel name from window_key
        novel = window_key.split(".")[0] if "." in window_key else "unknown"
        status = "OK" if parse_ok else "ERR"
        print(f"-> {status} | resp={len(response)}")

        time.sleep(0.3)

    return {
        "step": step,
        "total_samples": len(windows),
        "successful": len(results),
        "errors": errors,
        "total_prompt_tokens": total_prompt_tokens,
        "results": results,
    }


def run_alignment_step(
    step: int,
    system_prompt: str,
    output_dir: str,
    max_entries: int = ALIGNMENT_ENTRIES_TOTAL,
    distill_dir: str = DISTILL_DIR,
) -> Dict[str, Any]:
    """Run a prompt variant on alignment data for steps 6-7.

    Returns metadata dict.
    """
    runner = ExperimentLLMRunner()

    # Load alignment entries
    entries = load_all_alignment_entries(
        distill_dir=distill_dir,
        step=step,
        max_entries=max_entries,
    )
    if not entries:
        print("  No entries loaded.")
        return {"step": step, "total_samples": 0, "successful": 0, "errors": 0, "results": []}

    results = []
    errors = 0
    total_prompt_tokens = 0

    for ei, entry in enumerate(entries):
        user_prompt = entry.get("user_prompt", "")
        gt_parsed = entry.get("ground_truth_parsed", {})
        gt_action = entry.get("ground_truth_action", "")

        if not user_prompt:
            print(f"  [{ei+1}/{len(entries)}] EMPTY PROMPT, skipping")
            continue

        prompt_tokens = runner.estimate_tokens(system_prompt + user_prompt)
        total_prompt_tokens += prompt_tokens

        print(f"  [{ei+1}/{len(entries)}] action={gt_action} | prompt={len(user_prompt)} chars", end=" ")

        try:
            response = runner.call(system_prompt, user_prompt)
            parse_ok = True
        except Exception as e:
            print(f"ERROR: {e}")
            response = ""
            parse_ok = False
            errors += 1

        record = {
            "sample_id": f"alignment_{ei}",
            "user_prompt": user_prompt,
            "response": response,
            "ground_truth_parsed": gt_parsed,
            "ground_truth_action": gt_action,
            "metadata": {
                "entry_idx": ei,
                "prompt_tokens": prompt_tokens,
                "response_length": len(response),
                "parse_ok": parse_ok,
            },
        }
        results.append(record)

        status = "OK" if parse_ok else "ERR"
        print(f"-> {status}")

        time.sleep(0.3)

    return {
        "step": step,
        "total_samples": len(entries),
        "successful": len(results),
        "errors": errors,
        "total_prompt_tokens": total_prompt_tokens,
        "results": results,
    }


def run_experiment(
    step: int,
    system_prompt: Optional[str] = None,
    prompt_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> str:
    """Run a single prompt variant experiment. Returns output directory path.

    Args:
        step: Pipeline step (1-7)
        system_prompt: System prompt text (or None to use production)
        prompt_file: Path to file containing system prompt
        output_dir: Output directory (or auto-generated)
        max_samples: Number of test samples
    """
    # Resolve system prompt
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    elif system_prompt is None:
        system_prompt = get_production_prompt(step)

    # Resolve output dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(RESULTS_DIR, f"step{step}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    # Save prompt
    prompt_path = os.path.join(output_dir, "prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(system_prompt)

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "step": step,
            "max_samples": max_samples or FULL_SAMPLE_N,
            "timestamp": datetime.now().isoformat(),
            "prompt_length": len(system_prompt),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Step {step}: {STEP_DIRS[step]}")
    print(f"Prompt length: {len(system_prompt)} chars")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Run
    n = max_samples or FULL_SAMPLE_N
    if step >= 6:
        data = run_alignment_step(step, system_prompt, output_dir, max_entries=n)
    else:
        data = run_step(step, system_prompt, output_dir, max_samples=n)

    # Save responses
    responses_path = os.path.join(output_dir, "responses.json")
    with open(responses_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n  Results: {data['successful']}/{data['total_samples']} successful")
    print(f"  Saved to: {responses_path}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prompt variant experiment")
    parser.add_argument("--step", type=int, required=True, choices=range(1, 8))
    parser.add_argument("--prompt-file", type=str, default=None, help="Path to system prompt file")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--samples", type=int, default=None, help="Number of test samples")
    args = parser.parse_args()

    run_experiment(
        step=args.step,
        prompt_file=args.prompt_file,
        output_dir=args.output,
        max_samples=args.samples,
    )
