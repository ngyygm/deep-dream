"""Load test windows from distillation JSONL files."""
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import DISTILL_DIR, STEP_DIRS, NOVELS, NOVEL_WEIGHTS, TEST_WINDOWS_TOTAL, RANDOM_SEED, RESOURCES_DIR

# Re-export for convenience within this module
_RANDOM_SEED = RANDOM_SEED


def find_common_windows(
    distill_dir: str = DISTILL_DIR,
    novels: List[str] = NOVELS,
    novel_weights: Dict[str, int] = NOVEL_WEIGHTS,
    total: int = TEST_WINDOWS_TOTAL,
    seed: int = RANDOM_SEED,
) -> List[str]:
    """Find window keys present across ALL 7 step directories, stratified by novel."""
    random.seed(seed)
    step_names = sorted(STEP_DIRS.values())

    # Map window_key -> set of step dirs it appears in
    key_to_steps: Dict[str, set] = {}
    for step_name in step_names:
        step_path = os.path.join(distill_dir, step_name)
        if not os.path.isdir(step_path):
            print(f"  Warning: missing step dir {step_path}")
            continue
        for fn in os.listdir(step_path):
            if not fn.endswith(".jsonl"):
                continue
            # fn format: {novel}.txt_{hash}_{timestamp}.jsonl
            # window_key = novel + hash (without timestamp)
            parts = fn.rsplit("_", 1)  # ["红楼梦.txt_0229e258", "1774853451009.jsonl"]
            window_key = parts[0]
            key_to_steps.setdefault(window_key, set()).add(step_name)

    required = set(step_names)
    common_keys = [k for k, v in key_to_steps.items() if v >= required]

    # Group by novel
    novel_windows: Dict[str, List[str]] = {}
    for key in common_keys:
        novel = key.split(".")[0]
        novel_windows.setdefault(novel, []).append(key)

    print(f"  Common windows: {len(common_keys)} total")
    for n in novels:
        print(f"    {n}: {len(novel_windows.get(n, []))}")

    # Stratified sample
    selected = []
    for novel in novels:
        pool = novel_windows.get(novel, [])
        count = min(novel_weights.get(novel, 4), len(pool))
        if count > 0:
            selected.extend(random.sample(pool, count))

    print(f"  Selected {len(selected)} test windows")
    return selected


def load_window(distill_dir: str, step_name: str, window_key: str) -> Optional[Dict]:
    """Load a window's JSONL data for a given step. Returns parsed messages list or None."""
    step_path = os.path.join(distill_dir, step_name)
    if not os.path.isdir(step_path):
        return None
    for fn in os.listdir(step_path):
        if fn.startswith(window_key + "_") and fn.endswith(".jsonl"):
            fp = os.path.join(step_path, fn)
            with open(fp, "r", encoding="utf-8") as f:
                line = f.readline().strip()
                if line:
                    return json.loads(line)
    return None


def extract_tag_content(text: str, tag: str) -> str:
    """Extract content between <tag>...</tag>."""
    start_marker = f"<{tag}>"
    end_marker = f"</{tag}>"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start >= 0 and end >= 0:
        return text[start + len(start_marker):end].strip()
    return ""


def extract_input_text(messages: List[Dict]) -> str:
    """Extract <输入文本> from user messages."""
    for m in messages:
        if m.get("role") == "user":
            content = extract_tag_content(m["content"], "输入文本")
            if content:
                return content
    return ""


def extract_memory_cache(messages: List[Dict]) -> str:
    """Extract <记忆缓存> from user messages."""
    for m in messages:
        if m.get("role") == "user":
            content = extract_tag_content(m["content"], "记忆缓存")
            if content:
                return content
    return ""


def extract_entity_list(messages: List[Dict]) -> str:
    """Extract <概念实体列表> from user messages."""
    for m in messages:
        if m.get("role") == "user":
            content = extract_tag_content(m["content"], "概念实体列表")
            if content:
                return content
    return ""


def extract_system_prompt(messages: List[Dict]) -> str:
    """Get system prompt from messages."""
    for m in messages:
        if m.get("role") == "system":
            return m.get("content", "")
    return ""


def extract_assistant_response(messages: List[Dict]) -> str:
    """Get the first assistant response."""
    for m in messages:
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def parse_json_from_response(text: str) -> Any:
    """Parse JSON from LLM response (handles ```json ... ``` blocks)."""
    # Try markdown code block first
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Try raw JSON
    text = text.strip()
    if text.startswith("[") or text.startswith("{"):
        return json.loads(text)
    return None


def build_entity_catalog_from_step2(step2_response: str) -> List[Dict]:
    """Parse entity list from step 2 response."""
    try:
        entities = parse_json_from_response(step2_response)
        if isinstance(entities, list):
            return entities
    except:
        pass
    return []


# ============================================================
# Resource Library Loading (for diverse input types)
# ============================================================

def find_resource_windows(resources_dir: str = RESOURCES_DIR, total: int = 15) -> List[str]:
    """Scan resource library and return list of resource keys.

    Each .txt file under resources/ becomes a window key.
    Returns format: 'category/filename_without_ext'
    """
    if not os.path.isdir(resources_dir):
        print(f"  Warning: resources dir not found: {resources_dir}")
        return []

    resource_keys = []
    for category in sorted(os.listdir(resources_dir)):
        cat_path = os.path.join(resources_dir, category)
        if not os.path.isdir(cat_path):
            continue
        for fn in sorted(os.listdir(cat_path)):
            if fn.endswith(".txt"):
                key = f"{category}/{fn[:-4]}"  # strip .txt
                resource_keys.append(key)

    if total and len(resource_keys) > total:
        random.seed(RANDOM_SEED)
        resource_keys = random.sample(resource_keys, total)

    print(f"  Resource windows: {len(resource_keys)}")
    for k in resource_keys:
        print(f"    {k}")
    return resource_keys


def load_resource_window(
    resources_dir: str,
    resource_key: str,
) -> Dict[str, str]:
    """Load a resource text file and return metadata dict.

    Returns:
        {"content": str, "category": str, "source": str, "description": str}
    """
    fp = os.path.join(resources_dir, resource_key + ".txt")
    if not os.path.isfile(fp):
        return None

    with open(fp, "r", encoding="utf-8") as f:
        raw = f.read()

    # Parse header comments (# 类型: xxx, # 来源: xxx, # 描述: xxx)
    meta = {"content": "", "category": "", "source": "", "description": ""}
    lines = raw.split("\n")
    content_start = 0
    for i, line in enumerate(lines):
        if line.startswith("# "):
            if ":" in line:
                key, val = line[2:].split(":", 1)
                key = key.strip().lower()
                if key in meta:
                    meta[key] = val.strip()
            content_start = i + 1
        elif line.strip() == "":
            if content_start == i:
                content_start = i + 1
        else:
            break

    meta["content"] = "\n".join(lines[content_start:]).strip()
    meta["key"] = resource_key
    return meta


def build_resource_user_prompt(
    step: int,
    resource: Dict[str, str],
) -> str:
    """Build a synthetic user prompt from resource data for a given step.

    Steps 2/3: <输入文本> + optional <记忆缓存>
    Steps 1: Just the text
    """
    text = resource["content"]
    source = resource.get("source", resource.get("key", "unknown"))

    if step == 1:
        return f"<记忆缓存>\n（系统初始化，暂无缓存）\n</记忆缓存>\n\n<输入文本>\n{text}\n</输入文本>"
    elif step == 2:
        return f"<记忆缓存>\n（正在处理：{source}）\n</记忆缓存>\n\n<输入文本>\n{text}\n</输入文本>\n\n请从以上文本中抽取概念实体。"
    elif step == 3:
        return f"<记忆缓存>\n（正在处理：{source}）\n</记忆缓存>\n\n<输入文本>\n{text}\n</输入文本>\n\n请从以上文本中的概念实体间抽取概念关系。"
    elif step == 5:
        return f"<记忆缓存>\n（正在处理：{source}）\n</记忆缓存>\n\n请增强以下实体内容：\n实体名称：{resource.get('entity_name', '测试实体')}\n原始内容：{resource.get('entity_content', '待增强内容')}\n\n参考文本：\n{text[:500]}"
    else:
        return text


# ============================================================
# Alignment Data Loading (Steps 6/7 — multi-entry JSONL)
# ============================================================

def load_alignment_entries(
    distill_dir: str,
    step_name: str,
    window_key: str,
) -> List[Dict]:
    """Load ALL entries from a Step 6/7 JSONL file.

    Returns list of dicts, each with 'messages' field.
    """
    step_path = os.path.join(distill_dir, step_name)
    if not os.path.isdir(step_path):
        return []
    entries = []
    for fn in os.listdir(step_path):
        if fn.startswith(window_key + "_") and fn.endswith(".jsonl"):
            fp = os.path.join(step_path, fn)
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if isinstance(data, dict) and "messages" in data:
                        entries.append(data)
            break  # only one file per window_key
    return entries


def filter_batch_entries(
    entries: List[Dict],
    step: int,
) -> List[Dict]:
    """Filter entries to keep only batch-resolution / batch-relation type.

    Step 6: keep entries with '批量裁决' in system prompt first line.
    Step 7: keep entries with '一批新关系' in system prompt first line.
    """
    filtered = []
    for entry in entries:
        msgs = entry.get("messages", [])
        if not msgs:
            continue
        sp = msgs[0].get("content", "")
        first_line = sp.split("\n")[0]
        if step == 6 and "批量裁决" in first_line:
            filtered.append(entry)
        elif step == 7 and "一批新关系" in first_line:
            filtered.append(entry)
    return filtered


def _classify_s6_action(parsed: Dict) -> str:
    """Classify Step 6 batch action: match, relation, or create_new."""
    mid = str(parsed.get("match_existing_id", "")).strip()
    rtc = parsed.get("relations_to_create", [])
    has_match = mid != "" and mid != "None"
    has_relation = bool(rtc) and len(rtc) > 0
    if has_match and has_relation:
        return "match_relation"
    elif has_match:
        return "match"
    elif has_relation:
        return "relation"
    else:
        return "create_new"


def _classify_s7_action(parsed: Dict) -> str:
    """Classify Step 7 batch action: match_existing or create_new."""
    action = str(parsed.get("action", "")).strip().lower()
    if "match" in action:
        return "match_existing"
    else:
        return "create_new"


def sample_stratified(
    entries: List[Dict],
    step: int,
    total: int = 60,
    seed: int = RANDOM_SEED,
) -> List[Dict]:
    """Stratified sampling for balanced test set.

    Step 6: 30 match + 15 create_new + 15 relation/match_relation
    Step 7: 30 match_existing + 30 create_new

    Each returned dict has extra keys:
      - 'user_prompt': str
      - 'ground_truth_response': str
      - 'ground_truth_parsed': dict
      - 'ground_truth_action': str
    """
    random.seed(seed)
    by_category: Dict[str, List[Dict]] = {}

    for entry in entries:
        msgs = entry.get("messages", [])
        if len(msgs) < 3:
            continue
        resp_text = ""
        for m in msgs:
            if m.get("role") == "assistant":
                resp_text = m["content"]
                break
        if not resp_text:
            continue
        try:
            parsed = parse_json_from_response(resp_text)
        except (json.JSONDecodeError, ValueError):
            continue
        if parsed is None:
            continue

        user_prompt = ""
        for m in msgs:
            if m.get("role") == "user":
                user_prompt = m["content"]
                break

        if step == 6:
            action = _classify_s6_action(parsed)
        else:
            action = _classify_s7_action(parsed)

        record = {
            "messages": msgs,
            "user_prompt": user_prompt,
            "ground_truth_response": resp_text,
            "ground_truth_parsed": parsed,
            "ground_truth_action": action,
        }
        by_category.setdefault(action, []).append(record)

    # Sample from each category (proportional quotas)
    selected = []
    if step == 6:
        # Half match, quarter create_new, quarter split between relation/match_relation
        match_quota = max(1, total // 2)
        create_quota = max(1, total // 4)
        other_quota = total - match_quota - create_quota
        quotas = {"match": match_quota, "create_new": create_quota}
        relation_pool = by_category.get("relation", [])
        match_relation_pool = by_category.get("match_relation", [])
        if match_relation_pool and not relation_pool:
            quotas["match_relation"] = other_quota
        elif relation_pool and not match_relation_pool:
            quotas["relation"] = other_quota
        else:
            quotas["relation"] = max(0, other_quota // 2)
            quotas["match_relation"] = max(0, other_quota - other_quota // 2)
    else:
        quotas = {"match_existing": total // 2, "create_new": total - total // 2}

    print(f"  Stratified sampling (step={step}, total={total}):")
    for cat, quota in quotas.items():
        pool = by_category.get(cat, [])
        n = min(quota, len(pool))
        sampled = random.sample(pool, n) if n > 0 else []
        selected.extend(sampled)
        print(f"    {cat}: {len(pool)} available, sampled {n}")

    random.shuffle(selected)
    return selected


def load_all_alignment_entries(
    distill_dir: str,
    step: int,
    max_entries: int = 60,
    seed: int = RANDOM_SEED,
) -> List[Dict]:
    """Load, filter, and sample alignment entries for Steps 6/7.

    High-level function that scans all files, filters for batch type,
    and returns stratified sample.
    """
    step_name = STEP_DIRS[step]
    step_path = os.path.join(distill_dir, step_name)
    if not os.path.isdir(step_path):
        print(f"  Warning: step dir not found: {step_path}")
        return []

    # Collect all batch entries
    all_entries = []
    file_count = 0
    for fn in os.listdir(step_path):
        if not fn.endswith(".jsonl"):
            continue
        file_count += 1
        fp = os.path.join(step_path, fn)
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if isinstance(data, dict) and "messages" in data:
                    all_entries.append(data)

    print(f"  Loaded {len(all_entries)} entries from {file_count} files ({step_name})")

    # Filter for batch type
    batch_entries = filter_batch_entries(all_entries, step)
    print(f"  After filtering: {len(batch_entries)} batch entries")

    # Stratified sampling
    sampled = sample_stratified(batch_entries, step, total=max_entries, seed=seed)
    print(f"  Sampled {len(sampled)} entries for experiment")
    return sampled
