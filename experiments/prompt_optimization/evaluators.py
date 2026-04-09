"""Evaluation metrics for each pipeline step."""
import json
import math
import re
import sys
import os
from typing import Any, Dict, List

from .config import COMMON_VERBS, MENTION_PATTERNS, GENERIC_RELATION_PATTERNS

# Import section schema for alignment check
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from processor.content_schema import parse_markdown_sections, ENTITY_SECTIONS


def _parse_json_response(text: str):
    """Try to parse JSON from response text."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())
    text = text.strip()
    if text.startswith("[") or text.startswith("{"):
        return json.loads(text)
    return None


def _jaccard_chars(text_a: str, text_b: str) -> float:
    """Character-level Jaccard similarity."""
    set_a = set(text_a)
    set_b = set(text_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


# ---- Step 1: Memory Cache ----

def evaluate_step1(response: str, input_text: str, prompt_tokens: int, **kw) -> Dict[str, Any]:
    """Evaluate memory cache update quality."""
    m = {"parse_success": 1.0, "prompt_tokens": prompt_tokens}
    m["summary_length"] = len(response)

    # Verbatim ratio
    m["summary_verbatim_ratio"] = _jaccard_chars(response[:500], input_text[:500])

    # Thinking specificity: does 自我思考 or 预判 mention specific entity names?
    thinking = ""
    for header in ["## 自我思考", "## 预判"]:
        idx = response.find(header)
        if idx >= 0:
            rest = response[idx + len(header):]
            end = rest.find("\n## ")
            thinking = rest[:end] if end >= 0 else rest
            break

    # Extract 2-4 char Chinese names from input
    input_names = set(re.findall(r"[\u4e00-\u9fff]{2,4}", input_text[:200]))
    thinking_names = set(re.findall(r"[\u4e00-\u9fff]{2,4}", thinking))
    overlap = input_names & thinking_names
    m["thinking_specificity"] = len(overlap) / max(1, len(thinking_names))

    # Anchor coverage
    anchors = re.findall(r"第[一二三四五六七八九十百千零\d]+[回章节]", input_text)
    if anchors:
        covered = sum(1 for a in anchors if a in response)
        m["anchor_coverage"] = covered / len(anchors)
    else:
        m["anchor_coverage"] = 1.0  # No anchors to check

    return m


# ---- Step 2: Entity Extraction ----

def _is_noise_entity(name: str, content: str) -> bool:
    """Detect noise entities."""
    name = name.strip()
    content = content.strip()

    # Very short content
    if len(content) < 5:
        return True
    # Single category label
    if content.rstrip("，。、") in {"角色", "地点", "事件", "概念", "物品", "人物", "群体", "动作", "动词", "症状"}:
        return True
    # Single common verb as name
    if len(name) <= 2 and name in COMMON_VERBS:
        return True
    # Action phrase as name
    if re.match(r"^[\u4e00-\u9fff了着过]{2,6}$", name) and name.endswith(("了", "过", "着")):
        return True
    return False


def evaluate_step2(response: str, input_text: str, prompt_tokens: int, **kw) -> Dict[str, Any]:
    """Evaluate entity extraction quality."""
    m = {"prompt_tokens": prompt_tokens}

    try:
        entities = _parse_json_response(response)
        if entities is None:
            m["parse_success"] = 0.0
            m["entity_count"] = 0
            m["noise_entity_ratio"] = 1.0
            m["avg_content_length"] = 0.0
            m["name_type_coverage"] = 0.0
            return m
    except json.JSONDecodeError:
        m["parse_success"] = 0.0
        m["entity_count"] = 0
        m["noise_entity_ratio"] = 1.0
        m["avg_content_length"] = 0.0
        m["name_type_coverage"] = 0.0
        return m

    if not isinstance(entities, list):
        m["parse_success"] = 0.0
        m["entity_count"] = 0
        return m

    m["parse_success"] = 1.0
    m["entity_count"] = len(entities)

    noise = 0
    total_content = 0
    with_parens = 0
    entity_types = set()

    for e in entities:
        if not isinstance(e, dict):
            noise += 1
            continue
        name = e.get("name", "")
        content = e.get("content", "")
        total_content += len(content)

        if _is_noise_entity(name, content):
            noise += 1
        if "（" in name or "(" in name:
            with_parens += 1
            # Track concept types from parenthetical annotations
            m_match = re.search(r'[（(]([^）)]+)[）)]', name)
            if m_match:
                entity_types.add(m_match.group(1))

    m["noise_entity_ratio"] = noise / max(1, len(entities))
    m["avg_content_length"] = total_content / max(1, len(entities))
    m["name_type_coverage"] = with_parens / max(1, len(entities))
    m["concept_diversity"] = len(entity_types)

    return m


# ---- Step 3: Relation Extraction ----

def evaluate_step3(response: str, input_text: str, prompt_tokens: int,
                   entity_names: set = None, **kw) -> Dict[str, Any]:
    """Evaluate relation extraction quality."""
    m = {"prompt_tokens": prompt_tokens}

    try:
        relations = _parse_json_response(response)
        if relations is None:
            m["parse_success"] = 0.0
            m["relation_count"] = 0
            return m
    except json.JSONDecodeError:
        m["parse_success"] = 0.0
        m["relation_count"] = 0
        return m

    if not isinstance(relations, list):
        m["parse_success"] = 0.0
        m["relation_count"] = 0
        return m

    m["parse_success"] = 1.0
    m["relation_count"] = len(relations)

    mention_count = 0
    generic_count = 0
    total_content_len = 0
    valid_endpoints = 0

    for r in relations:
        if not isinstance(r, dict):
            continue
        e1 = r.get("entity1_name", "").strip()
        e2 = r.get("entity2_name", "").strip()
        content = r.get("content", "").strip()
        total_content_len += len(content)

        # "X提到Y" pattern
        if any(p in content for p in MENTION_PATTERNS) and len(content) < 20:
            mention_count += 1
        # Generic
        if any(p in content for p in GENERIC_RELATION_PATTERNS):
            generic_count += 1
        # Endpoint validity
        if entity_names and e1 in entity_names and e2 in entity_names:
            valid_endpoints += 1

    m["mention_pattern_ratio"] = mention_count / max(1, len(relations))
    m["generic_pattern_ratio"] = generic_count / max(1, len(relations))
    m["avg_content_length"] = total_content_len / max(1, len(relations))
    if entity_names:
        m["endpoint_validity"] = valid_endpoints / max(1, len(relations))
    else:
        m["endpoint_validity"] = 1.0  # Can't check without entity list

    return m


# ---- Step 4: Supplement Entities ----

def evaluate_step4(response: str, input_text: str, prompt_tokens: int,
                   requested_names: set = None, **kw) -> Dict[str, Any]:
    """Evaluate entity supplement quality."""
    m = {"prompt_tokens": prompt_tokens}

    try:
        entities = _parse_json_response(response)
        if entities is None:
            m["parse_success"] = 0.0
            return m
    except json.JSONDecodeError:
        m["parse_success"] = 0.0
        return m

    m["parse_success"] = 1.0

    total_content = 0
    name_matches = 0
    for e in entities:
        if not isinstance(e, dict):
            continue
        name = e.get("name", "").strip()
        content = e.get("content", "").strip()
        total_content += len(content)
        if requested_names and name in requested_names:
            name_matches += 1

    m["content_depth"] = total_content / max(1, len(entities))
    m["name_match_rate"] = name_matches / max(1, len(entities)) if requested_names else 1.0

    return m


# ---- Step 5: Entity Enhancement ----

def evaluate_step5(response: str, input_text: str, prompt_tokens: int,
                   original_content: str = "", **kw) -> Dict[str, Any]:
    """Evaluate entity enhancement quality."""
    m = {"prompt_tokens": prompt_tokens}

    try:
        result = _parse_json_response(response)
        if result is None or not isinstance(result, dict):
            m["parse_success"] = 0.0
            return m
    except json.JSONDecodeError:
        m["parse_success"] = 0.0
        return m

    m["parse_success"] = 1.0
    enhanced = result.get("content", "")

    if original_content:
        m["content_expansion_ratio"] = len(enhanced) / max(1, len(original_content))
        # Novelty: fraction of enhanced content NOT in original
        orig_chars = set(original_content)
        enhanced_chars = set(enhanced)
        if enhanced_chars:
            m["novelty_ratio"] = len(enhanced_chars - orig_chars) / len(enhanced_chars)
        else:
            m["novelty_ratio"] = 0.0
    else:
        m["content_expansion_ratio"] = 0.0
        m["novelty_ratio"] = 0.0

    m["enhanced_length"] = len(enhanced)

    # Section alignment: check if content sections match ENTITY_SECTIONS
    sections = parse_markdown_sections(enhanced)
    if sections:
        matched = sum(1 for s in ENTITY_SECTIONS if s in sections)
        m["section_alignment"] = matched / len(ENTITY_SECTIONS)
    else:
        m["section_alignment"] = 0.0

    return m


# ---- Step 6: Entity Alignment ----

def _classify_s6_action(parsed: dict) -> str:
    """Classify Step 6 action: match, match_relation, relation, or create_new."""
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


def evaluate_step6(response: str, input_text: str, prompt_tokens: int, **kw) -> Dict[str, Any]:
    """Evaluate entity alignment quality (format-only, for backward compat)."""
    m = {"prompt_tokens": prompt_tokens}

    try:
        result = _parse_json_response(response)
        if result is None or not isinstance(result, dict):
            m["parse_success"] = 0.0
            return m
    except json.JSONDecodeError:
        m["parse_success"] = 0.0
        return m

    m["parse_success"] = 1.0

    # Check for expected fields
    if "possible_merges" in result and "possible_relations" in result:
        # Preliminary screening
        m["result_type"] = "preliminary"
        total = len(result.get("possible_merges", [])) + len(result.get("possible_relations", [])) + len(result.get("no_action", []))
        m["candidates_classified"] = total
    elif "action" in result:
        # Detailed judgment
        m["result_type"] = "detailed"
        m["action"] = result.get("action", "unknown")
    elif "match_existing_id" in result:
        # Batch resolution
        m["result_type"] = "batch"
        m["update_mode"] = result.get("update_mode", "unknown")
        m["confidence"] = result.get("confidence", 0.0)

    return m


def evaluate_step6_batch(
    response: str,
    ground_truth: dict = None,
    prompt_tokens: int = 0,
    **kw,
) -> Dict[str, Any]:
    """Evaluate Step 6 batch resolution with correctness metrics."""
    m = {"prompt_tokens": prompt_tokens}

    try:
        result = _parse_json_response(response)
        if result is None or not isinstance(result, dict):
            m["parse_success"] = 0.0
            return m
    except json.JSONDecodeError:
        m["parse_success"] = 0.0
        return m

    m["parse_success"] = 1.0
    m["confidence"] = float(result.get("confidence", 0.0))

    # Always extract action classification
    m["predicted_action"] = _classify_s6_action(result)
    mid = str(result.get("match_existing_id", "")).strip()
    m["has_match"] = 1.0 if (mid and mid != "None") else 0.0
    rtc = result.get("relations_to_create", [])
    m["has_relation"] = 1.0 if (bool(rtc) and len(rtc) > 0) else 0.0

    if ground_truth:
        gt_action = _classify_s6_action(ground_truth)
        m["ground_truth_action"] = gt_action
        m["decision_match"] = 1.0 if m["predicted_action"] == gt_action else 0.0

        # For match cases, check if ID matches
        gt_mid = str(ground_truth.get("match_existing_id", "")).strip()
        if gt_mid and gt_mid != "None":
            m["match_id_accuracy"] = 1.0 if mid == gt_mid else 0.0

    return m


# ---- Step 7: Relation Alignment ----

def evaluate_step7(response: str, input_text: str, prompt_tokens: int, **kw) -> Dict[str, Any]:
    """Evaluate relation alignment quality (format-only, for backward compat)."""
    m = {"prompt_tokens": prompt_tokens}

    try:
        result = _parse_json_response(response)
        if result is None or not isinstance(result, dict):
            m["parse_success"] = 0.0
            return m
    except json.JSONDecodeError:
        m["parse_success"] = 0.0
        return m

    m["parse_success"] = 1.0
    m["action"] = result.get("action", "unknown")
    m["confidence"] = result.get("confidence", 0.0)

    return m


def evaluate_step7_batch(
    response: str,
    ground_truth: dict = None,
    prompt_tokens: int = 0,
    **kw,
) -> Dict[str, Any]:
    """Evaluate Step 7 batch relation matching with correctness metrics."""
    m = {"prompt_tokens": prompt_tokens}

    try:
        result = _parse_json_response(response)
        if result is None or not isinstance(result, dict):
            m["parse_success"] = 0.0
            return m
    except json.JSONDecodeError:
        m["parse_success"] = 0.0
        return m

    m["parse_success"] = 1.0
    m["confidence"] = float(result.get("confidence", 0.0))

    action = str(result.get("action", "")).strip().lower()
    if "match" in action:
        m["predicted_action"] = "match_existing"
    elif "create" in action:
        m["predicted_action"] = "create_new"
    else:
        m["predicted_action"] = "unknown"

    if ground_truth:
        gt_action_raw = str(ground_truth.get("action", "")).strip().lower()
        if "match" in gt_action_raw:
            gt_action = "match_existing"
        elif "create" in gt_action_raw:
            gt_action = "create_new"
        else:
            gt_action = "unknown"

        m["ground_truth_action"] = gt_action
        m["action_accuracy"] = 1.0 if m["predicted_action"] == gt_action else 0.0

        # Precision/recall for match_existing
        if m["predicted_action"] == "match_existing":
            m["match_correct"] = 1.0 if gt_action == "match_existing" else 0.0
        if gt_action == "match_existing":
            m["match_recalled"] = 1.0 if m["predicted_action"] == "match_existing" else 0.0

    return m


# ---- Aggregation ----

def aggregate_results(results: List[Dict]) -> Dict[str, Any]:
    """Compute mean/std for all numeric metrics."""
    if not results:
        return {}

    numeric_keys = set()
    for r in results:
        for k, v in r.items():
            if isinstance(v, (int, float)) and k not in ("window_idx",):
                numeric_keys.add(k)

    agg = {}
    for key in sorted(numeric_keys):
        values = [r[key] for r in results if key in r and isinstance(r[key], (int, float))]
        if not values:
            continue
        mean = sum(values) / len(values)
        agg[f"{key}_mean"] = mean
        if len(values) > 1:
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            agg[f"{key}_std"] = math.sqrt(variance)
        agg[f"{key}_n"] = len(values)

    return agg
