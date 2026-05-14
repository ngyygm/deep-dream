#!/usr/bin/env python3
"""
Response formatting helpers for Deep Dream MCP Server.

Functions for truncating, compacting, and formatting API responses to save
agent context tokens while preserving actionable information.
"""

import json


# ── Constants ─────────────────────────────────────────────────────────────

_MAX_RESPONSE_CHARS = 80000  # ~20k tokens safety cap

_TRIM_FIELDS = {"embedding", "embeddings", "content_hash", "raw_content", "vector"}

# Fields to truncate content to save agent tokens (content can be multi-KB markdown)
_CONTENT_TRUNCATE_LEN = 200
_SENTENCE_SEPARATORS = ('. ', '。', '\n', '！', '？', '; ', '；')

_NOISE_KEYS = {"success", "elapsed_ms", "timestamp"}

_MAX_ERROR_CHARS = 2000  # Errors should be concise — agents need hints, not stack traces


# ── Truncation helpers ────────────────────────────────────────────────────

def _truncate_text(text, max_len=_CONTENT_TRUNCATE_LEN):
    """Truncate text at sentence boundary when possible."""
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    # Try breaking at last sentence boundary in second half of snippet
    for sep in _SENTENCE_SEPARATORS:
        idx = cut.rfind(sep)
        if idx > max_len // 2:
            return text[:idx + len(sep)].rstrip() + "..."
    # No good break point — truncate at last space if possible
    last_space = cut.rfind(' ')
    if last_space > max_len // 2:
        return text[:last_space] + "..."
    return cut + "..."


# ── Compact helpers ───────────────────────────────────────────────────────

def _compact_entity(item):
    """Compact an entity dict: truncate content at sentence boundaries, keep essential fields including absolute_id for create_relation."""
    if not isinstance(item, dict):
        return item
    out = {}
    for k in ("family_id", "name", "summary", "absolute_id", "confidence"):
        if k in item:
            out[k] = item[k]
    if "_score" in item:
        out["_score"] = item["_score"]
    # Truncate content at sentence boundary
    for ck in ("content", "markdown_content"):
        if ck in item and isinstance(item[ck], str):
            out[ck] = _truncate_text(item[ck])
    if "event_time" in item:
        out["event_time"] = item["event_time"]
    if "version_count" in item:
        out["version_count"] = item["version_count"]
    # Keep small attributes; for large ones, keep key list as hint
    if "attributes" in item and isinstance(item["attributes"], dict) and item["attributes"]:
        attrs = item["attributes"]
        attr_len = sum(len(str(v)) for v in attrs.values())
        if attr_len <= 200:
            out["attributes"] = attrs
        else:
            out["_attr_keys"] = list(attrs.keys())[:8]
    return out


def _compact_relation(item):
    """Compact a relation dict: truncate content at sentence boundary, keep endpoints + confidence."""
    if not isinstance(item, dict):
        return item
    out = {}
    for k in ("family_id", "entity1_id", "entity2_id", "entity1_name", "entity2_name", "relation_type", "event_time", "confidence"):
        if k in item:
            out[k] = item[k]
    if "_score" in item:
        out["_score"] = item["_score"]
    if "content" in item and isinstance(item["content"], str):
        out["content"] = _truncate_text(item["content"])
    return out


def _compact_version(item):
    """Compact a version dict: truncate content at sentence boundary."""
    if not isinstance(item, dict):
        return item
    out = {}
    for k in ("absolute_id", "name", "event_time", "processed_time"):
        if k in item:
            out[k] = item[k]
    for ck in ("content", "markdown_content", "summary"):
        if ck in item and isinstance(item[ck], str):
            out[ck] = _truncate_text(item[ck])
    return out


def _compact_list(data, compact_fn, list_key=None):
    """Apply compact_fn to a list within a response dict.
    Works with patterns like {data: {entities: [...]}} or just {entities: [...]}."""
    if not isinstance(data, dict):
        return data

    # Check nested data.data pattern
    inner = data.get("data", data)
    keys_to_try = [list_key] if list_key else ["entities", "relations", "versions", "episodes", "items"]

    for key in keys_to_try:
        if key in inner and isinstance(inner[key], list):
            inner[key] = [compact_fn(item) for item in inner[key]]
            break

    return data


# ── Response trimming ─────────────────────────────────────────────────────

def _trim_response(data, max_chars=_MAX_RESPONSE_CHARS):
    """Remove bulky fields and unwrap boilerplate to save agent tokens."""
    if not isinstance(data, dict):
        text = json.dumps(data, ensure_ascii=False)
        if len(text) > max_chars:
            return text[:max_chars] + "\n... [truncated]"
        return text

    # Unwrap {success: true, data: {...}, elapsed_ms: ...} → just {...}
    inner = data.get("data")
    if isinstance(inner, dict) and "success" in data:
        # Keep only the inner data, but merge back any non-boilerplate top-level keys
        data = inner

    # Strip expensive fields from nested structures
    data = _strip_bulky(data)

    # Fast path: skip serialization if any known large list exceeds 3 items
    _needs_trim = False
    for key in ("entities", "relations", "versions", "episodes"):
        items = data.get(key)
        if isinstance(items, list) and len(items) > 3:
            _needs_trim = True
            break

    if not _needs_trim:
        text = json.dumps(data, ensure_ascii=False)
        if len(text) <= max_chars:
            return text
        # Fall through to trimming — reuse the serialization we already have
        _full_text = text
    else:
        _full_text = json.dumps(data, ensure_ascii=False)

    # Try progressively trimming large lists
    # Pre-compute per-key sizes for O(1) base_overhead
    _key_texts = {}
    for key in ("entities", "relations", "versions", "episodes"):
        items = data.get(key)
        if isinstance(items, list):
            _key_texts[key] = json.dumps(items, ensure_ascii=False)
    # _base_overhead for key K = full_text minus key K's serialized form (approximate)
    # This avoids re-serializing the entire dict per key.

    for key in ("entities", "relations", "versions", "episodes"):
        items = data.get(key)
        if not isinstance(items, list) or len(items) <= 3:
            continue
        # Pre-compute per-item serialized sizes
        _item_sizes = [len(json.dumps(item, ensure_ascii=False)) for item in items]
        # Estimate base overhead = total size minus the current key's value size + margin
        _key_text_len = len(_key_texts.get(key, ""))
        _base_overhead = len(_full_text) - _key_text_len
        # Estimate overhead for list brackets, commas, and metadata keys
        _est_overhead = _base_overhead + 30 + 5 * len(items)  # brackets + commas + metadata
        lo, hi = 1, min(len(items), 20)
        best_mid = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            est_size = _est_overhead + sum(_item_sizes[:mid])
            if est_size <= max_chars:
                best_mid = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best_mid > 0:
            kept = items[:best_mid]
            candidate = {**data, key: kept, f"{key}_total": len(items), f"{key}_shown": best_mid}
            candidate_text = json.dumps(candidate, ensure_ascii=False)
            if len(candidate_text) <= max_chars:
                omitted = len(items) - best_mid
                text = candidate_text
                if omitted > 0:
                    text += f"\n→ {omitted} more {key} omitted. Use offset/limit to fetch more."
                return text

    # Last resort: hard truncate
    text = _full_text
    return text[:max_chars] + "\n... [response truncated to fit context]"


def _strip_bulky(obj):
    """Recursively remove embedding/vector/hash fields from response objects."""
    if isinstance(obj, dict):
        # Fast path: if no bulky keys present, skip dict reconstruction
        if not _TRIM_FIELDS.intersection(obj):
            # Still recurse into values that may be dicts/lists
            _changed = False
            _result = {}
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    _v = _strip_bulky(v)
                    if _v is not v:
                        _changed = True
                        _result[k] = _v
                        continue
                _result[k] = v
            return _result if _changed else obj
        return {k: _strip_bulky(v) for k, v in obj.items() if k not in _TRIM_FIELDS}
    if isinstance(obj, list):
        return [_strip_bulky(item) for item in obj]
    return obj


# ── Hint helpers ──────────────────────────────────────────────────────────

def _error_hint(data):
    """Generate actionable hints from API error responses."""
    if not isinstance(data, dict):
        return ""
    msg = ""
    if isinstance(data.get("data"), dict):
        msg = data["data"].get("error", data["data"].get("message", ""))
    elif isinstance(data.get("error"), str):
        msg = data["error"]
    elif "detail" in data:
        msg = data["detail"]

    hints = []
    lower = msg.lower()
    if "not found" in lower and "entity" in lower:
        hints.append("Hint: use search_entities or find_entity_by_name to find the correct family_id.")
    if "not found" in lower and "relation" in lower:
        hints.append("Hint: use search_relations or get_entity_relations to find the correct relation.")
    if "not found" in lower and "episode" in lower:
        hints.append("Hint: use search_episodes or get_latest_episode to find valid cache_ids.")
    if "not found" in lower and "community" in lower:
        hints.append("Hint: use list_communities to see valid community IDs. Run detect_communities first if empty.")
    if "not found" in lower and "task" in lower:
        hints.append("Hint: use remember_tasks to list all tasks with valid IDs.")
    if "neo4j" in lower and "not available" in lower:
        hints.append("Hint: this feature requires Neo4j backend.")
    if "context budget" in lower or "token" in lower:
        hints.append("Hint: reduce max_entities/max_relations or shorten your query.")
    if "already exists" in lower:
        hints.append("Hint: use find_entity_by_name to check if the entity already exists.")
    if "timeout" in lower or "timed out" in lower:
        hints.append("Hint: the operation took too long. Try with smaller input or fewer items.")
    if "rate limit" in lower or "429" in lower:
        hints.append("Hint: too many requests. Wait a moment and retry.")
    if "invalid" in lower and ("id" in lower or "identifier" in lower):
        hints.append("Hint: check that the ID format is correct. family_ids start with 'ent_' or 'rel_', absolute_ids are UUIDs.")
    if "merge" in lower and ("same" in lower or "cannot" in lower or "error" in lower):
        hints.append("Hint: merge_entities requires at least 2 different entity family_ids.")
    if "cannot" in lower and ("delete" in lower or "remove" in lower):
        hints.append("Hint: the resource may be in use. Check system_tasks for running operations.")
    if "permission" in lower or "forbidden" in lower or "unauthorized" in lower:
        hints.append("Hint: check your API key configuration and graph_id access permissions.")
    if "validation" in lower or ("required" in lower and "field" in lower):
        hints.append("Hint: check the tool's required parameters. Missing or empty fields cause validation errors.")
    if "conflict" in lower or "409" in str(data.get("status_code", "")):
        hints.append("Hint: resource state conflict. The data may have changed since last read — refresh with get_entity or entity_profile.")

    return " ".join(hints) if hints else ""


def _empty_search_hint(data, query_param="query"):
    """Append hint when search returns zero results."""
    if not isinstance(data, dict):
        return data
    inner = _inner(data)
    entities = inner.get("entities", [])
    relations = inner.get("relations", [])
    if not entities and not relations:
        hint = f"\n→ No results found. Try: lower similarity_threshold, use search_mode='hybrid', or rephrase the {query_param}."
        _hint(data, hint)
    return data


def _pagination_hint(data, list_key, limit, offset=0):
    """Append hint when a list result equals the limit (suggesting more results exist).
    Only adds a hint if the result count matches the limit — a strong signal of truncation."""
    if not isinstance(data, dict) or not limit:
        return data
    inner = _inner(data)
    items = inner.get(list_key, [])
    if isinstance(items, list) and len(items) >= limit:
        next_offset = offset + limit
        hint = f"\n→ Result count matches limit ({limit}) — more results likely exist. Use offset={next_offset} to fetch the next page."
        _hint(data, hint)
    return data


def _hint(data, text):
    """Set a workflow hint on a response dict. Handles both wrapped and unwrapped responses."""
    if isinstance(data, dict):
        if isinstance(data.get("data"), dict):
            data["data"]["_hint"] = text
        else:
            data["_hint"] = text


def _inner(data):
    """Unwrap {data: {...}} boilerplate to get the inner dict."""
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        return data["data"]
    return data


# ── Error formatting ──────────────────────────────────────────────────────

def _compact_error(data):
    """Extract the essential error message from an API error response, discarding bulky fields."""
    if not isinstance(data, dict):
        return str(data)[:500]
    inner = data.get("data", data)
    # Extract core error fields
    msg = ""
    for key in ("error", "message", "detail", "error_message"):
        if key in inner and isinstance(inner[key], str):
            msg = inner[key]
            break
    if not msg and isinstance(inner.get("error"), dict):
        msg = inner["error"].get("message", "")
    if not msg:
        msg = str(data)[:500]
    # Truncate error message
    if len(msg) > 500:
        msg = msg[:500] + "..."
    # Include entity/relation ID context if present
    context_parts = []
    for k in ("family_id", "entity_id", "relation_id", "task_id", "cache_id"):
        if k in inner and isinstance(inner[k], str):
            context_parts.append(f"{k}={inner[k]}")
    result = {"error": msg}
    if context_parts:
        result["context"] = ", ".join(context_parts)
    return json.dumps(result, ensure_ascii=False)


# ── Main result formatter ─────────────────────────────────────────────────

def _result(data, code):
    # Extract workflow hints before trimming
    workflow_hint = ""
    if isinstance(data, dict):
        inner = _inner(data)
        workflow_hint = inner.pop("_hint", "") or data.pop("_hint", "")

    # For errors: use compact error format to save agent tokens
    if code >= 400:
        text = _compact_error(data)
        if workflow_hint:
            text = text.rstrip() + workflow_hint
        hint = _error_hint(data)
        if hint:
            text = text.rstrip() + "\n" + hint
        return {"content": [{"type": "text", "text": text}], "isError": True}

    text = _trim_response(data)
    if workflow_hint:
        text = text.rstrip() + workflow_hint
    return {"content": [{"type": "text", "text": text}]}
