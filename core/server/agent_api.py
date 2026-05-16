"""Agent-friendly REST API enhancements.

Migrated from the former MCP layer. Provides:
- Response compaction (?compact=true): strips embeddings, truncates content, trims large lists
- Error hints: auto-detects error patterns and appends actionable guidance
- ID validation: catches family_id vs absolute_id confusion
- Empty search hints: suggests alternatives when no results found
"""

import json
import re

# ── Constants ──────────────────────────────────────────────────────────────

_CONTENT_TRUNCATE_LEN = 200
_MAX_RESPONSE_CHARS = 80000
_TRIM_FIELDS = frozenset({"embedding", "embeddings", "content_hash", "raw_content", "vector"})
_SENTENCE_SEPARATORS = (".", "。", "\n", "！", "？", ";", "；")


# ── Text truncation ───────────────────────────────────────────────────────

def truncate_text(text, max_len=_CONTENT_TRUNCATE_LEN):
    if not isinstance(text, str) or len(text) <= max_len:
        return text
    cut = text[:max_len]
    for sep in _SENTENCE_SEPARATORS:
        idx = cut.rfind(sep)
        if idx > max_len // 2:
            return text[:idx + len(sep)].rstrip() + "..."
    last_space = cut.rfind(' ')
    if last_space > max_len // 2:
        return text[:last_space] + "..."
    return cut + "..."


# ── Compact helpers ───────────────────────────────────────────────────────

def compact_entity(item):
    if not isinstance(item, dict):
        return item
    out = {}
    for k in ("family_id", "name", "summary", "absolute_id", "confidence"):
        if k in item:
            out[k] = item[k]
    if "_score" in item:
        out["_score"] = item["_score"]
    for ck in ("content", "markdown_content"):
        if ck in item and isinstance(item[ck], str):
            out[ck] = truncate_text(item[ck])
    if "event_time" in item:
        out["event_time"] = item["event_time"]
    if "version_count" in item:
        out["version_count"] = item["version_count"]
    if "attributes" in item and isinstance(item["attributes"], dict) and item["attributes"]:
        attrs = item["attributes"]
        if sum(len(str(v)) for v in attrs.values()) <= 200:
            out["attributes"] = attrs
        else:
            out["_attr_keys"] = list(attrs.keys())[:8]
    return out


def compact_relation(item):
    if not isinstance(item, dict):
        return item
    out = {}
    for k in ("family_id", "entity1_family_id", "entity2_family_id",
               "entity1_name", "entity2_name", "relation_type", "event_time", "confidence"):
        if k in item:
            out[k] = item[k]
    if "_score" in item:
        out["_score"] = item["_score"]
    if "content" in item and isinstance(item["content"], str):
        out["content"] = truncate_text(item["content"])
    return out


def compact_version(item):
    if not isinstance(item, dict):
        return item
    out = {}
    for k in ("absolute_id", "name", "event_time", "processed_time"):
        if k in item:
            out[k] = item[k]
    for ck in ("content", "markdown_content", "summary"):
        if ck in item and isinstance(item[ck], str):
            out[ck] = truncate_text(item[ck])
    return out


def compact_item(item):
    """Auto-detect item type and apply the right compaction."""
    if not isinstance(item, dict):
        return item
    if "entity1_id" in item or "entity1_name" in item:
        return compact_relation(item)
    if "absolute_id" in item and "processed_time" in item and "family_id" not in item:
        return compact_version(item)
    return compact_entity(item)


# ── Bulky field stripping ─────────────────────────────────────────────────

def strip_bulky(obj):
    if isinstance(obj, dict):
        if not _TRIM_FIELDS.intersection(obj):
            _changed = False
            _result = {}
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    _v = strip_bulky(v)
                    if _v is not v:
                        _changed = True
                    _result[k] = _v
                else:
                    _result[k] = v
            return _result if _changed else obj
        return {k: strip_bulky(v) for k, v in obj.items() if k not in _TRIM_FIELDS}
    if isinstance(obj, list):
        return [strip_bulky(item) for item in obj]
    return obj


# ── List compaction with binary-search trimming ───────────────────────────

_LIST_KEYS = ("entities", "relations", "versions", "episodes", "items", "explored", "seeds",
              "latest_entities", "latest_relations")


def compact_lists(data, max_chars=_MAX_RESPONSE_CHARS):
    """Compact large lists in a response dict, keeping within max_chars."""
    if not isinstance(data, dict):
        return data

    # Recurse into nested dicts (e.g. results.entities in /find/ask)
    for k, v in list(data.items()):
        if isinstance(v, dict) and not k.endswith(("_total", "_shown")):
            data[k] = compact_lists(v, max_chars=max_chars)

    for key in _LIST_KEYS:
        items = data.get(key)
        if not isinstance(items, list) or len(items) <= 3:
            if isinstance(items, list):
                data[key] = [compact_item(i) for i in items]
            continue

        items = [compact_item(i) for i in items]
        _item_sizes = [len(json.dumps(i, ensure_ascii=False)) for i in items]

        full_size = len(json.dumps(data, ensure_ascii=False))
        key_text_len = len(json.dumps(items, ensure_ascii=False))
        base_overhead = full_size - key_text_len + 30 + 5 * len(items)

        lo, hi = 1, min(len(items), 50)
        best_mid = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            est_size = base_overhead + sum(_item_sizes[:mid])
            if est_size <= max_chars:
                best_mid = mid
                lo = mid + 1
            else:
                hi = mid - 1

        if best_mid > 0 and best_mid < len(items):
            data[key] = items[:best_mid]
            data[f"{key}_total"] = len(items)
            data[f"{key}_shown"] = best_mid
        else:
            data[key] = items

    return data


# ── Error hints ───────────────────────────────────────────────────────────

_HINT_PATTERNS = [
    (lambda m: ("not found" in m or "未找到" in m) and ("entity" in m or "实体" in m),
     "Hint: use search_entities (GET /find/entities/search?query_name=X) or find_entity_by_name to find the correct family_id."),
    (lambda m: ("not found" in m or "未找到" in m) and ("relation" in m or "关系" in m),
     "Hint: use get_relations_between or search_relations to find the correct relation."),
    (lambda m: ("not found" in m or "未找到" in m) and ("episode" in m or "记忆" in m),
     "Hint: use search_episodes or get_latest_episode to find valid cache_ids."),
    (lambda m: "not found" in m and "community" in m,
     "Hint: use list_communities (GET /communities) to see valid IDs. Run detect_communities first if empty."),
    (lambda m: "not found" in m and "task" in m,
     "Hint: use remember_tasks (GET /remember/tasks) to list all tasks with valid IDs."),
    (lambda m: "neo4j" in m and "not available" in m,
     "Hint: this feature requires Neo4j backend."),
    (lambda m: "context budget" in m or "token" in m,
     "Hint: reduce max_entities/max_relations or shorten your query."),
    (lambda m: "already exists" in m,
     "Hint: use find_entity_by_name (GET /find/entities/by-name/{name}) to check if it already exists."),
    (lambda m: "timeout" in m or "timed out" in m,
     "Hint: the operation took too long. Try with smaller input or fewer items."),
    (lambda m: "rate limit" in m or "429" in m,
     "Hint: too many requests. Wait a moment and retry."),
    (lambda m: "策略" in m or "strategy" in m,
     "Hint: check the strategy parameter. Valid strategies: random, hub, cross_community, orphan, time_gap, low_confidence."),
    (lambda m: "search_mode" in m,
     "Hint: check the search_mode parameter. Valid modes: hybrid, semantic, bm25."),
    (lambda m: ("invalid" in m or "无效" in m) and ("family_id" in m or "_id" in m or "identifier" in m),
     "Hint: check that the ID format is correct. family_ids start with 'ent_' or 'rel_', absolute_ids are UUIDs."),
    (lambda m: "merge" in m and ("same" in m or "cannot" in m or "error" in m),
     "Hint: merge_entities requires at least 2 different entity family_ids."),
    (lambda m: "合并" in m and "拒绝" in m,
     "Hint: merge was rejected because source and target names differ too much. Pass skip_name_check: true in the request body to force merge."),
    (lambda m: "合并" in m and "差异" in m,
     "Hint: merge was rejected because source and target names differ too much. Pass skip_name_check: true in the request body to force merge."),
    (lambda m: "cannot" in m and ("delete" in m or "remove" in m),
     "Hint: the resource may be in use. Check system_tasks for running operations."),
    (lambda m: "permission" in m or "forbidden" in m or "unauthorized" in m,
     "Hint: check your API key configuration and graph_id access permissions."),
    (lambda m: "validation" in m or ("required" in m and "field" in m),
     "Hint: check the required parameters. Missing or empty fields cause validation errors."),
    (lambda m: "conflict" in m,
     "Hint: resource state conflict. The data may have changed since last read — refresh with get_entity or entity_profile."),
    (lambda m: "至少需要提供" in m or "至少需要一个" in m,
     "Hint: provide at least one of: name, content, summary, or attributes in the request body."),
    (lambda m: "需为非空数组" in m or "non-empty array" in m,
     "Hint: the array parameter must contain at least one item."),
    (lambda m: "为必填" in m or "必填" in m or "required" in m.lower() and "parameter" in m.lower(),
     "Hint: a required parameter is missing. Check the API docs for required fields."),
    (lambda m: ("graph" in m or "图谱" in m) and ("not found" in m or "不存在" in m),
     "Hint: use list_graphs (GET /graphs) to see available graphs."),
    (lambda m: ("graph" in m or "图谱" in m) and "already exists" in m,
     "Hint: use a different graph_id. Use list_graphs (GET /graphs) to see existing graphs."),
]


def error_hint(error_message):
    """Auto-detect actionable hint from an error message string."""
    if not isinstance(error_message, str):
        return None
    lower = error_message.lower()
    for matcher, hint in _HINT_PATTERNS:
        if matcher(lower):
            return hint
    return None


def empty_search_hint(query_param="query"):
    return f"No results found. Try: lower similarity_threshold, use search_mode='hybrid', or rephrase the {query_param}."


# ── ID validation ─────────────────────────────────────────────────────────

_FAMILY_ID_RE = re.compile(r'^(ent|rel)_')
_UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)


def validate_family_id(value, param_name="family_id"):
    """Reject UUID-style absolute_ids where a family_id is expected."""
    if isinstance(value, str) and '-' in value and value.count('-') == 4 and len(value) == 36:
        prefix = value[:8]
        raise ValueError(
            f"'{prefix}...' looks like an absolute_id (UUID), but {param_name} requires a "
            f"family_id (e.g. 'ent_abc123' or 'rel_abc123'). "
            f"Use GET /find/entities/absolute/{value} if you need to access by version ID."
        )
    return value


def validate_absolute_id(value, param_name="absolute_id"):
    """Reject family_ids where an absolute_id is expected."""
    if isinstance(value, str) and _FAMILY_ID_RE.match(value) and '-' not in value:
        raise ValueError(
            f"'{value}' looks like a family_id, but {param_name} requires an absolute_id "
            f"(version ID like a UUID). Use GET /find/entities/{value} to find the current absolute_id."
        )
    return value
