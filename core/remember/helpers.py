"""Shared extraction utilities for the extraction pipeline.

Entity/relation validation, deduplication, normalization, and quality gates.
"""
from __future__ import annotations


import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from core.models import Entity
from core.utils import normalize_entity_pair


# ---------------------------------------------------------------------------
# Pre-compiled regex patterns (module-level for reuse)
# ---------------------------------------------------------------------------

_PAREN_RE = re.compile(r'[（(][^）)]+[)）]')
_ENGLISH_RE = re.compile(r'^[A-Za-z0-9\s\-_.]+$')
_EN_PAREN_RE = re.compile(r'[(（][A-Za-z]')

# Canonical parenthetical-annotation regex — re-exported by other modules.
# Matches any (full-width or half-width) parenthesized annotation.
_PAREN_ANNOTATION_RE = _PAREN_RE


# ---------------------------------------------------------------------------
# Entity name cleaning
# ---------------------------------------------------------------------------

def _clean_entity_name(name: str) -> str:
    """Strip parenthetical annotations from entity names.

    LLMs sometimes add scene/context annotations in parentheses.
    Strip them all — downstream entity alignment handles disambiguation.
    e.g. "曹操（汉中张鲁）" → "曹操", "曹操（魏王）" → "曹操"

    Also handles "/" compound names: "变更/Mutation" → "变更"
    """
    # Strip all parenthetical content (full-width and half-width)
    cleaned = _PAREN_RE.sub('', name)

    # Handle "/" compound names: strip English alias
    if "/" in cleaned:
        parts = [p.strip() for p in cleaned.split("/") if p.strip()]
        if len(parts) == 2:
            if _ENGLISH_RE.match(parts[1]) and not _ENGLISH_RE.match(parts[0]):
                cleaned = parts[0]
            elif _ENGLISH_RE.match(parts[0]) and not _ENGLISH_RE.match(parts[1]):
                cleaned = parts[1]

    return cleaned.strip() or name.strip()


# ---------------------------------------------------------------------------
# Relation content minimum length
# ---------------------------------------------------------------------------

# 关系内容最小字符数 — extraction 和 relation 模块统一使用
MIN_RELATION_CONTENT_LENGTH = 8


# ---------------------------------------------------------------------------
# Content-similarity dedup
# ---------------------------------------------------------------------------

def _word_set(text: str) -> set:
    """Extract bigram token set for Jaccard comparison.

    Uses bigrams (character pairs) instead of single characters to avoid
    false merges like "曹孟德"/"孟德曹" which have identical char-level
    token sets but are different entities. Bigram Jaccard is significantly
    more accurate for Chinese text.
    """
    # Extract meaningful characters first (Chinese chars + alphanumeric)
    chars = []
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            chars.append(char)
        elif char.isalnum():
            chars.append(char.lower())
    # Build bigram set from the filtered character sequence
    if len(chars) < 2:
        return set(chars)
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def _dedupe_by_content_similarity(
    entities: List[Dict[str, str]],
    relations: List[Dict[str, str]],
    jaccard_threshold: float = 0.65,
) -> List[Dict[str, str]]:
    """按内容相似度去重：如果多个实体的content高度相似，说明它们都在描述同一主体而非自身。"""
    if len(entities) <= 1:
        return entities

    # 收集关系端点，用于判断实体是否"有连接"
    related_names: set = set()
    for rel in relations:
        for key in ('entity1_name', 'entity2_name'):
            name = rel.get(key, '').strip()
            if name:
                related_names.add(name)

    content_words = [_word_set(e.get('content', '')) for e in entities]
    content_sizes = [len(w) for w in content_words]

    # Sort indices by content size (ascending) — comparing smaller sets first
    # means cheaper intersections and better early-exit pruning on size ratio
    _order = sorted(range(len(entities)), key=lambda i: content_sizes[i])

    # Union-Find 分组
    parent = list(range(len(entities)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for _ii in range(len(_order)):
        i = _order[_ii]
        si = content_sizes[i]
        if not si:
            continue
        for _jj in range(_ii + 1, len(_order)):
            j = _order[_jj]
            sj = content_sizes[j]
            if not sj:
                continue
            # Early exit: max possible Jaccard is min(si,sj)/max(si,sj)
            # Since _order is sorted by size, si <= sj, so ratio = si/sj
            if si < sj * jaccard_threshold:
                continue
            intersection = content_words[i] & content_words[j]
            union_set = si + sj - len(intersection)
            if union_set:
                jaccard = len(intersection) / union_set
                if jaccard > jaccard_threshold:
                    union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(entities)):
        root = find(i)
        groups[root].append(i)

    out: List[Dict[str, str]] = []
    for indices in groups.values():
        if len(indices) == 1:
            out.append(entities[indices[0]])
            continue
        # 多个实体内容高度相似：保留最有价值的一个
        def _entity_score(idx: int) -> tuple:
            _ent = entities[idx]
            name = _ent.get('name', '')
            is_related = name in related_names
            content_len = len(_ent.get('content', ''))
            return (is_related, content_len)

        best_idx = max(indices, key=_entity_score)
        out.append(entities[best_idx])

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_pair_for_relation(e1: str, e2: str) -> Tuple[str, str]:
    """无向边端点按字典序固定。委托给 processor.utils.normalize_entity_pair。"""
    return normalize_entity_pair(e1, e2)


# ---------------------------------------------------------------------------
# Entity name validation — structural checks only
# Content-based filtering is handled by prompt engineering
# ---------------------------------------------------------------------------

_MAX_ENTITY_NAME_LENGTH = 60
_MIN_ENTITY_NAME_LENGTH = 2


# ---------------------------------------------------------------------------
# Entity name validation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _is_valid_entity_name(name: str) -> bool:
    """Structural validation: length, format, not pure numbers."""
    if not name or len(name) < _MIN_ENTITY_NAME_LENGTH:
        return False

    _has_en_paren = bool(_EN_PAREN_RE.search(name))
    _max_len = 50 if (name.isascii() and ' ' in name) or _has_en_paren else _MAX_ENTITY_NAME_LENGTH
    if len(name) > _max_len:
        return False
    if name.strip().isdigit():
        return False
    return True


# ---------------------------------------------------------------------------
# Entity core name extraction
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2048)
def _core_entity_name(name: str) -> str:
    """提取实体名称的核心部分（去掉所有括号），用于去重比较。"""
    return _PAREN_RE.sub('', name).strip()


# ---------------------------------------------------------------------------
# Entity dedup
# ---------------------------------------------------------------------------

def dedupe_extracted_entities(entities: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """按实体 name（strip 后）去重；同名时保留 content 更长的条目。同时过滤非实体文本片段。"""
    name_to_index: Dict[str, int] = {}
    core_name_to_index: Dict[str, int] = {}
    out: List[Dict[str, str]] = []
    for e in entities or []:
        if not isinstance(e, dict):
            continue
        name = (e.get("name") or "").strip()
        if not name:
            continue

        cleaned_name = _clean_entity_name(name)
        if cleaned_name != name:
            name = cleaned_name

        if not _is_valid_entity_name(name):
            continue

        content = (e.get("content") or "").strip()
        if len(content) < 8:
            continue

        existing_idx = name_to_index.get(name)
        if existing_idx is not None:
            if len(content) > len(out[existing_idx]["content"]):
                out[existing_idx] = {"name": name, "content": content}
            continue

        core_name = _core_entity_name(name)
        core_idx = core_name_to_index.get(core_name)
        if core_idx is not None:
            existing = out[core_idx]
            if len(name) > len(existing["name"]):
                existing["name"] = name
            name_to_index[name] = core_idx
            if len(content) > len(existing["content"]):
                existing["content"] = content
            continue

        idx = len(out)
        name_to_index[name] = idx
        if core_name != name:
            core_name_to_index[core_name] = idx
        out.append({"name": name, "content": content})
    return out


# ---------------------------------------------------------------------------
# Relation content validation
# ---------------------------------------------------------------------------

def _is_valid_relation_content(content: str, entity1_name: str = "", entity2_name: str = "") -> bool:
    """Structural validation: content length only."""
    if not content or len(content) < MIN_RELATION_CONTENT_LENGTH:
        return False
    return True


# ---------------------------------------------------------------------------
# Relation dedup
# ---------------------------------------------------------------------------

def dedupe_extracted_relations(relations: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """关系去重：无向 (entity1, entity2) 字典序 + content（忽略大小写）。"""
    seen: set[Tuple[str, str, int]] = set()
    out: List[Dict[str, str]] = []
    for r in relations or []:
        if not isinstance(r, dict):
            continue
        e1 = (r.get("entity1_name") or "").strip()
        e2 = (r.get("entity2_name") or "").strip()
        content = (r.get("content") or "").strip()
        if not e1 or not e2 or not content:
            continue
        if e1 == e2:
            continue
        if not _is_valid_relation_content(content, e1, e2):
            continue
        n1, n2 = _normalize_pair_for_relation(e1, e2)
        key = (n1, n2, hash(content.lower()))
        if key in seen:
            continue
        seen.add(key)
        out.append({"entity1_name": n1, "entity2_name": n2, "content": content})
    return out


# ---------------------------------------------------------------------------
# Validate written entities
# ---------------------------------------------------------------------------

def validate_written_entities_with_report(
    entities: Optional[List[Dict[str, Any]]],
    stable_names: List[str],
    fallback_content_by_name: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """校验实体写作结果，确保实体名稳定且 content 不为空。"""
    stable_name_set = set(stable_names)
    fallback_content_by_name = fallback_content_by_name or {}
    name_to_best: Dict[str, Dict[str, str]] = {}
    rejected: List[Dict[str, str]] = []

    for entity in entities or []:
        if not isinstance(entity, dict):
            rejected.append({"reason": "invalid_payload", "name": ""})
            continue
        name = _clean_entity_name((entity.get("name") or "").strip())
        if not name:
            rejected.append({"reason": "missing_name", "name": ""})
            continue
        if stable_name_set and name not in stable_name_set:
            rejected.append({"reason": "unknown_stable_name", "name": name})
            continue
        content = (entity.get("content") or "").strip()
        if not content:
            content = fallback_content_by_name.get(name, "").strip()
        if not content:
            content = f'文本中出现了关于\u201c{name}\u201d的描述。'
            rejected.append({"reason": "empty_content_fallback", "name": name})

        candidate = {"name": name, "content": content}
        existing = name_to_best.get(name)
        if existing is None:
            name_to_best[name] = candidate
            continue
        if len(content) > len(existing.get("content", "")):
            name_to_best[name] = candidate
            rejected.append({"reason": "duplicate_name_replaced_shorter_content", "name": name})
        else:
            rejected.append({"reason": "duplicate_name", "name": name})

    out: List[Dict[str, str]] = []
    for name in stable_names:
        candidate = name_to_best.get(name)
        if candidate is not None:
            out.append(candidate)
            continue
        fallback = fallback_content_by_name.get(name, "").strip() or f'文本中出现了关于\u201c{name}\u201d的描述。'
        out.append({"name": name, "content": fallback})
        rejected.append({"reason": "missing_written_entity_fallback", "name": name})
    return out, rejected


# ---------------------------------------------------------------------------
# Validate extracted relations
# ---------------------------------------------------------------------------

def validate_extracted_relations_with_report(
    relations: Optional[List[Dict[str, Any]]],
    valid_entity_names: set[str],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """关系内容校验版，返回合法关系与拒绝原因。"""
    seen: set[Tuple[str, str, str]] = set()
    out: List[Dict[str, str]] = []
    rejected: List[Dict[str, str]] = []

    for relation in relations or []:
        if not isinstance(relation, dict):
            rejected.append({"reason": "invalid_payload"})
            continue
        entity1_name = (relation.get("entity1_name") or "").strip()
        entity2_name = (relation.get("entity2_name") or "").strip()
        content = (relation.get("content") or "").strip()
        if not entity1_name or not entity2_name:
            rejected.append({
                "reason": "missing_endpoint",
                "entity1_name": entity1_name,
                "entity2_name": entity2_name,
            })
            continue
        if entity1_name == entity2_name:
            rejected.append({
                "reason": "self_relation",
                "entity1_name": entity1_name,
                "entity2_name": entity2_name,
            })
            continue
        n1, n2 = _normalize_pair_for_relation(entity1_name, entity2_name)
        if n1 not in valid_entity_names or n2 not in valid_entity_names:
            rejected.append({
                "reason": "unknown_endpoint",
                "entity1_name": n1,
                "entity2_name": n2,
            })
            continue
        if not _is_valid_relation_content(content, n1, n2):
            rejected.append({
                "reason": "invalid_content",
                "entity1_name": n1,
                "entity2_name": n2,
                "content": content,
            })
            continue
        key = (n1, n2, hash(content.lower()))
        if key in seen:
            rejected.append({
                "reason": "duplicate_relation",
                "entity1_name": n1,
                "entity2_name": n2,
                "content": content,
            })
            continue
        seen.add(key)
        out.append({"entity1_name": n1, "entity2_name": n2, "content": content})

    return out, rejected


# ---------------------------------------------------------------------------
# Combined entity + relation dedup (used by orchestrator cache loader)
# ---------------------------------------------------------------------------

def dedupe_extraction_lists(
    entities: Optional[List[Dict[str, Any]]],
    relations: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """供缓存加载等场景：实体、关系各做一次列表级去重。"""
    return dedupe_extracted_entities(entities), dedupe_extracted_relations(relations)


# ---------------------------------------------------------------------------
# Alignment result dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _AlignResult:
    """步骤6（实体对齐）的输出，供步骤7使用。"""
    entity_name_to_id: Dict[str, str] = field(default_factory=dict)
    pending_relations: List[Dict] = field(default_factory=list)
    unique_entities: List[Entity] = field(default_factory=list)
    unique_pending_relations: List[Dict] = field(default_factory=list)
    resolved_family_ids: Optional[set] = None  # set of validated family_ids (skip re-resolution)
