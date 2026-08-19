"""Extraction pipeline helper functions and utilities.

Split from steps.py — contains name dedup, validation, prose indexing,
and entity fallback content generation.
"""
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from .helpers import _clean_entity_name, _is_valid_entity_name, _core_entity_name

# Pre-compiled patterns for _build_entity_fallback_content
_MD_HEADER_RE = re.compile(r'^#{1,6}\s')
_EMOJI_LEAD_RE = re.compile(r'^[\U0001F300-\U0001F9FF\U00002600-\U000027BF✂-➿]')
_BULLET_LABEL_RE = re.compile(r'^[-*]\s+[✅❌👉]')
_SENTENCE_SPLIT_RE = re.compile(r'[。！？\n]')


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    """Deterministic unordered pair key — avoids list alloc + sorted() + tuple()."""
    return (a, b) if a <= b else (b, a)


# ---------------------------------------------------------------------------
# Name cleaning & dedup
# ---------------------------------------------------------------------------


def _dedup_entity_names(names: List[str]) -> List[str]:
    """Deduplicate entity names using core-name matching."""
    seen_core: Dict[str, str] = {}
    result: List[str] = []
    _pos: Dict[str, int] = {}

    for name in names:
        if not _is_valid_entity_name(name):
            continue

        core = _core_entity_name(name)
        if not core:
            continue

        existing = seen_core.get(core)
        if existing is None:
            seen_core[core] = name
            _pos[core] = len(result)
            result.append(name)
        else:
            if "(" in name or "（" in name:
                if "(" not in existing and "（" not in existing:
                    if core in _pos:
                        result[_pos[core]] = name
                        seen_core[core] = name

    return result


def _normalize_and_dedup_entity_names(raw_names: List[str]) -> List[str]:
    """Clean, split, validate and dedup entity names in a single pass."""
    expanded: List[str] = []
    for name in raw_names:
        cleaned = _clean_entity_name(name)
        if "/" in cleaned:
            for part in cleaned.split("/"):
                part = part.strip()
                if part and len(part) >= 2:
                    expanded.append(part)
        else:
            expanded.append(cleaned)

    return _dedup_entity_names(expanded)


def _build_name_lookup(entity_name_set: Set[str]) -> Dict[str, Any]:
    """Pre-compute lookup structures for entity name resolution."""
    lower_map: Dict[str, str] = {}
    core_name_map: Dict[str, List[str]] = {}
    for name in entity_name_set:
        lower_map[name.lower()] = name
        core = _core_entity_name(name)
        if core not in core_name_map:
            core_name_map[core] = []
        core_name_map[core].append(name)
    return {"lower_map": lower_map, "core_name_map": core_name_map, "names": entity_name_set}


def _resolve_entity_name(raw_name: str, entity_name_set: Set[str],
                         _lookup: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Resolve a potentially fuzzy entity name to a known name.

    Args:
        _lookup: Pre-computed lookup from _build_name_lookup(). If provided,
                 avoids O(N) linear scans per call.
    """
    raw_name = raw_name.strip()
    if not raw_name:
        return None

    # Exact match
    if raw_name in entity_name_set:
        return raw_name

    if _lookup:
        # Case-insensitive match via dict lookup (O(1))
        lower_map = _lookup["lower_map"]
        match = lower_map.get(raw_name.lower())
        if match:
            return match

        # Core name match via dict lookup (O(1))
        core_name_map = _lookup["core_name_map"]
        raw_core = _core_entity_name(raw_name)
        matches = core_name_map.get(raw_core)
        if matches and len(matches) == 1:
            return matches[0]

        # Substring match (last resort, O(N))
        for known in entity_name_set:
            if raw_core in known or known in raw_core:
                return known
    else:
        # Case-insensitive match
        _raw_lower = raw_name.lower()
        for known in entity_name_set:
            if known.lower() == _raw_lower:
                return known

        # Core name match (strip parenthetical)
        raw_core = _core_entity_name(raw_name)
        matches = [n for n in entity_name_set if _core_entity_name(n) == raw_core]
        if len(matches) == 1:
            return matches[0]

        # Substring match (if raw is a substring of a known name or vice versa)
        for known in entity_name_set:
            if raw_core in known or known in raw_core:
                return known

    return None


# ---------------------------------------------------------------------------
# Quality gates — structural checks only
# ---------------------------------------------------------------------------

_MIN_ENTITY_CONTENT_LEN = 15
_MIN_RELATION_CONTENT_LEN = 10

_FILLER_PATTERNS = re.compile(
    r'^(?:'
    r'[^，。！？]{2,8}是(?:一个|一种)?(?:在文本中|本文中)?(?:被讨论|被提及|提到|涉及)的.{0,10}(?:概念|主题|内容|要素|方面)'
    r'|[^，。！？]{2,8}是(?:一个|一种)?(?:重要|核心|关键|主要)的.{0,10}(?:概念|主题|内容|要素|方面)'
    r'|.{0,20}具有特定的.{0,10}(?:语义|知识|内涵|意义)'
    r')$'
)


def _validate_entity(name: str, content: str) -> bool:
    """Structural + semantic validation: content length and filler detection."""
    if not content or len(content) < _MIN_ENTITY_CONTENT_LEN:
        return False
    if _FILLER_PATTERNS.match(content):
        return False
    return True


def _validate_relation(
    entity_a: str, entity_b: str, content: str, valid_entity_names: Set[str],
) -> bool:
    """Structural + semantic validation: content length, no self-relation, filler detection."""
    if not content or len(content) < _MIN_RELATION_CONTENT_LEN:
        return False
    if entity_a == entity_b:
        return False
    if _FILLER_PATTERNS.match(content):
        return False
    return True


def _prepare_prose_sentences(window_text: str) -> List[str]:
    """Pre-split window text into prose sentences for fallback content building."""
    if not window_text:
        return []

    raw_sentences = _SENTENCE_SPLIT_RE.split(window_text)
    prose = []
    for s in raw_sentences:
        s = s.strip()
        if not s or len(s) <= 5:
            continue
        if _MD_HEADER_RE.match(s):
            continue
        if _EMOJI_LEAD_RE.match(s):
            continue
        if _BULLET_LABEL_RE.match(s):
            continue
        prose.append(s)
    return prose


class _ProseIndex:
    """Pre-computed index over prose sentences for fast substring matching.

    Builds a bigram → {sentence_index} inverted index once, then serves
    O(1) lookups per entity name instead of O(M) full scans.
    """

    __slots__ = ('sentences', '_bigram_map')

    def __init__(self, sentences: List[str]):
        self.sentences = sentences
        self._bigram_map: Dict[str, Set[int]] = defaultdict(set)
        for i, s in enumerate(sentences):
            for j in range(len(s) - 1):
                self._bigram_map[s[j:j + 2]].add(i)


def _build_entity_fallback_content(name: str, prose_index: '_ProseIndex') -> str:
    """Build a context-aware fallback description when LLM content writing fails."""
    sentences = prose_index.sentences
    if not sentences:
        return f"文本中出现了关于{name}的描述。"

    name_bigrams = set(name[i:i + 2] for i in range(len(name) - 1))
    if name_bigrams:
        _bm = prose_index._bigram_map
        sorted_bgs = sorted(name_bigrams, key=lambda bg: len(_bm.get(bg, ())))
        candidates = None
        for bg in sorted_bgs:
            idx_set = _bm.get(bg)
            if idx_set is None:
                candidates = set()
                break
            if candidates is None:
                candidates = set(idx_set)
            else:
                candidates.intersection_update(idx_set)
            if not candidates:
                break
        if candidates:
            relevant = [sentences[i] for i in sorted(candidates)
                        if name in sentences[i]]
        else:
            relevant = []
    else:
        relevant = [s for s in sentences if name in s]

    if relevant:
        desc_parts = relevant[:3]
        desc = '。'.join(desc_parts)
        if len(desc) > 200:
            desc = desc[:197] + '...'
        if not desc.endswith('。'):
            desc += '。'
        return desc

    if len(name) >= 4:
        for part in name_bigrams:
            idx_set = prose_index._bigram_map.get(part)
            if idx_set:
                for si in sorted(idx_set):
                    return _format_desc(sentences[si])

    return f"文本中出现了关于{name}的描述。"


def _format_desc(sentence: str, max_len: int = 200) -> str:
    """Format a single sentence as a fallback description."""
    if len(sentence) > max_len:
        sentence = sentence[:max_len - 3] + '...'
    if not sentence.endswith('。'):
        sentence += '。'
    return sentence
