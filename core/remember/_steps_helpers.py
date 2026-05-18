"""Extraction pipeline helper functions and utilities.

Split from steps.py — contains name dedup, validation, prose indexing,
parallel map, and entity fallback content generation.
"""
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from core.utils import wprint_info

from .helpers import _clean_entity_name, _is_valid_entity_name, _core_entity_name
from ._shared import _get_or_create_pool

# Pre-compiled patterns for _build_entity_fallback_content
_MD_HEADER_RE = re.compile(r'^#{1,6}\s')
_EMOJI_LEAD_RE = re.compile(r'^[\U0001F300-\U0001F9FF\U00002600-\U000027BF✂-➿]')
_BULLET_LABEL_RE = re.compile(r'^[-*]\s+[✅❌👉]')
_SENTENCE_SPLIT_RE = re.compile(r'[。！？\n]')


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    """Deterministic unordered pair key — avoids list alloc + sorted() + tuple()."""
    return (a, b) if a <= b else (b, a)


# ---------------------------------------------------------------------------
# Parallel/sequential executor helper
# ---------------------------------------------------------------------------

# Lazily-initialized shared pool — avoids thread creation/teardown per call.
_SHARED_POOL: list = [None]
_SHARED_POOL_MAX_WORKERS: list = [1]


def _get_shared_pool(max_workers: int) -> ThreadPoolExecutor:
    """Return (and lazily create) the shared ThreadPoolExecutor."""
    return _get_or_create_pool(_SHARED_POOL, max_workers, _SHARED_POOL_MAX_WORKERS, "extract")


def _parallel_map(
    items: List,
    process_fn,
    fallback_fn=None,
    n_workers: int = 1,
    thread_prefix: str = "extract",
) -> List:
    """Process items in parallel, falling back to sequential on RuntimeError.

    Reuses a lazily-initialized module-level ThreadPoolExecutor to avoid
    thread creation/teardown overhead across pipeline invocations.
    """
    if not items:
        return []

    def _sequential():
        results = []
        for item in items:
            try:
                r = process_fn(item)
                if r is not None:
                    results.append(r)
            except Exception as exc:
                if fallback_fn:
                    results.append(fallback_fn(item, exc))
        return results

    if len(items) <= 1:
        return _sequential()

    n_workers = min(len(items), n_workers)
    pool = None
    try:
        pool = _get_shared_pool(n_workers)
        futures = {pool.submit(process_fn, item): item for item in items}
    except RuntimeError:
        pool = None

    if pool is None:
        return _sequential()

    results = []
    for fut in as_completed(futures):
        try:
            r = fut.result()
            if r is not None:
                results.append(r)
        except Exception as exc:
            if fallback_fn:
                results.append(fallback_fn(futures[fut], exc))

    return results


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
