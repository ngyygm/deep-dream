"""
Extraction Pipeline — "one small task per step".

Pipeline steps (within a sliding window):
  Step 2: Comprehensive entity extraction (one LLM call + refinement)
  Step 3: Entity dedup & normalization (rule-based)
  Step 4: Per-entity content writing (one LLM call per entity)
  Step 5: Entity quality gate (rule-based)
  Step 6: Comprehensive relation discovery (one LLM call + refinement)
  Step 7: Per-pair relation content writing (one LLM call per pair)
  Step 8: Relation quality gate (rule-based)

Step 1 (cache update) is in alignment.py.
Steps 9 (entity alignment) and 10 (relation alignment) are in alignment.py and orchestrator.py.
"""

import re
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from core.utils import wprint_info

# Pre-compiled patterns for _build_entity_fallback_content
_MD_HEADER_RE = re.compile(r'^#{1,6}\s')
_EMOJI_LEAD_RE = re.compile(r'^[\U0001F300-\U0001F9FF\U00002600-\U000027BF\u2702-\u27BF]')
_BULLET_LABEL_RE = re.compile(r'^[-*]\s+[✅❌👉]')
_SENTENCE_SPLIT_RE = re.compile(r'[。！？\n]')
from .helpers import _clean_entity_name, _is_valid_entity_name, _core_entity_name


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    """Deterministic unordered pair key — avoids list alloc + sorted() + tuple()."""
    return (a, b) if a <= b else (b, a)


# ---------------------------------------------------------------------------
# Parallel/sequential executor helper
# ---------------------------------------------------------------------------

# Lazily-initialized shared pool — avoids thread creation/teardown per call.
# _parallel_map is the hottest parallel utility in the pipeline; a persistent
# pool saves ~10 create/destroy cycles per remember invocation.
_SHARED_POOL: Optional[ThreadPoolExecutor] = None
_SHARED_POOL_MAX_WORKERS = 4  # covers the default n_workers=4


def _get_shared_pool(max_workers: int) -> ThreadPoolExecutor:
    """Return (and lazily create) the shared ThreadPoolExecutor."""
    global _SHARED_POOL, _SHARED_POOL_MAX_WORKERS
    if _SHARED_POOL is not None:
        # Grow pool if caller needs more workers
        if max_workers > _SHARED_POOL_MAX_WORKERS:
            try:
                _SHARED_POOL.shutdown(wait=False)
            except Exception:
                pass
            _SHARED_POOL = None
        else:
            return _SHARED_POOL
    _SHARED_POOL_MAX_WORKERS = max(max_workers, _SHARED_POOL_MAX_WORKERS)
    _SHARED_POOL = ThreadPoolExecutor(
        max_workers=_SHARED_POOL_MAX_WORKERS,
        thread_name_prefix="extract",
    )
    return _SHARED_POOL


def _parallel_map(
    items: List,
    process_fn,
    fallback_fn=None,
    n_workers: int = 4,
    thread_prefix: str = "extract",
) -> List:
    """Process items in parallel, falling back to sequential on RuntimeError.

    Reuses a lazily-initialized module-level ThreadPoolExecutor to avoid
    thread creation/teardown overhead across pipeline invocations.

    Args:
        items: Items to process.
        process_fn: Callable(item) -> result. Called in parallel.
        fallback_fn: Optional callable(item, exception) -> result for per-item failures.
        n_workers: Max parallel workers.
        thread_prefix: Thread name prefix for debugging.

    Returns:
        List of results (order not guaranteed).
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
    _pos: Dict[str, int] = {}  # core -> index in result

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
                    # Replace existing with the name that has parenthetical annotation
                    # Use dict to track positions instead of O(n) index() each time
                    if core in _pos:
                        result[_pos[core]] = name
                        seen_core[core] = name

    return result


def _normalize_and_dedup_entity_names(raw_names: List[str]) -> List[str]:
    """Clean, split, validate and dedup entity names in a single pass.

    Replaces the previous three-step: dedup → clean → split → dedup → dedup.
    """
    # Phase 1: clean each name, then split "/" compounds
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

    # Phase 2: single dedup pass using core-name matching
    return _dedup_entity_names(expanded)


# ---------------------------------------------------------------------------
# Quality gates — structural checks only
# ---------------------------------------------------------------------------

_MIN_ENTITY_CONTENT_LEN = 15
_MIN_RELATION_CONTENT_LEN = 10

# Generic filler patterns that indicate the LLM produced template content
# instead of meaningful entity/relation descriptions.
_FILLER_PATTERNS = re.compile(
    r'^(?:'
    r'[^，。！？]{2,8}是(?:一个|一种)?(?:在文本中|本文中)?(?:被讨论|被提及|提到|涉及)的.{0,10}(?:概念|主题|内容|要素|方面)'  # "X是一个被讨论的概念"
    r'|[^，。！？]{2,8}是(?:一个|一种)?(?:重要|核心|关键|主要)的.{0,10}(?:概念|主题|内容|要素|方面)'  # "X是一个重要的概念"
    r'|.{0,20}具有特定的.{0,10}(?:语义|知识|内涵|意义)'  # "具有特定的语义和知识内涵"
    r')$'
)


def _validate_entity(name: str, content: str) -> bool:
    """Structural + semantic validation: content length and filler detection."""
    if not content or len(content) < _MIN_ENTITY_CONTENT_LEN:
        return False
    # Reject generic filler content — these carry no useful information
    # for alignment or retrieval and waste storage.
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
    """Pre-split window text into prose sentences for fallback content building.

    Called once per window, reused across all entities in that window.
    Returns a list of clean prose sentences (non-markdown, non-emoji, adequate length).
    """
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
    """Build a context-aware fallback description when LLM content writing fails.

    Args:
        name: Entity name.
        prose_index: Pre-computed _ProseIndex from _prepare_prose_sentences().
    """
    sentences = prose_index.sentences
    if not sentences:
        return f"文本中出现了关于{name}的描述。"

    # Exact name match using pre-built index
    name_bigrams = set(name[i:i + 2] for i in range(len(name) - 1))
    if name_bigrams:
        # Find sentences containing ALL bigrams of the name (superset check via index)
        # Sort by ascending frequency for early pruning; use intersection_update to avoid allocs
        _bm = prose_index._bigram_map
        sorted_bgs = sorted(name_bigrams, key=lambda bg: len(_bm.get(bg, ())))
        candidates = None
        for bg in sorted_bgs:
            idx_set = _bm.get(bg)
            if idx_set is None:
                candidates = set()
                break
            if candidates is None:
                candidates = set(idx_set)  # copy only the first (smallest) set
            else:
                candidates.intersection_update(idx_set)  # in-place, no allocation
            if not candidates:
                break  # empty intersection, stop early
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

    # Partial bigram match — find sentences matching at least one bigram of name
    if len(name) >= 4:
        # Reuse name_bigrams computed above
        for part in name_bigrams:
            idx_set = prose_index._bigram_map.get(part)
            if idx_set:
                # Pick the first matching sentence (deterministic, sorted order)
                for si in sorted(idx_set):
                    return _format_desc(sentences[si])
        # If we got here, no bigram matched any sentence

    return f"文本中出现了关于{name}的描述。"


def _format_desc(sentence: str, max_len: int = 200) -> str:
    """Format a single sentence as a fallback description."""
    if len(sentence) > max_len:
        sentence = sentence[:max_len - 3] + '...'
    if not sentence.endswith('。'):
        sentence += '。'
    return sentence


def _build_relation_fallback_content(
    entity_a: str, entity_b: str, prose_index: '_ProseIndex',
) -> str:
    """Build relation content from window text when both entities co-occur."""
    sentences = prose_index.sentences
    if not sentences:
        return ""
    relevant = [s for s in sentences if entity_a in s and entity_b in s]
    if not relevant:
        return ""
    desc = '。'.join(relevant[:2])
    if len(desc) > 150:
        desc = desc[:147] + '...'
    if not desc.endswith('。'):
        desc += '。'
    return desc


# ---------------------------------------------------------------------------
# Extraction Pipeline Mixin
# ---------------------------------------------------------------------------

class _ExtractionStepsMixin:
    """
    Extraction pipeline mixin.

    Dual-model extraction: strong model for discovery, small model for content.
    Conversational refinement for both entities and relations.
    """

    def _extract_only(
        self,
        new_episode,
        input_text: str,
        document_name: str,
        verbose: bool = True,
        verbose_steps: bool = True,
        event_time=None,
        progress_callback=None,
        progress_range: tuple = (0.1, 0.5),
        window_index: int = 0,
        total_windows: int = 1,
        window_timings_ref: Optional[Dict[str, float]] = None,
        control_check_fn=None,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Dual-model extraction pipeline.

        Uses extraction_client (strong model) for entity/relation discovery,
        llm_client (small model) for content writing.

        Returns:
            (extracted_entities, extracted_relations)
        """
        from core.remember.orchestrator import RememberControlFlow

        def _check_control():
            if control_check_fn:
                action = control_check_fn()
                if action:
                    raise RememberControlFlow(action)

        p_lo, p_hi = progress_range
        _win = f"窗口 {window_index + 1}/{total_windows}"

        def _progress(frac, label, msg):
            if progress_callback:
                progress_callback(p_lo + (p_hi - p_lo) * frac, label, msg)

        def _record_timing(key: str, elapsed: float):
            if window_timings_ref is not None:
                window_timings_ref[key] = elapsed

        extraction_client = self.extraction_client

        # Set up LLM cancel checks so pause/delete can interrupt retry loops
        _cancel_check_fn = (lambda: control_check_fn() is not None) if control_check_fn else None
        if _cancel_check_fn:
            self.llm_client.set_cancel_check(_cancel_check_fn)
            if self.extraction_client_enabled:
                extraction_client.set_cancel_check(_cancel_check_fn)

        # ==============================================================
        # Step 2: Comprehensive entity extraction (strong model, think mode)
        # Step 1b: Conversational refinement
        # ==============================================================
        _progress(0.03, f"{_win} · 步骤2: 实体提取（强模型）", "开始")
        _t = _time.time()
        raw_names, ent_refine = extraction_client.extract_entities(
            input_text, max_refine_rounds=self.entity_rounds
        )
        _elapsed = _time.time() - _t
        _record_timing("step2_entity_extract", _elapsed)
        if verbose or verbose_steps:
            _refine_tag = ""
            if ent_refine["rounds_run"] > 0:
                _refine_tag = f" (精炼{ent_refine['rounds_run']}轮 +{ent_refine['refine_added']})"
            wprint_info(f"【步骤2】综合实体提取｜{ent_refine['initial']}个初始{_refine_tag} → 总{len(raw_names)}个候选｜{_elapsed:.1f}s")

        _check_control()
        # ==============================================================
        _t = _time.time()
        entity_names = _normalize_and_dedup_entity_names(raw_names)
        _elapsed = _time.time() - _t
        _record_timing("step3_entity_dedup", _elapsed)
        if verbose or verbose_steps:
            wprint_info(f"【步骤3】实体去重｜{len(entity_names)}个有效｜{_elapsed:.1f}s")

        _progress(0.15, f"{_win} · 步骤3: 实体去重", f"{len(entity_names)} 个实体")

        _check_control()
        # Step 3: Entity content writing — batch first, per-entity fallback
        # ==============================================================
        _progress(0.18, f"{_win} · 步骤4: 实体内容写作", f"开始写 {len(entity_names)} 个实体")

        # Cache for _build_entity_fallback_content — same name may be called multiple times
        _fallback_cache: Dict[str, str] = {}
        def _cached_fallback(name: str) -> str:
            if name not in _fallback_cache:
                _fallback_cache[name] = _build_entity_fallback_content(name, _prose_index)
            return _fallback_cache[name]

        _t = _time.time()

        # Pre-compute prose index once for all fallback content calls
        _prose_index = _ProseIndex(_prepare_prose_sentences(input_text))

        # All entities go through LLM for proper concept descriptions
        _fast_path_names: List[str] = []
        _needs_llm_names: List[str] = list(entity_names)

        # 4a: Batch write only for entities needing LLM content
        batch_results: Dict[str, str] = {}
        if _needs_llm_names:
            _batch_chunk_size = getattr(self, 'remember_entity_content_batch_size', 10)
            batch_results = self.llm_client.batch_write_entity_content(
                _needs_llm_names, input_text,
                chunk_size=_batch_chunk_size,
            )

            # 4b: Identify entities the batch missed
            _min_content_len = 10
            _missing_names = [
                n for n in _needs_llm_names
                if n not in batch_results or len(batch_results[n]) < _min_content_len
            ]
            if verbose_steps and _missing_names:
                wprint_info(f"  │  S4 batch命中{len(_needs_llm_names) - len(_missing_names)}/{len(_needs_llm_names)}，{_missing_names} 需回退")

            # 4c: Per-entity fallback for missing entities (parallelized)
            if _missing_names:
                def _write_one_entity(name: str) -> Dict[str, str]:
                    content = self.llm_client.write_entity_content(name, input_text)
                    if not content or len(content) < _min_content_len:
                        content = _cached_fallback(name)
                    return {"name": name, "content": content}

                _entity_fallback = lambda name, exc: {"name": name, "content": _cached_fallback(name)}

                _fallback_results = _parallel_map(
                    _missing_names, _write_one_entity,
                    fallback_fn=_entity_fallback,
                    n_workers=4, thread_prefix="extract-econtent",
                )
                for e in _fallback_results:
                    batch_results[e["name"]] = e["content"]

        # Merge fast-path results
        for name in _fast_path_names:
            batch_results[name] = _fallback_cache[name]

        # Assemble final list preserving entity_names order
        extracted_entities = []
        for name in entity_names:
            content = batch_results.get(name, "")
            if not content or len(content) < _MIN_ENTITY_CONTENT_LEN:
                content = _cached_fallback(name)
            extracted_entities.append({"name": name, "content": content})

        _elapsed = _time.time() - _t
        _record_timing("step4_entity_content", _elapsed)
        if verbose or verbose_steps:
            _detail = f"{len(_fast_path_names)}个快速+{len(_needs_llm_names)}个LLM" if _fast_path_names else f"{len(extracted_entities)}个完成"
            wprint_info(f"【步骤4】实体内容写作｜{_detail} → {len(extracted_entities)}个完成｜{_elapsed:.1f}s")

        _check_control()
        #   When content fails validation, try fallback before dropping.
        #   If ALL entities are filtered, keep them with forced fallback
        #   (better to have imperfect entities than lose entire window).
        # ==============================================================
        _t = _time.time()
        valid_entities = []
        rejected_entities = []
        for e in extracted_entities:
            if _validate_entity(e["name"], e["content"]):
                valid_entities.append(e)
            else:
                # Try fallback content extracted from window text
                fallback = _cached_fallback(e["name"])
                if _validate_entity(e["name"], fallback):
                    e["content"] = fallback
                    valid_entities.append(e)
                    if verbose_steps:
                        wprint_info(f"  │  实体质量门挽救: {e['name']} (使用窗口文本回退)")
                else:
                    rejected_entities.append((e, fallback))
                    if verbose_steps:
                        _content_preview = e["content"][:60] if e.get("content") else "(空)"
                        _fallback_preview = fallback[:60] if fallback else "(空)"
                        wprint_info(f"  │  实体质量门拒绝: {e['name']}")
                        wprint_info(f"  │    原始内容: {_content_preview}")
                        wprint_info(f"  │    回退内容: {_fallback_preview}")

        # Emergency rescue: if ALL entities filtered, keep with forced fallback
        if not valid_entities and rejected_entities:
            if verbose_steps:
                wprint_info("  │  ⚠ 全部实体被过滤，启动紧急保留")
            for e, fallback in rejected_entities:
                # Try fallback from window text first (already computed)
                if fallback and _validate_entity(e["name"], fallback):
                    e["content"] = fallback
                else:
                    # Build a minimal but valid description that passes quality gate
                    e["content"] = _cached_fallback(e["name"])
                valid_entities.append(e)

        if verbose or verbose_steps:
            rejected = len(extracted_entities) - len(valid_entities)
            if rejected:
                wprint_info(f"【步骤5】实体质量门｜{rejected}个被过滤，{len(valid_entities)}个通过")

        _record_timing("step5_entity_quality", _time.time() - _t)
        extracted_entities = valid_entities
        entity_name_list = [e["name"] for e in extracted_entities]
        entity_name_set = set(entity_name_list)

        _progress(0.50, f"{_win} · 步骤5: 实体质量门", f"{len(extracted_entities)} 个有效实体")

        _check_control()
        # Step 6b: Conversational refinement
        # ==============================================================
        _t = _time.time()
        relation_pairs = []
        if len(extracted_entities) >= 2:
            _progress(0.53, f"{_win} · 步骤6: 关系发现（强模型）", "开始")

            # Normalize helper
            seen_pairs = set()
            _name_lookup = self._build_name_lookup(entity_name_set)
            def _add_pairs(raw_list):
                added = 0
                for a, b in raw_list:
                    a = self._resolve_entity_name(a, entity_name_set, _lookup=_name_lookup)
                    b = self._resolve_entity_name(b, entity_name_set, _lookup=_name_lookup)
                    if a and b and a != b:
                        pair_key = _pair_key(a, b)
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            relation_pairs.append((a, b))
                            added += 1
                return added

            # Single conversation: initial + refine (coverage handled by refine prompt)
            raw_pairs, rel_stats = extraction_client.discover_relations(
                entity_name_list, input_text, max_refine_rounds=self.relation_rounds
            )
            _initial_count = _add_pairs(raw_pairs)
            _refine_added = len(relation_pairs) - _initial_count

            if verbose or verbose_steps:
                _elapsed5 = _time.time() - _t
                _ref_tag = f" +精炼{_refine_added}对" if _refine_added else ""
                wprint_info(f"【步骤6】关系对发现｜{_initial_count}对初始{_ref_tag} → 总{len(relation_pairs)}对｜{_elapsed5:.1f}s")
            else:
                _elapsed5 = _time.time() - _t
        _record_timing("step6_relation_discovery", _elapsed5)

        _progress(0.60, f"{_win} · 步骤6: 关系发现完成", f"{len(relation_pairs)} 对")

        _check_control()
        # ==============================================================
        _t = _time.time()

        # 7a: All relation pairs go through LLM for proper descriptions
        _fast_rel_results: Dict[Tuple[str, str], str] = {}
        _needs_llm_pairs: List[Tuple[str, str]] = list(relation_pairs)

        # 7b: Batch write for pairs needing LLM content
        batch_rel_results: Dict[Tuple[str, str], str] = {}
        if _needs_llm_pairs:
            _rel_batch_size = getattr(self, 'remember_relation_content_batch_size', 10)
            batch_rel_results = self.llm_client.batch_write_relation_content(
                _needs_llm_pairs, input_text,
                chunk_size=_rel_batch_size,
            )

        # 7c: Per-pair fallback for batch misses
        _pair_keys = {id(p): _pair_key(p[0], p[1]) for p in _needs_llm_pairs}
        _missing_pairs = [
            p for p in _needs_llm_pairs
            if _pair_keys[id(p)] not in batch_rel_results
            or len(batch_rel_results.get(_pair_keys[id(p)], "")) < _MIN_RELATION_CONTENT_LEN
        ]
        _fallback_rels: List[Dict[str, str]] = []
        if _missing_pairs:
            def _write_one_relation(pair: Tuple[str, str]) -> Optional[Dict[str, str]]:
                a, b = pair
                content = self.llm_client.write_relation_content(a, b, input_text)
                if content:
                    return {"entity1_name": a, "entity2_name": b, "content": content}
                return None

            _fallback_rels = _parallel_map(
                _missing_pairs, _write_one_relation,
                n_workers=4, thread_prefix="extract-rcontent",
            )

        # Assemble final list: batch > fast-path > per-pair fallback
        extracted_relations = []
        covered_keys = set()
        for p in relation_pairs:
            key = _pair_key(p[0], p[1])
            content = batch_rel_results.get(key, "")
            if not content or len(content) < _MIN_RELATION_CONTENT_LEN:
                content = _fast_rel_results.get(key, "")
            if content and len(content) >= _MIN_RELATION_CONTENT_LEN:
                extracted_relations.append({"entity1_name": p[0], "entity2_name": p[1], "content": content})
                covered_keys.add(key)

        for r in _fallback_rels:
            key = _pair_key(r["entity1_name"], r["entity2_name"])
            if key not in covered_keys:
                extracted_relations.append(r)
                covered_keys.add(key)

        if verbose or verbose_steps:
            _fast_n = len(_fast_rel_results)
            _batch_n = len(batch_rel_results)
            _fallback_n = len(_fallback_rels)
            _detail_parts = []
            if _fast_n:
                _detail_parts.append(f"快速{_fast_n}条")
            if _batch_n:
                _detail_parts.append(f"批量{_batch_n}条")
            if _fallback_n:
                _detail_parts.append(f"回退{_fallback_n}条")
            _detail = " + ".join(_detail_parts) or f"{len(extracted_relations)}条"
            _elapsed6 = _time.time() - _t
            wprint_info(f"【步骤7】关系内容写作｜{_detail} → {len(extracted_relations)}条完成｜{_elapsed6:.1f}s")
        else:
            _elapsed6 = _time.time() - _t
        _record_timing("step7_relation_content", _elapsed6)

        _check_control()
        # ==============================================================
        _t = _time.time()
        valid_relations = []
        for r in extracted_relations:
            if _validate_relation(r["entity1_name"], r["entity2_name"],
                                  r["content"], entity_name_set):
                valid_relations.append(r)

        if verbose or verbose_steps:
            rejected_r = len(extracted_relations) - len(valid_relations)
            if rejected_r:
                wprint_info(f"【步骤8】关系质量门｜{rejected_r}条被过滤，{len(valid_relations)}条通过")
        _elapsed7 = _time.time() - _t
        _record_timing("step8_relation_quality", _elapsed7)

        _progress(0.95, f"{_win} · 完成",
                   f"{len(extracted_entities)} 实体, {len(valid_relations)} 关系")

        return extracted_entities, valid_relations

    # ------------------------------------------------------------------
    # Helper: resolve LLM-returned entity name to known entity name
    # ------------------------------------------------------------------

    @staticmethod
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

    @staticmethod
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
