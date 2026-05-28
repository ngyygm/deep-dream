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

import time as _time
from typing import Any, Dict, List, Optional, Set, Tuple

from core.llm.client import (
    LLM_PRIORITY_STEP2,
    LLM_PRIORITY_STEP3,
    LLM_PRIORITY_STEP4,
    LLM_PRIORITY_STEP5,
)
from core.utils import wprint_info
from .helpers import _core_entity_name
from ._steps_helpers import (
    _pair_key, _get_shared_pool, _parallel_map,
    _normalize_and_dedup_entity_names, _validate_entity, _validate_relation,
    _prepare_prose_sentences, _ProseIndex, _build_entity_fallback_content,
    _MIN_ENTITY_CONTENT_LEN, _MIN_RELATION_CONTENT_LEN,
)


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
        early_entity_done_fn=None,
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

        extraction_client = self.extraction_client or self.llm_client

        def _with_llm_priority(client, priority: int, fn):
            if client is None:
                raise RuntimeError("LLM client is not configured; set llm.api_key, llm.base_url, and llm.model before running remember")
            previous = getattr(client._priority_local, "priority", None)
            client._priority_local.priority = priority
            try:
                return fn()
            finally:
                if previous is None:
                    try:
                        del client._priority_local.priority
                    except AttributeError:
                        pass
                else:
                    client._priority_local.priority = previous

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
        raw_names, ent_refine = _with_llm_priority(
            extraction_client,
            LLM_PRIORITY_STEP2,
            lambda: extraction_client.extract_entities(
                input_text, max_refine_rounds=self.entity_rounds
            ),
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
                _fallback_cache[name] = _build_entity_fallback_content(name, _get_prose_index())
            return _fallback_cache[name]

        _t = _time.time()

        # Lazy prose index — only build when fallback content is needed
        _prose_index = None
        _prose_sentences = None

        def _get_prose_index():
            nonlocal _prose_index, _prose_sentences
            if _prose_index is None:
                _prose_sentences = _prepare_prose_sentences(input_text)
                _prose_index = _ProseIndex(_prose_sentences)
            return _prose_index

        # All entities go through LLM for proper concept descriptions
        _fast_path_names: List[str] = []
        _needs_llm_names: List[str] = list(entity_names)

        # ── Launch Step 6 (relation discovery) in parallel with Step 4 ──
        # Step 6 only needs entity names (available after Step 3), not entity content.
        _step6_future = None
        _step6_raw_pairs: List = []
        _step6_stats: Dict = {}
        _step6_entity_names = list(entity_names)  # snapshot for thread safety

        if len(entity_names) >= 2:
            def _run_step6():
                _t6 = _time.time()
                _raw, _stats = _with_llm_priority(
                    extraction_client,
                    LLM_PRIORITY_STEP3,
                    lambda: extraction_client.discover_relations(
                        _step6_entity_names, input_text, max_refine_rounds=self.relation_rounds
                    ),
                )
                return _raw, _stats, _time.time() - _t6

            _step6_future = _get_shared_pool(1).submit(_run_step6)

        # 4a: Batch write only for entities needing LLM content
        batch_results: Dict[str, str] = {}
        if _needs_llm_names:
            _batch_chunk_size = getattr(self, 'remember_entity_content_batch_size', 10)
            _step4_workers = getattr(self, 'llm_threads', 1)
            if _step6_future is not None and _step4_workers > 1:
                # Keep one LLM slot available for relation discovery. This preserves
                # the original prompts and steps while avoiding a full-slot step4
                # batch blocking the next high-value extraction call.
                _step4_workers = max(1, _step4_workers - 1)
            _t4_batch = _time.time()
            batch_results = _with_llm_priority(
                self.llm_client,
                LLM_PRIORITY_STEP4,
                lambda: self.llm_client.batch_write_entity_content(
                    _needs_llm_names, input_text,
                    chunk_size=_batch_chunk_size,
                    max_workers=_step4_workers,
                ),
            )
            _record_timing("step4_entity_content_batch_llm", _time.time() - _t4_batch)

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
                _t4_fallback_llm = _time.time()
                def _write_one_entity(name: str) -> Dict[str, str]:
                    content = _with_llm_priority(
                        self.llm_client,
                        LLM_PRIORITY_STEP4,
                        lambda: self.llm_client.write_entity_content(name, input_text),
                    )
                    if not content or len(content) < _min_content_len:
                        content = _cached_fallback(name)
                    return {"name": name, "content": content}

                _entity_fallback = lambda name, exc: {"name": name, "content": _cached_fallback(name)}

                _fallback_results = _parallel_map(
                    _missing_names, _write_one_entity,
                    fallback_fn=_entity_fallback,
                    n_workers=_step4_workers, thread_prefix="extract-econtent",
                )
                for e in _fallback_results:
                    batch_results[e["name"]] = e["content"]
                _record_timing("step4_entity_content_fallback_llm", _time.time() - _t4_fallback_llm)

        # Merge fast-path results
        for name in _fast_path_names:
            batch_results[name] = _fallback_cache[name]

        # Assemble final list preserving entity_names order
        _t4_code_fallback = _time.time()
        _code_fallback_used = False
        extracted_entities = []
        for name in entity_names:
            content = batch_results.get(name, "")
            if not content or len(content) < _MIN_ENTITY_CONTENT_LEN:
                content = _cached_fallback(name)
                _code_fallback_used = True
            extracted_entities.append({"name": name, "content": content})
        if _code_fallback_used:
            _record_timing("step4_entity_content_code_fallback", _time.time() - _t4_code_fallback)

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

        _elapsed4q = _time.time() - _t
        _record_timing("step5_entity_quality", _elapsed4q)
        if verbose or verbose_steps:
            _q_rejected = len(extracted_entities) - len(valid_entities) + len(rejected_entities)
            if not (verbose or verbose_steps) or not rejected:
                wprint_info(f"【步骤5】实体质量门｜{_elapsed4q:.1f}s｜{len(valid_entities)}个通过")
        entity_name_list = [e["name"] for e in extracted_entities]
        entity_name_set = set(entity_name_list)

        _progress(0.50, f"{_win} · 步骤5: 实体质量门", f"{len(extracted_entities)} 个有效实体")

        if early_entity_done_fn:
            early_entity_done_fn(valid_entities)

        _check_control()
        # ==============================================================
        # Step 6: Relation discovery (may have been launched in parallel with Step 4)
        # ==============================================================
        _t = _time.time()
        relation_pairs = []
        if _step6_future is not None:
            # Collect result from parallel step 6
            _progress(0.53, f"{_win} · 步骤6: 关系发现（强模型）", "等待结果")
            _t6_wait = _time.time()
            try:
                _step6_raw_pairs, _step6_stats, _step6_wall = _step6_future.result()
            except Exception:
                _step6_raw_pairs, _step6_stats, _step6_wall = [], {}, 0.0
            _step6_future = None
            _record_timing("step6_relation_wait", _time.time() - _t6_wait)

            # Normalize pairs using the entity name set from step 3
            _t6_norm = _time.time()
            seen_pairs = set()
            _name_lookup = self._build_name_lookup(set(_step6_entity_names))
            def _add_pairs(raw_list):
                added = 0
                for a, b in raw_list:
                    a = self._resolve_entity_name(a, set(_step6_entity_names), _lookup=_name_lookup)
                    b = self._resolve_entity_name(b, set(_step6_entity_names), _lookup=_name_lookup)
                    if a and b and a != b:
                        pair_key = _pair_key(a, b)
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            relation_pairs.append((a, b))
                            added += 1
                return added

            _initial_count = _add_pairs(_step6_raw_pairs)
            _refine_added = len(relation_pairs) - _initial_count
            _elapsed5 = _step6_wall
            _record_timing("step6_relation_normalize", _time.time() - _t6_norm)

            if verbose or verbose_steps:
                _ref_tag = f" +精炼{_refine_added}对" if _refine_added else ""
                wprint_info(f"【步骤6】关系对发现（并行）｜{_initial_count}对初始{_ref_tag} → 总{len(relation_pairs)}对｜{_elapsed5:.1f}s (wall)")
        else:
            _elapsed5 = 0.0
        _record_timing("step6_relation_discovery", _elapsed5)

        _progress(0.60, f"{_win} · 步骤6: 关系发现完成", f"{len(relation_pairs)} 对")

        _check_control()
        # ==============================================================
        _progress(0.65, f"{_win} · 步骤7: 关系内容写作", f"开始写 {len(relation_pairs)} 对关系")
        _t = _time.time()

        # 7a: All relation pairs go through LLM for proper descriptions
        _fast_rel_results: Dict[Tuple[str, str], str] = {}
        _needs_llm_pairs: List[Tuple[str, str]] = list(relation_pairs)

        # 7b: Batch write for pairs needing LLM content
        batch_rel_results: Dict[Tuple[str, str], str] = {}
        if _needs_llm_pairs:
            _rel_batch_size = getattr(self, 'remember_relation_content_batch_size', 20)
            _step7_workers = getattr(self, 'llm_threads', 1)
            _t7_batch = _time.time()
            batch_rel_results = _with_llm_priority(
                self.llm_client,
                LLM_PRIORITY_STEP5,
                lambda: self.llm_client.batch_write_relation_content(
                    _needs_llm_pairs, input_text,
                    chunk_size=_rel_batch_size,
                    max_workers=_step7_workers,
                ),
            )
            _record_timing("step7_relation_content_batch_llm", _time.time() - _t7_batch)

        # 7c: Per-pair fallback for batch misses
        _pair_keys = {id(p): _pair_key(p[0], p[1]) for p in _needs_llm_pairs}
        _missing_pairs = [
            p for p in _needs_llm_pairs
            if _pair_keys[id(p)] not in batch_rel_results
            or len(batch_rel_results.get(_pair_keys[id(p)], "")) < _MIN_RELATION_CONTENT_LEN
        ]
        _fallback_rels: List[Dict[str, str]] = []
        if _missing_pairs:
            _t7_fallback_llm = _time.time()
            def _write_one_relation(pair: Tuple[str, str]) -> Optional[Dict[str, str]]:
                a, b = pair
                content = _with_llm_priority(
                    self.llm_client,
                    LLM_PRIORITY_STEP5,
                    lambda: self.llm_client.write_relation_content(a, b, input_text),
                )
                if content:
                    return {"entity1_name": a, "entity2_name": b, "content": content}
                return None

            _fallback_rels = _parallel_map(
                _missing_pairs, _write_one_relation,
                n_workers=_step7_workers, thread_prefix="extract-rcontent",
            )
            _record_timing("step7_relation_content_fallback_llm", _time.time() - _t7_fallback_llm)

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

        _progress(0.80, f"{_win} · 步骤7: 关系内容写作完成", f"{len(extracted_relations)} 条关系")

        _check_control()
        # ==============================================================
        _progress(0.85, f"{_win} · 步骤8: 关系质量门", "开始")
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
        if verbose or verbose_steps:
            wprint_info(f"【步骤8】关系质量门｜{_elapsed7:.1f}s｜{len(valid_relations)}条通过")

        _progress(0.90, f"{_win} · 步骤8: 关系质量门完成", f"{len(valid_relations)} 条有效关系")

        _progress(0.95, f"{_win} · 完成",
                   f"{len(extracted_entities)} 实体, {len(valid_relations)} 关系")

        self.llm_client.clear_cancel_check()
        if self.extraction_client_enabled:
            extraction_client.clear_cancel_check()
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
