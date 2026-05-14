"""
Pipeline worker threads for step9 (entity alignment) and step10 (relation alignment).

Extracted from orchestrator.py.  The workers receive the ``processor`` (i.e. the
``TemporalMemoryGraphProcessor`` instance) so they can call mixin methods such
as ``_align_entities``, ``_align_relations``, etc. without importing
orchestrator.py (avoiding circular imports).
"""
import logging
import threading
import time
from typing import Any, Callable, Optional

from core.utils import (
    clear_parallel_log_context,
    set_pipeline_role,
    set_window_label,
    wprint_info,
)
from core.log import info as _log_info

from .pipeline_state import (
    poll_control,
    record_window_error,
    safe_prefetch_submit,
    safe_progress,
    signal_control_stop,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Slot management (step1-5 window slots)
# ------------------------------------------------------------------

def acquire_window_slot(processor) -> None:
    """Acquire a concurrency slot and bump the active-main-pipeline counter."""
    processor._window_slot.acquire()
    with processor._runtime_lock:
        processor._active_main_pipeline_windows += 1


def release_window_slot(processor) -> None:
    processor._window_slot.release()
    with processor._runtime_lock:
        processor._active_main_pipeline_windows = max(0, processor._active_main_pipeline_windows - 1)


# ------------------------------------------------------------------
# Extraction job wrapper
# ------------------------------------------------------------------

def run_extraction_job(
    processor,
    new_episode,
    input_text: str,
    document_name: str,
    verbose: bool = True,
    verbose_steps: bool = True,
    event_time=None,
    control_check_fn=None,
):
    with processor._runtime_lock:
        processor._active_window_extractions += 1
        processor._peak_window_extractions = max(
            processor._peak_window_extractions,
            processor._active_window_extractions,
        )
    try:
        return processor._process_extraction(
            new_episode,
            input_text,
            document_name,
            verbose=verbose,
            verbose_steps=verbose_steps,
            event_time=event_time,
            control_check_fn=control_check_fn,
        )
    finally:
        with processor._runtime_lock:
            processor._active_window_extractions = max(0, processor._active_window_extractions - 1)
        release_window_slot(processor)


# ------------------------------------------------------------------
# Step-9 worker
# ------------------------------------------------------------------

def run_step9_worker(processor, state, start_chunk, total_chunks, doc_name,
                     verbose, verbose_steps, event_time, progress_callback,
                     step9_chunk_done_callback,
                     RememberControlFlow):
    """Step-9 worker thread: entity alignment, chained across windows."""
    _emb_available = bool(processor.storage.embedding_client and processor.storage.embedding_client.is_available())
    for i in range(state.N):
        state.extract_done[i].wait()
        _action = poll_control(state, None)
        if _action:
            signal_control_stop(state, _action, i, set_extract=False, set_step9=True, set_step10=True)
            break
        set_window_label(f"W{start_chunk + i + 1}/{total_chunks}")
        set_pipeline_role("步骤9")
        _er = state.extract_results[i]
        emb_prefetch_future = None
        if _er is not None:
            _ents, _ = _er
            if _ents and _emb_available:
                emb_prefetch_future = safe_prefetch_submit(
                    state,
                    processor.entity_processor.encode_entities_for_candidate_table,
                    _ents,
                )
        if i > 0:
            state.step9_done_ev[i - 1].wait()
        _action = poll_control(state, None)
        if _action:
            signal_control_stop(state, _action, i, set_extract=False, set_step9=True, set_step10=True)
            break
        with processor._runtime_lock:
            processor._active_step9 += 1
        _already_versioned = set()
        _t_step9_start = time.time()
        try:
            mc = state.episodes[i]
            _success = False
            if _er is None:
                _upstream = state.window_failures[i]
                if _upstream is not None:
                    _stage, _exc = _upstream
                    if verbose or verbose_steps:
                        wprint_info(f"【步骤9】跳过｜上游｜{_stage} {_exc}")
                    continue
                raise RuntimeError(
                    f"step9 skipped for window {start_chunk + i}: extract result is None (extraction failed)"
                )
            ents, rels = _er
            if verbose:
                wprint_info("【步骤9】实体｜就绪｜本窗1–5完成或缓存")
            elif verbose_steps:
                wprint_info("【步骤9】实体｜开始｜前置1–5已就绪")
            _wi = start_chunk + i
            _g_lo = _wi / total_chunks
            _g_hi = (_wi + 1) / total_chunks
            _span = _g_hi - _g_lo
            _pr_step9 = (_g_lo + _span * (8.0 / 10.0), _g_lo + _span * (9.0 / 10.0))
            ar = processor._align_entities(
                ents, rels, mc, state.input_texts[i], doc_name,
                verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
                progress_callback=lambda p, l, m: safe_progress(progress_callback, p, l, m, "step9"),
                progress_range=_pr_step9,
                window_index=start_chunk + i, total_windows=total_chunks,
                entity_embedding_prefetch=emb_prefetch_future,
                already_versioned_family_ids=_already_versioned,
                window_timings_ref=state.window_timings[i],
                control_check_fn=lambda: poll_control(state, None),
            )
            state.align_results[i] = ar
            _success = True
            _step9_elapsed = time.time() - _t_step9_start
            state.window_timings[i]["step9"] = _step9_elapsed
            if verbose or verbose_steps:
                wprint_info(f"【步骤9】完成｜{_step9_elapsed:.1f}s")
        except Exception as e:
            if isinstance(e, RememberControlFlow):
                signal_control_stop(state, e.remember_control_action, i, set_extract=False, set_step9=True, set_step10=True)
            if record_window_error(state, "step9", i, e):
                logger.error("step9 window %d error: %s", i, e, exc_info=True)
        finally:
            with processor._runtime_lock:
                processor._active_step9 = max(0, processor._active_step9 - 1)
            state.step9_done_ev[i].set()
            # Free raw extraction data now that step9 has consumed it
            # NOTE: Do NOT nullify state.input_texts[i] here — step10 still needs it
            if _success:
                state.extract_results[i] = None
            if _success and step9_chunk_done_callback:
                step9_chunk_done_callback(start_chunk + i + 1)
            clear_parallel_log_context()


# ------------------------------------------------------------------
# Step-10 worker
# ------------------------------------------------------------------

def run_step10_worker(processor, state, start_chunk, total_chunks, doc_name,
                      verbose, verbose_steps, event_time, progress_callback,
                      chunk_done_callback,
                      RememberControlFlow):
    """Step-10 worker thread: relation alignment, chained across windows."""
    for i in range(state.N):
        state.step9_done_ev[i].wait()
        _action = poll_control(state, None)
        if _action:
            signal_control_stop(state, _action, i, set_extract=False, set_step9=False, set_step10=True)
            break
        set_window_label(f"W{start_chunk + i + 1}/{total_chunks}")
        set_pipeline_role("步骤10")
        ar = state.align_results[i]
        step10_inputs_cache = None
        rel_prefetch_future = None
        if ar is not None:
            try:
                step10_inputs_cache = processor._build_step10_relation_inputs_from_align_result(ar)
                _ri, _eid, _, _ = step10_inputs_cache
                if i > 0 and _ri:
                    rel_prefetch_future = safe_prefetch_submit(
                        state,
                        processor.relation_processor.build_relations_by_pair_from_inputs,
                        _ri,
                        _eid,
                    )
            except Exception as exc:
                wprint_info(f"  │  step10 输入构建失败: {exc}")
                step10_inputs_cache = None
                rel_prefetch_future = None
        if i > 0:
            state.step10_done_ev[i - 1].wait()
        _action = poll_control(state, None)
        if _action:
            signal_control_stop(state, _action, i, set_extract=False, set_step9=False, set_step10=True)
            break
        prepared_relations_by_pair = None
        if rel_prefetch_future is not None:
            try:
                prepared_relations_by_pair, _ = rel_prefetch_future.result()
            except Exception as exc:
                wprint_info(f"  │  关系预取结果获取失败: {exc}")
                prepared_relations_by_pair = None
        with processor._runtime_lock:
            processor._active_step10 += 1
        _t_step10_start = time.time()
        _success = False
        _window_has_entities = False
        try:
            if ar is None:
                _upstream = state.window_failures[i]
                if _upstream is not None:
                    _stage, _exc = _upstream
                    if verbose or verbose_steps:
                        wprint_info(f"【步骤10】跳过｜上游｜{_stage} {_exc}")
                    continue
                raise RuntimeError(
                    f"step9 result for window {start_chunk + i} is None"
                )
            mc = state.episodes[i]
            _wi = start_chunk + i
            _g_lo = _wi / total_chunks
            _g_hi = (_wi + 1) / total_chunks
            _span = _g_hi - _g_lo
            _pr_step10 = (_g_lo + _span * (9.0 / 10.0), _g_hi)
            processed_rels = processor._align_relations(
                ar, mc, state.input_texts[i], doc_name,
                verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
                progress_callback=lambda p, l, m: safe_progress(progress_callback, p, l, m, "step10"),
                progress_range=_pr_step10,
                window_index=start_chunk + i, total_windows=total_chunks,
                prepared_relations_by_pair=prepared_relations_by_pair,
                step10_inputs_cache=step10_inputs_cache,
                window_timings_ref=state.window_timings[i],
                control_check_fn=lambda: poll_control(state, None),
            )
            state.step10_results[i] = processed_rels
            _success = True
            _window_has_entities = bool(ar.unique_entities)
            _step10_elapsed = time.time() - _t_step10_start
            state.window_timings[i]["step10"] = _step10_elapsed
            if verbose or verbose_steps:
                wprint_info(f"【步骤10】完成｜{_step10_elapsed:.1f}s")

            # Phase C-2: Record Episode -> Relation MENTIONS
            if processed_rels:
                try:
                    _t_mentions = time.time()
                    _rel_abs_ids = list(set(
                        r.absolute_id for r in processed_rels if r.absolute_id
                    ))
                    if _rel_abs_ids:
                        processor.storage.save_episode_mentions(
                            mc.absolute_id, _rel_abs_ids,
                            target_type="relation",
                        )
                        if verbose or verbose_steps:
                            wprint_info(f"【步骤10】MENTIONS｜Relation｜{len(_rel_abs_ids)}条")
                    state.window_timings[i]["step10-relation_mentions"] = time.time() - _t_mentions
                except Exception as _me:
                    logger.warning("Relation MENTIONS 记录失败: %s", _me)

            if _window_has_entities:
                safe_progress(progress_callback,
                    _g_lo + _span * (9.0 / 10.0 + 0.9 / 10.0),
                    f"窗口 {start_chunk + i + 1}/{total_chunks} · 步骤10/10: 孤立实体处理", "", "step10")
                try:
                    _t_orphan = time.time()
                    _orphan_count = processor._cleanup_orphaned_entities(
                        ar.unique_entities,
                        verbose=verbose or verbose_steps,
                        window_text=state.input_texts[i],
                        all_entity_names=[e.name for e in ar.unique_entities] if ar.unique_entities else [],
                        episode_id=getattr(mc, 'cache_id', ''),
                        source_document=doc_name,
                    )
                    state.window_timings[i]["step10-orphan_cleanup"] = time.time() - _t_orphan
                    if _orphan_count > 0:
                        _window_has_entities = bool(ar.unique_entities) and _orphan_count < len(ar.unique_entities)
                except Exception as _oe:
                    logger.warning("孤立实体清理失败: %s", _oe)
        except Exception as e:
            if isinstance(e, RememberControlFlow):
                signal_control_stop(state, e.remember_control_action, i, set_extract=False, set_step9=False, set_step10=True)
            if record_window_error(state, "step10", i, e):
                logger.error("step10 window %d error: %s", i, e, exc_info=True)
        finally:
            with processor._runtime_lock:
                processor._active_step10 = max(0, processor._active_step10 - 1)
            state.step10_done_ev[i].set()
            # Free alignment data now that step10 has consumed it
            if _success:
                state.align_results[i] = None
                state.input_texts[i] = None
                state.episodes[i] = None
            if _success and chunk_done_callback:
                chunk_done_callback(start_chunk + i + 1)
            if _success and not _window_has_entities:
                wprint_info("提示: step10 完成但本窗无实体，仍已计入进度（避免断点卡死）")
            clear_parallel_log_context()


# ------------------------------------------------------------------
# Timing summary
# ------------------------------------------------------------------

def summarize_window_timings(window_timings):
    """Log timing summary across all windows."""
    _all_steps = ["step1", "step2-8", "step9", "step10"]
    _step_labels = {"step1": "1-缓存", "step2-8": "2-8-抽取", "step9": "9-实体对齐", "step10": "10-关系对齐"}
    _sub_step_labels = {
        "step2_entity_extract": "2-实体提取",
        "step3_entity_dedup": "3-实体去重",
        "step4_entity_content": "4-实体内容",
        "step5_entity_quality": "5-实体质量门",
        "step6_relation_discovery": "6-关系发现",
        "step7_relation_content": "7-关系内容",
        "step8_relation_quality": "8-关系质量门",
        "step9-process_entities": "9a-实体处理",
        "step9-dedup_merge": "9b-同名去重",
        "step9-resolve_missing_names": "9c-名称解析",
        "step9-convert_to_ids": "9d-ID转换",
        "step9-entity_mentions": "9e-Entity记录",
        "step10-input_build": "10a-输入构建",
        "step10-process_relations": "10b-关系处理",
        "step10a-db_read_relations": "  b1-DB读关系",
        "step10a-db_fetch_entities": "  b2-DB取实体",
        "step10a-embedding_prep": "  b3-Embedding",
        "step10b-process_loop": "  b4-处理循环",
        "step10c-refresh_edges": "10c-边刷新",
        "step10d-dream_corroboration": "10d-Dream佐证",
        "step10-relation_mentions": "10e-Relation记录",
        "step10-orphan_cleanup": "10f-孤立处理",
    }
    _step_totals = {s: 0.0 for s in _all_steps}
    _sub_totals = {k: 0.0 for k in _sub_step_labels}
    for _wt in window_timings:
        for _s in _all_steps:
            _step_totals[_s] += _wt.get(_s, 0.0)
        for _sk in _sub_step_labels:
            _sub_totals[_sk] += _wt.get(_sk, 0.0)
    _total_elapsed = sum(_step_totals.values())
    if _total_elapsed > 0:
        _timing_detail = " | ".join(
            f"{_step_labels[s]}:{_step_totals[s]:.1f}s"
            for s in _all_steps if _step_totals[s] > 0
        )
        _log_info("Remember", f"计时汇总｜共{_total_elapsed:.1f}s｜{_timing_detail}")
        _active_subs = {k: v for k, v in _sub_totals.items() if v > 0.01}
        if _active_subs:
            _sub_detail = " | ".join(
                f"{_sub_step_labels[k]}:{v:.1f}s"
                for k, v in sorted(_active_subs.items(), key=lambda x: -x[1])
            )
            _log_info("Remember", f"子步骤明细｜{_sub_detail}")
