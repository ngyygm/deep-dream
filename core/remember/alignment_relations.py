"""Extraction pipeline mixin: relation alignment (step10).

Extracted from alignment.py to keep each file under 800 lines.
"""
from __future__ import annotations

import time as _time
from datetime import datetime
from typing import Dict, Optional

from core.models import Episode
from core.debug_log import log as dbg, _ENABLED as _dbg_enabled
from core.utils import wprint_info
from core.llm.client import LLM_PRIORITY_ALIGN
from .helpers import _AlignResult


class _RelationAlignMixin:
    """Relation alignment, verification, and serial window processing.

    Mixed into _PipelineExtractionMixin alongside entity alignment and
    sub-concern mixins.  All attributes are resolved via ``self`` on the
    combined class (llm_client, storage, relation_processor, etc.).
    """

    # =========================================================================
    # 步骤10：关系对齐（写存储，串行跨窗口）
    # =========================================================================

    def _align_relations(self, align_result: _AlignResult,
                         new_episode: Episode, input_text: str,
                         document_name: str, verbose: bool = True,
                         verbose_steps: bool = True,
                         event_time: Optional[datetime] = None,
                         progress_callback=None,
                         progress_range: tuple = (0.75, 1.0),
                         window_index: int = 0,
                         total_windows: int = 1,
                         prepared_relations_by_pair=None,
                         step10_inputs_cache=None,
                         window_timings_ref: Optional[Dict[str, float]] = None,
                         control_check_fn=None,
                         ):

        p_lo, p_hi = progress_range
        _win_label = f"窗口 {window_index + 1}/{total_windows}"
        _step_size = p_hi - p_lo

        self.llm_client._priority_local.priority = LLM_PRIORITY_ALIGN
        if verbose:
            wprint_info("【步骤10】关系｜开始｜对齐写入")
        elif verbose_steps:
            wprint_info("【步骤10】关系｜开始｜")
        _ent_count = len(align_result.unique_entities) if align_result and align_result.unique_entities else 0
        _rel_count = len(align_result.pending_relations) if align_result and align_result.pending_relations else 0
        if progress_callback:
            progress_callback(p_lo,
                f"{_win_label} · 步骤10/10: 关系对齐 · 开始",
                f"{_ent_count}个实体, {_rel_count}条待处理关系")

        self.llm_client._current_distill_step = "07_relation_alignment"

        if control_check_fn:
            action = control_check_fn()
            if action:
                from core.remember.orchestrator import RememberControlFlow
                raise RememberControlFlow(action)
            _cancel_bool_fn = lambda: control_check_fn() is not None
            self.llm_client.set_cancel_check(_cancel_bool_fn)

        unique_entities = align_result.unique_entities

        if step10_inputs_cache is not None:
            relation_inputs, entity_name_to_id, unique_pending_relations, all_pending_relations = step10_inputs_cache
        else:
            _t_inputs = _time.time()
            relation_inputs, entity_name_to_id, unique_pending_relations, all_pending_relations = (
                self._build_step10_relation_inputs_from_align_result(align_result)
            )
            if window_timings_ref is not None:
                window_timings_ref["step10-input_build"] = _time.time() - _t_inputs

        if verbose:
            duplicate_count = len(all_pending_relations) - len(unique_pending_relations)
            if duplicate_count > 0:
                wprint_info(
                    f"【步骤10】关系｜待处理｜{len(all_pending_relations)}→去重{len(unique_pending_relations)}"
                )
            else:
                wprint_info(f"【步骤10】关系｜待处理｜{len(unique_pending_relations)}个")

        _upr_count = len(unique_pending_relations)
        if _upr_count == 0:
            if verbose:
                wprint_info("【步骤10】关系｜跳过｜无待处理")
        else:
            if verbose:
                wprint_info(
                    f"【步骤10】关系｜待处理｜去重{_upr_count}·原{len(all_pending_relations)}"
                )
        dbg(f"步骤10: 去重后待处理关系 {len(unique_pending_relations)} 个 (去重前 {len(all_pending_relations)} 个)")
        if _dbg_enabled:
            for _upr in unique_pending_relations:
                dbg(f"  待处理: '{_upr.get('entity1_name', '')}' <-> '{_upr.get('entity2_name', '')}' (e1_id={_upr.get('entity1_id', '?')}, e2_id={_upr.get('entity2_id', '?')})  content='{_upr.get('content', '')[:100]}'")

        _rel_done = [0]

        def _on_relation_pair_done(done, total):
            _rel_done[0] = done
            if progress_callback:
                frac = done / max(1, total)
                progress_callback(p_lo + _step_size * 0.05 + _step_size * 0.85 * frac,
                    f"{_win_label} · 步骤10/10: 关系对齐 ({done}/{total})",
                    f"关系对齐 {done}/{total}")

        if progress_callback:
            progress_callback(p_lo + _step_size * 0.01,
                f"{_win_label} · 步骤10/10: 关系输入构建（{len(unique_pending_relations)}条）", "")

        _t_rel_start = _time.time()
        all_processed_relations = self.relation_processor.process_relations_batch(
            relation_inputs,
            entity_name_to_id,
            new_episode.absolute_id,
            source_document=document_name,
            base_time=new_episode.event_time,
            # Conservative mode: serial (1 worker). Non-conservative: llm_threads for parallel processing.
            max_workers=(1 if getattr(self, "remember_alignment_conservative", False) else self.llm_threads),
            on_relation_done=_on_relation_pair_done,
            # detail 模式常开 verbose、关 verbose_steps：避免逐条 [关系操作] 刷屏
            verbose_relation=bool(verbose and verbose_steps),
            prepared_relations_by_pair=prepared_relations_by_pair,
            window_timings_ref=window_timings_ref,
            source_text=input_text,
        )
        _t_rel_elapsed = _time.time() - _t_rel_start
        if window_timings_ref is not None:
            window_timings_ref["step10-process_relations"] = _t_rel_elapsed
        if verbose or verbose_steps:
            wprint_info(f"【步骤10】process_relations_batch｜{_t_rel_elapsed:.1f}s｜{len(all_processed_relations)}个关系")

        if verbose:
            if not all_processed_relations:
                wprint_info("【步骤10】关系｜小结｜无新")
            else:
                wprint_info(f"【步骤10】关系｜小结｜{len(all_processed_relations)}个")
        elif verbose_steps:
            wprint_info("【步骤10】关系｜完成｜")

        if progress_callback:
            progress_callback(p_lo + _step_size * 0.92,
                f"{_win_label} · 步骤10/10: Episode-Relation关联记录", "")

        if verbose:
            wprint_info("【窗口】流水｜结束｜")
        _final_ents = len(unique_entities)
        _final_rels = len(all_processed_relations)
        if verbose:
            if _final_ents == 0 and _final_rels == 0:
                wprint_info("【窗口】汇总｜空｜无新实体关系")
            else:
                wprint_info(
                    f"【窗口】汇总｜得｜实体{_final_ents} 关系{_final_rels}·待{len(unique_pending_relations)}"
                )
        elif verbose_steps:
            wprint_info(f"【窗口】汇总｜得｜实体{_final_ents} 关系{_final_rels}")
        dbg(f"窗口处理完成: {len(unique_entities)} 个实体, {len(all_processed_relations)} 个关系 (从 {len(unique_pending_relations)} 个待处理)")

        if progress_callback:
            progress_callback(p_hi,
                f"{_win_label} · 步骤10/10: 窗口完成",
                f"{len(unique_entities)} 个实体, {len(all_processed_relations)} 个关系")

        # Phase B+: 自动关系矛盾检测 — disabled (too expensive for auto pipeline)

        self.llm_client._current_distill_step = None
        self.llm_client._distill_task_id = None
        self.llm_client.clear_cancel_check()

        return all_processed_relations

