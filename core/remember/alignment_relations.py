"""Extraction pipeline mixin: relation alignment (step10) + orphan cleanup."""
from __future__ import annotations

import time as _time
from datetime import datetime
from typing import Dict, List, Optional

from core.models import Episode
from core.debug_log import log as dbg, _ENABLED as _dbg_enabled
from core.utils import wprint_info, wprint_debug
from core.llm.client import LLM_PRIORITY_ALIGN
from core.llm.prompts import RELATION_DISCOVER_SYSTEM, ORPHAN_RECOVERY_USER
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
            def _cancel_bool_fn():
                return control_check_fn() is not None
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


class _OrphanMixin:
    """Orphan entity handling: cleanup, fallback cooccurrence relations, and LLM-based recovery."""

    def _cleanup_orphaned_entities(
        self,
        saved_entities: list,
        verbose: bool = False,
        window_text: str = "",
        all_entity_names: Optional[List[str]] = None,
        episode_id: str = "",
        source_document: str = "",
        progress_callback=None,
    ) -> int:
        """处理孤立实体：先尝试补救（找关系），再为无法补救的创建兜底共现关系。

        在 step10（关系存储）完成后调用。此时关系已经全部写入，
        可以准确判断哪些实体是孤立的。

        补救流程：对孤立实体调用 LLM 寻找与其他实体的关系，写入后重新检查度数，
        仍然为 0 的创建兜底共现关系（不再删除）。

        Args:
            saved_entities: step9 存入的实体列表（_AlignResult.unique_entities）
            verbose: 是否打印日志
            window_text: 当前窗口文本（补救用）
            all_entity_names: 当前窗口所有实体名称（补救用）
            episode_id: 当前 episode ID（补救写关系时使用）
            source_document: 来源文档名（补救写关系时使用）

        Returns:
            删除的孤立实体数量（始终为 0，不再删除）
        """
        if not saved_entities:
            return 0

        new_family_ids = [e.family_id for e in saved_entities if hasattr(e, 'family_id') and e.family_id]
        if not new_family_ids:
            return 0

        # 批量查询度数（关系数）
        batch_fn = getattr(self.storage, 'batch_get_entity_degrees', None)
        if batch_fn is None:
            return 0

        try:
            degree_map = batch_fn(new_family_ids)
        except Exception:
            return 0

        # 收集度数为 0 的实体（无任何关系）
        orphan_fids = [fid for fid, deg in degree_map.items() if deg == 0]
        if not orphan_fids:
            if progress_callback:
                progress_callback(1.0, "无孤立实体", f"{len(new_family_ids)}个实体均有关系", "step10")
            return 0

        # 区分「全新实体」和「对齐到已有实体的更新」
        # 批量查询版本数：版本数 > 1 说明实体在本次处理前就已存在
        version_counts = {}
        try:
            version_counts = self.storage.get_entity_version_counts(orphan_fids)
        except Exception:
            pass  # 查询失败则保守不删

        # 只处理真正全新创建的孤立实体（版本数 == 1 且无关系）
        truly_new_orphans = [fid for fid in orphan_fids
                             if version_counts.get(fid, 1) <= 1]

        if not truly_new_orphans:
            if progress_callback:
                progress_callback(1.0, "无孤立实体", "均非全新实体", "step10")
            return 0

        # ---- 补救阶段：尝试为孤立实体找关系 ----
        recovered = 0
        if window_text and all_entity_names and truly_new_orphans:
            if progress_callback:
                progress_callback(0.2, "补救中", f"{len(truly_new_orphans)}个孤立实体 · 调用LLM找关系", "step10")
            recovered = self._recover_orphan_relations(
                truly_new_orphans, saved_entities, all_entity_names,
                window_text, episode_id, source_document, verbose,
            )

        # 补救后重新查询度数，只删除仍然孤立的
        if recovered > 0:
            try:
                degree_map = batch_fn(truly_new_orphans)
                truly_new_orphans = [fid for fid, deg in degree_map.items() if deg == 0]
            except Exception:
                pass  # 查询失败则保守不删

        if not truly_new_orphans:
            if progress_callback:
                progress_callback(1.0, "补救完成", f"成功补救{recovered}个实体", "step10")
            return 0

        # ---- 兜底阶段：为仍然孤立的实体创建共现关系 ----
        if progress_callback:
            progress_callback(0.6, "创建兜底关系", f"为{len(truly_new_orphans)}个未补救实体创建共现关系", "step10")
        _fallback_count = self._create_fallback_cooccurrence_relations(
            truly_new_orphans, saved_entities,
            episode_id, source_document, verbose,
        )

        if progress_callback:
            progress_callback(1.0, "完成",
                f"补救{recovered}个 · 兜底{_fallback_count}个",
                "step10")

        return 0  # 不再删除孤立实体

    def _create_fallback_cooccurrence_relations(
        self,
        orphan_fids: List[str],
        saved_entities: list,
        episode_id: str,
        source_document: str,
        verbose: bool,
    ) -> int:
        """为孤立实体创建兜底共现关系，确保每个实体至少有一个关系链接。"""
        if not orphan_fids:
            return 0

        # 构建 family_id → entity 映射
        fid_to_entity = {}
        for e in saved_entities:
            fid = getattr(e, 'family_id', None)
            if fid:
                fid_to_entity[fid] = e

        # 非孤立实体作为关系目标
        orphan_fid_set = set(orphan_fids)
        non_orphan_entities = [
            e for e in saved_entities
            if hasattr(e, 'family_id') and e.family_id
            and e.family_id not in orphan_fid_set
        ]

        if not non_orphan_entities:
            if verbose:
                wprint_info("  │  孤立实体兜底｜无法创建共现关系（无非孤立实体）")
            return 0

        relation_processor = getattr(self, 'relation_processor', None)
        if not relation_processor or not episode_id:
            return 0

        if verbose:
            _orphan_names = [getattr(fid_to_entity.get(fid), 'name', '?') for fid in orphan_fids]
            wprint_info(f"  │  孤立实体兜底｜为 {len(orphan_fids)} 个实体创建共现关系: {', '.join(_orphan_names[:5])}")

        fallback_count = 0
        for i, orphan_fid in enumerate(orphan_fids):
            orphan_entity = fid_to_entity.get(orphan_fid)
            if not orphan_entity:
                continue

            # 选择非孤立实体作为关系目标（轮询分配）
            target_entity = non_orphan_entities[i % len(non_orphan_entities)]

            try:
                rel = relation_processor._build_new_relation(
                    orphan_fid,
                    target_entity.family_id,
                    f"{orphan_entity.name}与{target_entity.name}在同一文本中出现",
                    episode_id,
                    entity1_name=orphan_entity.name,
                    entity2_name=target_entity.name,
                    verbose_relation=False,
                    source_document=source_document,
                    confidence=0.3,
                )
                if rel is not None:
                    relation_processor.storage.save_relation(rel)
                    fallback_count += 1
                    if verbose:
                        wprint_debug(f"  │  兜底共现关系: {orphan_entity.name} <-> {target_entity.name}")
            except Exception:
                pass

        if verbose:
            wprint_info(f"  │  孤立实体兜底｜{fallback_count}/{len(orphan_fids)} 个实体成功创建共现关系")
        return fallback_count

    def _recover_orphan_relations(
        self,
        orphan_fids: List[str],
        saved_entities: list,
        all_entity_names: List[str],
        window_text: str,
        episode_id: str,
        source_document: str,
        verbose: bool,
    ) -> int:
        """尝试为孤立实体找到并建立关系。

        Returns:
            成功补救的实体数量（度数从 0 变为 > 0）
        """
        # 构建 family_id → entity 映射
        fid_to_entity = {}
        for e in saved_entities:
            fid = getattr(e, 'family_id', None)
            if fid and fid in orphan_fids:
                fid_to_entity[fid] = e

        # 构建 entity_name → family_id 映射（所有实体，包括非孤儿）
        name_to_fid = {}
        for e in saved_entities:
            fid = getattr(e, 'family_id', None)
            name = getattr(e, 'name', None)
            if fid and name:
                name_to_fid[name] = fid

        orphan_names = [getattr(fid_to_entity[fid], 'name', '?') for fid in orphan_fids if fid in fid_to_entity]
        other_names = [n for n in all_entity_names if n not in orphan_names]

        if not orphan_names or not other_names:
            return 0

        if verbose:
            wprint_info(f"  │  孤立实体补救｜尝试为 {len(orphan_names)} 个实体找关系: {', '.join(orphan_names[:5])}")

        # 调用 LLM 寻找关系对
        try:
            user_prompt = ORPHAN_RECOVERY_USER.format(
                orphan_names="、".join(orphan_names),
                other_entity_names="、".join(other_names),
                window_text=window_text,
            )
            messages = [
                {"role": "system", "content": RELATION_DISCOVER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
            parsed, _ = self.llm_client.call_llm_until_json_parses(
                messages,
                parse_fn=self.llm_client._parse_pair_list,
                timeout=120,
            )
            raw_pairs = parsed or []
        except Exception as e:
            if verbose:
                wprint_debug(f"  │  孤立实体补救 LLM 调用失败: {e}")
            return 0

        if not raw_pairs:
            if verbose:
                wprint_info("  │  孤立实体补救｜LLM 未发现新关系")
            return 0

        # 解析并写入关系
        entity_name_set = set(all_entity_names)
        recovered_fids = set()
        relation_processor = getattr(self, 'relation_processor', None)
        from ._steps_helpers import _build_name_lookup, _resolve_entity_name

        # Phase 1: Resolve names + parallel LLM content writing
        _name_lookup = _build_name_lookup(entity_name_set)
        resolved_pairs = []
        for pair in raw_pairs:
            a, b = pair[0], pair[1]
            resolved_a = _resolve_entity_name(a, entity_name_set, _lookup=_name_lookup)
            resolved_b = _resolve_entity_name(b, entity_name_set, _lookup=_name_lookup)
            if not resolved_a or not resolved_b or resolved_a == resolved_b:
                continue
            fid_a = name_to_fid.get(resolved_a)
            fid_b = name_to_fid.get(resolved_b)
            if not fid_a or not fid_b:
                continue
            resolved_pairs.append((resolved_a, resolved_b, fid_a, fid_b))

        # Batch LLM content writing (1 call instead of N parallel calls)
        batch_fn = getattr(self.llm_client, 'batch_write_relation_content', None)
        batch_results = {}
        if batch_fn and resolved_pairs:
            try:
                batch_results = batch_fn(
                    [(a, b) for a, b, _, _ in resolved_pairs], window_text,
                )
            except Exception:
                pass

        content_results = []
        for resolved_a, resolved_b, fid_a, fid_b in resolved_pairs:
            content = batch_results.get((resolved_a, resolved_b), "")
            if not content:
                content = batch_results.get((resolved_b, resolved_a), "")
            if not content:
                try:
                    content = self.llm_client.write_relation_content(resolved_a, resolved_b, window_text)
                except Exception:
                    content = ""
            content_results.append((resolved_a, resolved_b, fid_a, fid_b, content))

        # Phase 2: Build relations in batch, then bulk-save
        if relation_processor and episode_id:
            batch_relations = []
            batch_fids = []
            for resolved_a, resolved_b, fid_a, fid_b, content in content_results:
                try:
                    rel = relation_processor._build_new_relation(
                        fid_a, fid_b, content, episode_id,
                        entity1_name=resolved_a, entity2_name=resolved_b,
                        verbose_relation=False, source_document=source_document,
                    )
                    if rel is not None:
                        batch_relations.append(rel)
                        batch_fids.append((resolved_a, resolved_b, fid_a, fid_b))
                except Exception:
                    pass
            if batch_relations:
                try:
                    relation_processor.storage.bulk_save_relations(batch_relations)
                except Exception:
                    # Fallback: save individually
                    for rel in batch_relations:
                        try:
                            relation_processor.storage.save_relation(rel)
                        except Exception:
                            pass
                for resolved_a, resolved_b, fid_a, fid_b in batch_fids:
                    recovered_fids.add(fid_a)
                    recovered_fids.add(fid_b)
                    if verbose:
                        wprint_debug(f"  │  补救关系: {resolved_a} <-> {resolved_b}")

        recovered_count = len(recovered_fids & set(orphan_fids))
        if verbose:
            wprint_info(f"  │  孤立实体补救｜{recovered_count}/{len(orphan_names)} 个实体成功建立关系")
        return recovered_count
