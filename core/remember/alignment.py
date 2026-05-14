"""Extraction pipeline mixin: entity/relation alignment, contradiction detection, debug snapshots.

Shared utilities (_AlignResult, etc.) → extraction_utils.py
Extraction logic → extraction_pipeline.py
Sub-mixins → alignment_contradiction.py, alignment_resolution.py,
             alignment_orphan.py, alignment_cache.py
"""
from __future__ import annotations

import time as _time
from typing import Any, Dict, List, Optional
from collections import defaultdict

from core.models import Episode
from core.debug_log import log as dbg, log_section as dbg_section, _ENABLED as _dbg_enabled
from core.utils import wprint_info, wprint_warn
from core.llm.client import (
    LLM_PRIORITY_STEP6, LLM_PRIORITY_STEP7,
)
from .helpers import _AlignResult
from .helpers import (
    _is_valid_entity_name,
    _PAREN_ANNOTATION_RE as _PAREN_ANNOTATION_STRIP_RE,
)
from .alignment_contradiction import _ContradictionMixin
from .alignment_resolution import _ResolutionMixin
from .alignment_orphan import _OrphanMixin
from .alignment_cache import _CacheMixin

# System leak patterns for entity content quality checks
_SYSTEM_LEAK_PATTERNS = ("处理进度", "步骤", "缓存", "抽取", "token", "api")


class _PipelineExtractionMixin(_ContradictionMixin, _ResolutionMixin, _OrphanMixin, _CacheMixin):
    """Core pipeline extraction mixin — step9/step10 alignment plus sub-concerns.

    Composes:
      - _ContradictionMixin: contradiction detection + summary evolution
      - _ResolutionMixin: same-name conflicts, missing-name resolution, name→ID
      - _OrphanMixin: orphan entity cleanup, fallback cooccurrence, recovery
      - _CacheMixin: step 1 cache update, debug directory
    """

    def _extract_only(self, new_episode: Episode, input_text: str,
                      document_name: str, verbose: bool = True,
                      verbose_steps: bool = True,
                      event_time: Optional[datetime] = None,
                      progress_callback=None,
                      progress_range: tuple = (0.1, 0.5),
                      window_index: int = 0,
                      total_windows: int = 1,
                      window_timings_ref: Optional[Dict[str, float]] = None,
                      control_check_fn=None,
                      early_entity_done_fn=None) -> Tuple[List[Dict], List[Dict]]:
        """Dispatch extraction to V2 or V3 pipeline. No storage writes; safe for thread pools.

        Returns:
            (extracted_entities, extracted_relations) — dict lists, no family_id.
        """
        mode = getattr(self, "remember_mode", "dual_model")
        if mode in ("dual_model", "standard"):
            return super()._extract_only(
                new_episode, input_text, document_name,
                verbose=verbose, verbose_steps=verbose_steps,
                event_time=event_time, progress_callback=progress_callback,
                progress_range=progress_range,
                window_index=window_index, total_windows=total_windows,
                window_timings_ref=window_timings_ref,
                control_check_fn=control_check_fn,
                early_entity_done_fn=early_entity_done_fn,
            )
        raise ValueError(f"Unsupported extraction mode: {mode!r}")

    # =========================================================================
    # 步骤9：实体对齐（写存储，必须串行跨窗口）
    # =========================================================================

    def _post_align_entity_maintenance(self, unique_entities, verbose=False):
        """Contradiction detection & summary evolution — disabled in auto pipeline (too expensive).
        Manual API endpoints (/contradictions, /resolve-contradiction) still work."""
        return

    def _record_entity_mentions(self, unique_entities, entity_name_to_id,
                                 new_episode, verbose=False):
        """Record Episode → Entity MENTIONS and run corroboration."""
        try:
            _seen_fids = set()
            all_mentioned_entity_ids = []
            for _e in unique_entities:
                if _e and _e.absolute_id and _e.family_id:
                    if _e.family_id not in _seen_fids:
                        _seen_fids.add(_e.family_id)
                        all_mentioned_entity_ids.append(_e.absolute_id)
            # Batch fetch entities not yet seen (replaces N individual calls)
            _unseen_fids = [_fid for _fid in entity_name_to_id.values()
                            if _fid and _fid not in _seen_fids]
            if _unseen_fids:
                try:
                    _batch_ents = self.storage.get_entities_by_family_ids(_unseen_fids)
                    for _fid, _ent in _batch_ents.items():
                        if _ent and _ent.absolute_id:
                            all_mentioned_entity_ids.append(_ent.absolute_id)
                            _seen_fids.add(_fid)
                except Exception:
                    pass
            if all_mentioned_entity_ids:
                self.storage.save_episode_mentions(
                    new_episode.absolute_id, all_mentioned_entity_ids,
                    target_type="entity",
                )
                # Alignment trace: mention recording
                _mention_names = []
                for _e in unique_entities:
                    if _e and _e.family_id:
                        _mention_names.append(f"{_e.name}(fid={_e.family_id})")
                dbg(f"MENTIONS: ep={new_episode.absolute_id} → {len(all_mentioned_entity_ids)} entities: {', '.join(_mention_names[:10])}")
            # Batch corroboration adjustment
            _fids_list = list(_seen_fids)
            if _fids_list:
                try:
                    batch_fn = getattr(self.storage, 'adjust_confidence_on_corroboration_batch', None)
                    if batch_fn:
                        batch_fn(_fids_list, source_type="entity")
                    else:
                        for _fid in _fids_list:
                            try:
                                self.storage.adjust_confidence_on_corroboration(_fid, source_type="entity")
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception as e:
            if verbose:
                wprint_info(f"【步骤9】MENTIONS｜Entity｜失败｜{e}")

    def _build_step10_relation_inputs_from_align_result(
        self, align_result: _AlignResult
    ):
        """从步骤9输出构造步骤10批处理输入；与 _align_relations 内逻辑一致，供预取与步骤10共用。"""
        entity_name_to_id = dict(align_result.entity_name_to_id)
        pending_relations_from_entities = align_result.pending_relations
        updated_pending_relations = align_result.unique_pending_relations

        # Fast path: if alignment already validated all family_ids, skip DB re-resolution
        _pre_resolved = align_result.resolved_family_ids
        eids_to_resolve = [(name, eid) for name, eid in entity_name_to_id.items() if eid]
        valid_eids = set()

        if _pre_resolved is not None and all(eid in _pre_resolved for _, eid in eids_to_resolve):
            valid_eids = _pre_resolved
        elif eids_to_resolve:
            # 某些并行实体对齐分支可能留下只存在于内存中的临时 family_id；
            # Step7 开始前按名称刷新一次，避免关系写入时再命中"family_id 不存在"。
            resolve_fn = getattr(self.storage, 'resolve_family_ids', None)
            if resolve_fn:
                try:
                    unique_eids = list(set(eid for _, eid in eids_to_resolve))
                    resolved_map = resolve_fn(unique_eids) or {}
                    for name, eid in eids_to_resolve:
                        entity_name_to_id[name] = resolved_map.get(eid, eid)
                    # resolve_family_ids 返回存在的映射，有效 ID = 键 ∪ 值
                    valid_eids = set(resolved_map.keys()) | set(resolved_map.values())
                except Exception:
                    _resolved_cache = {}
                    for name, eid in eids_to_resolve:
                        if eid not in _resolved_cache:
                            _resolved_cache[eid] = self.storage.resolve_family_id(eid)
                        entity_name_to_id[name] = _resolved_cache[eid]
            else:
                _resolved_cache = {}
                for name, eid in eids_to_resolve:
                    if eid not in _resolved_cache:
                        _resolved_cache[eid] = self.storage.resolve_family_id(eid)
                    entity_name_to_id[name] = _resolved_cache[eid]

        if not valid_eids:
            # Fallback: batch check validity
            _candidate_eids = list(set(eid for eid in entity_name_to_id.values() if eid))
            if _candidate_eids:
                try:
                    valid_eids = set(self.storage.get_entities_by_family_ids(_candidate_eids).keys())
                except Exception:
                    pass

        invalid_names = [
            name for name, eid in entity_name_to_id.items()
            if eid and eid not in valid_eids
        ]
        if invalid_names:
            refreshed_map = self.storage.get_family_ids_by_names(invalid_names)
            for name, refreshed_id in refreshed_map.items():
                if refreshed_id:
                    entity_name_to_id[name] = refreshed_id

        all_pending_relations = updated_pending_relations.copy()

        for rel_info in pending_relations_from_entities:
            entity1_name = rel_info.get("entity1_name", "")
            entity2_name = rel_info.get("entity2_name", "")
            content = rel_info.get("content", "")
            relation_type = rel_info.get("relation_type", "normal")

            entity1_id = entity_name_to_id.get(entity1_name)
            entity2_id = entity_name_to_id.get(entity2_name)

            if entity1_id and entity2_id:
                if entity1_id == entity2_id:
                    continue
                all_pending_relations.append({
                    "entity1_id": entity1_id,
                    "entity2_id": entity2_id,
                    "entity1_name": entity1_name,
                    "entity2_name": entity2_name,
                    "content": content,
                    "relation_type": relation_type
                })

        seen_relations = set()
        unique_pending_relations = []
        for rel in all_pending_relations:
            entity1_id = rel.get("entity1_id")
            entity2_id = rel.get("entity2_id")
            content = rel.get("content", "")
            if entity1_id and entity2_id:
                pair_key = (entity1_id, entity2_id) if entity1_id <= entity2_id else (entity2_id, entity1_id)
                content_hash = hash(content.strip().lower()) & 0xFFFFFFFFFFFF
                relation_key = (pair_key, content_hash)
                if relation_key not in seen_relations:
                    seen_relations.add(relation_key)
                    unique_pending_relations.append(rel)

        relation_inputs = [
            {
                "entity1_name": rel_info.get("entity1_name", ""),
                "entity2_name": rel_info.get("entity2_name", ""),
                "content": rel_info.get("content", ""),
            }
            for rel_info in unique_pending_relations
        ]

        return relation_inputs, entity_name_to_id, unique_pending_relations, all_pending_relations

    def _align_entities(self, extracted_entities: List[Dict], extracted_relations: List[Dict],
                        new_episode: Episode, input_text: str,
                        document_name: str, verbose: bool = True,
                        verbose_steps: bool = True,
                        event_time: Optional[datetime] = None,
                        progress_callback=None,
                        progress_range: tuple = (0.5, 0.75),
                        window_index: int = 0,
                        total_windows: int = 1,
                        entity_embedding_prefetch=None,
                        already_versioned_family_ids: Optional[set] = None,
                        window_timings_ref: Optional[Dict[str, float]] = None,
                        control_check_fn=None) -> _AlignResult:
        """步骤9：实体对齐（搜索、合并、写入存储）。必须串行跨窗口。

        Returns:
            _AlignResult 包含 entity_name_to_id、pending_relations 等，供步骤10使用。
        """

        p_lo, p_hi = progress_range
        _win_label = f"窗口 {window_index + 1}/{total_windows}"

        self.llm_client._priority_local.priority = LLM_PRIORITY_STEP6
        if verbose:
            wprint_info("【步骤9】实体｜开始｜对齐写入")
        elif verbose_steps:
            wprint_info("【步骤9】实体｜开始｜")
        if progress_callback:
            progress_callback(p_lo,
                f"{_win_label} · 步骤9/10: 实体对齐 · 开始",
                f"{len(extracted_entities)}个实体, {len(extracted_relations) if extracted_relations else 0}条待处理关系")

        self.llm_client._current_distill_step = "06_entity_alignment"

        if control_check_fn:
            action = control_check_fn()
            if action:
                from core.remember.orchestrator import RememberControlFlow
                raise RememberControlFlow(action)
            _cancel_bool_fn = lambda: control_check_fn() is not None
            self.llm_client.set_cancel_check(_cancel_bool_fn)

        # 记录原始实体名称列表（用于后续建立映射）
        original_entity_names = [e['name'] for e in extracted_entities]

        # 用于存储待处理的关系（使用实体名称）
        all_pending_relations_by_name = []
        if extracted_relations:
            for rel in extracted_relations:
                entity1_name = rel.get('entity1_name') or rel.get('from_entity_name', '').strip()
                entity2_name = rel.get('entity2_name') or rel.get('to_entity_name', '').strip()
                content = rel.get('content', '').strip()
                if entity1_name and entity2_name:
                    all_pending_relations_by_name.append({
                        "entity1_name": entity1_name,
                        "entity2_name": entity2_name,
                        "content": content,
                        "relation_type": "normal"
                    })

        entity_name_to_id_from_entities = {}
        _entity_total = len(extracted_entities)
        _entity_done = 0
        _step_size = p_hi - p_lo

        def on_entity_processed_callback(entity, current_entity_name_to_id, current_pending_relations):
            nonlocal all_pending_relations_by_name, entity_name_to_id_from_entities, _entity_done
            _entity_done += 1
            entity_name_to_id_from_entities.update(current_entity_name_to_id)
            all_pending_relations_by_name.extend(current_pending_relations)
            if progress_callback:
                frac = _entity_done / max(1, _entity_total)
                progress_callback(p_lo + _step_size * frac,
                    f"{_win_label} · 步骤9/10: 实体对齐 ({_entity_done}/{_entity_total})",
                    f"实体对齐 {_entity_done}/{_entity_total}")

        _t_align_start = _time.time()
        processed_entities, pending_relations_from_entities, entity_name_to_id_from_entities_final = self.entity_processor.process_entities(
            extracted_entities,
            new_episode.absolute_id,
            self.similarity_threshold,
            episode=new_episode,
            source_document=document_name,
            context_text=input_text,
            extracted_relations=extracted_relations,
            jaccard_search_threshold=self.jaccard_search_threshold,
            embedding_name_search_threshold=self.embedding_name_search_threshold,
            embedding_full_search_threshold=self.embedding_full_search_threshold,
            on_entity_processed=on_entity_processed_callback,
            base_time=new_episode.event_time,
            # Conservative mode: serial (1 worker). Non-conservative: llm_threads for parallel processing.
            max_workers=(1 if getattr(self, "remember_alignment_conservative", False) else self.llm_threads),
            verbose=verbose,
            entity_embedding_prefetch=entity_embedding_prefetch,
            already_versioned_family_ids=already_versioned_family_ids,
        )
        _t_align_elapsed = _time.time() - _t_align_start
        if window_timings_ref is not None:
            window_timings_ref["step9-process_entities"] = _t_align_elapsed
        if verbose or verbose_steps:
            wprint_info(f"【步骤9】process_entities｜{_t_align_elapsed:.1f}s｜{_entity_total}个实体")

        entity_name_to_id_from_entities.update(entity_name_to_id_from_entities_final)
        pending_relations_from_entities = all_pending_relations_by_name

        # 按family_id去重，只保留最新版本
        unique_entities_dict = {}
        for entity in processed_entities:
            if entity.family_id not in unique_entities_dict:
                unique_entities_dict[entity.family_id] = entity
            else:
                if entity.processed_time > unique_entities_dict[entity.family_id].processed_time:
                    unique_entities_dict[entity.family_id] = entity

        unique_entities = list(unique_entities_dict.values())

        # 构建完整的实体名称到family_id的映射
        _name_to_fids: Dict[str, set] = defaultdict(set)
        for entity in unique_entities:
            _name_to_fids[entity.name].add(entity.family_id)

        for name, family_id in entity_name_to_id_from_entities.items():
            _name_to_fids[name].add(family_id)

        for i, entity in enumerate(processed_entities):
            if i < len(original_entity_names):
                original_name = original_entity_names[i]
                _name_to_fids[original_name].add(entity.family_id)

        entity_name_to_ids = {name: list(fids) for name, fids in _name_to_fids.items()}

        # 检测和处理同名实体冲突
        if progress_callback:
            progress_callback(p_lo + _step_size * 0.85,
                f"{_win_label} · 步骤9/10: 同名实体冲突合并", "")
        _t_dup_start = _time.time()
        entity_name_to_id, ambiguous_duplicate_names = self._resolve_same_name_conflicts(
            entity_name_to_ids, verbose=verbose
        )
        _t_dup_elapsed = _time.time() - _t_dup_start
        if window_timings_ref is not None:
            window_timings_ref["step9-dedup_merge"] = _t_dup_elapsed
        if (verbose or verbose_steps) and _t_dup_elapsed > 0.5:
            wprint_info(f"【步骤9】同名去重｜{_t_dup_elapsed:.1f}s")

        merged_mappings = []
        for i, entity in enumerate(processed_entities):
            if i < len(original_entity_names):
                original_name = original_entity_names[i]
                if original_name != entity.name:
                    merged_mappings.append((original_name, entity.name, entity.family_id))

        if verbose:
            if not unique_entities:
                wprint_info(
                    f"【步骤9】小结｜实体｜无新·抽{len(original_entity_names)}个已存在"
                )
            else:
                wprint_info(
                    f"【步骤9】小结｜实体｜唯一{len(unique_entities)}·原{len(original_entity_names)}"
                )
            if merged_mappings:
                wprint_info(f"【步骤9】映射｜合并｜{len(merged_mappings)}个")

        # 步骤9：构建完整的实体名称→ID映射表，防止关系丢失
        if progress_callback:
            progress_callback(p_lo + _step_size * 0.89,
                f"{_win_label} · 步骤9/10: 关系端点名称解析", "")
        _t_resolve = _time.time()
        entity_name_to_id, _db_matched, _fuzzy_matched = self._resolve_missing_relation_entity_names(
            pending_relations_from_entities, entity_name_to_id, ambiguous_duplicate_names
        )
        if window_timings_ref is not None:
            window_timings_ref["step9-resolve_missing_names"] = _time.time() - _t_resolve

        # 名称→ID转换
        if progress_callback:
            progress_callback(p_lo + _step_size * 0.93,
                f"{_win_label} · 步骤9/10: 名称→ID转换", "")
        _t_convert = _time.time()
        updated_pending_relations, _skipped_relations, _self_relations = self._convert_pending_relations_to_ids(
            pending_relations_from_entities, entity_name_to_id, verbose=verbose
        )
        if window_timings_ref is not None:
            window_timings_ref["step9-convert_to_ids"] = _time.time() - _t_convert

        if _skipped_relations or _self_relations > 0:
            _parts = [f"成功解析 {len(updated_pending_relations)} 个"]
            if _db_matched > 0:
                _parts.append(f"数据库补全 {_db_matched} 个")
            if _fuzzy_matched > 0:
                _parts.append(f"模糊匹配 {_fuzzy_matched} 个")
            if _self_relations > 0:
                _parts.append(f"自关系 {_self_relations} 个")
            if _skipped_relations:
                _parts.append(f"无法解析 {len(_skipped_relations)} 个")
            if verbose:
                wprint_info(
                    f"【步骤9】关系｜待处理｜{len(pending_relations_from_entities)}→{', '.join(_parts)}"
                )
                if _skipped_relations:
                    _n_known = len(entity_name_to_id)
                    wprint_info(
                        f"【步骤9】映射｜表｜{_n_known}名 "
                        f"{', '.join(list(entity_name_to_id)[:15])}{'...' if _n_known > 15 else ''}"
                    )
                    for _sr in _skipped_relations[:10]:
                        wprint_info(f"【步骤9】关系｜跳过｜{_sr}")
                    if len(_skipped_relations) > 10:
                        wprint_info(f"【步骤9】关系｜跳过｜余{len(_skipped_relations) - 10}条")
        else:
            if verbose:
                wprint_info(
                    f"【步骤9】关系｜待处理｜{len(pending_relations_from_entities)}→全解析"
                    + (f"·库补{_db_matched}" if _db_matched > 0 else "")
                )

        if verbose_steps and not verbose:
            wprint_info("【步骤9】实体｜完成｜映射")

        dbg_section("步骤9: 实体名称→family_id映射")
        if _dbg_enabled:
            dbg(f"entity_name_to_id 映射 ({len(entity_name_to_id)} 个):")
            for _mn, _mid in entity_name_to_id.items():
                dbg(f"  '{_mn}' -> {_mid}")
            dbg(f"待处理关系 {len(pending_relations_from_entities)} 个 → 成功 {len(updated_pending_relations)}, 自关系 {_self_relations}, 跳过 {len(_skipped_relations)}")
            for _sr in _skipped_relations:
                dbg(f"  跳过: {_sr}")

        self.llm_client._current_distill_step = None

        # Phase B+: 自动矛盾检测 + Phase B++: 自动摘要进化
        self._post_align_entity_maintenance(unique_entities, verbose=verbose)

        # Episode→Entity MENTIONS + corroboration
        if progress_callback:
            progress_callback(p_lo + _step_size * 0.97,
                f"{_win_label} · 步骤9/10: Episode-Entity关联记录", "")

        if progress_callback:
            progress_callback(p_hi,
                f"{_win_label} · 步骤9/10: 实体对齐",
                f"实体对齐完成，共 {len(unique_entities)} 个实体")

        # Phase C: 记录 Episode → Entity MENTIONS
        _t_mentions = _time.time()
        self._record_entity_mentions(unique_entities, entity_name_to_id, new_episode, verbose=verbose)
        if window_timings_ref is not None:
            window_timings_ref["step9-entity_mentions"] = _time.time() - _t_mentions

        # Capture validated family_ids to skip redundant re-resolution in step 7
        _validated_fids = set(entity_name_to_id.values()) - {""}

        self.llm_client.clear_cancel_check()
        return _AlignResult(
            entity_name_to_id=entity_name_to_id,
            pending_relations=pending_relations_from_entities,
            unique_entities=unique_entities,
            unique_pending_relations=updated_pending_relations,
            resolved_family_ids=_validated_fids,
            ambiguous_duplicate_names=ambiguous_duplicate_names,
        )

    def _complete_align_relations(self, phase_a_result: _AlignResult,
                                   extracted_relations: List[Dict],
                                   verbose: bool = True,
                                   verbose_steps: bool = True,
                                   progress_callback=None,
                                   progress_range: tuple = (0.5, 0.75),
                                   window_index: int = 0,
                                   total_windows: int = 1,
                                   window_timings_ref: Optional[Dict[str, float]] = None):
        """步骤9 Phase B: 将抽取结果中的关系数据附加到 Phase A 的对齐结果中。

        在 Phase A (实体处理) 完成后、_extract_only 全部步骤完成后调用。
        用 entity_name_to_id 把关系端点名称解析为 family_id，构建完整的 _AlignResult。
        """
        p_lo, p_hi = progress_range
        _win_label = f"窗口 {window_index + 1}/{total_windows}"

        # Build pending relations from real extracted_relations
        all_pending_relations_by_name = []
        if extracted_relations:
            for rel in extracted_relations:
                entity1_name = rel.get('entity1_name') or rel.get('from_entity_name', '').strip()
                entity2_name = rel.get('entity2_name') or rel.get('to_entity_name', '').strip()
                content = rel.get('content', '').strip()
                if entity1_name and entity2_name:
                    all_pending_relations_by_name.append({
                        "entity1_name": entity1_name,
                        "entity2_name": entity2_name,
                        "content": content,
                        "relation_type": "normal"
                    })

        if not all_pending_relations_by_name:
            return phase_a_result

        entity_name_to_id = dict(phase_a_result.entity_name_to_id)
        _ambiguous = phase_a_result.ambiguous_duplicate_names

        # Resolve missing entity names
        _t_resolve = _time.time()
        entity_name_to_id, _db_matched, _fuzzy_matched = self._resolve_missing_relation_entity_names(
            all_pending_relations_by_name, entity_name_to_id, _ambiguous
        )
        if window_timings_ref is not None:
            window_timings_ref["step9b-resolve_missing_names"] = _time.time() - _t_resolve

        # Convert to IDs
        _t_convert = _time.time()
        updated_pending_relations, _skipped_relations, _self_relations = self._convert_pending_relations_to_ids(
            all_pending_relations_by_name, entity_name_to_id, verbose=verbose
        )
        if window_timings_ref is not None:
            window_timings_ref["step9b-convert_to_ids"] = _time.time() - _t_convert

        if _skipped_relations or _self_relations > 0:
            _parts = [f"成功解析 {len(updated_pending_relations)} 个"]
            if _db_matched > 0:
                _parts.append(f"数据库补全 {_db_matched} 个")
            if _fuzzy_matched > 0:
                _parts.append(f"模糊匹配 {_fuzzy_matched} 个")
            if _self_relations > 0:
                _parts.append(f"自关系 {_self_relations} 个")
            if _skipped_relations:
                _parts.append(f"无法解析 {len(_skipped_relations)} 个")
            if verbose:
                wprint_info(
                    f"【步骤9B】关系｜待处理｜{len(all_pending_relations_by_name)}→{', '.join(_parts)}"
                )
                if _skipped_relations:
                    for _sr in _skipped_relations[:5]:
                        wprint_info(f"【步骤9B】关系｜跳过｜{_sr}")
        else:
            if verbose:
                wprint_info(
                    f"【步骤9B】关系｜待处理｜{len(all_pending_relations_by_name)}→全解析"
                    + (f"·库补{_db_matched}" if _db_matched > 0 else "")
                )
        if verbose_steps and not verbose:
            wprint_info("【步骤9B】关系｜完成｜映射")

        _validated_fids = set(entity_name_to_id.values()) - {""}
        return _AlignResult(
            entity_name_to_id=entity_name_to_id,
            pending_relations=all_pending_relations_by_name,
            unique_entities=phase_a_result.unique_entities,
            unique_pending_relations=updated_pending_relations,
            resolved_family_ids=_validated_fids,
            ambiguous_duplicate_names=_ambiguous,
        )

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

        self.llm_client._priority_local.priority = LLM_PRIORITY_STEP7
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

    # =========================================================================
    # 兼容入口：串行执行步骤2-7（_process_window 旧路径使用）
    # =========================================================================

    def _verify_window_results(
        self,
        entities: list,
        relations: list,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """步骤8: 纯代码校验（零LLM调用）。返回校验报告。"""

        report = {
            "entity_count": len(entities),
            "relation_count": len(relations),
            "issues": [],
            "warnings": [],
        }

        # Check 1+3: Relations — collect entity_ids AND check content quality in one pass
        entity_ids_in_relations = set()
        for rel in relations:
            entity_ids_in_relations.add(getattr(rel, 'entity1_id', None))
            entity_ids_in_relations.add(getattr(rel, 'entity2_id', None))
            content = getattr(rel, 'content', '') or ''
            rid = getattr(rel, 'family_id', '?')
            if len(content) < 8:
                report["issues"].append({
                    "type": "relation_content_too_short",
                    "relation_id": rid,
                })
        entity_ids_in_relations.discard(None)
        isolated = [e for e in entities if getattr(e, 'family_id', None) not in entity_ids_in_relations]
        if isolated:
            report["warnings"].append({
                "type": "isolated_entities",
                "count": len(isolated),
                "names": [getattr(e, 'name', '?') for e in isolated[:5]],
            })

        # Check 2+4+5: Entities — content quality + core name dedup + name validity in one pass
        core_name_map: Dict[str, list] = defaultdict(list)
        for e in entities:
            name = getattr(e, 'name', '')
            fid = getattr(e, 'family_id', '')
            content = getattr(e, 'content', '') or ''
            # Check 2: content quality
            if len(content) < 10:
                report["issues"].append({
                    "type": "entity_content_too_short",
                    "entity_name": name or '?',
                    "family_id": fid or '?',
                })
            else:
                content_lower = content.lower()
                for pattern in _SYSTEM_LEAK_PATTERNS:
                    if pattern in content_lower:
                        report["issues"].append({
                            "type": "entity_content_system_leak",
                            "entity_name": name or '?',
                            "pattern": pattern,
                        })
                        break
            # Check 4: core name dedup
            core = _PAREN_ANNOTATION_STRIP_RE.sub('', name).strip()
            core_name_map[core].append(fid)
            # Check 5: name validity
            if name and not _is_valid_entity_name(name):
                report["issues"].append({
                    "type": "invalid_entity_name",
                    "entity_name": name,
                    "family_id": fid,
                })

        for core, fids in core_name_map.items():
            if len(set(fids)) > 1:
                report["warnings"].append({
                    "type": "duplicate_core_names",
                    "core_name": core,
                    "family_ids": list(set(fids)),
                })

        if verbose and (report["issues"] or report["warnings"]):
            wprint_info(f"【步骤8】校验｜问题{len(report['issues'])} 警告{len(report['warnings'])}")
            for issue in report["issues"][:5]:
                wprint_warn(f"  ⚠ 问题: {issue['type']} — {issue.get('entity_name', '') or issue.get('relation_id', '')}")
            for warn in report["warnings"][:5]:
                wprint_warn(f"  ⚡ 警告: {warn['type']} — {warn.get('names', warn.get('core_name', ''))}")

        return report

    def _process_extraction(self, new_episode: Episode, input_text: str,
                            document_name: str, verbose: bool = True,
                            verbose_steps: bool = True,
                            event_time: Optional[datetime] = None,
                            progress_callback=None,
                            progress_range: tuple = (0.1, 1.0),
                            window_index: int = 0,
                            total_windows: int = 1,
                            control_check_fn=None):
        """兼容入口：串行执行步骤2-7（_process_window 等旧路径使用）。"""

        # 步骤2-5 占 progress_range 的 5/7，步骤9 占 1/7，步骤10 占 1/7
        total_size = progress_range[1] - progress_range[0]
        p1_end = progress_range[0] + total_size * 5 / 7
        p2_end = progress_range[0] + total_size * 6 / 7

        extracted_entities, extracted_relations = self._extract_only(
            new_episode, input_text, document_name,
            verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
            progress_callback=progress_callback,
            progress_range=(progress_range[0], p1_end),
            window_index=window_index, total_windows=total_windows,
            control_check_fn=control_check_fn,
        )

        align_result = self._align_entities(
            extracted_entities, extracted_relations,
            new_episode, input_text, document_name,
            verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
            progress_callback=progress_callback,
            progress_range=(p1_end, p2_end),
            window_index=window_index, total_windows=total_windows,
            control_check_fn=control_check_fn,
        )

        processed_relations = self._align_relations(
            align_result,
            new_episode, input_text, document_name,
            verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
            progress_callback=progress_callback,
            progress_range=(p2_end, progress_range[1]),
            window_index=window_index, total_windows=total_windows,
            control_check_fn=control_check_fn,
        )

        # Phase C-2: 记录 Episode → Relation MENTIONS（串行路径）
        if processed_relations:
            try:
                rel_abs_ids = list(set(
                    r.absolute_id for r in processed_relations if r.absolute_id
                ))
                if rel_abs_ids:
                    self.storage.save_episode_mentions(
                        new_episode.absolute_id, rel_abs_ids,
                        target_type="relation",
                    )
                    if verbose:
                        wprint_info(f"【步骤10】MENTIONS｜Relation｜{len(rel_abs_ids)}条")
                # 注意：关系置信度 corroboration 已在 relation.py _process_relations_parallel 中统一处理
            except Exception as e:
                wprint_warn(f"【步骤10】MENTIONS｜Relation｜失败｜{e}")

        # 步骤8: 纯代码校验
        self._verify_window_results(
            align_result.unique_entities,
            processed_relations or [],
            verbose=verbose,
        )

    def _process_window(self, input_text: str, document_name: str,
                       is_new_document: bool, text_start_pos: int = 0,
                       text_end_pos: int = 0, total_text_length: int = 0,
                       verbose: bool = True, verbose_steps: bool = True,
                       document_path: str = "",
                       event_time: Optional[datetime] = None,
                       window_index: int = 0, total_windows: int = 1):
        """兼容入口：串行执行 cache 更新 + 抽取处理（process_documents 等旧路径使用）。"""
        if verbose:
            wprint_info(f"\n{'='*60}")
            wprint_info(f"处理窗口 (文档: {document_name}, 位置: {text_start_pos}-{text_end_pos}/{total_text_length})")
            wprint_info(f"输入文本长度: {len(input_text)} 字符")
            wprint_info(f"{'='*60}\n")
        elif verbose_steps:
            wprint_info(f"窗口开始 · {document_name}  [{text_start_pos}-{text_end_pos}/{total_text_length}]")

        with self._cache_lock:
            new_mc = self._update_cache(
                input_text, document_name,
                text_start_pos=text_start_pos, text_end_pos=text_end_pos,
                total_text_length=total_text_length, verbose=verbose,
                verbose_steps=verbose_steps,
                document_path=document_path, event_time=event_time,
                window_index=window_index, total_windows=total_windows,
            )
        self._process_extraction(new_mc, input_text, document_name,
                                 verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
                                 window_index=window_index, total_windows=total_windows)
