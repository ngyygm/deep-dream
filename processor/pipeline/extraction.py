"""Extraction pipeline mixin: entity/relation alignment, contradiction detection, debug snapshots.

V1-only standalone functions → _v1_legacy.py
Shared utilities (_AlignResult, etc.) → extraction_utils.py
V2/V3 extraction logic → new_extraction.py
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from math import ceil
from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future

from ..models import Episode, Entity
from ..debug_log import log as dbg, log_section as dbg_section
from ..utils import compute_doc_hash, wprint_info, wprint_warn, wprint_debug
from ..llm.client import (
    LLM_PRIORITY_STEP1, LLM_PRIORITY_STEP6, LLM_PRIORITY_STEP7,
)
from .extraction_utils import _AlignResult
from ._v1_legacy import (
    _core_entity_name,
    _is_valid_entity_name,
    validate_written_entities_with_report,
    validate_extracted_relations_with_report,
)


class _ExtractionMixin:
    """抽取相关流水线步骤（mixin，通过 TemporalMemoryGraphProcessor 多继承使用）。"""

    def _detect_and_apply_contradictions(self, family_ids: List[str], verbose: bool = False):
        """对多版本实体运行矛盾检测，发现高严重性矛盾时自动降低置信度。

        这是 remember 流水线的自动矛盾检测步骤：
        1. 对每个 family_id 获取版本历史
        2. 调用 LLM detect_contradictions 检测矛盾
        3. 对 medium/high 严重性矛盾调用 adjust_confidence_on_contradiction
        """
        for fid in family_ids:
            try:
                versions = self.storage.get_entity_versions(fid)
                if len(versions) < 2:
                    continue
                contradictions = self.llm_client.detect_contradictions(fid, versions)
                if not contradictions:
                    continue
                # 只对 medium/high 严重性矛盾降低置信度
                high_severity = [c for c in contradictions
                                 if c.get("severity") in ("high", "medium")]
                if high_severity:
                    self.storage.adjust_confidence_on_contradiction(fid, source_type="entity")
                    if verbose:
                        wprint_info(f"【矛盾检测】{fid}: 发现 {len(high_severity)} 个中/高严重性矛盾，降低置信度")
            except Exception as e:
                # 矛盾检测失败不应阻断流水线
                if verbose:
                    wprint_info(f"【矛盾检测】{fid}: 检测失败 ({e})")

    def _detect_and_apply_relation_contradictions(self, family_ids: List[str], verbose: bool = False):
        """对多版本关系运行矛盾检测，发现高严重性矛盾时自动降低置信度。

        这是 remember 流水线的自动关系矛盾检测步骤：
        1. 对每个 family_id 获取版本历史
        2. 调用 LLM detect_contradictions(concept_type="relation") 检测矛盾
        3. 对 medium/high 严重性矛盾调用 adjust_confidence_on_contradiction
        """
        for fid in family_ids:
            try:
                versions = self.storage.get_relation_versions(fid)
                if len(versions) < 2:
                    continue
                contradictions = self.llm_client.detect_contradictions(fid, versions, concept_type="relation")
                if not contradictions:
                    continue
                high_severity = [c for c in contradictions
                                 if c.get("severity") in ("high", "medium")]
                if high_severity:
                    self.storage.adjust_confidence_on_contradiction(fid, source_type="relation")
                    if verbose:
                        wprint_warn(f"【关系矛盾检测】{fid}: 发现 {len(high_severity)} 个中/高严重性矛盾，降低置信度")
            except Exception as e:
                if verbose:
                    wprint_warn(f"【关系矛盾检测】{fid}: 检测失败 ({e})")

    # =========================================================================
    # 自动摘要进化
    # =========================================================================
    SUMMARY_EVOLVE_MIN_VERSIONS = 3  # 至少 3 个版本才触发摘要进化

    def _auto_evolve_summaries(self, family_ids: List[str], verbose: bool = False):
        """对版本数足够的实体自动进化摘要。

        当实体积累了多个版本后，其 _extract_summary (首行截断) 已无法反映完整信息。
        此方法调用 LLM 生成综合性摘要，覆盖存储中的 summary 字段。

        阈值：version_count >= SUMMARY_EVOLVE_MIN_VERSIONS
        """
        import asyncio

        for fid in family_ids:
            try:
                versions = self.storage.get_entity_versions(fid)
                if len(versions) < self.SUMMARY_EVOLVE_MIN_VERSIONS:
                    continue

                current = versions[0]  # 最新版本
                old_version = versions[1] if len(versions) > 1 else None

                # 检查当前 summary 是否已经是 LLM 生成的高质量摘要
                # 如果 summary 长度 > 50 且包含多种信息，跳过（避免每次都重跑）
                existing_summary = getattr(current, 'summary', '') or ''
                if len(existing_summary) > 50:
                    # 已有较长摘要，仅当内容有显著变化时才重新生成
                    if old_version and old_version.content == current.content:
                        continue  # 内容未变，无需进化

                # 调用 async 方法：pipeline 在同步线程中运行，asyncio.run() 安全
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    summary = pool.submit(
                        asyncio.run,
                        self.llm_client.evolve_entity_summary(current, old_version)
                    ).result()

                if summary and summary.strip():
                    self.storage.update_entity_summary(fid, summary.strip())
                    if verbose:
                        wprint_info(f"【摘要进化】{fid} ({current.name}): 摘要已更新")

            except Exception as e:
                # 摘要进化失败不应阻断流水线
                if verbose:
                    wprint_info(f"【摘要进化】{fid}: 进化失败 ({e})")

    def _update_cache(self, input_text: str, document_name: str,
                      text_start_pos: int = 0, text_end_pos: int = 0,
                      total_text_length: int = 0, verbose: bool = True,
                      verbose_steps: bool = True,
                      document_path: str = "",
                      event_time: Optional[datetime] = None,
                      window_index: int = 0, total_windows: int = 0) -> Episode:
        """步骤1：更新记忆缓存。必须在 _cache_lock 下调用，保证 cache 链串行。"""
        self.llm_client._priority_local.priority = LLM_PRIORITY_STEP1
        if verbose:
            wprint_info("【步骤1】缓存｜开始｜")
        elif verbose_steps:
            wprint_info("【步骤1】缓存｜开始｜")

        # 蒸馏数据准备：确保 task_id 在步骤1前生成
        if self.llm_client._distill_data_dir:
            if not self.llm_client._distill_task_id:
                self.llm_client._distill_task_id = f"{document_name}_{uuid.uuid4().hex[:8]}_{int(time.time() * 1000)}"
            self.llm_client._current_distill_step = "01_update_cache"

        new_episode = self.llm_client.update_episode(
            self.current_episode,
            input_text,
            document_name=document_name,
            text_start_pos=text_start_pos,
            text_end_pos=text_end_pos,
            total_text_length=total_text_length,
            event_time=event_time,
            window_index=window_index,
            total_windows=total_windows,
        )

        self.llm_client._current_distill_step = None

        doc_hash = compute_doc_hash(input_text) if input_text else ""
        self.storage.save_episode(new_episode, text=input_text, document_path=document_path, doc_hash=doc_hash)
        self.current_episode = new_episode

        if verbose:
            wprint_info(f"【步骤1】缓存｜写入｜ID {new_episode.absolute_id}")
        elif verbose_steps:
            wprint_info("【步骤1】缓存｜完成｜已更新")

        return new_episode

    def _remember_debug_base_dir(self, document_name: str) -> Optional[Path]:
        root = getattr(self.llm_client, "_distill_data_dir", None)
        if not root:
            return None
        task_id = getattr(self.llm_client, "_distill_task_id", None) or f"adhoc_{document_name}"
        return Path(root) / "remember_debug" / task_id

    def _write_remember_step_snapshot(
        self,
        *,
        document_name: str,
        window_index: int,
        step_name: str,
        payload: Dict[str, Any],
    ) -> None:
        base_dir = self._remember_debug_base_dir(document_name)
        if base_dir is None:
            return
        base_dir.mkdir(parents=True, exist_ok=True)
        filename = f"window_{window_index + 1:03d}_{step_name}.json"
        try:
            with open(base_dir / filename, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    @staticmethod
    def _soft_entity_target(input_text: str, entities_per_100_chars: float) -> int:
        if entities_per_100_chars <= 0:
            return 0
        if not input_text:
            return 0
        return max(1, int(ceil(len(input_text) / 100.0 * entities_per_100_chars)))

    def _pre_alignment_validate_extraction(
        self,
        *,
        stable_names: List[str],
        extracted_entities: List[Dict[str, Any]],
        extracted_relations: List[Dict[str, Any]],
        fallback_content_by_name: Dict[str, str],
        relation_candidates: List[Dict[str, Any]],
        input_text: str,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any]]:
        valid_entities, entity_rejected = validate_written_entities_with_report(
            extracted_entities,
            stable_names,
            fallback_content_by_name=fallback_content_by_name,
        )
        valid_name_set = {entity["name"] for entity in valid_entities}
        valid_relations, relation_rejected = validate_extracted_relations_with_report(
            extracted_relations,
            valid_name_set,
        )

        soft_target = self._soft_entity_target(
            input_text,
            getattr(self, "remember_min_entities_per_100_chars_soft_target", 0.0),
        )
        report = {
            "stable_entity_count": len(stable_names),
            "written_entity_count": len(extracted_entities),
            "validated_entity_count": len(valid_entities),
            "relation_candidate_count": len(relation_candidates),
            "written_relation_count": len(extracted_relations),
            "validated_relation_count": len(valid_relations),
            "entity_rejections": entity_rejected,
            "relation_rejections": relation_rejected,
            "soft_targets": {
                "min_entities_per_100_chars": getattr(
                    self, "remember_min_entities_per_100_chars_soft_target", 0.0
                ),
                "target_entity_count": soft_target,
                "min_relation_candidates_per_window": getattr(
                    self, "remember_min_relation_candidates_per_window", 0
                ),
                "entity_target_met": len(valid_entities) >= soft_target if soft_target > 0 else True,
                "relation_candidate_target_met": len(relation_candidates) >= getattr(
                    self, "remember_min_relation_candidates_per_window", 0
                ),
            },
        }
        return valid_entities, valid_relations, report


    def _extract_only(self, new_episode: Episode, input_text: str,
                      document_name: str, verbose: bool = True,
                      verbose_steps: bool = True,
                      event_time: Optional[datetime] = None,
                      progress_callback=None,
                      progress_range: tuple = (0.1, 0.5),
                      window_index: int = 0,
                      total_windows: int = 1,
                      window_timings_ref: Optional[Dict[str, float]] = None) -> Tuple[List[Dict], List[Dict]]:
        """Dispatch extraction to V2 or V3 pipeline. No storage writes; safe for thread pools.

        Returns:
            (extracted_entities, extracted_relations) — dict lists, no family_id.
        """
        mode = getattr(self, "remember_mode", "v3")
        if mode == "v3":
            return self._extract_only_v3(
                new_episode, input_text, document_name,
                verbose=verbose, verbose_steps=verbose_steps,
                event_time=event_time, progress_callback=progress_callback,
                progress_range=progress_range,
                window_index=window_index, total_windows=total_windows,
                window_timings_ref=window_timings_ref,
            )
        if mode == "v2":
            return self._extract_only_v2(
                new_episode, input_text, document_name,
                verbose=verbose, verbose_steps=verbose_steps,
                event_time=event_time, progress_callback=progress_callback,
                progress_range=progress_range,
                window_index=window_index, total_windows=total_windows,
            )
        raise ValueError(f"Unsupported extraction mode: {mode!r}. Only 'v2' and 'v3' are supported.")

    # =========================================================================
    # 步骤6：实体对齐（写存储，必须串行跨窗口）
    # =========================================================================

    def _build_step7_relation_inputs_from_align_result(
        self, align_result: _AlignResult
    ) -> Tuple[List[Dict[str, str]], Dict[str, str], List[Dict], List[Dict]]:
        """从步骤6输出构造步骤7批处理输入；与 _align_relations 内逻辑一致，供预取与步骤7共用。"""
        entity_name_to_id = dict(align_result.entity_name_to_id)
        pending_relations_from_entities = align_result.pending_relations
        updated_pending_relations = align_result.unique_pending_relations

        # 某些并行实体对齐分支可能留下只存在于内存中的临时 family_id；
        # Step7 开始前按名称刷新一次，避免关系写入时再命中”family_id 不存在”。
        eids_to_resolve = [(name, eid) for name, eid in entity_name_to_id.items() if eid]
        valid_eids = set()
        if eids_to_resolve:
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
                    for name, eid in eids_to_resolve:
                        entity_name_to_id[name] = self.storage.resolve_family_id(eid)
            else:
                for name, eid in eids_to_resolve:
                    entity_name_to_id[name] = self.storage.resolve_family_id(eid)

        if not valid_eids:
            # Fallback: 逐条检查有效性
            for eid in set(eid for eid in entity_name_to_id.values() if eid):
                if self.storage.get_entity_by_family_id(eid) is not None:
                    valid_eids.add(eid)

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
                pair_key = tuple(sorted([entity1_id, entity2_id]))
                content_hash = hashlib.md5(content.strip().lower().encode()).hexdigest()[:12]
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
                        entity_embedding_prefetch: Optional[Future] = None,
                        already_versioned_family_ids: Optional[set] = None,
                        window_timings_ref: Optional[Dict[str, float]] = None) -> _AlignResult:
        """步骤6：实体对齐（搜索、合并、写入存储）。必须串行跨窗口。

        Returns:
            _AlignResult 包含 entity_name_to_id、pending_relations 等，供步骤7使用。
        """

        import time as _time
        p_lo, p_hi = progress_range
        _win_label = f"窗口 {window_index + 1}/{total_windows}"

        self.llm_client._priority_local.priority = LLM_PRIORITY_STEP6
        if verbose:
            wprint_info("【步骤6】实体｜开始｜对齐写入")
        elif verbose_steps:
            wprint_info("【步骤6】实体｜开始｜")

        self.llm_client._current_distill_step = "06_entity_alignment"

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
                    f"{_win_label} · 步骤6/7: 实体对齐 ({_entity_done}/{_entity_total})",
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
            max_workers=(1 if getattr(self, "remember_alignment_conservative", False) else self.llm_threads),
            verbose=verbose,
            entity_embedding_prefetch=entity_embedding_prefetch,
            already_versioned_family_ids=already_versioned_family_ids,
        )
        _t_align_elapsed = _time.time() - _t_align_start
        if window_timings_ref is not None:
            window_timings_ref["step6-process_entities"] = _t_align_elapsed
        if verbose or verbose_steps:
            wprint_info(f"【步骤6】process_entities｜{_t_align_elapsed:.1f}s｜{_entity_total}个实体")

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
        entity_name_to_ids = {}
        for entity in unique_entities:
            if entity.name not in entity_name_to_ids:
                entity_name_to_ids[entity.name] = []
            if entity.family_id not in entity_name_to_ids[entity.name]:
                entity_name_to_ids[entity.name].append(entity.family_id)

        for name, family_id in entity_name_to_id_from_entities.items():
            if name not in entity_name_to_ids:
                entity_name_to_ids[name] = []
            if family_id not in entity_name_to_ids[name]:
                entity_name_to_ids[name].append(family_id)

        for i, entity in enumerate(processed_entities):
            if i < len(original_entity_names):
                original_name = original_entity_names[i]
                if original_name not in entity_name_to_ids:
                    entity_name_to_ids[original_name] = []
                if entity.family_id not in entity_name_to_ids[original_name]:
                    entity_name_to_ids[original_name].append(entity.family_id)

        # 检测和处理同名实体冲突
        duplicate_names = {name: ids for name, ids in entity_name_to_ids.items() if len(ids) > 1}

        _t_dup_start = _time.time()
        ambiguous_duplicate_names = set()
        if duplicate_names:
            if verbose:
                wprint_info(f"【步骤6】警告｜同名｜{len(duplicate_names)}处")
                for name, ids in duplicate_names.items():
                    wprint_info(
                        f"【步骤6】冲突｜详情｜{name} {len(ids)}id {ids[:3]}{'...' if len(ids) > 3 else ''}"
                    )

            entity_name_to_id = {}
            conservative_alignment = bool(getattr(self, "remember_alignment_conservative", False))
            for name, ids in entity_name_to_ids.items():
                if len(ids) > 1:
                    versions_map = {}
                    contents_map = {}
                    for fid in ids:
                        vs = self.storage.get_entity_versions(fid)
                        versions_map[fid] = len(vs)
                        contents_map[fid] = (vs[0].content or "")[:300] if vs else ""

                    # Same-name entities: always merge — name match is strong signal.
                    # Content descriptions vary across windows (auto-generated by LLM),
                    # so Jaccard on content is unreliable for entity identity.
                    should_merge = True

                    if should_merge:
                        primary_id = max(ids, key=lambda fid: versions_map.get(fid, 0))
                        entity_name_to_id[name] = primary_id
                        for fid in ids:
                            if fid and fid != primary_id:
                                self.storage.register_entity_redirect(fid, primary_id)
                        if verbose:
                            wprint_info(
                                f"【步骤6】冲突｜主实体｜{name}->{primary_id} v{versions_map.get(primary_id, 0)}"
                            )
                    else:
                        ambiguous_duplicate_names.add(name)
                        if verbose:
                            reason = "保守对齐，不确定则新建" if conservative_alignment else f"内容差异过大 (Jaccard={jaccard:.2f})"
                            wprint_info(f"【步骤6】冲突｜跳过｜同名实体 '{name}' {reason}，不自动映射")
                        continue
                else:
                    entity_name_to_id[name] = ids[0]
        else:
            entity_name_to_id = {name: ids[0] for name, ids in entity_name_to_ids.items()}

        _t_dup_elapsed = _time.time() - _t_dup_start
        if window_timings_ref is not None:
            window_timings_ref["step6-dedup_merge"] = _t_dup_elapsed
        if (verbose or verbose_steps) and _t_dup_elapsed > 0.5:
            wprint_info(f"【步骤6】同名去重｜{_t_dup_elapsed:.1f}s")

        merged_mappings = []
        for i, entity in enumerate(processed_entities):
            if i < len(original_entity_names):
                original_name = original_entity_names[i]
                if original_name != entity.name:
                    merged_mappings.append((original_name, entity.name, entity.family_id))

        if verbose:
            if len(unique_entities) == 0:
                wprint_info(
                    f"【步骤6】小结｜实体｜无新·抽{len(original_entity_names)}个已存在"
                )
            else:
                wprint_info(
                    f"【步骤6】小结｜实体｜唯一{len(unique_entities)}·原{len(original_entity_names)}"
                )
            if merged_mappings:
                wprint_info(f"【步骤6】映射｜合并｜{len(merged_mappings)}个")

        # 步骤6.3：构建完整的实体名称→ID映射表，防止关系丢失
        # 收集关系中引用的所有实体名称
        _rel_entity_names = set()
        for rel_info in pending_relations_from_entities:
            n1 = rel_info.get("entity1_name", "")
            n2 = rel_info.get("entity2_name", "")
            if n1:
                _rel_entity_names.add(n1)
            if n2:
                _rel_entity_names.add(n2)
        # 补全：关系中引用但映射表里缺失的名称
        _missing_names = [n for n in _rel_entity_names if n not in entity_name_to_id and n not in ambiguous_duplicate_names]
        _db_matched = 0
        _fuzzy_matched = 0

        # 第一轮：精确匹配数据库
        if _missing_names:
            _db_map = self.storage.get_family_ids_by_names(_missing_names)
            for name, eid in _db_map.items():
                if name not in entity_name_to_id:
                    entity_name_to_id[name] = eid
                    _db_matched += 1

        # 第二轮：核心名称模糊匹配（去掉括号后比较）
        # 处理 LLM 在关系中使用带括号别名但实体列表中用简名的情况
        # 例如：关系端点"Docker（开源容器引擎）"匹配到实体"Docker"
        _still_missing = [n for n in _rel_entity_names if n not in entity_name_to_id]
        if _still_missing:
            # 构建核心名称→family_id 反查表
            _core_name_map: Dict[str, str] = {}
            for name, eid in entity_name_to_id.items():
                core = _core_entity_name(name)
                if core and core not in _core_name_map:
                    _core_name_map[core] = eid

            for missing_name in _still_missing:
                core_missing = _core_entity_name(missing_name)
                if not core_missing:
                    continue
                # 先在已有映射中找核心名称匹配
                if core_missing in _core_name_map:
                    entity_name_to_id[missing_name] = _core_name_map[core_missing]
                    _fuzzy_matched += 1
                    continue
                # 再在数据库中按核心名称搜索
                _db_core_map = self.storage.get_family_ids_by_names([core_missing])
                if _db_core_map:
                    for _, eid in _db_core_map.items():
                        entity_name_to_id[missing_name] = eid
                        _core_name_map[core_missing] = eid
                        _fuzzy_matched += 1
                        break

        # 第三轮：大小写不敏感匹配
        _still_missing = [n for n in _rel_entity_names if n not in entity_name_to_id]
        if _still_missing:
            _lower_map: Dict[str, str] = {}
            for name, eid in entity_name_to_id.items():
                low = name.lower()
                if low not in _lower_map:
                    _lower_map[low] = eid
            for missing_name in _still_missing:
                low_missing = missing_name.lower()
                if low_missing in _lower_map:
                    entity_name_to_id[missing_name] = _lower_map[low_missing]
                    _fuzzy_matched += 1

        # 第四轮：子字符串双向模糊匹配
        # 处理 LLM 在关系端点使用全称/简称但实体列表使用另一种形式的情况
        # 例如：关系端点"李世民"匹配到实体"唐太宗李世民"，或反之
        _still_missing = [n for n in _rel_entity_names if n not in entity_name_to_id]
        if _still_missing:
            _known_names = list(entity_name_to_id.keys())
            for missing_name in _still_missing:
                core_miss = _core_entity_name(missing_name).lower()
                if not core_miss or len(core_miss) < 2:
                    continue
                best_match = None
                best_len = 0
                for known in _known_names:
                    core_known = _core_entity_name(known).lower()
                    if not core_known or len(core_known) < 2:
                        continue
                    # 双向包含：missing 包含 known 或 known 包含 missing
                    if core_miss in core_known or core_known in core_miss:
                        # 选最长的匹配（更精确）
                        match_len = min(len(core_miss), len(core_known))
                        if match_len > best_len:
                            best_len = match_len
                            best_match = known
                if best_match:
                    entity_name_to_id[missing_name] = entity_name_to_id[best_match]
                    _fuzzy_matched += 1

        # 名称→ID转换
        updated_pending_relations = []
        _skipped_relations = []
        _self_relations = 0
        for rel_info in pending_relations_from_entities:
            entity1_name = rel_info.get("entity1_name", "")
            entity2_name = rel_info.get("entity2_name", "")
            content = rel_info.get("content", "")
            relation_type = rel_info.get("relation_type", "normal")

            entity1_id = entity_name_to_id.get(entity1_name)
            entity2_id = entity_name_to_id.get(entity2_name)

            if entity1_id and entity2_id:
                if entity1_id == entity2_id:
                    _self_relations += 1
                    continue
                updated_pending_relations.append({
                    "entity1_id": entity1_id,
                    "entity2_id": entity2_id,
                    "entity1_name": entity1_name,
                    "entity2_name": entity2_name,
                    "content": content,
                    "relation_type": relation_type
                })
            else:
                _reason = []
                if not entity1_id:
                    _reason.append(f"entity1='{entity1_name}'")
                if not entity2_id:
                    _reason.append(f"entity2='{entity2_name}'")
                _skipped_relations.append(f"  {entity1_name} <-> {entity2_name} (无法解析: {', '.join(_reason)})")

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
                    f"【步骤6】关系｜待处理｜{len(pending_relations_from_entities)}→{', '.join(_parts)}"
                )
                if _skipped_relations:
                    wprint_info(
                        f"【步骤6】映射｜表｜{len(entity_name_to_id)}名 "
                        f"{', '.join(list(entity_name_to_id.keys())[:15])}{'...' if len(entity_name_to_id) > 15 else ''}"
                    )
                    for _sr in _skipped_relations[:10]:
                        wprint_info(f"【步骤6】关系｜跳过｜{_sr}")
                    if len(_skipped_relations) > 10:
                        wprint_info(f"【步骤6】关系｜跳过｜余{len(_skipped_relations) - 10}条")
        else:
            if verbose:
                wprint_info(
                    f"【步骤6】关系｜待处理｜{len(pending_relations_from_entities)}→全解析"
                    + (f"·库补{_db_matched}" if _db_matched > 0 else "")
                )

        if verbose_steps and not verbose:
            wprint_info("【步骤6】实体｜完成｜映射")

        dbg_section("步骤6.3: 实体名称→family_id映射")
        dbg(f"entity_name_to_id 映射 ({len(entity_name_to_id)} 个):")
        for _mn, _mid in entity_name_to_id.items():
            dbg(f"  '{_mn}' -> {_mid}")
        dbg(f"待处理关系 {len(pending_relations_from_entities)} 个 → 成功 {len(updated_pending_relations)}, 自关系 {_self_relations}, 跳过 {len(_skipped_relations)}")
        for _sr in _skipped_relations:
            dbg(f"  跳过: {_sr}")

        self.llm_client._current_distill_step = None

        # Phase B+: 自动矛盾检测 — 对刚创建新版本的实体检查版本间矛盾
        # 仅对版本数 ≥ 2 的实体运行，发现高严重性矛盾时自动降低置信度
        _versioned_fids = []
        for entity in unique_entities:
            fid = entity.family_id
            vc = self.storage.get_entity_version_counts([fid])
            if vc.get(fid, 0) >= 2:
                _versioned_fids.append(fid)
        if _versioned_fids:
            self._detect_and_apply_contradictions(
                _versioned_fids, verbose=verbose,
            )

        # Phase B++: 自动摘要进化 — 对版本数足够的实体重新生成 LLM 摘要
        # 仅对版本数 >= 3 的实体运行，避免对新建实体浪费 LLM 调用
        _evolve_fids = []
        _vc_map = self.storage.get_entity_version_counts(
            [e.family_id for e in unique_entities]
        )
        for entity in unique_entities:
            if _vc_map.get(entity.family_id, 0) >= self.SUMMARY_EVOLVE_MIN_VERSIONS:
                _evolve_fids.append(entity.family_id)
        if _evolve_fids:
            self._auto_evolve_summaries(_evolve_fids, verbose=verbose)

        if progress_callback:
            progress_callback(p_hi,
                f"{_win_label} · 步骤6/7: 实体对齐",
                f"实体对齐完成，共 {len(unique_entities)} 个实体")

        # Phase C: 记录 Episode → Entity MENTIONS（无条件：所有提及的实体，包括新建和已存在的）
        try:
            # 直接使用 unique_entities 中的 absolute_id（精确到版本，避免重解析竞态）
            _seen_fids = set()
            all_mentioned_entity_ids = []
            for _e in unique_entities:
                if _e and _e.absolute_id and _e.family_id:
                    if _e.family_id not in _seen_fids:
                        _seen_fids.add(_e.family_id)
                        all_mentioned_entity_ids.append(_e.absolute_id)
            # fallback: entity_name_to_id 中有但 unique_entities 未覆盖的 family_id
            for _fid in entity_name_to_id.values():
                if _fid and _fid not in _seen_fids:
                    _ent = self.storage.get_entity_by_family_id(_fid)
                    if _ent and _ent.absolute_id:
                        all_mentioned_entity_ids.append(_ent.absolute_id)
                        _seen_fids.add(_fid)
            if all_mentioned_entity_ids:
                self.storage.save_episode_mentions(
                    new_episode.absolute_id, all_mentioned_entity_ids,
                    target_type="entity",
                )
            # Phase C-1b: Active corroboration for all mentioned entities
            # Vision: "多个独立来源印证同一个事实，置信度上升"
            for _fid in _seen_fids:
                try:
                    self.storage.adjust_confidence_on_corroboration(_fid, source_type="entity")
                except Exception:
                    pass  # corroboration failure must not block pipeline
        except Exception as e:
            if verbose:
                wprint_info(f"【步骤6】MENTIONS｜Entity｜失败｜{e}")

        return _AlignResult(
            entity_name_to_id=entity_name_to_id,
            pending_relations=pending_relations_from_entities,
            unique_entities=unique_entities,
            unique_pending_relations=updated_pending_relations,
        )

    # =========================================================================
    # 步骤7：关系对齐（写存储，串行跨窗口）
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
                         prepared_relations_by_pair: Optional[Dict[Tuple[str, str], List[Dict[str, str]]]] = None,
                         step7_inputs_cache: Optional[Tuple[List[Dict[str, str]], Dict[str, str], List[Dict], List[Dict]]] = None,
                         window_timings_ref: Optional[Dict[str, float]] = None,
                         ) -> List:
        """步骤7：关系对齐（搜索、合并、写入存储）。串行跨窗口。

        Args:
            align_result: 步骤6的输出，包含 entity_name_to_id 和 pending_relations。
            prepared_relations_by_pair: 可选，跨窗预取的按实体对分组结果（须在上一窗 step7 完成后读库）。
            step7_inputs_cache: 可选，与 _build_step7_relation_inputs_from_align_result 返回值一致，避免重复计算。
        """

        import time as _time
        p_lo, p_hi = progress_range
        _win_label = f"窗口 {window_index + 1}/{total_windows}"
        _step_size = p_hi - p_lo

        self.llm_client._priority_local.priority = LLM_PRIORITY_STEP7
        if verbose:
            wprint_info("【步骤7】关系｜开始｜对齐写入")
        elif verbose_steps:
            wprint_info("【步骤7】关系｜开始｜")

        self.llm_client._current_distill_step = "07_relation_alignment"

        unique_entities = align_result.unique_entities

        if step7_inputs_cache is not None:
            relation_inputs, entity_name_to_id, unique_pending_relations, all_pending_relations = step7_inputs_cache
        else:
            relation_inputs, entity_name_to_id, unique_pending_relations, all_pending_relations = (
                self._build_step7_relation_inputs_from_align_result(align_result)
            )

        if verbose:
            duplicate_count = len(all_pending_relations) - len(unique_pending_relations)
            if duplicate_count > 0:
                wprint_info(
                    f"【步骤7】关系｜待处理｜{len(all_pending_relations)}→去重{len(unique_pending_relations)}"
                )
            else:
                wprint_info(f"【步骤7】关系｜待处理｜{len(unique_pending_relations)}个")

        _upr_count = len(unique_pending_relations)
        if _upr_count == 0:
            if verbose:
                wprint_info("【步骤7】关系｜跳过｜无待处理")
        else:
            if verbose:
                wprint_info(
                    f"【步骤7】关系｜待处理｜去重{_upr_count}·原{len(all_pending_relations)}"
                )
        dbg(f"步骤7: 去重后待处理关系 {len(unique_pending_relations)} 个 (去重前 {len(all_pending_relations)} 个)")
        for _upr in unique_pending_relations:
            dbg(f"  待处理: '{_upr.get('entity1_name', '')}' <-> '{_upr.get('entity2_name', '')}' (e1_id={_upr.get('entity1_id', '?')}, e2_id={_upr.get('entity2_id', '?')})  content='{_upr.get('content', '')[:100]}'")

        _rel_done = [0]

        def _on_relation_pair_done(done, total):
            _rel_done[0] = done
            if progress_callback:
                frac = done / max(1, total)
                progress_callback(p_lo + _step_size * frac,
                    f"{_win_label} · 步骤7/7: 关系对齐 ({done}/{total})",
                    f"关系对齐 {done}/{total}")

        _t_rel_start = _time.time()
        all_processed_relations = self.relation_processor.process_relations_batch(
            relation_inputs,
            entity_name_to_id,
            new_episode.absolute_id,
            source_document=document_name,
            base_time=new_episode.event_time,
            max_workers=(1 if getattr(self, "remember_alignment_conservative", False) else self.llm_threads),
            on_relation_done=_on_relation_pair_done,
            # detail 模式常开 verbose、关 verbose_steps：避免逐条 [关系操作] 刷屏
            verbose_relation=bool(verbose and verbose_steps),
            prepared_relations_by_pair=prepared_relations_by_pair,
        )
        _t_rel_elapsed = _time.time() - _t_rel_start
        if window_timings_ref is not None:
            window_timings_ref["step7-process_relations"] = _t_rel_elapsed
        if verbose or verbose_steps:
            wprint_info(f"【步骤7】process_relations_batch｜{_t_rel_elapsed:.1f}s｜{len(all_processed_relations)}个关系")

        if verbose:
            if len(all_processed_relations) == 0:
                wprint_info("【步骤7】关系｜小结｜无新")
            else:
                wprint_info(f"【步骤7】关系｜小结｜{len(all_processed_relations)}个")
        elif verbose_steps:
            wprint_info("【步骤7】关系｜完成｜")

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
                f"{_win_label} · 步骤7/7: 窗口完成",
                f"{len(unique_entities)} 个实体, {len(all_processed_relations)} 个关系")

        # Phase B+: 自动关系矛盾检测 — 对刚创建新版本的关系检查版本间矛盾
        if all_processed_relations:
            _rel_versioned_fids = []
            for rel in all_processed_relations:
                fid = rel.family_id
                try:
                    vc = self.storage.get_relation_version_counts([fid])
                    if vc.get(fid, 0) >= 2:
                        _rel_versioned_fids.append(fid)
                except Exception:
                    pass
            if _rel_versioned_fids:
                self._detect_and_apply_relation_contradictions(
                    _rel_versioned_fids, verbose=verbose,
                )

        self.llm_client._current_distill_step = None
        self.llm_client._distill_task_id = None

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

        # Check 1: 孤立实体（无关系连接）
        entity_ids_in_relations = set()
        for rel in relations:
            entity_ids_in_relations.add(getattr(rel, 'entity1_id', None))
            entity_ids_in_relations.add(getattr(rel, 'entity2_id', None))
        entity_ids_in_relations.discard(None)
        isolated = [e for e in entities if getattr(e, 'family_id', None) not in entity_ids_in_relations]
        if isolated:
            report["warnings"].append({
                "type": "isolated_entities",
                "count": len(isolated),
                "names": [getattr(e, 'name', '?') for e in isolated[:5]],
            })

        # Check 2: 实体内容质量
        for e in entities:
            content = getattr(e, 'content', '') or ''
            name = getattr(e, 'name', '?')
            fid = getattr(e, 'family_id', '?')
            if len(content) < 10:
                report["issues"].append({
                    "type": "entity_content_too_short",
                    "entity_name": name,
                    "family_id": fid,
                })
            content_lower = content.lower()
            for pattern in ["处理进度", "步骤", "缓存", "抽取", "token", "api"]:
                if pattern in content_lower:
                    report["issues"].append({
                        "type": "entity_content_system_leak",
                        "entity_name": name,
                        "pattern": pattern,
                    })
                    break

        # Check 3: 关系内容质量
        for rel in relations:
            content = getattr(rel, 'content', '') or ''
            rid = getattr(rel, 'family_id', '?')
            if len(content) < 8:
                report["issues"].append({
                    "type": "relation_content_too_short",
                    "relation_id": rid,
                })

        # Check 4: 核心名称重复（去括号后同名）
        core_name_map: Dict[str, list] = {}
        for e in entities:
            name = getattr(e, 'name', '')
            fid = getattr(e, 'family_id', '')
            core = re.sub(r'[（(][^）)]+[）)]', '', name).strip()
            if core not in core_name_map:
                core_name_map[core] = []
            core_name_map[core].append(fid)
        for core, fids in core_name_map.items():
            if len(set(fids)) > 1:
                report["warnings"].append({
                    "type": "duplicate_core_names",
                    "core_name": core,
                    "family_ids": list(set(fids)),
                })

        # Check 5: 实体名称有效性
        for e in entities:
            name = getattr(e, 'name', '')
            fid = getattr(e, 'family_id', '')
            if name and not _is_valid_entity_name(name):
                report["issues"].append({
                    "type": "invalid_entity_name",
                    "entity_name": name,
                    "family_id": fid,
                })

        if verbose and (report["issues"] or report["warnings"]):
            wprint_info(f"【步骤8】校验｜问题{len(report['issues'])} 警告{len(report['warnings'])}")
            for issue in report["issues"][:5]:
                wprint_warn(f"  ⚠ 问题: {issue['type']} — {issue.get('entity_name', '') or issue.get('relation_id', '')}")
            for warn in report["warnings"][:5]:
                wprint_warn(f"  ⚡ 警告: {warn['type']} — {warn.get('names', warn.get('core_name', ''))}")

        return report

    def _cleanup_orphaned_entities(
        self,
        saved_entities: list,
        verbose: bool = False,
    ) -> int:
        """清理孤立实体：仅删除本次窗口新创建且最终没有任何关系的实体。

        在 step7（关系存储）完成后调用。此时关系已经全部写入，
        可以准确判断哪些实体是孤立的。

        重要：只删除「本次窗口新创建」的孤立实体。如果实体在对齐前就存在
        （有历史版本），即使当前无关系也不删除——因为历史版本可能携带重要信息。

        Args:
            saved_entities: step6 存入的实体列表（_AlignResult.unique_entities）
            verbose: 是否打印日志

        Returns:
            删除的孤立实体数量
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
            return 0

        # 区分「全新实体」和「对齐到已有实体的更新」
        # 批量查询版本数：版本数 > 1 说明实体在本次处理前就已存在
        version_counts = {}
        try:
            version_counts = self.storage.get_entity_version_counts(orphan_fids)
        except Exception:
            pass  # 查询失败则保守不删

        # 只删除真正全新创建的孤立实体（版本数 == 1 且无关系）
        truly_new_orphans = [fid for fid in orphan_fids
                             if version_counts.get(fid, 1) <= 1]

        # 删除孤立实体
        deleted = 0
        for fid in truly_new_orphans:
            try:
                cnt = self.storage.delete_entity_by_id(fid)
                if cnt > 0:
                    deleted += 1
                    if verbose:
                        entity_name = "?"
                        for e in saved_entities:
                            if getattr(e, 'family_id', None) == fid:
                                entity_name = getattr(e, 'name', '?')
                                break
                        wprint_debug(f"  │  清理孤立实体(新): {entity_name} ({fid})")
            except Exception:
                pass

        if deleted > 0:
            # 清理缓存
            try:
                self.storage._cache.invalidate("entity:")
                self.storage._cache.invalidate("graph_stats")
            except Exception:
                pass

        return deleted

    def _process_extraction(self, new_episode: Episode, input_text: str,
                            document_name: str, verbose: bool = True,
                            verbose_steps: bool = True,
                            event_time: Optional[datetime] = None,
                            progress_callback=None,
                            progress_range: tuple = (0.1, 1.0),
                            window_index: int = 0,
                            total_windows: int = 1):
        """兼容入口：串行执行步骤2-7（_process_window 等旧路径使用）。"""

        # 步骤2-5 占 progress_range 的 5/7，步骤6 占 1/7，步骤7 占 1/7
        total_size = progress_range[1] - progress_range[0]
        p1_end = progress_range[0] + total_size * 5 / 7
        p2_end = progress_range[0] + total_size * 6 / 7

        extracted_entities, extracted_relations = self._extract_only(
            new_episode, input_text, document_name,
            verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
            progress_callback=progress_callback,
            progress_range=(progress_range[0], p1_end),
            window_index=window_index, total_windows=total_windows,
        )

        align_result = self._align_entities(
            extracted_entities, extracted_relations,
            new_episode, input_text, document_name,
            verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
            progress_callback=progress_callback,
            progress_range=(p1_end, p2_end),
            window_index=window_index, total_windows=total_windows,
        )

        processed_relations = self._align_relations(
            align_result,
            new_episode, input_text, document_name,
            verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
            progress_callback=progress_callback,
            progress_range=(p2_end, progress_range[1]),
            window_index=window_index, total_windows=total_windows,
        )

        # Phase C-2: 记录 Episode → Relation MENTIONS（无条件：所有新建的关系）
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
                # 注意：关系置信度 corroboration 已在 relation.py _process_relations_parallel 中统一处理
            except Exception as e:
                if verbose:
                    wprint_info(f"【步骤7】MENTIONS｜Relation｜失败｜{e}")

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
