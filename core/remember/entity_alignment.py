"""Entity alignment layer: batch candidate resolution + parallel adjudication.

Merged from entity_batch.py + entity_parallel.py (phase M, pure move).
_EntityBatchMixin handles the per-entity batch-resolution LLM path;
module functions drive sequential/parallel window entity processing.

Lazy import rule: _preprocess_extraction_context is imported at function
level from entity.py to avoid a module-level import cycle.
"""
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
import threading
import time
import logging

import numpy as np

from core.models import Entity
from core.storage.sqlite.manager import SQLiteGraphStorageManager
from core.llm.client import LLMClient, LLM_PRIORITY_ALIGN
from core.utils import (
    wprint_info,
    capture_log_context as _capture_log_ctx,
    set_window_label as _set_wl,
    set_pipeline_role as _set_pr,
)
from core.debug_log import log_struct as _dbg_struct
from ._shared import _doc_basename

logger = logging.getLogger(__name__)


def entity_needs_batch_llm(entity_name: str, candidates: List[Dict[str, Any]]) -> bool:
    """预筛：该实体是否会走到批量裁决 LLM 调用（用于窗口级批量预裁决）。

    与 _process_entity_with_batch_candidates 内的免 LLM 快路径保持一致：
    - 无候选 → 直建，不需要 LLM
    - 首候选精确同名 + score≥0.85 + merge_safe → 快路径，不需要 LLM
    - 首候选 score<0.25 → 直建，不需要 LLM
    其余（含 alias merge 可达路径）保守视为需要 LLM；alias merge 命中时预裁决自然作废。
    """
    if not candidates:
        return False
    top = candidates[0]
    if top.get("name") == entity_name and top.get("combined_score", 0) >= 0.85 \
            and top.get("merge_safe", True):
        return False
    if top.get("combined_score", 0) < 0.25:
        return False
    return True


class _EntityBatchMixin:
    """Mixin providing the batch-candidate processing method.

    Expects the host class to provide:
      - self.storage
      - self.llm_client
      - self._entity_tree_log() -> bool
      - self._build_new_entity(...)
      - self._build_entity_version(...)
      - self._merge_two_contents(...)
      - self._mark_versioned(...)
      - self._alignment_guard(...)
      - self._try_context_alias_merge(...)
      - self._process_entity_sequential_fallback(...)
    """

    def _process_entity_with_batch_candidates(self,
                                     extracted_entity: Dict[str, str],
                                     candidates: List[Dict[str, Any]],
                                     episode_id: str,
                                     similarity_threshold: float,
                                     episode=None,
                                     source_document: str = "",
                                     context_text: Optional[str] = None,
                                     entity_index: int = 0,
                                     total_entities: int = 0,
                                     extracted_entity_names: Optional[set] = None,
                                     extracted_relation_pairs: Optional[set] = None,
                                     jaccard_search_threshold: Optional[float] = None,
                                     embedding_name_search_threshold: Optional[float] = None,
                                     embedding_full_search_threshold: Optional[float] = None,
                                     base_time: Optional[datetime] = None,
                                     already_versioned_family_ids: Optional[set] = None,
                                     _version_lock: Optional[Any] = None,
                                     entity_name_to_id: Optional[Dict[str, str]] = None,
                                     prefetched_embedding: Optional[Any] = None,
                                     precomputed_verdict: Optional[Dict[str, Any]] = None,
                                     _llm_budget: Optional[Any] = None) -> Tuple:
        """批量候选 + 批量裁决主路径，低置信度时回退旧逻辑。

        Args:
            already_versioned_family_ids: 已创建版本的 family_id 集合，防止同窗口重复版本化。
            _version_lock: 可选线程锁，保护 already_versioned_family_ids 的并发访问。
            precomputed_verdict: 窗口级批量预裁决结果（strong-v1）；schema 与
                resolve_entity_candidates_batch 返回一致，非空时跳过单体 LLM 调用。
            _llm_budget: 单窗口逐实体补裁配额（_WindowAlignLLMBudget）；None=不限。
                用尽时合成安全缺省 create_new 判决，免掉单体 LLM 调用。
        """
        entity_name = extracted_entity["name"]
        entity_content = extracted_entity["content"]
        _t_entity_start = time.monotonic()
        if self._entity_tree_log() and total_entities > 0:
            wprint_info(f"  ├─ 处理实体 [{entity_index}/{total_entities}]: {entity_name}")

        # ── Alignment trace: entity start ──
        _dbg_struct("entity_start",
                    name=entity_name,
                    content_snippet=(entity_content or "")[:120],
                    episode_id=episode_id,
                    n_candidates=len(candidates) if candidates else 0,
                    already_versioned_count=len(already_versioned_family_ids) if already_versioned_family_ids else 0)

        if not candidates:
            new_entity = self._gate_create_entity(entity_name, entity_content, episode_id, source_document, base_time=base_time)
            if self._entity_tree_log():
                wprint_info(f"  │  未找到候选实体，批量路径创建新实体: {new_entity.family_id}")
            _dbg_struct("decision_no_candidates",
                        name=entity_name, new_family_id=new_entity.family_id)
            wprint_info(f"[entity_timing] '{entity_name}' no_candidates → {time.monotonic() - _t_entity_start:.1f}s")
            self._mark_versioned(new_entity.family_id, already_versioned_family_ids, _version_lock)
            return new_entity, [], {entity_name: new_entity.family_id, new_entity.name: new_entity.family_id}, new_entity

        if self._entity_tree_log():
            wprint_info(f"  │  批量候选生成: {len(candidates)} 个")

        # ── Alignment trace: candidate summary ──
        _cand_summary = "; ".join(
            f"{c.get('name','?')}(fid={c.get('family_id','?')},score={c.get('combined_score',0):.3f},safe={c.get('merge_safe',True)},type={c.get('name_match_type','?')})"
            for c in candidates[:5]
        )
        _dbg_struct("candidates_top",
                    name=entity_name, top_n=min(len(candidates), 5),
                    candidates=_cand_summary)

        # ---- Fix 2a: 精确名称匹配 + 高embedding相似度 → 同窗口复用/跨窗口创建版本，跳过LLM ----
        top = candidates[0]
        _exact_match_skip_guard = (
            top["name"] == entity_name
            and top.get("combined_score", 0) >= 0.85
            and top.get("merge_safe", True)
            and top.get("name_match_type", "none") in ("exact", "substring")
        )
        if (top["name"] == entity_name
            and top.get("combined_score", 0) >= 0.85
            and top.get("merge_safe", True)):
            # 优先使用候选中已携带的实体对象，避免重复 DB 查询
            latest = top.get("entity") or self.storage.get_entity_by_family_id(top["family_id"])
            if latest:
                # Skip alignment guard for merge_safe exact/substring matches — the candidate
                # table already confirmed strong name + embedding similarity. The guard adds
                # ~20-40s LLM call per entity with near-zero value for these high-confidence cases.
                if not _exact_match_skip_guard:
                    # ---- Three-way alignment guard for exact name matches (Phase 4) ----
                    # Even with exact name match, check if content describes a different entity
                    # This catches "张伟(教授)" vs "张伟(CEO)" cases
                    _guard = self._alignment_guard(
                        entity_name, entity_content, latest.name, latest.content or "",
                        name_match_type=top.get("name_match_type", "none"),
                    )
                    if _guard:
                        _align_verdict, _align_confidence = _guard
                        if self._entity_tree_log():
                            _label = "同名但不同实体" if _align_verdict == "different" else "保守策略"
                            wprint_info(f"  │  快捷路径三值对齐: verdict={_align_verdict} (conf={_align_confidence:.2f}), {_label}→新建")
                        _dbg_struct("decision_exact_match_guard_reject",
                                    name=entity_name, matched_name=top.get("name","?"),
                                    matched_fid=top.get("family_id","?"),
                                    verdict=_align_verdict, guard_conf=f"{_align_confidence:.2f}",
                                    action="create_new")
                        # 同名候选已被守卫判为"不同实体"——gate 不得覆盖该裁决
                        new_entity = self._gate_create_entity(
                            entity_name, entity_content, episode_id, source_document,
                            base_time=base_time,
                            judged_candidate_names=[c.get("name", "") for c in candidates])
                        self._mark_versioned(new_entity.family_id, already_versioned_family_ids, _version_lock)
                        return new_entity, [], {entity_name: new_entity.family_id, new_entity.name: new_entity.family_id}, new_entity
                        # verdict == "same" → proceed with fast path merge

                # 同窗口内已有版本 → 直接复用，避免同窗口重复版本化（加锁防竞态）
                def _fast_path_create_version():
                    """在锁保护下检查+创建版本，防止并行线程重复版本化。"""
                    if already_versioned_family_ids and latest.family_id in already_versioned_family_ids:
                        if self._entity_tree_log():
                            wprint_info(f"  │  快捷路径：同窗口复用 {latest.family_id}")
                        _dbg_struct("decision_exact_same_window_reuse",
                                    name=entity_name, family_id=latest.family_id,
                                    action="reuse_existing_version")
                        return latest, [], {entity_name: latest.family_id, latest.name: latest.family_id}, None

                    # 内容完全相同 → 直接复用旧 content（零 LLM 开销）
                    old_content = (latest.content or "").strip()
                    new_content = entity_content.strip()
                    if old_content and old_content == new_content:
                        entity_version = self._build_entity_version(
                            latest.family_id, entity_name, latest.content,
                            episode_id, source_document, base_time=base_time,
                            old_content=latest.content or "",
                            old_content_format=latest.content_format or "plain",
                        )
                        entity_version.embedding = latest.embedding
                        self._mark_versioned(latest.family_id, already_versioned_family_ids, _version_lock)
                        if self._entity_tree_log():
                            wprint_info(f"  │  快捷路径：内容相同，直接复用 {latest.family_id}")
                        _dbg_struct("decision_exact_content_identical",
                                    name=entity_name, family_id=latest.family_id,
                                    action="reuse_content_new_version")
                        return entity_version, [], {entity_name: latest.family_id, latest.name: latest.family_id}, entity_version

                    # 内容有差异 → 增量合并（git-like editing）
                    merged_content = self._merge_two_contents(
                        latest, entity_name, entity_content,
                        source_document, episode_id, base_time=base_time,
                    )
                    final_name = entity_name

                    entity_version = self._build_entity_version(
                        latest.family_id, final_name, merged_content,
                        episode_id, source_document, base_time=base_time,
                        old_content=latest.content or "",
                        old_content_format=latest.content_format or "plain",
                    )
                    if (merged_content or "").strip() == (latest.content or "").strip():
                        entity_version.embedding = latest.embedding
                    self._mark_versioned(latest.family_id, already_versioned_family_ids, _version_lock)
                    if self._entity_tree_log():
                        wprint_info(f"  │  快捷路径：增量合并新版本 {latest.family_id}")
                    _dbg_struct("decision_exact_incremental_merge",
                                name=entity_name, family_id=latest.family_id,
                                action="merge_and_new_version")
                    return entity_version, [], {entity_name: latest.family_id, latest.name: latest.family_id}, entity_version

                if _version_lock:
                    with _version_lock:
                        _r = _fast_path_create_version()
                        wprint_info(f"[entity_timing] '{entity_name}' exact_match_fast → {time.monotonic() - _t_entity_start:.1f}s")
                        return _r
                else:
                    _r = _fast_path_create_version()
                    wprint_info(f"[entity_timing] '{entity_name}' exact_match_fast → {time.monotonic() - _t_entity_start:.1f}s")
                    return _r

        # ---- Low similarity fast path: skip LLM when best candidate score is very low ----
        if candidates[0].get("combined_score", 0) < 0.25:
            if self._entity_tree_log():
                wprint_info(f"  │  快捷路径：候选相似度过低({candidates[0].get('combined_score', 0):.2f})→新建")
            _dbg_struct("decision_low_similarity",
                        name=entity_name, best_score=f"{candidates[0].get('combined_score', 0):.3f}",
                        best_name=candidates[0].get('name', '?'), action="create_new")
            new_entity = self._gate_create_entity(
                entity_name, entity_content, episode_id, source_document,
                base_time=base_time,
                judged_candidate_names=[c.get("name", "") for c in candidates])
            if new_entity:
                self._mark_versioned(new_entity.family_id, already_versioned_family_ids, _version_lock)
            if new_entity:
                wprint_info(f"[entity_timing] '{entity_name}' low_similarity(score<0.25) → {time.monotonic() - _t_entity_start:.1f}s")
                return new_entity, [], {entity_name: new_entity.family_id, new_entity.name: new_entity.family_id}, new_entity

        # ---- Context-based alias bypass (skip LLM for obvious aliases) ----
        alias_merged = self._try_context_alias_merge(
            entity_name=entity_name,
            entity_content=entity_content,
            candidates=candidates,
            context_text=context_text,
            episode_id=episode_id,
            source_document=source_document,
            base_time=base_time,
            already_versioned_family_ids=already_versioned_family_ids,
            _version_lock=_version_lock,
            entity_name_to_id=entity_name_to_id,
        )
        if alias_merged is not None:
            _dbg_struct("decision_alias_merge",
                        name=entity_name, matched_name=candidates[0].get('name', '?') if candidates else '?',
                        matched_fid=candidates[0].get('family_id', '?') if candidates else '?',
                        combined_score=f"{candidates[0].get('combined_score', 0):.3f}" if candidates else "0",
                        action="alias_merge_guard_verified")
            wprint_info(f"[entity_timing] '{entity_name}' alias_merge → {time.monotonic() - _t_entity_start:.1f}s")
            return alias_merged
        if precomputed_verdict is not None:
            batch_result = precomputed_verdict
        elif _llm_budget is not None and not _llm_budget.take():
            # 窗口补裁配额用尽：合成安全缺省判决。多余的 create_new family 可由
            # 簇收敛/跨窗口去重后续合并（可逆），错并进同一家族不可逆——宁可新建。
            # 与 LLM 判 create_new 走完全相同的下游路径，只是免掉这次调用。
            _dbg_struct("decision_align_budget_exhausted",
                        name=entity_name,
                        n_candidates=len(candidates),
                        best_score=f"{candidates[0].get('combined_score', 0):.3f}" if candidates else "0",
                        action="create_new")
            wprint_info(
                f"[align-budget] '{entity_name}' 窗口对齐配额用尽 → create_new"
                f" (top={candidates[0].get('name', '?')} {candidates[0].get('combined_score', 0):.2f})"
                if candidates else f"[align-budget] '{entity_name}' 窗口对齐配额用尽 → create_new"
            )
            batch_result = {
                "update_mode": "create_new",
                # 0.7 对齐其它免 LLM 新建路径（无候选/低相似）的默认置信度：
                # 这是全库受验最少的实体，存 1.0 会让真家族的 corroboration
                # 加成（+0.05，上限 1.0）永远追不上配额复制体，检索排序被污染。
                "confidence": 0.7,
                "match_existing_id": "",
                "merged_name": "",
                "relations_to_create": [],
            }
        else:
            batch_result = self.llm_client.resolve_entity_candidates_batch(
                {
                    "family_id": "NEW_ENTITY",
                    "name": entity_name,
                    "content": entity_content,
                    "source_document": _doc_basename(source_document),
                    "version_count": 0,
                },
                candidates,
                context_text=context_text,
            )
        confidence = float(batch_result.get("confidence", 0.0) or 0.0)
        update_mode = batch_result.get("update_mode") or "reuse_existing"

        # ── Alignment trace: batch LLM decision ──
        _dbg_struct("batch_llm_decision",
                    name=entity_name, confidence=f"{confidence:.3f}",
                    update_mode=update_mode,
                    match_existing_id=batch_result.get("match_existing_id", ""),
                    merged_name=batch_result.get("merged_name", ""),
                    n_relations=len(batch_result.get("relations_to_create", []) or []))
        # ── 置信度门 + kill-switch（f7）──
        # 低置信度 merge_into 判决不得直接执行：把不相关实体错并进同一家族
        # 无法自动恢复，而多余的 create_new 可由跨窗口去重/后续 run 收敛——
        # 与关系腿 relation.py 的 batch_resolution_confidence_threshold 门语义
        # 对称（默认 0.75，conservative 档由 orchestrator 抬到 0.9）。
        _batch_conf_threshold = float(
            getattr(self, "batch_resolution_confidence_threshold", 0.75) or 0.0)
        if (update_mode == "merge_into_latest"
                and confidence < _batch_conf_threshold):
            _dbg_struct("decision_batch_merge_low_confidence",
                        name=entity_name, batch_conf=f"{confidence:.2f}",
                        threshold=f"{_batch_conf_threshold:.2f}",
                        matched_fid=batch_result.get("match_existing_id", ""),
                        action="degrade_to_create_new")
            if self._entity_tree_log():
                wprint_info(f"  │  批量裁决置信度不足(conf={confidence:.2f}<{_batch_conf_threshold:.2f})，merge_into 降级为新建")
            update_mode = "create_new"
        # batch_resolution_enabled kill-switch：显式关闭批量裁决时走完整
        # sequential fallback（getattr 兜底默认开启，兼容未设该属性的宿主）
        if not getattr(self, "batch_resolution_enabled", True):
            update_mode = "fallback"
        # create_new（含上面置信度降级来的）是安全判决：不进入 match_existing 分支
        _safe_create_new = (update_mode == "create_new")

        _need_full_fallback = update_mode == "fallback"
        if _need_full_fallback:
            _dbg_struct("decision_fallback",
                        name=entity_name, batch_conf=f"{confidence:.2f}",
                        update_mode=update_mode,
                        reason="fallback_mode",
                        action="sequential_fallback")
            if self._entity_tree_log():
                wprint_info(f"  │  批量裁决置信度不足，回退到旧逻辑 (confidence={confidence:.2f})")
            entity, relations, name_mapping = self._process_entity_sequential_fallback(
                extracted_entity,
                episode_id,
                similarity_threshold,
                episode,
                source_document,
                context_text,
                entity_index=entity_index,
                total_entities=total_entities,
                extracted_entity_names=extracted_entity_names,
                extracted_relation_pairs=extracted_relation_pairs,
                jaccard_search_threshold=jaccard_search_threshold,
                embedding_name_search_threshold=embedding_name_search_threshold,
                embedding_full_search_threshold=embedding_full_search_threshold,
                base_time=base_time,
                already_versioned_family_ids=already_versioned_family_ids,
                _version_lock=_version_lock,
                prefetched_embedding=prefetched_embedding,
                prebuilt_candidates=candidates,
            )
            wprint_info(f"[entity_timing] '{entity_name}' fallback_sequential(conf={confidence:.2f}) → {time.monotonic() - _t_entity_start:.1f}s")
            return entity, relations, name_mapping, None

        wprint_info(f"[entity_timing] '{entity_name}' batch_resolve(conf={confidence:.2f},{update_mode}) → {time.monotonic() - _t_entity_start:.1f}s (past fallback check)")

        # Pre-build family_id → candidate dict for O(1) lookups (avoids 4× linear scans)
        _cand_by_fid = {c.get("family_id"): c for c in candidates if c.get("family_id")}
        relations_to_create: List[Dict] = []
        for relation in batch_result.get("relations_to_create", []) or []:
            candidate = _cand_by_fid.get(relation.get("family_id"))
            if not candidate:
                continue
            relation_content = (relation.get("relation_content") or "").strip()
            if not relation_content:
                continue
            relations_to_create.append({
                "entity1_name": entity_name,
                "entity2_name": candidate.get("name", ""),
                "content": relation_content,
                "relation_type": "alias" if ("别名" in relation_content or "简称" in relation_content or "称呼" in relation_content) else "normal",
            })

        match_existing_id = (batch_result.get("match_existing_id") or "").strip()
        # _safe_create_new 生效：判决为 create_new（或被置信度门降级）时即便
        # LLM 顺手填了 match_existing_id 也不得按 reuse/merge 处理
        if match_existing_id and not _safe_create_new:
            matched_candidate = _cand_by_fid.get(match_existing_id)
            latest_entity = matched_candidate.get("entity") if matched_candidate else None
            if not latest_entity:
                # Try redirect resolution first
                resolved_id = self.storage.resolve_family_id(match_existing_id)
                if resolved_id and resolved_id != match_existing_id:
                    latest_entity = self.storage.get_entity_by_family_id(resolved_id)
            if not latest_entity:
                # Entity not found (merged/deleted) — create new directly instead of
                # expensive fallback. Register redirect so future lookups find the new entity.
                if self._entity_tree_log():
                    wprint_info(f"  │  批量裁决命中的实体不存在: {match_existing_id}，直接新建")
                new_entity = self._gate_create_entity(
                    entity_name, entity_content, episode_id, source_document,
                    base_time=base_time, confidence=confidence,
                    judged_candidate_names=[c.get("name", "") for c in candidates])
                self._mark_versioned(new_entity.family_id, already_versioned_family_ids, _version_lock)
                try:
                    self.storage.register_entity_redirect(match_existing_id, new_entity.family_id)
                except Exception:
                    pass
                wprint_info(f"[entity_timing] '{entity_name}' entity_not_found→create_new(conf={confidence:.2f}) → {time.monotonic() - _t_entity_start:.1f}s")
                return new_entity, relations_to_create, {entity_name: new_entity.family_id, new_entity.name: new_entity.family_id}, new_entity

            if update_mode == "merge_into_latest":
                # 防止同窗口内重复版本化（加锁防竞态）
                def _batch_merge_create_version():
                    if already_versioned_family_ids and match_existing_id in already_versioned_family_ids:
                        if self._entity_tree_log():
                            wprint_info(f"  │  批量裁决: family_id {match_existing_id} 已在本次处理中创建版本，复用已有实体")
                        _dbg_struct("decision_batch_merge_same_window_reuse",
                                    name=entity_name, family_id=match_existing_id,
                                    action="reuse_existing_version")
                        return latest_entity, relations_to_create, {
                            entity_name: latest_entity.family_id,
                            latest_entity.name: latest_entity.family_id,
                        }, None

                    merged_name = (batch_result.get("merged_name") or latest_entity.name).strip()

                    # 增量合并：使用专用 merge 函数，而非 batch 裁决的 merged_content
                    # 确保 CONTENT_MERGE_REQUIREMENTS 的六条增量规则始终生效
                    merged_content = self._merge_two_contents(
                        latest_entity, entity_name, entity_content,
                        source_document, episode_id, base_time=base_time,
                    )

                    # 始终创建新版本（每个 episode 提及的概念都版本化）
                    entity_version = self._build_entity_version(
                        latest_entity.family_id,
                        merged_name,
                        merged_content,
                        episode_id,
                        source_document,
                        base_time=base_time,
                        old_content=latest_entity.content or "",
                        old_content_format=latest_entity.content_format or "plain",
                    )
                    if (merged_content or "").strip() == (latest_entity.content or "").strip():
                        entity_version.embedding = latest_entity.embedding
                    self._mark_versioned(latest_entity.family_id, already_versioned_family_ids, _version_lock)
                    if self._entity_tree_log():
                        wprint_info(f"  │  批量裁决: 增量合并到已有实体 {latest_entity.family_id} 并生成新版本")
                    _dbg_struct("decision_batch_merge",
                                name=entity_name, family_id=latest_entity.family_id,
                                merged_name=merged_name,
                                confidence=f"{confidence:.2f}",
                                action="merge_incremental_new_version")
                    return entity_version, relations_to_create, {
                        entity_name: latest_entity.family_id,
                        entity_version.name: latest_entity.family_id,
                    }, entity_version

                if _version_lock:
                    with _version_lock:
                        _r = _batch_merge_create_version()
                        wprint_info(f"[entity_timing] '{entity_name}' batch_merge(conf={confidence:.2f}) → {time.monotonic() - _t_entity_start:.1f}s")
                        return _r
                else:
                    _r = _batch_merge_create_version()
                    wprint_info(f"[entity_timing] '{entity_name}' batch_merge(conf={confidence:.2f}) → {time.monotonic() - _t_entity_start:.1f}s")
                    return _r

            # reuse_existing: 跨窗口再次遇到已知实体 → 创建新版本（同窗口内已有版本则复用）
            # 使用锁保护 check+create，防止并行线程重复版本化（TOCTOU 竞态）
            def _batch_reuse_create_version():
                if already_versioned_family_ids and latest_entity.family_id in already_versioned_family_ids:
                    if self._entity_tree_log():
                        wprint_info(f"  │  批量裁决: 同窗口复用已有实体 {latest_entity.family_id}")
                    _dbg_struct("decision_batch_reuse_same_window",
                                name=entity_name, family_id=latest_entity.family_id,
                                action="reuse_existing_version")
                    return latest_entity, relations_to_create, {
                        entity_name: latest_entity.family_id,
                        latest_entity.name: latest_entity.family_id,
                    }, None
                # 始终创建新版本（每个 episode 提及的概念都版本化）
                # reuse_existing: 保留已有实体的名称和内容（新信息已被已有内容覆盖）
                entity_version = self._build_entity_version(
                    latest_entity.family_id, latest_entity.name, latest_entity.content or entity_content,
                    episode_id, source_document, base_time=base_time,
                    old_content=latest_entity.content or "",
                    old_content_format=latest_entity.content_format or "plain",
                )
                if latest_entity.content and latest_entity.content.strip() == (latest_entity.content or entity_content).strip():
                    # Content unchanged — reuse existing embedding
                    entity_version.embedding = latest_entity.embedding
                elif latest_entity.embedding:
                    # Content changed but old embedding exists — keep stale embedding
                    # rather than None; it will be overwritten when a new embedding
                    # is computed in the embedding step.
                    entity_version.embedding = latest_entity.embedding
                self._mark_versioned(latest_entity.family_id, already_versioned_family_ids, _version_lock)
                if self._entity_tree_log():
                    wprint_info(f"  │  批量裁决: 跨窗口创建新版本 {latest_entity.family_id}")
                _dbg_struct("decision_batch_reuse_cross_window",
                            name=entity_name, family_id=latest_entity.family_id,
                            confidence=f"{confidence:.2f}",
                            action="reuse_existing_new_version")
                return entity_version, relations_to_create, {
                    entity_name: latest_entity.family_id,
                    latest_entity.name: latest_entity.family_id,
                }, entity_version

            if _version_lock:
                with _version_lock:
                    _r = _batch_reuse_create_version()
                    wprint_info(f"[entity_timing] '{entity_name}' batch_reuse(conf={confidence:.2f}) → {time.monotonic() - _t_entity_start:.1f}s")
                    return _r
            else:
                _r = _batch_reuse_create_version()
                wprint_info(f"[entity_timing] '{entity_name}' batch_reuse(conf={confidence:.2f}) → {time.monotonic() - _t_entity_start:.1f}s")
                return _r

        merged_name = (batch_result.get("merged_name") or entity_name).strip() or entity_name
        new_entity = self._gate_create_entity(
            merged_name, entity_content, episode_id, source_document,
            base_time=base_time, confidence=confidence,
            judged_candidate_names=[c.get("name", "") for c in candidates])
        # 标记新实体的 family_id 已创建版本
        self._mark_versioned(new_entity.family_id, already_versioned_family_ids, _version_lock)
        if self._entity_tree_log():
            wprint_info(f"  │  批量裁决: 创建新实体 '{entity_name}' {new_entity.family_id} (had {len(candidates)} cands, best={candidates[0].get('name','?')} score={candidates[0].get('combined_score',0):.2f}, LLM chose create_new conf={confidence:.2f})")
        _dbg_struct("decision_batch_create_new",
                    name=entity_name, new_family_id=new_entity.family_id,
                    confidence=f"{confidence:.2f}",
                    best_candidate=candidates[0].get('name', '?'),
                    best_score=f"{candidates[0].get('combined_score', 0):.3f}",
                    action="create_new")
        wprint_info(f"[entity_timing] '{entity_name}' batch_create_new(conf={confidence:.2f}) → {time.monotonic() - _t_entity_start:.1f}s")
        return new_entity, relations_to_create, {
            entity_name: new_entity.family_id,
            new_entity.name: new_entity.family_id,
        }, new_entity


def _build_window_batch_verdicts(
    llm_client: LLMClient,
    extracted_entities: List[Dict[str, str]],
    candidate_table: Dict[int, List[Dict[str, Any]]],
    context_text: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """窗口级实体批量预裁决：一次 LLM 调用覆盖所有需要裁决的实体。

    返回 {entity_name: verdict}；失败/无待裁决实体时返回空 dict（逐实体回退）。
    """
    if llm_client is None:
        return {}
    needs = []
    cands_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for idx, ent in enumerate(extracted_entities):
        name = str(ent.get("name", ""))
        cands = candidate_table.get(idx, []) or []
        if name and entity_needs_batch_llm(name, cands):
            if name not in cands_by_name:
                needs.append({"name": name, "content": ent.get("content", "")})
                cands_by_name[name] = cands
    if not needs:
        return {}
    try:
        return llm_client.resolve_entities_window_batch(
            needs, cands_by_name, context_text=context_text) or {}
    except Exception as exc:
        wprint_info(f"[window_batch] 实体窗口批量裁决异常，回退逐实体: {exc}")
        return {}


class _WindowAlignLLMBudget:
    """单窗口逐实体对齐补裁的 LLM 调用配额（线程安全）。

    pipeline.remember.window_align_llm_cap（0/None=不限）。窗口批裁决缺票的
    歧义带实体每个要补一次单体裁决调用；scope 尾部实体池变大时该数失控
    （9→40+ calls/window，吞吐三倍化劣化），配额只剪病理窗口——正常窗口
    补裁数在 2-4 个，默认帽不会触发。
    """

    __slots__ = ("remaining", "lock")

    def __init__(self, cap: int):
        self.remaining = max(0, int(cap))
        self.lock = threading.Lock()

    def take(self) -> bool:
        with self.lock:
            if self.remaining <= 0:
                return False
            self.remaining -= 1
            return True


def _process_entities_sequential(
    storage: SQLiteGraphStorageManager,
    llm_client: LLMClient,
    candidate_builder,  # EntityCandidateBuilder
    entity_tree_log: bool,
    build_entity_candidate_table_fn,  # callable
    process_entity_with_batch_candidates_fn,  # callable
    extracted_entities: List[Dict[str, str]],
    episode_id: str,
    similarity_threshold: float = 0.7,
    episode=None,
    source_document: str = "",
    context_text: Optional[str] = None,
    extracted_relations: Optional[List[Dict[str, str]]] = None,
    jaccard_search_threshold: Optional[float] = None,
    embedding_name_search_threshold: Optional[float] = None,
    embedding_full_search_threshold: Optional[float] = None,
    on_entity_processed: Optional[callable] = None,
    base_time=None,
    prefetched_embeddings: Optional[Tuple[Optional[Any], Optional[Any]]] = None,
    already_versioned_family_ids: Optional[set] = None,
    window_timings_ref: Optional[Dict[str, float]] = None,
    window_batch_alignment: bool = False,
    window_align_llm_cap: int = 0,
) -> Tuple[List[Entity], List[Dict], Dict[str, str]]:
    """串行处理实体（原逻辑）。"""
    from core.remember.entity import _preprocess_extraction_context

    processed_entities: List[Entity] = []
    pending_relations: List[Dict] = []
    entity_name_to_id: Dict[str, str] = {}
    _corroborated_fids: List[str] = []

    extracted_entity_names, extracted_relation_pairs, related_entity_names = _preprocess_extraction_context(
        extracted_entities, extracted_relations,
    )

    _t_candidate = time.monotonic()
    candidate_table = build_entity_candidate_table_fn(
        extracted_entities,
        similarity_threshold=similarity_threshold,
        jaccard_search_threshold=jaccard_search_threshold,
        embedding_name_search_threshold=embedding_name_search_threshold,
        embedding_full_search_threshold=embedding_full_search_threshold,
        prefetched_embeddings=prefetched_embeddings,
    )
    if window_timings_ref is not None:
        window_timings_ref["step9-entity_candidate_table"] = time.monotonic() - _t_candidate

    # strong-v1 窗口级批量预裁决：一次调用覆盖全部待裁决实体
    _window_verdicts: Dict[str, Dict[str, Any]] = {}
    if window_batch_alignment:
        _t_wb = time.monotonic()
        _window_verdicts = _build_window_batch_verdicts(
            llm_client, extracted_entities, candidate_table, context_text)
        if window_timings_ref is not None:
            window_timings_ref["step9-window_batch_alignment"] = time.monotonic() - _t_wb
    # 单窗口逐实体补裁配额（0=不限）
    _llm_budget = _WindowAlignLLMBudget(window_align_llm_cap) if window_align_llm_cap else None

    total_entities = len(extracted_entities)
    _skipped_orphans = 0
    # Extract per-entity full-text embeddings from prefetch for sequential path
    _prefetched_full_embs = None
    if prefetched_embeddings is not None:
        try:
            _, _full_embs = prefetched_embeddings
            if _full_embs is not None:
                _prefetched_full_embs = _full_embs
        except Exception:
            pass
    _t_loop = time.monotonic()
    for idx, extracted_entity in enumerate(extracted_entities, 1):
        candidates = candidate_table.get(idx - 1, [])
        _ent_emb = None
        if _prefetched_full_embs is not None and (idx - 1) < len(_prefetched_full_embs):
            try:
                _ent_emb = np.array(_prefetched_full_embs[idx - 1], dtype=np.float32)
            except Exception:
                pass
        entity, relations, name_mapping, to_persist = process_entity_with_batch_candidates_fn(
            extracted_entity=extracted_entity,
            candidates=candidates,
            episode_id=episode_id,
            similarity_threshold=similarity_threshold,
            episode=episode,
            source_document=source_document,
            context_text=context_text,
            entity_index=idx,
            total_entities=total_entities,
            extracted_entity_names=extracted_entity_names,
            extracted_relation_pairs=extracted_relation_pairs,
            jaccard_search_threshold=jaccard_search_threshold,
            embedding_name_search_threshold=embedding_name_search_threshold,
            embedding_full_search_threshold=embedding_full_search_threshold,
            base_time=base_time,
            already_versioned_family_ids=already_versioned_family_ids,
            entity_name_to_id=entity_name_to_id,
            prefetched_embedding=_ent_emb,
            precomputed_verdict=_window_verdicts.get(extracted_entity.get("name", "")) or None,
            _llm_budget=_llm_budget,
        )

        if entity:
            processed_entities.append(entity)
            entity_name_to_id[entity.name] = entity.family_id
            entity_name_to_id[extracted_entity['name']] = entity.family_id
        if relations:
            pending_relations.extend(relations)
        if name_mapping:
            entity_name_to_id.update(name_mapping)
        if to_persist:
            storage.save_entity(to_persist)
            _ent_patches = getattr(to_persist, '_pending_patches', None) or []
            if _ent_patches:
                try:
                    storage.save_content_patches(_ent_patches)
                except Exception:
                    pass
            if to_persist.family_id:
                _corroborated_fids.append(to_persist.family_id)
        if on_entity_processed and entity:
            on_entity_processed(entity, entity_name_to_id, relations or [])
    if window_timings_ref is not None:
        window_timings_ref["step9-entity_align_loop"] = time.monotonic() - _t_loop

    # ALIGN-V2：窗口实体全部落库后，应用窗口批量裁决带出的候选等价组（flag 关闭时为 no-op）
    from core.remember.align_v2 import maybe_apply_window_cluster_dupes
    maybe_apply_window_cluster_dupes(storage, _window_verdicts, verbose=entity_tree_log)

    # Batch corroboration: 独立来源印证 → 置信度提升
    if _corroborated_fids:
        _t_corro = time.monotonic()
        _unique_fids = list(set(_corroborated_fids))
        try:
            storage.adjust_confidence_on_corroboration_batch(_unique_fids, source_type="entity")
        except Exception:
            pass
        if window_timings_ref is not None:
            window_timings_ref["step9-entity_corroboration"] = time.monotonic() - _t_corro

    return processed_entities, pending_relations, entity_name_to_id


def _process_entities_parallel(
    storage: SQLiteGraphStorageManager,
    llm_client: LLMClient,
    candidate_builder,
    entity_tree_log: bool,
    build_entity_candidate_table_fn,  # callable
    process_entity_with_batch_candidates_fn,  # callable
    get_entity_pool_fn,  # callable: (max_workers) -> ThreadPoolExecutor
    extracted_entities: List[Dict[str, str]],
    episode_id: str,
    similarity_threshold: float = 0.7,
    episode=None,
    source_document: str = "",
    context_text: Optional[str] = None,
    extracted_relations: Optional[List[Dict[str, str]]] = None,
    jaccard_search_threshold: Optional[float] = None,
    embedding_name_search_threshold: Optional[float] = None,
    embedding_full_search_threshold: Optional[float] = None,
    on_entity_processed: Optional[callable] = None,
    base_time=None,
    max_workers: int = 1,
    prefetched_embeddings: Optional[Tuple[Optional[Any], Optional[Any]]] = None,
    already_versioned_family_ids: Optional[set] = None,
    window_timings_ref: Optional[Dict[str, float]] = None,
    window_batch_alignment: bool = False,
    window_align_llm_cap: int = 0,
) -> Tuple[List[Entity], List[Dict], Dict[str, str]]:
    """多线程处理实体；合并冲突时以数据库中已存在的 family_id 为准。"""
    from core.remember.entity import _preprocess_extraction_context

    extracted_entity_names, extracted_relation_pairs, related_entity_names = _preprocess_extraction_context(
        extracted_entities, extracted_relations,
    )

    # 不再过滤孤立实体：所有通过验证的实体都应被处理
    # 孤立实体仍然有价值（如对话中提到的技术选型），丢弃会导致信息损失
    _skipped_orphans = 0
    _orig_indices = list(range(len(extracted_entities)))
    filtered_entities = extracted_entities

    _t_candidate = time.monotonic()
    candidate_table = build_entity_candidate_table_fn(
        extracted_entities,
        similarity_threshold=similarity_threshold,
        jaccard_search_threshold=jaccard_search_threshold,
        embedding_name_search_threshold=embedding_name_search_threshold,
        embedding_full_search_threshold=embedding_full_search_threshold,
        prefetched_embeddings=prefetched_embeddings,
    )
    if window_timings_ref is not None:
        window_timings_ref["step9-entity_candidate_table"] = time.monotonic() - _t_candidate
    total_entities = len(extracted_entities)

    # strong-v1 窗口级批量预裁决：一次调用覆盖全部待裁决实体（线程启动前完成）
    _window_verdicts: Dict[str, Dict[str, Any]] = {}
    if window_batch_alignment:
        _window_verdicts = _build_window_batch_verdicts(
            llm_client, extracted_entities, candidate_table, context_text)
    # 单窗口逐实体补裁配额（0=不限）；工作线程共享同一实例，锁保护递减
    _llm_budget = _WindowAlignLLMBudget(window_align_llm_cap) if window_align_llm_cap else None

    _distill_step = llm_client._current_distill_step
    _priority = getattr(llm_client._priority_local, 'priority', LLM_PRIORITY_ALIGN)
    _version_lock = threading.RLock()
    # Extract per-entity full-text embeddings from prefetch
    _prefetched_full_embs = None
    if prefetched_embeddings is not None:
        try:
            _, _full_embs = prefetched_embeddings
            if _full_embs is not None:
                _prefetched_full_embs = _full_embs
        except Exception:
            pass

    # Pre-seed entity name cache from candidate entities to reduce hidden reads
    for _cand_row in candidate_table.values():
        for _cand in _cand_row:
            _cand_ent = _cand.get("entity")
            if _cand_ent and hasattr(_cand_ent, 'absolute_id') and hasattr(_cand_ent, 'name'):
                storage._cache_entity_name(_cand_ent.absolute_id, _cand_ent.name)

    def task(idx: int, extracted_entity: Dict[str, str], orig_idx: int):
        # 将主线程的 distill step 和优先级传播到工作线程（threading.local）
        llm_client._current_distill_step = _distill_step
        llm_client._priority_local.priority = _priority
        candidates = candidate_table.get(orig_idx, [])
        _ent_emb = None
        if _prefetched_full_embs is not None and orig_idx < len(_prefetched_full_embs):
            try:
                _ent_emb = np.array(_prefetched_full_embs[orig_idx], dtype=np.float32)
            except Exception:
                pass
        entity, relations, name_mapping, to_persist = process_entity_with_batch_candidates_fn(
            extracted_entity=extracted_entity,
            candidates=candidates,
            episode_id=episode_id,
            similarity_threshold=similarity_threshold,
            episode=episode,
            source_document=source_document,
            context_text=context_text,
            entity_index=idx,
            total_entities=total_entities,
            extracted_entity_names=extracted_entity_names,
            extracted_relation_pairs=extracted_relation_pairs,
            jaccard_search_threshold=jaccard_search_threshold,
            embedding_name_search_threshold=embedding_name_search_threshold,
            embedding_full_search_threshold=embedding_full_search_threshold,
            base_time=base_time,
            already_versioned_family_ids=already_versioned_family_ids,
            _version_lock=_version_lock,
            prefetched_embedding=_ent_emb,
            precomputed_verdict=_window_verdicts.get(extracted_entity.get("name", "")) or None,
            _llm_budget=_llm_budget,
        )
        return (idx, entity, relations, name_mapping, to_persist)

    results: List[Tuple[int, Optional[Entity], List[Dict], Dict[str, str], Optional[Entity]]] = []
    executor = get_entity_pool_fn(max_workers)
    from concurrent.futures import as_completed
    # 捕获父线程日志上下文，让实体对齐子线程也能显示窗口/步骤
    _parent_ctx = _capture_log_ctx()

    def _task_with_ctx(i, ent, oi):
        label, role = _parent_ctx
        if label:
            _set_wl(label)
        if role:
            _set_pr(role)
        try:
            return task(i, ent, oi)
        finally:
            _set_wl(None)
            _set_pr(None)

    _submit_fn = _task_with_ctx if (_parent_ctx[0] or _parent_ctx[1]) else task
    futures = {
        executor.submit(_submit_fn, idx, extracted_entity, orig_idx): idx
        for idx, (extracted_entity, orig_idx) in enumerate(
            zip(filtered_entities, _orig_indices), 1
        )
    }
    _t_workers = time.monotonic()
    for future in as_completed(futures):
        results.append(future.result())
    results.sort(key=lambda r: r[0])
    if window_timings_ref is not None:
        window_timings_ref["step9-entity_parallel_resolve"] = time.monotonic() - _t_workers

    _t_merge = time.monotonic()
    name_to_ids: Dict[str, set] = defaultdict(set)
    all_candidate_eids = set()
    for idx, entity, relations, name_mapping, to_persist in results:
        if name_mapping:
            for name, eid in name_mapping.items():
                if name and eid:
                    name_to_ids[name].add(eid)
                    all_candidate_eids.add(eid)

    entity_name_to_id: Dict[str, str] = {}
    if all_candidate_eids:
        # resolve_family_ids 返回存在的映射；不存在的 eid 会被过滤
        try:
            _resolve_fn = getattr(storage, 'resolve_family_ids', None)
            if _resolve_fn:
                resolved_map = _resolve_fn(list(all_candidate_eids)) or {}
                existing_eids = set(resolved_map.keys()) | set(resolved_map.values())
            else:
                _batch_result = storage.get_entities_by_family_ids(list(all_candidate_eids))
                existing_eids = set(_batch_result.keys())
        except Exception:
            existing_eids = set()
    else:
        existing_eids = set()

    for name, ids in name_to_ids.items():
        # 优先使用数据库中已存在的 family_id（同名实体被多个线程分别匹配到不同候选）
        in_storage = [eid for eid in ids if eid in existing_eids]
        if in_storage:
            entity_name_to_id[name] = in_storage[0]
        else:
            entity_name_to_id[name] = min(ids)

    redirect_pairs = []
    for name, ids in name_to_ids.items():
        canonical_id = entity_name_to_id.get(name)
        if not canonical_id:
            continue
        for eid in ids:
            if eid and eid != canonical_id:
                redirect_pairs.append((eid, canonical_id))
    if redirect_pairs:
        if hasattr(storage, 'register_entity_redirects_batch'):
            storage.register_entity_redirects_batch(dict(redirect_pairs))
        else:
            for source_id, canonical_id in redirect_pairs:
                storage.register_entity_redirect(source_id, canonical_id)

    # 对于被合并到 canonical ID 的非 canonical 实体，需要从 results 中修正
    _canonical_ids_to_fetch = set()
    for idx, entity, relations, name_mapping, to_persist in results:
        if entity and entity.family_id != entity_name_to_id.get(entity.name):
            canonical_id = entity_name_to_id.get(entity.name)
            if canonical_id:
                _canonical_ids_to_fetch.add(canonical_id)
    if _canonical_ids_to_fetch:
        try:
            _canonical_ent_map = storage.get_entities_by_family_ids(
                list(_canonical_ids_to_fetch))
        except Exception:
            _canonical_ent_map = {}
        for i, (idx, entity, relations, name_mapping, to_persist) in enumerate(results):
            if entity and entity.family_id != entity_name_to_id.get(entity.name):
                canonical_id = entity_name_to_id.get(entity.name)
                if canonical_id:
                    canonical_entity = _canonical_ent_map.get(canonical_id)
                    if canonical_entity:
                        results[i] = (idx, canonical_entity, relations, name_mapping, to_persist)
    if window_timings_ref is not None:
        window_timings_ref["step9-entity_mapping_merge"] = time.monotonic() - _t_merge

    canonical_ids = set(entity_name_to_id.values())
    all_to_persist: List[Entity] = [r[4] for r in results if r[4] is not None]
    entities_to_persist_final = [e for e in all_to_persist if e.family_id in canonical_ids]
    # 按 family_id 去重：同一 family_id 只保留一个待持久化实体（避免重复写入）
    if entities_to_persist_final:
        _seen_fids = set()
        _deduped = []
        for e in entities_to_persist_final:
            if e.family_id not in _seen_fids:
                _seen_fids.add(e.family_id)
                _deduped.append(e)
        if len(_deduped) < len(entities_to_persist_final):
            _dup_count = len(entities_to_persist_final) - len(_deduped)
            if entity_tree_log:
                wprint_info(f"  │  持久化去重: 移除 {_dup_count} 个重复 family_id 的待持久化实体")
            entities_to_persist_final = _deduped
        # 批量保存实体（UNWIND 一次写入，减少 Neo4j 连接数）
        _corro_fids = []
        # 预计算所有 embedding（CPU 密集，不需要 Neo4j session）
        _t_embed = time.monotonic()
        batch_embed_fn = getattr(storage, '_compute_entity_embeddings_batch', None)
        _missing_embedding_entities = [e for e in entities_to_persist_final if not getattr(e, "embedding", None)]
        if batch_embed_fn and _missing_embedding_entities:
            try:
                for e, emb in zip(_missing_embedding_entities, batch_embed_fn(_missing_embedding_entities)):
                    if emb is not None:
                        e.embedding = emb[0]
            except Exception:
                for e in _missing_embedding_entities:
                    try:
                        _emb_result = storage._compute_entity_embedding(e)
                        if _emb_result is not None:
                            e.embedding = _emb_result[0]
                    except Exception:
                        pass
        elif _missing_embedding_entities:
            for e in _missing_embedding_entities:
                try:
                    _emb_result = storage._compute_entity_embedding(e)
                    if _emb_result is not None:
                        e.embedding = _emb_result[0]
                except Exception:
                    pass
        if window_timings_ref is not None:
            window_timings_ref["step9-entity_persist_embedding"] = time.monotonic() - _t_embed
        # 一次 UNWIND 写入所有实体
        _t_persist = time.monotonic()
        try:
            storage.bulk_save_entities(entities_to_persist_final)
        except Exception as _bulk_err:
            # Fallback: 逐条写入
            _saved = 0
            for e in entities_to_persist_final:
                try:
                    storage.save_entity(e)
                    _saved += 1
                except Exception as _e:
                    wprint_info(f"[entity_persist] 逐条保存失败: {getattr(e, 'name', '?')} -> {_e}")
            wprint_info(f"[entity_persist] 批量写入失败({type(_bulk_err).__name__}: {_bulk_err}), 逐条保存成功 {_saved}/{len(entities_to_persist_final)}")
        if window_timings_ref is not None:
            window_timings_ref["step9-entity_persist_db"] = time.monotonic() - _t_persist
        # 一次写入所有 patches
        _all_patches = []
        for e in entities_to_persist_final:
            _ent_patches = getattr(e, '_pending_patches', None) or []
            _all_patches.extend(_ent_patches)
            if e.family_id:
                _corro_fids.append(e.family_id)
        if _all_patches:
            _t_patches = time.monotonic()
            try:
                storage.save_content_patches(_all_patches)
            except Exception:
                pass
            if window_timings_ref is not None:
                window_timings_ref["step9-entity_persist_patches"] = time.monotonic() - _t_patches
        # Batch corroboration
        if _corro_fids:
            _t_corro = time.monotonic()
            try:
                storage.adjust_confidence_on_corroboration_batch(list(set(_corro_fids)), source_type="entity")
            except Exception:
                pass
            if window_timings_ref is not None:
                window_timings_ref["step9-entity_corroboration"] = time.monotonic() - _t_corro

    processed_entities = [r[1] for r in results if r[1] is not None]
    pending_relations: List[Dict] = []
    for r in results:
        if r[2]:
            pending_relations.extend(r[2])
    if on_entity_processed:
        for r in results:
            if r[1]:
                on_entity_processed(r[1], entity_name_to_id, r[2] or [])

    # ALIGN-V2：窗口实体全部落库后，应用窗口批量裁决带出的候选等价组（flag 关闭时为 no-op）
    from core.remember.align_v2 import maybe_apply_window_cluster_dupes
    maybe_apply_window_cluster_dupes(storage, _window_verdicts, verbose=entity_tree_log)

    return processed_entities, pending_relations, entity_name_to_id
