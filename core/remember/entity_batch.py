"""
Batch candidate processing mixin for EntityProcessor.

Extracted from entity.py to keep file sizes manageable. Contains the
_process_entity_with_batch_candidates method which handles the main
batch-resolution logic path.
"""
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import time

import logging

logger = logging.getLogger(__name__)

from core.debug_log import log_struct as _dbg_struct
from core.utils import wprint_info
from ._shared import _doc_basename


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
      - self.batch_resolution_enabled (bool)
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
                                     prefetched_embedding: Optional[Any] = None) -> Tuple:
        """批量候选 + 批量裁决主路径，低置信度时回退旧逻辑。

        Args:
            already_versioned_family_ids: 已创建版本的 family_id 集合，防止同窗口重复版本化。
            _version_lock: 可选线程锁，保护 already_versioned_family_ids 的并发访问。
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
            new_entity = self._build_new_entity(entity_name, entity_content, episode_id, source_document, base_time=base_time)
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
                        new_entity = self._build_new_entity(entity_name, entity_content, episode_id, source_document, base_time=base_time)
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
            new_entity = self._build_new_entity(entity_name, entity_content, episode_id, source_document, base_time=base_time)
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
        # Trust batch_resolve decisions directly — update_mode is the LLM's judgment,
        # confidence is just self-reported noise. create_new = no match = fast path.
        _safe_create_new = (update_mode == "create_new")


        _need_full_fallback = (not self.batch_resolution_enabled) or update_mode == "fallback"
        if _need_full_fallback:
            _dbg_struct("decision_fallback",
                        name=entity_name, batch_conf=f"{confidence:.2f}",
                        update_mode=update_mode,
                        reason="disabled" if not self.batch_resolution_enabled else
                               "fallback_mode" if update_mode == "fallback" else "low_confidence",
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
        # Handle within-batch alias matches (__batch_ prefixed IDs)
        if match_existing_id.startswith("__batch_"):
            batch_idx_str = match_existing_id[len("__batch_"):]
            try:
                batch_idx = int(batch_idx_str)
            except ValueError:
                batch_idx = -1
            if batch_idx >= 0:
                matched_candidate = _cand_by_fid.get(match_existing_id)
                if matched_candidate:
                    batch_name = matched_candidate.get("name", "")
                    # Resolve via entity_name_to_id dict (populated incrementally during sequential processing)
                    if batch_name:
                        resolved_id = (entity_name_to_id or {}).get(batch_name)
                        if resolved_id:
                            match_existing_id = resolved_id
                            if self._entity_tree_log():
                                wprint_info(f"  │  Within-batch alias resolved: __batch_{batch_idx} '{batch_name}' → {match_existing_id}")
                        else:
                            # Entity not yet resolved — create new entity, let the other entity merge later
                            match_existing_id = ""
                            if self._entity_tree_log():
                                wprint_info(f"  │  Within-batch alias: '{batch_name}' not yet in entity_name_to_id, creating new entity")
                    else:
                        match_existing_id = ""
        if match_existing_id:
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
                new_entity = self._build_new_entity(entity_name, entity_content, episode_id, source_document, base_time=base_time, confidence=confidence)
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
        new_entity = self._build_new_entity(merged_name, entity_content, episode_id, source_document, base_time=base_time, confidence=confidence)
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
