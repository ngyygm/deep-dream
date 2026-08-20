"""Extraction pipeline mixin: entity alignment (step9) + name resolution + step1 cache writer.

Relation alignment & orphan cleanup live in alignment_relations.py.
"""
from __future__ import annotations

import time as _time
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np

from core.models import Episode
from core.debug_log import log as dbg, log_section as dbg_section, _ENABLED as _dbg_enabled
from core.utils import compute_doc_hash, wprint_info, cosine_similarity
from core.llm.client import LLM_PRIORITY_ALIGN, LLM_PRIORITY_EXTRACT
from .helpers import _AlignResult, _core_entity_name
from .alignment_relations import _OrphanMixin, _RelationAlignMixin

# 同名强制合并的 embedding 相似度门槛（与 cross_window.py 同名合并阈值一致，0.75）。
# 名称相同是强信号但不足以断言同概念（"苹果"公司 vs"苹果"水果）：
# 双方都有已存 embedding 时，余弦相似度低于该值则跳过合并、记入歧义集合；
# 任一方无 embedding 可比时维持合并（日志标注 no-embedding-guard）。
_SAME_NAME_MERGE_SIM_THRESHOLD = 0.75


class _ResolutionMixin:
    """Same-name conflict resolution, missing-name resolution, and name-to-ID conversion."""

    def _fetch_stored_entity_vectors(self, family_ids):
        """批量读取实体已持久化的 embedding 向量（BLOB→float32 向量）。

        纯本地 DB 读取，不触发 embedding 编码/LLM 调用；
        无向量或读取失败时该 fid 对应值保持 None（调用方按"不可比"处理）。
        """
        vectors: Dict[str, Optional[np.ndarray]] = {fid: None for fid in family_ids}
        batch_fn = getattr(self.storage, 'get_entities_by_family_ids', None)
        fetched = {}
        if batch_fn:
            try:
                fetched = batch_fn(list(family_ids)) or {}
            except Exception:
                fetched = {}
        else:
            single_fn = getattr(self.storage, 'get_entity_by_family_id', None)
            for fid in family_ids:
                try:
                    ent = single_fn(fid) if single_fn else None
                    if ent:
                        fetched[fid] = ent
                except Exception:
                    continue
        for fid, ent in fetched.items():
            blob = getattr(ent, 'embedding', None)
            if not blob:
                continue
            try:
                vec = np.frombuffer(blob, dtype=np.float32)
                if vec.size:
                    vectors[fid] = vec
            except Exception:
                continue
        return vectors

    @staticmethod
    def _same_name_pair_similarity(vec_map, fid_a, fid_b):
        """返回 (余弦相似度, 是否可比)。任一方无已存向量时不可比。"""
        va, vb = vec_map.get(fid_a), vec_map.get(fid_b)
        if va is None or vb is None:
            return 0.0, False
        try:
            return cosine_similarity(va, vb), True
        except Exception:
            # 维度不一致（更换 embedding 模型后新旧向量共存）等计算失败——
            # 按不可比处理（维持合并现状），不让守卫打穿步骤9（cross_window 同保护）
            return 0.0, False

    def _resolve_same_name_conflicts(self, entity_name_to_ids, verbose=False):
        """Detect and resolve same-name entity conflicts by merging into primary.

        同名不再无条件合并：双方都有已存 embedding 时按
        _SAME_NAME_MERGE_SIM_THRESHOLD 门槛守卫（与跨窗口同名合并同阈值），
        低相似跳过合并并记入 ambiguous_duplicate_names。
        """
        duplicate_names = {name: ids for name, ids in entity_name_to_ids.items() if len(ids) > 1}
        ambiguous_duplicate_names = set()

        if not duplicate_names:
            entity_name_to_id = {name: ids[0] for name, ids in entity_name_to_ids.items()}
            return entity_name_to_id, ambiguous_duplicate_names

        if verbose:
            wprint_info(f"【步骤9】警告｜同名｜{len(duplicate_names)}处")
            for name, ids in duplicate_names.items():
                wprint_info(
                    f"【步骤9】冲突｜详情｜{name} {len(ids)}id {ids[:3]}{'...' if len(ids) > 3 else ''}"
                )

        entity_name_to_id = {}
        # Batch-fetch version counts for all duplicate-name entities
        _all_dup_fids = [fid for ids in entity_name_to_ids.values() if len(ids) > 1 for fid in ids]
        _dup_vc_map = self.storage.get_entity_version_counts(_all_dup_fids) if _all_dup_fids else {}
        # 同名合并 embedding 守卫：批量预取已存向量（纯本地 DB 读取，不发 LLM/embedding 调用）
        _dup_vec_map = self._fetch_stored_entity_vectors(_all_dup_fids) if _all_dup_fids else {}
        for name, ids in entity_name_to_ids.items():
            if len(ids) > 1:
                versions_map = {fid: _dup_vc_map.get(fid, 0) for fid in ids}

                primary_id = max(ids, key=lambda fid: versions_map.get(fid, 0))
                duplicate_pairs = []
                _ambiguous = False
                for fid in ids:
                    if not fid or fid == primary_id:
                        continue
                    sim, comparable = self._same_name_pair_similarity(_dup_vec_map, primary_id, fid)
                    if comparable and sim < _SAME_NAME_MERGE_SIM_THRESHOLD:
                        # 同名不同概念（如"苹果"公司 vs"苹果"水果）——跳过合并并记歧义
                        _ambiguous = True
                        ambiguous_duplicate_names.add(name)
                        dbg(f"step9: 同名歧义跳过合并 | {name} sim={sim:.3f} {fid}≠{primary_id}")
                        if verbose:
                            wprint_info(
                                f"【步骤9】冲突｜歧义｜{name} sim={sim:.2f} 保留 {fid}≠{primary_id}"
                            )
                        continue
                    if comparable:
                        dbg(f"step9: 同名合并 | {name} sim={sim:.3f} {fid}->{primary_id}")
                    else:
                        # 无 embedding 可比——维持合并现状
                        dbg(f"step9: 同名合并 no-embedding-guard | {name} {fid}->{primary_id}")
                    duplicate_pairs.append((fid, primary_id))
                if duplicate_pairs:
                    batch_fn = getattr(self.storage, 'register_entity_redirects_batch', None)
                    if batch_fn:
                        batch_fn(dict(duplicate_pairs))
                    else:
                        for fid, pid in duplicate_pairs:
                            self.storage.register_entity_redirect(fid, pid)
                if _ambiguous:
                    # 名称歧义——不建立名称→ID 映射，关系解析沿用 ambiguous 机制跳过/延后
                    dbg(f"step9: 同名歧义不建映射 | {name}")
                else:
                    entity_name_to_id[name] = primary_id
                    if verbose:
                        wprint_info(
                            f"【步骤9】冲突｜主实体｜{name}->{primary_id} v{versions_map.get(primary_id, 0)}"
                        )
            else:
                entity_name_to_id[name] = ids[0]

        return entity_name_to_id, ambiguous_duplicate_names

    def _resolve_missing_relation_entity_names(self, pending_relations, entity_name_to_id,
                                                 ambiguous_duplicate_names):
        """Resolve entity names referenced in relations but missing from the name-to-id map.

        Runs 4 rounds: DB exact match → core-name fuzzy → case-insensitive → substring.
        Returns (entity_name_to_id, db_matched, fuzzy_matched).
        """
        _rel_entity_names = set()
        # _core_entity_name（= entity_match_key）自带 lru_cache — 无需本地缓存

        for rel_info in pending_relations:
            n1 = rel_info.get("entity1_name", "")
            n2 = rel_info.get("entity2_name", "")
            if n1:
                _rel_entity_names.add(n1)
            if n2:
                _rel_entity_names.add(n2)

        _missing_names = [n for n in _rel_entity_names
                          if n not in entity_name_to_id and n not in ambiguous_duplicate_names]
        _db_matched = 0
        _fuzzy_matched = 0

        # Rounds 1+2 merged: single DB query with both exact and core names
        if _missing_names:
            # Build combined name set: original names + core names for fuzzy match
            _core_name_map: Dict[str, str] = {}
            for name, eid in entity_name_to_id.items():
                core = _core_entity_name(name)
                if core and core not in _core_name_map:
                    _core_name_map[core] = eid

            _query_names = set(_missing_names)
            for missing_name in _missing_names:
                core_missing = _core_entity_name(missing_name)
                if core_missing and core_missing not in _core_name_map:
                    _query_names.add(core_missing)

            _db_map = self.storage.get_family_ids_by_names(list(_query_names))

            # Round 1: resolve exact matches
            for name in _missing_names:
                if name in _db_map and name not in entity_name_to_id:
                    entity_name_to_id[name] = _db_map[name]
                    _db_matched += 1

            # Round 2: resolve core-name fuzzy matches
            # db_map 以查询原文为键——并入匹配键空间时统一归一
            for db_name, eid in _db_map.items():
                db_key = _core_entity_name(db_name)
                if db_key and db_key not in _core_name_map:
                    _core_name_map[db_key] = eid

            for missing_name in _missing_names:
                if missing_name in entity_name_to_id:
                    continue
                core_missing = _core_entity_name(missing_name)
                if core_missing and core_missing in _core_name_map:
                    entity_name_to_id[missing_name] = _core_name_map[core_missing]
                    _fuzzy_matched += 1

        # Rounds 3+4: Build lookup structures once, then iterate remaining missing names once
        _still_missing = [n for n in _rel_entity_names if n not in entity_name_to_id]
        if _still_missing:
            # Round 3 structures: 统一匹配键精确查找（casefold+剥注记/称号）
            _key_map: Dict[str, str] = {}
            # Round 4 structures: core name + substring matching
            _known_cores = []
            _core_to_known: Dict[str, str] = {}
            for name, eid in entity_name_to_id.items():
                key = _core_entity_name(name)
                if key and key not in _key_map:
                    _key_map[key] = eid
                if key and len(key) >= 2:
                    _known_cores.append((name, key))
                    if key not in _core_to_known:
                        _core_to_known[key] = name

            for missing_name in _still_missing:
                # Round 3: unified match key
                key_missing = _core_entity_name(missing_name)
                if key_missing in _key_map:
                    entity_name_to_id[missing_name] = _key_map[key_missing]
                    _fuzzy_matched += 1
                    continue
                # Round 4: substring fuzzy match
                core_miss = key_missing
                if not core_miss or len(core_miss) < 2:
                    continue
                if core_miss in _core_to_known:
                    entity_name_to_id[missing_name] = entity_name_to_id[_core_to_known[core_miss]]
                    _fuzzy_matched += 1
                    continue
                best_match = None
                best_len = 0
                for known, core_known in _known_cores:
                    if core_miss in core_known or core_known in core_miss:
                        short_len = min(len(core_miss), len(core_known))
                        long_len = max(len(core_miss), len(core_known))
                        # P2 收紧：无 LLM 验证的兜底路径——短核被长名
                        # 偶然包含（如 "公司" ⊂ "阿里巴巴集团公司"）不自动
                        # 归并。包含比 ≥0.5 覆盖真实别名形态（甄士隐/士隐、
                        # Docker容器/Docker）；更松的别名交给有 LLM 验证的
                        # cross_window 别名去重。
                        if short_len * 2 < long_len:
                            continue
                        if short_len > best_len:
                            best_len = short_len
                            best_match = known
                if best_match:
                    entity_name_to_id[missing_name] = entity_name_to_id[best_match]
                    _fuzzy_matched += 1

        return entity_name_to_id, _db_matched, _fuzzy_matched

    def _convert_pending_relations_to_ids(self, pending_relations, entity_name_to_id,
                                           verbose=False):
        """Convert relation endpoint names to family_ids. Returns (updated_relations, skipped, self_rels)."""
        updated_pending_relations = []
        _skipped_relations = []
        _self_relations = 0
        for rel_info in pending_relations:
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

        return updated_pending_relations, _skipped_relations, _self_relations


class _Step1CacheWriterMixin:
    """Step 1 cache update and debug directory helpers."""

    def _update_cache(self, input_text: str, document_name: str,
                      text_start_pos: int = 0, text_end_pos: int = 0,
                      total_text_length: int = 0, verbose: bool = True,
                      verbose_steps: bool = True,
                      document_path: str = "",
                      event_time: Optional[datetime] = None,
                      window_index: int = 0, total_windows: int = 0,
                      doc_hash: str = "",
                      heading_path: str = "",
                      episode_type: str = "",
                      run_id: str = "") -> Episode:
        """步骤1：更新记忆缓存。必须在 _cache_lock 下调用，保证 cache 链串行。"""
        self.llm_client._priority_local.priority = LLM_PRIORITY_EXTRACT
        if verbose:
            wprint_info("【步骤1】缓存｜开始｜")
        elif verbose_steps:
            wprint_info("【步骤1】缓存｜开始｜")

        # 蒸馏数据准备：确保 task_id 在步骤1前生成
        if self.llm_client._distill_data_dir:
            if not self.llm_client._distill_task_id:
                self.llm_client._distill_task_id = f"{document_name}_{uuid.uuid4().hex[:8]}_{int(_time.time() * 1000)}"
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

        # Thread new fields onto the Episode DTO before persisting
        if heading_path:
            new_episode.heading_path = heading_path
        if episode_type:
            new_episode.episode_type = episode_type

        doc_hash = doc_hash or (compute_doc_hash(input_text) if input_text else "")
        _override = getattr(self, '_pipeline_override_doc_id', '') or ''
        # strong-v1 检索切片：窗口 episode 之外按 ~N 字追加薄 FTS 切片行（0=关闭）
        _slice_chars = int(getattr(self, 'remember_episode_slice_chars', 0) or 0)
        self.storage.save_episode(
            new_episode,
            text=input_text,
            document_path=document_path,
            doc_hash=doc_hash,
            start_offset=text_start_pos,
            end_offset=text_end_pos,
            override_doc_id=_override,
            heading_path=heading_path,
            episode_type=episode_type,
            run_id=run_id,
            retrieval_slice_chars=_slice_chars,
        )
        self.current_episode = new_episode

        if verbose:
            wprint_info(f"【步骤1】缓存｜写入｜ID {new_episode.absolute_id}")
        elif verbose_steps:
            wprint_info("【步骤1】缓存｜完成｜已更新")

        return new_episode


class _PipelineExtractionMixin(_ResolutionMixin, _OrphanMixin, _Step1CacheWriterMixin, _RelationAlignMixin):
    """Core pipeline extraction mixin — step9/step10 alignment plus sub-concerns.

    Composes:
      - _ResolutionMixin: same-name conflicts, missing-name resolution, name→ID
      - _OrphanMixin: orphan entity cleanup, fallback cooccurrence, recovery
      - _CacheMixin: step 1 cache update, debug directory
      - _RelationAlignMixin: relation alignment, verification, serial window processing
    """

    # =========================================================================
    # 步骤9：实体对齐（写存储，必须串行跨窗口）
    # =========================================================================

    def _record_entity_mentions(self, unique_entities, entity_name_to_id,
                                 new_episode, verbose=False):
        """Record Episode → Entity MENTIONS and run corroboration."""
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
            try:
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
            except Exception as _me:
                if verbose:
                    wprint_info(f"MENTIONS | Entity | failed: {_me}")
        # Batch corroboration adjustment is auxiliary; it must not decide window success.
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

        # Extra resolution pass: catch entity names that step 9 missed.
        # Relations may reference entity names not present in the entity list
        # (e.g. mentioned in relation text but not extracted as entities).
        _missing_in_rels = set()
        for _pr in pending_relations_from_entities:
            for _k in ("entity1_name", "entity2_name"):
                _n = _pr.get(_k, "")
                if _n and _n not in entity_name_to_id:
                    _missing_in_rels.add(_n)
        if _missing_in_rels:
            entity_name_to_id, _xdb, _xfz = self._resolve_missing_relation_entity_names(
                pending_relations_from_entities, entity_name_to_id,
                align_result.ambiguous_duplicate_names or set(),
            )
            dbg(f"step10-input: 额外名称解析 | 缺失{len(_missing_in_rels)} DB+{_xdb} 模糊+{_xfz}")

        all_pending_relations = updated_pending_relations.copy()

        _extra_matched = 0
        _extra_self = 0
        _extra_missing = 0
        for rel_info in pending_relations_from_entities:
            entity1_name = rel_info.get("entity1_name", "")
            entity2_name = rel_info.get("entity2_name", "")
            content = rel_info.get("content", "")
            relation_type = rel_info.get("relation_type", "normal")

            entity1_id = entity_name_to_id.get(entity1_name)
            entity2_id = entity_name_to_id.get(entity2_name)

            if entity1_id and entity2_id:
                if entity1_id == entity2_id:
                    _extra_self += 1
                    continue
                new_rel = {
                    "entity1_id": entity1_id,
                    "entity2_id": entity2_id,
                    "entity1_name": entity1_name,
                    "entity2_name": entity2_name,
                    "content": content,
                    "relation_type": relation_type,
                }
                all_pending_relations.append(new_rel)
                _extra_matched += 1
            else:
                _extra_missing += 1
        if _extra_missing > 0 or _extra_self > 0:
            dbg(f"step10-input: 原始关系追加 | 总{len(pending_relations_from_entities)} 匹配{_extra_matched} 自关系{_extra_self} 缺失{_extra_missing}")

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

        relation_inputs = []
        for rel_info in unique_pending_relations:
            _ri = {
                "entity1_name": rel_info.get("entity1_name", ""),
                "entity2_name": rel_info.get("entity2_name", ""),
                "content": rel_info.get("content", ""),
            }
            relation_inputs.append(_ri)

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

        self.llm_client._priority_local.priority = LLM_PRIORITY_ALIGN
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
            def _cancel_bool_fn():
                return control_check_fn() is not None
            self.llm_client.set_cancel_check(_cancel_bool_fn)

        # LLM JSON 偶尔会在数组里混入 null/非对象项；这里仅丢弃坏项，不改变 prompt 或流程语义。
        extracted_entities = [
            e for e in (extracted_entities or [])
            if isinstance(e, dict) and str(e.get("name") or "").strip()
        ]
        extracted_relations = [
            r for r in (extracted_relations or [])
            if isinstance(r, dict)
        ]

        # 记录原始实体名称列表（用于后续建立映射）
        original_entity_names = [str(e.get('name') or '').strip() for e in extracted_entities]

        # 用于存储待处理的关系（使用实体名称）
        all_pending_relations_by_name = []
        if extracted_relations:
            for rel in extracted_relations:
                entity1_name = (rel.get('entity1_name') or rel.get('from_entity_name', '')).strip()
                entity2_name = (rel.get('entity2_name') or rel.get('to_entity_name', '')).strip()
                content = rel.get('content', '').strip()
                if entity1_name and entity2_name:
                    _rel = {
                        "entity1_name": entity1_name,
                        "entity2_name": entity2_name,
                        "content": content,
                        "relation_type": "normal",
                    }
                    all_pending_relations_by_name.append(_rel)

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
            window_timings_ref=window_timings_ref,
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

        # 注：不做 processed_entities[i] ↔ original_entity_names[i] 的位置关联——
        # 并行处理（max_workers>1）下顺序不保证，错位会把原始名挂到错误 family。
        # entity_name_to_id_from_entities 是权威映射（process_entities 已注册
        # 原始名与合并后规范名双键）。

        entity_name_to_ids = {name: list(fids) for name, fids in _name_to_fids.items()}
        dbg(f"step9: _name_to_fids | 名称变体{len(_name_to_fids)} 原始名{len(original_entity_names)} 处理后{len(processed_entities)} 唯一{len(unique_entities)}")

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
        for name, fid in entity_name_to_id_from_entities.items():
            for entity in unique_entities:
                if entity.family_id == fid and entity.name != name and name in original_entity_names:
                    merged_mappings.append((name, entity.name, entity.family_id))
                    break
        dbg(f"step9: 合并映射 | {len(merged_mappings)} 个名称变更")

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

