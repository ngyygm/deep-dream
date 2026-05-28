"""
关系处理模块：关系搜索、对齐、更新/新建
"""
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import numpy as np

from core.models import Relation
from core.llm.client import LLMClient
from core.debug_log import log as dbg, log_section as dbg_section, _ENABLED as _dbg_enabled
from core.content_schema import RELATION_SECTIONS, compute_content_patches
import time as _time

from core.utils import wprint_info, normalize_entity_pair, cosine_similarity
import logging as _logging
_log_fn = _logging.getLogger(__name__).warning

from .helpers import MIN_RELATION_CONTENT_LENGTH


def _get_entity_names(relation: Dict[str, str]) -> Tuple[str, str]:
    """Extract entity names from relation dict, supporting both old and new formats."""
    return (
        relation.get('entity1_name') or relation.get('from_entity_name', ''),
        relation.get('entity2_name') or relation.get('to_entity_name', ''),
    )


def _embedding_to_bytes(embedding: Any) -> Optional[bytes]:
    if embedding is None:
        return None
    if isinstance(embedding, bytes):
        return embedding
    try:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32, copy=False).tobytes()
    except Exception:
        return None

# Shared pool for batch relation processing
_REL_POOL: list = [None]
_REL_POOL_MAX: list = [1]

def _get_rel_pool(max_workers: int) -> ThreadPoolExecutor:
    """Return (and lazily create/upgrade) the shared relation ThreadPoolExecutor."""
    return _get_or_create_pool(_REL_POOL, max_workers, _REL_POOL_MAX, "tmg-rel")


from ._shared import _doc_basename, _get_or_create_pool
from .relation_construction import _RelationConstructionMixin


class RelationProcessor(_RelationConstructionMixin):
    """关系处理器 - 负责关系的搜索、对齐、更新和新建"""
    
    def __init__(self, storage, llm_client: LLMClient):
        self.storage = storage
        self.llm_client = llm_client
        self.batch_resolution_enabled = True
        self.batch_resolution_confidence_threshold = 0.70
        self.preserve_distinct_relations_per_pair = False
        self.emb_new_threshold = 0.80
        self._corroboration_queue: List[str] = []  # Batch corroboration family_ids
    
    def build_relations_by_pair_from_inputs(
        self,
        extracted_relations: List[Dict[str, str]],
        entity_name_to_id: Dict[str, str],
    ) -> Tuple[Dict[Tuple[str, str], List[Dict[str, str]]], int]:
        """去重合并后按实体对分组，不含读库。供步骤10跨窗预取，与 process_relations_batch 前半段一致。"""
        merged_relations = self._dedupe_and_merge_relations(extracted_relations, entity_name_to_id)
        if not merged_relations:
            return {}, 0

        relations_by_pair: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
        _batch_filtered = 0
        for merged_relation in merged_relations:
            entity1_name, entity2_name = _get_entity_names(merged_relation)
            if not entity1_name or not entity2_name:
                _batch_filtered += 1
                continue
            entity1_id = entity_name_to_id.get(entity1_name)
            entity2_id = entity_name_to_id.get(entity2_name)
            if not entity1_id or not entity2_id or entity1_id == entity2_id:
                _batch_filtered += 1
                continue
            pair_key = (entity1_id, entity2_id) if entity1_id <= entity2_id else (entity2_id, entity1_id)
            relations_by_pair[pair_key].append(merged_relation)

        return relations_by_pair, _batch_filtered

    def process_relations_batch(self,
                                extracted_relations: List[Dict[str, str]],
                                entity_name_to_id: Dict[str, str],
                                episode_id: str,
                                source_document: str = "",
                                base_time: Optional[datetime] = None,
                                fallback_to_single: bool = True,
                                max_workers: Optional[int] = None,
                                on_relation_done: Optional[callable] = None,
                                verbose_relation: bool = True,
                                prepared_relations_by_pair: Optional[Dict[Tuple[str, str], List[Dict[str, str]]]] = None,
                                window_timings_ref: Optional[Dict[str, float]] = None,
                                ) -> List[Relation]:
        """按实体对批量 upsert 关系，低置信度时回退单条逻辑。max_workers>1 且实体对数量>1 时并行处理。"""
        dbg(f"process_relations_batch: 输入 {len(extracted_relations)} 个关系, entity_name_to_id 有 {len(entity_name_to_id)} 个映射")
        if prepared_relations_by_pair is not None:
            relations_by_pair = prepared_relations_by_pair
            dbg(f"process_relations_batch: 使用预分组实体对 {len(relations_by_pair)} 个")
        else:
            relations_by_pair, _batch_filtered = self.build_relations_by_pair_from_inputs(
                extracted_relations, entity_name_to_id
            )
            dbg(f"process_relations_batch: 从 {len(extracted_relations)} 个关系中构建了 {len(relations_by_pair)} 个实体对 (过滤 {_batch_filtered})")
            if not relations_by_pair:
                return []

        _t0_prb = _time.monotonic()
        existing_relations_by_pair = self.storage.get_relations_by_entity_pairs(list(relations_by_pair))
        _t_prb_elapsed = _time.monotonic() - _t0_prb
        dbg(f"[step10_timing] get_relations_by_entity_pairs: {_t_prb_elapsed:.2f}s ({len(relations_by_pair)} pairs)")
        if window_timings_ref is not None:
            window_timings_ref["step10a-db_read_relations"] = _t_prb_elapsed

        # 批量预取所有涉及的实体，避免 _build_new_relation/_build_relation_version 中逐对查询
        unique_eids = set()
        for e1, e2 in relations_by_pair:
            unique_eids.add(e1)
            unique_eids.add(e2)
        entity_lookup: Dict[str, Any] = {}
        if unique_eids:
            _t1 = _time.monotonic()
            batch_fn = getattr(self.storage, 'get_entities_by_family_ids', None)
            if batch_fn:
                try:
                    entity_lookup = batch_fn(list(unique_eids)) or {}
                except Exception:
                    for eid in unique_eids:
                        ent = self.storage.get_entity_by_family_id(eid)
                        if ent:
                            entity_lookup[eid] = ent
            else:
                for eid in unique_eids:
                    ent = self.storage.get_entity_by_family_id(eid)
                    if ent:
                        entity_lookup[eid] = ent
            dbg(f"[step10_timing] get_entities_by_family_ids: {_time.monotonic()-_t1:.2f}s ({len(unique_eids)} entities)")
            if window_timings_ref is not None:
                window_timings_ref["step10a-db_fetch_entities"] = _time.monotonic() - _t1
            # Pre-seed entity name cache to eliminate hidden reads in save_relation()
            for _ent in entity_lookup.values():
                if hasattr(_ent, 'absolute_id') and hasattr(_ent, 'name'):
                    self.storage._cache_entity_name(_ent.absolute_id, _ent.name)

        # Embedding fast-path prep
        embedding_ctx = None
        emb_client = getattr(self.storage, 'embedding_client', None)
        if emb_client and emb_client.is_available():
            _t_emb = _time.monotonic()
            # 1) 批量获取已有关系的 embedding
            _existing_fids = set()
            for pair_rels in existing_relations_by_pair.values():
                for r in pair_rels:
                    if r.family_id:
                        _existing_fids.add(r.family_id)
            existing_emb_map = {}
            if _existing_fids:
                existing_emb_map = self.storage.get_relation_embeddings(list(_existing_fids))

            # 2) 批量编码新关系内容
            _emb_texts = []
            _text_to_content = {}
            for pair_key, pair_rels in relations_by_pair.items():
                e1_name, e2_name = _get_entity_names(pair_rels[0])
                for rel in pair_rels:
                    c = rel.get('content', '')
                    if c:
                        text = f"# {e1_name} → {e2_name}\n{c[:512]}"
                        if text not in _text_to_content:
                            _emb_texts.append(text)
                            _text_to_content[text] = c

            new_content_embs = {}
            if _emb_texts:
                emb_arrays = emb_client.encode(_emb_texts)
                if emb_arrays is not None:
                    for text, emb in zip(_emb_texts, emb_arrays):
                        new_content_embs[_text_to_content[text]] = emb

            if existing_emb_map or new_content_embs:
                embedding_ctx = {
                    'new_embs': new_content_embs,
                    'existing_embs': existing_emb_map,
                }
            dbg(f"[step10_timing] embedding prep: {_time.monotonic()-_t_emb:.2f}s ({len(existing_emb_map)} exist, {len(new_content_embs)} new)")
            if window_timings_ref is not None:
                window_timings_ref["step10a-embedding_prep"] = _time.monotonic() - _t_emb

        processed_relations: List[Relation] = []
        _incremental_save_count = 0
        all_corroborated_family_ids: set = set()
        _t_loop = _time.monotonic()

        use_parallel = max_workers is not None and max_workers > 1 and len(relations_by_pair) > 1
        total_pairs = len(relations_by_pair)
        _rel_done = 0

        if use_parallel:
            pair_items = list(relations_by_pair.items())
            results: List[Optional[Tuple[List[Relation], List[Relation]]]] = [None] * len(pair_items)
            _distill_step = self.llm_client._current_distill_step
            _priority = getattr(self.llm_client._priority_local, 'priority', 6)

            def task(idx: int, pair_key: Tuple[str, str], pair_relations: List[Dict[str, str]]):
                # 将主线程的 distill step 和优先级传播到工作线程（threading.local）
                self.llm_client._current_distill_step = _distill_step
                self.llm_client._priority_local.priority = _priority
                existing_relations = existing_relations_by_pair.get(pair_key, [])
                entity1_name, entity2_name = _get_entity_names(pair_relations[0])
                return idx, self._process_one_relation_pair(
                    pair_key=pair_key,
                    pair_relations=pair_relations,
                    existing_relations=existing_relations,
                    entity1_name=entity1_name,
                    entity2_name=entity2_name,
                    episode_id=episode_id,
                    source_document=source_document,
                    base_time=base_time,
                    fallback_to_single=fallback_to_single,
                    verbose_relation=verbose_relation,
                    entity_lookup=entity_lookup,
                    embedding_ctx=embedding_ctx,
                )

            executor = _get_rel_pool(max_workers)
            futures = {
                executor.submit(task, idx, pair_key, pair_relations): idx
                for idx, (pair_key, pair_relations) in enumerate(pair_items)
            }
            for future in as_completed(futures):
                idx, pair_result = future.result()
                results[idx] = pair_result
                _rel_done += 1
                if on_relation_done:
                    on_relation_done(_rel_done, total_pairs)
            _all_to_persist = []
            _all_corrob_fids = set()
            for res in results:
                if res is None:
                    continue
                proc, to_persist, corrob_fids = res
                if proc:
                    processed_relations.extend(proc)
                if to_persist:
                    _all_to_persist.extend(to_persist)
                if corrob_fids:
                    _all_corrob_fids.update(corrob_fids)
            if _all_to_persist:
                _missing_embedding_relations = [
                    _rel for _rel in _all_to_persist
                    if not getattr(_rel, "embedding", None)
                ]
                batch_embed_fn = getattr(self.storage, '_compute_relation_embeddings_batch', None)
                if batch_embed_fn and _missing_embedding_relations:
                    try:
                        for _rel, _emb_result in zip(_missing_embedding_relations, batch_embed_fn(_missing_embedding_relations)):
                            if _emb_result is not None:
                                _rel.embedding = _emb_result
                    except Exception:
                        for _rel in _missing_embedding_relations:
                            try:
                                _emb_result = self.storage._compute_relation_embedding(_rel)
                                if _emb_result is not None:
                                    _rel.embedding = _emb_result
                            except Exception:
                                pass
                elif _missing_embedding_relations:
                    for _rel in _missing_embedding_relations:
                        try:
                            _emb_result = self.storage._compute_relation_embedding(_rel)
                            if _emb_result is not None:
                                _rel.embedding = _emb_result
                        except Exception:
                            pass
                try:
                    self.storage.bulk_save_relations_with_embedding(_all_to_persist)
                except Exception as _bulk_err:
                    _saved = 0
                    for _rel in _all_to_persist:
                        try:
                            self.storage.save_relation(_rel)
                            _saved += 1
                        except Exception as _e:
                            _log_fn(f"[relation_persist] 逐条保存失败: {getattr(_rel, 'name', '?')} -> {_e}")
                    _log_fn(f"[relation_persist] 批量写入失败({type(_bulk_err).__name__}: {_bulk_err}), 逐条保存成功 {_saved}/{len(_all_to_persist)}")
                _incremental_save_count += len(_all_to_persist)
                _all_patches = []
                for _rel in _all_to_persist:
                    _rel_patches = getattr(_rel, '_pending_patches', None) or []
                    _all_patches.extend(_rel_patches)
                if _all_patches:
                    try:
                        self.storage.save_content_patches(_all_patches)
                    except Exception:
                        pass
            all_corroborated_family_ids.update(_all_corrob_fids)
        else:
            _sum_pair_time = 0.0
            _seq_all_persist = []
            for pair_key, pair_relations in relations_by_pair.items():
                _t_pair = _time.monotonic()
                entity1_id, entity2_id = pair_key
                entity1_name, entity2_name = _get_entity_names(pair_relations[0])
                existing_relations = existing_relations_by_pair.get(pair_key, [])
                proc, to_persist, corrob_fids = self._process_one_relation_pair(
                    pair_key=pair_key,
                    pair_relations=pair_relations,
                    existing_relations=existing_relations,
                    entity1_name=entity1_name,
                    entity2_name=entity2_name,
                    episode_id=episode_id,
                    source_document=source_document,
                    base_time=base_time,
                    fallback_to_single=fallback_to_single,
                    verbose_relation=verbose_relation,
                    entity_lookup=entity_lookup,
                    embedding_ctx=embedding_ctx,
                )
                _pair_elapsed = _time.monotonic() - _t_pair
                _sum_pair_time += _pair_elapsed
                if _pair_elapsed > 0.05:
                    _existing = existing_relations_by_pair.get(pair_key, [])
                    _path = "LLM" if _pair_elapsed > 0.5 else ("VERSION" if len(_existing) > 0 and len(to_persist) > 0 else "NEW")
                    dbg(f"[step10_timing] pair {_rel_done+1}/{total_pairs}: {_pair_elapsed:.2f}s ({entity1_name}-{entity2_name}, exist={len(_existing)}, persist={len(to_persist)}, {_path})")
                if proc:
                    processed_relations.extend(proc)
                if to_persist:
                    _seq_all_persist.extend(to_persist)
                if corrob_fids:
                    all_corroborated_family_ids.update(corrob_fids)
                _rel_done += 1
                if on_relation_done:
                    on_relation_done(_rel_done, total_pairs)
            if _seq_all_persist:
                _missing_embedding_relations = [
                    _rel for _rel in _seq_all_persist
                    if not getattr(_rel, "embedding", None)
                ]
                batch_embed_fn = getattr(self.storage, '_compute_relation_embeddings_batch', None)
                if batch_embed_fn and _missing_embedding_relations:
                    try:
                        for _rel, _emb_result in zip(_missing_embedding_relations, batch_embed_fn(_missing_embedding_relations)):
                            if _emb_result is not None:
                                _rel.embedding = _emb_result
                    except Exception:
                        for _rel in _missing_embedding_relations:
                            try:
                                _emb_result = self.storage._compute_relation_embedding(_rel)
                                if _emb_result is not None:
                                    _rel.embedding = _emb_result
                            except Exception:
                                pass
                elif _missing_embedding_relations:
                    for _rel in _missing_embedding_relations:
                        try:
                            _emb_result = self.storage._compute_relation_embedding(_rel)
                            if _emb_result is not None:
                                _rel.embedding = _emb_result
                        except Exception:
                            pass
                try:
                    self.storage.bulk_save_relations_with_embedding(_seq_all_persist)
                except Exception as _bulk_err:
                    _saved = 0
                    for _rel in _seq_all_persist:
                        try:
                            self.storage.save_relation(_rel)
                            _saved += 1
                        except Exception as _e:
                            _log_fn(f"[relation_persist] 逐条保存失败: {getattr(_rel, 'name', '?')} -> {_e}")
                    _log_fn(f"[relation_persist] 批量写入失败({type(_bulk_err).__name__}: {_bulk_err}), 逐条保存成功 {_saved}/{len(_seq_all_persist)}")
                _incremental_save_count += len(_seq_all_persist)
                _seq_all_patches = []
                for _rel in _seq_all_persist:
                    _rel_patches = getattr(_rel, '_pending_patches', None) or []
                    _seq_all_patches.extend(_rel_patches)
                if _seq_all_patches:
                    try:
                        self.storage.save_content_patches(_seq_all_patches)
                    except Exception:
                        pass

        dbg(f"[step10_timing] process loop: {_time.monotonic()-_t_loop:.2f}s ({_incremental_save_count} saved incrementally)")
        if window_timings_ref is not None:
            window_timings_ref["step10b-process_loop"] = _time.monotonic() - _t_loop

        # Incremental refresh: fix edges for entity families involved in this batch
        if relations_by_pair and hasattr(self.storage, 'refresh_relates_to_edges'):
            _refresh_fids = list({fid for pair in relations_by_pair for fid in pair})
            _t_ref = _time.monotonic()
            self.storage.refresh_relates_to_edges(family_ids=_refresh_fids)
            dbg(f"[step10_timing] refresh_relates_to_edges: {_time.monotonic()-_t_ref:.2f}s ({len(_refresh_fids)} families)")
            if window_timings_ref is not None:
                window_timings_ref["step10c-refresh_edges"] = _time.monotonic() - _t_ref

        # Confidence corroboration: version-updated relations get confidence boost
        if all_corroborated_family_ids:
            try:
                self.storage.adjust_confidence_on_corroboration_batch(
                    list(all_corroborated_family_ids), source_type="relation",
                )
            except Exception:
                pass

        # Flush queued corroboration from single-relation fallback path
        self.flush_corroboration_batch()

        # Vision 原则「内容版本和关联解耦」：MENTIONS 必须无条件建立。
        # processed_relations 可能不包含所有 resolved pair 的关系（如 _build_new_relation 返回 None）。
        # 从 existing_relations_by_pair 补充缺失的 relation absolute_ids。
        seen_abs_ids = {r.absolute_id for r in processed_relations if r and r.absolute_id}
        for pair_key, existing in existing_relations_by_pair.items():
            for r in existing:
                if r.absolute_id and r.absolute_id not in seen_abs_ids:
                    processed_relations.append(r)
                    seen_abs_ids.add(r.absolute_id)

        return processed_relations

    def _process_one_relation_pair(self,
                                   pair_key: Tuple[str, str],
                                   pair_relations: List[Dict[str, str]],
                                   existing_relations: List[Relation],
                                   entity1_name: str,
                                   entity2_name: str,
                                   episode_id: str,
                                   source_document: str = "",
                                   base_time: Optional[datetime] = None,
                                   fallback_to_single: bool = True,
                                   verbose_relation: bool = True,
                                   entity_lookup: Optional[Dict[str, Any]] = None,
                                   embedding_ctx: Optional[Dict[str, Any]] = None,
                                   ) -> Tuple[List[Relation], List[Relation], set]:
        """处理单个实体对的关系，返回 (processed_relations, relations_to_persist, corroborated_family_ids)。"""
        dbg(f"ENTER {entity1_name}-{entity2_name}: ctx={embedding_ctx is not None} exist={len(existing_relations)}")
        entity1_id, entity2_id = pair_key
        processed_relations: List[Relation] = []
        relations_to_persist: List[Relation] = []
        corroborated_family_ids: set = set()
        new_contents = [c for rel in pair_relations if (c := rel.get("content", ""))]

        # 快速检查：如果所有新内容都已在已有关系中完全存在，跳过 LLM 但创建版本
        # Pre-build indexes for O(1) lookups (avoid repeated linear scans)
        _existing_by_fid = {r.family_id: r for r in existing_relations if r.family_id}
        _existing_by_content_lower = {r.content.strip().lower(): r for r in existing_relations if r.content}
        _existing_contents_lower = _existing_by_content_lower.keys()
        # Single pass: compute lower keys once, cache for reuse below
        _new_content_lower: Dict[str, str] = {}
        _lower_to_orig: Dict[str, str] = {}
        truly_new_contents = []
        for c in new_contents:
            key = c.strip().lower()
            _new_content_lower[c] = key
            _lower_to_orig[key] = c
            if key not in _existing_contents_lower:
                truly_new_contents.append(key)
        dbg(f"[step10_path] {entity1_name}-{entity2_name}: existing={len(existing_relations)}, new={len(new_contents)}, truly_new={len(truly_new_contents)}")
        if not truly_new_contents:
            # 所有新内容都已是已有关系的精确重复 → 直接复用，跳过 _construct_relation
            processed_relations.extend(existing_relations)
            for r in existing_relations:
                if r.family_id:
                    corroborated_family_ids.add(r.family_id)
            return processed_relations, relations_to_persist, corroborated_family_ids

        if self.preserve_distinct_relations_per_pair:
            for merged_relation in pair_relations:
                relation = self._process_single_relation(
                    merged_relation,
                    entity1_id,
                    entity2_id,
                    episode_id,
                    entity1_name,
                    entity2_name,
                    verbose_relation=verbose_relation,
                    source_document=source_document,
                    base_time=base_time,
                    pre_fetched_relations=existing_relations,
                )
                if relation:
                    processed_relations.append(relation)
            return processed_relations, relations_to_persist, corroborated_family_ids

        # No existing relations → create directly, skip LLM
        if not existing_relations:
            dbg(f"[step10_path] NO_EXISTING: {entity1_name}-{entity2_name} ({len(truly_new_contents)} new)")
            if len(truly_new_contents) == 1:
                merged_content = truly_new_contents[0]
            else:
                merged_content = "；".join(truly_new_contents[:3])
            new_rel = self._build_new_relation(
                entity1_id, entity2_id, merged_content, episode_id,
                entity1_name=entity1_name, entity2_name=entity2_name,
                source_document=source_document, base_time=base_time,
                entity_lookup=entity_lookup,
            )
            if new_rel:
                if embedding_ctx and len(truly_new_contents) == 1:
                    _emb = embedding_ctx.get('new_embs', {}).get(merged_content)
                    new_rel.embedding = _embedding_to_bytes(_emb) or new_rel.embedding
                processed_relations.append(new_rel)
                relations_to_persist.append(new_rel)
            return processed_relations, relations_to_persist, corroborated_family_ids

        # ---- Embedding 相似度快速过滤 ----
        if embedding_ctx and existing_relations:
            _EMB_NEW_THRESHOLD = self.emb_new_threshold

            _new_embs = embedding_ctx.get('new_embs', {})
            _existing_embs = embedding_ctx.get('existing_embs', {})

            # truly_new_contents = lower-cased keys; _new_embs keys = original content
            # _lower_to_orig built in initial pass above

            max_sim = 0.0
            _emb_miss_reason = ""
            for c_key in truly_new_contents:
                orig = _lower_to_orig.get(c_key)
                if not orig:
                    _emb_miss_reason = "no_orig"
                    continue
                new_emb = _new_embs.get(orig)
                if new_emb is None:
                    _emb_miss_reason = f"no_new_emb(orig={orig[:40]})"
                    continue
                for r in existing_relations:
                    exist_emb = _existing_embs.get(r.family_id)
                    if exist_emb is None:
                        continue
                    sim = cosine_similarity(new_emb, exist_emb)
                    if sim > max_sim:
                        max_sim = sim
            dbg(f"{entity1_name}-{entity2_name}: max_sim={max_sim:.4f} exist={len(existing_relations)} new_embs={len(_new_embs)} exist_embs={len(_existing_embs)} truly_new={len(truly_new_contents)} miss={_emb_miss_reason or 'none'} decision={'NEW' if max_sim < _EMB_NEW_THRESHOLD else 'LLM'}")

            if max_sim < _EMB_NEW_THRESHOLD:
                if len(truly_new_contents) == 1:
                    merged_content = _lower_to_orig.get(truly_new_contents[0], truly_new_contents[0])
                else:
                    merged_content = "；".join(truly_new_contents[:3])
                new_rel = self._build_new_relation(
                    entity1_id, entity2_id, merged_content, episode_id,
                    entity1_name=entity1_name, entity2_name=entity2_name,
                    source_document=source_document, base_time=base_time,
                    entity_lookup=entity_lookup,
                )
                if new_rel:
                    if len(truly_new_contents) == 1:
                        _emb = _new_embs.get(merged_content)
                        new_rel.embedding = _embedding_to_bytes(_emb) or new_rel.embedding
                    processed_relations.append(new_rel)
                    relations_to_persist.append(new_rel)
                return processed_relations, relations_to_persist, corroborated_family_ids

        # Build relations info only when needed for LLM call (lazy construction)
        existing_relations_info = [
            {
                "family_id": relation.family_id,
                "content": relation.content,
                "source_document": relation.source_document,
            }
            for relation in existing_relations
        ]

        batch_result = self.llm_client.resolve_relation_pair_batch(
            entity1_name=entity1_name,
            entity2_name=entity2_name,
            new_relation_contents=new_contents,
            existing_relations=existing_relations_info,
            new_source_document=_doc_basename(source_document),
        )

        _action = batch_result.get("action", "")
        confidence = float(batch_result.get("confidence", 0.0) or 0.0)
        if (not self.batch_resolution_enabled) or _action == "fallback" or (confidence < self.batch_resolution_confidence_threshold and fallback_to_single):
            for merged_relation in pair_relations:
                relation = self._process_single_relation(
                    merged_relation,
                    entity1_id,
                    entity2_id,
                    episode_id,
                    entity1_name,
                    entity2_name,
                    verbose_relation=verbose_relation,
                    source_document=source_document,
                    base_time=base_time,
                    pre_fetched_relations=existing_relations,
                    _pre_built_relations_info=existing_relations_info,
                )
                if relation:
                    processed_relations.append(relation)
            return processed_relations, relations_to_persist, corroborated_family_ids

        if _action == "match_existing":
            matched_family_id = batch_result.get("matched_relation_id") or ""
            latest_relation = _existing_by_fid.get(matched_family_id)
            _need_update = batch_result.get("need_update")
            if latest_relation and _need_update:
                merged_content = (batch_result.get("merged_content") or "").strip()
                if not merged_content:
                    merged_content = self.llm_client.merge_multiple_relation_contents(
                        [latest_relation.content] + new_contents,
                        relation_sources=[latest_relation.source_document] + [source_document] * len(new_contents),
                        entity_pair=(entity1_name, entity2_name),
                    )
                new_relation = self._build_relation_version(
                    matched_family_id,
                    entity1_id,
                    entity2_id,
                    merged_content,
                    episode_id,
                    source_document=source_document,
                    entity1_name=entity1_name,
                    entity2_name=entity2_name,
                    base_time=base_time,
                    entity_lookup=entity_lookup,
                    old_content=latest_relation.content or "",
                    old_content_format=latest_relation.content_format or "plain",
                )
                if new_relation is not None:
                    if (merged_content or "").strip() == (latest_relation.content or "").strip():
                        new_relation.embedding = latest_relation.embedding
                    relations_to_persist.append(new_relation)
                    processed_relations.append(new_relation)
                    corroborated_family_ids.add(matched_family_id)
                else:
                    # LLM said need_update but content is whitespace-equivalent — keep existing
                    processed_relations.append(latest_relation)
            elif latest_relation:
                # 始终创建版本（内容无变化时复制已有内容）
                new_relation = self._build_relation_version(
                    matched_family_id,
                    entity1_id,
                    entity2_id,
                    latest_relation.content,
                    episode_id,
                    source_document=source_document,
                    entity1_name=entity1_name,
                    entity2_name=entity2_name,
                    base_time=base_time,
                    entity_lookup=entity_lookup,
                    old_content=latest_relation.content or "",
                    old_content_format=latest_relation.content_format or "plain",
                )
                if new_relation is not None:
                    new_relation.embedding = latest_relation.embedding
                    relations_to_persist.append(new_relation)
                    processed_relations.append(new_relation)
                    corroborated_family_ids.add(matched_family_id)
                else:
                    processed_relations.append(latest_relation)
            else:
                fallback_content = batch_result.get("merged_content")
                if not fallback_content:
                    fallback_content = self.llm_client.merge_multiple_relation_contents(
                        new_contents,
                        relation_sources=[source_document] * len(new_contents),
                        entity_pair=(entity1_name, entity2_name),
                    )
                new_relation = self._build_new_relation(
                    entity1_id,
                    entity2_id,
                    fallback_content,
                    episode_id,
                    entity1_name=entity1_name,
                    entity2_name=entity2_name,
                    source_document=source_document,
                    base_time=base_time,
                    entity_lookup=entity_lookup,
                    confidence=confidence,
                )
                if new_relation is not None:
                    if embedding_ctx:
                        _emb = embedding_ctx.get('new_embs', {}).get(fallback_content)
                        new_relation.embedding = _embedding_to_bytes(_emb) or new_relation.embedding
                    relations_to_persist.append(new_relation)
                    processed_relations.append(new_relation)
        else:
            _merged_raw = batch_result.get("merged_content") or ""
            merged_content = _merged_raw.strip()
            if not merged_content:
                merged_content = self.llm_client.merge_multiple_relation_contents(
                    new_contents,
                    relation_sources=[source_document] * len(new_contents),
                    entity_pair=(entity1_name, entity2_name),
                )
            new_relation = self._build_new_relation(
                entity1_id,
                entity2_id,
                merged_content,
                episode_id,
                entity1_name=entity1_name,
                entity2_name=entity2_name,
                source_document=source_document,
                base_time=base_time,
                entity_lookup=entity_lookup,
                confidence=confidence,
            )
            if new_relation is not None:
                if embedding_ctx:
                    _emb = embedding_ctx.get('new_embs', {}).get(merged_content)
                    new_relation.embedding = _embedding_to_bytes(_emb) or new_relation.embedding
                relations_to_persist.append(new_relation)
                processed_relations.append(new_relation)
        return processed_relations, relations_to_persist, corroborated_family_ids
