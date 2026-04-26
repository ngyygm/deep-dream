"""
关系处理模块：关系搜索、对齐、更新/新建
"""
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

from core.models import Relation
from core.storage.manager import StorageManager
from core.llm.client import LLMClient
from core.debug_log import log as dbg, log_section as dbg_section, _ENABLED as _dbg_enabled
from core.utils import wprint_info, normalize_entity_pair
from .helpers import MIN_RELATION_CONTENT_LENGTH


def _get_entity_names(relation: Dict[str, str]) -> Tuple[str, str]:
    """Extract entity names from relation dict, supporting both old and new formats."""
    return (
        relation.get('entity1_name') or relation.get('from_entity_name', ''),
        relation.get('entity2_name') or relation.get('to_entity_name', ''),
    )

# Shared pool for corroboration calls (avoids per-call thread churn)
_corrob_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="corrob")

# Shared pool for batch relation processing
_REL_POOL: ThreadPoolExecutor | None = None
_REL_POOL_MAX = 2

def _get_rel_pool(max_workers: int) -> ThreadPoolExecutor:
    """Return (and lazily create/upgrade) the shared relation ThreadPoolExecutor."""
    global _REL_POOL, _REL_POOL_MAX
    if _REL_POOL is not None:
        if max_workers > _REL_POOL_MAX:
            try:
                _REL_POOL.shutdown(wait=False)
            except Exception:
                pass
            _REL_POOL = None
        else:
            return _REL_POOL
    _REL_POOL_MAX = max(max_workers, _REL_POOL_MAX)
    _REL_POOL = ThreadPoolExecutor(
        max_workers=_REL_POOL_MAX,
        thread_name_prefix="tmg-rel",
    )
    return _REL_POOL


def _doc_basename(source_document: str) -> str:
    return source_document.rpartition('/')[-1] if source_document else ""


class RelationProcessor:
    """关系处理器 - 负责关系的搜索、对齐、更新和新建"""
    
    def __init__(self, storage: StorageManager, llm_client: LLMClient):
        self.storage = storage
        self.llm_client = llm_client
        self.batch_resolution_enabled = True
        self.batch_resolution_confidence_threshold = 0.70
        self.preserve_distinct_relations_per_pair = False
        self._corroboration_queue: List[str] = []  # Batch corroboration family_ids
    
    def build_relations_by_pair_from_inputs(
        self,
        extracted_relations: List[Dict[str, str]],
        entity_name_to_id: Dict[str, str],
    ) -> Tuple[Dict[Tuple[str, str], List[Dict[str, str]]], int]:
        """去重合并后按实体对分组，不含读库。供步骤7跨窗预取，与 process_relations_batch 前半段一致。"""
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

        existing_relations_by_pair = self.storage.get_relations_by_entity_pairs(list(relations_by_pair))

        # 批量预取所有涉及的实体，避免 _build_new_relation/_build_relation_version 中逐对查询
        unique_eids = set()
        for e1, e2 in relations_by_pair:
            unique_eids.add(e1)
            unique_eids.add(e2)
        entity_lookup: Dict[str, Any] = {}
        if unique_eids:
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

        processed_relations: List[Relation] = []
        relations_to_persist: List[Relation] = []
        all_corroborated_family_ids: set = set()

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
            for res in results:
                if res is None:
                    continue
                proc, to_persist, corrob_fids = res
                if proc:
                    processed_relations.extend(proc)
                if to_persist:
                    relations_to_persist.extend(to_persist)
                if corrob_fids:
                    all_corroborated_family_ids.update(corrob_fids)
        else:
            for pair_key, pair_relations in relations_by_pair.items():
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
                )
                if proc:
                    processed_relations.extend(proc)
                if to_persist:
                    relations_to_persist.extend(to_persist)
                if corrob_fids:
                    all_corroborated_family_ids.update(corrob_fids)
                _rel_done += 1
                if on_relation_done:
                    on_relation_done(_rel_done, total_pairs)

        if relations_to_persist:
            self.storage.bulk_save_relations(relations_to_persist)
            # Incremental refresh: only fix edges for entity families involved in this batch
            if hasattr(self.storage, 'refresh_relates_to_edges'):
                _refresh_fids = list({fid for pair in relations_by_pair for fid in pair})
                self.storage.refresh_relates_to_edges(family_ids=_refresh_fids)

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

        # Dream 候选层佐证：remember 提取的关系与 dream 候选关系匹配时，自动佐证
        if hasattr(self.storage, 'corroborate_dream_relations_batch'):
            try:
                self.storage.corroborate_dream_relations_batch(
                    list(relations_by_pair), corroboration_source="remember",
                )
            except Exception:
                pass
        else:
            # Batch corroborate in parallel to avoid N+1 sequential DB calls
            _pair_keys = list(relations_by_pair)
            if _pair_keys:
                def _corroborate_one(pair_key):
                    try:
                        self.storage.corroborate_dream_relation(
                            pair_key[0], pair_key[1], corroboration_source="remember",
                        )
                    except Exception:
                        pass

                if len(_pair_keys) > 1:
                    list(_corrob_pool.map(_corroborate_one, _pair_keys))
                else:
                    _corroborate_one(_pair_keys[0])

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
                                   entity_lookup: Optional[Dict[str, Any]] = None) -> Tuple[List[Relation], List[Relation], set]:
        """处理单个实体对的关系，返回 (processed_relations, relations_to_persist, corroborated_family_ids)。"""
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
        truly_new_contents = []
        for c in new_contents:
            key = c.strip().lower()
            _new_content_lower[c] = key
            if key not in _existing_contents_lower:
                truly_new_contents.append(key)
        if not truly_new_contents:
            # 所有新内容都已是已有关系的精确重复 → 为匹配的关系创建版本（复制内容）
            for nc in new_contents:
                _nc_lower = _new_content_lower[nc]  # reuse pre-computed lower key
                matched = _existing_by_content_lower.get(_nc_lower)
                if matched:
                    new_rel = self._build_relation_version(
                        matched.family_id, entity1_id, entity2_id,
                        matched.content, episode_id,
                        source_document=source_document, base_time=base_time,
                        entity1_name=entity1_name, entity2_name=entity2_name,
                        entity_lookup=entity_lookup,
                    )
                    if new_rel is not None:
                        relations_to_persist.append(new_rel)
                        processed_relations.append(new_rel)
                        corroborated_family_ids.add(matched.family_id)
                    else:
                        processed_relations.append(matched)
                else:
                    # 兜底：没匹配到则加入已有关系
                    processed_relations.extend(existing_relations)
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

        # ---- Fix 3: 无已有关系 → 直接创建新关系，跳过batch LLM ----
        if not existing_relations:
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
                )
                if new_relation is not None:
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
                )
                if new_relation is not None:
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
                relations_to_persist.append(new_relation)
                processed_relations.append(new_relation)
        return processed_relations, relations_to_persist, corroborated_family_ids

    def _dedupe_and_merge_relations(self, extracted_relations: List[Dict[str, str]],
                                    entity_name_to_id: Dict[str, str]) -> List[Dict[str, str]]:
        """
        对相同实体对的关系进行去重和合并
        
        Args:
            extracted_relations: 抽取的关系列表
            entity_name_to_id: 实体名称到family_id的映射
        
        Returns:
            去重合并后的关系列表
        """
        # 按实体对分组（使用标准化后的实体对，使关系无向化）
        relations_by_pair = {}
        filtered_count = 0
        filtered_relations = []
        dbg_section("RelationProcessor._dedupe_and_merge_relations")
        dbg(f"输入关系数: {len(extracted_relations)}")
        if _dbg_enabled:
            dbg(f"entity_name_to_id 映射 ({len(entity_name_to_id)} 个): {list(entity_name_to_id)[:20]}")
        
        for relation in extracted_relations:
            # 支持新旧格式
            entity1_name, entity2_name = _get_entity_names(relation)

            if not entity1_name or not entity2_name:
                filtered_count += 1
                filtered_relations.append({
                    'entity1': entity1_name or '(空)',
                    'entity2': entity2_name or '(空)',
                    'reason': '实体名称为空'
                })
                dbg(f"  过滤(空名): e1='{entity1_name}' e2='{entity2_name}'")
                continue
            
            # 检查实体ID是否存在
            missing_entities = []
            entity1_id = entity_name_to_id.get(entity1_name)
            entity2_id = entity_name_to_id.get(entity2_name)
            if not entity1_id:
                missing_entities.append(f'entity1: {entity1_name}')
            if not entity2_id:
                missing_entities.append(f'entity2: {entity2_name}')
            
            if missing_entities:
                filtered_count += 1
                filtered_relations.append({
                    'entity1': entity1_name,
                    'entity2': entity2_name,
                    'reason': f'实体不在当前窗口的实体列表中: {", ".join(missing_entities)}'
                })
                dbg(f"  过滤(不在映射): e1='{entity1_name}' e2='{entity2_name}' 缺少: {missing_entities}")
                continue
            
            # 检查两个实体是否是同一个实体（通过family_id比较）
            if entity1_id and entity2_id and entity1_id == entity2_id:
                filtered_count += 1
                filtered_relations.append({
                    'entity1': entity1_name,
                    'entity2': entity2_name,
                    'reason': f'两个实体是同一个实体（family_id: {entity1_id}）'
                })
                dbg(f"  过滤(自关系): e1='{entity1_name}' e2='{entity2_name}' family_id={entity1_id}")
                continue
            
            # 标准化实体对（按字母顺序排序，使关系无向化）
            normalized_pair = normalize_entity_pair(entity1_name, entity2_name)
            
            if normalized_pair not in relations_by_pair:
                relations_by_pair[normalized_pair] = []
            # 确保关系使用标准化后的实体对（only copy if names actually changed）
            _needs_copy = (entity1_name != normalized_pair[0] or entity2_name != normalized_pair[1])
            if _needs_copy:
                relation_copy = relation.copy()
                relation_copy['entity1_name'] = normalized_pair[0]
                relation_copy['entity2_name'] = normalized_pair[1]
            else:
                relation_copy = relation
            relations_by_pair[normalized_pair].append(relation_copy)
        
        # 对每个实体对的关系进行合并或保留多条语义关系
        merged_relations = []
        for pair, relations in relations_by_pair.items():
            if self.preserve_distinct_relations_per_pair:
                seen_contents = set()
                for relation in relations:
                    content_key = (relation.get('content') or '').strip().lower()
                    if not content_key or content_key in seen_contents:
                        continue
                    seen_contents.add(content_key)
                    merged_relations.append(relation)
                continue
            if len(relations) == 1:
                # 只有一个关系，直接添加
                merged_relations.append(relations[0])
            else:
                # 多个关系，需要合并
                merged_relation = self._merge_relations_for_pair(pair, relations)
                if merged_relation:
                    merged_relations.append(merged_relation)
        
        dbg(f"去重合并结果: 过滤 {filtered_count}, 合并后通过 {len(merged_relations)}")
        for _mr in merged_relations:
            dbg(f"  通过: '{_mr.get('entity1_name', '')}' <-> '{_mr.get('entity2_name', '')}'  content='{_mr.get('content', '')[:100]}'")

        return merged_relations
    
    def _merge_relations_for_pair(self, pair: tuple, 
                                  relations: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """
        合并同一实体对的多个关系
        
        Args:
            pair: 实体对 (entity1_name, entity2_name)
            relations: 该实体对的所有关系列表
        
        Returns:
            合并后的关系
        """
        if not relations:
            return None
        
        if len(relations) == 1:
            return relations[0]
        
        # 提取所有关系内容
        relation_contents = [c for rel in relations if (c := rel.get('content', ''))]
        
        if not relation_contents:
            return relations[0]  # 如果没有content，返回第一个
        
        if len(relation_contents) == 1:
            return relations[0]  # 只有一个有content的关系
        
        # 快速拼接，LLM合并推迟到process_relations_batch的批量阶段
        merged_content = "；".join(relation_contents)
        
        # 构建合并后的关系
        merged_relation = {
            'entity1_name': pair[0],
            'entity2_name': pair[1],
            'content': merged_content
        }
        
        return merged_relation
    
    def _process_single_relation(self, extracted_relation: Dict[str, str],
                                 entity1_id: str,
                                 entity2_id: str,
                                 episode_id: str,
                                 entity1_name: str = "",
                                 entity2_name: str = "",
                                 verbose_relation: bool = True,
                                 source_document: str = "",
                                 base_time: Optional[datetime] = None,
                                 pre_fetched_relations: Optional[List[Relation]] = None,
                                 _pre_built_relations_info: Optional[List[Dict]] = None) -> Optional[Relation]:
        """
        处理单个关系
        
        注意：参数 entity1_id 和 entity2_id 是实体的 family_id（不是绝对ID）
        在创建关系时，会通过 family_id 获取实体的最新版本，然后使用绝对ID存储
        
        流程：
        1. 根据两个实体ID查找所有已有关系
        2. 用LLM判断是否匹配
        3. 如果匹配且需要更新，更新；如果不匹配，新建
        """
        relation_content = extracted_relation['content']
        if not entity1_name or not entity2_name:
            _e1, _e2 = _get_entity_names(extracted_relation)
            entity1_name = entity1_name or _e1
            entity2_name = entity2_name or _e2
        # 步骤1：根据两个实体的 family_id 查找所有已有关系
        # 优先使用预获取的结果，避免冗余DB查询
        if pre_fetched_relations is not None:
            existing_relations = pre_fetched_relations
        else:
            existing_relations = self.storage.get_relations_by_entities(
                entity1_id,
                entity2_id
            )
        
        if not existing_relations:
            return self._create_new_relation(
                entity1_id,
                entity2_id,
                relation_content,
                episode_id,
                entity1_name,
                entity2_name,
                verbose_relation,
                source_document,
                base_time=base_time,
            )
        
        # 步骤2：准备已有关系信息供LLM判断
        # get_relations_by_entities 已按 family_id 去重，直接使用
        # Use pre-built info if available (avoids redundant dict construction in fallback path)
        existing_relations_info = _pre_built_relations_info or [
            {
                'family_id': r.family_id,
                'content': r.content,
                'source_document': r.source_document,
            }
            for r in existing_relations
        ]
        
        # 步骤3：用LLM判断是否匹配
        match_result = self.llm_client.judge_relation_match(
            extracted_relation,
            existing_relations_info,
            new_source_document=_doc_basename(source_document),
        )
        # LLM 有时返回 list 而非 dict，统一取第一个元素
        if isinstance(match_result, list) and match_result:
            match_result = match_result[0] if isinstance(match_result[0], dict) else None
        elif not isinstance(match_result, dict):
            match_result = None

        if match_result and match_result.get('family_id'):
            # 匹配到已有关系
            family_id = match_result['family_id']

            # 获取最新版本的content
            latest_relation = next(
                (r for r in existing_relations if r.family_id == family_id), None
            )
            if not latest_relation:
                return self._create_new_relation(
                    entity1_id,
                    entity2_id,
                    relation_content,
                    episode_id,
                    entity1_name,
                    entity2_name,
                    verbose_relation,
                    source_document,
                    base_time=base_time,
                )
            
            # 始终创建版本：快速比较内容是否变化（避免多余的合并 LLM 调用）
            _old_content = (latest_relation.content or "").strip()
            _new_content = relation_content.strip()
            if _old_content == _new_content:
                # 内容相同 → 直接复制创建版本（不调 LLM）
                new_relation = self._create_relation_version(
                    family_id,
                    entity1_id,
                    entity2_id,
                    latest_relation.content,
                    episode_id,
                    verbose_relation,
                    source_document,
                    entity1_name,
                    entity2_name,
                    base_time=base_time,
                )
                return new_relation
            else:
                # 内容不同 → 合并内容 + 创建版本
                record_count = 0
                if verbose_relation:
                    try:
                        vc_map = self.storage.get_relation_version_counts([family_id])
                        record_count = vc_map.get(family_id, 0)
                    except Exception:
                        pass

                merged_content = self.llm_client.merge_relation_content(
                    latest_relation.content,
                    relation_content,
                    old_source_document=latest_relation.source_document,
                    new_source_document=source_document,
                    entity1_name=entity1_name,
                    entity2_name=entity2_name,
                )

                if verbose_relation:
                    wprint_info(f"[关系操作] 🔄 更新关系: {entity1_name} <-> {entity2_name} (family_id: {family_id}, 版本数: {record_count})")

                new_relation = self._create_relation_version(
                    family_id,
                    entity1_id,
                    entity2_id,
                    merged_content,
                    episode_id,
                    verbose_relation,
                    source_document,
                    entity1_name,
                    entity2_name,
                    base_time=base_time,
                )

                return new_relation
        else:
            return self._create_new_relation(
                entity1_id,
                entity2_id,
                relation_content,
                episode_id,
                entity1_name,
                entity2_name,
                verbose_relation,
                source_document,
                base_time=base_time,
            )
    
    def _construct_relation(self, entity1_id: str, entity2_id: str,
                            content: str, episode_id: str,
                            family_id: str,
                            entity1_name: str = "", entity2_name: str = "",
                            verbose_relation: bool = True, source_document: str = "",
                            base_time: Optional[datetime] = None,
                            entity_lookup: Optional[Dict[str, Any]] = None,
                            skip_label: str = "关系创建",
                            confidence: Optional[float] = None) -> Optional[Relation]:
        """Shared helper: resolve entities, validate, and construct a Relation object.

        Args:
            confidence: Initial confidence from LLM extraction (0.0–1.0).
                        Falls back to 0.7 if not provided.
        """
        entity1 = (entity_lookup or {}).get(entity1_id) or self.storage.get_entity_by_family_id(entity1_id)
        entity2 = (entity_lookup or {}).get(entity2_id) or self.storage.get_entity_by_family_id(entity2_id)

        if not entity1 or not entity2:
            missing_info = []
            if not entity1:
                missing_info.append(f"entity1: {entity1_name or '(未提供名称)'} (family_id: {entity1_id})")
            if not entity2:
                missing_info.append(f"entity2: {entity2_name or '(未提供名称)'} (family_id: {entity2_id})")
            if verbose_relation:
                wprint_info(f"[关系操作] ⚠️  警告: 无法找到实体: {', '.join(missing_info)}，跳过{skip_label}")
            return None

        _now = datetime.now()
        ts = base_time if base_time is not None else _now
        processed_time = _now
        relation_record_id = f"relation_{processed_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        if entity1.name <= entity2.name:
            entity1_absolute_id, entity2_absolute_id = entity1.absolute_id, entity2.absolute_id
        else:
            entity1_absolute_id, entity2_absolute_id = entity2.absolute_id, entity1.absolute_id

        source_document_only = _doc_basename(source_document)
        initial_confidence = confidence if confidence is not None else 0.7
        initial_confidence = max(0.0, min(1.0, initial_confidence))
        return Relation(
            absolute_id=relation_record_id,
            family_id=family_id,
            entity1_absolute_id=entity1_absolute_id,
            entity2_absolute_id=entity2_absolute_id,
            content=content,
            event_time=ts,
            processed_time=processed_time,
            episode_id=episode_id,
            source_document=source_document_only,
            content_format="markdown",
            summary=content[:200].strip(),
            confidence=initial_confidence,
        )

    def _build_new_relation(self, entity1_id: str, entity2_id: str,
                            content: str, episode_id: str,
                            entity1_name: str = "", entity2_name: str = "",
                            verbose_relation: bool = True, source_document: str = "",
                            base_time: Optional[datetime] = None,
                            entity_lookup: Optional[Dict[str, Any]] = None,
                            confidence: Optional[float] = None) -> Optional[Relation]:
        """构建新关系对象，但不立即写库。"""
        _cs = content.strip() if content else ""
        if len(_cs) < MIN_RELATION_CONTENT_LENGTH:
            if verbose_relation:
                wprint_info(f"[关系操作] ⚠️  跳过: 关系内容过短 ({len(_cs)}字符): {entity1_name} <-> {entity2_name}")
            return None

        return self._construct_relation(
            entity1_id, entity2_id, content, episode_id,
            family_id=f"rel_{uuid.uuid4().hex[:12]}",
            entity1_name=entity1_name, entity2_name=entity2_name,
            verbose_relation=verbose_relation, source_document=source_document,
            base_time=base_time, entity_lookup=entity_lookup,
            skip_label="关系创建",
            confidence=confidence,
        )

    def _create_new_relation(self, entity1_id: str, entity2_id: str,
                            content: str, episode_id: str,
                            entity1_name: str = "", entity2_name: str = "",
                            verbose_relation: bool = True, source_document: str = "",
                            base_time: Optional[datetime] = None,
                            confidence: Optional[float] = None) -> Optional[Relation]:
        """创建新关系"""
        relation = self._build_new_relation(
            entity1_id, entity2_id, content, episode_id,
            entity1_name=entity1_name, entity2_name=entity2_name,
            verbose_relation=verbose_relation, source_document=source_document, base_time=base_time,
            confidence=confidence,
        )
        if relation:
            self.storage.save_relation(relation)
            if verbose_relation:
                wprint_info(f"[关系操作] ✅ 创建新关系: {entity1_name} <-> {entity2_name} (family_id: {relation.family_id})")
        return relation

    def _build_relation_version(self, family_id: str, entity1_id: str,
                                 entity2_id: str, content: str,
                                 episode_id: str,
                                 verbose_relation: bool = True,
                                 source_document: str = "",
                                 entity1_name: str = "",
                                 entity2_name: str = "",
                                 base_time: Optional[datetime] = None,
                                 entity_lookup: Optional[Dict[str, Any]] = None,
                                 _existing_relation: Optional[Relation] = None) -> Optional[Relation]:
        """构建关系新版本对象，但不立即写库。"""
        # 内容过短时，尝试使用已有关系的内容（始终版本化原则）
        _cs = content.strip() if content else ""
        if len(_cs) < MIN_RELATION_CONTENT_LENGTH:
            if _existing_relation and _existing_relation.content and len(_existing_relation.content.strip()) >= MIN_RELATION_CONTENT_LENGTH:
                content = _existing_relation.content
            else:
                # 兜底：获取存储中的最新版本内容
                try:
                    versions = self.storage.get_relation_versions(family_id)
                    for v in versions:
                        if v.content and len(v.content.strip()) >= MIN_RELATION_CONTENT_LENGTH:
                            content = v.content
                            break
                except Exception:
                    pass
            # 如果仍然太短，则跳过（无法创建有意义的版本）
            _cs2 = content.strip() if content else ""
            if len(_cs2) < MIN_RELATION_CONTENT_LENGTH:
                if verbose_relation:
                    wprint_info(f"[关系操作] ⚠️  跳过版本: 内容过短且无可用历史内容 ({len(_cs2)}字符): {family_id}")
                return None

        # 始终构建版本（每个 episode 提及的关系都版本化）

        return self._construct_relation(
            entity1_id, entity2_id, content, episode_id,
            family_id=family_id,
            entity1_name=entity1_name, entity2_name=entity2_name,
            verbose_relation=verbose_relation, source_document=source_document,
            base_time=base_time, entity_lookup=entity_lookup,
            skip_label="关系版本创建",
        )

    def _create_relation_version(self, family_id: str, entity1_id: str,
                                 entity2_id: str, content: str,
                                 episode_id: str,
                                 verbose_relation: bool = True,
                                 source_document: str = "",
                                 entity1_name: str = "",
                                 entity2_name: str = "",
                                 base_time: Optional[datetime] = None,
                                 entity_lookup: Optional[Dict[str, Any]] = None) -> Optional[Relation]:
        """创建关系的新版本（始终创建，不跳过）。"""
        # 始终创建新版本（每个 episode 提及的关系都版本化）
        relation = self._build_relation_version(
            family_id, entity1_id, entity2_id, content, episode_id,
            verbose_relation=verbose_relation, source_document=source_document,
            entity1_name=entity1_name, entity2_name=entity2_name, base_time=base_time,
            entity_lookup=entity_lookup,
            _existing_relation=None,
        )
        if relation:
            self.storage.save_relation(relation)
            # 置信度演化：收集 family_id 批量处理（避免 N+1 单条 UPDATE）
            self._corroboration_queue.append(family_id)
        return relation

    def flush_corroboration_batch(self):
        """Flush queued corroboration updates as a single batch SQL operation."""
        if not self._corroboration_queue:
            return
        unique_fids = list(set(self._corroboration_queue))
        self._corroboration_queue.clear()
        try:
            self.storage.adjust_confidence_on_corroboration_batch(
                unique_fids, source_type="relation",
            )
        except Exception:
            # Fallback: use individual calls
            for fid in unique_fids:
                try:
                    self.storage.adjust_confidence_on_corroboration(fid, source_type="relation")
                except Exception:
                    pass
