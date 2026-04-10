"""
关系处理模块：关系搜索、对齐、更新/新建
"""
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

from ..models import Relation
from ..storage.manager import StorageManager
from ..llm.client import LLMClient
from ..debug_log import log as dbg, log_section as dbg_section
from ..utils import wprint, normalize_entity_pair


class RelationProcessor:
    """关系处理器 - 负责关系的搜索、对齐、更新和新建"""
    
    def __init__(self, storage: StorageManager, llm_client: LLMClient):
        self.storage = storage
        self.llm_client = llm_client
        self.batch_resolution_enabled = True
        self.batch_resolution_confidence_threshold = 0.70
    
    def build_relations_by_pair_from_inputs(
        self,
        extracted_relations: List[Dict[str, str]],
        entity_name_to_id: Dict[str, str],
    ) -> Tuple[Dict[Tuple[str, str], List[Dict[str, str]]], int]:
        """去重合并后按实体对分组，不含读库。供步骤7跨窗预取，与 process_relations_batch 前半段一致。"""
        merged_relations = self._dedupe_and_merge_relations(extracted_relations, entity_name_to_id)
        if not merged_relations:
            return {}, 0

        relations_by_pair: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
        _batch_filtered = 0
        for merged_relation in merged_relations:
            entity1_name = merged_relation.get('entity1_name') or merged_relation.get('from_entity_name', '')
            entity2_name = merged_relation.get('entity2_name') or merged_relation.get('to_entity_name', '')
            if not entity1_name or not entity2_name:
                _batch_filtered += 1
                continue
            entity1_id = entity_name_to_id.get(entity1_name)
            entity2_id = entity_name_to_id.get(entity2_name)
            if not entity1_id or not entity2_id or entity1_id == entity2_id:
                _batch_filtered += 1
                continue
            pair_key = tuple(sorted((entity1_id, entity2_id)))
            relations_by_pair.setdefault(pair_key, []).append(merged_relation)

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
            merged_relations = self._dedupe_and_merge_relations(extracted_relations, entity_name_to_id)
            dbg(f"process_relations_batch: 去重合并后 {len(merged_relations)} 个关系")
            if not merged_relations:
                return []

            relations_by_pair = {}
            _batch_filtered = 0
            for merged_relation in merged_relations:
                entity1_name = merged_relation.get('entity1_name') or merged_relation.get('from_entity_name', '')
                entity2_name = merged_relation.get('entity2_name') or merged_relation.get('to_entity_name', '')
                if not entity1_name or not entity2_name:
                    dbg(f"  batch过滤(空名): e1='{entity1_name}' e2='{entity2_name}'")
                    _batch_filtered += 1
                    continue
                entity1_id = entity_name_to_id.get(entity1_name)
                entity2_id = entity_name_to_id.get(entity2_name)
                if not entity1_id or not entity2_id or entity1_id == entity2_id:
                    dbg(f"  batch过滤(无ID/自关系): e1='{entity1_name}'(id={entity1_id}) e2='{entity2_name}'(id={entity2_id})")
                    _batch_filtered += 1
                    continue
                pair_key = tuple(sorted((entity1_id, entity2_id)))
                relations_by_pair.setdefault(pair_key, []).append(merged_relation)

            dbg(f"process_relations_batch: 第二次过滤 {_batch_filtered} 个, 剩余 {len(relations_by_pair)} 个实体对")

        existing_relations_by_pair = self.storage.get_relations_by_entity_pairs(list(relations_by_pair.keys()))

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
                entity1_name = pair_relations[0].get('entity1_name') or pair_relations[0].get('from_entity_name', '')
                entity2_name = pair_relations[0].get('entity2_name') or pair_relations[0].get('to_entity_name', '')
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

            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tmg-llm") as executor:
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
                proc, to_persist = res
                if proc:
                    processed_relations.extend(proc)
                if to_persist:
                    relations_to_persist.extend(to_persist)
        else:
            for pair_key, pair_relations in relations_by_pair.items():
                entity1_id, entity2_id = pair_key
                entity1_name = pair_relations[0].get('entity1_name') or pair_relations[0].get('from_entity_name', '')
                entity2_name = pair_relations[0].get('entity2_name') or pair_relations[0].get('to_entity_name', '')
                existing_relations = existing_relations_by_pair.get(pair_key, [])
                proc, to_persist = self._process_one_relation_pair(
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
                _rel_done += 1
                if on_relation_done:
                    on_relation_done(_rel_done, total_pairs)

        if relations_to_persist:
            self.storage.bulk_save_relations(relations_to_persist)
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
                                   entity_lookup: Optional[Dict[str, Any]] = None) -> Tuple[List[Relation], List[Relation]]:
        """处理单个实体对的关系，返回 (processed_relations, relations_to_persist)。"""
        entity1_id, entity2_id = pair_key
        processed_relations: List[Relation] = []
        relations_to_persist: List[Relation] = []
        new_contents = [rel.get("content", "") for rel in pair_relations if rel.get("content", "")]
        existing_relations_info = [
            {
                "family_id": relation.family_id,
                "content": relation.content,
                "source_document": relation.source_document,
            }
            for relation in existing_relations
        ]

        # 快速检查：如果所有新内容都已在已有关系中完全存在，直接跳过 LLM 调用
        existing_contents_lower = {r.content.strip().lower() for r in existing_relations}
        truly_new_contents = [c for c in new_contents if c.strip().lower() not in existing_contents_lower]
        if not truly_new_contents:
            # 所有新内容都已是已有关系的精确重复，无需处理
            processed_relations.extend(existing_relations)
            return processed_relations, relations_to_persist

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
            return processed_relations, relations_to_persist

        batch_result = self.llm_client.resolve_relation_pair_batch(
            entity1_name=entity1_name,
            entity2_name=entity2_name,
            new_relation_contents=new_contents,
            existing_relations=existing_relations_info,
            new_source_document=source_document.split('/')[-1] if source_document else "",
        )

        confidence = float(batch_result.get("confidence", 0.0) or 0.0)
        if (not self.batch_resolution_enabled) or batch_result.get("action") == "fallback" or (confidence < self.batch_resolution_confidence_threshold and fallback_to_single):
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
                )
                if relation:
                    processed_relations.append(relation)
            return processed_relations, relations_to_persist

        if batch_result.get("action") == "match_existing":
            matched_family_id = batch_result.get("matched_relation_id") or ""
            latest_relation = next((rel for rel in existing_relations if rel.family_id == matched_family_id), None)
            if latest_relation and batch_result.get("need_update"):
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
            elif latest_relation:
                processed_relations.append(latest_relation)
            else:
                fallback_content = batch_result.get("merged_content") or self.llm_client.merge_multiple_relation_contents(new_contents)
                if not batch_result.get("merged_content"):
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
                )
                if new_relation is not None:
                    relations_to_persist.append(new_relation)
                    processed_relations.append(new_relation)
        else:
            merged_content = (batch_result.get("merged_content") or "").strip()
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
            )
            if new_relation is not None:
                relations_to_persist.append(new_relation)
                processed_relations.append(new_relation)
        return processed_relations, relations_to_persist
    
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
        dbg(f"entity_name_to_id 映射 ({len(entity_name_to_id)} 个): {list(entity_name_to_id.keys())[:20]}")
        
        for relation in extracted_relations:
            # 支持新旧格式
            entity1_name = relation.get('entity1_name') or relation.get('from_entity_name', '')
            entity2_name = relation.get('entity2_name') or relation.get('to_entity_name', '')
            
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
            # 确保关系使用标准化后的实体对
            relation_copy = relation.copy()
            relation_copy['entity1_name'] = normalized_pair[0]
            relation_copy['entity2_name'] = normalized_pair[1]
            relations_by_pair[normalized_pair].append(relation_copy)
        
        # 对每个实体对的关系进行合并
        merged_relations = []
        for pair, relations in relations_by_pair.items():
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
        relation_contents = [rel.get('content', '') for rel in relations if rel.get('content')]
        
        if not relation_contents:
            return relations[0]  # 如果没有content，返回第一个
        
        if len(relation_contents) == 1:
            return relations[0]  # 只有一个有content的关系
        
        # 使用LLM合并多个关系内容
        merged_content = self.llm_client.merge_multiple_relation_contents(
            relation_contents
        )
        
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
                                 base_time: Optional[datetime] = None) -> Optional[Relation]:
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
        entity1_name = entity1_name or extracted_relation.get('entity1_name') or extracted_relation.get('from_entity_name', '')
        entity2_name = entity2_name or extracted_relation.get('entity2_name') or extracted_relation.get('to_entity_name', '')
        
        # 步骤1：根据两个实体的 family_id 查找所有已有关系
        # 注意：这里传入的是 family_id，方法内部会转换为所有版本的绝对ID来查询
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
        existing_relations_info = [
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
            new_source_document=source_document.split('/')[-1] if source_document else "",
        )
        # LLM 有时返回 list 而非 dict，统一取第一个元素
        if isinstance(match_result, list) and len(match_result) > 0:
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
            
            # 判断是否需要更新：比较最新版本的content和当前抽取的content
            need_update = self.llm_client.judge_content_need_update(
                latest_relation.content,
                relation_content,
                old_source_document=latest_relation.source_document,
                new_source_document=source_document,
                old_name=f"{entity1_name}<->{entity2_name}",
                new_name=f"{entity1_name}<->{entity2_name}",
                object_type="关系",
            )
            
            if need_update:
                # 需要更新：合并内容
                if verbose_relation:
                    current_versions = self.storage.get_relation_versions(family_id)
                    record_count = len(current_versions)

                # 合并内容
                merged_content = self.llm_client.merge_relation_content(
                    latest_relation.content,
                    relation_content,
                    old_source_document=latest_relation.source_document,
                    new_source_document=source_document,
                    entity1_name=entity1_name,
                    entity2_name=entity2_name,
                )

                # 创建新版本
                if verbose_relation:
                    wprint(f"[关系操作] 🔄 更新关系: {entity1_name} <-> {entity2_name} (family_id: {family_id}, 版本数: {record_count})")

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
                    skip_if_unchanged=True,
                )

                return new_relation
            else:
                # 不需要更新，返回最新版本
                if verbose_relation:
                    # 获取数据库中该 family_id 的版本数量
                    current_versions = self.storage.get_relation_versions(family_id)
                    version_count = len(current_versions)
                    wprint(f"[关系操作] ⏭️  匹配但无需更新: {entity1_name} <-> {entity2_name} (family_id: {family_id}, 数据库中有 {version_count} 个版本)")
                return latest_relation
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
    
    def _build_new_relation(self, entity1_id: str, entity2_id: str,
                            content: str, episode_id: str,
                            entity1_name: str = "", entity2_name: str = "",
                            verbose_relation: bool = True, source_document: str = "",
                            base_time: Optional[datetime] = None,
                            entity_lookup: Optional[Dict[str, Any]] = None) -> Optional[Relation]:
        """构建新关系对象，但不立即写库。"""
        # 内容校验：过短的内容（< 5字符）不是有效关系描述
        if not content or len(content.strip()) < 5:
            if verbose_relation:
                wprint(f"[关系操作] ⚠️  跳过: 关系内容过短 ({len(content.strip()) if content else 0}字符): {entity1_name} <-> {entity2_name}")
            return None

        # 通过 family_id 获取实体的最新版本（优先使用预取缓存）
        entity1 = (entity_lookup or {}).get(entity1_id) or self.storage.get_entity_by_family_id(entity1_id)
        entity2 = (entity_lookup or {}).get(entity2_id) or self.storage.get_entity_by_family_id(entity2_id)
        
        if not entity1 or not entity2:
            missing_info = []
            if not entity1:
                missing_info.append(f"entity1: {entity1_name or '(未提供名称)'} (family_id: {entity1_id})")
            if not entity2:
                missing_info.append(f"entity2: {entity2_name or '(未提供名称)'} (family_id: {entity2_id})")
            
            if verbose_relation:
                wprint(f"[关系操作] ⚠️  警告: 无法找到实体: {', '.join(missing_info)}，跳过关系创建")
            return None
        
        ts = base_time if base_time is not None else datetime.now()
        processed_time = datetime.now()
        family_id = f"rel_{uuid.uuid4().hex[:12]}"
        relation_record_id = f"relation_{processed_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        if entity1.name <= entity2.name:
            entity1_absolute_id = entity1.absolute_id
            entity2_absolute_id = entity2.absolute_id
        else:
            entity1_absolute_id = entity2.absolute_id
            entity2_absolute_id = entity1.absolute_id

        source_document_only = source_document.split('/')[-1] if source_document else ""

        relation = Relation(
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
            confidence=0.7,
        )
        return relation

    def _create_new_relation(self, entity1_id: str, entity2_id: str,
                            content: str, episode_id: str,
                            entity1_name: str = "", entity2_name: str = "",
                            verbose_relation: bool = True, source_document: str = "",
                            base_time: Optional[datetime] = None) -> Optional[Relation]:
        """创建新关系"""
        relation = self._build_new_relation(
            entity1_id, entity2_id, content, episode_id,
            entity1_name=entity1_name, entity2_name=entity2_name,
            verbose_relation=verbose_relation, source_document=source_document, base_time=base_time,
        )
        if relation:
            self.storage.save_relation(relation)
            if verbose_relation:
                relation_versions = self.storage.get_relation_versions(relation.family_id)
                version_count = len(relation_versions)
                wprint(f"[关系操作] ✅ 创建新关系: {entity1_name} <-> {entity2_name} (family_id: {relation.family_id}, 数据库中有 {version_count} 个版本)")
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

        # 内容校验：过短的内容不是有效关系描述
        if not content or len(content.strip()) < 5:
            if verbose_relation:
                wprint(f"[关系操作] ⚠️  跳过版本: 内容过短 ({len(content.strip()) if content else 0}字符): {family_id}")
            return None

        # 内容未变化则跳过版本创建，避免版本膨胀（优先使用传入的已有关系）
        existing_relation = _existing_relation or self.storage.get_relation_by_family_id(family_id)
        if existing_relation and existing_relation.content:
            if content.strip() == existing_relation.content.strip() or \
               content.replace(' ', '').replace('\n', '') == existing_relation.content.replace(' ', '').replace('\n', ''):
                if verbose_relation:
                    wprint(f"[关系操作] 内容未变化，跳过版本创建: {family_id}")
                return None

        # 通过 family_id 获取实体的最新版本（优先使用预取缓存）
        entity1 = (entity_lookup or {}).get(entity1_id) or self.storage.get_entity_by_family_id(entity1_id)
        entity2 = (entity_lookup or {}).get(entity2_id) or self.storage.get_entity_by_family_id(entity2_id)
        
        if not entity1 or not entity2:
            missing_info = []
            if not entity1:
                missing_info.append(f"entity1: {entity1_name or '(未提供名称)'} (family_id: {entity1_id})")
            if not entity2:
                missing_info.append(f"entity2: {entity2_name or '(未提供名称)'} (family_id: {entity2_id})")
            
            if verbose_relation:
                wprint(f"[关系操作] ⚠️  警告: 无法找到实体: {', '.join(missing_info)}，跳过关系版本创建")
            return None
        
        ts = base_time if base_time is not None else datetime.now()
        processed_time = datetime.now()
        relation_record_id = f"relation_{processed_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        if entity1.name <= entity2.name:
            entity1_absolute_id = entity1.absolute_id
            entity2_absolute_id = entity2.absolute_id
        else:
            entity1_absolute_id = entity2.absolute_id
            entity2_absolute_id = entity1.absolute_id

        source_document_only = source_document.split('/')[-1] if source_document else ""

        relation = Relation(
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
            confidence=0.7,
        )
        return relation

    def _create_relation_version(self, family_id: str, entity1_id: str,
                                 entity2_id: str, content: str,
                                 episode_id: str,
                                 verbose_relation: bool = True,
                                 source_document: str = "",
                                 entity1_name: str = "",
                                 entity2_name: str = "",
                                 base_time: Optional[datetime] = None,
                                 skip_if_unchanged: bool = False,
                                 entity_lookup: Optional[Dict[str, Any]] = None) -> Optional[Relation]:
        """创建关系的新版本"""
        # 如果启用 skip_if_unchanged，检查内容是否真正变化
        _existing_relation = None
        if skip_if_unchanged and content:
            _existing_relation = self.storage.get_relation_by_family_id(family_id)
            if _existing_relation and _existing_relation.content:
                if content.strip() == _existing_relation.content.strip():
                    return _existing_relation
        relation = self._build_relation_version(
            family_id, entity1_id, entity2_id, content, episode_id,
            verbose_relation=verbose_relation, source_document=source_document,
            entity1_name=entity1_name, entity2_name=entity2_name, base_time=base_time,
            entity_lookup=entity_lookup,
            _existing_relation=_existing_relation,
        )
        if relation:
            self.storage.save_relation(relation)
            # 置信度演化：关系被新文本再次提及 → 独立来源印证 → 置信度提升
            self.storage.adjust_confidence_on_corroboration(family_id, source_type="relation")
        return relation
